from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.research.evaluation import summarize_predictions
from src.research.features import BASE_FEATURES, build_labeled_samples
from src.research.models import MODEL_NAMES, FittedCandidate, fit_candidate, predict_candidate


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts"
REPORT_PATH = ARTIFACT_ROOT / "latest_report.json"
PRODUCTION_DIR = ARTIFACT_ROOT / "production"


def _fit_calibration_split(
    pool: pd.DataFrame,
    calibration_fraction: float,
    purge_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pool["datetime"].unique()))
    if len(dates) < max(150, purge_bars * 4):
        raise ValueError("Too few dates for a purged train/calibration split.")
    calibration_start = dates[max(1, int(len(dates) * (1.0 - calibration_fraction)))]
    train = pool.loc[
        pool["datetime"].lt(calibration_start)
        & pool["label_end_datetime"].lt(calibration_start)
    ].copy()
    calibration = pool.loc[pool["datetime"].ge(calibration_start)].copy()
    return train, calibration


def make_walk_forward_folds(
    development_samples: pd.DataFrame,
    n_splits: int = 3,
    min_train_fraction: float = 0.55,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding chronological validation blocks, never shuffled."""
    dates = np.array(sorted(development_samples["datetime"].unique()))
    if len(dates) < 300:
        raise ValueError("At least 300 unique timestamps are required for walk-forward evaluation.")
    first_validation = max(1, int(len(dates) * min_train_fraction))
    validation_dates = dates[first_validation:]
    blocks = [block for block in np.array_split(validation_dates, n_splits) if len(block)]
    folds = []
    for block in blocks:
        start, end = block[0], block[-1]
        valid_mask = development_samples["datetime"].between(start, end, inclusive="both").to_numpy()
        train_mask = development_samples["datetime"].lt(start).to_numpy()
        folds.append((train_mask, valid_mask))
    return folds


def _predict_validation(
    candidate: FittedCandidate,
    train_samples: pd.DataFrame,
    calibration_samples: pd.DataFrame,
    valid_samples: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    context = pd.concat(
        [
            train_samples.groupby(["symbol", "side"], sort=False).tail(candidate.lookback - 1),
            calibration_samples,
            valid_samples,
        ],
        ignore_index=True,
    ).sort_values(["symbol", "side", "datetime"]).reset_index(drop=True)
    probabilities, aligned = predict_candidate(candidate, context)
    validation_times = set(valid_samples["datetime"])
    mask = aligned["datetime"].isin(validation_times).to_numpy()
    return probabilities[mask], aligned.loc[mask].reset_index(drop=True)


def _evaluate_one_model(
    model_name: str,
    development_samples: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    costs: dict[str, float],
) -> tuple[dict, list[dict], list[pd.DataFrame]]:
    fold_reports: list[dict] = []
    fold_trades: list[pd.DataFrame] = []
    fold_briers: list[float] = []
    lookback = int(config["lookback_bars"])
    min_expected = float(config["minimum_expected_net_r"])
    for fold_number, (_, valid_mask) in enumerate(folds, start=1):
        validation = development_samples.loc[valid_mask].copy()
        validation_start = validation["datetime"].min()
        pool = development_samples.loc[
            development_samples["datetime"].lt(validation_start)
            & development_samples["label_end_datetime"].lt(validation_start)
        ].copy()
        train, calibration = _fit_calibration_split(pool, 0.20, int(config["holding_bars"]))
        if len(train) < 500 or len(calibration) < max(100, lookback * 2):
            raise ValueError(
                f"Fold {fold_number} has insufficient train/calibration samples "
                f"({len(train)}/{len(calibration)})."
            )
        candidate, _, _ = fit_candidate(
            model_name,
            train,
            calibration,
            feature_names=tuple(BASE_FEATURES),
            lookback=lookback,
            epochs=int(config.get("epochs", 8)),
            seed=1729 + fold_number,
        )
        probabilities, aligned = _predict_validation(candidate, train, calibration, validation)
        if len(aligned) != len(probabilities) or aligned.empty:
            raise ValueError(f"Fold {fold_number} produced no validation predictions.")
        metrics, trades = summarize_predictions(
            aligned,
            probabilities,
            float(config["target_atr"]),
            float(config["stop_atr"]),
            costs,
            min_expected,
            candidate.timeout_payoffs,
        )
        metrics["fold"] = fold_number
        fold_reports.append(metrics)
        fold_trades.append(trades.assign(fold=fold_number))
        fold_briers.append(metrics["brier_score"])

    all_trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame()
    all_costs_configured = bool(costs) and set(development_samples["symbol"].unique()).issubset(costs)
    cv_net = (
        float(all_trades["realized_net_r"].mean())
        if all_costs_configured and not all_trades.empty
        else None
    )
    positive_folds = sum(
        1
        for report in fold_reports
        if report["realized_net_r_per_trade"] is not None
        and report["realized_net_r_per_trade"] > 0
        and report["trade_count"] > 0
    )
    summary = {
        "model": model_name,
        "folds_completed": len(fold_reports),
        "mean_brier_score": float(np.mean(fold_briers)) if fold_briers else None,
        "mean_log_loss": float(np.mean([item["log_loss"] for item in fold_reports]))
        if fold_reports
        else None,
        "cv_trade_count": int(len(all_trades)),
        "cv_realized_net_r_per_trade": cv_net,
        "cv_positive_folds": int(positive_folds),
        "fold_metrics": fold_reports,
    }
    return summary, fold_reports, fold_trades


def _fit_final_candidate(
    model_name: str,
    development_samples: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[FittedCandidate, pd.DataFrame, pd.DataFrame]:
    train, calibration = _fit_calibration_split(
        development_samples,
        calibration_fraction=0.20,
        purge_bars=int(config["holding_bars"]),
    )
    candidate, _, _ = fit_candidate(
        model_name,
        train,
        calibration,
        feature_names=tuple(BASE_FEATURES),
        lookback=int(config["lookback_bars"]),
        epochs=int(config.get("epochs", 8)),
        seed=1729,
    )
    return candidate, train, calibration


def _write_report(report: dict, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def train_and_audit(
    shared_market_data: pd.DataFrame,
    config: dict[str, Any],
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict:
    """Train all candidates, compare them walk-forward, then gate promotion."""
    samples = build_labeled_samples(
        shared_market_data,
        holding_bars=int(config["holding_bars"]),
        target_atr=float(config["target_atr"]),
        stop_atr=float(config["stop_atr"]),
    )
    samples["datetime"] = pd.to_datetime(samples["datetime"], utc=True)
    samples["label_end_datetime"] = pd.to_datetime(samples["label_end_datetime"], utc=True)
    timestamps = np.array(sorted(samples["datetime"].unique()))
    holdout_fraction = float(config.get("holdout_fraction", 0.15))
    holdout_start = timestamps[max(1, int((1.0 - holdout_fraction) * len(timestamps)))]
    development = samples.loc[
        samples["datetime"].lt(holdout_start)
        & samples["label_end_datetime"].lt(holdout_start)
    ].copy()
    holdout = samples.loc[samples["datetime"].ge(holdout_start)].copy()
    folds = make_walk_forward_folds(development, int(config.get("cv_folds", 3)))
    costs = {
        str(symbol): float(value)
        for symbol, value in config.get("round_trip_cost_bps", {}).items()
        if value is not None and float(value) >= 0
    }
    reports: dict[str, dict] = {}
    fit_candidates: dict[str, FittedCandidate] = {}
    cv_failures: dict[str, str] = {}

    for model_name in MODEL_NAMES:
        try:
            summary, _, _ = _evaluate_one_model(model_name, development, folds, config, costs)
            candidate, final_train, final_calibration = _fit_final_candidate(
                model_name, development, config
            )
            probabilities, aligned = _predict_validation(
                candidate,
                final_train,
                final_calibration,
                holdout,
            )
            holdout_metrics, holdout_trades = summarize_predictions(
                aligned,
                probabilities,
                float(config["target_atr"]),
                float(config["stop_atr"]),
                costs,
                float(config["minimum_expected_net_r"]),
                candidate.timeout_payoffs,
            )
            summary["holdout_metrics"] = holdout_metrics
            summary["holdout_trade_rows"] = int(len(holdout_trades))
            reports[model_name] = summary
            fit_candidates[model_name] = candidate
        except Exception as error:
            cv_failures[model_name] = f"{type(error).__name__}: {error}"

    if not reports:
        raise ValueError("All model candidates failed. " + " | ".join(f"{k}: {v}" for k, v in cv_failures.items()))

    costs_complete = set(shared_market_data["symbol"].unique()).issubset(costs)
    if costs_complete:
        ranking_key = lambda item: (
            item[1]["cv_realized_net_r_per_trade"]
            if item[1]["cv_realized_net_r_per_trade"] is not None
            else -float("inf")
        )
    else:
        ranking_key = lambda item: -(
            item[1]["mean_brier_score"]
            if item[1]["mean_brier_score"] is not None
            else float("inf")
        )
    selected_name, selected_report = max(reports.items(), key=ranking_key)
    holdout_metrics = selected_report["holdout_metrics"]
    required_cv_folds = max(1, (int(config.get("cv_folds", 3)) + 1) // 2)
    promotion_reasons = []
    if set(reports) != set(MODEL_NAMES):
        promotion_reasons.append("All three required model candidates did not complete successfully.")
    if not costs_complete:
        promotion_reasons.append("Per-symbol round-trip costs are not configured for every active instrument.")
    if not config.get("cost_estimates_verified", False):
        promotion_reasons.append("Round-trip costs are provisional and have not been verified against broker execution pricing.")
    if selected_report["cv_trade_count"] < int(config.get("minimum_cv_trades", 30)):
        promotion_reasons.append("Walk-forward sample produced too few candidate trades.")
    if selected_report["cv_positive_folds"] < required_cv_folds:
        promotion_reasons.append("Candidate did not have positive net performance in enough walk-forward folds.")
    if holdout_metrics["trade_count"] < int(config.get("minimum_holdout_trades", 10)):
        promotion_reasons.append("Untouched holdout produced too few candidate trades.")
    if holdout_metrics["realized_net_r_per_trade"] is None or holdout_metrics["realized_net_r_per_trade"] <= 0:
        promotion_reasons.append("Untouched holdout net return per trade was not positive after configured costs.")
    promoted = not promotion_reasons

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_start_utc": str(shared_market_data["datetime"].min()),
        "data_end_utc": str(shared_market_data["datetime"].max()),
        "sample_count": int(len(samples)),
        "symbol_count": int(shared_market_data["symbol"].nunique()),
        "active_symbols": sorted(shared_market_data["symbol"].unique().tolist()),
        "config": config,
        "costs_complete": costs_complete,
        "cost_estimates_verified": bool(config.get("cost_estimates_verified", False)),
        "selected_candidate": selected_name,
        "promoted": promoted,
        "promotion_reasons": promotion_reasons,
        "model_results": reports,
        "model_failures": cv_failures,
    }

    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_report(report, artifact_root / "latest_report.json")
    if promoted:
        production_dir = artifact_root / "production"
        production_dir.mkdir(parents=True, exist_ok=True)
        for model_name, candidate in fit_candidates.items():
            temp_path = production_dir / f"{model_name}.joblib.tmp"
            joblib.dump(candidate, temp_path)
            temp_path.replace(production_dir / f"{model_name}.joblib")
        _write_report(report, production_dir / "metadata.json")
    return report
