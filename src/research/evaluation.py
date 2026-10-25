from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


CLASS_NAMES = ["stop", "timeout", "target"]


def summarize_predictions(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    target_atr: float,
    stop_atr: float,
    round_trip_cost_bps: dict[str, float],
    min_expected_net_r: float,
    timeout_payoffs: dict[str, float] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Score probabilities and simulate a non-overlapping per-symbol signal stream."""
    if len(rows) != len(probabilities):
        raise ValueError("Prediction count does not match evaluation rows.")
    result = rows.reset_index(drop=True).copy()
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    result[["p_stop", "p_timeout", "p_target"]] = probabilities
    reward_r = target_atr / stop_atr
    timeout_payoffs = timeout_payoffs or {"__global__": 0.0}
    result["timeout_mean_r"] = [
        timeout_payoffs.get(f"{symbol}|{int(side)}", timeout_payoffs.get("__global__", 0.0))
        for symbol, side in zip(result["symbol"], result["side"])
    ]
    result["expected_gross_r"] = (
        result["p_target"] * reward_r
        - result["p_stop"]
        + result["p_timeout"] * result["timeout_mean_r"]
    )
    costs_known = result["symbol"].isin(round_trip_cost_bps)
    result["cost_known"] = costs_known
    result["cost_r"] = 0.0
    for symbol, basis_points in round_trip_cost_bps.items():
        mask = result["symbol"].eq(symbol)
        notional_cost = float(basis_points) / 10_000.0 * result.loc[mask, "entry_price"]
        result.loc[mask, "cost_r"] = notional_cost / result.loc[mask, "risk_distance"].replace(0, np.nan)
    result["expected_net_r"] = result["expected_gross_r"] - result["cost_r"]
    result["realized_gross_r"] = result["realized_r"]
    result["realized_net_r"] = result["realized_gross_r"] - result["cost_r"]

    eligible = result.loc[result["expected_net_r"].ge(min_expected_net_r)].copy()
    eligible = eligible.sort_values(["datetime", "expected_net_r"], ascending=[True, False])
    # Keep one position open per symbol. The label's exit timestamp is conservatively used
    # as the earliest time that a subsequent signal may enter.
    selected_indices: list[int] = []
    last_exit_by_symbol: dict[str, pd.Timestamp] = {}
    for index, row in eligible.iterrows():
        prior_exit = last_exit_by_symbol.get(row["symbol"])
        if prior_exit is not None and row["datetime"] <= prior_exit:
            continue
        selected_indices.append(index)
        last_exit_by_symbol[row["symbol"]] = row["label_end_datetime"]
    trades = eligible.loc[selected_indices].copy()
    # Of simultaneous symbols with similar exposure, retain all for evaluation; portfolio
    # sizing is a separate stage and must account for covariance and risk limits.
    labels = result["label"].to_numpy(dtype=int)
    one_hot = np.eye(3, dtype=float)[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    try:
        multiclass_log_loss = float(log_loss(labels, probabilities, labels=[0, 1, 2]))
    except ValueError:
        multiclass_log_loss = float("nan")

    metrics = {
        "sample_count": int(len(result)),
        "brier_score": brier,
        "log_loss": multiclass_log_loss,
        "trade_count": int(len(trades)),
        "costs_configured": bool(result["cost_known"].all()),
        "expected_net_r_per_trade": float(trades["expected_net_r"].mean()) if len(trades) else None,
        "realized_net_r_per_trade": (
            float(trades["realized_net_r"].mean())
            if len(trades) and result["cost_known"].all()
            else None
        ),
        "realized_gross_r_per_trade": float(trades["realized_gross_r"].mean()) if len(trades) else None,
        "win_rate": float((trades["realized_net_r"] > 0).mean()) if len(trades) else None,
        "positive_symbols": int((trades.groupby("symbol")["realized_gross_r"].mean() > 0).sum())
        if len(trades)
        else 0,
        "total_symbols": int(trades["symbol"].nunique()) if len(trades) else 0,
    }
    return metrics, trades.reset_index(drop=True)


def make_trade_advice(
    rows: pd.DataFrame,
    probabilities_by_model: dict[str, np.ndarray],
    target_atr: float,
    stop_atr: float,
    round_trip_cost_bps: dict[str, float],
    min_expected_net_r: float,
    agreement_tolerance: float = 0.20,
    required_model_count: int = 3,
    timeout_payoffs_by_model: dict[str, dict[str, float]] | None = None,
    cost_estimates_verified: bool = False,
) -> pd.DataFrame:
    """Combine calibrated candidates conservatively; disagreement yields no trade."""
    model_names = list(probabilities_by_model)
    if not model_names:
        return pd.DataFrame()
    per_model = []
    for model_name, probabilities in probabilities_by_model.items():
        frame = rows.reset_index(drop=True).copy()
        if len(frame) != len(probabilities):
            continue
        probabilities = np.asarray(probabilities, dtype=float)
        reward_r = target_atr / stop_atr
        frame["model"] = model_name
        frame["p_stop"] = probabilities[:, 0]
        frame["p_timeout"] = probabilities[:, 1]
        frame["p_target"] = probabilities[:, 2]
        timeout_payoffs = (timeout_payoffs_by_model or {}).get(model_name, {"__global__": 0.0})
        frame["timeout_mean_r"] = [
            timeout_payoffs.get(f"{symbol}|{int(side)}", timeout_payoffs.get("__global__", 0.0))
            for symbol, side in zip(frame["symbol"], frame["side"])
        ]
        frame["expected_gross_r"] = (
            frame["p_target"] * reward_r
            - frame["p_stop"]
            + frame["p_timeout"] * frame["timeout_mean_r"]
        )
        frame["cost_r"] = 0.0
        for symbol, basis_points in round_trip_cost_bps.items():
            mask = frame["symbol"].eq(symbol)
            frame.loc[mask, "cost_r"] = (
                float(basis_points) / 10_000.0 * frame.loc[mask, "entry_price"]
                / frame.loc[mask, "risk_distance"].replace(0, np.nan)
            )
        frame["expected_net_r"] = frame["expected_gross_r"] - frame["cost_r"]
        per_model.append(frame)
    if not per_model:
        return pd.DataFrame()

    stacked = pd.concat(per_model, ignore_index=True)
    keys = ["datetime", "symbol", "side"]
    summary = stacked.groupby(keys, as_index=False).agg(
        p_stop=("p_stop", "mean"),
        p_timeout=("p_timeout", "mean"),
        p_target=("p_target", "mean"),
        expected_net_r=("expected_net_r", "mean"),
        model_count=("model", "nunique"),
        model_expected_r_min=("expected_net_r", "min"),
        model_expected_r_max=("expected_net_r", "max"),
        timeout_mean_r=("timeout_mean_r", "mean"),
        entry_price=("entry_price", "first"),
        risk_distance=("risk_distance", "first"),
        label=("label", "first"),
        realized_r=("realized_r", "first"),
        label_end_datetime=("label_end_datetime", "first"),
    )
    summary["direction"] = np.where(summary["side"].gt(0), "Long", "Short")
    summary["model_agreement"] = (
        summary["model_expected_r_max"] - summary["model_expected_r_min"]
    ).le(agreement_tolerance)
    all_costs_present = summary["symbol"].isin(round_trip_cost_bps) & cost_estimates_verified
    summary["recommendation"] = np.where(
        all_costs_present
        & summary["model_agreement"]
        & summary["model_count"].eq(required_model_count)
        & summary["expected_net_r"].ge(min_expected_net_r),
        summary["direction"],
        "No trade",
    )
    return summary.sort_values(["expected_net_r", "datetime"], ascending=[False, False]).reset_index(drop=True)


def apply_portfolio_risk_limits(
    advice: pd.DataFrame,
    market_data: pd.DataFrame,
    max_positions: int = 3,
    max_abs_factor_units: float = 2.0,
    max_same_direction_correlation: float = 0.85,
    correlation_lookback: int = 168,
) -> pd.DataFrame:
    """Suppress redundant simultaneous signals using lagged correlation and factor limits."""
    result = advice.copy()
    result["risk_filter_reason"] = ""
    signal_mask = result["recommendation"].isin(["Long", "Short"])
    signals = result.loc[signal_mask].sort_values("expected_net_r", ascending=False)
    if signals.empty:
        return result

    clean = market_data.sort_values(["symbol", "datetime"]).copy()
    clean["log_return"] = clean.groupby("symbol")["close"].transform(lambda values: np.log(values).diff())
    returns = clean.pivot(index="datetime", columns="symbol", values="log_return").tail(correlation_lookback)
    correlation = returns.corr(min_periods=max(24, correlation_lookback // 3))

    factor_exposure: dict[str, float] = {}
    accepted: list[pd.Series] = []
    rejected: dict[int, str] = {}
    for row_index, row in signals.iterrows():
        if len(accepted) >= max_positions:
            rejected[row_index] = f"Portfolio limit: at most {max_positions} concurrent positions."
            continue
        position_sign = 1.0 if row["recommendation"] == "Long" else -1.0
        legs = str(row["symbol"]).upper().split("/")
        candidate_exposure: dict[str, float] = {}
        if len(legs) == 2:
            candidate_exposure[legs[0]] = position_sign
            candidate_exposure[legs[1]] = -position_sign
        for factor, amount in candidate_exposure.items():
            if abs(factor_exposure.get(factor, 0.0) + amount) > max_abs_factor_units:
                rejected[row_index] = f"Portfolio limit: {factor} exposure cap reached."
                break
        if row_index in rejected:
            continue

        redundant = False
        for accepted_row in accepted:
            pair_corr = correlation.get(accepted_row["symbol"], pd.Series(dtype=float)).get(
                row["symbol"], np.nan
            )
            aligned_corr = pair_corr * position_sign * (
                1.0 if accepted_row["recommendation"] == "Long" else -1.0
            )
            if pd.notna(aligned_corr) and aligned_corr >= max_same_direction_correlation:
                rejected[row_index] = (
                    f"Portfolio limit: highly correlated with {accepted_row['symbol']} exposure."
                )
                redundant = True
                break
        if redundant:
            continue
        accepted.append(row)
        for factor, amount in candidate_exposure.items():
            factor_exposure[factor] = factor_exposure.get(factor, 0.0) + amount

    for row_index, reason in rejected.items():
        result.loc[row_index, "recommendation"] = "No trade"
        result.loc[row_index, "risk_filter_reason"] = reason
    return result
