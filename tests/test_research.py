from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib

from src.research.evaluation import apply_portfolio_risk_limits, make_trade_advice
from src.research.features import BASE_FEATURES, _label_direction, build_inference_rows, build_labeled_samples
from src.research.models import MODEL_NAMES, fit_candidate, predict_candidate
from src.research.deployment import training_config_mismatches
from src.research.pipeline import make_walk_forward_folds, train_and_audit
from src.data import shared_feed
from src.data import twelve_data
from src.data.twelve_data import TwelveDataError, _request_json


def _synthetic_market_data(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(73)
    dates = pd.date_range("2022-01-01", periods=rows, freq="h", tz="UTC")
    frames = []
    for symbol, asset_class, drift in (
        ("AUD/EUR", "Forex", 0.00001),
        ("XAU/EUR", "Precious Metal", 0.00003),
    ):
        returns = rng.normal(drift, 0.001, rows)
        close = 1.5 * np.exp(np.cumsum(returns))
        open_price = np.r_[close[0], close[:-1]]
        high = np.maximum(open_price, close) * (1 + rng.uniform(0.0001, 0.001, rows))
        low = np.minimum(open_price, close) * (1 - rng.uniform(0.0001, 0.001, rows))
        frames.append(pd.DataFrame({
            "datetime": dates,
            "symbol": symbol,
            "asset_class": asset_class,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }))
    return pd.concat(frames, ignore_index=True)


def test_causal_features_do_not_change_when_future_bars_change():
    market = _synthetic_market_data(220)
    original = build_inference_rows(market)
    cutoff = market["datetime"].sort_values().unique()[170]
    altered = market.copy()
    future_mask = altered["datetime"].gt(cutoff)
    altered.loc[future_mask, ["open", "high", "low", "close"]] *= 1.4
    changed = build_inference_rows(altered)
    check_time = pd.to_datetime(cutoff, utc=True)
    left = original.loc[original["datetime"].eq(check_time), BASE_FEATURES].sort_index(axis=1)
    right = changed.loc[changed["datetime"].eq(check_time), BASE_FEATURES].sort_index(axis=1)
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_ambiguous_same_bar_barriers_resolve_to_stop():
    bars = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        "open": [100.0, 100.0, 100.0, 100.0, 100.0],
        "high": [100.5, 100.5, 102.5, 100.5, 100.5],
        "low": [99.5, 99.5, 97.5, 99.5, 99.5],
        "close": [100.0] * 5,
    })
    features = pd.DataFrame({"atr": [1.0] * 5})
    labels = _label_direction(bars, features, side=1, holding_bars=2, target_atr=2, stop_atr=1)
    first = labels.iloc[0]
    assert first["label"] == 0
    assert first["realized_r"] == pytest.approx(-1.0)


def test_opening_gap_through_target_resolves_before_later_same_bar_stop():
    bars = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
        "open": [100.0, 100.0, 103.0, 100.0],
        "high": [100.5, 100.5, 104.0, 100.5],
        "low": [99.5, 99.5, 98.0, 99.5],
        "close": [100.0, 100.0, 102.0, 100.0],
    })
    features = pd.DataFrame({"atr": [1.0] * 4})
    labels = _label_direction(bars, features, side=1, holding_bars=2, target_atr=2, stop_atr=1)
    assert labels.iloc[0]["label"] == 2
    assert labels.iloc[0]["realized_r"] == pytest.approx(2.0)


def test_labeled_samples_use_both_directions_and_end_before_latest_bar():
    market = _synthetic_market_data(260)
    samples = build_labeled_samples(market, holding_bars=8)
    assert set(samples["side"]) == {-1.0, 1.0}
    assert set(samples["label"]).issubset({0, 1, 2})
    assert samples["label_end_datetime"].max() < market["datetime"].max()


def test_walk_forward_folds_are_chronological_and_disjoint():
    dates = pd.date_range("2020-01-01", periods=400, freq="h", tz="UTC")
    samples = pd.DataFrame({"datetime": dates, "label_end_datetime": dates + pd.Timedelta(minutes=30)})
    folds = make_walk_forward_folds(samples, n_splits=3)
    assert len(folds) == 3
    previous_end = None
    for train_mask, valid_mask in folds:
        assert train_mask.any() and valid_mask.any()
        train_end = samples.loc[train_mask, "datetime"].max()
        valid_start = samples.loc[valid_mask, "datetime"].min()
        assert train_end < valid_start
        if previous_end is not None:
            assert valid_start > previous_end
        previous_end = samples.loc[valid_mask, "datetime"].max()


def _synthetic_samples(rows_per_side: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    dates = pd.date_range("2021-01-01", periods=rows_per_side, freq="h", tz="UTC")
    samples = []
    for symbol in ("AUD/EUR", "XAU/EUR"):
        for side in (1.0, -1.0):
            for index, date in enumerate(dates):
                features = rng.normal(size=len(BASE_FEATURES)).astype(float)
                labels = index % 3
                row = dict(zip(BASE_FEATURES, features))
                row.update({
                    "datetime": date,
                    "label_end_datetime": date + pd.Timedelta(hours=1),
                    "symbol": symbol,
                    "asset_class": "Forex" if symbol == "AUD/EUR" else "Precious Metal",
                    "side": side,
                    "label": labels,
                    "entry_price": 100.0,
                    "risk_distance": 1.0,
                    "realized_r": float(labels - 1),
                })
                samples.append(row)
    return pd.DataFrame(samples).sort_values(["datetime", "symbol", "side"]).reset_index(drop=True)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_each_model_fits_serializes_and_returns_three_probabilities(model_name: str, tmp_path):
    samples = _synthetic_samples()
    dates = sorted(samples["datetime"].unique())
    train = samples.loc[samples["datetime"].lt(dates[125])].copy()
    calibration = samples.loc[samples["datetime"].ge(dates[125])].copy()
    candidate, probabilities, aligned = fit_candidate(
        model_name,
        train,
        calibration,
        lookback=8,
        epochs=1,
        seed=42,
    )
    assert probabilities.shape == (len(aligned), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    artifact_path = tmp_path / f"{model_name}.joblib"
    joblib.dump(candidate, artifact_path)
    candidate = joblib.load(artifact_path)
    predicted, prediction_rows = predict_candidate(candidate, calibration)
    assert predicted.shape[1] == 3
    assert len(predicted) == len(prediction_rows)


def test_trade_advisor_abstains_without_costs():
    row = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AUD/EUR"],
        "side": [1.0],
        "entry_price": [100.0],
        "risk_distance": [1.0],
        "label": [2],
        "realized_r": [2.0],
        "label_end_datetime": [pd.Timestamp("2024-01-02", tz="UTC")],
    })
    probabilities = {name: np.array([[0.05, 0.05, 0.90]]) for name in MODEL_NAMES}
    advice = make_trade_advice(
        row, probabilities, target_atr=2, stop_atr=1,
        round_trip_cost_bps={}, min_expected_net_r=0.1,
    )
    assert advice.loc[0, "recommendation"] == "No trade"


def test_trade_advisor_abstains_when_costs_are_provisional():
    row = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AUD/EUR"],
        "side": [1.0],
        "entry_price": [100.0],
        "risk_distance": [1.0],
        "label": [2],
        "realized_r": [2.0],
        "label_end_datetime": [pd.Timestamp("2024-01-02", tz="UTC")],
    })
    probabilities = {name: np.array([[0.05, 0.05, 0.90]]) for name in MODEL_NAMES}
    advice = make_trade_advice(
        row, probabilities, target_atr=2, stop_atr=1,
        round_trip_cost_bps={"AUD/EUR": 20.0}, min_expected_net_r=0.1,
    )
    assert advice.loc[0, "recommendation"] == "No trade"


def test_portfolio_overlay_limits_shared_eur_factor_exposure():
    advice = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01", tz="UTC")] * 2,
        "symbol": ["AUD/EUR", "CAD/EUR"],
        "recommendation": ["Long", "Long"],
        "expected_net_r": [0.5, 0.4],
    })
    dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    market = pd.concat([
        pd.DataFrame({"datetime": dates, "symbol": symbol, "close": np.exp(np.linspace(0, 0.01, len(dates)))})
        for symbol in ("AUD/EUR", "CAD/EUR")
    ], ignore_index=True)
    result = apply_portfolio_risk_limits(advice, market, max_abs_factor_units=1.0)
    assert result.loc[result["symbol"].eq("AUD/EUR"), "recommendation"].iloc[0] == "Long"
    assert result.loc[result["symbol"].eq("CAD/EUR"), "recommendation"].iloc[0] == "No trade"


def test_shared_feed_fetches_each_configured_symbol_once(monkeypatch):
    calls = []

    def fake_get_time_series(api_key, symbol, interval, output_size):
        calls.append(symbol)
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
        })

    monkeypatch.setattr(shared_feed, "get_time_series", fake_get_time_series)
    symbols = [
        {"symbol": "AUD/EUR", "asset_class": "Forex"},
        {"symbol": "XAU/EUR", "asset_class": "Precious Metal"},
    ]
    data = shared_feed.fetch_shared_market_data("fake", symbols, "1h", 3)
    assert calls == ["AUD/EUR", "XAU/EUR"]
    assert data["symbol"].nunique() == 2
    assert len(data) == 4  # The last possibly incomplete bar is discarded per symbol.


def test_shared_feed_partial_mode_keeps_available_symbols_but_strict_mode_fails(monkeypatch):
    def fake_get_time_series(api_key, symbol, interval, output_size):
        if symbol == "XAU/EUR":
            raise shared_feed.TwelveDataError("Twelve Data returned HTTP 404.")
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
        })

    monkeypatch.setattr(shared_feed, "get_time_series", fake_get_time_series)
    symbols = [
        {"symbol": "AUD/EUR", "asset_class": "Forex"},
        {"symbol": "XAU/EUR", "asset_class": "Precious Metal"},
    ]
    partial = shared_feed.fetch_shared_market_data("fake", symbols, "1h", 3, allow_partial=True)
    assert partial["symbol"].unique().tolist() == ["AUD/EUR"]
    assert partial.attrs["partial_universe"]
    assert partial.attrs["load_failures"] == ["XAU/EUR: Twelve Data returned HTTP 404."]
    with pytest.raises(shared_feed.TwelveDataError, match="Data load incomplete"):
        shared_feed.fetch_shared_market_data("fake", symbols, "1h", 3)


def test_provider_error_never_exposes_api_key(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {}

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"status": "error", "code": 400, "message": "bad key secret-token-123"}

    monkeypatch.setattr("src.data.twelve_data.requests.get", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(TwelveDataError) as error:
        _request_json("time_series", {"apikey": "secret-token-123"})
    assert "secret-token-123" not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_catalog_discovery_uses_two_shared_catalog_requests(monkeypatch):
    calls = []

    def fake_request(endpoint, params):
        calls.append(endpoint)
        if endpoint == "forex_pairs":
            return {"data": [
                {"symbol": "AUD/EUR", "currency_group": "Minor"},
                {"symbol": "AUD/NZD", "currency_group": "Minor"},
            ]}
        return {"data": [
            {"symbol": "XAU/EUR", "name": "Gold Spot", "category": "Precious Metal"}
        ]}

    monkeypatch.setattr(twelve_data, "_request_json", fake_request)
    requested = [
        {"symbol": "AUD/EUR", "asset_class": "Forex"},
        {"symbol": "AUD/NZD", "asset_class": "Forex"},
        {"symbol": "XAU/EUR", "asset_class": "Precious Metal"},
    ]
    universe = twelve_data.discover_eur_universe("fake", requested)
    assert calls == ["forex_pairs", "commodities"]
    assert universe["symbol"].tolist() == ["AUD/EUR", "AUD/NZD", "XAU/EUR"]


def test_changed_cost_or_ticker_universe_invalidates_stored_models():
    original = {
        "interval": "1h",
        "lookback_bars": 64,
        "holding_bars": 24,
        "target_atr": 2.0,
        "stop_atr": 1.0,
        "minimum_expected_net_r": 0.1,
        "round_trip_cost_bps": {"AUD/EUR": 1.2},
        "symbols": [{"symbol": "AUD/EUR", "asset_class": "Forex"}],
    }
    assert training_config_mismatches(original, original) == []
    changed = {**original, "round_trip_cost_bps": {"AUD/EUR": 2.0}}
    assert "round_trip_cost_bps" in training_config_mismatches(changed, original)
    changed = {**original, "symbols": [{"symbol": "XAU/EUR", "asset_class": "Precious Metal"}]}
    assert "symbols" in training_config_mismatches(changed, original)


def test_all_three_models_complete_a_small_walk_forward_audit(tmp_path):
    market = _synthetic_market_data(620)
    config = {
        "holding_bars": 8,
        "target_atr": 2.0,
        "stop_atr": 1.0,
        "lookback_bars": 8,
        "minimum_expected_net_r": 0.10,
        "round_trip_cost_bps": {"AUD/EUR": 1.0, "XAU/EUR": 1.0},
        "minimum_cv_trades": 100_000,
        "minimum_holdout_trades": 1,
        "cv_folds": 2,
        "epochs": 1,
    }
    report = train_and_audit(market, config, artifact_root=tmp_path / "artifacts")
    assert set(report["model_results"]) == set(MODEL_NAMES)
    assert report["sample_count"] > 0
    assert (tmp_path / "artifacts" / "latest_report.json").exists()
    assert not report["promoted"]
    assert not report["cost_estimates_verified"]
    assert any("provisional" in reason for reason in report["promotion_reasons"])
