from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "volatility_24",
    "volatility_72",
    "volatility_ratio",
    "range_atr",
    "body_atr",
    "close_location",
    "trend_efficiency_24",
    "peer_fx_return",
    "peer_fx_breadth",
    "peer_fx_dispersion",
    "peer_metal_return",
    "peer_metal_breadth",
    "peer_return",
    "peer_count",
    "asset_is_metal",
    "side",
]

TARGET_NAMES = {0: "stop", 1: "timeout", 2: "target"}
N_CLASSES = 3


def validate_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize long-form OHLC data without filling missing bars."""
    required = {"datetime", "symbol", "asset_class", "open", "high", "low", "close"}
    missing = required.difference(market_data.columns)
    if missing:
        raise ValueError("Market data missing fields: " + ", ".join(sorted(missing)))
    frame = market_data.loc[:, sorted(required)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime", "symbol", "asset_class", "open", "high", "low", "close"])
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    valid_ohlc = (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )
    frame = frame.loc[valid_ohlc]
    frame = frame.drop_duplicates(["symbol", "datetime"], keep="last")
    return frame.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def _build_causal_features(market_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns_by_symbol: dict[str, pd.Series] = {}
    classes: dict[str, str] = {}
    per_symbol: dict[str, pd.DataFrame] = {}

    for symbol, group in market_data.groupby("symbol", sort=False):
        bars = group.sort_values("datetime").set_index("datetime")
        close = bars["close"].astype(float)
        log_close = np.log(close)
        ret_1 = log_close.diff()
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                bars["high"] - bars["low"],
                (bars["high"] - previous_close).abs(),
                (bars["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        volatility_24 = ret_1.rolling(24, min_periods=24).std(ddof=0)
        volatility_72 = ret_1.rolling(72, min_periods=72).std(ddof=0)

        features = pd.DataFrame(index=bars.index)
        for lag in (1, 3, 6, 12, 24):
            features[f"ret_{lag}"] = log_close - log_close.shift(lag)
        features["volatility_24"] = volatility_24
        features["volatility_72"] = volatility_72
        features["volatility_ratio"] = volatility_24 / volatility_72.replace(0, np.nan)
        features["range_atr"] = true_range / atr.replace(0, np.nan)
        features["body_atr"] = (bars["close"] - bars["open"]) / atr.replace(0, np.nan)
        candle_range = (bars["high"] - bars["low"]).replace(0, np.nan)
        features["close_location"] = (bars["close"] - bars["low"]) / candle_range
        features["trend_efficiency_24"] = (
            (log_close - log_close.shift(24)).abs()
            / ret_1.abs().rolling(24, min_periods=24).sum().replace(0, np.nan)
        )
        features["atr"] = atr
        features["asset_class"] = str(group["asset_class"].iloc[-1])
        per_symbol[str(symbol)] = features
        returns_by_symbol[str(symbol)] = ret_1
        classes[str(symbol)] = str(group["asset_class"].iloc[-1])

    return_matrix = pd.concat(returns_by_symbol, axis=1).sort_index()
    # Lag other instruments by one complete bar to avoid relying on simultaneous close timing.
    lagged_peer_returns = return_matrix.shift(1)
    forex_columns = [symbol for symbol, kind in classes.items() if kind.casefold() == "forex"]
    metal_columns = [symbol for symbol, kind in classes.items() if "metal" in kind.casefold()]

    for symbol, features in per_symbol.items():
        peer_columns = [column for column in lagged_peer_returns.columns if column != symbol]
        fx_peers = [column for column in forex_columns if column != symbol]
        metal_peers = [column for column in metal_columns if column != symbol]

        def summarize(columns: list[str], prefix: str) -> None:
            if not columns:
                features[f"{prefix}_return"] = 0.0
                features[f"{prefix}_breadth"] = 0.0
                features[f"{prefix}_dispersion"] = 0.0
                return
            peer_frame = lagged_peer_returns[columns]
            features[f"{prefix}_return"] = peer_frame.mean(axis=1)
            features[f"{prefix}_breadth"] = (peer_frame > 0).mean(axis=1)
            features[f"{prefix}_dispersion"] = peer_frame.std(axis=1, ddof=0)

        summarize(fx_peers, "peer_fx")
        summarize(metal_peers, "peer_metal")
        peer_frame = (
            lagged_peer_returns[peer_columns]
            if peer_columns
            else pd.DataFrame(index=return_matrix.index)
        )
        features["peer_return"] = peer_frame.mean(axis=1)
        features["peer_count"] = peer_frame.notna().sum(axis=1).astype(float)
        features["asset_is_metal"] = float("metal" in classes[symbol].casefold())
        features["symbol"] = symbol
        features["datetime"] = features.index

    return per_symbol


def _label_direction(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    side: int,
    holding_bars: int,
    target_atr: float,
    stop_atr: float,
) -> pd.DataFrame:
    records: list[dict] = []
    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    atr_values = features["atr"].to_numpy(dtype=float)
    datetimes = pd.to_datetime(bars["datetime"], utc=True).to_numpy()
    n_rows = len(bars)

    for decision_index in range(n_rows - holding_bars):
        entry_index = decision_index + 1
        entry_price = opens[entry_index]
        risk_distance = atr_values[decision_index] * stop_atr
        if not np.isfinite(risk_distance) or risk_distance <= 0 or entry_price <= 0:
            continue

        target_price = entry_price + side * atr_values[decision_index] * target_atr
        stop_price = entry_price - side * risk_distance
        outcome_class = 1
        realized_r = 0.0
        end_index = min(n_rows - 1, entry_index + holding_bars - 1)
        exit_index = end_index
        for future_index in range(entry_index, end_index + 1):
            if side == 1:
                if opens[future_index] <= stop_price:
                    stop_hit, target_hit = True, False
                elif opens[future_index] >= target_price:
                    stop_hit, target_hit = False, True
                else:
                    stop_hit = lows[future_index] <= stop_price
                    target_hit = highs[future_index] >= target_price
            else:
                if opens[future_index] >= stop_price:
                    stop_hit, target_hit = True, False
                elif opens[future_index] <= target_price:
                    stop_hit, target_hit = False, True
                else:
                    stop_hit = highs[future_index] >= stop_price
                    target_hit = lows[future_index] <= target_price
            # If OHLC cannot establish intrabar ordering, resolve the collision as a stop.
            if stop_hit:
                outcome_class = 0
                stop_fill = min(opens[future_index], stop_price) if side == 1 else max(opens[future_index], stop_price)
                realized_r = side * (stop_fill - entry_price) / risk_distance
                exit_index = future_index
                break
            if target_hit:
                outcome_class = 2
                realized_r = target_atr / stop_atr
                exit_index = future_index
                break
            if future_index == end_index:
                realized_r = side * (closes[future_index] - entry_price) / risk_distance

        records.append(
            {
                "datetime": datetimes[decision_index],
                "side": float(side),
                "label": outcome_class,
                "realized_r": float(realized_r),
                "label_end_datetime": datetimes[exit_index],
                "entry_price": float(entry_price),
                "risk_distance": float(risk_distance),
            }
        )
    return pd.DataFrame.from_records(records)


def build_labeled_samples(
    market_data: pd.DataFrame,
    holding_bars: int = 24,
    target_atr: float = 2.0,
    stop_atr: float = 1.0,
) -> pd.DataFrame:
    """Build long/short three-outcome samples from causal per-bar features."""
    clean = validate_market_data(market_data)
    if holding_bars < 1 or target_atr <= 0 or stop_atr <= 0:
        raise ValueError("Holding period and ATR barriers must be positive.")
    features_by_symbol = _build_causal_features(clean)
    sample_frames: list[pd.DataFrame] = []

    for symbol, feature_frame in features_by_symbol.items():
        symbol_bars = clean.loc[clean["symbol"].eq(symbol)].sort_values("datetime").reset_index(drop=True)
        feature_frame = feature_frame.reset_index(drop=True)
        if len(symbol_bars) != len(feature_frame):
            raise ValueError(f"Feature/bar row mismatch for {symbol}.")
        for side in (1, -1):
            labels = _label_direction(
                symbol_bars,
                feature_frame,
                side,
                holding_bars,
                target_atr,
                stop_atr,
            )
            if labels.empty:
                continue
            labels["symbol"] = symbol
            labels["asset_class"] = feature_frame["asset_class"].iloc[0]
            feature_columns = [column for column in BASE_FEATURES if column != "side"]
            input_features = feature_frame[["datetime", *feature_columns]].copy()
            joined = labels.merge(input_features, on="datetime", how="left", validate="many_to_one")
            joined["side"] = joined["side"].astype(float)
            joined = joined.dropna(subset=[*BASE_FEATURES, "label", "label_end_datetime"])
            sample_frames.append(joined)

    if not sample_frames:
        raise ValueError("Not enough clean history to form labeled samples.")
    samples = pd.concat(sample_frames, ignore_index=True)
    samples = samples.sort_values(["datetime", "symbol", "side"]).reset_index(drop=True)
    return samples


def build_inference_rows(
    market_data: pd.DataFrame,
    stop_atr: float = 1.0,
) -> pd.DataFrame:
    """Build current causal feature rows for both directions without future labels."""
    clean = validate_market_data(market_data)
    feature_frames = _build_causal_features(clean)
    rows: list[pd.DataFrame] = []
    for symbol, frame in feature_frames.items():
        frame = frame.reset_index(drop=True)
        frame["symbol"] = symbol
        frame["asset_class"] = str(frame["asset_class"].iloc[0])
        frame["entry_price"] = clean.loc[clean["symbol"].eq(symbol)].sort_values("datetime")["close"].to_numpy()
        frame["risk_distance"] = frame["atr"] * stop_atr
        frame["label"] = 1
        frame["realized_r"] = 0.0
        frame["label_end_datetime"] = frame["datetime"]
        for side in (1.0, -1.0):
            sided = frame.copy()
            sided["side"] = side
            rows.append(sided)
    if not rows:
        raise ValueError("No features could be built from market data.")
    inference = pd.concat(rows, ignore_index=True)
    inference = inference.dropna(subset=BASE_FEATURES)
    return inference.sort_values(["symbol", "side", "datetime"]).reset_index(drop=True)
