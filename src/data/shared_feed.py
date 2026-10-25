from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.twelve_data import TwelveDataError, get_time_series


ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_CONFIG_PATH = ROOT / "config" / "universe.json"


def load_universe_config(path: Path = UNIVERSE_CONFIG_PATH) -> dict:
    """Load the editable symbol universe and research parameters."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read universe configuration at {path}.") from error
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("Universe configuration must contain a non-empty symbols list.")
    seen: set[str] = set()
    for instrument in symbols:
        if not isinstance(instrument, dict):
            raise ValueError("Each configured instrument must be an object.")
        symbol = str(instrument.get("symbol", "")).strip()
        asset_class = str(instrument.get("asset_class", "")).strip()
        if not symbol or not asset_class or symbol in seen:
            raise ValueError("Each symbol needs a unique symbol and an asset_class.")
        seen.add(symbol)
    return config


def fetch_shared_market_data(
    api_key: str,
    symbols: list[dict[str, str]],
    interval: str,
    output_size: int,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Fetch one shared OHLC dataset; training stays strict unless explicitly relaxed."""
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for instrument in symbols:
        symbol = instrument["symbol"]
        try:
            bars = get_time_series(
                api_key=api_key,
                symbol=symbol,
                interval=interval,
                output_size=output_size,
            )
        except TwelveDataError as error:
            failures.append(f"{symbol}: {error}")
            continue
        # The most recent intraday bar can still be forming; exclude it conservatively.
        if len(bars) > 1:
            bars = bars.iloc[:-1].copy()
        bars = bars.copy()
        bars["symbol"] = symbol
        bars["asset_class"] = instrument["asset_class"]
        frames.append(bars)

    if failures and not allow_partial:
        raise TwelveDataError("Data load incomplete. " + " | ".join(failures))
    if not frames:
        detail = " " + " | ".join(failures) if failures else ""
        raise TwelveDataError("No configured market data could be loaded." + detail)
    shared = pd.concat(frames, ignore_index=True)
    shared = shared.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    shared.attrs["load_failures"] = failures
    shared.attrs["partial_universe"] = bool(failures)
    return shared
