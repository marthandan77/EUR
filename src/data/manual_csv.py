from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable
from io import BytesIO

import pandas as pd


REQUIRED_OHLC_COLUMNS = {"datetime", "open", "high", "low", "close"}


def _normalize_symbol(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def load_uploaded_market_csvs(
    uploads: Iterable,
    configured_symbols: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Validate user-supplied OHLC CSVs into the shared market-data format.

    CSVs can contain a symbol column or be named after one configured symbol,
    such as ``AUD_EUR.csv``. The latest row per symbol is dropped because CSV
    export timestamps cannot tell us whether that bar has completed.
    """
    if not uploads:
        raise ValueError("Select at least one OHLC CSV file.")

    canonical_symbols = {_normalize_symbol(symbol): (symbol, asset_class)
                         for symbol, asset_class in configured_symbols}
    frames = []
    errors = []

    for upload in uploads:
        try:
            frame = pd.read_csv(BytesIO(upload.getvalue()))
        except Exception as error:
            errors.append(f"{upload.name}: could not read CSV ({type(error).__name__}).")
            continue
        if frame.empty:
            errors.append(f"{upload.name}: file contains no rows.")
            continue

        frame.columns = [str(column).strip().casefold() for column in frame.columns]
        aliases = {
            "date": "datetime",
            "timestamp": "datetime",
            "time": "datetime",
            "ticker": "symbol",
            "instrument": "symbol",
        }
        frame = frame.rename(columns={column: aliases.get(column, column) for column in frame.columns})
        missing = REQUIRED_OHLC_COLUMNS.difference(frame.columns)
        if missing:
            errors.append(f"{upload.name}: missing columns {', '.join(sorted(missing))}.")
            continue

        if "symbol" in frame.columns:
            symbols = frame["symbol"].astype(str).str.strip()
        else:
            filename_symbol = _normalize_symbol(Path(upload.name).stem)
            matched = canonical_symbols.get(filename_symbol)
            if matched is None:
                errors.append(
                    f"{upload.name}: add a symbol column or name the file after a configured ticker."
                )
                continue
            symbols = pd.Series(matched[0], index=frame.index)

        frame["_symbol_key"] = symbols.map(_normalize_symbol)
        unknown = sorted(set(frame["_symbol_key"]) - set(canonical_symbols))
        if unknown:
            unknown_names = sorted(set(symbols[frame["_symbol_key"].isin(unknown)].astype(str)))
            errors.append(f"{upload.name}: unconfigured symbols: {', '.join(unknown_names)}.")
            continue
        frame["symbol"] = frame["_symbol_key"].map(
            lambda key: canonical_symbols[key][0]
        )
        frame["asset_class"] = frame["_symbol_key"].map(
            lambda key: canonical_symbols[key][1]
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame["datetime"].isna()
        invalid |= frame[["open", "high", "low", "close"]].isna().any(axis=1)
        invalid |= frame[["open", "high", "low", "close"]].le(0).any(axis=1)
        invalid |= frame["high"].lt(frame["low"])
        invalid |= frame["high"].lt(frame[["open", "close"]].max(axis=1))
        invalid |= frame["low"].gt(frame[["open", "close"]].min(axis=1))
        if invalid.any():
            errors.append(f"{upload.name}: contains invalid timestamps or OHLC values.")
            continue

        frames.append(frame[["datetime", "open", "high", "low", "close", "symbol", "asset_class"]])

    if errors:
        raise ValueError("CSV load failed: " + " ".join(errors))
    if not frames:
        raise ValueError("No usable OHLC bars were found in the uploaded files.")

    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(["symbol", "datetime"], keep=False)
    if duplicates.any():
        duplicated_rows = combined.loc[duplicates]
        same_values = not duplicated_rows.groupby(["symbol", "datetime"])[
            ["open", "high", "low", "close"]
        ].nunique().gt(1).any().any()
        if not same_values:
            raise ValueError("CSV files contain conflicting OHLC values for the same symbol and timestamp.")
        combined = combined.drop_duplicates(["symbol", "datetime"], keep="last")

    combined = combined.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    last_bar_indices = combined.groupby("symbol").tail(1).index
    combined = combined.drop(index=last_bar_indices).reset_index(drop=True)
    if combined.empty:
        raise ValueError("At least two bars per ticker are required; the latest possibly incomplete bar is discarded.")
    return combined
