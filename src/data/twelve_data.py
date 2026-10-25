from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.twelvedata.com"
REQUEST_TIMEOUT_SECONDS = 25


class TwelveDataError(RuntimeError):
    """Raised when Twelve Data returns an error or unusable response."""


def _request_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Call Twelve Data and normalize transport and provider errors."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    payload: dict[str, Any] | None = None
    last_exception: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(60.0, max(1.0, float(retry_after))) if retry_after else 2.0 ** attempt
                    except ValueError:
                        delay = 2.0 ** attempt
                    if response.status_code == 429 and not retry_after:
                        delay = 60.0
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and (payload.get("status") == "error" or payload.get("code")):
                provider_message = str(payload.get("message", "")).casefold()
                rate_limited = "per minute" in provider_message or "rate limit" in provider_message
                if rate_limited and attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(60.0, max(1.0, float(retry_after))) if retry_after else 60.0
                    except ValueError:
                        delay = 60.0
                    time.sleep(delay)
                    continue
            break
        except requests.Timeout as error:
            last_exception = error
            if attempt < 2:
                time.sleep(2.0 ** attempt)
                continue
            raise TwelveDataError("The request timed out after retries. Try again shortly.") from error
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else "unknown"
            raise TwelveDataError(f"Twelve Data returned HTTP {status_code}.") from error
        except requests.RequestException as error:
            last_exception = error
            if attempt < 2:
                time.sleep(2.0 ** attempt)
                continue
            raise TwelveDataError("Could not connect to Twelve Data after retries.") from error
        except ValueError as error:
            raise TwelveDataError("Twelve Data returned a response that was not valid JSON.") from error

    if payload is None:
        raise TwelveDataError("Twelve Data request failed after retries.") from last_exception
    if not isinstance(payload, dict):
        raise TwelveDataError("Twelve Data returned an unexpected response format.")

    if payload.get("status") == "error" or payload.get("code"):
        message = str(payload.get("message", "Unknown provider error"))
        # Never surface credentials if the provider echoes request details.
        api_key = str(params.get("apikey", ""))
        if api_key:
            message = message.replace(api_key, "[redacted]")
        raise TwelveDataError(message)
    return payload


def discover_eur_universe(
    api_key: str, configured_symbols: list[dict[str, str]]
) -> pd.DataFrame:
    """Validate configured symbols using one request per provider catalog."""
    forex_payload = _request_json(
        "forex_pairs",
        {"apikey": api_key, "outputsize": 5000, "format": "JSON"},
    )
    commodity_payload = _request_json(
        "commodities",
        {"apikey": api_key, "outputsize": 5000, "format": "JSON"},
    )
    forex_records = forex_payload.get("data", [])
    commodity_records = commodity_payload.get("data", [])
    if not isinstance(forex_records, list) or not isinstance(commodity_records, list):
        raise TwelveDataError("A Twelve Data catalog response did not contain a symbol list.")
    forex_by_symbol = {
        str(item.get("symbol", "")).casefold(): item
        for item in forex_records
        if isinstance(item, dict)
    }
    commodity_by_symbol = {
        str(item.get("symbol", "")).casefold(): item
        for item in commodity_records
        if isinstance(item, dict)
    }
    records = []
    missing_symbols = []
    for configured in configured_symbols:
        symbol = str(configured["symbol"]).strip()
        asset_class = str(configured["asset_class"]).strip()
        catalog = forex_by_symbol if asset_class.casefold() == "forex" else commodity_by_symbol
        item = catalog.get(symbol.casefold())
        if item is None:
            missing_symbols.append(symbol)
            continue
        records.append(
            {
                "symbol": str(item.get("symbol", symbol)),
                "asset_class": str(item.get("category", asset_class)),
                "name": str(item.get("name", item.get("currency_group", symbol))),
            }
        )
    if missing_symbols:
        raise TwelveDataError(
            "Symbols not found in the configured Twelve Data catalogs: "
            + ", ".join(missing_symbols)
        )
    return pd.DataFrame(records, columns=["symbol", "asset_class", "name"])


def get_time_series(
    api_key: str,
    symbol: str,
    interval: str = "1h",
    output_size: int = 1000,
) -> pd.DataFrame:
    """Fetch and validate OHLC bars, ordered oldest to newest in UTC."""
    payload = _request_json(
        "time_series",
        {
            "apikey": api_key,
            "symbol": symbol,
            "interval": interval,
            "outputsize": int(output_size),
            "timezone": "UTC",
            "format": "JSON",
        },
    )
    values = payload.get("values", [])
    if not isinstance(values, list) or not values:
        raise TwelveDataError(f"No historical bars were returned for {symbol}.")

    frame = pd.DataFrame(values)
    required_columns = {"datetime", "open", "high", "low", "close"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise TwelveDataError(
            "Historical response is missing required fields: "
            + ", ".join(sorted(missing_columns))
        )

    frame = frame.loc[:, ["datetime", "open", "high", "low", "close"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["datetime", "open", "high", "low", "close"])
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise TwelveDataError(f"No valid OHLC bars were returned for {symbol}.")
    return frame
