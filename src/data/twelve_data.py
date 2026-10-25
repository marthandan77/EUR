from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


BASE_URL = "https://api.twelvedata.com"
REQUEST_TIMEOUT_SECONDS = 25


class TwelveDataError(RuntimeError):
    """Raised when Twelve Data returns an error or unusable response."""


def _request_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Call Twelve Data and normalize transport and provider errors."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as error:
        raise TwelveDataError("The request timed out. Try again shortly.") from error
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        raise TwelveDataError(f"Twelve Data returned HTTP {status_code}.") from error
    except requests.RequestException as error:
        raise TwelveDataError("Could not connect to Twelve Data. Check the app's network access.") from error

    try:
        payload = response.json()
    except ValueError as error:
        raise TwelveDataError("Twelve Data returned a response that was not valid JSON.") from error

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


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _get_eur_forex_pairs(api_key: str) -> pd.DataFrame:
    payload = _request_json(
        "forex_pairs",
        {"apikey": api_key, "currency_quote": "EUR", "format": "JSON"},
    )
    records = payload.get("data", [])
    if not isinstance(records, list):
        raise TwelveDataError("The forex catalog response did not contain a symbol list.")

    rows = []
    for item in records:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        quote_currency = str(item.get("currency_quote", "")).upper()
        if symbol.endswith("/EUR") or quote_currency == "EUR":
            rows.append(
                {
                    "symbol": symbol,
                    "asset_class": "Forex",
                    "name": str(item.get("currency_group", "Forex pair")),
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "asset_class", "name"])


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _get_eur_metals(api_key: str) -> pd.DataFrame:
    payload = _request_json(
        "commodities",
        {"apikey": api_key, "format": "JSON"},
    )
    records = payload.get("data", [])
    if not isinstance(records, list):
        raise TwelveDataError("The commodity catalog response did not contain a symbol list.")

    rows = []
    for item in records:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        category = str(item.get("category", "")).strip()
        if symbol.upper().endswith("/EUR") and "metal" in category.casefold():
            rows.append(
                {
                    "symbol": symbol,
                    "asset_class": category,
                    "name": str(item.get("name", category or "Metal")),
                }
            )
    return pd.DataFrame(rows, columns=["symbol", "asset_class", "name"])


def discover_eur_universe(api_key: str) -> pd.DataFrame:
    """Return forex and precious-metal instruments quoted in EUR."""
    forex = _get_eur_forex_pairs(api_key)
    metals = _get_eur_metals(api_key)
    instruments = pd.concat([forex, metals], ignore_index=True)
    if instruments.empty:
        return instruments
    return (
        instruments.drop_duplicates(subset=["symbol", "asset_class"])
        .sort_values(["asset_class", "symbol"])
        .reset_index(drop=True)
    )


@st.cache_data(ttl=15 * 60, show_spinner=False)
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
