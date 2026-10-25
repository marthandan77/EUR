from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Mapping

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.shared_feed import fetch_shared_market_data, load_universe_config
from src.data.manual_csv import load_uploaded_market_csvs
from src.data.twelve_data import TwelveDataError, discover_eur_universe
from src.research.evaluation import apply_portfolio_risk_limits, make_trade_advice
from src.research.deployment import training_config_mismatches
from src.research.features import build_inference_rows
from src.research.models import MODEL_NAMES, predict_candidate


ROOT = Path(__file__).resolve().parent
PRODUCTION_DIR = ROOT / "artifacts" / "production"
REPORT_PATH = ROOT / "artifacts" / "latest_report.json"


st.set_page_config(
    page_title="EUR Asset Regime & Trend Breakout Backtester",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_discover_universe(api_key: str, configured_symbols: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    symbols = [{"symbol": symbol, "asset_class": asset_class} for symbol, asset_class in configured_symbols]
    return discover_eur_universe(api_key, symbols)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def cached_shared_market_data(
    api_key: str,
    configured_symbols: tuple[tuple[str, str], ...],
    interval: str,
    output_size: int,
) -> pd.DataFrame:
    symbols = [{"symbol": symbol, "asset_class": asset_class} for symbol, asset_class in configured_symbols]
    return fetch_shared_market_data(api_key, symbols, interval, output_size)


@st.cache_resource(show_spinner=False)
def cached_load_candidate(path: str, file_modified_ns: int):
    del file_modified_ns  # Included in the cache key so a promoted artifact reloads automatically.
    return joblib.load(path)


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_twelve_data_secret(secrets) -> tuple[str, tuple[str, ...]]:
    """Read common Twelve Data secret names without exposing their values."""
    visible_names = []
    for name, value in secrets.items():
        normalized_name = re.sub(r"[^a-z0-9]", "", str(name).lower())
        visible_names.append(str(name))
        if normalized_name in {"twelvedatakey", "twelvedataapikey"} and value:
            return str(value).strip(), tuple(visible_names)
        if normalized_name == "twelvedata" and isinstance(value, Mapping):
            for child_name, child_value in value.items():
                visible_names.append(f"{name}.{child_name}")
                normalized_child = re.sub(r"[^a-z0-9]", "", str(child_name).lower())
                if normalized_child in {"key", "apikey"} and child_value:
                    return str(child_value).strip(), tuple(visible_names)
    return "", tuple(visible_names)


st.title("EUR Asset Regime & Trend Breakout Advisor")
st.caption("Shared Twelve Data feed · LSTM · TCN · gradient-boosted trees")

try:
    twelve_data_key, visible_secret_names = resolve_twelve_data_secret(st.secrets)
except Exception:
    twelve_data_key = ""
    visible_secret_names = ()

try:
    config = load_universe_config()
    configured_symbols = tuple(
        (str(item["symbol"]), str(item["asset_class"])) for item in config["symbols"]
    )
except (ValueError, KeyError) as error:
    st.error(f"Universe configuration error: {error}")
    st.stop()

if not twelve_data_key:
    visible_names_text = ", ".join(visible_secret_names) if visible_secret_names else "none"
    st.warning(
        "Twelve Data is unavailable in this app runtime. You can still load historical OHLC "
        "CSV files in the Shared market data tab. Secret field names visible to this app: "
        f"{visible_names_text}."
    )

universe_tab, data_tab, advisor_tab, audit_tab = st.tabs(
    ["Universe", "Shared data", "Trade advisor", "Model audit"]
)

with universe_tab:
    st.subheader("Configurable active universe")
    st.write(
        "The active ticker list is in `config/universe.json`. Add or remove symbols there; "
        "the scheduled workflow will rebuild the shared features and retrain all models."
    )
    configured_frame = pd.DataFrame(
        [{"symbol": symbol, "asset_class": asset_class} for symbol, asset_class in configured_symbols]
    )
    if twelve_data_key:
        with st.spinner("Checking configured symbols against Twelve Data catalogs…"):
            try:
                instrument_universe = cached_discover_universe(twelve_data_key, configured_symbols)
                st.success(f"Validated {len(instrument_universe)} configured symbols.")
            except TwelveDataError as error:
                instrument_universe = pd.DataFrame()
                st.error(f"Symbol catalog validation failed: {error}")
    else:
        instrument_universe = configured_frame
    st.dataframe(instrument_universe if not instrument_universe.empty else configured_frame,
                 use_container_width=True, hide_index=True)
    st.caption(f"{len(configured_symbols)} configured tickers. One shared time series per ticker feeds all models.")

with data_tab:
    st.subheader("Shared market data")
    interval = str(config.get("interval", "1h"))
    history_bars = int(config.get("history_bars", 5000))
    st.caption(
        "Without an API key, upload broker or data-vendor OHLC CSV exports below. "
        "This fallback loads historical data for inspection; it does not enable model promotion or trade advice."
    )
    manual_uploads = st.file_uploader(
        "Upload OHLC CSV files (one combined file with a symbol column, or one file per ticker)",
        type=["csv"],
        accept_multiple_files=True,
        key="manual_ohlc_uploads",
    )
    st.download_button(
        "Download CSV template",
        data="datetime,open,high,low,close,symbol\n",
        file_name="ohlc_template.csv",
        mime="text/csv",
    )
    if st.button("Load uploaded CSV data", disabled=not manual_uploads):
        try:
            st.session_state["shared_market_data"] = load_uploaded_market_csvs(
                manual_uploads, configured_symbols
            )
            st.session_state["shared_data_source"] = "Uploaded CSV"
            st.session_state["shared_data_error"] = None
        except ValueError as error:
            st.session_state["shared_data_error"] = str(error)
    if not twelve_data_key:
        st.info("Twelve Data is unavailable in this runtime. Use the CSV upload above to inspect historical market data.")
    elif st.button("Fetch shared data", type="primary"):
        with st.spinner("Fetching one OHLC history per configured symbol…"):
            try:
                st.session_state["shared_market_data"] = cached_shared_market_data(
                    twelve_data_key, configured_symbols, interval, history_bars
                )
                st.session_state["shared_data_source"] = "Twelve Data"
                st.session_state["shared_data_error"] = None
            except TwelveDataError as error:
                st.session_state["shared_data_error"] = str(error)
    if st.session_state.get("shared_data_error"):
        st.error(st.session_state["shared_data_error"])
    shared_data = st.session_state.get("shared_market_data")
    if shared_data is not None and not shared_data.empty:
        st.caption(f"Data source: {st.session_state.get('shared_data_source', 'Unknown')}")
        summary = (
            shared_data.groupby(["symbol", "asset_class"], as_index=False)
            .agg(bars=("datetime", "size"), first_bar=("datetime", "min"), last_bar=("datetime", "max"))
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        selected_symbol = st.selectbox("Preview ticker", summary["symbol"].tolist())
        symbol_bars = shared_data.loc[shared_data["symbol"].eq(selected_symbol)].sort_values("datetime")
        chart = go.Figure(
            data=[go.Candlestick(
                x=symbol_bars["datetime"],
                open=symbol_bars["open"],
                high=symbol_bars["high"],
                low=symbol_bars["low"],
                close=symbol_bars["close"],
                name=selected_symbol,
            )]
        )
        chart.update_layout(
            template="plotly_dark", height=480, xaxis_rangeslider_visible=False,
            xaxis_title="Time (UTC)", yaxis_title="Price",
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(symbol_bars.tail(100), use_container_width=True, hide_index=True)

with advisor_tab:
    st.subheader("Current model view")
    metadata = load_json(PRODUCTION_DIR / "metadata.json")
    if not config.get("cost_estimates_verified", False):
        st.warning(
            "Trade advice is disabled because the configured round-trip costs are provisional. "
            "Verify them against your broker's spreads, commissions, and expected slippage first."
        )
    elif not metadata or not metadata.get("promoted"):
        st.info(
            "No promoted model set is available yet. Run the scheduled GitHub Action after "
            "adding `TWELVE_DATA_KEY` to GitHub repository Actions secrets and configuring "
            "realistic per-symbol costs."
        )
    else:
        trained_config = metadata.get("config", {})
        mismatches = training_config_mismatches(config, trained_config)
        latest_report = load_json(REPORT_PATH)
        production_date = metadata.get("generated_at_utc", "")
        latest_date = latest_report.get("generated_at_utc", "") if latest_report else ""
        if mismatches:
            st.warning(
                "Model configuration changed since training ("
                + ", ".join(mismatches)
                + "). Run training before using these models."
            )
        elif latest_report and latest_date > production_date and not latest_report.get("promoted"):
            st.warning(
                "The latest scheduled audit did not promote a replacement; the prior production "
                "model set is retained. Review the Model audit tab before acting on signals."
            )
        elif not twelve_data_key:
            st.warning("Add the Twelve Data key to load current bars for model scoring.")
        elif st.button("Score latest completed bars", type="primary"):
            with st.spinner("Fetching one shared snapshot and scoring all three models…"):
                try:
                    market_data = cached_shared_market_data(
                        twelve_data_key,
                        configured_symbols,
                        interval,
                        history_bars,
                    )
                    inference_rows = build_inference_rows(
                        market_data,
                        stop_atr=float(config.get("stop_atr", 1.0)),
                    )
                    model_probabilities = {}
                    model_timeout_payoffs = {}
                    aligned_rows = None
                    for model_name in MODEL_NAMES:
                        model_path = PRODUCTION_DIR / f"{model_name}.joblib"
                        if not model_path.exists():
                            continue
                        candidate = cached_load_candidate(str(model_path), model_path.stat().st_mtime_ns)
                        probabilities, rows = predict_candidate(candidate, inference_rows)
                        model_probabilities[model_name] = probabilities
                        model_timeout_payoffs[model_name] = candidate.timeout_payoffs
                        if aligned_rows is None:
                            aligned_rows = rows
                    if len(model_probabilities) != len(MODEL_NAMES) or aligned_rows is None:
                        st.error("The promoted model set is incomplete. No trade advice generated.")
                    else:
                        latest_indices = (
                            aligned_rows.reset_index()
                            .sort_values("datetime")
                            .groupby(["symbol", "side"], sort=False)["index"]
                            .tail(1)
                            .to_numpy()
                        )
                        latest_rows = aligned_rows.iloc[latest_indices].reset_index(drop=True)
                        latest_probabilities = {
                            name: values[latest_indices]
                            for name, values in model_probabilities.items()
                        }
                        advice = make_trade_advice(
                            latest_rows,
                            latest_probabilities,
                            target_atr=float(config["target_atr"]),
                            stop_atr=float(config["stop_atr"]),
                            round_trip_cost_bps=config.get("round_trip_cost_bps", {}),
                            min_expected_net_r=float(config.get("minimum_expected_net_r", 0.10)),
                            required_model_count=3,
                            timeout_payoffs_by_model=model_timeout_payoffs,
                            cost_estimates_verified=bool(config.get("cost_estimates_verified", False)),
                        )
                        advice = advice.sort_values("expected_net_r", ascending=False).drop_duplicates(
                            "symbol", keep="first"
                        )
                        advice = apply_portfolio_risk_limits(
                            advice,
                            market_data,
                            max_positions=int(config.get("max_simultaneous_positions", 3)),
                            max_abs_factor_units=float(config.get("max_factor_exposure_units", 2.0)),
                            max_same_direction_correlation=float(
                                config.get("max_same_direction_correlation", 0.85)
                            ),
                            correlation_lookback=int(config.get("correlation_lookback_bars", 168)),
                        )
                        if advice.empty:
                            st.warning("No usable predictions were produced for the latest bars.")
                        else:
                            latest_advice = advice
                            st.caption(
                                "Probabilities are calibrated on earlier data. Signal prices use the latest "
                                "completed bar close; they are not executable quotes."
                            )
                            st.dataframe(
                                latest_advice[[
                                    "symbol", "direction", "recommendation", "p_target", "p_stop",
                                    "p_timeout", "expected_net_r", "model_agreement", "model_count",
                                    "timeout_mean_r", "entry_price", "risk_filter_reason",
                                ]],
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.session_state["latest_advice"] = latest_advice
                except (TwelveDataError, ValueError, OSError) as error:
                    st.error(f"Scoring failed: {error}")
        if metadata.get("promotion_reasons"):
            st.caption("Latest promotion audit: " + " ".join(metadata["promotion_reasons"]))

with audit_tab:
    st.subheader("Automated model comparison and promotion gate")
    latest_report = load_json(REPORT_PATH)
    if not latest_report:
        st.info("No training report is available. The scheduled GitHub Action writes the first report.")
    else:
        report_summary = {
            key: latest_report.get(key)
            for key in (
                "generated_at_utc", "data_start_utc", "data_end_utc", "sample_count",
                "symbol_count", "active_symbols", "selected_candidate", "promoted",
                "promotion_reasons", "model_failures",
            )
        }
        st.json(report_summary, expanded=False)
        model_rows = []
        for model_name, model_report in latest_report.get("model_results", {}).items():
            holdout = model_report.get("holdout_metrics", {})
            model_rows.append({
                "model": model_name,
                "walk_forward Brier": model_report.get("mean_brier_score"),
                "walk_forward log loss": model_report.get("mean_log_loss"),
                "walk_forward trades": model_report.get("cv_trade_count"),
                "walk_forward net R/trade": model_report.get("cv_realized_net_r_per_trade"),
                "holdout trades": holdout.get("trade_count"),
                "holdout net R/trade": holdout.get("realized_net_r_per_trade"),
            })
        if model_rows:
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Training is run by GitHub Actions. Current schedule: weekly. "
            "Model promotion is blocked until every active ticker has a configured round-trip cost "
            "and the out-of-sample gates pass."
        )
