from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.twelve_data import TwelveDataError, discover_eur_universe, get_time_series


st.set_page_config(
    page_title="EUR Asset Regime & Trend Breakout Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("EUR Asset Regime & Trend Breakout Backtester")
st.caption("Research workspace for EUR-quoted forex pairs and precious metals.")

try:
    twelve_data_key = st.secrets.get("twelve_data_key", "").strip()
except Exception:
    twelve_data_key = ""

if not twelve_data_key:
    st.error(
        "Twelve Data API key is missing. Add `twelve_data_key` in the app's "
        "Streamlit Secrets settings, then restart the app."
    )
    st.stop()

universe_tab, data_tab, research_tab = st.tabs(
    ["Universe", "Data health", "Research"]
)

with st.spinner("Loading EUR-quoted forex and precious-metal symbols…"):
    try:
        instrument_universe = discover_eur_universe(twelve_data_key)
        universe_error = None
    except TwelveDataError as error:
        instrument_universe = pd.DataFrame()
        universe_error = str(error)

if universe_error:
    st.error(f"Could not load the Twelve Data instrument catalogs: {universe_error}")
elif instrument_universe.empty:
    st.warning("No EUR-quoted forex or precious-metal symbols were returned.")

with universe_tab:
    st.subheader("Available EUR-quoted instruments")
    if not instrument_universe.empty:
        forex_count = int(instrument_universe["asset_class"].eq("Forex").sum())
        metal_count = int(instrument_universe["asset_class"].eq("Precious metal").sum())
        metric_columns = st.columns(2)
        metric_columns[0].metric("Forex pairs", forex_count)
        metric_columns[1].metric("Precious metals", metal_count)
        st.dataframe(
            instrument_universe,
            use_container_width=True,
            hide_index=True,
        )
    elif not universe_error:
        st.info("The filtered universe is empty. Check catalog coverage in your Twelve Data plan.")

with data_tab:
    st.subheader("Historical OHLC data")
    if not instrument_universe.empty:
        selected_symbol = st.selectbox(
            "Instrument",
            instrument_universe["symbol"].tolist(),
            format_func=lambda symbol: (
                f"{symbol} · "
                f"{instrument_universe.loc[instrument_universe['symbol'].eq(symbol), 'asset_class'].iloc[0]}"
            ),
        )
        selected_interval = st.selectbox(
            "Bar interval",
            ["1h", "4h", "1day"],
            index=0,
            format_func={"1h": "1 hour", "4h": "4 hours", "1day": "1 day"}.get,
        )
        output_size = st.slider("Historical bars", min_value=100, max_value=5000, value=1000, step=100)

        if st.button("Load history", type="primary"):
            with st.spinner(f"Requesting {selected_symbol} history…"):
                try:
                    historical_data = get_time_series(
                        api_key=twelve_data_key,
                        symbol=selected_symbol,
                        interval=selected_interval,
                        output_size=output_size,
                    )
                except TwelveDataError as error:
                    st.error(f"Could not load historical data: {error}")
                else:
                    st.session_state["historical_data"] = historical_data
                    st.session_state["historical_symbol"] = selected_symbol
                    st.session_state["historical_interval"] = selected_interval

        historical_data = st.session_state.get("historical_data")
        if historical_data is not None and not historical_data.empty:
            history_symbol = st.session_state.get("historical_symbol", selected_symbol)
            history_interval = st.session_state.get("historical_interval", selected_interval)
            st.caption(
                f"{history_symbol} · {history_interval} · "
                f"{len(historical_data):,} bars · "
                f"{historical_data['datetime'].min():%Y-%m-%d %H:%M UTC} to "
                f"{historical_data['datetime'].max():%Y-%m-%d %H:%M UTC}"
            )
            chart = go.Figure(
                data=[
                    go.Candlestick(
                        x=historical_data["datetime"],
                        open=historical_data["open"],
                        high=historical_data["high"],
                        low=historical_data["low"],
                        close=historical_data["close"],
                        name=history_symbol,
                    )
                ]
            )
            chart.update_layout(
                template="plotly_dark",
                xaxis_title="Time (UTC)",
                yaxis_title="Price",
                height=560,
                xaxis_rangeslider_visible=False,
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
            )
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(historical_data.tail(100), use_container_width=True, hide_index=True)
    elif not universe_error:
        st.info("Load an instrument universe before requesting history.")

with research_tab:
    st.subheader("Research pipeline")
    st.info(
        "The API connection and OHLC data view are in place. Causal feature engineering, "
        "barrier labels, and walk-forward evaluation are the next research milestones."
    )
