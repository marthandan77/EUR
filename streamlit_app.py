import streamlit as st

st.set_page_config(
    page_title="EUR Asset Regime & Trend Breakout Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("EUR Asset Regime & Trend Breakout Backtester")
st.caption("Research workspace for EUR-quoted forex and metal instruments.")

universe_tab, data_tab, research_tab = st.tabs(
    ["Universe", "Data health", "Research"]
)

with universe_tab:
    st.info("Instrument discovery will be added in the first data-engineering milestone.")

with data_tab:
    st.info("Twelve Data history checks will appear here after API configuration.")

with research_tab:
    st.info("Walk-forward evaluation and model research will be added after data validation.")
