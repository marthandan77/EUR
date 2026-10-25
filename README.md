# EUR Asset Regime & Trend Breakout Backtester

A Streamlit research application for selected forex pairs and EUR-quoted metals. The app discovers the configured forex universe plus available EUR-quoted metals from Twelve Data and can load cached OHLC history for inspection. Causal feature engineering, barrier labels, and walk-forward backtesting are planned next.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Configure Twelve Data

Create `.streamlit/secrets.toml` locally with:

```toml
twelve_data_key = "YOUR_API_KEY"
```

Do not commit that file. For Streamlit Community Cloud, add the same key in the app's Secrets settings. The code reads the secret but never displays it.

