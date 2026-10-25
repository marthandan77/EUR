# EUR Asset Regime & Trend Breakout Backtester

A Streamlit research application for EUR-quoted forex pairs and precious metals. The initial commit provides the app shell; data discovery, causal feature engineering, and walk-forward backtesting will be added in subsequent milestones.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Configure Twelve Data

Create `.streamlit/secrets.toml` locally with:

```toml
twelve_data_key = "YOUR_API_KEY"
```

Do not commit that file. For Streamlit Community Cloud, add the same key in the app's Secrets settings.
