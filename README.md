# EUR Asset Regime & Trend Breakout Advisor

A research system using one shared Twelve Data OHLC feed for three candidate classifiers: gradient-boosted trees, an LSTM, and a temporal convolutional network (TCN). It predicts whether an ATR target is reached first, an ATR stop is reached first, or neither barrier is reached within the configured holding period. It does not forecast exact prices.

## Active universe and experiment settings

Edit [`config/universe.json`](config/universe.json) to add or remove tickers. Every configured symbol is fetched once into the shared panel used by all models. Adding a symbol triggers feature rebuilding and retraining; the existing model set is not used for trade advice until it has been retrained on the same universe.

The initial experiment uses hourly bars, a 64-bar model window, a 24-bar maximum holding period, a 2×ATR target, and a 1×ATR stop. These are research defaults, not optimized trading parameters. Update them only as explicit, recorded experiments.

`round_trip_cost_bps` contains provisional conservative scenario assumptions for every currently active FX symbol: 20 bps for AUD/EUR, CAD/EUR, CHF/EUR, GBP/EUR, and JPY/EUR; 200 bps for IDR/EUR; 30 bps for NZD/EUR; and 25 bps for AUD/NZD. They are round-trip costs as a fraction of notional and are not broker quotes or measured historical costs. Twelve Data OHLC bars cannot establish broker spreads, commissions, or slippage. `cost_estimates_verified` is therefore `false`; promotion and trade recommendations remain disabled until you replace these assumptions with broker-specific estimates and explicitly set it to `true`.

## Shared data and model workflow

`src/data/shared_feed.py` fetches one dataset per configured ticker. The shared feature builder creates causal log-return, rolling-volatility, ATR-normalized price-action, and cross-market context features. It does not forward-fill missing prices. Features at time `t` use data through the completed bar at `t`; labels enter at the next bar open and use only subsequent bars. Same-bar target/stop ambiguity resolves as a stop, with adverse gap fills on stops.

If Twelve Data is unavailable, the Streamlit **Shared market data** tab accepts uploaded OHLC CSV files. Use one combined file with `datetime,open,high,low,close,symbol` columns, or one file per configured symbol named like `AUD_EUR.csv`. The loader validates prices and timestamps and drops the latest row per ticker because an exported bar may still be forming. Uploaded data supports inspection in the app; the scheduled trainer still requires Twelve Data, and the provisional cost flag continues to block model promotion and advice.

The Streamlit data tab can display successfully fetched tickers when another configured symbol fails. This partial panel is for inspection only. The scheduled trainer remains all-or-nothing and requires data for every configured symbol, preventing a reduced universe from being labeled and promoted as the complete one.

The scheduled GitHub Actions workflow runs weekly. It trains all three models on identical chronological folds, calibrates probabilities on a later slice inside each training window, evaluates on walk-forward periods, and reserves the latest 15% of timestamps as a holdout. Promotion requires configured costs, enough trades, positive net results across multiple folds, and positive net results on the holdout. A failed candidate does not replace the existing production set.

The advisor requires all three calibrated model predictions to agree within a configured tolerance, positive expected net R above the configured threshold, a matching trained ticker universe, and costs for all symbols. Timeout payoffs are estimated from matured training outcomes by symbol and direction; the held-out backtest uses each sample's realized timeout return. Otherwise the advisor reports **No trade**. It also withholds signals when the latest scheduled audit fails promotion or model-defining settings no longer match the stored model. Current bar-close prices are signal references, not executable quotes.

## Configure secrets

For local Streamlit use, create `.streamlit/secrets.toml`:

```toml
twelve_data_key = "YOUR_API_KEY"
```

Never commit that file. For the scheduled trainer, add the same Twelve Data key as a GitHub repository Actions secret named `TWELVE_DATA_KEY` under **Settings → Secrets and variables → Actions**. Streamlit Secrets and GitHub Actions secrets are separate settings.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the full research training cycle locally with `TWELVE_DATA_KEY` set in the environment:

```powershell
$env:TWELVE_DATA_KEY = "YOUR_API_KEY"
python train.py
```

The same workflow can be started manually from GitHub's **Actions → Scheduled model learning → Run workflow**. Scheduled jobs use the latest `main` branch. GitHub Actions may delay scheduled starts during busy periods; this is a weekly research cadence, not a real-time execution service.

## Artifacts and persistence

The latest evaluation report is written to `artifacts/latest_report.json`. Promoted model files and their metadata are written to `artifacts/production/` and committed by the workflow, allowing Streamlit Community Cloud to load the latest set from the repository. This is suitable for compact research models; a larger or higher-frequency deployment should move market history and model artifacts to durable database/object storage rather than growing the Git history.

## Important research boundary

The backtest is a candidate-selection tool, not evidence of live profitability. It uses OHLC bars and configurable cost estimates, has no bid/ask execution stream, and does not model market impact or portfolio-level capital allocation. Inspect the held-out results, per-symbol behavior, model calibration, and actual broker execution before relying on any signal.
