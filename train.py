from __future__ import annotations

import json
import os
import sys

from src.data.shared_feed import fetch_shared_market_data, load_universe_config
from src.research.pipeline import train_and_audit


def main() -> int:
    api_key = os.environ.get("TWELVE_DATA_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TWELVE_DATA_KEY is not set. Add it as a GitHub Actions repository secret."
        )
    config = load_universe_config()
    market_data = fetch_shared_market_data(
        api_key=api_key,
        symbols=config["symbols"],
        interval=config["interval"],
        output_size=int(config["history_bars"]),
    )
    report = train_and_audit(market_data, config)
    print(json.dumps({
        "sample_count": report["sample_count"],
        "active_symbols": report["active_symbols"],
        "selected_candidate": report["selected_candidate"],
        "promoted": report["promoted"],
        "promotion_reasons": report["promotion_reasons"],
        "model_failures": report["model_failures"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Training failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
