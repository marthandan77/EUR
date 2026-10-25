from __future__ import annotations

import math


MODEL_RETRAIN_KEYS = (
    "interval",
    "lookback_bars",
    "holding_bars",
    "target_atr",
    "stop_atr",
    "minimum_expected_net_r",
    "cost_estimates_verified",
)


def training_config_mismatches(current: dict, trained: dict) -> list[str]:
    """Return setting names that make stored models incompatible with current settings."""
    mismatches = []
    for key in MODEL_RETRAIN_KEYS:
        current_value = current.get(key)
        trained_value = trained.get(key)
        if isinstance(current_value, (int, float)) and isinstance(trained_value, (int, float)):
            matches = math.isclose(float(current_value), float(trained_value), rel_tol=1e-10, abs_tol=1e-12)
        else:
            matches = current_value == trained_value
        if not matches:
            mismatches.append(key)

    def symbols(config: dict) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(
            (
                str(item.get("symbol", "")).casefold(),
                str(item.get("asset_class", "")).casefold(),
            )
            for item in config.get("symbols", [])
        ))

    if symbols(current) != symbols(trained):
        mismatches.append("symbols")

    def costs(config: dict) -> dict[str, float]:
        return {
            str(symbol): float(value)
            for symbol, value in config.get("round_trip_cost_bps", {}).items()
            if value is not None
        }

    if costs(current) != costs(trained):
        mismatches.append("round_trip_cost_bps")
    return mismatches
