"""Bootstrap confidence intervals on backtest means (docs/metrics.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

INTERVAL_COLUMNS = ["metric", "mean", "lo", "hi"]


def bootstrap_intervals(
    races: pd.DataFrame,
    columns: list[str],
    R: int = 1000,
    seed: int = 0,
    level: float = 0.9,
) -> pd.DataFrame:
    """Resample races with replacement `R` times and report the central `level` interval of
    each column's mean. One row per column: `metric, mean, lo, hi`."""
    rng = np.random.default_rng(seed)
    n = len(races)
    if n == 0:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    idx = rng.integers(0, n, size=(R, n))
    lo_q, hi_q = 100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0
    rows = []
    for c in columns:
        v = races[c].to_numpy(dtype=float)
        means = np.nanmean(v[idx], axis=1)
        lo, hi = np.percentile(means, [lo_q, hi_q])
        rows.append({"metric": c, "mean": float(np.nanmean(v)), "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
