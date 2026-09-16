"""Qualifying lap times as a percentage gap to pole. Spec 2.2 (qualifying likelihood).

The best qualifying lap is the cleanest pre-race measure of car pace and is available for
almost every driver-race since 2010, so the Bayesian pace model treats it as a second
observation of the same latent pace.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

QUALIFYING_COLUMNS = ("race_id", "driver_id", "quali_position", "best_ms", "gap_pct")
SEGMENTS = ("q1", "q2", "q3")


def parse_lap_ms(text: str | float | None) -> float | None:
    """`"1:26.572"` -> 86572.0, `"59.999"` -> 59999.0, anything blank or NaN -> None."""
    if text is None:
        return None
    if isinstance(text, float) and math.isnan(text):
        return None
    s = str(text).strip()
    if not s or s in {"\\N", "nan", "None"}:
        return None
    minutes = 0.0
    if ":" in s:
        head, s = s.split(":", 1)
        minutes = float(head)
    try:
        seconds = float(s)
    except ValueError:
        return None
    return round(1000.0 * (60.0 * minutes + seconds), 3)


def qualifying_gaps(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per qualifying entry: the driver's best lap and its percent gap to pole.

    `best_ms` is the fastest of Q1, Q2 and Q3; `gap_pct = 100 * (best_ms - pole_ms) / pole_ms`
    where `pole_ms` is the fastest `best_ms` of that race. A driver who set no time keeps a
    `quali_position` but gets NaN for both.
    """
    q = raw["qualifying"]
    drivers = raw["drivers"][["driverId", "driverRef"]]
    df = q.merge(drivers, on="driverId", how="left")
    best = []
    for row in df[list(SEGMENTS)].itertuples(index=False):
        times = [t for t in (parse_lap_ms(v) for v in row) if t is not None]
        best.append(min(times) if times else np.nan)
    out = pd.DataFrame(
        {
            "race_id": df["raceId"].astype(int),
            "driver_id": df["driverRef"].astype(str),
            "quali_position": pd.to_numeric(df["position"], errors="coerce").astype("Int64"),
            "best_ms": np.asarray(best, dtype=float),
        }
    )
    pole = out.groupby("race_id")["best_ms"].transform("min")
    out["gap_pct"] = 100.0 * (out["best_ms"] - pole) / pole
    out = out.sort_values(["race_id", "quali_position"], kind="stable").reset_index(drop=True)
    return out[list(QUALIFYING_COLUMNS)]
