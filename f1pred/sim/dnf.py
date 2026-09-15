"""Per-driver DNF probability from recent driver, constructor and circuit history. Spec 6.3."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from f1pred.config import ModelParams

DEFAULT_GLOBAL_RATE = 0.1
P_MIN, P_MAX = 0.01, 0.9


def _rate(flags: pd.Series, window: int) -> tuple[float, int]:
    recent = flags.tail(window)
    if len(recent) == 0:
        return 0.0, 0
    return float(recent.mean()), int(len(recent))


def _global_rate(past: pd.DataFrame, params: ModelParams) -> float:
    recent = past.tail(params.dnf_window * 20)
    return float(recent["dnf"].mean()) if len(recent) else DEFAULT_GLOBAL_RATE


def _circuit_factor(
    past: pd.DataFrame, circuit_id: str, global_rate: float, params: ModelParams
) -> float:
    circ = past[past["circuit_id"] == circuit_id]
    if len(circ) < params.circuit_min_rows or global_rate <= 0:
        return 1.0
    return float(
        np.clip(
            circ["dnf"].mean() / global_rate, params.circuit_factor_min, params.circuit_factor_max
        )
    )


def _probability(
    past: pd.DataFrame, driver_id: str, constructor_id: str, circuit_id: str, params: ModelParams
) -> float:
    global_rate = _global_rate(past, params)
    d_rate, n_d = _rate(past.loc[past["driver_id"] == driver_id, "dnf"], params.dnf_window)
    c_rate, n_c = _rate(
        past.loc[past["constructor_id"] == constructor_id, "dnf"], params.dnf_window * 2
    )
    if n_d and n_c:
        raw, n = (d_rate + c_rate) / 2, (n_d + n_c) / 2
    elif n_d:
        raw, n = d_rate, n_d
    elif n_c:
        raw, n = c_rate, n_c
    else:
        raw, n = global_rate, 0
    shrunk = global_rate + (raw - global_rate) * n / (n + params.shrink_dnf)
    p = shrunk * _circuit_factor(past, circuit_id, global_rate, params)
    return float(np.clip(p, P_MIN, P_MAX))


def dnf_probability(
    table: pd.DataFrame,
    driver_id: str,
    constructor_id: str,
    circuit_id: str,
    before_date: pd.Timestamp,
    params: ModelParams,
) -> float:
    past = table[(~table["is_sprint"]) & (table["date"] < before_date)]
    return _probability(past, driver_id, constructor_id, circuit_id, params)


def dnf_cache_for_races(
    table: pd.DataFrame, race_ids: Iterable[int], params: ModelParams
) -> dict[tuple[int, str], float]:
    """Pre-race p_dnf for every entrant of every race in `race_ids`, keyed (race_id, driver_id)."""
    races = table[~table["is_sprint"]]
    out: dict[tuple[int, str], float] = {}
    for race_id in race_ids:
        rows = races[races["race_id"] == race_id]
        if rows.empty:
            continue
        date = rows["date"].iloc[0]
        past = races[races["date"] < date]
        for r in rows.itertuples(index=False):
            out[(int(race_id), r.driver_id)] = _probability(
                past, r.driver_id, r.constructor_id, r.circuit_id, params
            )
    return out
