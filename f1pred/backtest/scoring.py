"""Probability scores and the two dumb baselines. Spec section 8."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

EPS = 1e-6
POLE_MASS = 0.9
PODIUM = 3


def winner_log_loss(p_win: Mapping[str, float], winner: str) -> float:
    p = min(max(p_win.get(winner, 0.0), EPS), 1.0)
    return -math.log(p)


def podium_brier(p_podium: Mapping[str, float], podium: set[str]) -> float:
    terms = [(p - (1.0 if d in podium else 0.0)) ** 2 for d, p in p_podium.items()]
    return sum(terms) / len(terms)


def position_spearman(expected: Mapping[str, float], actual: Mapping[str, int]) -> float:
    common = [d for d in expected if d in actual]
    if len(common) < 3:
        return math.nan
    e = pd.Series([expected[d] for d in common]).rank()
    a = pd.Series([actual[d] for d in common]).rank()
    # Spearman is Pearson on ranks; pandas' method="spearman" would pull in scipy.
    if e.nunique() < 2 or a.nunique() < 2:
        return math.nan  # a constant vector has no rank correlation
    return float(e.corr(a))


@dataclass
class Baseline:
    p_win: dict[str, float]
    p_podium: dict[str, float]
    expected_position: dict[str, float]


def pole_baseline(grid: Mapping[str, int]) -> Baseline:
    """90% on the pole sitter (top three for podium), the rest spread evenly."""
    n = len(grid)
    ordered = sorted(grid, key=lambda d: grid[d])
    rest_win = (1.0 - POLE_MASS) / max(n - 1, 1)
    p_win = {d: (POLE_MASS if i == 0 else rest_win) for i, d in enumerate(ordered)}
    k = min(PODIUM, n)
    rest_podium = (PODIUM - POLE_MASS * k) / max(n - k, 1)
    p_podium = {d: (POLE_MASS if i < k else rest_podium) for i, d in enumerate(ordered)}
    return Baseline(p_win, p_podium, {d: float(grid[d]) for d in grid})


def uniform_baseline(driver_ids: Iterable[str]) -> Baseline:
    ids = list(driver_ids)
    n = len(ids)
    mid = (n + 1) / 2
    return Baseline({d: 1.0 / n for d in ids}, {d: PODIUM / n for d in ids}, {d: mid for d in ids})
