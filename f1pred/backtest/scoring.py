"""Probability scores and the two dumb baselines. Spec section 8.

Phase 7a adds the distribution-aware scores (RPS, top-3 set log loss, ECE, sharpness and skill
scores); each is defined in `docs/metrics.md`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6
SET_EPS = 1e-4
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


def rps(position_row: np.ndarray, actual: int) -> float:
    """Ranked probability score of one driver's finishing-position distribution.

    `sum_{k=1..n-1} (F(k) - 1[actual <= k])^2 / (n - 1)`: 0 when the whole mass sits on the
    actual position, 1 when it sits at the opposite end of the order.
    """
    p = np.asarray(position_row, dtype=float)
    n = len(p)
    if n < 2:
        return 0.0
    cdf = np.cumsum(p)[:-1]
    outcome = (np.arange(1, n) >= actual).astype(float)
    return float(((cdf - outcome) ** 2).sum() / (n - 1))


def mean_rps(
    position_matrix: np.ndarray, driver_ids: Sequence[str], actual: Mapping[str, int]
) -> float:
    """RPS averaged over the classified drivers (those with an actual position)."""
    scores = [rps(position_matrix[i], actual[d]) for i, d in enumerate(driver_ids) if d in actual]
    return float(np.mean(scores)) if scores else math.nan


def set_log_loss(p: float) -> float:
    """`-log p` for a set probability, floored at `SET_EPS` (a Monte Carlo zero is not a zero)."""
    return -math.log(min(max(p, SET_EPS), 1.0))


def top3_set_log_loss(positions: np.ndarray, driver_ids: Sequence[str], podium: set[str]) -> float:
    """Log loss of the exact podium set, from the `(n_runs, n)` matrix of simulated positions."""
    podium_mask = np.isin(np.asarray(driver_ids), list(podium))
    on_podium = np.asarray(positions) <= PODIUM
    hit = (on_podium == podium_mask[None, :]).all(axis=1)
    return set_log_loss(float(hit.mean()))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins: the count-weighted mean gap between
    the predicted and observed frequency in each bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def sharpness(p_win_per_race: Iterable[np.ndarray]) -> float:
    """Mean over races of the largest win probability: how decisive the forecasts are."""
    maxima = [float(np.max(p)) for p in p_win_per_race]
    return float(np.mean(maxima)) if maxima else math.nan


def skill_score(model: float, baseline: float) -> float:
    """`1 - model / baseline`: 0 matches the baseline, 1 is perfect, negative is worse."""
    if baseline == 0.0:
        return math.nan
    return 1.0 - model / baseline


@dataclass
class Baseline:
    p_win: dict[str, float]
    p_podium: dict[str, float]
    expected_position: dict[str, float]
    # Pole baseline: each driver's grid slot rank (1..n); None for the uniform baseline.
    slot: dict[str, int] | None = None

    def position_matrix(self, driver_ids: Sequence[str]) -> np.ndarray:
        """`(n, n)` finishing-position distribution, row i for driver i of `driver_ids`."""
        n = len(driver_ids)
        if self.slot is None:
            return np.full((n, n), 1.0 / n)
        rest = (1.0 - POLE_MASS) / max(n - 1, 1)
        rows = np.full((n, n), rest)
        for i, d in enumerate(driver_ids):
            rows[i] = rest if n > 1 else 0.0
            rows[i, min(self.slot[d], n) - 1] = POLE_MASS if n > 1 else 1.0
        return rows

    def top3_set_prob(self, podium: set[str]) -> float:
        n = len(self.p_win)
        n_sets = math.comb(n, PODIUM) if n >= PODIUM else 1
        if self.slot is None:
            return 1.0 / n_sets
        top = {d for d, s in self.slot.items() if s <= PODIUM}
        return POLE_MASS if top == set(podium) else (1.0 - POLE_MASS) / n_sets


def pole_baseline(grid: Mapping[str, int]) -> Baseline:
    """90% on the pole sitter (top three for podium), the rest spread evenly."""
    n = len(grid)
    ordered = sorted(grid, key=lambda d: grid[d])
    rest_win = (1.0 - POLE_MASS) / max(n - 1, 1)
    p_win = {d: (POLE_MASS if i == 0 else rest_win) for i, d in enumerate(ordered)}
    k = min(PODIUM, n)
    rest_podium = (PODIUM - POLE_MASS * k) / max(n - k, 1)
    p_podium = {d: (POLE_MASS if i < k else rest_podium) for i, d in enumerate(ordered)}
    slot = {d: i + 1 for i, d in enumerate(ordered)}
    return Baseline(p_win, p_podium, {d: float(grid[d]) for d in grid}, slot)


def uniform_baseline(driver_ids: Iterable[str]) -> Baseline:
    ids = list(driver_ids)
    n = len(ids)
    mid = (n + 1) / 2
    return Baseline({d: 1.0 / n for d in ids}, {d: PODIUM / n for d in ids}, {d: mid for d in ids})
