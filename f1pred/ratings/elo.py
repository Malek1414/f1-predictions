"""Pairwise Elo for drivers and constructors. Spec section 5."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from f1pred.config import ModelParams

Finisher = tuple[str, str, int]  # driver_id, constructor_id, classified position


@dataclass
class RaceUpdate:
    driver_delta: dict[str, float] = field(default_factory=dict)
    constructor_delta: dict[str, float] = field(default_factory=dict)


def expected_score(strength_a: float, strength_b: float) -> float:
    """Probability that A finishes ahead of B."""
    return 1.0 / (1.0 + 10 ** ((strength_b - strength_a) / 400.0))


def race_update(
    finishers: list[Finisher],
    driver_ratings: Mapping[str, float],
    constructor_ratings: Mapping[str, float],
    params: ModelParams,
    k_scale: float = 1.0,
) -> RaceUpdate:
    """Rating changes from one session. Every pair of classified finishers is one matchup.

    The driver ahead scores 1, the driver behind 0. Each side moves by K * (actual - expected).
    Teammate pairs are weighted by teammate_weight for drivers and skipped for constructors.
    Deltas are scaled by 1 / (n - 1) so field size does not change the rating speed.
    """
    ordered = sorted(finishers, key=lambda f: f[2])
    n = len(ordered)
    if n < 2:
        return RaceUpdate()
    scale = 1.0 / (n - 1)
    dd: dict[str, float] = defaultdict(float)
    dc: dict[str, float] = defaultdict(float)
    for i in range(n):
        di, ci, _ = ordered[i]
        s_i = driver_ratings[di] + constructor_ratings[ci]
        for j in range(i + 1, n):
            dj, cj, _ = ordered[j]
            s_j = driver_ratings[dj] + constructor_ratings[cj]
            surprise = 1.0 - expected_score(s_i, s_j)
            same_team = ci == cj
            w = params.teammate_weight if same_team else 1.0
            step_d = params.k_driver * k_scale * w * surprise * scale
            dd[di] += step_d
            dd[dj] -= step_d
            if not same_team:
                step_c = params.k_constructor * k_scale * surprise * scale
                dc[ci] += step_c
                dc[cj] -= step_c
    return RaceUpdate(dict(dd), dict(dc))


def apply_update(ratings: dict[str, float], delta: Mapping[str, float]) -> None:
    for key, change in delta.items():
        ratings[key] = ratings.get(key, 0.0) + change


def regress(ratings: dict[str, float], fraction: float, initial: float) -> None:
    """Pull every rating toward `initial` by `fraction` of its distance."""
    for key, value in ratings.items():
        ratings[key] = initial + (value - initial) * (1.0 - fraction)
