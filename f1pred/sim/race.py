"""Vectorised Monte Carlo of one race. Spec section 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.config import ModelParams

PODIUM = 3
POINTS_POSITIONS = 10
DNF_KEY = -1e9  # anything below every real performance


@dataclass(frozen=True)
class Entrant:
    driver_id: str
    constructor_id: str
    strength: float
    grid: int | None
    p_dnf: float
    low_confidence: bool = False


@dataclass
class RaceForecast:
    driver_ids: list[str]
    p_win: np.ndarray
    p_podium: np.ndarray
    p_points: np.ndarray
    expected_position: np.ndarray
    position_p10: np.ndarray
    position_p90: np.ndarray
    position_matrix: np.ndarray  # (n_entrants, n_positions), row i = P(driver i finishes k+1)
    n_runs: int
    used_grid: bool

    def as_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "driver_id": self.driver_ids,
                "p_win": self.p_win,
                "p_podium": self.p_podium,
                "p_points": self.p_points,
                "expected_position": self.expected_position,
                "position_p10": self.position_p10,
                "position_p90": self.position_p90,
            }
        )
        return df.sort_values("p_win", ascending=False, kind="stable").reset_index(drop=True)

    def prob(self, driver_id: str, kind: str) -> float:
        i = self.driver_ids.index(driver_id)
        return float({"win": self.p_win, "podium": self.p_podium, "points": self.p_points}[kind][i])


def simulate_positions(
    entrants: list[Entrant],
    params: ModelParams,
    n_runs: int,
    rng: np.random.Generator,
    use_grid: bool = True,
) -> tuple[np.ndarray, bool]:
    """One finishing order per run. Returns (positions of shape (n_runs, n), used_grid)."""
    n = len(entrants)
    strength = np.array([e.strength for e in entrants], dtype=float)
    p_dnf = np.array([e.p_dnf for e in entrants], dtype=float)
    used_grid = use_grid and all(e.grid is not None for e in entrants)

    teams = sorted({e.constructor_id for e in entrants})
    team_idx = np.array([teams.index(e.constructor_id) for e in entrants])
    team_noise = rng.normal(0.0, params.sigma_team, size=(n_runs, len(teams)))[:, team_idx]

    if used_grid:
        grid = np.array([e.grid for e in entrants], dtype=float)
        grid_term = params.grid_bonus * (n - grid)
        sigma_driver = params.sigma_driver
    else:
        grid_term = np.zeros(n)
        # A uniform draw over n grid slots has std n / sqrt(12); fold that into driver noise.
        sigma_driver = float(np.sqrt(params.sigma_driver**2 + (params.grid_bonus * n) ** 2 / 12))
    driver_noise = rng.normal(0.0, sigma_driver, size=(n_runs, n))

    performance = strength + grid_term + team_noise + driver_noise
    dnf = rng.random((n_runs, n)) < p_dnf
    key = np.where(dnf, DNF_KEY + rng.random((n_runs, n)), performance)

    order = np.argsort(-key, axis=1, kind="stable")  # order[r, k] = entrant index at position k+1
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.arange(1, n + 1)[None, :].repeat(n_runs, 0), axis=1)
    return positions, used_grid


def simulate_race(
    entrants: list[Entrant],
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int | None = None,
    use_grid: bool = True,
) -> RaceForecast:
    rng = np.random.default_rng(seed)
    n = len(entrants)
    positions, used_grid = simulate_positions(entrants, params, n_runs, rng, use_grid)
    position_matrix = (
        np.stack([np.bincount(positions[:, i] - 1, minlength=n) for i in range(n)]).astype(float)
        / n_runs
    )
    ranks = np.arange(1, n + 1)
    p10, p90 = np.percentile(positions, [10, 90], axis=0)
    return RaceForecast(
        driver_ids=[e.driver_id for e in entrants],
        p_win=position_matrix[:, 0].copy(),
        p_podium=position_matrix[:, :PODIUM].sum(axis=1),
        p_points=position_matrix[:, :POINTS_POSITIONS].sum(axis=1),
        expected_position=(position_matrix * ranks).sum(axis=1),
        position_p10=p10,
        position_p90=p90,
        position_matrix=position_matrix,
        n_runs=n_runs,
        used_grid=used_grid,
    )
