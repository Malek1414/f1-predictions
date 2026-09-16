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
    strength_wet: float | None = None
    # Driver profile (spec 6.5); all three are centred at 0 and inert while the scales are 0.
    aggression: float = 0.0
    risk: float = 0.0
    form: float = 0.0

    @property
    def wet_strength(self) -> float:
        return self.strength if self.strength_wet is None else self.strength_wet


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
    rain_probability: float = 0.0
    positions: np.ndarray | None = None  # (n_runs, n) when simulate_race(keep_positions=True)

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


def grid_term(grid: np.ndarray, n: int, params: ModelParams) -> np.ndarray:
    """Concave grid advantage (spec 2.1): pole gets `grid_bonus * n`, P2 half of that at shape 1,
    and the back rows are nearly indistinguishable."""
    return params.grid_bonus * n / np.asarray(grid, dtype=float) ** params.grid_shape


def no_grid_sigma(n: int, params: ModelParams) -> float:
    """Spread of the grid term over a full field, folded into driver noise without a grid."""
    return float(np.std(grid_term(np.arange(1, n + 1), n, params)))


def simulate_positions(
    entrants: list[Entrant],
    params: ModelParams,
    n_runs: int,
    rng: np.random.Generator,
    use_grid: bool = True,
    rain_probability: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """One finishing order per run. Returns (positions of shape (n_runs, n), used_grid).

    Spec 6.7: each run first draws wet ~ Bernoulli(rain_probability). Wet runs use the wet
    strength, scale both noise terms by wet_noise_factor and DNF odds by wet_dnf_factor.
    With rain_probability == 0 the draw is skipped so earlier seeds reproduce exactly.
    """
    n = len(entrants)
    strength = np.array([e.strength for e in entrants], dtype=float)
    p_dnf = np.array([e.p_dnf for e in entrants], dtype=float)
    used_grid = use_grid and all(e.grid is not None for e in entrants)

    # Spec 6.5: aggression and form as pace, risk as wider driver noise and more DNFs. No extra
    # RNG draws, and every adjustment is an exact no-op while its scale is 0.
    profile_pace = np.zeros(n)
    risk_noise = np.ones(n)
    if params.use_profile:
        aggression = np.array([e.aggression for e in entrants], dtype=float)
        risk = np.array([e.risk for e in entrants], dtype=float)
        form = np.array([e.form for e in entrants], dtype=float)
        profile_pace = params.aggression_scale * aggression + params.form_scale * form
        risk_noise = np.clip(1.0 + params.risk_noise_scale * risk, 0.5, 3.0)
        risk_dnf = 1.0 + params.risk_dnf_scale * risk
        # Clip only where the multiplier bites so a certain (or impossible) DNF stays exact.
        p_dnf = np.where(risk_dnf != 1.0, np.clip(p_dnf * risk_dnf, 0.01, 0.95), p_dnf)

    if rain_probability > 0.0:
        wet = rng.random(n_runs) < rain_probability
    else:
        wet = np.zeros(n_runs, dtype=bool)
    strength_wet = np.array([e.wet_strength for e in entrants], dtype=float)
    strength_run = np.where(wet[:, None], strength_wet[None, :], strength[None, :])
    noise_scale = np.where(wet, params.wet_noise_factor, 1.0)[:, None]

    teams = sorted({e.constructor_id for e in entrants})
    team_idx = np.array([teams.index(e.constructor_id) for e in entrants])
    team_noise = rng.normal(0.0, params.sigma_team, size=(n_runs, len(teams)))[:, team_idx]

    if used_grid:
        grid = np.array([e.grid for e in entrants], dtype=float)
        grid_pace = grid_term(grid, n, params)
        sigma_driver = params.sigma_driver
    else:
        grid_pace = np.zeros(n)
        # An unknown grid slot is a uniform draw over 1..n; fold its spread into driver noise.
        sigma_driver = float(np.sqrt(params.sigma_driver**2 + no_grid_sigma(n, params) ** 2))
    driver_noise = rng.normal(0.0, sigma_driver, size=(n_runs, n)) * risk_noise[None, :]

    performance = (
        strength_run
        + grid_pace
        + profile_pace[None, :]
        + team_noise * noise_scale
        + driver_noise * noise_scale
    )
    # Clip only the wet branch so a dry run keeps p_dnf exactly (a certain DNF stays certain).
    p_dnf_wet = np.clip(p_dnf * params.wet_dnf_factor, 0.0, 0.95)
    p_dnf_run = np.where(wet[:, None], p_dnf_wet[None, :], p_dnf[None, :])
    dnf = rng.random((n_runs, n)) < p_dnf_run
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
    rain_probability: float = 0.0,
    keep_positions: bool = False,
) -> RaceForecast:
    rng = np.random.default_rng(seed)
    n = len(entrants)
    positions, used_grid = simulate_positions(
        entrants, params, n_runs, rng, use_grid, rain_probability=rain_probability
    )
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
        rain_probability=rain_probability,
        positions=positions if keep_positions else None,
    )
