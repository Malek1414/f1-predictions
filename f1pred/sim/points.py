"""Points rules by season. Spec section 7."""

from __future__ import annotations

import numpy as np

RACE_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
SPRINT_POINTS_2021 = np.array([3, 2, 1], dtype=float)
SPRINT_POINTS_2022 = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)


def race_points(season: int) -> np.ndarray:
    return RACE_POINTS.copy()


def sprint_points(season: int) -> np.ndarray:
    if season < 2021:
        return np.array([], dtype=float)
    if season == 2021:
        return SPRINT_POINTS_2021.copy()
    return SPRINT_POINTS_2022.copy()


def had_fastest_lap_bonus(season: int) -> bool:
    """One bonus point for the fastest lap by a top-10 finisher, 2019 to 2024. Not simulated."""
    return 2019 <= season <= 2024


def points_for_positions(positions: np.ndarray, table: np.ndarray) -> np.ndarray:
    padded = np.concatenate([table.astype(float), np.zeros(1)])
    idx = np.clip(np.asarray(positions) - 1, 0, len(table))
    return padded[idx]
