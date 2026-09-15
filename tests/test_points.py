import numpy as np
import pytest

from f1pred.sim.points import (
    had_fastest_lap_bonus,
    points_for_positions,
    race_points,
    sprint_points,
)


def test_race_points_modern():
    np.testing.assert_array_equal(race_points(2026), [25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
    np.testing.assert_array_equal(race_points(2010), race_points(2026))


def test_sprint_points_by_season():
    assert sprint_points(2020).size == 0
    np.testing.assert_array_equal(sprint_points(2021), [3, 2, 1])
    np.testing.assert_array_equal(sprint_points(2024), [8, 7, 6, 5, 4, 3, 2, 1])


@pytest.mark.parametrize(
    "season,expected", [(2018, False), (2019, True), (2024, True), (2025, False)]
)
def test_fastest_lap_bonus(season, expected):
    assert had_fastest_lap_bonus(season) is expected


def test_points_for_positions_shapes_and_zero_beyond_table():
    pos = np.array([[1, 2, 11, 20], [10, 3, 1, 5]])
    out = points_for_positions(pos, race_points(2025))
    np.testing.assert_array_equal(out, [[25, 18, 0, 0], [1, 15, 25, 10]])
    assert out.dtype == float


def test_points_for_positions_empty_table():
    out = points_for_positions(np.array([1, 2, 3]), sprint_points(2020))
    np.testing.assert_array_equal(out, [0, 0, 0])
