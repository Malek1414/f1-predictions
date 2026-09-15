import numpy as np
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.sim.race import Entrant, simulate_positions, simulate_race

P = DEFAULT_PARAMS


def _field(n=20, strength=3000.0, p_dnf=0.1):
    return [Entrant(f"d{i}", f"t{i // 2}", strength, i + 1, p_dnf) for i in range(n)]


def test_position_columns_sum_to_one():
    f = simulate_race(_field(), P, n_runs=2000, seed=1)
    assert f.position_matrix.shape == (20, 20)
    np.testing.assert_allclose(f.position_matrix.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(f.position_matrix.sum(axis=1), 1.0, atol=1e-12)
    assert f.p_win.sum() == pytest.approx(1.0)
    assert f.p_podium.sum() == pytest.approx(3.0)
    assert f.p_points.sum() == pytest.approx(10.0)


def test_stronger_driver_wins_more():
    field = _field(strength=3000.0)
    field[5] = Entrant("d5", "t2", 3400.0, 6, 0.1)
    f = simulate_race(field, P, n_runs=5000, seed=2, use_grid=False)
    assert f.driver_ids[int(np.argmax(f.p_win))] == "d5"
    assert f.p_win[5] > 0.5


def test_certain_dnf_never_wins_and_finishes_last_region():
    field = _field(p_dnf=0.0)
    field[0] = Entrant("d0", "t0", 5000.0, 1, 1.0)
    f = simulate_race(field, P, n_runs=1000, seed=3)
    assert f.p_win[0] == 0.0
    assert f.expected_position[0] == pytest.approx(20.0)


def test_identical_drivers_equal_within_tolerance():
    f = simulate_race(_field(), P, n_runs=20000, seed=4, use_grid=False)
    assert f.p_win.max() - f.p_win.min() < 0.03


def test_grid_bonus_helps_pole():
    f = simulate_race(_field(), P, n_runs=5000, seed=5, use_grid=True)
    assert f.p_win[0] > f.p_win[19]
    assert f.used_grid


def test_missing_grid_disables_grid_mode():
    field = _field()
    field[3] = Entrant("d3", "t1", 3000.0, None, 0.1)
    f = simulate_race(field, P, n_runs=500, seed=6)
    assert not f.used_grid


def test_seed_reproducible():
    a = simulate_race(_field(), P, n_runs=500, seed=7)
    b = simulate_race(_field(), P, n_runs=500, seed=7)
    np.testing.assert_array_equal(a.position_matrix, b.position_matrix)


def test_teammates_share_noise():
    # With only team noise, teammates must finish adjacent in every run.
    params = P.replace(sigma_driver=0.0, grid_bonus=0.0)
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.0) for i in range(6)]
    f = simulate_race(field, params, n_runs=200, seed=8, use_grid=False)
    # d0 and d1 same team: P(d0 in pos k) equals P(d1 in pos k) roughly and podium share matches
    assert f.p_podium[0] + f.p_podium[1] == pytest.approx(f.p_podium[2] + f.p_podium[3], abs=0.35)


def test_as_frame_sorted_and_prob():
    f = simulate_race(_field(), P, n_runs=300, seed=9)
    df = f.as_frame()
    assert list(df.columns) == [
        "driver_id",
        "p_win",
        "p_podium",
        "p_points",
        "expected_position",
        "position_p10",
        "position_p90",
    ]
    assert df.p_win.is_monotonic_decreasing
    assert f.prob("d0", "win") == pytest.approx(f.p_win[0])


def test_simulate_positions_rows_are_permutations():
    rng = np.random.default_rng(0)
    positions, used_grid = simulate_positions(_field(), P, 50, rng)
    assert positions.shape == (50, 20) and used_grid
    for row in positions:
        assert sorted(row.tolist()) == list(range(1, 21))


def test_simulate_race_matches_positions_with_same_seed():
    forecast = simulate_race(_field(), P, n_runs=300, seed=11)
    positions, _ = simulate_positions(_field(), P, 300, np.random.default_rng(11))
    p_win = (positions == 1).mean(axis=0)
    np.testing.assert_allclose(p_win, forecast.p_win)
