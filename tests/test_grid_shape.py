import numpy as np
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.sim.race import Entrant, grid_term, simulate_race

P = DEFAULT_PARAMS


def test_grid_term_is_concave_and_scaled():
    n = 20
    g = grid_term(np.arange(1, n + 1), n, P.replace(grid_bonus=10.0, grid_shape=1.0))
    assert g[0] == pytest.approx(200.0)
    assert g[1] == pytest.approx(100.0)
    assert g[-1] == pytest.approx(10.0)
    diffs = -np.diff(g)
    assert (np.diff(diffs) <= 1e-9).all()  # gaps shrink toward the back


def test_back_of_grid_still_wins_sometimes():
    # A dominant driver from P19 must not be priced at zero.
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.05) for i in range(20)]
    field[18] = Entrant("d18", "t9", 3400.0, 19, 0.05)
    f = simulate_race(field, P.replace(grid_bonus=8.0, grid_shape=1.0), n_runs=20000, seed=1)
    assert f.p_win[18] > 0.01


def test_no_grid_variance_matches_term_spread():
    from f1pred.sim.race import no_grid_sigma

    p = P.replace(grid_bonus=8.0, grid_shape=1.0)
    assert no_grid_sigma(20, p) == pytest.approx(np.std(grid_term(np.arange(1, 21), 20, p)))
