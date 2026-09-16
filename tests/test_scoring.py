import math

import numpy as np
import pytest

from f1pred.backtest.scoring import (
    ece,
    mean_rps,
    podium_brier,
    pole_baseline,
    position_spearman,
    rps,
    sharpness,
    skill_score,
    top3_set_log_loss,
    uniform_baseline,
    winner_log_loss,
)


def test_winner_log_loss():
    assert winner_log_loss({"a": 0.5, "b": 0.5}, "a") == pytest.approx(math.log(2))
    assert winner_log_loss({"a": 1.0}, "a") == pytest.approx(0.0)
    assert winner_log_loss({"a": 1.0}, "zzz") == pytest.approx(-math.log(1e-6))


def test_podium_brier():
    p = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
    assert podium_brier(p, {"a", "b", "c"}) == 0.0
    assert podium_brier({"a": 0.5, "b": 0.5}, {"a"}) == pytest.approx(0.25)


def test_position_spearman():
    assert position_spearman({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}) == pytest.approx(
        1.0
    )
    assert position_spearman({"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 2, "c": 1}) == pytest.approx(
        -1.0
    )
    assert math.isnan(position_spearman({"a": 1, "b": 2}, {"a": 1, "b": 2}))


def test_pole_baseline():
    b = pole_baseline({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    assert b.p_win["a"] == pytest.approx(0.9)
    assert b.p_win["e"] == pytest.approx(0.025)
    assert sum(b.p_win.values()) == pytest.approx(1.0)
    assert b.p_podium["b"] == pytest.approx(0.9)
    assert b.p_podium["d"] == pytest.approx(0.15)
    assert sum(b.p_podium.values()) == pytest.approx(3.0)
    assert b.expected_position == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


def test_uniform_baseline():
    b = uniform_baseline(["a", "b", "c", "d"])
    assert b.p_win == pytest.approx({k: 0.25 for k in "abcd"})
    assert b.p_podium == pytest.approx({k: 0.75 for k in "abcd"})
    assert len(set(b.expected_position.values())) == 1


def test_position_spearman_constant_vector_is_nan_without_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(position_spearman({"a": 1, "b": 2, "c": 3}, {"a": 20, "b": 20, "c": 20}))


def test_rps_perfect_and_worst():
    p = np.zeros(5)
    p[2] = 1.0  # certain P3
    assert rps(p, 3) == 0.0
    assert rps(p, 5) == pytest.approx((1 + 1) / 4)  # F=1 at k=3,4 while 1[a<=k]=0 -> two unit terms
    flat = np.full(5, 0.2)
    assert rps(flat, 1) == pytest.approx(
        ((0.2 - 1) ** 2 + (0.4 - 1) ** 2 + (0.6 - 1) ** 2 + (0.8 - 1) ** 2) / 4
    )


def test_top3_set_log_loss():
    positions = np.array([[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 4, 3], [4, 3, 2, 1]])
    ids = ["a", "b", "c", "d"]
    assert top3_set_log_loss(positions, ids, {"a", "b", "c"}) == pytest.approx(-math.log(0.5))
    assert top3_set_log_loss(positions, ids, {"b", "c", "d"}) == pytest.approx(-math.log(0.25))
    # Run 3 ([1, 2, 4, 3]) puts d on the podium, so {a, b, d} is 1/4; {a, c, d} never happens.
    assert top3_set_log_loss(positions, ids, {"a", "b", "d"}) == pytest.approx(-math.log(0.25))
    assert top3_set_log_loss(positions, ids, {"a", "c", "d"}) == pytest.approx(-math.log(1e-4))


def test_ece_and_sharpness_and_skill():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y, bins=10) == pytest.approx(0.1)
    assert sharpness([np.array([0.5, 0.5]), np.array([0.9, 0.1])]) == pytest.approx(0.7)
    assert skill_score(1.0, 2.0) == pytest.approx(0.5)


def test_mean_rps_skips_unclassified_and_baseline_matrices():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ids = ["a", "b", "c"]
    # c did not finish: only a (P2) and b (P1) count.
    assert mean_rps(matrix, ids, {"a": 2, "b": 1}) == pytest.approx(
        (rps(matrix[0], 2) + rps(matrix[1], 1)) / 2
    )
    pole = pole_baseline({"a": 1, "b": 2, "c": 3, "d": 4})
    assert pole.position_matrix(["a", "b", "c", "d"])[0].tolist() == pytest.approx(
        [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
    )
    assert pole.top3_set_prob({"a", "b", "c"}) == pytest.approx(0.9)
    assert pole.top3_set_prob({"a", "b", "d"}) == pytest.approx(0.1 / 4)
    uni = uniform_baseline(["a", "b", "c", "d"])
    np.testing.assert_allclose(uni.position_matrix(["a", "b", "c", "d"]), np.full((4, 4), 0.25))
    assert uni.top3_set_prob({"a", "b", "c"}) == pytest.approx(0.25)
