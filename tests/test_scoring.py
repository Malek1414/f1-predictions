import math

import pytest

from f1pred.backtest.scoring import (
    podium_brier,
    pole_baseline,
    position_spearman,
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
