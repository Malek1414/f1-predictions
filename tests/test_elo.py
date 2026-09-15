import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.elo import apply_update, expected_score, race_update, regress

P = DEFAULT_PARAMS  # k_driver 24, k_constructor 32, teammate_weight 2


def test_expected_score_symmetry_and_scale():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) == pytest.approx(1 / (1 + 10 ** (-0.5)))
    assert expected_score(1500, 1700) + expected_score(1700, 1500) == pytest.approx(1.0)


def test_two_drivers_different_teams_hand_computed():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1500.0, "y": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "y", 2)], d, c, P)
    # expected 0.5, surprise 0.5, n=2 so scale 1
    assert upd.driver_delta == pytest.approx({"a": 12.0, "b": -12.0})
    assert upd.constructor_delta == pytest.approx({"x": 16.0, "y": -16.0})


def test_teammates_weighted_and_no_constructor_change():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "x", 2)], d, c, P)
    assert upd.driver_delta == pytest.approx({"a": 24.0, "b": -24.0})
    assert upd.constructor_delta.get("x", 0.0) == pytest.approx(0.0)


def test_scaled_by_field_size_and_zero_sum():
    d = {f"d{i}": 1500.0 for i in range(20)}
    c = {f"c{i // 2}": 1500.0 for i in range(20)}
    finishers = [(f"d{i}", f"c{i // 2}", i + 1) for i in range(20)]
    upd = race_update(finishers, d, c, P)
    # winner beats 19 drivers at surprise 0.5, one of them a teammate (weight 2): (18 + 2) * 12 / 19
    assert upd.driver_delta["d0"] == pytest.approx(20 * 12 / 19)
    assert sum(upd.driver_delta.values()) == pytest.approx(0.0)
    assert sum(upd.constructor_delta.values()) == pytest.approx(0.0)


def test_strength_uses_driver_plus_constructor():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1700.0, "y": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "y", 2)], d, c, P)
    surprise = 1 - expected_score(3200, 3000)
    assert upd.driver_delta["a"] == pytest.approx(24 * surprise)


def test_k_scale_for_sprints():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1500.0, "y": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "y", 2)], d, c, P, k_scale=0.5)
    assert upd.driver_delta["a"] == pytest.approx(6.0)


def test_single_finisher_no_change():
    upd = race_update([("a", "x", 1)], {"a": 1500.0}, {"x": 1500.0}, P)
    assert upd.driver_delta == {} and upd.constructor_delta == {}


def test_apply_and_regress():
    r = {"a": 1600.0, "b": 1400.0}
    apply_update(r, {"a": 10.0})
    assert r == {"a": 1610.0, "b": 1400.0}
    regress(r, 0.5, 1500.0)
    assert r == pytest.approx({"a": 1555.0, "b": 1450.0})
