import numpy as np
import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import replay
from f1pred.sim.race import Entrant
from f1pred.sim.season import (
    RemainingRace,
    current_standings,
    remaining_calendar,
    season_entrants,
    simulate_season,
)

P = DEFAULT_PARAMS


def _race(i, has_sprint=False):
    return RemainingRace(1000 + i, 20 + i, f"Race {i}", "c", pd.Timestamp("2030-01-01"), has_sprint)


def _entrants():
    return [Entrant(f"d{i}", f"t{i // 2}", 3000.0, None, 0.0) for i in range(4)]


def test_unassailable_lead_gives_probability_one():
    remaining = [_race(0), _race(1)]
    pts = {"d0": 500.0, "d1": 100.0, "d2": 0.0, "d3": 0.0}
    cons = {"t0": 600.0, "t1": 0.0}
    mapping = {d: f"t{int(d[1]) // 2}" for d in pts}
    f = simulate_season(
        2025, remaining, [_entrants(), _entrants()], pts, cons, mapping, P, n_runs=200, seed=0
    )
    assert f.p_driver_title[f.driver_ids.index("d0")] == 1.0
    assert f.p_constructor_title[f.constructor_ids.index("t0")] == 1.0
    assert f.n_remaining == 2
    assert f.expected_driver_points[f.driver_ids.index("d0")] > 500.0


def test_no_remaining_races_returns_current_leader():
    pts = {"a": 10.0, "b": 20.0}
    f = simulate_season(2024, [], [], pts, {"x": 30.0}, {"a": "x", "b": "x"}, P, n_runs=10, seed=0)
    assert f.driver_ids[0] == "b" and f.p_driver_title[0] == 1.0
    assert f.p_constructor_title[0] == 1.0
    np.testing.assert_array_equal(f.expected_driver_points, [20.0, 10.0])


def test_points_awarded_per_race_and_sprint():
    # One driver far stronger than the rest wins every race and sprint:
    # 25 + 8 on the sprint weekend, 25 on the other.
    ents = [Entrant("s", "t0", 6000.0, None, 0.0)] + [
        Entrant(f"d{i}", f"t{1 + i // 2}", 3000.0, None, 0.0) for i in range(3)
    ]
    remaining = [_race(0, has_sprint=True), _race(1)]
    pts = {e.driver_id: 0.0 for e in ents}
    mapping = {e.driver_id: e.constructor_id for e in ents}
    f = simulate_season(2025, remaining, [ents, ents], pts, {}, mapping, P, n_runs=50, seed=1)
    assert f.expected_driver_points[f.driver_ids.index("s")] == pytest.approx(25 + 8 + 25)
    assert f.expected_constructor_points[f.constructor_ids.index("t0")] == pytest.approx(58)


def test_position_matrix_and_frames():
    remaining = [_race(0)]
    pts = {"d0": 0.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}
    mapping = {d: f"t{int(d[1]) // 2}" for d in pts}
    f = simulate_season(2025, remaining, [_entrants()], pts, {}, mapping, P, n_runs=400, seed=2)
    np.testing.assert_allclose(f.driver_position_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(f.driver_position_matrix.sum(axis=0), 1.0)
    df = f.drivers_frame()
    assert list(df.columns) == [
        "driver_id",
        "current_points",
        "expected_points",
        "p_title",
        "p_top3",
    ]
    assert df.p_title.is_monotonic_decreasing
    assert f.p_driver_title.sum() == pytest.approx(1.0)
    cdf = f.constructors_frame()
    assert list(cdf.columns) == ["constructor_id", "current_points", "expected_points", "p_title"]


def test_retired_driver_keeps_points_but_scores_no_more():
    remaining = [_race(0)]
    pts = {"old": 30.0, "d0": 0.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}
    mapping = {"old": "t9", **{d: f"t{int(d[1]) // 2}" for d in pts if d != "old"}}
    f = simulate_season(
        2025, remaining, [_entrants()], pts, {"t9": 30.0}, mapping, P, n_runs=100, seed=3
    )
    assert f.expected_driver_points[f.driver_ids.index("old")] == 30.0
    assert "t9" in f.constructor_ids
    assert f.expected_constructor_points[f.constructor_ids.index("t9")] == 30.0


def test_remaining_calendar_and_standings_on_sample(raw_sample, driver_race):
    last_id = int(driver_race[driver_race.season == 2024].race_id.max())
    table = driver_race[driver_race.race_id != last_id]
    remaining = remaining_calendar(raw_sample, table, 2024)
    assert [r.round for r in remaining] == [24]
    assert remaining[0].name == "Abu Dhabi Grand Prix" and not remaining[0].has_sprint
    assert remaining_calendar(raw_sample, driver_race, 2024) == []
    full = remaining_calendar(raw_sample, driver_race[driver_race.season == 2023], 2024)
    assert len(full) == 24 and sum(r.has_sprint for r in full) == 6

    drivers, cons, mapping = current_standings(driver_race, 2024)
    assert drivers["max_verstappen"] == pytest.approx(437.0)
    assert cons["mclaren"] == pytest.approx(666.0)
    assert mapping["max_verstappen"] == "red_bull"
    assert current_standings(driver_race, 1999) == ({}, {}, {})


def test_season_entrants_on_sample(driver_race):
    _, state = replay(driver_race, P)
    race = RemainingRace(999, 1, "Next", "bahrain", pd.Timestamp("2025-03-16"), False)
    # Phase 3 arithmetic holds with track ratings off (Phase 4 adds a track-type shrink).
    ents = season_entrants(driver_race, state, 2025, race, P.replace(use_track=False))
    assert len(ents) == 20 and all(e.grid is None for e in ents)
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    assert e.strength == pytest.approx(
        1500
        + (state.driver["max_verstappen"] - 1500) * 0.9
        + 1500
        + (state.constructor["red_bull"] - 1500) * 0.6
    )
    assert 0.01 <= e.p_dnf <= 0.9


def test_rain_by_race_changes_outcome():
    ents = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, None, 0.0) for i in range(4)]
    ents[3] = Entrant("d3", "t1", 3000.0, None, 0.0, strength_wet=3800.0)
    remaining = [_race(0)]
    pts = {e.driver_id: 0.0 for e in ents}
    mapping = {e.driver_id: e.constructor_id for e in ents}
    dry = simulate_season(
        2025, remaining, [ents], pts, {}, mapping, P, n_runs=400, seed=0, rain_by_race=[0.0]
    )
    wet = simulate_season(
        2025, remaining, [ents], pts, {}, mapping, P, n_runs=400, seed=0, rain_by_race=[1.0]
    )
    assert (
        wet.p_driver_title[wet.driver_ids.index("d3")]
        > dry.p_driver_title[dry.driver_ids.index("d3")]
    )
    with pytest.raises(ValueError):
        simulate_season(
            2025, remaining, [ents], pts, {}, mapping, P, n_runs=10, rain_by_race=[0.0, 0.0]
        )


def test_season_entrants_carry_wet_strength(driver_race):
    _, state = replay(driver_race, P)
    race = RemainingRace(
        999, 1, "Next", "interlagos", pd.Timestamp("2025-03-16"), False, track_type="mixed"
    )
    ents = season_entrants(driver_race, state, 2025, race, P)
    assert all(e.strength_wet is not None for e in ents)
    assert any(e.strength_wet != e.strength for e in ents)


def test_season_entrants_take_profiles(driver_race):
    from f1pred.ratings.profile import Profile

    _, state = replay(driver_race, P)
    race = RemainingRace(999, 1, "Next", "bahrain", pd.Timestamp("2025-03-16"), False)
    ents = season_entrants(
        driver_race, state, 2025, race, P, {"max_verstappen": Profile(1.0, 0.5, 0.2, 9)}
    )
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    assert (e.aggression, e.risk, e.form) == (1.0, 0.5, 0.2)
