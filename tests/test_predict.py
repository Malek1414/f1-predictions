import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.predict import UnknownRaceError, build_prediction_inputs, resolve_race
from f1pred.ratings.history import replay

P = DEFAULT_PARAMS


@pytest.fixture(scope="module")
def replayed(driver_race):
    return replay(driver_race, P)


def test_resolve_by_round(raw_sample, driver_race):
    ref = resolve_race(raw_sample, driver_race, 2024, round=8)
    assert ref.circuit_id == "monaco" and ref.has_results and ref.name == "Monaco Grand Prix"


def test_resolve_by_name_substring_and_circuit_ref(raw_sample, driver_race):
    assert resolve_race(raw_sample, driver_race, 2024, race="MONZA").circuit_id == "monza"
    assert resolve_race(raw_sample, driver_race, 2024, race="british").round == 12


def test_resolve_unknown_lists_choices(raw_sample, driver_race):
    with pytest.raises(UnknownRaceError) as exc:
        resolve_race(raw_sample, driver_race, 2024, race="mars")
    assert any("Monaco" in c for c in exc.value.choices)
    with pytest.raises(UnknownRaceError):
        resolve_race(raw_sample, driver_race, 2024, round=99)


def test_past_race_inputs(raw_sample, driver_race, replayed):
    history, state = replayed
    ref = resolve_race(raw_sample, driver_race, 2024, round=8)
    inputs = build_prediction_inputs(raw_sample, driver_race, history, state, ref, P)
    assert inputs.use_grid and inputs.note is None
    assert len(inputs.entrants) == 20
    leclerc = next(e for e in inputs.entrants if e.driver_id == "leclerc")
    assert leclerc.grid == 1
    assert inputs.names["leclerc"] == "Charles Leclerc"
    h = history[(history.race_id == ref.race_id) & (~history.is_sprint)].set_index("driver_id")
    assert leclerc.strength == pytest.approx(
        h.loc["leclerc", "driver_rating_pre"] + h.loc["leclerc", "constructor_rating_pre"]
    )


def test_future_race_with_qualifying(raw_sample, driver_race, replayed):
    history, state = replayed
    # Pretend the last 2024 race has not happened: drop its results from the table.
    last_id = int(driver_race[driver_race.season == 2024].race_id.max())
    table = driver_race[driver_race.race_id != last_id]
    hist = history[history.race_id != last_id]
    ref = resolve_race(raw_sample, table, 2024, round=24)
    assert not ref.has_results
    inputs = build_prediction_inputs(raw_sample, table, hist, state, ref, P)
    assert inputs.use_grid and inputs.note is None
    grids = sorted(e.grid for e in inputs.entrants)
    assert grids[:3] == [1, 2, 3]
    e = inputs.entrants[0]
    assert e.strength == pytest.approx(
        state.driver.get(e.driver_id, 1500.0) + state.constructor.get(e.constructor_id, 1500.0)
    )


def test_future_race_without_qualifying(raw_sample, driver_race, replayed):
    history, state = replayed
    raw = dict(raw_sample)
    last_id = int(driver_race[driver_race.season == 2024].race_id.max())
    raw["qualifying"] = raw_sample["qualifying"][raw_sample["qualifying"].raceId != last_id]
    table = driver_race[driver_race.race_id != last_id]
    ref = resolve_race(raw, table, 2024, round=24)
    inputs = build_prediction_inputs(raw, table, history, state, ref, P)
    assert not inputs.use_grid
    assert "qualifying" in inputs.note.lower()
    assert all(e.grid is None for e in inputs.entrants)
    assert len(inputs.entrants) == 20


def test_next_season_applies_regression(raw_sample, driver_race, replayed):
    history, state = replayed
    races = raw_sample["races"].copy()
    new = races.iloc[[-1]].copy()
    new["raceId"] = 999999
    new["year"] = 2025
    new["round"] = 1
    new["date"] = "2025-03-16"
    raw = dict(raw_sample)
    raw["races"] = pd.concat([races, new], ignore_index=True)
    raw["qualifying"] = raw_sample["qualifying"][raw_sample["qualifying"].raceId < 0]
    ref = resolve_race(raw, driver_race, 2025, round=1)
    inputs = build_prediction_inputs(raw, driver_race, history, state, ref, P)
    # Ergast driverRef for Max is "max_verstappen"; plain "verstappen" is Jos.
    e = next(x for x in inputs.entrants if x.driver_id == "max_verstappen")
    d = 1500 + (state.driver["max_verstappen"] - 1500) * 0.9
    c = 1500 + (state.constructor[e.constructor_id] - 1500) * 0.6
    assert e.strength == pytest.approx(d + c)
