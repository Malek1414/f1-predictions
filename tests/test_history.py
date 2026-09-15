import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import RatingState, replay, state_for_season, strength

P = DEFAULT_PARAMS


def _row(season, rnd, race_id, date, driver, team, grid, pos, is_sprint=False):
    return {
        "season": season,
        "round": rnd,
        "race_id": race_id,
        "date": pd.Timestamp(date),
        "circuit_id": "c",
        "race_name": "r",
        "driver_id": driver,
        "driver_code": driver[:3].upper(),
        "driver_name": driver,
        "constructor_id": team,
        "constructor_name": team,
        "grid": grid,
        "position": pos,
        "status": "Finished" if pos is not None else "Engine",
        "dnf": pos is None,
        "dnf_kind": None if pos is not None else "mechanical",
        "points": 0.0,
        "is_sprint": is_sprint,
    }


@pytest.fixture
def tiny_table():
    rows = [
        # 2020 round 1: a beats b (different teams)
        _row(2020, 1, 1, "2020-03-01", "a", "x", 1, 1),
        _row(2020, 1, 1, "2020-03-01", "b", "y", 2, 2),
        # 2020 round 2: b beats a; a DNFs
        _row(2020, 2, 2, "2020-03-08", "b", "y", 1, 1),
        _row(2020, 2, 2, "2020-03-08", "a", "x", 2, None),
        # 2021 round 1
        _row(2021, 1, 3, "2021-03-01", "a", "x", 1, 1),
        _row(2021, 1, 3, "2021-03-01", "b", "y", 2, 2),
    ]
    df = pd.DataFrame(rows)
    df["position"] = df["position"].astype("Int64")
    return df


def test_pre_race_ratings_exclude_own_result(tiny_table):
    history, state = replay(tiny_table, P)
    r1 = history[history.race_id == 1].set_index("driver_id")
    assert r1.loc["a", "driver_rating_pre"] == 1500.0
    assert r1.loc["a", "driver_races_pre"] == 0
    r2 = history[history.race_id == 2].set_index("driver_id")
    assert r2.loc["a", "driver_rating_pre"] == pytest.approx(1512.0)
    assert r2.loc["b", "driver_rating_pre"] == pytest.approx(1488.0)
    assert r2.loc["a", "constructor_rating_pre"] == pytest.approx(1516.0)
    assert r2.loc["a", "driver_races_pre"] == 1


def test_dnf_does_not_change_rating(tiny_table):
    history, _ = replay(tiny_table, P)
    r3 = history[history.race_id == 3].set_index("driver_id")
    # race 2 had one finisher (b) so no update; then season regression (driver 0.1)
    assert r3.loc["a", "driver_rating_pre"] == pytest.approx(1500 + 12 * 0.9)
    assert r3.loc["b", "driver_rating_pre"] == pytest.approx(1500 - 12 * 0.9)
    assert r3.loc["a", "driver_races_pre"] == 1
    assert r3.loc["b", "driver_races_pre"] == 2


def test_constructor_regression_uses_regulation_fraction(tiny_table):
    params = P.replace(regulation_seasons=(2021,))
    history, _ = replay(tiny_table, params)
    r3 = history[history.race_id == 3].set_index("driver_id")
    assert r3.loc["a", "constructor_rating_pre"] == pytest.approx(1500 + 16 * 0.3)


def test_final_state_and_state_for_new_season(tiny_table):
    _, state = replay(tiny_table, P)
    assert state.season == 2021
    # race 3 step is k_driver * surprise = 24 * (1 - E); the "12" elsewhere is 24 * 0.5.
    a_after = 1500 + 12 * 0.9 + 24 * (1 - _e(state, "a", "b"))
    assert state.driver["a"] == pytest.approx(a_after, abs=1e-6)
    nxt = state_for_season(state, 2022, P)
    assert nxt.driver["a"] == pytest.approx(1500 + (state.driver["a"] - 1500) * 0.9)
    assert state_for_season(state, 2021, P).driver["a"] == state.driver["a"]


def _e(state, a, b):
    from f1pred.ratings.elo import expected_score

    # ratings before race 3: a 1510.8 + x 1509.6 vs b 1489.2 + y 1490.4
    return expected_score(1510.8 + 1509.6, 1489.2 + 1490.4)


def test_history_columns_and_order(tiny_table):
    history, _ = replay(tiny_table, P)
    assert list(history.columns) == [
        "race_id",
        "is_sprint",
        "season",
        "round",
        "date",
        "driver_id",
        "constructor_id",
        "driver_rating_pre",
        "constructor_rating_pre",
        "driver_races_pre",
    ]
    assert history.date.is_monotonic_increasing


def test_sprint_uses_half_k(tiny_table):
    sprint = pd.DataFrame(
        [
            _row(2020, 1, 1, "2020-02-29", "a", "x", 1, 1, is_sprint=True),
            _row(2020, 1, 1, "2020-02-29", "b", "y", 2, 2, is_sprint=True),
        ]
    )
    sprint["position"] = sprint["position"].astype("Int64")
    table = pd.concat([sprint, tiny_table], ignore_index=True)
    history, _ = replay(table, P)
    race1 = history[(history.race_id == 1) & (~history.is_sprint)].set_index("driver_id")
    assert race1.loc["a", "driver_rating_pre"] == pytest.approx(1506.0)


def test_strength_and_json_roundtrip(tmp_path):
    state = RatingState({"a": 1600.0}, {"x": 1450.0}, {"a": 3}, 2024)
    assert strength(state, "a", "x", P) == 3050.0
    assert strength(state, "new", "x", P) == 1500.0 + 1450.0
    path = tmp_path / "state.json"
    state.to_json(path)
    loaded = RatingState.from_json(path)
    assert loaded == state


def test_replay_on_real_sample(driver_race):
    history, state = replay(driver_race, P)
    assert len(history) == len(driver_race)
    top = max(state.driver, key=state.driver.get)
    # Ergast driverRef for Max is "max_verstappen"; plain "verstappen" is Jos.
    assert top in {"max_verstappen", "norris", "leclerc", "piastri", "hamilton", "russell", "sainz"}
    assert max(state.constructor, key=state.constructor.get) in {"red_bull", "mclaren", "ferrari"}
