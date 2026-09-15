import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import replay
from f1pred.ratings.profile import (
    PROFILE_COLUMNS,
    Profile,
    latest_profiles,
    profile_features,
    profile_lookup,
)

P = DEFAULT_PARAMS


def _table(rows):
    cols = [
        "season",
        "round",
        "race_id",
        "date",
        "circuit_id",
        "race_name",
        "driver_id",
        "driver_code",
        "driver_name",
        "constructor_id",
        "constructor_name",
        "grid",
        "position",
        "status",
        "dnf",
        "dnf_kind",
        "points",
        "is_sprint",
        "is_wet",
        "track_type",
        "lap1_position",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["position"] = df["position"].astype("Int64")
    df["lap1_position"] = df["lap1_position"].astype("Int64")
    return df


def _row(i, driver, team, grid, pos, lap1, kind=None):
    return [
        2020,
        i + 1,
        i,
        f"2020-03-{1 + i:02d}",
        "c",
        "r",
        driver,
        driver[:3],
        driver,
        team,
        team,
        grid,
        pos,
        "Finished" if pos is not None else "Accident",
        pos is None,
        kind,
        0.0,
        False,
        False,
        "mixed",
        lap1,
    ]


@pytest.fixture
def two_driver_history():
    rows = []
    for i in range(12):
        # "hero" starts 4th, is 2nd after lap 1 and finishes 2nd; "dull" starts 1st, stays 1st.
        rows.append(_row(i, "dull", "x", 1, 1, 1))
        rows.append(_row(i, "hero", "y", 4, 2, 2))
        rows.append(_row(i, "meh1", "z", 2, 3, 3))
        rows.append(_row(i, "meh2", "w", 3, 4, 4))
    return _table(rows)


def test_features_are_pre_race_and_shrunk(two_driver_history):
    history, _ = replay(two_driver_history, P)
    f = profile_features(two_driver_history, history, P)
    assert list(f.columns) == PROFILE_COLUMNS
    first = f[(f.race_id == 0)]
    assert (first.aggression == 0).all() and (first.risk == 0).all() and (first.form == 0).all()
    hero_last = f[(f.race_id == 11) & (f.driver_id == "hero")].iloc[0]
    # hero gains 2 places (lap1: 4-2, race: 4-2) every race; field mean gain is 0 (a zero-sum grid)
    assert hero_last.profile_n == 11
    assert hero_last.aggression == pytest.approx(2.0 * 11 / 21)
    dull_last = f[(f.race_id == 11) & (f.driver_id == "dull")].iloc[0]
    assert dull_last.aggression == pytest.approx(0.0 * 11 / 21)
    assert (f.risk == 0).all()


def test_risk_relative_to_field():
    rows = []
    for i in range(20):
        rows.append(_row(i, "crash", "x", 1, None, 1, "accident"))
        rows.append(_row(i, "safe1", "y", 2, 1, 2))
        rows.append(_row(i, "safe2", "z", 3, 2, 3))
        rows.append(_row(i, "safe3", "w", 4, 3, 4))
    t = _table(rows)
    history, _ = replay(t, P)
    f = profile_features(t, history, P)
    last = f[f.race_id == 19].set_index("driver_id")
    # crash: rate 1.0 vs field 0.25 -> (4 - 1) shrunk by 19/29
    assert last.loc["crash", "risk"] == pytest.approx(3.0 * 19 / 29)
    assert last.loc["safe1", "risk"] == pytest.approx(-1.0 * 19 / 29)


def test_form_positive_when_beating_expectation():
    rows = []
    # "under" wins the first six races and is rated top, then finishes 3rd while "over" wins;
    # the ratings lag the swing, so over beats its expected position and under misses it.
    # (The plan's fixture had the same order every race, so the ratings matched the finishes
    # from race 1 and every form_raw in the window was 0.)
    for i in range(12):
        if i < 6:
            rows.append(_row(i, "under", "z", 1, 1, 1))
            rows.append(_row(i, "mid", "y", 2, 2, 2))
            rows.append(_row(i, "over", "x", 3, 3, 3))
        else:
            rows.append(_row(i, "over", "x", 3, 1, 3))
            rows.append(_row(i, "mid", "y", 2, 2, 2))
            rows.append(_row(i, "under", "z", 1, 3, 1))
    t = _table(rows)
    history, _ = replay(t, P)
    f = profile_features(t, history, P)
    last = f[f.race_id == 11].set_index("driver_id")
    assert last.loc["over", "form"] > 0 > last.loc["under", "form"]


def test_latest_profiles_and_lookup(driver_race):
    history, _ = replay(driver_race, P)
    feats = profile_features(driver_race, history, P)
    lookup = profile_lookup(feats)
    key = next(iter(lookup))
    assert isinstance(lookup[key], Profile)
    latest = latest_profiles(driver_race, history, P)
    assert set(latest) >= {"max_verstappen", "norris"}
    assert latest["max_verstappen"].n > 0
    assert Profile.zero() == Profile(0.0, 0.0, 0.0, 0)
    # every non-sprint driver-race has a feature row
    assert len(feats) == (~driver_race.is_sprint).sum()
    assert not feats[["aggression", "risk", "form"]].isna().any().any()
