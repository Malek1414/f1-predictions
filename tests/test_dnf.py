import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.sim.dnf import dnf_cache_for_races, dnf_probability

P = DEFAULT_PARAMS


def _table(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "race_id",
            "driver_id",
            "constructor_id",
            "circuit_id",
            "dnf",
            "is_sprint",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["is_sprint"] = df["is_sprint"].astype(bool)
    df["dnf"] = df["dnf"].astype(bool)
    return df


def _season(n_races, dnf_for, drivers=("a", "b", "c", "d"), circuit="c1"):
    rows = []
    for i in range(n_races):
        for d in drivers:
            rows.append(
                (
                    f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}",
                    i,
                    d,
                    f"t{d}",
                    circuit,
                    dnf_for(i, d),
                    False,
                )
            )
    return _table(rows)


def test_no_history_uses_default_global():
    t = _table([])
    p = dnf_probability(t, "a", "x", "c1", pd.Timestamp("2020-01-01"), P)
    assert p == pytest.approx(0.1)


def test_unknown_driver_gets_global_rate():
    t = _season(10, lambda i, d: i % 5 == 0)  # 20% global
    p = dnf_probability(t, "zzz", "tzzz", "c1", pd.Timestamp("2021-01-01"), P)
    assert p == pytest.approx(0.2, abs=1e-9)


def test_crash_prone_driver_shrinks_toward_global():
    t = _season(20, lambda i, d: d == "a")  # a always DNFs; global 25%
    p = dnf_probability(t, "a", "ta", "c1", pd.Timestamp("2021-01-01"), P)
    # raw = (1.0 + 1.0)/2 = 1.0, n = 20 -> shrunk = 0.25 + 0.75 * 20/30 = 0.75; circuit factor 1
    assert p == pytest.approx(0.75, abs=1e-9)


def test_only_past_rows_count():
    t = _season(20, lambda i, d: d == "a")
    early = dnf_probability(t, "a", "ta", "c1", pd.Timestamp("2020-01-01"), P)
    assert early == pytest.approx(0.1)


def test_circuit_factor_clipped():
    rows = []
    for i in range(10):
        for d in "abcd":
            rows.append((f"2020-01-{1 + i:02d}", i, d, f"t{d}", "safe", False, False))
    for i in range(10):
        for d in "abcd":
            rows.append((f"2020-02-{1 + i:02d}", 100 + i, d, f"t{d}", "deadly", True, False))
    t = _table(rows)
    global_rate = 0.5
    safe = dnf_probability(t, "zzz", "tzzz", "safe", pd.Timestamp("2021-01-01"), P)
    deadly = dnf_probability(t, "zzz", "tzzz", "deadly", pd.Timestamp("2021-01-01"), P)
    assert safe == pytest.approx(global_rate * P.circuit_factor_min)
    assert deadly == pytest.approx(min(0.9, global_rate * P.circuit_factor_max))


def test_sprints_ignored():
    t = _season(10, lambda i, d: True)
    t["is_sprint"] = True
    p = dnf_probability(t, "a", "ta", "c1", pd.Timestamp("2021-01-01"), P)
    assert p == pytest.approx(0.1)


def test_cache_matches_direct_call(driver_race):
    race_ids = driver_race[
        (driver_race.season == 2024) & (~driver_race.is_sprint)
    ].race_id.unique()[:3]
    cache = dnf_cache_for_races(driver_race, race_ids, P)
    row = driver_race[(driver_race.race_id == race_ids[2]) & (~driver_race.is_sprint)].iloc[0]
    direct = dnf_probability(
        driver_race, row.driver_id, row.constructor_id, row.circuit_id, row.date, P
    )
    assert cache[(int(race_ids[2]), row.driver_id)] == pytest.approx(direct)
    assert 0.01 <= direct <= 0.9
