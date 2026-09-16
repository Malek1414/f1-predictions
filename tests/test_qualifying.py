import math

import pandas as pd
import pytest

from f1pred.data.frame import build_driver_race_table
from f1pred.data.qualifying import QUALIFYING_COLUMNS, parse_lap_ms, qualifying_gaps


def _monaco_2024_id(raw_sample) -> int:
    races = raw_sample["races"]
    row = races[(races["year"] == 2024) & (races["name"].str.contains("Monaco"))]
    return int(row["raceId"].iloc[0])


def test_parse_lap_ms():
    assert parse_lap_ms("1:26.572") == 86572.0
    assert parse_lap_ms("59.999") == 59999.0
    assert parse_lap_ms(float("nan")) is None
    assert parse_lap_ms("") is None
    assert parse_lap_ms(None) is None


def test_gaps_on_sample(raw_sample):
    g = qualifying_gaps(raw_sample)
    assert list(g.columns) == list(QUALIFYING_COLUMNS)
    monaco = g[g.race_id == _monaco_2024_id(raw_sample)]
    assert monaco.set_index("driver_id").loc["leclerc", "gap_pct"] == 0.0
    assert (monaco.gap_pct.dropna() >= 0).all() and monaco.gap_pct.dropna().max() < 5
    assert g.gap_pct.notna().mean() > 0.95


def test_best_ms_is_the_fastest_of_the_three_segments(raw_sample):
    g = qualifying_gaps(raw_sample).set_index(["race_id", "driver_id"])
    q = raw_sample["qualifying"]
    drivers = raw_sample["drivers"][["driverId", "driverRef"]]
    row = q.merge(drivers, on="driverId").iloc[0]
    times = [parse_lap_ms(row[c]) for c in ("q1", "q2", "q3")]
    times = [t for t in times if t is not None]
    assert g.loc[(int(row.raceId), row.driverRef), "best_ms"] == pytest.approx(min(times))


def test_driver_race_table_carries_quali_gap(raw_sample, weather_sample, lap1_sample):
    gaps = qualifying_gaps(raw_sample)
    table = build_driver_race_table(
        raw_sample, start_season=2010, weather=weather_sample, lap1=lap1_sample, qualifying=gaps
    )
    assert "quali_gap_pct" in table.columns
    races = table[~table.is_sprint]
    assert races.quali_gap_pct.notna().mean() > 0.9
    # Exactly one pole sitter per race.
    assert len(races[races.quali_gap_pct == 0.0]) == races.race_id.nunique()


def test_driver_race_table_without_qualifying_is_all_nan(raw_sample):
    table = build_driver_race_table(raw_sample, start_season=2010)
    assert "quali_gap_pct" in table.columns
    assert table.quali_gap_pct.isna().all()


def test_gap_pct_arithmetic():
    raw = {
        "qualifying": pd.DataFrame(
            {
                "raceId": [1, 1],
                "driverId": [10, 11],
                "position": [1, 2],
                "q1": ["1:40.000", "1:41.000"],
                "q2": [float("nan"), float("nan")],
                "q3": [float("nan"), float("nan")],
            }
        ),
        "drivers": pd.DataFrame({"driverId": [10, 11], "driverRef": ["a", "b"]}),
    }
    g = qualifying_gaps(raw).set_index("driver_id")
    assert g.loc["a", "gap_pct"] == 0.0
    assert g.loc["b", "gap_pct"] == pytest.approx(100.0 * 1000.0 / 100000.0)
    assert math.isclose(g.loc["b", "best_ms"], 101000.0)
