import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.data import weather
from f1pred.data.weather import (
    build_weather_table,
    circuit_wet_rate,
    fetch_race_rain_mm,
    fetch_rain_probability,
    race_window,
    window_sum,
)

P = DEFAULT_PARAMS


def _hourly(day, values, key="precipitation"):
    return {"hourly": {"time": [f"{day}T{h:02d}:00" for h in range(24)], key: values}}


def test_race_window_uses_time_or_default():
    s, e = race_window("2024-06-09", "18:00:00", 3)
    assert s == pd.Timestamp("2024-06-09 18:00", tz="UTC") and e - s == pd.Timedelta(hours=3)
    s, _ = race_window(pd.Timestamp("2024-06-09"), None, 3)
    assert s.hour == 13


def test_window_sum():
    vals = [0.0] * 24
    vals[18], vals[19], vals[20], vals[21] = 1.0, 2.0, 0.5, 9.0
    s, e = race_window("2024-06-09", "18:00:00", 3)
    assert window_sum(_hourly("2024-06-09", vals), s, e, "precipitation") == pytest.approx(3.5)
    assert window_sum({"error": True}, s, e, "precipitation") is None


def test_fetch_race_rain_mm_and_probability(monkeypatch):
    calls = []

    def fake(url, params):
        calls.append((url, params))
        if "archive" in url:
            v = [0.0] * 24
            v[13] = 0.6
            return _hourly(params["start_date"], v)
        v = [0] * 24
        v[13] = 40  # inside the 11:00-14:00 window; the window end is exclusive (see window_sum)
        return _hourly(params["start_date"], v, key="precipitation_probability")

    monkeypatch.setattr(weather, "get_json", fake)
    assert fetch_race_rain_mm(1.0, 2.0, "2024-06-09", "13:00:00", 3) == pytest.approx(0.6)
    assert fetch_rain_probability(1.0, 2.0, "2026-09-26", "11:00:00", 3) == pytest.approx(0.4)
    assert calls[0][1]["hourly"] == "precipitation" and calls[0][1]["timezone"] == "UTC"


def test_fetch_returns_none_on_failure(monkeypatch):
    def boom(url, params):
        raise OSError("offline")

    monkeypatch.setattr(weather, "get_json", boom)
    assert fetch_race_rain_mm(1.0, 2.0, "2024-06-09", None, 3) is None
    monkeypatch.setattr(weather, "get_json", lambda url, params: {"error": True, "reason": "x"})
    assert fetch_rain_probability(1.0, 2.0, "2030-01-01", None, 3) is None


def test_build_weather_table_incremental(raw_sample, tmp_path, monkeypatch):
    fetched = []

    def fake_rain(lat, lng, date, time, hours):
        fetched.append(str(date)[:10])
        return 2.0 if str(date).startswith("2024-11-03") else 0.0  # Brazil 2024 wet

    monkeypatch.setattr(weather, "fetch_race_rain_mm", fake_rain)
    today = pd.Timestamp("2025-01-01")
    w = build_weather_table(raw_sample, tmp_path, P, today=today)
    assert list(w.columns) == ["race_id", "rain_mm", "is_wet"]
    assert len(w) == 46 and w.is_wet.sum() == 1
    assert (tmp_path / "weather.parquet").exists()
    n_first = len(fetched)
    w2 = build_weather_table(raw_sample, tmp_path, P, today=today)
    assert len(fetched) == n_first and len(w2) == 46


def test_build_weather_table_retries_failed_rows(raw_sample, tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "fetch_race_rain_mm", lambda *a: None)
    w = build_weather_table(raw_sample, tmp_path, P, today=pd.Timestamp("2025-01-01"))
    assert w.rain_mm.isna().all() and not w.is_wet.any()
    monkeypatch.setattr(weather, "fetch_race_rain_mm", lambda *a: 3.0)
    w = build_weather_table(raw_sample, tmp_path, P, today=pd.Timestamp("2025-01-01"))
    assert w.is_wet.all()


def test_build_weather_table_skips_future_races(raw_sample, tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "fetch_race_rain_mm", lambda *a: 0.0)
    w = build_weather_table(raw_sample, tmp_path, P, today=pd.Timestamp("2024-01-01"))
    assert len(w) == 22


def test_circuit_wet_rate(driver_race):
    # driver_race carries is_wet from the fixture after Task 4; before that, this test is skipped.
    if "is_wet" not in driver_race.columns:
        pytest.skip("is_wet added in Task 4")
    rate = circuit_wet_rate(driver_race, "interlagos", pd.Timestamp("2025-01-01"))
    assert rate == pytest.approx(
        driver_race[~driver_race.is_sprint].groupby("race_id").is_wet.first().mean()
    )
    assert circuit_wet_rate(driver_race, "nowhere", pd.Timestamp("2000-01-01")) == 0.1
