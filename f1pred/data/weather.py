"""Race-day rain from Open-Meteo. Spec 3.2 and 6.7."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from f1pred.config import ModelParams

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_START = "13:00:00"
WEATHER_FILE = "weather.parquet"
WEATHER_COLUMNS = ["race_id", "rain_mm", "is_wet"]
FALLBACK_WET_RATE = 0.1
MIN_CIRCUIT_RACES = 3


def get_json(url: str, params: dict) -> dict:
    return requests.get(url, params=params, timeout=30).json()


def race_window(
    date: pd.Timestamp | str, time: str | None, hours: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    clock = time if isinstance(time, str) and time else DEFAULT_START
    start = pd.Timestamp(f"{day} {clock}", tz="UTC")
    return start, start + pd.Timedelta(hours=hours)


def window_sum(hourly: dict, start: pd.Timestamp, end: pd.Timestamp, key: str) -> float | None:
    block = hourly.get("hourly") if isinstance(hourly, dict) else None
    if not block or key not in block or "time" not in block:
        return None
    times = pd.to_datetime(block["time"], utc=True)
    values = pd.Series(block[key], index=times, dtype=float).fillna(0.0)
    return float(values[(times >= start) & (times < end)].sum())


def _window_max(hourly: dict, start: pd.Timestamp, end: pd.Timestamp, key: str) -> float | None:
    block = hourly.get("hourly") if isinstance(hourly, dict) else None
    if not block or key not in block or "time" not in block:
        return None
    times = pd.to_datetime(block["time"], utc=True)
    values = pd.Series(block[key], index=times, dtype=float).fillna(0.0)
    inside = values[(times >= start) & (times < end)]
    return float(inside.max()) if len(inside) else None


def _params(lat: float, lng: float, start: pd.Timestamp, end: pd.Timestamp, hourly: str) -> dict:
    return {
        "latitude": lat,
        "longitude": lng,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": hourly,
        "timezone": "UTC",
    }


def fetch_race_rain_mm(lat: float, lng: float, date, time, hours: int) -> float | None:
    start, end = race_window(date, time, hours)
    try:
        data = get_json(ARCHIVE_URL, _params(lat, lng, start, end, "precipitation"))
    except Exception as exc:  # network / JSON
        log.warning("weather archive failed for %s: %s", date, exc)
        return None
    return window_sum(data, start, end, "precipitation")


def fetch_rain_probability(lat: float, lng: float, date, time, hours: int) -> float | None:
    start, end = race_window(date, time, hours)
    try:
        data = get_json(
            FORECAST_URL, _params(lat, lng, start, end, "precipitation_probability,precipitation")
        )
    except Exception as exc:
        log.warning("weather forecast failed for %s: %s", date, exc)
        return None
    prob = _window_max(data, start, end, "precipitation_probability")
    return None if prob is None else prob / 100.0


def load_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_wet"] = df["is_wet"].astype(bool)
    return df[WEATHER_COLUMNS]


def build_weather_table(
    raw: dict, cache_dir: Path, params: ModelParams, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Race-day rain for every completed race since start_season, cached in weather.parquet."""
    cache_dir = Path(cache_dir)
    # Timestamp.now("UTC") replaces the deprecated Timestamp.utcnow(); same instant.
    today = pd.Timestamp.now("UTC").normalize().tz_localize(None) if today is None else today
    races = raw["races"].merge(raw["circuits"][["circuitId", "lat", "lng"]], on="circuitId")
    races = races[(races["year"] >= params.start_season) & (pd.to_datetime(races["date"]) < today)]
    path = cache_dir / WEATHER_FILE
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=WEATHER_COLUMNS)
    known = cached[cached["rain_mm"].notna()]
    todo = races[~races["raceId"].isin(known["race_id"])]
    rows = []
    for r in todo.itertuples():
        mm = fetch_race_rain_mm(
            float(r.lat), float(r.lng), r.date, r.time, params.race_window_hours
        )
        rows.append({"race_id": int(r.raceId), "rain_mm": mm})
    fresh = pd.DataFrame(rows, columns=["race_id", "rain_mm"])
    out = pd.concat([known[["race_id", "rain_mm"]], fresh], ignore_index=True)
    out["rain_mm"] = out["rain_mm"].astype(float)
    out["is_wet"] = out["rain_mm"].fillna(-1.0) >= params.wet_threshold_mm
    out = out.sort_values("race_id").reset_index(drop=True)[WEATHER_COLUMNS]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    failed = int(out["rain_mm"].isna().sum())
    if failed:
        log.warning("%d races have no weather data (treated as dry; retried next update)", failed)
    return out


def circuit_wet_rate(table: pd.DataFrame, circuit_id: str, before_date: pd.Timestamp) -> float:
    past = table[(~table["is_sprint"]) & (table["date"] < before_date)]
    if past.empty or "is_wet" not in past.columns:
        return FALLBACK_WET_RATE
    per_race = past.groupby("race_id").agg(circuit=("circuit_id", "first"), wet=("is_wet", "first"))
    here = per_race[per_race["circuit"] == circuit_id]
    if len(here) >= MIN_CIRCUIT_RACES:
        return float(here["wet"].mean())
    return float(per_race["wet"].mean())
