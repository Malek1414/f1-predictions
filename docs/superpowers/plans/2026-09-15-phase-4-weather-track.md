# F1 Predictions Phase 4: Weather and Track Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add wet-weather and track-type awareness: race-day rain from Open-Meteo, a hand-curated circuit type file, per-driver and per-constructor conditional Elo ratings shrunk toward the overall rating, rain-forecast dice rolls inside the Monte Carlo, and an ablation backtest that reports scores with and without each addition.

**Architecture:** The driver-race table gains `is_wet` and `track_type` columns, populated in `f1pred data update` from a cached weather table and the YAML mapping. The ratings replay maintains conditional ratings alongside the overall ones and records them in the history. A single helper turns (overall, conditional, count) into an effective rating. `Entrant` gains `strength_wet`; `simulate_positions` gains `rain_probability` and draws wet/dry per run. Prediction of a future race asks Open-Meteo for the rain probability and falls back to the circuit's historical wet rate. Nothing in Phases 1 to 3 changes behaviour when `use_weather` and `use_track` are both false.

**Tech Stack:** adds `requests` and `pyyaml`. Everything else unchanged.

Spec: `docs/superpowers/specs/2026-09-15-f1-predictions-design.md` sections 3.2, 3.3, 3.5, 5.5, 6.3, 6.7, 8.1, 10. Existing code to read first: `f1pred/data/frame.py`, `f1pred/ratings/history.py`, `f1pred/ratings/elo.py`, `f1pred/sim/race.py`, `f1pred/sim/season.py`, `f1pred/backtest/run.py`, `f1pred/predict.py`, `f1pred/cli.py`, `f1pred/tune.py`, `tests/conftest.py`.

## Global Constraints

- Same as earlier plans: `uv run` everything; ruff line length 100; `uv run ruff check . && uv run ruff format --check .` clean before every commit; tests never touch the network (Open-Meteo calls are monkeypatched); `tests/fixtures/sample/weather.csv` (already committed: `race_id, rain_mm, is_wet` for the 46 sample races) is the weather fixture.
- Commit trailer on every commit:
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01YJeHtrUFY8CEqTesftaBuB
  ```
- Wet definition: precipitation summed over the three hours from race start (UTC `time` from the races table; 13:00 UTC if missing) is at least `wet_threshold_mm = 0.5`.
- Open-Meteo endpoints (verified 2026-09-15):
  - archive: `https://archive-api.open-meteo.com/v1/archive?latitude&longitude&start_date&end_date&hourly=precipitation&timezone=UTC` → `{"hourly": {"time": ["2023-03-05T00:00", ...], "precipitation": [0.0, ...]}}`
  - forecast: `https://api.open-meteo.com/v1/forecast?latitude&longitude&start_date&end_date&hourly=precipitation_probability,precipitation&timezone=UTC` → `{"hourly": {"time": [...], "precipitation_probability": [2, ...], "precipitation": [...]}}`; probability is 0 to 100; dates beyond about 16 days return an error object without `hourly`.
- All new knobs default to values that keep the Phase 1 to 3 outputs unchanged when `use_weather=False, use_track=False`; the ablation test in Task 7 enforces this.

## File Structure

```
pyproject.toml                  MODIFY: add requests, pyyaml
f1pred/config.py                MODIFY: new params
f1pred/data/weather.py          Open-Meteo fetch, race window, weather table build/cache, forecast, fallback rate
f1pred/data/track_types.yaml    circuit_id -> type
f1pred/data/track_types.py      load_track_types(), track_type_for(circuit_id, mapping)
f1pred/data/frame.py            MODIFY: is_wet + track_type columns
f1pred/ratings/conditional.py   effective_rating(), strengths_for_row()
f1pred/ratings/history.py       MODIFY: conditional ratings in RatingState, replay, history columns
f1pred/sim/race.py              MODIFY: Entrant.strength_wet, rain_probability in simulate_positions / simulate_race
f1pred/sim/season.py            MODIFY: rain_by_race, strength_wet in season_entrants
f1pred/backtest/run.py          MODIFY: conditional strengths, rain_probability from is_wet, ablation helper
f1pred/predict.py               MODIFY: rain probability (forecast / fallback), track type, strength_wet
f1pred/tune.py                  MODIFY: search space additions
f1pred/cli.py                   MODIFY: data update builds weather, backtest --ablate, predict prints rain chance
tests/conftest.py               MODIFY: weather_sample fixture, driver_race built with weather + track types
tests/test_weather.py
tests/test_track_types.py
tests/test_frame.py             MODIFY
tests/test_conditional.py
tests/test_history.py           MODIFY
tests/test_race_sim.py          MODIFY
tests/test_backtest.py          MODIFY
tests/test_predict.py           MODIFY
tests/test_cli.py               MODIFY
docs/weather-and-track.md
README.md                       MODIFY
```

---

### Task 1: Params and dependencies

**Files:**
- Modify: `pyproject.toml`, `f1pred/config.py`, `tests/test_config.py`

**Interfaces:**
- New `ModelParams` fields (append after the DNF block):
  ```python
  # Weather and track (Phase 4)
  use_weather: bool = True
  use_track: bool = True
  wet_threshold_mm: float = 0.5
  race_window_hours: int = 3
  shrink_wet: float = 8.0
  shrink_track: float = 8.0
  wet_noise_factor: float = 1.5
  wet_dnf_factor: float = 1.5
  ```

- [ ] **Step 1: Add `"requests>=2.32"` and `"pyyaml>=6"` to `dependencies` in `pyproject.toml`; run `uv sync`.**

- [ ] **Step 2: Failing test in `tests/test_config.py`**

```python
def test_phase4_defaults():
    p = DEFAULT_PARAMS
    assert p.use_weather and p.use_track
    assert p.wet_threshold_mm == 0.5 and p.race_window_hours == 3
    assert p.shrink_wet == 8.0 and p.shrink_track == 8.0
    assert p.wet_noise_factor == 1.5 and p.wet_dnf_factor == 1.5
```

- [ ] **Step 3: Run, see it fail, add the fields, run green, lint, commit**

```bash
git add pyproject.toml uv.lock f1pred/config.py tests/test_config.py
git commit -m "feat: weather and track params"
```

---

### Task 2: Weather module

**Files:**
- Create: `f1pred/data/weather.py`, `tests/test_weather.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- `ARCHIVE_URL`, `FORECAST_URL` constants.
- `get_json(url: str, params: dict) -> dict`: thin wrapper over `requests.get(url, params=params, timeout=30).json()`; tests monkeypatch this.
- `race_window(date: pd.Timestamp | str, time: str | None, hours: int) -> tuple[pd.Timestamp, pd.Timestamp]` (UTC-aware start and end; `time` missing → 13:00).
- `window_sum(hourly: dict, start, end, key: str) -> float | None`: sum of `hourly[key]` over `start <= t < end`, `None` if `hourly` missing the key.
- `fetch_race_rain_mm(lat: float, lng: float, date, time, hours: int) -> float | None` (archive; `None` on error or missing data).
- `fetch_rain_probability(lat: float, lng: float, date, time, hours: int) -> float | None`: max of `precipitation_probability` over the window divided by 100; `None` if the response has no `hourly` or the request raises.
- `build_weather_table(raw: dict, cache_dir: Path, params: ModelParams, today: pd.Timestamp | None = None) -> pd.DataFrame` with columns `race_id, rain_mm, is_wet`: for every race in `raw["races"]` with `year >= params.start_season` and `date < today`, reuse rows already in `cache_dir/weather.parquet`, fetch the rest, write the parquet, return the full table. Rows whose fetch failed are stored with `rain_mm = NaN, is_wet = False` and are retried on the next update. `today` defaults to `pd.Timestamp.utcnow().normalize()`.
- `circuit_wet_rate(table: pd.DataFrame, circuit_id: str, before_date: pd.Timestamp) -> float`: share of past non-sprint races at that circuit with `is_wet`; falls back to the global past wet rate if the circuit has fewer than 3 races; 0.1 if no history at all.
- `load_weather_csv(path: Path) -> pd.DataFrame` for the fixture.

- [ ] **Step 1: conftest additions**

```python
from f1pred.data.weather import load_weather_csv


@pytest.fixture(scope="session")
def weather_sample() -> pd.DataFrame:
    return load_weather_csv(SAMPLE_DIR / "weather.csv")
```
(The `driver_race` fixture is changed in Task 4 to use it.)

- [ ] **Step 2: Failing tests `tests/test_weather.py`**

```python
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
        v[14] = 40
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
    assert rate == pytest.approx(driver_race[~driver_race.is_sprint].groupby("race_id").is_wet.first().mean())
    assert circuit_wet_rate(driver_race, "nowhere", pd.Timestamp("2000-01-01")) == 0.1
```

Note on the last test: circuits with fewer than 3 past races (Interlagos has 2 in the sample) fall back to the global rate, which is what the assertion computes.

- [ ] **Step 3: Run to verify failure, then write `f1pred/data/weather.py`**

```python
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
    today = pd.Timestamp.utcnow().normalize().tz_localize(None) if today is None else today
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
```

- [ ] **Step 4: Run tests and lint, commit**

Run: `uv run pytest tests/test_weather.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: all pass except `test_circuit_wet_rate` which skips until Task 4.

```bash
git add f1pred/data/weather.py tests/test_weather.py tests/conftest.py
git commit -m "feat: Open-Meteo race-day rain with cached weather table"
```

---

### Task 3: Track types

**Files:**
- Create: `f1pred/data/track_types.yaml`, `f1pred/data/track_types.py`, `tests/test_track_types.py`

**Interfaces:**
- `TRACK_TYPES = ("street", "high_speed", "high_downforce", "mixed")`, `DEFAULT_TRACK_TYPE = "mixed"`
- `load_track_types(path: Path | None = None) -> dict[str, str]` (validates every value is in `TRACK_TYPES`, raises `ValueError` otherwise)
- `track_type_for(circuit_id: str, mapping: Mapping[str, str]) -> str` (warns once per unknown circuit, returns `mixed`)

- [ ] **Step 1: Write `f1pred/data/track_types.yaml`**

```yaml
# Hand-curated circuit types for every circuit raced since 2010. Spec 3.3.
# street: temporary or semi-permanent street layouts, walls close, low average speed
# high_speed: long straights, low downforce, power-sensitive
# high_downforce: flowing medium/high-speed corners, aero-sensitive
# mixed: everything else
albert_park: mixed
americas: mixed
bahrain: mixed
baku: street
buddh: mixed
catalunya: high_downforce
hockenheimring: mixed
hungaroring: high_downforce
imola: mixed
interlagos: mixed
istanbul: mixed
jeddah: street
losail: high_downforce
madring: street
marina_bay: street
miami: street
monaco: street
monza: high_speed
mugello: high_speed
nurburgring: mixed
portimao: mixed
red_bull_ring: high_speed
ricard: mixed
rodriguez: high_speed
sepang: high_downforce
shanghai: mixed
silverstone: high_speed
sochi: street
spa: high_speed
suzuka: high_downforce
valencia: street
vegas: street
villeneuve: high_speed
yas_marina: mixed
yeongam: mixed
zandvoort: high_downforce
```

- [ ] **Step 2: Failing tests `tests/test_track_types.py`**

```python
import logging

import pytest

from f1pred.data.track_types import TRACK_TYPES, load_track_types, track_type_for


def test_mapping_covers_sample_circuits(driver_race):
    mapping = load_track_types()
    missing = set(driver_race.circuit_id) - set(mapping)
    assert not missing
    assert set(mapping.values()) <= set(TRACK_TYPES)
    assert mapping["monaco"] == "street" and mapping["monza"] == "high_speed"


def test_unknown_circuit_falls_back_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert track_type_for("moon", {"monaco": "street"}) == "mixed"
    assert "moon" in caplog.text


def test_invalid_value_rejected(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("monaco: oval\n")
    with pytest.raises(ValueError, match="oval"):
        load_track_types(p)
```

- [ ] **Step 3: Write `f1pred/data/track_types.py`**

```python
"""Circuit type lookup from the hand-curated YAML. Spec 3.3."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

TRACK_TYPES = ("street", "high_speed", "high_downforce", "mixed")
DEFAULT_TRACK_TYPE = "mixed"
DEFAULT_PATH = Path(__file__).parent / "track_types.yaml"
_warned: set[str] = set()


def load_track_types(path: Path | None = None) -> dict[str, str]:
    data = yaml.safe_load(Path(path or DEFAULT_PATH).read_text()) or {}
    for circuit, kind in data.items():
        if kind not in TRACK_TYPES:
            raise ValueError(f"track type for {circuit!r} is {kind!r}, expected one of {TRACK_TYPES}")
    return {str(k): str(v) for k, v in data.items()}


def track_type_for(circuit_id: str, mapping: Mapping[str, str]) -> str:
    kind = mapping.get(circuit_id)
    if kind is None:
        if circuit_id not in _warned:
            log.warning("circuit %r has no track type; using %s", circuit_id, DEFAULT_TRACK_TYPE)
            _warned.add(circuit_id)
        return DEFAULT_TRACK_TYPE
    return kind
```

- [ ] **Step 4: Run, lint, commit**

```bash
git add f1pred/data/track_types.yaml f1pred/data/track_types.py tests/test_track_types.py
git commit -m "feat: hand-curated track types"
```

---

### Task 4: `is_wet` and `track_type` in the driver-race table

**Files:**
- Modify: `f1pred/data/frame.py`, `tests/test_frame.py`, `tests/conftest.py`

**Interfaces:**
- `build_driver_race_table(raw, start_season=2010, weather: pd.DataFrame | None = None, track_types: Mapping[str, str] | None = None)`; `DRIVER_RACE_COLUMNS` gains `is_wet` (bool) and `track_type` (str) at the end. With `weather=None`, `is_wet` is all False; with `track_types=None`, `track_type` is `mixed` for every row. Sprint rows take the same `is_wet` as their race (a simplification, documented).

- [ ] **Step 1: conftest change**

```python
from f1pred.data.track_types import load_track_types


@pytest.fixture(scope="session")
def driver_race(raw_sample, weather_sample) -> pd.DataFrame:
    return build_driver_race_table(
        raw_sample, start_season=2010, weather=weather_sample, track_types=load_track_types()
    )
```

- [ ] **Step 2: Failing tests in `tests/test_frame.py`**

Update `test_table_columns_and_types` (the column list assertion now includes the two new columns via `DRIVER_RACE_COLUMNS`, so it still passes once implemented) and add:

```python
def test_wet_and_track_columns(driver_race, raw_sample):
    assert driver_race.is_wet.dtype == bool
    brazil = driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "interlagos")]
    assert brazil.is_wet.all()
    assert driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "bahrain")].is_wet.eq(False).all()
    assert driver_race[driver_race.circuit_id == "monaco"].track_type.eq("street").all()
    plain = build_driver_race_table(raw_sample)
    assert not plain.is_wet.any() and plain.track_type.eq("mixed").all()
```

- [ ] **Step 3: Implement in `frame.py`**

Add the two names to `DRIVER_RACE_COLUMNS`. In `build_driver_race_table`, after the `concat` and before sorting:

```python
    if weather is not None and len(weather):
        wet = weather.set_index("race_id")["is_wet"].astype(bool)
        table["is_wet"] = table["race_id"].map(wet).fillna(False).astype(bool)
    else:
        table["is_wet"] = False
    mapping = track_types or {}
    table["track_type"] = table["circuit_id"].map(lambda c: track_type_for(c, mapping) if mapping else DEFAULT_TRACK_TYPE)
```
with `from f1pred.data.track_types import DEFAULT_TRACK_TYPE, track_type_for`. Keep `out[list(DRIVER_RACE_COLUMNS)]` selection at the end (add the columns to `_session_rows` output as placeholders or select after adding; simplest: build `out` in `_session_rows` without them and only reorder at the end of `build_driver_race_table`).

- [ ] **Step 4: Run the full suite (the skipped weather test now runs), lint, commit**

```bash
git add f1pred/data/frame.py tests/test_frame.py tests/conftest.py
git commit -m "feat: is_wet and track_type columns in driver-race table"
```

---

### Task 5: Conditional ratings

**Files:**
- Create: `f1pred/ratings/conditional.py`, `tests/test_conditional.py`
- Modify: `f1pred/ratings/history.py`, `tests/test_history.py`

**Interfaces:**
- `conditional.effective_rating(overall: float, conditional: float, n: int, shrink: float) -> float` = `overall + (conditional - overall) * n / (n + shrink)`.
- `conditional.strengths(driver_overall, constructor_overall, driver_track, constructor_track, n_driver_track, n_constructor_track, driver_wet, constructor_wet, n_driver_wet, n_constructor_wet, params) -> tuple[float, float]` returning `(strength_dry, strength_wet)`:
  - `dry = eff(d, d_track, n_dt, shrink_track) + eff(c, c_track, n_ct, shrink_track)` if `params.use_track` else `d + c`
  - `wet = dry + (eff(d, d_wet, n_dw, shrink_wet) - d) + (eff(c, c_wet, n_cw, shrink_wet) - c)` if `params.use_weather` else `dry`
- `RatingState` gains: `driver_wet: dict[str, float]`, `constructor_wet: dict[str, float]`, `driver_wet_n: dict[str, int]`, `constructor_wet_n: dict[str, int]`, `driver_track: dict[str, dict[str, float]]` (track_type → driver → rating), `constructor_track: dict[str, dict[str, float]]`, `driver_track_n: dict[str, dict[str, int]]`, `constructor_track_n: dict[str, dict[str, int]]`. All default empty. JSON round trip must include them. `from_json` must tolerate a Phase 1 file without these keys.
- `HISTORY_COLUMNS` gains, in this order after `driver_races_pre`: `is_wet, track_type, driver_wet_pre, constructor_wet_pre, driver_wet_n_pre, constructor_wet_n_pre, driver_track_pre, constructor_track_pre, driver_track_n_pre, constructor_track_n_pre`.
- `replay` rules: for each session, after the overall update, (a) compute a second `race_update` with `driver_ratings=state.driver_track[track_type]` and `constructor_ratings=state.constructor_track[track_type]` (defaulting missing entries to the current overall rating of that driver/constructor at first sight) and apply it to those dicts, bumping the `_n` counts for finishers; (b) if `is_wet`, do the same with the wet dicts. Season regression applies to all conditional dicts with the same fractions as their overall counterpart. A table without `is_wet`/`track_type` columns is treated as all-dry, all-`mixed`.
- `state_for_season` unchanged in signature, regresses conditional dicts too.
- New helper `history.state_strengths(state: RatingState, driver_id: str, constructor_id: str, track_type: str, params: ModelParams) -> tuple[float, float]` that reads the dicts and calls `conditional.strengths`.

- [ ] **Step 1: Failing tests `tests/test_conditional.py`**

```python
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.conditional import effective_rating, strengths

P = DEFAULT_PARAMS


def test_effective_rating_shrinks_with_evidence():
    assert effective_rating(1500, 1600, 0, 8) == 1500
    assert effective_rating(1500, 1600, 8, 8) == pytest.approx(1550)
    assert effective_rating(1500, 1600, 10_000, 8) == pytest.approx(1600, abs=0.1)


def test_strengths_flags():
    args = dict(
        driver_overall=1500, constructor_overall=1500,
        driver_track=1580, constructor_track=1500, n_driver_track=8, n_constructor_track=0,
        driver_wet=1500, constructor_wet=1420, n_driver_wet=0, n_constructor_wet=8,
    )
    dry, wet = strengths(**args, params=P)
    assert dry == pytest.approx(3040) and wet == pytest.approx(3000)
    dry, wet = strengths(**args, params=P.replace(use_track=False))
    assert dry == 3000 and wet == pytest.approx(2960)
    dry, wet = strengths(**args, params=P.replace(use_weather=False))
    assert dry == pytest.approx(3040) and wet == dry
```

- [ ] **Step 2: Failing tests added to `tests/test_history.py`**

```python
def test_conditional_ratings_update_only_on_matching_condition(tiny_table):
    t = tiny_table.copy()
    t["is_wet"] = [True, True, False, False, False, False]
    t["track_type"] = ["street", "street", "mixed", "mixed", "street", "street"]
    history, state = replay(t, P)
    # race 1 was wet and street: a beat b
    assert state.driver_wet["a"] > 1500 > state.driver_wet["b"]
    assert state.driver_wet_n["a"] == 1 and state.driver_wet_n["b"] == 1
    assert state.driver_track["street"]["a"] > 1500
    assert "mixed" not in state.driver_track or state.driver_track_n.get("mixed", {}).get("a", 0) == 0
    r3 = history[history.race_id == 3].set_index("driver_id")
    assert r3.loc["a", "driver_wet_n_pre"] == 1 and r3.loc["a", "track_type"] == "street"
    assert r3.loc["a", "driver_track_n_pre"] == 1
    assert not r3.loc["a", "is_wet"]


def test_replay_without_condition_columns_is_all_dry_mixed(tiny_table):
    history, state = replay(tiny_table, P)
    assert not history.is_wet.any() and history.track_type.eq("mixed").all()
    assert state.driver_wet == {} and state.driver_wet_n == {}
    assert state.driver_track_n["mixed"]["a"] == 2


def test_state_json_roundtrip_with_conditionals(tiny_table, tmp_path):
    _, state = replay(tiny_table, P)
    state.to_json(tmp_path / "s.json")
    assert RatingState.from_json(tmp_path / "s.json") == state


def test_state_strengths_matches_conditional_helper(tiny_table):
    from f1pred.ratings.conditional import effective_rating
    from f1pred.ratings.history import state_strengths

    _, state = replay(tiny_table, P)
    dry, wet = state_strengths(state, "a", "x", "mixed", P)
    d = state.driver["a"]
    dt = effective_rating(d, state.driver_track["mixed"]["a"], state.driver_track_n["mixed"]["a"], P.shrink_track)
    c = state.constructor["x"]
    ct = effective_rating(c, state.constructor_track["mixed"]["x"], state.constructor_track_n["mixed"]["x"], P.shrink_track)
    assert dry == pytest.approx(dt + ct) and wet == pytest.approx(dry)
```
Also update the existing `test_history_columns_and_order` expected list to the new `HISTORY_COLUMNS`.

- [ ] **Step 3: Write `f1pred/ratings/conditional.py`**

```python
"""Shrink a conditional (wet / track-type) rating toward the overall rating. Spec 5.5."""

from __future__ import annotations

from f1pred.config import ModelParams


def effective_rating(overall: float, conditional: float, n: int, shrink: float) -> float:
    return overall + (conditional - overall) * n / (n + shrink)


def strengths(
    driver_overall: float,
    constructor_overall: float,
    driver_track: float,
    constructor_track: float,
    n_driver_track: int,
    n_constructor_track: int,
    driver_wet: float,
    constructor_wet: float,
    n_driver_wet: int,
    n_constructor_wet: int,
    params: ModelParams,
) -> tuple[float, float]:
    """(dry strength, wet strength) for one entrant on one circuit."""
    if params.use_track:
        dry = effective_rating(
            driver_overall, driver_track, n_driver_track, params.shrink_track
        ) + effective_rating(
            constructor_overall, constructor_track, n_constructor_track, params.shrink_track
        )
    else:
        dry = driver_overall + constructor_overall
    if not params.use_weather:
        return dry, dry
    wet = (
        dry
        + effective_rating(driver_overall, driver_wet, n_driver_wet, params.shrink_wet)
        - driver_overall
        + effective_rating(constructor_overall, constructor_wet, n_constructor_wet, params.shrink_wet)
        - constructor_overall
    )
    return dry, wet
```

- [ ] **Step 4: Modify `f1pred/ratings/history.py`**

Replace the module with this (it keeps every existing public name):

```python
"""Replay every session in date order and record the rating each entrant had going in."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from f1pred.config import ModelParams
from f1pred.data.track_types import DEFAULT_TRACK_TYPE
from f1pred.ratings.conditional import strengths
from f1pred.ratings.elo import apply_update, race_update, regress

HISTORY_COLUMNS = [
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
    "is_wet",
    "track_type",
    "driver_wet_pre",
    "constructor_wet_pre",
    "driver_wet_n_pre",
    "constructor_wet_n_pre",
    "driver_track_pre",
    "constructor_track_pre",
    "driver_track_n_pre",
    "constructor_track_n_pre",
]


@dataclass
class RatingState:
    driver: dict[str, float] = field(default_factory=dict)
    constructor: dict[str, float] = field(default_factory=dict)
    driver_races: dict[str, int] = field(default_factory=dict)
    season: int | None = None
    driver_wet: dict[str, float] = field(default_factory=dict)
    constructor_wet: dict[str, float] = field(default_factory=dict)
    driver_wet_n: dict[str, int] = field(default_factory=dict)
    constructor_wet_n: dict[str, int] = field(default_factory=dict)
    driver_track: dict[str, dict[str, float]] = field(default_factory=dict)
    constructor_track: dict[str, dict[str, float]] = field(default_factory=dict)
    driver_track_n: dict[str, dict[str, int]] = field(default_factory=dict)
    constructor_track_n: dict[str, dict[str, int]] = field(default_factory=dict)

    def copy(self) -> RatingState:
        return copy.deepcopy(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> RatingState:
        data = json.loads(Path(path).read_text())
        return cls(**data)


def _regress_all(state: RatingState, driver_fraction: float, constructor_fraction: float, initial: float) -> None:
    regress(state.driver, driver_fraction, initial)
    regress(state.constructor, constructor_fraction, initial)
    regress(state.driver_wet, driver_fraction, initial)
    regress(state.constructor_wet, constructor_fraction, initial)
    for d in state.driver_track.values():
        regress(d, driver_fraction, initial)
    for c in state.constructor_track.values():
        regress(c, constructor_fraction, initial)


def start_season(state: RatingState, season: int, params: ModelParams) -> None:
    """Spec 5.3: when a new season begins, regress ratings toward the initial value."""
    if state.season is not None and season > state.season:
        fraction = (
            params.regress_constructor_regulation
            if season in params.regulation_seasons
            else params.regress_constructor
        )
        _regress_all(state, params.regress_driver, fraction, params.initial_rating)
    state.season = season


def state_for_season(state: RatingState, season: int, params: ModelParams) -> RatingState:
    """A copy of `state` as it would be at the start of `season`."""
    out = state.copy()
    start_season(out, season, params)
    return out


def strength(state: RatingState, driver_id: str, constructor_id: str, params: ModelParams) -> float:
    return state.driver.get(driver_id, params.initial_rating) + state.constructor.get(
        constructor_id, params.initial_rating
    )


def state_strengths(
    state: RatingState, driver_id: str, constructor_id: str, track_type: str, params: ModelParams
) -> tuple[float, float]:
    """(dry, wet) strength using the conditional ratings held in `state`."""
    d = state.driver.get(driver_id, params.initial_rating)
    c = state.constructor.get(constructor_id, params.initial_rating)
    return strengths(
        driver_overall=d,
        constructor_overall=c,
        driver_track=state.driver_track.get(track_type, {}).get(driver_id, d),
        constructor_track=state.constructor_track.get(track_type, {}).get(constructor_id, c),
        n_driver_track=state.driver_track_n.get(track_type, {}).get(driver_id, 0),
        n_constructor_track=state.constructor_track_n.get(track_type, {}).get(constructor_id, 0),
        driver_wet=state.driver_wet.get(driver_id, d),
        constructor_wet=state.constructor_wet.get(constructor_id, c),
        n_driver_wet=state.driver_wet_n.get(driver_id, 0),
        n_constructor_wet=state.constructor_wet_n.get(constructor_id, 0),
        params=params,
    )


def _apply_conditional(
    ratings_d: dict[str, float],
    ratings_c: dict[str, float],
    counts_d: dict[str, int],
    counts_c: dict[str, int],
    finishers: list[tuple[str, str, int]],
    state: RatingState,
    params: ModelParams,
    k_scale: float,
) -> None:
    for d, c, _ in finishers:
        ratings_d.setdefault(d, state.driver[d])
        ratings_c.setdefault(c, state.constructor[c])
    upd = race_update(finishers, ratings_d, ratings_c, params, k_scale=k_scale)
    apply_update(ratings_d, upd.driver_delta)
    apply_update(ratings_c, upd.constructor_delta)
    for d, c, _ in finishers:
        counts_d[d] = counts_d.get(d, 0) + 1
        counts_c[c] = counts_c.get(c, 0) + 1


def replay(table: pd.DataFrame, params: ModelParams) -> tuple[pd.DataFrame, RatingState]:
    """Walk the table in date order. Returns (history, final state).

    history has one row per table row with the ratings *before* that session.
    """
    state = RatingState()
    rows: list[dict] = []
    ordered = table.sort_values(["date", "race_id", "is_sprint"], kind="stable")
    has_wet = "is_wet" in ordered.columns
    has_track = "track_type" in ordered.columns
    for (date, race_id, is_sprint), grp in ordered.groupby(
        ["date", "race_id", "is_sprint"], sort=True
    ):
        season = int(grp["season"].iloc[0])
        is_wet = bool(grp["is_wet"].iloc[0]) if has_wet else False
        track_type = str(grp["track_type"].iloc[0]) if has_track else DEFAULT_TRACK_TYPE
        start_season(state, season, params)
        d_track = state.driver_track.setdefault(track_type, {})
        c_track = state.constructor_track.setdefault(track_type, {})
        d_track_n = state.driver_track_n.setdefault(track_type, {})
        c_track_n = state.constructor_track_n.setdefault(track_type, {})
        for r in grp.itertuples(index=False):
            d = state.driver.setdefault(r.driver_id, params.initial_rating)
            c = state.constructor.setdefault(r.constructor_id, params.initial_rating)
            rows.append(
                {
                    "race_id": int(race_id),
                    "is_sprint": bool(is_sprint),
                    "season": season,
                    "round": int(r.round),
                    "date": date,
                    "driver_id": r.driver_id,
                    "constructor_id": r.constructor_id,
                    "driver_rating_pre": d,
                    "constructor_rating_pre": c,
                    "driver_races_pre": state.driver_races.get(r.driver_id, 0),
                    "is_wet": is_wet,
                    "track_type": track_type,
                    "driver_wet_pre": state.driver_wet.get(r.driver_id, d),
                    "constructor_wet_pre": state.constructor_wet.get(r.constructor_id, c),
                    "driver_wet_n_pre": state.driver_wet_n.get(r.driver_id, 0),
                    "constructor_wet_n_pre": state.constructor_wet_n.get(r.constructor_id, 0),
                    "driver_track_pre": d_track.get(r.driver_id, d),
                    "constructor_track_pre": c_track.get(r.constructor_id, c),
                    "driver_track_n_pre": d_track_n.get(r.driver_id, 0),
                    "constructor_track_n_pre": c_track_n.get(r.constructor_id, 0),
                }
            )
        finishers = [
            (r.driver_id, r.constructor_id, int(r.position))
            for r in grp.itertuples(index=False)
            if pd.notna(r.position)
        ]
        k_scale = params.sprint_weight if is_sprint else 1.0
        upd = race_update(finishers, state.driver, state.constructor, params, k_scale=k_scale)
        _apply_conditional(d_track, c_track, d_track_n, c_track_n, finishers, state, params, k_scale)
        if is_wet:
            _apply_conditional(
                state.driver_wet,
                state.constructor_wet,
                state.driver_wet_n,
                state.constructor_wet_n,
                finishers,
                state,
                params,
                k_scale,
            )
        apply_update(state.driver, upd.driver_delta)
        apply_update(state.constructor, upd.constructor_delta)
        for driver_id, _, _ in finishers:
            state.driver_races[driver_id] = state.driver_races.get(driver_id, 0) + 1
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return history, state
```

Note the order: conditional updates are computed **before** the overall deltas are applied, and they seed missing conditional entries from the pre-race overall rating, so a driver's first wet race starts their wet rating at their current overall rating.

- [ ] **Step 5: Run the full suite, lint, commit**

Existing Phase 1 history tests must still pass unchanged (the overall ratings are computed exactly as before).

```bash
git add f1pred/ratings/conditional.py f1pred/ratings/history.py tests/test_conditional.py tests/test_history.py
git commit -m "feat: wet and track-type conditional Elo ratings"
```

---

### Task 6: Rain in the simulation

**Files:**
- Modify: `f1pred/sim/race.py`, `f1pred/sim/season.py`, `tests/test_race_sim.py`, `tests/test_season_sim.py`

**Interfaces:**
- `Entrant` gains `strength_wet: float | None = None` (last field). Helper property `wet_strength` returns `strength_wet if strength_wet is not None else strength`.
- `simulate_positions(entrants, params, n_runs, rng, use_grid=True, rain_probability: float = 0.0)`: draw `wet = rng.random(n_runs) < rain_probability` **first** (before any other draw), then per run use `wet_strength` for wet rows, multiply `sigma_team` and `sigma_driver` by `wet_noise_factor` on wet rows (scale the noise matrices row-wise), multiply `p_dnf` by `wet_dnf_factor` on wet rows (clip at 0.95). With `rain_probability == 0.0` skip the draw entirely so Phase 1 to 3 seeds reproduce exactly.
- `simulate_race(..., rain_probability: float = 0.0)`; `RaceForecast` gains `rain_probability: float`.
- `season.season_entrants(...)` sets `strength_wet` via `state_strengths(season_state, driver, constructor, track_type, params)` where `track_type = track_type_for(race.circuit_id, load_track_types())` and dry strength also comes from `state_strengths` (so track ratings apply).
- `season.simulate_season(..., rain_by_race: list[float] | None = None)`; `None` means all 0.0. Length must match `remaining`.
- `season.RemainingRace` gains `track_type: str = "mixed"` and `rain_probability: float = 0.0` (populated by the CLI in Task 9 from `circuit_wet_rate`).

- [ ] **Step 1: Failing tests in `tests/test_race_sim.py`**

```python
def test_rain_zero_reproduces_phase1_draws():
    a = simulate_race(_field(), P, n_runs=300, seed=5)
    b = simulate_race(_field(), P, n_runs=300, seed=5, rain_probability=0.0)
    np.testing.assert_array_equal(a.position_matrix, b.position_matrix)


def test_wet_specialist_wins_more_in_rain():
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.05) for i in range(20)]
    field[7] = Entrant("d7", "t3", 3000.0, 8, 0.05, strength_wet=3400.0)
    dry = simulate_race(field, P, n_runs=3000, seed=1, rain_probability=0.0)
    wet = simulate_race(field, P, n_runs=3000, seed=1, rain_probability=1.0)
    assert wet.p_win[7] > 0.5 > dry.p_win[7]
    assert wet.rain_probability == 1.0


def test_rain_raises_dnfs_and_spread():
    field = _field(p_dnf=0.2)
    dry_pos, _ = simulate_positions(field, P, 4000, np.random.default_rng(2), rain_probability=0.0)
    wet_pos, _ = simulate_positions(field, P, 4000, np.random.default_rng(2), rain_probability=1.0)
    # With everyone equal, more DNFs in the wet means the pole sitter finishes last more often.
    assert (wet_pos[:, 0] == 20).mean() > (dry_pos[:, 0] == 20).mean()


def test_partial_rain_is_a_mixture():
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.0) for i in range(20)]
    field[0] = Entrant("d0", "t0", 3000.0, 1, 0.0, strength_wet=3600.0)
    half = simulate_race(field, P, n_runs=4000, seed=3, rain_probability=0.5)
    wet = simulate_race(field, P, n_runs=4000, seed=3, rain_probability=1.0)
    dry = simulate_race(field, P, n_runs=4000, seed=3, rain_probability=0.0)
    assert dry.p_win[0] < half.p_win[0] < wet.p_win[0]
```

- [ ] **Step 2: Failing tests in `tests/test_season_sim.py`**

```python
def test_rain_by_race_changes_outcome():
    ents = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, None, 0.0) for i in range(4)]
    ents[3] = Entrant("d3", "t1", 3000.0, None, 0.0, strength_wet=3800.0)
    remaining = [_race(0)]
    pts = {e.driver_id: 0.0 for e in ents}
    mapping = {e.driver_id: e.constructor_id for e in ents}
    dry = simulate_season(2025, remaining, [ents], pts, {}, mapping, P, n_runs=400, seed=0, rain_by_race=[0.0])
    wet = simulate_season(2025, remaining, [ents], pts, {}, mapping, P, n_runs=400, seed=0, rain_by_race=[1.0])
    assert wet.p_driver_title[wet.driver_ids.index("d3")] > dry.p_driver_title[dry.driver_ids.index("d3")]
    with pytest.raises(ValueError):
        simulate_season(2025, remaining, [ents], pts, {}, mapping, P, n_runs=10, rain_by_race=[0.0, 0.0])


def test_season_entrants_carry_wet_strength(driver_race):
    _, state = replay(driver_race, P)
    race = RemainingRace(999, 1, "Next", "interlagos", pd.Timestamp("2025-03-16"), False, track_type="mixed")
    ents = season_entrants(driver_race, state, 2025, race, P)
    assert all(e.strength_wet is not None for e in ents)
    assert any(e.strength_wet != e.strength for e in ents)
```

- [ ] **Step 3: Implement in `f1pred/sim/race.py`**

```python
@dataclass(frozen=True)
class Entrant:
    driver_id: str
    constructor_id: str
    strength: float
    grid: int | None
    p_dnf: float
    low_confidence: bool = False
    strength_wet: float | None = None

    @property
    def wet_strength(self) -> float:
        return self.strength if self.strength_wet is None else self.strength_wet
```

Add `rain_probability: float = 0.0` to `RaceForecast` (last field) and to both functions. In `simulate_positions`, after computing `used_grid` and before the team-noise draw:

```python
    if rain_probability > 0.0:
        wet = rng.random(n_runs) < rain_probability
    else:
        wet = np.zeros(n_runs, dtype=bool)
    strength_wet = np.array([e.wet_strength for e in entrants], dtype=float)
    strength_run = np.where(wet[:, None], strength_wet[None, :], strength[None, :])
    noise_scale = np.where(wet, params.wet_noise_factor, 1.0)[:, None]
```
Then use `team_noise * noise_scale`, `driver_noise * noise_scale`, `performance = strength_run + grid_term + ...`, and `p_dnf_run = np.clip(np.where(wet[:, None], p_dnf * params.wet_dnf_factor, p_dnf), 0.0, 0.95)` with `dnf = rng.random((n_runs, n)) < p_dnf_run`. The draw order must stay: wet (only if rain_probability > 0), team noise, driver noise, dnf, dnf-key randoms.

- [ ] **Step 4: Implement in `f1pred/sim/season.py`**

`RemainingRace` gains `track_type: str = "mixed"` and `rain_probability: float = 0.0`. `season_entrants` uses `state_strengths(season_state, r.driver_id, r.constructor_id, race.track_type, params)` for `(strength, strength_wet)`. `simulate_season(..., rain_by_race=None)` validates length and passes `rain_probability=rain_by_race[i]` to both the sprint and race draws.

- [ ] **Step 5: Run the full suite, lint, commit**

```bash
git add f1pred/sim/race.py f1pred/sim/season.py tests/test_race_sim.py tests/test_season_sim.py
git commit -m "feat: rain-aware Monte Carlo with wet strengths"
```

---

### Task 7: Backtest with conditional strengths and ablation

**Files:**
- Modify: `f1pred/backtest/run.py`, `tests/test_backtest.py`

**Interfaces:**
- `entrants_for_past_race(race_rows, history_rows, dnf_lookup, params)` now computes `(strength, strength_wet)` with `conditional.strengths(...)` from the history row's `*_pre` columns (falls back to Phase 1 behaviour if the conditional columns are absent).
- `run_backtest(...)` passes `rain_probability = 1.0 if (params.use_weather and race is_wet) else 0.0` to `simulate_race`.
- New `ablation_backtest(table, history, seasons, params, n_runs, seed, dnf_cache=None) -> pd.DataFrame` with columns `variant, season, n_races, model_logloss, model_brier, model_spearman` for variants `base` (`use_weather=False, use_track=False`), `weather`, `track`, `full`, each row a season plus an `all` row per variant with the mean over all races.

- [ ] **Step 1: Failing tests in `tests/test_backtest.py`**

```python
def test_entrants_use_conditional_strengths(driver_race, sample_history):
    from f1pred.ratings.conditional import strengths

    race_id = int(driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "interlagos")].race_id.iloc[0])
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    ents = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)
    h = hist.set_index("driver_id")
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    r = h.loc["max_verstappen"]
    dry, wet = strengths(
        r.driver_rating_pre, r.constructor_rating_pre,
        r.driver_track_pre, r.constructor_track_pre, r.driver_track_n_pre, r.constructor_track_n_pre,
        r.driver_wet_pre, r.constructor_wet_pre, r.driver_wet_n_pre, r.constructor_wet_n_pre, P,
    )
    assert e.strength == pytest.approx(dry) and e.strength_wet == pytest.approx(wet)


def test_base_variant_matches_flags_off(driver_race, sample_history):
    from f1pred.backtest.run import ablation_backtest

    off = P.replace(use_weather=False, use_track=False)
    direct = run_backtest(driver_race, sample_history, [2024], off, n_runs=200, seed=1)
    abl = ablation_backtest(driver_race, sample_history, [2024], P, n_runs=200, seed=1)
    base = abl[(abl.variant == "base") & (abl.season == "2024")].iloc[0]
    assert base.model_logloss == pytest.approx(direct.seasons.model_logloss.iloc[0])
    assert set(abl.variant) == {"base", "weather", "track", "full"}
    assert (abl.season == "all").sum() == 4
```

- [ ] **Step 2: Implement**

In `entrants_for_past_race`:

```python
        if "driver_wet_pre" in hist.columns:
            dry, wet = strengths(
                float(h.driver_rating_pre), float(h.constructor_rating_pre),
                float(h.driver_track_pre), float(h.constructor_track_pre),
                int(h.driver_track_n_pre), int(h.constructor_track_n_pre),
                float(h.driver_wet_pre), float(h.constructor_wet_pre),
                int(h.driver_wet_n_pre), int(h.constructor_wet_n_pre), params,
            )
        else:
            dry = wet = float(h.driver_rating_pre + h.constructor_rating_pre)
```
and pass `strength=dry, strength_wet=wet`. In `run_backtest`, compute `is_wet = bool(race_rows["is_wet"].iloc[0]) if "is_wet" in race_rows.columns else False` and call `simulate_race(..., rain_probability=1.0 if (params.use_weather and is_wet) else 0.0)`.

```python
VARIANTS = {
    "base": {"use_weather": False, "use_track": False},
    "weather": {"use_weather": True, "use_track": False},
    "track": {"use_weather": False, "use_track": True},
    "full": {"use_weather": True, "use_track": True},
}


def ablation_backtest(table, history, seasons, params, n_runs=10_000, seed=0, dnf_cache=None) -> pd.DataFrame:
    seasons = list(seasons)
    race_ids = table[(~table["is_sprint"]) & (table["season"].isin(seasons))]["race_id"].unique()
    if dnf_cache is None:
        dnf_cache = dnf_cache_for_races(table, race_ids, params)
    rows = []
    for name, flags in VARIANTS.items():
        result = run_backtest(table, history, seasons, params.replace(**flags), n_runs, seed, dnf_cache)
        for r in result.seasons.itertuples(index=False):
            rows.append({"variant": name, "season": str(int(r.season)), "n_races": int(r.n_races),
                         "model_logloss": r.model_logloss, "model_brier": r.model_brier,
                         "model_spearman": r.model_spearman})
        rows.append({"variant": name, "season": "all", "n_races": len(result.races),
                     "model_logloss": result.races.model_logloss.mean(),
                     "model_brier": result.races.model_brier.mean(),
                     "model_spearman": result.races.model_spearman.mean()})
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Run the full suite, lint, commit**

```bash
git add f1pred/backtest/run.py tests/test_backtest.py
git commit -m "feat: conditional strengths in backtest and ablation variants"
```

---

### Task 8: Prediction inputs with rain forecast

**Files:**
- Modify: `f1pred/predict.py`, `tests/test_predict.py`

**Interfaces:**
- `RaceRef` gains `time: str | None` (race start, from `raw["races"]["time"]`), `lat: float`, `lng: float`, `track_type: str`.
- `PredictionInputs` gains `rain_probability: float` and `rain_source: str` (`"observed"` for past races, `"forecast"`, or `"historical"`).
- `build_prediction_inputs(..., rain_probability: float | None = None)`: an explicit value overrides everything (for `--rain 0.3` in the CLI). Otherwise: past race → `is_wet` from the table (1.0 or 0.0, `"observed"`); future race → `fetch_rain_probability(lat, lng, date, time, params.race_window_hours)`; if `None` → `circuit_wet_rate(table, circuit_id, date)` with source `"historical"`. When `params.use_weather` is False, rain_probability is 0.0 and source `"disabled"`.
- Future entrants use `state_strengths(season_state, driver, constructor, ref.track_type, params)`; past entrants come from `entrants_for_past_race` (already conditional).

- [ ] **Step 1: Failing tests in `tests/test_predict.py`**

```python
def test_past_race_uses_observed_wet(raw_sample, driver_race, replayed):
    history, state = replayed
    ref = resolve_race(raw_sample, driver_race, 2024, race="brazil")
    assert ref.track_type == "mixed" and ref.lat != 0
    inputs = build_prediction_inputs(raw_sample, driver_race, history, state, ref, P)
    assert inputs.rain_probability == 1.0 and inputs.rain_source == "observed"
    assert all(e.strength_wet is not None for e in inputs.entrants)


def test_future_race_uses_forecast_then_fallback(raw_sample, driver_race, replayed, monkeypatch):
    from f1pred import predict as predict_mod

    history, state = replayed
    last_id = int(driver_race[driver_race.season == 2024].race_id.max())
    table = driver_race[driver_race.race_id != last_id]
    ref = resolve_race(raw_sample, table, 2024, round=24)
    monkeypatch.setattr(predict_mod, "fetch_rain_probability", lambda *a: 0.35)
    inputs = build_prediction_inputs(raw_sample, table, history, state, ref, P)
    assert inputs.rain_probability == 0.35 and inputs.rain_source == "forecast"
    monkeypatch.setattr(predict_mod, "fetch_rain_probability", lambda *a: None)
    inputs = build_prediction_inputs(raw_sample, table, history, state, ref, P)
    assert inputs.rain_source == "historical" and 0 <= inputs.rain_probability <= 1
    inputs = build_prediction_inputs(raw_sample, table, history, state, ref, P, rain_probability=0.9)
    assert inputs.rain_probability == 0.9 and inputs.rain_source == "override"
    inputs = build_prediction_inputs(raw_sample, table, history, state, ref, P.replace(use_weather=False))
    assert inputs.rain_probability == 0.0 and inputs.rain_source == "disabled"
```
Update `test_resolve_by_name_substring_and_circuit_ref` and friends only if a `RaceRef` positional construction breaks; use keyword construction.

- [ ] **Step 2: Implement** per the interfaces. `resolve_race` merges `lat, lng` from circuits and reads `time`; `track_type = track_type_for(circuitRef, load_track_types())`. Import `fetch_rain_probability` and `circuit_wet_rate` from `f1pred.data.weather` at module level (so tests can monkeypatch `predict_mod.fetch_rain_probability`).

- [ ] **Step 3: Full suite, lint, commit**

```bash
git add f1pred/predict.py tests/test_predict.py
git commit -m "feat: rain forecast and track type in prediction inputs"
```

---

### Task 9: CLI, tuning space, docs

**Files:**
- Modify: `f1pred/cli.py`, `f1pred/tune.py`, `tests/test_cli.py`, `tests/test_tune.py`
- Create: `docs/weather-and-track.md`

**Interfaces:**
- `data update`: after downloading, `weather = build_weather_table(raw, cache_dir, params)` (wrapped in try/except: on total failure warn and continue with `weather=None`), then `build_driver_race_table(raw, start_season, weather=weather, track_types=load_track_types())`. Prints the wet-race count.
- `predict`: new option `--rain FLOAT` (override), passes `rain_probability` through to `simulate_race`, prints `Rain chance: 35% (forecast)` / `(historical rate; forecast unavailable)` / `(observed)`; charts unchanged.
- `season`: builds `RemainingRace` entries with `track_type` and `rain_probability = circuit_wet_rate(table, circuit_id, date)` and passes `rain_by_race`.
- `backtest --ablate`: prints an extra table from `ablation_backtest` and writes `outputs/ablation.csv`.
- `tune.SEARCH_SPACE` gains `"shrink_wet": [4.0, 8.0, 16.0], "shrink_track": [4.0, 8.0, 16.0], "wet_noise_factor": [1.0, 1.5, 2.0], "wet_dnf_factor": [1.0, 1.5, 2.0]`.

- [ ] **Step 1: Failing tests**

`tests/test_cli.py`:

```python
def test_data_update_builds_weather(tmp_path, monkeypatch):
    from f1pred.data import weather as weather_mod

    monkeypatch.setattr(
        hub, "hf_hub_download", lambda repo_id, filename, repo_type: str(SAMPLE_DIR / Path(filename).name)
    )
    monkeypatch.setattr(weather_mod, "fetch_race_rain_mm", lambda *a: 0.0)
    r = runner.invoke(app, ["data", "update", "--cache-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "weather.parquet").exists()
    assert "wet" in r.output.lower()
    table = pd.read_parquet(tmp_path / "driver_race.parquet")
    assert "is_wet" in table.columns and "track_type" in table.columns


def test_predict_rain_override_and_print(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        ["predict", "--season", "2024", "--race", "brazil", "--runs", "200", "--rain", "0.7",
         "--cache-dir", str(cache_dir), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    assert "Rain chance: 70%" in r.output


def test_backtest_ablate(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        ["backtest", "--seasons", "2024", "--runs", "100", "--ablate", "--cache-dir", str(cache_dir), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    abl = pd.read_csv(tmp_path / "ablation.csv")
    assert set(abl.variant) == {"base", "weather", "track", "full"}
```

`tests/test_tune.py`:

```python
def test_search_space_has_phase4_knobs():
    from f1pred.tune import SEARCH_SPACE

    assert {"shrink_wet", "shrink_track", "wet_noise_factor", "wet_dnf_factor"} <= set(SEARCH_SPACE)
```

- [ ] **Step 2: Implement** the CLI changes. The `ratings build` command needs no change (replay reads the new columns). The `cache_dir` conftest fixture must keep writing `driver_race.parquet` from the `driver_race` fixture (which now has the columns).

- [ ] **Step 3: Write `docs/weather-and-track.md`** (about 350 words): where rain data comes from and the three-hour window; the 0.5 mm threshold and its known false positives; how a wet rating is shrunk toward the overall one and why 8 races of evidence is the half-way point; the four track types and the honest caveat that they are hand-labelled; how a 40% forecast becomes a mixture inside the Monte Carlo; what the ablation table tells you; the fallbacks when Open-Meteo is unreachable.

- [ ] **Step 4: Full suite, lint, commit**

```bash
git add f1pred/cli.py f1pred/tune.py tests/test_cli.py tests/test_tune.py docs/weather-and-track.md
git commit -m "feat: weather and track type in CLI, ablation, tuning space"
```

---

### Task 10: Real data run, re-tune, README

Needs network for `data update` (Hugging Face plus about 340 Open-Meteo calls, one per race since 2010; takes a few minutes) and for the 2026 forecast.

- [ ] **Step 1: Run**

```bash
uv run f1pred data update
uv run f1pred ratings build
uv run f1pred backtest --seasons 2023-2025 --ablate
```
Record the ablation table. Sanity: wet race count since 2010 should be roughly 30 to 60; `base` numbers must equal the Phase 1 README numbers within Monte Carlo noise.

- [ ] **Step 2: Re-tune**

```bash
uv run f1pred tune --train 2015-2022 --test 2023-2025 --passes 2 --runs 2000
uv run f1pred ratings build
uv run f1pred backtest --seasons 2023-2025 --ablate
uv run f1pred predict --season 2026 --round 15
uv run f1pred season --season 2026
cp outputs/2026-15-win.png docs/results/example-win.png
cp outputs/2026-15-positions.png docs/results/example-positions.png
cp outputs/2026-title.png docs/results/example-title.png
cp outputs/calibration.png docs/results/backtest-calibration.png
```
If a later 2026 round is the next unraced one, use that.

- [ ] **Step 3: README**

- "How it works": add a "Weather and track type" paragraph linking `docs/weather-and-track.md`.
- "Backtest results": add the ablation table (variants × seasons, log loss and Brier) with two honest sentences on whether weather and track type helped.
- Usage: mention `--rain` and `--ablate`.
- Roadmap: Phase 4 done, Phase 5 remaining.
- Data credits: add Open-Meteo (CC BY 4.0).

- [ ] **Step 4: Final check, commit, push**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
git add README.md f1pred/tuned.json docs/results
git commit -m "docs: weather and track ablation results"
git push origin HEAD:main
```

---

## Self-review notes

- Spec coverage: 3.2 Task 2; 3.3 Task 3; 3.5 (`is_wet`, `track_type`) Task 4; 5.5 Task 5; 6.3 wet DNF factor Task 6; 6.7 Task 6 and 8; 8.1 additions Task 9; 10 fallbacks Tasks 2, 3, 8. The spec's "sprint rows share the race's is_wet" simplification is stated in Task 4.
- Backward compatibility: with `use_weather=False, use_track=False` the strengths reduce to `driver + constructor` and `rain_probability` is 0, so `simulate_positions` skips the wet draw and reproduces Phase 1 to 3 draws for a given seed (Task 6 test 1, Task 7 test 2).
- Type consistency: `strengths(...)` argument order is the same in `conditional.py`, `history.state_strengths`, and `backtest.entrants_for_past_race`; `RemainingRace` fields `track_type` and `rain_probability` are read in `season_entrants` and the CLI.
