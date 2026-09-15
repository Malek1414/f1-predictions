# F1 Predictions Phase 5: Driver Profile (Aggression, Risk, Form) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every driver three "mental edge" numbers computed from history before each race: **aggression** (places gained on lap 1 and over the race), **risk** (accident-type DNF rate relative to the field), **form** (recent finishes versus what the ratings expected). Feed them into the Monte Carlo through tunable scales that default to zero, tune them, and report an ablation.

**Architecture:** `data update` additionally downloads `lap_times.csv`, keeps lap-1 rows, and stores `lap1.parquet`; the driver-race table gains `lap1_position`. A new `f1pred/ratings/profile.py` computes the three measures vectorised for every driver-race (pre-race, leak-free) and for "now". `Entrant` gains `aggression, risk, form`; `simulate_positions` applies them. Backtest, predict and season pass them through; the ablation gets a `profile` variant; tuning gets the four scales; a new `f1pred profile` command prints the current numbers.

**Tech Stack:** unchanged.

Spec: `docs/superpowers/specs/2026-09-15-f1-predictions-design.md` sections 6.5 and 6.6. Existing code to read first: `f1pred/data/hub.py`, `f1pred/data/frame.py`, `f1pred/sim/race.py`, `f1pred/backtest/run.py`, `f1pred/predict.py`, `f1pred/sim/season.py`, `f1pred/cli.py`, `f1pred/tune.py`, `tests/conftest.py`.

## Global Constraints

- Same as earlier plans (uv, ruff 100, offline tests, commit trailer):
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01YJeHtrUFY8CEqTesftaBuB
  ```
- `tests/fixtures/sample/lap1.csv` is already committed: columns `raceId, driverId, position` for lap 1 of the 46 sample races (895 rows).
- All new scales default to 0.0 so Phases 1 to 4 outputs are unchanged until tuning sets them; the ablation `base` variant must reproduce `full` from Phase 4 exactly.
- Deviation from spec 6.6, decided here: the "expected position" used by the form measure is the driver's rank by pre-race strength among that race's entrants (deterministic, from the ratings history), not a 1,000-run simulation. Same information, no nested simulation. Documented in `docs/driver-profile.md`.

## File Structure

```
f1pred/config.py               MODIFY: profile params
f1pred/data/laps.py            download lap-1 positions, load fixture csv
f1pred/data/frame.py           MODIFY: lap1_position column
f1pred/ratings/profile.py      profile_features(table, history, params), latest_profiles(...)
f1pred/sim/race.py             MODIFY: Entrant fields, effects in simulate_positions
f1pred/backtest/run.py         MODIFY: profile lookup in entrants, `profile` ablation variant
f1pred/predict.py              MODIFY: profiles for past and future entrants
f1pred/sim/season.py           MODIFY: profiles in season_entrants
f1pred/report/tables.py        MODIFY: profile_table
f1pred/cli.py                  MODIFY: data update, `profile` command
f1pred/tune.py                 MODIFY: search space
tests/test_laps.py
tests/test_frame.py            MODIFY
tests/test_profile.py
tests/test_race_sim.py         MODIFY
tests/test_backtest.py         MODIFY
tests/test_predict.py          MODIFY
tests/test_cli.py              MODIFY
tests/conftest.py              MODIFY
docs/driver-profile.md
README.md                      MODIFY
```

---

### Task 1: Params

**Files:**
- Modify: `f1pred/config.py`, `tests/test_config.py`

New `ModelParams` fields (append):
```python
    # Driver profile (Phase 5)
    use_profile: bool = True
    profile_window: int = 20
    form_window: int = 6
    shrink_profile: float = 10.0
    aggression_scale: float = 0.0
    risk_noise_scale: float = 0.0
    risk_dnf_scale: float = 0.0
    form_scale: float = 0.0
```

- [ ] **Step 1: Failing test**

```python
def test_phase5_defaults():
    p = DEFAULT_PARAMS
    assert p.use_profile and p.profile_window == 20 and p.form_window == 6
    assert p.shrink_profile == 10.0
    assert (p.aggression_scale, p.risk_noise_scale, p.risk_dnf_scale, p.form_scale) == (0, 0, 0, 0)
```

- [ ] **Step 2: Implement, run, lint, commit** `feat: driver profile params`

---

### Task 2: Lap-1 positions

**Files:**
- Create: `f1pred/data/laps.py`, `tests/test_laps.py`
- Modify: `f1pred/data/frame.py`, `tests/test_frame.py`, `tests/conftest.py`

**Interfaces:**
- `laps.LAP1_FILE = "lap1.parquet"`, `laps.LAP1_COLUMNS = ["raceId", "driverId", "lap1_position"]`
- `laps.download_lap1(cache_dir: Path) -> pd.DataFrame`: `hf_hub_download(REPO_ID, "data/lap_times.csv", repo_type="dataset")` (imported from `f1pred.data.hub` so tests monkeypatch `hub.hf_hub_download`), read with `read_table_csv`, keep `lap == 1`, rename `position` → `lap1_position`, write parquet, return.
- `laps.load_lap1_csv(path) -> pd.DataFrame` (fixture loader; renames `position` → `lap1_position`).
- `laps.load_cached_lap1(cache_dir) -> pd.DataFrame | None` (None if missing).
- `frame.build_driver_race_table(..., lap1: pd.DataFrame | None = None)`: new column `lap1_position` (nullable `Int64`) joined on `(raceId, driverId)` for non-sprint rows; NA for sprint rows and when `lap1` is None. Appended to `DRIVER_RACE_COLUMNS`.

- [ ] **Step 1: Failing tests**

`tests/test_laps.py`:
```python
from pathlib import Path

import pandas as pd

from f1pred.data import hub
from f1pred.data.laps import LAP1_COLUMNS, download_lap1, load_cached_lap1, load_lap1_csv

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


def test_load_lap1_csv():
    df = load_lap1_csv(SAMPLE_DIR / "lap1.csv")
    assert list(df.columns) == LAP1_COLUMNS and len(df) == 895


def test_download_lap1_filters_and_caches(tmp_path, monkeypatch):
    src = tmp_path / "lap_times.csv"
    pd.DataFrame(
        {"raceId": [1, 1, 1], "driverId": [7, 7, 8], "lap": [1, 2, 1], "position": [3, 2, 1],
         "time": ["1:30.0"] * 3, "milliseconds": [90000] * 3}
    ).to_csv(src, index=False)
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo_id, filename, repo_type: str(src))
    df = download_lap1(tmp_path)
    assert list(df.columns) == LAP1_COLUMNS and len(df) == 2
    assert load_cached_lap1(tmp_path).equals(df)
    assert load_cached_lap1(tmp_path / "empty") is None
```

`tests/test_frame.py` addition:
```python
def test_lap1_position_column(driver_race, raw_sample):
    assert driver_race.lap1_position.dtype == "Int64"
    races = driver_race[~driver_race.is_sprint]
    assert races.lap1_position.notna().mean() > 0.95
    assert driver_race[driver_race.is_sprint].lap1_position.isna().all()
    row = races[(races.season == 2024) & (races.circuit_id == "monaco") & (races.driver_id == "leclerc")].iloc[0]
    assert row.lap1_position == 1
    assert build_driver_race_table(raw_sample).lap1_position.isna().all()
```

`tests/conftest.py`: add `lap1_sample` fixture (`load_lap1_csv(SAMPLE_DIR / "lap1.csv")`) and pass `lap1=lap1_sample` in the `driver_race` fixture; in the `cache_dir` fixture also write `lap1_sample.to_parquet(d / "lap1.parquet")`.

- [ ] **Step 2: Implement**

`f1pred/data/laps.py`:
```python
"""Lap-1 positions from the lap_times table, for the aggression measure. Spec 6.6."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pred.data import hub
from f1pred.data.hub import REPO_ID, read_table_csv

LAP1_FILE = "lap1.parquet"
LAP1_COLUMNS = ["raceId", "driverId", "lap1_position"]


def _lap1_rows(lap_times: pd.DataFrame) -> pd.DataFrame:
    df = lap_times[lap_times["lap"] == 1][["raceId", "driverId", "position"]]
    return df.rename(columns={"position": "lap1_position"}).reset_index(drop=True)[LAP1_COLUMNS]


def load_lap1_csv(path: Path) -> pd.DataFrame:
    df = read_table_csv(Path(path))
    if "lap" in df.columns:
        return _lap1_rows(df)
    return df.rename(columns={"position": "lap1_position"})[LAP1_COLUMNS]


def download_lap1(cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = hub.hf_hub_download(REPO_ID, "data/lap_times.csv", repo_type="dataset")
    df = _lap1_rows(read_table_csv(Path(local)))
    df.to_parquet(cache_dir / LAP1_FILE, index=False)
    return df


def load_cached_lap1(cache_dir: Path) -> pd.DataFrame | None:
    path = Path(cache_dir) / LAP1_FILE
    return pd.read_parquet(path) if path.exists() else None
```

In `frame.py`, `_session_rows` gains a `lap1: pd.DataFrame | None` argument; for race rows merge on `["raceId", "driverId"]` (left) and take `lap1_position` as `Int64`; for sprint rows and when `lap1` is None set `pd.array([pd.NA] * len(df), dtype="Int64")`.

- [ ] **Step 3: Full suite, lint, commit** `feat: lap-1 positions in driver-race table`

---

### Task 3: Profile features

**Files:**
- Create: `f1pred/ratings/profile.py`, `tests/test_profile.py`, `docs/driver-profile.md`

**Interfaces:**
- `PROFILE_COLUMNS = ["race_id", "driver_id", "aggression", "risk", "form", "profile_n"]`
- `profile_features(table: pd.DataFrame, history: pd.DataFrame, params: ModelParams) -> pd.DataFrame` with `PROFILE_COLUMNS`, one row per non-sprint driver-race, values computed from rows strictly before that race (shift by one within driver):
  - Inputs per row (non-sprint only, sorted by date then race_id):
    - `strength_pre = driver_rating_pre + constructor_rating_pre` from `history`; `expected_position = rank of strength_pre (desc) within race` (ties: average).
    - `gain_lap1 = grid - lap1_position` (NaN if lap1 missing), `gain_race = grid - position` (NaN if not classified), `gain = nanmean(gain_lap1, gain_race)` (NaN if both missing).
    - `accident = dnf_kind == "accident"` (0/1).
    - `form_raw = expected_position - position` (NaN if not classified).
  - Per driver, rolling over the previous `profile_window` rows (`.shift(1)` so the current race is excluded): `gain_mean`, `gain_n` (count of non-NaN), `acc_rate` (mean of accident over rows), `acc_n`; over the previous `form_window` rows: `form_mean`, `form_n`.
  - Field baselines, over the previous `profile_window * 20` rows of the whole table in date order (`shift(1)`): `field_gain` (mean gain), `field_acc` (mean accident, floored at 0.01).
  - `aggression = (gain_mean - field_gain) * gain_n / (gain_n + shrink_profile)`; 0 when `gain_n == 0`.
  - `risk = (acc_rate / field_acc - 1) * acc_n / (acc_n + shrink_profile)`; 0 when `acc_n == 0`.
  - `form = form_mean * form_n / (form_n + shrink_profile)`; 0 when `form_n == 0`.
  - `profile_n = gain_n`.
- `latest_profiles(table, history, params) -> dict[str, Profile]`: same computation but taken **after** each driver's last race (no shift), keyed by driver_id. `@dataclass(frozen=True) Profile(aggression: float, risk: float, form: float, n: int)`; `Profile.zero()`.
- `profile_lookup(features: pd.DataFrame) -> dict[tuple[int, str], Profile]`.

- [ ] **Step 1: Failing tests `tests/test_profile.py`**

```python
import numpy as np
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
    cols = ["season", "round", "race_id", "date", "circuit_id", "race_name", "driver_id", "driver_code",
            "driver_name", "constructor_id", "constructor_name", "grid", "position", "status", "dnf",
            "dnf_kind", "points", "is_sprint", "is_wet", "track_type", "lap1_position"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["position"] = df["position"].astype("Int64")
    df["lap1_position"] = df["lap1_position"].astype("Int64")
    return df


def _row(i, driver, team, grid, pos, lap1, kind=None):
    return [2020, i + 1, i, f"2020-03-{1 + i:02d}", "c", "r", driver, driver[:3], driver, team, team,
            grid, pos, "Finished" if pos is not None else "Accident", pos is None,
            kind, 0.0, False, False, "mixed", lap1]


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
    for i in range(8):
        # "under" is rated top after a few races but keeps finishing 3rd; "over" keeps finishing 1st
        rows.append(_row(i, "over", "x", 3, 1, 3))
        rows.append(_row(i, "mid", "y", 2, 2, 2))
        rows.append(_row(i, "under", "z", 1, 3, 1))
    t = _table(rows)
    history, _ = replay(t, P)
    f = profile_features(t, history, P)
    last = f[f.race_id == 7].set_index("driver_id")
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
```

- [ ] **Step 2: Implement `f1pred/ratings/profile.py`**

```python
"""Aggression, risk and form per driver, computed leak-free before each race. Spec 6.6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.config import ModelParams

PROFILE_COLUMNS = ["race_id", "driver_id", "aggression", "risk", "form", "profile_n"]
FIELD_ACC_FLOOR = 0.01


@dataclass(frozen=True)
class Profile:
    aggression: float
    risk: float
    form: float
    n: int

    @classmethod
    def zero(cls) -> Profile:
        return cls(0.0, 0.0, 0.0, 0)


def _raw_signals(table: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    races = table[~table["is_sprint"]].copy()
    hist = history[~history["is_sprint"]][
        ["race_id", "driver_id", "driver_rating_pre", "constructor_rating_pre"]
    ]
    df = races.merge(hist, on=["race_id", "driver_id"], how="left")
    df["strength_pre"] = df["driver_rating_pre"] + df["constructor_rating_pre"]
    df["expected_position"] = df.groupby("race_id")["strength_pre"].rank(
        ascending=False, method="average"
    )
    position = df["position"].astype(float)
    lap1 = df["lap1_position"].astype(float) if "lap1_position" in df.columns else np.nan
    gain_lap1 = df["grid"] - lap1
    gain_race = df["grid"] - position
    df["gain"] = pd.concat([gain_lap1, gain_race], axis=1).mean(axis=1, skipna=True)
    df["accident"] = (df["dnf_kind"] == "accident").astype(float)
    df["form_raw"] = df["expected_position"] - position
    return df.sort_values(["date", "race_id", "driver_id"], kind="stable").reset_index(drop=True)


def _rolling(df: pd.DataFrame, params: ModelParams, shift: int) -> pd.DataFrame:
    """Per-driver and field rolling statistics. shift=1 excludes the current row."""
    g = df.groupby("driver_id", sort=False)
    w, fw = params.profile_window, params.form_window

    def roll(col: str, window: int, func: str) -> pd.Series:
        return g[col].transform(
            lambda s: getattr(s.rolling(window, min_periods=1), func)().shift(shift)
        )

    out = pd.DataFrame(index=df.index)
    out["gain_mean"] = roll("gain", w, "mean")
    out["gain_n"] = roll("gain", w, "count").fillna(0)
    out["acc_rate"] = roll("accident", w, "mean")
    out["acc_n"] = roll("accident", w, "count").fillna(0)
    out["form_mean"] = roll("form_raw", fw, "mean")
    out["form_n"] = roll("form_raw", fw, "count").fillna(0)
    field_window = w * 20
    out["field_gain"] = df["gain"].rolling(field_window, min_periods=1).mean().shift(shift)
    out["field_acc"] = (
        df["accident"].rolling(field_window, min_periods=1).mean().shift(shift).clip(lower=FIELD_ACC_FLOOR)
    )
    return out


def _combine(stats: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    s = params.shrink_profile
    gain_w = stats["gain_n"] / (stats["gain_n"] + s)
    acc_w = stats["acc_n"] / (stats["acc_n"] + s)
    form_w = stats["form_n"] / (stats["form_n"] + s)
    out = pd.DataFrame(index=stats.index)
    out["aggression"] = ((stats["gain_mean"] - stats["field_gain"]) * gain_w).fillna(0.0)
    out["risk"] = ((stats["acc_rate"] / stats["field_acc"] - 1.0) * acc_w).fillna(0.0)
    out["form"] = (stats["form_mean"] * form_w).fillna(0.0)
    out["profile_n"] = stats["gain_n"].astype(int)
    return out


def profile_features(table: pd.DataFrame, history: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    df = _raw_signals(table, history)
    if df.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    stats = _rolling(df, params, shift=1)
    # A driver's very first race: the field baseline is defined but their own stats are NaN -> 0.
    out = _combine(stats, params)
    out.insert(0, "driver_id", df["driver_id"].to_numpy())
    out.insert(0, "race_id", df["race_id"].astype(int).to_numpy())
    return out[PROFILE_COLUMNS].reset_index(drop=True)


def latest_profiles(table: pd.DataFrame, history: pd.DataFrame, params: ModelParams) -> dict[str, Profile]:
    df = _raw_signals(table, history)
    if df.empty:
        return {}
    out = _combine(_rolling(df, params, shift=0), params)
    out["driver_id"] = df["driver_id"].to_numpy()
    last = out.groupby("driver_id").tail(1)
    return {
        r.driver_id: Profile(float(r.aggression), float(r.risk), float(r.form), int(r.profile_n))
        for r in last.itertuples(index=False)
    }


def profile_lookup(features: pd.DataFrame) -> dict[tuple[int, str], Profile]:
    return {
        (int(r.race_id), r.driver_id): Profile(
            float(r.aggression), float(r.risk), float(r.form), int(r.profile_n)
        )
        for r in features.itertuples(index=False)
    }
```

Note for `test_features_are_pre_race_and_shrunk`: the field gain in that fixture is exactly 0 in every race (grid places gained sum to zero across finishers when everyone finishes), so `aggression = gain_mean * n/(n+10)`. Hero's `gain` is 2.0 every race; after 11 races `gain_n = 11`.

Note for `test_risk_relative_to_field`: field accident rate over the previous rows is 0.25 (one of four drivers crashes every race), crash's own rate is 1.0, so `(1/0.25 - 1) = 3`, shrunk by `19/29`; safe drivers have rate 0, so `(0 - 1) = -1` shrunk by `19/29`.

- [ ] **Step 3: Write `docs/driver-profile.md`** (about 350 words): what each number means with an F1 example (a lap-1 charger, a driver with a crash habit, a driver on a hot streak); why they are centred at zero and shrunk; why form uses strength rank rather than a nested simulation; how the scales enter the Monte Carlo (aggression and form as pace, risk as wider spread and more accidents); the honest caveat that this can add colour without adding accuracy, which is why the ablation decides whether the scales stay above zero.

- [ ] **Step 4: Full suite, lint, commit** `feat: aggression, risk and form driver profile features`

---

### Task 4: Profile effects in the simulation

**Files:**
- Modify: `f1pred/sim/race.py`, `tests/test_race_sim.py`

**Interfaces:**
- `Entrant` gains `aggression: float = 0.0`, `risk: float = 0.0`, `form: float = 0.0` (after `strength_wet`).
- In `simulate_positions`, when `params.use_profile`:
  - `performance += params.aggression_scale * aggression + params.form_scale * form` (per entrant, broadcast over runs)
  - driver noise sigma per entrant: `sigma_driver * clip(1 + risk_noise_scale * risk, 0.5, 3.0)`
  - `p_dnf` per entrant: `clip(p_dnf * (1 + risk_dnf_scale * risk), 0.01, 0.95)` (applied before the wet factor)
  - No extra RNG draws, so Phase 4 seeds reproduce when all scales are 0 (or `use_profile=False`).

- [ ] **Step 1: Failing tests**

```python
def test_profile_scales_zero_reproduce_phase4():
    field = _field()
    field[2] = Entrant("d2", "t1", 3000.0, 3, 0.1, aggression=2.0, risk=1.0, form=1.5)
    a = simulate_race(_field(), P, n_runs=300, seed=21)
    b = simulate_race(field, P, n_runs=300, seed=21)
    np.testing.assert_array_equal(a.position_matrix, b.position_matrix)


def test_aggression_and_form_add_pace():
    params = P.replace(aggression_scale=50.0, form_scale=50.0)
    field = _field()
    field[9] = Entrant("d9", "t4", 3000.0, 10, 0.1, aggression=3.0, form=3.0)
    f = simulate_race(field, params, n_runs=3000, seed=22, use_grid=False)
    assert f.driver_ids[int(np.argmax(f.p_win))] == "d9"


def test_risk_widens_and_crashes():
    params = P.replace(risk_noise_scale=1.0, risk_dnf_scale=2.0)
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.1) for i in range(20)]
    field[0] = Entrant("d0", "t0", 3000.0, 1, 0.1, risk=1.0)
    risky = simulate_race(field, params, n_runs=4000, seed=23, use_grid=False)
    calm = simulate_race(_field(), params, n_runs=4000, seed=23, use_grid=False)
    assert risky.position_p90[0] > calm.position_p90[0]
    assert risky.expected_position[0] > calm.expected_position[0]


def test_use_profile_false_ignores_fields():
    params = P.replace(aggression_scale=50.0, use_profile=False)
    field = _field()
    field[9] = Entrant("d9", "t4", 3000.0, 10, 0.1, aggression=3.0)
    f = simulate_race(field, params, n_runs=300, seed=24)
    g = simulate_race(_field(), params, n_runs=300, seed=24)
    np.testing.assert_array_equal(f.position_matrix, g.position_matrix)
```

- [ ] **Step 2: Implement, full suite, lint, commit** `feat: profile effects in the race Monte Carlo`

---

### Task 5: Wire profiles through backtest, predict, season

**Files:**
- Modify: `f1pred/backtest/run.py`, `f1pred/predict.py`, `f1pred/sim/season.py`, `tests/test_backtest.py`, `tests/test_predict.py`, `tests/test_season_sim.py`

**Interfaces:**
- `backtest.entrants_for_past_race(race_rows, history_rows, dnf_lookup, params, profiles: Mapping[tuple[int, str], Profile] | None = None)`: fills the three fields (zero when `profiles` is None or key missing).
- `backtest.run_backtest(..., profiles=None)`: computes `profile_lookup(profile_features(table, history, params))` once when `None` and `params.use_profile`.
- `backtest.VARIANTS` gains `"profile": {"use_weather": True, "use_track": True, "use_profile": True}` and every other variant sets `"use_profile": False`. (`full` now means weather + track, profile off; `profile` is everything.)
- `predict.build_prediction_inputs(...)`: past race → profiles from `profile_lookup`; future race → `latest_profiles(table, history, params)`.
- `season.season_entrants(table, state, season, race, params, profiles: Mapping[str, Profile] | None = None)`.

- [ ] **Step 1: Failing tests**

`tests/test_backtest.py`:
```python
def test_entrants_carry_profiles(driver_race, sample_history):
    from f1pred.ratings.profile import profile_features, profile_lookup

    race_id = int(driver_race[(driver_race.season == 2024) & (~driver_race.is_sprint)].race_id.iloc[-1])
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    lookup = profile_lookup(profile_features(driver_race, sample_history, P))
    ents = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P, lookup)
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    p = lookup[(race_id, "max_verstappen")]
    assert (e.aggression, e.risk, e.form) == (p.aggression, p.risk, p.form)
    plain = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)
    assert all(x.aggression == 0 and x.risk == 0 and x.form == 0 for x in plain)


def test_ablation_has_profile_variant(driver_race, sample_history):
    from f1pred.backtest.run import ablation_backtest

    abl = ablation_backtest(driver_race, sample_history, [2024], P, n_runs=100, seed=1)
    assert set(abl.variant) == {"base", "weather", "track", "full", "profile"}
```

`tests/test_predict.py`:
```python
def test_prediction_inputs_include_profiles(raw_sample, driver_race, replayed):
    history, state = replayed
    ref = resolve_race(raw_sample, driver_race, 2024, round=8)
    inputs = build_prediction_inputs(raw_sample, driver_race, history, state, ref, P)
    assert any(e.aggression != 0 or e.risk != 0 or e.form != 0 for e in inputs.entrants)
    last_id = int(driver_race[driver_race.season == 2024].race_id.max())
    table = driver_race[driver_race.race_id != last_id]
    ref = resolve_race(raw_sample, table, 2024, round=24)
    inputs = build_prediction_inputs(raw_sample, table, history, state, ref, P)
    assert any(e.aggression != 0 for e in inputs.entrants)
```

`tests/test_season_sim.py`:
```python
def test_season_entrants_take_profiles(driver_race):
    from f1pred.ratings.profile import Profile

    _, state = replay(driver_race, P)
    race = RemainingRace(999, 1, "Next", "bahrain", pd.Timestamp("2025-03-16"), False)
    ents = season_entrants(driver_race, state, 2025, race, P, {"max_verstappen": Profile(1.0, 0.5, 0.2, 9)})
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    assert (e.aggression, e.risk, e.form) == (1.0, 0.5, 0.2)
```

- [ ] **Step 2: Implement, full suite, lint, commit** `feat: profiles in backtest, prediction and season inputs`

---

### Task 6: CLI, tuning, README

**Files:**
- Modify: `f1pred/cli.py`, `f1pred/tune.py`, `f1pred/report/tables.py`, `tests/test_cli.py`, `tests/test_tune.py`, `tests/test_report.py`

**Interfaces:**
- `data update` also calls `download_lap1(cache_dir)` (wrapped: on failure warn and continue with `lap1=None`), passes `lap1` to `build_driver_race_table`, prints lap-1 coverage.
- `season` command computes `latest_profiles` once and passes it to every `season_entrants` call.
- New command `f1pred profile [--season 2026] [--cache-dir]`: prints `profile_table(latest_profiles(...), names, drivers=<entrants of the latest race>)` with columns `Driver, Aggression, Risk, Form, Races`, sorted by aggression desc; one line under the table explaining the sign of each.
- `tables.profile_table(profiles: Mapping[str, Profile], names: Mapping[str, str], drivers: Iterable[str]) -> Table`.
- `tune.SEARCH_SPACE` gains `"aggression_scale": [0.0, 5.0, 10.0, 20.0], "risk_noise_scale": [0.0, 0.25, 0.5], "risk_dnf_scale": [0.0, 0.25, 0.5], "form_scale": [0.0, 5.0, 10.0]`. `objective` must build the profile lookup per params (it depends on `history`, which depends on K) and pass it to `run_backtest`.

- [ ] **Step 1: Failing tests**

`tests/test_cli.py`:
```python
def test_profile_command(cache_dir):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(app, ["profile", "--cache-dir", str(cache_dir)])
    assert r.exit_code == 0, r.output
    assert "Aggression" in r.output and "Max Verstappen" in r.output


def test_data_update_downloads_lap1(tmp_path, monkeypatch):
    from f1pred.data import weather as weather_mod

    def fake_download(repo_id, filename, repo_type):
        name = Path(filename).name
        if name == "lap_times.csv":
            return str(SAMPLE_DIR / "lap1.csv")
        return str(SAMPLE_DIR / name)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(weather_mod, "fetch_race_rain_mm", lambda *a: 0.0)
    r = runner.invoke(app, ["data", "update", "--cache-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "lap1.parquet").exists()
    assert pd.read_parquet(tmp_path / "driver_race.parquet").lap1_position.notna().any()
```
(`download_lap1` must therefore accept a CSV that already lacks a `lap` column: route it through `load_lap1_csv`-style handling, i.e. use `_lap1_rows` only when `lap` is present.)

`tests/test_tune.py`:
```python
def test_search_space_has_profile_scales():
    from f1pred.tune import SEARCH_SPACE

    assert {"aggression_scale", "risk_noise_scale", "risk_dnf_scale", "form_scale"} <= set(SEARCH_SPACE)
```

`tests/test_report.py`:
```python
def test_profile_table():
    from f1pred.ratings.profile import Profile
    from f1pred.report.tables import profile_table

    t = profile_table({"d0": Profile(1.2, -0.3, 0.8, 12), "d1": Profile(-0.5, 0.9, -0.2, 4)}, NAMES, ["d0", "d1"])
    text = _render(t)
    assert "Driver 0" in text and "+1.20" in text and "-0.30" in text
```

- [ ] **Step 2: Implement, full suite, lint, commit** `feat: profile command, lap-1 download, profile tuning space`

---

### Task 7: Real data run, tune, README

- [ ] **Step 1: Run**

```bash
uv run f1pred data update           # now also fetches lap_times.csv (about 25 MB)
uv run f1pred ratings build
uv run f1pred profile
uv run f1pred backtest --seasons 2023-2025 --ablate
```
Sanity: the `profile` table should show known lap-1 chargers and crash-prone drivers where you would expect them; `base` through `full` rows match the Phase 4 ablation within Monte Carlo noise; `profile` equals `full` exactly while all scales are 0.

- [ ] **Step 2: Tune**

```bash
uv run f1pred tune --train 2015-2022 --test 2023-2025 --passes 2 --runs 2000
uv run f1pred ratings build
uv run f1pred backtest --seasons 2023-2025 --ablate
uv run f1pred predict --season 2026 --round 15    # or next unraced round
uv run f1pred season --season 2026
cp outputs/2026-*-win.png docs/results/example-win.png
cp outputs/2026-*-positions.png docs/results/example-positions.png
cp outputs/2026-title.png docs/results/example-title.png
cp outputs/calibration.png docs/results/backtest-calibration.png
```

- [ ] **Step 3: README**

- "How it works": add "Driver profile (mental edge)" paragraph linking `docs/driver-profile.md`.
- "Example output": add the `f1pred profile` table (top and bottom 5 by aggression).
- "Backtest results": replace the ablation table with the five-variant one; two honest sentences on whether the profile helped and which scales tuning kept above zero.
- Usage: add `f1pred profile`.
- Roadmap: all five phases done; list two or three candidate next steps (qualifying-time pace signal from FastF1, live ratings update within a simulated season, Streamlit dashboard).

- [ ] **Step 4: Final check, commit, push**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
git add README.md f1pred/tuned.json docs/results
git commit -m "docs: driver profile results and final ablation"
git push origin HEAD:main
```

---

## Self-review notes

- Spec coverage: 6.5 Task 4; 6.6 Task 3 (with the stated deviation for form's expected position); lap_times ingestion (3.1) Task 2; tuning and ablation (8.1) Tasks 5 and 6; docs and README Task 7.
- Inertness: all four scales default to 0 and `simulate_positions` adds no RNG draws for the profile, so Phase 4 outputs are reproduced bit-for-bit until tuning changes the scales (Task 4 test 1, Task 5 ablation).
- Type consistency: `Profile` fields `(aggression, risk, form, n)` are read identically in `backtest.entrants_for_past_race`, `predict`, `season.season_entrants`, and `report.profile_table`.
