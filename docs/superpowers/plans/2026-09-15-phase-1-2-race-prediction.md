# F1 Predictions Phase 1 + 2: Race Prediction Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `f1pred` package end to end for Phases 1 and 2 of the spec: Hugging Face data → driver-race table → Elo ratings → Monte Carlo race simulation → backtest with baselines → CLI with tables and charts, plus tuning and docs.

**Architecture:** Five modules in strict dependency order, `data → ratings → sim → backtest → report`, with `cli.py` and `predict.py` on top. Every module exposes plain functions over pandas DataFrames and small dataclasses. The ratings replay writes a "rating before each race" history, which is the only rating source the simulation and backtest read, guaranteeing no leakage.

**Tech Stack:** Python 3.12, `uv`, pandas, pyarrow, numpy, huggingface_hub, typer, rich, matplotlib, pytest, ruff.

Spec: `docs/superpowers/specs/2026-09-15-f1-predictions-design.md`. Read sections 3 to 11 before starting.

## Global Constraints

- Python `>=3.12`, managed with `uv`. Run everything as `uv run <cmd>`.
- Package name `f1pred`, CLI entry point `f1pred`.
- Data source: Hugging Face dataset `tracinginsights/RaceData`, CSV files under `data/<table>.csv`, nulls encoded as the literal string `\N`.
- Rating replay uses `start_season = 2010` onward.
- Initial rating 1500 for every driver and constructor.
- Default simulation runs: 10,000 for a race.
- `data/cache/` and `outputs/` are git-ignored (already in `.gitignore`).
- Every commit message ends with:
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01YJeHtrUFY8CEqTesftaBuB
  ```
- Line length 100, `ruff check` and `ruff format --check` clean before each commit.
- Tests never touch the network. Real-data tests use `tests/fixtures/sample/` (2023 and 2024 seasons, already committed).
- Windows-style `\r\n` never; UTF-8 everywhere.

## File Structure

```
pyproject.toml                    project metadata, deps, ruff + pytest config
LICENSE                           MIT
README.md                         overview, usage, results (written last)
f1pred/__init__.py
f1pred/config.py                  ModelParams dataclass, defaults, paths, tuned-params loading
f1pred/data/__init__.py
f1pred/data/hub.py                download CSVs from the Hub, Parquet cache, fallback rules
f1pred/data/frame.py              build the driver-race table, DNF classification
f1pred/ratings/__init__.py
f1pred/ratings/elo.py             expected score, pairwise race update, regression
f1pred/ratings/history.py         replay history, RatingState, pre-race ratings table
f1pred/sim/__init__.py
f1pred/sim/dnf.py                 DNF probability
f1pred/sim/race.py                Entrant, RaceForecast, simulate_race
f1pred/backtest/__init__.py
f1pred/backtest/scoring.py        log loss, Brier, Spearman, baselines
f1pred/backtest/run.py            run_backtest, calibration table
f1pred/report/__init__.py
f1pred/report/tables.py           rich tables
f1pred/report/charts.py           matplotlib PNGs
f1pred/predict.py                 resolve a race, assemble entrants for past or future race
f1pred/tune.py                    coordinate-descent parameter search
f1pred/cli.py                     Typer app
tests/conftest.py                 fixtures: raw sample tables, driver-race table
tests/test_hub.py
tests/test_frame.py
tests/test_elo.py
tests/test_history.py
tests/test_dnf.py
tests/test_race_sim.py
tests/test_scoring.py
tests/test_backtest.py
tests/test_predict.py
tests/test_cli.py
docs/elo.md  docs/monte-carlo.md  docs/dnf-model.md  docs/backtesting.md
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `f1pred/__init__.py`, `f1pred/config.py`, `f1pred/data/__init__.py`, `f1pred/ratings/__init__.py`, `f1pred/sim/__init__.py`, `f1pred/backtest/__init__.py`, `f1pred/report/__init__.py`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `f1pred.config.ModelParams` (frozen dataclass, fields below), `DEFAULT_PARAMS`, `load_params(path: Path | None = None) -> ModelParams`, `DEFAULT_CACHE_DIR = Path("data/cache")`, `DEFAULT_OUTPUT_DIR = Path("outputs")`, `TUNED_PARAMS_PATH = Path(__file__).parent / "tuned.json"`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "f1pred"
version = "0.1.0"
description = "F1 race and championship predictions with Elo ratings and Monte Carlo simulation"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=17",
    "numpy>=2.0",
    "huggingface_hub>=0.25",
    "typer>=0.12",
    "rich>=13",
    "matplotlib>=3.9",
]

[project.scripts]
f1pred = "f1pred.cli:app"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["f1pred"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `LICENSE`** (MIT, copyright 2026 Malek Hassan) and empty `__init__.py` files listed above. `f1pred/__init__.py` contains `__version__ = "0.1.0"`.

- [ ] **Step 3: Write the failing test `tests/test_config.py`**

```python
import json

from f1pred.config import DEFAULT_PARAMS, ModelParams, load_params


def test_defaults_match_spec():
    p = DEFAULT_PARAMS
    assert p.initial_rating == 1500.0
    assert p.start_season == 2010
    assert p.regulation_seasons == (2014, 2017, 2022, 2026)
    assert p.teammate_weight == 2.0
    assert p.sprint_weight == 0.5


def test_load_params_overrides_from_json(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"k_driver": 40.0, "sigma_team": 70.0}))
    p = load_params(path)
    assert p.k_driver == 40.0
    assert p.sigma_team == 70.0
    assert p.k_constructor == DEFAULT_PARAMS.k_constructor


def test_load_params_missing_file_returns_defaults(tmp_path):
    assert load_params(tmp_path / "nope.json") == DEFAULT_PARAMS


def test_replace_returns_new_params():
    p = DEFAULT_PARAMS.replace(k_driver=1.0)
    assert p.k_driver == 1.0
    assert isinstance(p, ModelParams)
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'f1pred.config'`

- [ ] **Step 5: Write `f1pred/config.py`**

```python
"""Model parameters and default paths."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_OUTPUT_DIR = Path("outputs")
TUNED_PARAMS_PATH = Path(__file__).parent / "tuned.json"


@dataclass(frozen=True)
class ModelParams:
    """All tunable knobs. See spec sections 5, 6 and 8."""

    # Elo
    initial_rating: float = 1500.0
    k_driver: float = 24.0
    k_constructor: float = 32.0
    teammate_weight: float = 2.0
    sprint_weight: float = 0.5
    regress_driver: float = 0.1
    regress_constructor: float = 0.4
    regress_constructor_regulation: float = 0.7
    regulation_seasons: tuple[int, ...] = (2014, 2017, 2022, 2026)
    start_season: int = 2010
    min_races_for_confidence: int = 5
    # Simulation
    sigma_team: float = 60.0
    sigma_driver: float = 80.0
    grid_bonus: float = 8.0
    # DNF
    dnf_window: int = 20
    shrink_dnf: float = 10.0
    circuit_min_rows: int = 40
    circuit_factor_min: float = 0.5
    circuit_factor_max: float = 2.0

    def replace(self, **changes: object) -> ModelParams:
        return dataclasses.replace(self, **changes)


DEFAULT_PARAMS = ModelParams()


def load_params(path: Path | None = None) -> ModelParams:
    """Defaults overridden by a JSON file of field values, if the file exists."""
    path = TUNED_PARAMS_PATH if path is None else path
    if not path.exists():
        return DEFAULT_PARAMS
    overrides = json.loads(path.read_text())
    if "regulation_seasons" in overrides:
        overrides["regulation_seasons"] = tuple(overrides["regulation_seasons"])
    return DEFAULT_PARAMS.replace(**overrides)
```

- [ ] **Step 6: Run tests and lint**

Run: `uv run pytest tests/test_config.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 4 passed, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml LICENSE uv.lock f1pred tests
git commit -m "feat: project scaffold with ModelParams config"
```

---

### Task 2: Hub download and Parquet cache

**Files:**
- Create: `f1pred/data/hub.py`, `tests/conftest.py`, `tests/test_hub.py`

**Interfaces:**
- Produces:
  - `REPO_ID = "tracinginsights/RaceData"`, `TABLES = ("races", "results", "qualifying", "sprint_results", "drivers", "constructors", "circuits", "status")`
  - `read_table_csv(path: Path) -> pd.DataFrame` (handles `\N` nulls)
  - `load_raw_from_dir(directory: Path, tables=TABLES) -> dict[str, pd.DataFrame]` (reads CSVs; used by tests and fixtures)
  - `download_tables(cache_dir: Path, tables=TABLES) -> dict[str, pd.DataFrame]` (fetches from Hub, writes Parquet)
  - `load_cached_tables(cache_dir: Path, tables=TABLES) -> dict[str, pd.DataFrame]` (raises `CacheMissingError`)
  - `load_tables(cache_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]` (download-or-cache with the spec's fallback rules; raises `DataUnavailableError`)
  - Exceptions `CacheMissingError(Exception)`, `DataUnavailableError(Exception)`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

from f1pred.data.hub import load_raw_from_dir

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


@pytest.fixture(scope="session")
def raw_sample() -> dict:
    return load_raw_from_dir(SAMPLE_DIR)
```

- [ ] **Step 2: Write the failing tests `tests/test_hub.py`**

```python
import logging
from pathlib import Path

import pandas as pd
import pytest

from f1pred.data import hub
from f1pred.data.hub import (
    TABLES,
    CacheMissingError,
    DataUnavailableError,
    load_cached_tables,
    load_tables,
    read_table_csv,
)

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


def test_read_table_csv_turns_backslash_n_into_null():
    df = read_table_csv(SAMPLE_DIR / "results.csv")
    assert df["position"].isna().any()
    assert pd.api.types.is_numeric_dtype(df["position"])
    assert df["statusId"].notna().all()


def test_raw_sample_has_all_tables(raw_sample):
    assert set(raw_sample) == set(TABLES)
    assert len(raw_sample["races"]) == 46


def _fake_download(repo_id, filename, repo_type):
    assert repo_id == hub.REPO_ID and repo_type == "dataset"
    return str(SAMPLE_DIR / Path(filename).name)


def test_load_tables_downloads_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "hf_hub_download", _fake_download)
    tables = load_tables(tmp_path)
    assert set(tables) == set(TABLES)
    assert (tmp_path / "results.parquet").exists()
    cached = load_cached_tables(tmp_path)
    assert len(cached["results"]) == len(tables["results"])


def test_load_tables_falls_back_to_cache_when_offline(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(hub, "hf_hub_download", _fake_download)
    load_tables(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(hub, "hf_hub_download", boom)
    with caplog.at_level(logging.WARNING):
        tables = load_tables(tmp_path, refresh=True)
    assert "cache" in caplog.text.lower()
    assert len(tables["races"]) == 46


def test_load_tables_raises_when_offline_and_no_cache(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(hub, "hf_hub_download", boom)
    with pytest.raises(DataUnavailableError, match="f1pred data update"):
        load_tables(tmp_path)


def test_load_cached_tables_missing_raises(tmp_path):
    with pytest.raises(CacheMissingError):
        load_cached_tables(tmp_path)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_hub.py -v`
Expected: FAIL with `ImportError` on `f1pred.data.hub`.

- [ ] **Step 4: Write `f1pred/data/hub.py`**

```python
"""Download RaceData tables from the Hugging Face Hub and cache them as Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

REPO_ID = "tracinginsights/RaceData"
TABLES = (
    "races",
    "results",
    "qualifying",
    "sprint_results",
    "drivers",
    "constructors",
    "circuits",
    "status",
)
NA_VALUES = ["\\N"]


class CacheMissingError(Exception):
    """A required table is not in the local cache."""


class DataUnavailableError(Exception):
    """Neither the Hub nor the cache can supply the data."""


def read_table_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)


def load_raw_from_dir(directory: Path, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    return {t: read_table_csv(Path(directory) / f"{t}.csv") for t in tables}


def download_tables(cache_dir: Path, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        local = hf_hub_download(REPO_ID, f"data/{table}.csv", repo_type="dataset")
        df = read_table_csv(Path(local))
        df.to_parquet(cache_dir / f"{table}.parquet", index=False)
        out[table] = df
        log.info("downloaded %s (%d rows)", table, len(df))
    return out


def load_cached_tables(cache_dir: Path, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        path = cache_dir / f"{table}.parquet"
        if not path.exists():
            raise CacheMissingError(f"{path} not found; run `f1pred data update`")
        out[table] = pd.read_parquet(path)
    return out


def load_tables(cache_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Spec 3.4: download when asked or when the cache is incomplete; otherwise use the cache.

    If the download fails and a cache exists, warn and use the cache.
    If the download fails and there is no cache, raise DataUnavailableError.
    """
    cache_dir = Path(cache_dir)
    have_cache = all((cache_dir / f"{t}.parquet").exists() for t in TABLES)
    if have_cache and not refresh:
        return load_cached_tables(cache_dir)
    try:
        return download_tables(cache_dir)
    except Exception as exc:  # network, HTTP, disk
        if have_cache:
            log.warning("Hugging Face download failed (%s); using local cache", exc)
            return load_cached_tables(cache_dir)
        raise DataUnavailableError(
            "Could not reach Hugging Face and no local cache exists. "
            "Run `f1pred data update` once you are online."
        ) from exc
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/test_hub.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add f1pred/data/hub.py tests/conftest.py tests/test_hub.py
git commit -m "feat: Hub download with Parquet cache and offline fallback"
```

---

### Task 3: Driver-race table

**Files:**
- Create: `f1pred/data/frame.py`, `tests/test_frame.py`
- Modify: `tests/conftest.py` (add `driver_race` fixture)

**Interfaces:**
- Consumes: raw tables dict from Task 2.
- Produces:
  - `classify_dnf(status: str, classified: bool) -> str | None` returning `"accident"`, `"mechanical"`, `"other"`, or `None`
  - `build_driver_race_table(raw: dict[str, pd.DataFrame], start_season: int = 2010) -> pd.DataFrame` with columns exactly: `season:int, round:int, race_id:int, date:datetime64, circuit_id:str, race_name:str, driver_id:str, driver_code:str, driver_name:str, constructor_id:str, constructor_name:str, grid:int, position:Int64 (nullable), status:str, dnf:bool, dnf_kind:object, points:float, is_sprint:bool`, sorted by `date`, then `race_id`, then `grid`. Sprint rows carry `date = sprint_date`.
  - `DRIVER_RACE_COLUMNS` tuple of those names.

- [ ] **Step 1: Add fixture to `tests/conftest.py`**

```python
from f1pred.data.frame import build_driver_race_table


@pytest.fixture(scope="session")
def driver_race(raw_sample) -> pd.DataFrame:
    return build_driver_race_table(raw_sample, start_season=2010)
```
(add `import pandas as pd` at the top).

- [ ] **Step 2: Write the failing tests `tests/test_frame.py`**

```python
import pandas as pd
import pytest

from f1pred.data.frame import DRIVER_RACE_COLUMNS, classify_dnf


@pytest.mark.parametrize(
    "status,classified,expected",
    [
        ("Finished", True, None),
        ("+1 Lap", True, None),
        ("Accident", False, "accident"),
        ("Collision", False, "accident"),
        ("Spun off", False, "accident"),
        ("Collision damage", False, "accident"),
        ("Engine", False, "mechanical"),
        ("Gearbox", False, "mechanical"),
        ("Puncture", False, "mechanical"),
        ("Power Unit", False, "mechanical"),
        ("Disqualified", False, "other"),
        ("Retired", False, "other"),
        ("Withdrew", False, "other"),
        ("Illness", False, "other"),
    ],
)
def test_classify_dnf(status, classified, expected):
    assert classify_dnf(status, classified) == expected


def test_table_columns_and_types(driver_race):
    assert list(driver_race.columns) == list(DRIVER_RACE_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(driver_race["date"])
    assert driver_race["position"].dtype == "Int64"
    assert driver_race["dnf"].dtype == bool
    assert driver_race["is_sprint"].dtype == bool


def test_table_covers_sample_seasons(driver_race):
    races = driver_race[~driver_race.is_sprint]
    assert set(races.season.unique()) == {2023, 2024}
    assert races.groupby("season").race_id.nunique().to_dict() == {2023: 22, 2024: 24}
    sprints = driver_race[driver_race.is_sprint]
    assert sprints.groupby("season").race_id.nunique().to_dict() == {2023: 6, 2024: 6}


def test_ids_are_refs_not_numbers(driver_race):
    assert "verstappen" in set(driver_race.driver_id)
    assert "red_bull" in set(driver_race.constructor_id)
    assert "monaco" in set(driver_race.circuit_id)


def test_known_result_2024_monaco(driver_race):
    row = driver_race[
        (driver_race.season == 2024)
        & (driver_race.circuit_id == "monaco")
        & (~driver_race.is_sprint)
        & (driver_race.driver_id == "leclerc")
    ].iloc[0]
    assert row.position == 1 and row.grid == 1 and row.points == 25 and not row.dnf


def test_pit_lane_start_mapped_to_last(driver_race):
    assert (driver_race.grid >= 1).all()


def test_dnf_flag_matches_kind(driver_race):
    assert (driver_race.dnf == driver_race.dnf_kind.isin(["accident", "mechanical"])).all()
    assert driver_race.loc[driver_race.position.notna(), "dnf_kind"].isna().all()


def test_sprint_rows_use_sprint_date(driver_race):
    china_2024 = driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "shanghai")]
    sprint_date = china_2024[china_2024.is_sprint].date.iloc[0]
    race_date = china_2024[~china_2024.is_sprint].date.iloc[0]
    assert sprint_date < race_date


def test_sorted_by_date(driver_race):
    assert driver_race.date.is_monotonic_increasing


def test_start_season_filter(raw_sample):
    from f1pred.data.frame import build_driver_race_table

    only_2024 = build_driver_race_table(raw_sample, start_season=2024)
    assert set(only_2024.season) == {2024}
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_frame.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Write `f1pred/data/frame.py`**

```python
"""Build the one-row-per-driver-per-session table that every downstream module consumes."""

from __future__ import annotations

import pandas as pd

DRIVER_RACE_COLUMNS = (
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
)

ACCIDENT_STATUSES = {"Accident", "Collision", "Spun off", "Collision damage", "Damage"}
OTHER_STATUSES = {
    "Disqualified",
    "Retired",
    "Withdrew",
    "Did not qualify",
    "Did not prequalify",
    "Not classified",
    "Excluded",
    "107% Rule",
    "Injured",
    "Injury",
    "Illness",
    "Fatal accident",
    "Safety concerns",
    "Not restarted",
    "Driver unwell",
    "Eye injury",
    "Physical",
    "Safety",
    "Finished",
}


def classify_dnf(status: str, classified: bool) -> str | None:
    """Spec 3.5: accident, mechanical, other, or None for a classified finisher."""
    if classified:
        return None
    if status in ACCIDENT_STATUSES:
        return "accident"
    if status in OTHER_STATUSES or status.startswith("+"):
        return "other"
    return "mechanical"


def _session_rows(
    results: pd.DataFrame,
    races: pd.DataFrame,
    lookups: dict[str, pd.DataFrame],
    date_col: str,
    is_sprint: bool,
) -> pd.DataFrame:
    df = results.merge(races, on="raceId", how="inner")
    df = df.merge(lookups["drivers"], on="driverId", how="left")
    df = df.merge(lookups["constructors"], on="constructorId", how="left")
    df = df.merge(lookups["circuits"], on="circuitId", how="left")
    df = df.merge(lookups["status"], on="statusId", how="left")
    df = df[df[date_col].notna()].copy()

    n_entrants = df.groupby("raceId")["driverId"].transform("count")
    grid = df["grid"].fillna(0).astype(int)
    df["grid_fixed"] = grid.where(grid >= 1, n_entrants)

    position = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    classified = position.notna()
    dnf_kind = [
        classify_dnf(s, c) for s, c in zip(df["status"].fillna(""), classified, strict=True)
    ]

    out = pd.DataFrame(
        {
            "season": df["year"].astype(int),
            "round": df["round"].astype(int),
            "race_id": df["raceId"].astype(int),
            "date": pd.to_datetime(df[date_col]),
            "circuit_id": df["circuitRef"].astype(str),
            "race_name": df["name_race"].astype(str),
            "driver_id": df["driverRef"].astype(str),
            "driver_code": df["code"].fillna("").astype(str),
            "driver_name": (df["forename"] + " " + df["surname"]).astype(str),
            "constructor_id": df["constructorRef"].astype(str),
            "constructor_name": df["name_constructor"].astype(str),
            "grid": df["grid_fixed"].astype(int),
            "position": position,
            "status": df["status"].fillna("").astype(str),
            "dnf_kind": pd.Series(dnf_kind, index=df.index, dtype=object),
            "points": df["points"].fillna(0).astype(float),
            "is_sprint": is_sprint,
        }
    )
    out["dnf"] = out["dnf_kind"].isin(["accident", "mechanical"])
    return out[list(DRIVER_RACE_COLUMNS)]


def build_driver_race_table(raw: dict[str, pd.DataFrame], start_season: int = 2010) -> pd.DataFrame:
    races = raw["races"].rename(columns={"name": "name_race"})
    races = races[races["year"] >= start_season][
        ["raceId", "year", "round", "circuitId", "name_race", "date", "sprint_date"]
    ]
    lookups = {
        "drivers": raw["drivers"][["driverId", "driverRef", "code", "forename", "surname"]],
        "constructors": raw["constructors"][["constructorId", "constructorRef", "name"]].rename(
            columns={"name": "name_constructor"}
        ),
        "circuits": raw["circuits"][["circuitId", "circuitRef"]],
        "status": raw["status"][["statusId", "status"]],
    }
    race_rows = _session_rows(raw["results"], races, lookups, "date", is_sprint=False)
    sprint_rows = _session_rows(raw["sprint_results"], races, lookups, "sprint_date", is_sprint=True)
    table = pd.concat([race_rows, sprint_rows], ignore_index=True)
    table = table.sort_values(["date", "race_id", "grid"], kind="stable").reset_index(drop=True)
    return table
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/test_frame.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. If `test_known_result_2024_monaco` fails on `points`, check that `points` in the sample is 25.0 for Leclerc; the test expects 25.

- [ ] **Step 6: Commit**

```bash
git add f1pred/data/frame.py tests/conftest.py tests/test_frame.py
git commit -m "feat: driver-race table with DNF classification"
```

---

### Task 4: Elo engine

**Files:**
- Create: `f1pred/ratings/elo.py`, `tests/test_elo.py`, `docs/elo.md`

**Interfaces:**
- Consumes: `ModelParams` from Task 1.
- Produces:
  - `expected_score(strength_a: float, strength_b: float) -> float`
  - `Finisher = tuple[str, str, int]` (driver_id, constructor_id, position)
  - `@dataclass RaceUpdate(driver_delta: dict[str, float], constructor_delta: dict[str, float])`
  - `race_update(finishers: list[Finisher], driver_ratings: Mapping[str, float], constructor_ratings: Mapping[str, float], params: ModelParams, k_scale: float = 1.0) -> RaceUpdate`
  - `apply_update(ratings: dict[str, float], delta: Mapping[str, float]) -> None`
  - `regress(ratings: dict[str, float], fraction: float, initial: float) -> None`

- [ ] **Step 1: Write the failing tests `tests/test_elo.py`**

```python
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.elo import apply_update, expected_score, race_update, regress

P = DEFAULT_PARAMS  # k_driver 24, k_constructor 32, teammate_weight 2


def test_expected_score_symmetry_and_scale():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) == pytest.approx(1 / (1 + 10 ** (-0.5)))
    assert expected_score(1500, 1700) + expected_score(1700, 1500) == pytest.approx(1.0)


def test_two_drivers_different_teams_hand_computed():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1500.0, "y": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "y", 2)], d, c, P)
    # expected 0.5, surprise 0.5, n=2 so scale 1
    assert upd.driver_delta == pytest.approx({"a": 12.0, "b": -12.0})
    assert upd.constructor_delta == pytest.approx({"x": 16.0, "y": -16.0})


def test_teammates_weighted_and_no_constructor_change():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "x", 2)], d, c, P)
    assert upd.driver_delta == pytest.approx({"a": 24.0, "b": -24.0})
    assert upd.constructor_delta.get("x", 0.0) == pytest.approx(0.0)


def test_scaled_by_field_size_and_zero_sum():
    d = {f"d{i}": 1500.0 for i in range(20)}
    c = {f"c{i // 2}": 1500.0 for i in range(20)}
    finishers = [(f"d{i}", f"c{i // 2}", i + 1) for i in range(20)]
    upd = race_update(finishers, d, c, P)
    # winner beats 19 drivers at surprise 0.5, one of them a teammate (weight 2): (18 + 2) * 12 / 19
    assert upd.driver_delta["d0"] == pytest.approx(20 * 12 / 19)
    assert sum(upd.driver_delta.values()) == pytest.approx(0.0)
    assert sum(upd.constructor_delta.values()) == pytest.approx(0.0)


def test_strength_uses_driver_plus_constructor():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1700.0, "y": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "y", 2)], d, c, P)
    surprise = 1 - expected_score(3200, 3000)
    assert upd.driver_delta["a"] == pytest.approx(24 * surprise)


def test_k_scale_for_sprints():
    d = {"a": 1500.0, "b": 1500.0}
    c = {"x": 1500.0, "y": 1500.0}
    upd = race_update([("a", "x", 1), ("b", "y", 2)], d, c, P, k_scale=0.5)
    assert upd.driver_delta["a"] == pytest.approx(6.0)


def test_single_finisher_no_change():
    upd = race_update([("a", "x", 1)], {"a": 1500.0}, {"x": 1500.0}, P)
    assert upd.driver_delta == {} and upd.constructor_delta == {}


def test_apply_and_regress():
    r = {"a": 1600.0, "b": 1400.0}
    apply_update(r, {"a": 10.0})
    assert r == {"a": 1610.0, "b": 1400.0}
    regress(r, 0.5, 1500.0)
    assert r == pytest.approx({"a": 1555.0, "b": 1450.0})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_elo.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/ratings/elo.py`**

```python
"""Pairwise Elo for drivers and constructors. Spec section 5."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from f1pred.config import ModelParams

Finisher = tuple[str, str, int]  # driver_id, constructor_id, classified position


@dataclass
class RaceUpdate:
    driver_delta: dict[str, float] = field(default_factory=dict)
    constructor_delta: dict[str, float] = field(default_factory=dict)


def expected_score(strength_a: float, strength_b: float) -> float:
    """Probability that A finishes ahead of B."""
    return 1.0 / (1.0 + 10 ** ((strength_b - strength_a) / 400.0))


def race_update(
    finishers: list[Finisher],
    driver_ratings: Mapping[str, float],
    constructor_ratings: Mapping[str, float],
    params: ModelParams,
    k_scale: float = 1.0,
) -> RaceUpdate:
    """Rating changes from one session. Every pair of classified finishers is one matchup.

    The driver ahead scores 1, the driver behind 0. Each side moves by K * (actual - expected).
    Teammate pairs are weighted by teammate_weight for drivers and skipped for constructors.
    Deltas are scaled by 1 / (n - 1) so field size does not change the rating speed.
    """
    ordered = sorted(finishers, key=lambda f: f[2])
    n = len(ordered)
    if n < 2:
        return RaceUpdate()
    scale = 1.0 / (n - 1)
    dd: dict[str, float] = defaultdict(float)
    dc: dict[str, float] = defaultdict(float)
    for i in range(n):
        di, ci, _ = ordered[i]
        s_i = driver_ratings[di] + constructor_ratings[ci]
        for j in range(i + 1, n):
            dj, cj, _ = ordered[j]
            s_j = driver_ratings[dj] + constructor_ratings[cj]
            surprise = 1.0 - expected_score(s_i, s_j)
            same_team = ci == cj
            w = params.teammate_weight if same_team else 1.0
            step_d = params.k_driver * k_scale * w * surprise * scale
            dd[di] += step_d
            dd[dj] -= step_d
            if not same_team:
                step_c = params.k_constructor * k_scale * surprise * scale
                dc[ci] += step_c
                dc[cj] -= step_c
    return RaceUpdate(dict(dd), dict(dc))


def apply_update(ratings: dict[str, float], delta: Mapping[str, float]) -> None:
    for key, change in delta.items():
        ratings[key] = ratings.get(key, 0.0) + change


def regress(ratings: dict[str, float], fraction: float, initial: float) -> None:
    """Pull every rating toward `initial` by `fraction` of its distance."""
    for key, value in ratings.items():
        ratings[key] = initial + (value - initial) * (1.0 - fraction)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_elo.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 8 passed.

- [ ] **Step 5: Write `docs/elo.md`** (plain English, about 250 words)

Cover: the chess origin; "expected score" as a probability from the rating gap with 400 points meaning roughly 10 to 1; why driver + constructor is the race strength; why teammate matchups count double; why DNFs are ignored here; why the sum of changes is zero; why season regression pulls ratings back and pulls constructors harder in rule-change years. End with the two formulas from spec 5.1 and 5.3.

- [ ] **Step 6: Commit**

```bash
git add f1pred/ratings/elo.py tests/test_elo.py docs/elo.md
git commit -m "feat: pairwise Elo update with teammate weighting"
```

---

### Task 5: Ratings history replay

**Files:**
- Create: `f1pred/ratings/history.py`, `tests/test_history.py`

**Interfaces:**
- Consumes: driver-race table (Task 3), `race_update`, `apply_update`, `regress` (Task 4).
- Produces:
  - `@dataclass RatingState(driver: dict[str, float], constructor: dict[str, float], driver_races: dict[str, int], season: int | None)` with `.copy()`, `.to_json(path)`, `RatingState.from_json(path)`
  - `start_season(state: RatingState, season: int, params: ModelParams) -> None` (applies regression when the season changes)
  - `replay(table: pd.DataFrame, params: ModelParams) -> tuple[pd.DataFrame, RatingState]` where the history DataFrame has columns `race_id, is_sprint, season, round, date, driver_id, constructor_id, driver_rating_pre, constructor_rating_pre, driver_races_pre`
  - `state_for_season(state: RatingState, season: int, params: ModelParams) -> RatingState` (copy with regression applied if `season` is a new season)
  - `strength(state: RatingState, driver_id: str, constructor_id: str, params: ModelParams) -> float`

- [ ] **Step 1: Write the failing tests `tests/test_history.py`**

```python
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
    a_after = 1500 + 12 * 0.9 + 12 * (1 - _e(state, "a", "b"))
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
    assert top in {"verstappen", "norris", "leclerc", "piastri", "hamilton", "russell", "sainz"}
    assert max(state.constructor, key=state.constructor.get) in {"red_bull", "mclaren", "ferrari"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_history.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/ratings/history.py`**

```python
"""Replay every session in date order and record the rating each entrant had going in."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from f1pred.config import ModelParams
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
]


@dataclass
class RatingState:
    driver: dict[str, float] = field(default_factory=dict)
    constructor: dict[str, float] = field(default_factory=dict)
    driver_races: dict[str, int] = field(default_factory=dict)
    season: int | None = None

    def copy(self) -> RatingState:
        return copy.deepcopy(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> RatingState:
        data = json.loads(Path(path).read_text())
        return cls(**data)


def start_season(state: RatingState, season: int, params: ModelParams) -> None:
    """Spec 5.3: when a new season begins, regress ratings toward the initial value."""
    if state.season is not None and season > state.season:
        regress(state.driver, params.regress_driver, params.initial_rating)
        fraction = (
            params.regress_constructor_regulation
            if season in params.regulation_seasons
            else params.regress_constructor
        )
        regress(state.constructor, fraction, params.initial_rating)
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


def replay(table: pd.DataFrame, params: ModelParams) -> tuple[pd.DataFrame, RatingState]:
    """Walk the table in date order. Returns (history, final state).

    history has one row per table row with the ratings *before* that session.
    """
    state = RatingState()
    rows: list[dict] = []
    ordered = table.sort_values(["date", "race_id", "is_sprint"], kind="stable")
    for (date, race_id, is_sprint), grp in ordered.groupby(
        ["date", "race_id", "is_sprint"], sort=True
    ):
        season = int(grp["season"].iloc[0])
        start_season(state, season, params)
        for r in grp.itertuples(index=False):
            state.driver.setdefault(r.driver_id, params.initial_rating)
            state.constructor.setdefault(r.constructor_id, params.initial_rating)
            rows.append(
                {
                    "race_id": int(race_id),
                    "is_sprint": bool(is_sprint),
                    "season": season,
                    "round": int(r.round),
                    "date": date,
                    "driver_id": r.driver_id,
                    "constructor_id": r.constructor_id,
                    "driver_rating_pre": state.driver[r.driver_id],
                    "constructor_rating_pre": state.constructor[r.constructor_id],
                    "driver_races_pre": state.driver_races.get(r.driver_id, 0),
                }
            )
        finishers = [
            (r.driver_id, r.constructor_id, int(r.position))
            for r in grp.itertuples(index=False)
            if pd.notna(r.position)
        ]
        k_scale = params.sprint_weight if is_sprint else 1.0
        upd = race_update(finishers, state.driver, state.constructor, params, k_scale=k_scale)
        apply_update(state.driver, upd.driver_delta)
        apply_update(state.constructor, upd.constructor_delta)
        for driver_id, _, _ in finishers:
            state.driver_races[driver_id] = state.driver_races.get(driver_id, 0) + 1
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return history, state
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_history.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: all pass. Hand-check for `test_final_state_and_state_for_new_season`: after race 1, a=1512, b=1488, x=1516, y=1484. Race 2 has one finisher, no update. Season 2021 regression: a=1510.8, b=1489.2, x=1509.6, y=1490.4. Race 3: surprise = 1 - E(3020.4, 2979.6).

- [ ] **Step 5: Commit**

```bash
git add f1pred/ratings/history.py tests/test_history.py
git commit -m "feat: ratings history replay with pre-race ratings"
```

---

### Task 6: DNF probability

**Files:**
- Create: `f1pred/sim/dnf.py`, `tests/test_dnf.py`, `docs/dnf-model.md`

**Interfaces:**
- Consumes: driver-race table (Task 3), `ModelParams`.
- Produces:
  - `dnf_probability(table: pd.DataFrame, driver_id: str, constructor_id: str, circuit_id: str, before_date: pd.Timestamp, params: ModelParams) -> float` in `[0.01, 0.9]`
  - `dnf_cache_for_races(table: pd.DataFrame, race_ids: Iterable[int], params: ModelParams) -> dict[tuple[int, str], float]` (pre-race p_dnf for every driver in every listed race; keyed by `(race_id, driver_id)`)

- [ ] **Step 1: Write the failing tests `tests/test_dnf.py`**

```python
import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.sim.dnf import dnf_cache_for_races, dnf_probability

P = DEFAULT_PARAMS


def _table(rows):
    df = pd.DataFrame(
        rows,
        columns=["date", "race_id", "driver_id", "constructor_id", "circuit_id", "dnf", "is_sprint"],
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
                (f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}", i, d, f"t{d}", circuit, dnf_for(i, d), False)
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
    race_ids = driver_race[(driver_race.season == 2024) & (~driver_race.is_sprint)].race_id.unique()[:3]
    cache = dnf_cache_for_races(driver_race, race_ids, P)
    row = driver_race[(driver_race.race_id == race_ids[2]) & (~driver_race.is_sprint)].iloc[0]
    direct = dnf_probability(
        driver_race, row.driver_id, row.constructor_id, row.circuit_id, row.date, P
    )
    assert cache[(int(race_ids[2]), row.driver_id)] == pytest.approx(direct)
    assert 0.01 <= direct <= 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dnf.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/sim/dnf.py`**

```python
"""Per-driver DNF probability from recent driver, constructor and circuit history. Spec 6.3."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from f1pred.config import ModelParams

DEFAULT_GLOBAL_RATE = 0.1
P_MIN, P_MAX = 0.01, 0.9


def _rate(flags: pd.Series, window: int) -> tuple[float, int]:
    recent = flags.tail(window)
    if len(recent) == 0:
        return 0.0, 0
    return float(recent.mean()), int(len(recent))


def _global_rate(past: pd.DataFrame, params: ModelParams) -> float:
    recent = past.tail(params.dnf_window * 20)
    return float(recent["dnf"].mean()) if len(recent) else DEFAULT_GLOBAL_RATE


def _circuit_factor(past: pd.DataFrame, circuit_id: str, global_rate: float, params: ModelParams) -> float:
    circ = past[past["circuit_id"] == circuit_id]
    if len(circ) < params.circuit_min_rows or global_rate <= 0:
        return 1.0
    return float(np.clip(circ["dnf"].mean() / global_rate, params.circuit_factor_min, params.circuit_factor_max))


def _probability(
    past: pd.DataFrame, driver_id: str, constructor_id: str, circuit_id: str, params: ModelParams
) -> float:
    global_rate = _global_rate(past, params)
    d_rate, n_d = _rate(past.loc[past["driver_id"] == driver_id, "dnf"], params.dnf_window)
    c_rate, n_c = _rate(
        past.loc[past["constructor_id"] == constructor_id, "dnf"], params.dnf_window * 2
    )
    if n_d and n_c:
        raw, n = (d_rate + c_rate) / 2, (n_d + n_c) / 2
    elif n_d:
        raw, n = d_rate, n_d
    elif n_c:
        raw, n = c_rate, n_c
    else:
        raw, n = global_rate, 0
    shrunk = global_rate + (raw - global_rate) * n / (n + params.shrink_dnf)
    p = shrunk * _circuit_factor(past, circuit_id, global_rate, params)
    return float(np.clip(p, P_MIN, P_MAX))


def dnf_probability(
    table: pd.DataFrame,
    driver_id: str,
    constructor_id: str,
    circuit_id: str,
    before_date: pd.Timestamp,
    params: ModelParams,
) -> float:
    past = table[(~table["is_sprint"]) & (table["date"] < before_date)]
    return _probability(past, driver_id, constructor_id, circuit_id, params)


def dnf_cache_for_races(
    table: pd.DataFrame, race_ids: Iterable[int], params: ModelParams
) -> dict[tuple[int, str], float]:
    """Pre-race p_dnf for every entrant of every race in `race_ids`, keyed (race_id, driver_id)."""
    races = table[~table["is_sprint"]]
    out: dict[tuple[int, str], float] = {}
    for race_id in race_ids:
        rows = races[races["race_id"] == race_id]
        if rows.empty:
            continue
        date = rows["date"].iloc[0]
        past = races[races["date"] < date]
        for r in rows.itertuples(index=False):
            out[(int(race_id), r.driver_id)] = _probability(
                past, r.driver_id, r.constructor_id, r.circuit_id, params
            )
    return out
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_dnf.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 7 passed. In `test_circuit_factor_clipped` the global rate over the last 400 rows is 0.5 (40 safe rows, 40 deadly rows) and the `deadly` circuit has 40 rows, at the `circuit_min_rows` threshold.

- [ ] **Step 5: Write `docs/dnf-model.md`** (about 200 words): why reliability is separate from pace, what "shrinking toward the average" means with the `n / (n + 10)` picture, why the circuit multiplier is clipped, and where the numbers come from.

- [ ] **Step 6: Commit**

```bash
git add f1pred/sim/dnf.py tests/test_dnf.py docs/dnf-model.md
git commit -m "feat: DNF probability with shrinkage and circuit factor"
```

---

### Task 7: Race Monte Carlo

**Files:**
- Create: `f1pred/sim/race.py`, `tests/test_race_sim.py`, `docs/monte-carlo.md`

**Interfaces:**
- Consumes: `ModelParams`.
- Produces:
  - `@dataclass(frozen=True) Entrant(driver_id: str, constructor_id: str, strength: float, grid: int | None, p_dnf: float, low_confidence: bool = False)`
  - `@dataclass RaceForecast(driver_ids: list[str], p_win: np.ndarray, p_podium: np.ndarray, p_points: np.ndarray, expected_position: np.ndarray, position_p10: np.ndarray, position_p90: np.ndarray, position_matrix: np.ndarray, n_runs: int, used_grid: bool)` with `.as_frame() -> pd.DataFrame` (columns `driver_id, p_win, p_podium, p_points, expected_position, position_p10, position_p90`, sorted by `p_win` desc) and `.prob(driver_id, kind) -> float`
  - `simulate_race(entrants: list[Entrant], params: ModelParams, n_runs: int = 10_000, seed: int | None = None, use_grid: bool = True) -> RaceForecast`

- [ ] **Step 1: Write the failing tests `tests/test_race_sim.py`**

```python
import numpy as np
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.sim.race import Entrant, simulate_race

P = DEFAULT_PARAMS


def _field(n=20, strength=3000.0, p_dnf=0.1):
    return [
        Entrant(f"d{i}", f"t{i // 2}", strength, i + 1, p_dnf) for i in range(n)
    ]


def test_position_columns_sum_to_one():
    f = simulate_race(_field(), P, n_runs=2000, seed=1)
    assert f.position_matrix.shape == (20, 20)
    np.testing.assert_allclose(f.position_matrix.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(f.position_matrix.sum(axis=1), 1.0, atol=1e-12)
    assert f.p_win.sum() == pytest.approx(1.0)
    assert f.p_podium.sum() == pytest.approx(3.0)
    assert f.p_points.sum() == pytest.approx(10.0)


def test_stronger_driver_wins_more():
    field = _field(strength=3000.0)
    field[5] = Entrant("d5", "t2", 3400.0, 6, 0.1)
    f = simulate_race(field, P, n_runs=5000, seed=2, use_grid=False)
    assert f.driver_ids[int(np.argmax(f.p_win))] == "d5"
    assert f.p_win[5] > 0.5


def test_certain_dnf_never_wins_and_finishes_last_region():
    field = _field(p_dnf=0.0)
    field[0] = Entrant("d0", "t0", 5000.0, 1, 1.0)
    f = simulate_race(field, P, n_runs=1000, seed=3)
    assert f.p_win[0] == 0.0
    assert f.expected_position[0] == pytest.approx(20.0)


def test_identical_drivers_equal_within_tolerance():
    f = simulate_race(_field(), P, n_runs=20000, seed=4, use_grid=False)
    assert f.p_win.max() - f.p_win.min() < 0.03


def test_grid_bonus_helps_pole():
    f = simulate_race(_field(), P, n_runs=5000, seed=5, use_grid=True)
    assert f.p_win[0] > f.p_win[19]
    assert f.used_grid


def test_missing_grid_disables_grid_mode():
    field = _field()
    field[3] = Entrant("d3", "t1", 3000.0, None, 0.1)
    f = simulate_race(field, P, n_runs=500, seed=6)
    assert not f.used_grid


def test_seed_reproducible():
    a = simulate_race(_field(), P, n_runs=500, seed=7)
    b = simulate_race(_field(), P, n_runs=500, seed=7)
    np.testing.assert_array_equal(a.position_matrix, b.position_matrix)


def test_teammates_share_noise():
    # With only team noise, teammates must finish adjacent in every run.
    params = P.replace(sigma_driver=0.0, grid_bonus=0.0)
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.0) for i in range(6)]
    f = simulate_race(field, params, n_runs=200, seed=8, use_grid=False)
    # d0 and d1 same team: P(d0 in pos k) equals P(d1 in pos k) roughly and podium share matches
    assert f.p_podium[0] + f.p_podium[1] == pytest.approx(f.p_podium[2] + f.p_podium[3], abs=0.35)


def test_as_frame_sorted_and_prob():
    f = simulate_race(_field(), P, n_runs=300, seed=9)
    df = f.as_frame()
    assert list(df.columns) == [
        "driver_id",
        "p_win",
        "p_podium",
        "p_points",
        "expected_position",
        "position_p10",
        "position_p90",
    ]
    assert df.p_win.is_monotonic_decreasing
    assert f.prob("d0", "win") == pytest.approx(f.p_win[0])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_race_sim.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/sim/race.py`**

```python
"""Vectorised Monte Carlo of one race. Spec section 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.config import ModelParams

PODIUM = 3
POINTS_POSITIONS = 10
DNF_KEY = -1e9  # anything below every real performance


@dataclass(frozen=True)
class Entrant:
    driver_id: str
    constructor_id: str
    strength: float
    grid: int | None
    p_dnf: float
    low_confidence: bool = False


@dataclass
class RaceForecast:
    driver_ids: list[str]
    p_win: np.ndarray
    p_podium: np.ndarray
    p_points: np.ndarray
    expected_position: np.ndarray
    position_p10: np.ndarray
    position_p90: np.ndarray
    position_matrix: np.ndarray  # (n_entrants, n_positions), row i = P(driver i finishes k+1)
    n_runs: int
    used_grid: bool

    def as_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "driver_id": self.driver_ids,
                "p_win": self.p_win,
                "p_podium": self.p_podium,
                "p_points": self.p_points,
                "expected_position": self.expected_position,
                "position_p10": self.position_p10,
                "position_p90": self.position_p90,
            }
        )
        return df.sort_values("p_win", ascending=False, kind="stable").reset_index(drop=True)

    def prob(self, driver_id: str, kind: str) -> float:
        i = self.driver_ids.index(driver_id)
        return float({"win": self.p_win, "podium": self.p_podium, "points": self.p_points}[kind][i])


def simulate_race(
    entrants: list[Entrant],
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int | None = None,
    use_grid: bool = True,
) -> RaceForecast:
    rng = np.random.default_rng(seed)
    n = len(entrants)
    strength = np.array([e.strength for e in entrants], dtype=float)
    p_dnf = np.array([e.p_dnf for e in entrants], dtype=float)
    used_grid = use_grid and all(e.grid is not None for e in entrants)

    teams = sorted({e.constructor_id for e in entrants})
    team_idx = np.array([teams.index(e.constructor_id) for e in entrants])
    team_noise = rng.normal(0.0, params.sigma_team, size=(n_runs, len(teams)))[:, team_idx]

    if used_grid:
        grid = np.array([e.grid for e in entrants], dtype=float)
        grid_term = params.grid_bonus * (n - grid)
        sigma_driver = params.sigma_driver
    else:
        grid_term = np.zeros(n)
        # A uniform draw over n grid slots has std n / sqrt(12); fold that into driver noise.
        sigma_driver = float(np.sqrt(params.sigma_driver**2 + (params.grid_bonus * n) ** 2 / 12))
    driver_noise = rng.normal(0.0, sigma_driver, size=(n_runs, n))

    performance = strength + grid_term + team_noise + driver_noise
    dnf = rng.random((n_runs, n)) < p_dnf
    key = np.where(dnf, DNF_KEY + rng.random((n_runs, n)), performance)

    order = np.argsort(-key, axis=1, kind="stable")  # order[r, k] = entrant index at position k+1
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.arange(1, n + 1)[None, :].repeat(n_runs, 0), axis=1)

    position_matrix = np.stack(
        [np.bincount(positions[:, i] - 1, minlength=n) for i in range(n)]
    ).astype(float) / n_runs
    ranks = np.arange(1, n + 1)
    p10, p90 = np.percentile(positions, [10, 90], axis=0)
    return RaceForecast(
        driver_ids=[e.driver_id for e in entrants],
        p_win=position_matrix[:, 0].copy(),
        p_podium=position_matrix[:, :PODIUM].sum(axis=1),
        p_points=position_matrix[:, :POINTS_POSITIONS].sum(axis=1),
        expected_position=(position_matrix * ranks).sum(axis=1),
        position_p10=p10,
        position_p90=p90,
        position_matrix=position_matrix,
        n_runs=n_runs,
        used_grid=used_grid,
    )
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_race_sim.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 9 passed. In `test_certain_dnf_never_wins_and_finishes_last_region` every other entrant has `p_dnf=0.0`, so d0 is the only DNF and always finishes 20th.

- [ ] **Step 5: Write `docs/monte-carlo.md`** (about 300 words): what one "run" is, the performance formula in words, why team noise is shared, why DNFs go to the back in random order, how counting runs turns into probabilities, why 10,000 runs, and how `--no-grid` folds the unknown grid into noise.

- [ ] **Step 6: Commit**

```bash
git add f1pred/sim/race.py tests/test_race_sim.py docs/monte-carlo.md
git commit -m "feat: vectorised race Monte Carlo with team noise and DNF layer"
```

---

### Task 8: Scoring and baselines

**Files:**
- Create: `f1pred/backtest/scoring.py`, `tests/test_scoring.py`

**Interfaces:**
- Produces:
  - `winner_log_loss(p_win: Mapping[str, float], winner: str) -> float`
  - `podium_brier(p_podium: Mapping[str, float], podium: set[str]) -> float`
  - `position_spearman(expected: Mapping[str, float], actual: Mapping[str, int]) -> float` (nan if fewer than 3 common drivers)
  - `@dataclass Baseline(p_win: dict[str, float], p_podium: dict[str, float], expected_position: dict[str, float])`
  - `pole_baseline(grid: Mapping[str, int]) -> Baseline`
  - `uniform_baseline(driver_ids: Iterable[str]) -> Baseline`

- [ ] **Step 1: Write the failing tests `tests/test_scoring.py`**

```python
import math

import pytest

from f1pred.backtest.scoring import (
    podium_brier,
    pole_baseline,
    position_spearman,
    uniform_baseline,
    winner_log_loss,
)


def test_winner_log_loss():
    assert winner_log_loss({"a": 0.5, "b": 0.5}, "a") == pytest.approx(math.log(2))
    assert winner_log_loss({"a": 1.0}, "a") == pytest.approx(0.0)
    assert winner_log_loss({"a": 1.0}, "zzz") == pytest.approx(-math.log(1e-6))


def test_podium_brier():
    p = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
    assert podium_brier(p, {"a", "b", "c"}) == 0.0
    assert podium_brier({"a": 0.5, "b": 0.5}, {"a"}) == pytest.approx(0.25)


def test_position_spearman():
    assert position_spearman({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}) == pytest.approx(1.0)
    assert position_spearman({"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 2, "c": 1}) == pytest.approx(-1.0)
    assert math.isnan(position_spearman({"a": 1, "b": 2}, {"a": 1, "b": 2}))


def test_pole_baseline():
    b = pole_baseline({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    assert b.p_win["a"] == pytest.approx(0.9)
    assert b.p_win["e"] == pytest.approx(0.025)
    assert sum(b.p_win.values()) == pytest.approx(1.0)
    assert b.p_podium["b"] == pytest.approx(0.9)
    assert b.p_podium["d"] == pytest.approx(0.15)
    assert sum(b.p_podium.values()) == pytest.approx(3.0)
    assert b.expected_position == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


def test_uniform_baseline():
    b = uniform_baseline(["a", "b", "c", "d"])
    assert b.p_win == pytest.approx({k: 0.25 for k in "abcd"})
    assert b.p_podium == pytest.approx({k: 0.75 for k in "abcd"})
    assert len(set(b.expected_position.values())) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/backtest/scoring.py`**

```python
"""Probability scores and the two dumb baselines. Spec section 8."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

EPS = 1e-6
POLE_MASS = 0.9
PODIUM = 3


def winner_log_loss(p_win: Mapping[str, float], winner: str) -> float:
    p = min(max(p_win.get(winner, 0.0), EPS), 1.0)
    return -math.log(p)


def podium_brier(p_podium: Mapping[str, float], podium: set[str]) -> float:
    terms = [(p - (1.0 if d in podium else 0.0)) ** 2 for d, p in p_podium.items()]
    return sum(terms) / len(terms)


def position_spearman(expected: Mapping[str, float], actual: Mapping[str, int]) -> float:
    common = [d for d in expected if d in actual]
    if len(common) < 3:
        return math.nan
    e = pd.Series([expected[d] for d in common])
    a = pd.Series([actual[d] for d in common])
    return float(e.corr(a, method="spearman"))


@dataclass
class Baseline:
    p_win: dict[str, float]
    p_podium: dict[str, float]
    expected_position: dict[str, float]


def pole_baseline(grid: Mapping[str, int]) -> Baseline:
    """90% on the pole sitter (top three for podium), the rest spread evenly."""
    n = len(grid)
    ordered = sorted(grid, key=lambda d: grid[d])
    rest_win = (1.0 - POLE_MASS) / max(n - 1, 1)
    p_win = {d: (POLE_MASS if i == 0 else rest_win) for i, d in enumerate(ordered)}
    k = min(PODIUM, n)
    rest_podium = (PODIUM - POLE_MASS * k) / max(n - k, 1)
    p_podium = {d: (POLE_MASS if i < k else rest_podium) for i, d in enumerate(ordered)}
    return Baseline(p_win, p_podium, {d: float(grid[d]) for d in grid})


def uniform_baseline(driver_ids: Iterable[str]) -> Baseline:
    ids = list(driver_ids)
    n = len(ids)
    mid = (n + 1) / 2
    return Baseline({d: 1.0 / n for d in ids}, {d: PODIUM / n for d in ids}, {d: mid for d in ids})
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_scoring.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add f1pred/backtest/scoring.py tests/test_scoring.py
git commit -m "feat: log loss, Brier, Spearman and baselines"
```

---

### Task 9: Backtest runner

**Files:**
- Create: `f1pred/backtest/run.py`, `tests/test_backtest.py`, `docs/backtesting.md`

**Interfaces:**
- Consumes: driver-race table, history (Task 5), `dnf_cache_for_races` (Task 6), `Entrant`, `simulate_race` (Task 7), scoring (Task 8).
- Produces:
  - `entrants_for_past_race(race_rows: pd.DataFrame, history_rows: pd.DataFrame, dnf_lookup: Mapping[tuple[int, str], float], params: ModelParams) -> list[Entrant]`
  - `@dataclass BacktestResult(races: pd.DataFrame, seasons: pd.DataFrame, calibration: pd.DataFrame)`
  - `run_backtest(table, history, seasons: Iterable[int], params, n_runs: int = 10_000, seed: int = 0, dnf_cache: dict | None = None) -> BacktestResult`
  - `calibration_table(p: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame` with columns `bin_low, bin_high, predicted, observed, count`
  - `RACE_SCORE_COLUMNS` and `SEASON_SCORE_COLUMNS`

  `races` columns: `season, round, race_id, race_name, winner, winner_p, model_logloss, pole_logloss, uniform_logloss, model_brier, pole_brier, uniform_brier, model_spearman, pole_spearman`.
  `seasons` columns: `season, n_races, model_logloss, pole_logloss, uniform_logloss, model_brier, pole_brier, uniform_brier, model_spearman, pole_spearman` (means).

- [ ] **Step 1: Write the failing tests `tests/test_backtest.py`**

```python
import math

import numpy as np
import pandas as pd
import pytest

from f1pred.backtest.run import (
    RACE_SCORE_COLUMNS,
    SEASON_SCORE_COLUMNS,
    calibration_table,
    entrants_for_past_race,
    run_backtest,
)
from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import replay

P = DEFAULT_PARAMS


@pytest.fixture(scope="module")
def sample_history(driver_race):
    history, _ = replay(driver_race, P)
    return history


def test_entrants_use_pre_race_ratings(driver_race, sample_history):
    race_id = int(driver_race[(driver_race.season == 2024) & (~driver_race.is_sprint)].race_id.iloc[-1])
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    ents = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)
    assert len(ents) == len(rows)
    by_id = {e.driver_id: e for e in ents}
    h = hist.set_index("driver_id")
    for d, e in by_id.items():
        assert e.strength == pytest.approx(
            h.loc[d, "driver_rating_pre"] + h.loc[d, "constructor_rating_pre"]
        )
        assert e.grid == int(rows.set_index("driver_id").loc[d, "grid"])
        assert e.p_dnf == 0.1
        assert e.low_confidence == (h.loc[d, "driver_races_pre"] < P.min_races_for_confidence)


def test_run_backtest_shapes_and_baselines(driver_race, sample_history):
    result = run_backtest(driver_race, sample_history, [2024], P, n_runs=500, seed=1)
    assert list(result.races.columns) == RACE_SCORE_COLUMNS
    assert list(result.seasons.columns) == SEASON_SCORE_COLUMNS
    assert len(result.races) == 24
    assert result.seasons.n_races.iloc[0] == 24
    assert (result.races.uniform_logloss == pytest.approx(math.log(20), abs=0.06)).all()
    assert result.races.model_logloss.between(0, -math.log(1e-6)).all()
    assert result.calibration.columns.tolist() == ["bin_low", "bin_high", "predicted", "observed", "count"]
    assert result.calibration["count"].sum() == 24 * 20


def test_backtest_never_reads_future(driver_race, sample_history, monkeypatch):
    """Every history row used for a race must carry that race's own id (pre-race snapshot)."""
    import f1pred.backtest.run as run_mod

    seen = []
    original = run_mod.entrants_for_past_race

    def spy(rows, hist, dnf, params):
        seen.append((set(rows.race_id), set(hist.race_id)))
        return original(rows, hist, dnf, params)

    monkeypatch.setattr(run_mod, "entrants_for_past_race", spy)
    run_backtest(driver_race, sample_history, [2023], P, n_runs=50, seed=0)
    assert seen and all(r == h and len(r) == 1 for r, h in seen)


def test_calibration_table_bins():
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.95])
    y = np.array([0, 0, 1, 1, 1])
    cal = calibration_table(p, y, bins=10)
    assert cal["count"].sum() == 5
    row = cal[cal.bin_low == 0.1].iloc[0]
    assert row.predicted == pytest.approx(0.15) and row.observed == pytest.approx(0.5)
    assert cal[cal.bin_low == 0.9].iloc[0].observed == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/backtest/run.py`**

```python
"""Replay past seasons, predicting each race from pre-race data only. Spec section 8."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.backtest.scoring import (
    podium_brier,
    pole_baseline,
    position_spearman,
    uniform_baseline,
    winner_log_loss,
)
from f1pred.config import ModelParams
from f1pred.sim.dnf import dnf_cache_for_races
from f1pred.sim.race import Entrant, simulate_race

RACE_SCORE_COLUMNS = [
    "season",
    "round",
    "race_id",
    "race_name",
    "winner",
    "winner_p",
    "model_logloss",
    "pole_logloss",
    "uniform_logloss",
    "model_brier",
    "pole_brier",
    "uniform_brier",
    "model_spearman",
    "pole_spearman",
]
SEASON_SCORE_COLUMNS = ["season", "n_races"] + RACE_SCORE_COLUMNS[6:]


@dataclass
class BacktestResult:
    races: pd.DataFrame
    seasons: pd.DataFrame
    calibration: pd.DataFrame


def entrants_for_past_race(
    race_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
    dnf_lookup: Mapping[tuple[int, str], float],
    params: ModelParams,
) -> list[Entrant]:
    hist = history_rows.set_index("driver_id")
    out = []
    for r in race_rows.itertuples(index=False):
        h = hist.loc[r.driver_id]
        out.append(
            Entrant(
                driver_id=r.driver_id,
                constructor_id=r.constructor_id,
                strength=float(h.driver_rating_pre + h.constructor_rating_pre),
                grid=int(r.grid),
                p_dnf=float(dnf_lookup[(int(r.race_id), r.driver_id)]),
                low_confidence=bool(h.driver_races_pre < params.min_races_for_confidence),
            )
        )
    return out


def calibration_table(p: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed": float(outcome[mask].mean()) if mask.any() else np.nan,
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(
    table: pd.DataFrame,
    history: pd.DataFrame,
    seasons: Iterable[int],
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int = 0,
    dnf_cache: dict[tuple[int, str], float] | None = None,
) -> BacktestResult:
    seasons = list(seasons)
    races = table[(~table["is_sprint"]) & (table["season"].isin(seasons))]
    race_ids = races.sort_values("date")["race_id"].unique()
    if dnf_cache is None:
        dnf_cache = dnf_cache_for_races(table, race_ids, params)
    race_history = history[~history["is_sprint"]]

    rows, all_p, all_won = [], [], []
    for i, race_id in enumerate(race_ids):
        race_rows = races[races["race_id"] == race_id]
        hist_rows = race_history[race_history["race_id"] == race_id]
        entrants = entrants_for_past_race(race_rows, hist_rows, dnf_cache, params)
        forecast = simulate_race(entrants, params, n_runs=n_runs, seed=seed + i)

        classified = race_rows[race_rows["position"].notna()]
        actual = dict(zip(classified["driver_id"], classified["position"].astype(int), strict=True))
        winner = next(d for d, pos in actual.items() if pos == 1)
        podium = {d for d, pos in actual.items() if pos <= 3}
        grid = dict(zip(race_rows["driver_id"], race_rows["grid"].astype(int), strict=True))

        ids = forecast.driver_ids
        p_win = dict(zip(ids, forecast.p_win, strict=True))
        p_podium = dict(zip(ids, forecast.p_podium, strict=True))
        expected = dict(zip(ids, forecast.expected_position, strict=True))
        pole = pole_baseline(grid)
        uni = uniform_baseline(ids)

        first = race_rows.iloc[0]
        rows.append(
            {
                "season": int(first.season),
                "round": int(first["round"]),
                "race_id": int(race_id),
                "race_name": first.race_name,
                "winner": winner,
                "winner_p": p_win[winner],
                "model_logloss": winner_log_loss(p_win, winner),
                "pole_logloss": winner_log_loss(pole.p_win, winner),
                "uniform_logloss": winner_log_loss(uni.p_win, winner),
                "model_brier": podium_brier(p_podium, podium),
                "pole_brier": podium_brier(pole.p_podium, podium),
                "uniform_brier": podium_brier(uni.p_podium, podium),
                "model_spearman": position_spearman(expected, actual),
                "pole_spearman": position_spearman(pole.expected_position, actual),
            }
        )
        all_p.extend(forecast.p_win.tolist())
        all_won.extend([1.0 if d == winner else 0.0 for d in ids])

    race_df = pd.DataFrame(rows, columns=RACE_SCORE_COLUMNS)
    season_df = (
        race_df.groupby("season")
        .agg(n_races=("race_id", "count"), **{c: (c, "mean") for c in RACE_SCORE_COLUMNS[6:]})
        .reset_index()[SEASON_SCORE_COLUMNS]
    )
    calibration = calibration_table(np.array(all_p), np.array(all_won))
    return BacktestResult(race_df, season_df, calibration)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_backtest.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 4 passed. Note the 2024 sample has 20 entrants per race; the uniform log loss check allows for a race with a different field size.

- [ ] **Step 5: Write `docs/backtesting.md`** (about 250 words): why we only score with pre-race information, what log loss and Brier mean in words ("how surprised were we"), why we compare to "pole wins" and "everyone equal", what a calibration curve shows, and why tuning seasons and reporting seasons are kept separate.

- [ ] **Step 6: Commit**

```bash
git add f1pred/backtest/run.py tests/test_backtest.py docs/backtesting.md
git commit -m "feat: leakage-safe backtest with baselines and calibration"
```

---

### Task 10: Race resolution and entrants for future races

**Files:**
- Create: `f1pred/predict.py`, `tests/test_predict.py`

**Interfaces:**
- Consumes: raw tables (Task 2), driver-race table (Task 3), `RatingState`, `state_for_season`, `strength` (Task 5), `dnf_probability` (Task 6), `Entrant` (Task 7), `entrants_for_past_race` (Task 9).
- Produces:
  - `class UnknownRaceError(Exception)` with attribute `choices: list[str]`
  - `@dataclass RaceRef(race_id: int, season: int, round: int, name: str, circuit_id: str, date: pd.Timestamp, has_results: bool)`
  - `resolve_race(raw: dict, table: pd.DataFrame, season: int, round: int | None = None, race: str | None = None) -> RaceRef`
  - `@dataclass PredictionInputs(race: RaceRef, entrants: list[Entrant], names: dict[str, str], use_grid: bool, note: str | None)`
  - `build_prediction_inputs(raw, table, history, state, race: RaceRef, params, use_grid: bool = True) -> PredictionInputs`

Rules:
  - Past race (`has_results`): entrants from the table rows and pre-race history rows (reuse `entrants_for_past_race`) with `dnf_probability` computed at the race date.
  - Future race with qualifying rows: entrants from `qualifying` joined to `drivers`/`constructors`, `grid = position`, strengths from `state_for_season(state, season)`, `p_dnf` at the race date.
  - Future race without qualifying: entrants of the most recent session in the table (any season), `grid=None`, `use_grid=False`, `note="No qualifying data yet; running without grid"`.

- [ ] **Step 1: Write the failing tests `tests/test_predict.py`**

```python
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
    e = next(x for x in inputs.entrants if x.driver_id == "verstappen")
    d = 1500 + (state.driver["verstappen"] - 1500) * 0.9
    c = 1500 + (state.constructor[e.constructor_id] - 1500) * 0.6
    assert e.strength == pytest.approx(d + c)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_predict.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/predict.py`**

```python
"""Turn a season/round or race name into simulation inputs, for past or future races."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from f1pred.backtest.run import entrants_for_past_race
from f1pred.config import ModelParams
from f1pred.ratings.history import RatingState, state_for_season, strength
from f1pred.sim.dnf import dnf_probability
from f1pred.sim.race import Entrant

NO_QUALIFYING_NOTE = "No qualifying data yet; running without grid"


class UnknownRaceError(Exception):
    def __init__(self, message: str, choices: list[str]):
        super().__init__(message)
        self.choices = choices


@dataclass
class RaceRef:
    race_id: int
    season: int
    round: int
    name: str
    circuit_id: str
    date: pd.Timestamp
    has_results: bool


@dataclass
class PredictionInputs:
    race: RaceRef
    entrants: list[Entrant]
    names: dict[str, str]
    use_grid: bool
    note: str | None


def _season_races(raw: dict, season: int) -> pd.DataFrame:
    races = raw["races"].merge(raw["circuits"][["circuitId", "circuitRef"]], on="circuitId")
    return races[races["year"] == season].sort_values("round")


def resolve_race(
    raw: dict,
    table: pd.DataFrame,
    season: int,
    round: int | None = None,
    race: str | None = None,
) -> RaceRef:
    races = _season_races(raw, season)
    choices = [f"{int(r.round):2d}  {r.name}  ({r.circuitRef})" for r in races.itertuples()]
    if races.empty:
        raise UnknownRaceError(f"No races found for season {season}", choices)
    if round is not None:
        match = races[races["round"] == round]
    elif race is not None:
        needle = race.lower()
        match = races[
            (races["circuitRef"].str.lower() == needle)
            | races["name"].str.lower().str.contains(needle, regex=False)
        ]
    else:
        raise UnknownRaceError("Give --round or --race", choices)
    if len(match) != 1:
        what = f"round {round}" if round is not None else f"race '{race}'"
        raise UnknownRaceError(f"{what} did not match exactly one race in {season}", choices)
    r = match.iloc[0]
    race_id = int(r.raceId)
    has_results = bool(((table["race_id"] == race_id) & (~table["is_sprint"])).any())
    return RaceRef(
        race_id=race_id,
        season=season,
        round=int(r["round"]),
        name=str(r["name"]),
        circuit_id=str(r["circuitRef"]),
        date=pd.Timestamp(r["date"]),
        has_results=has_results,
    )


def _names_from_table(table: pd.DataFrame) -> dict[str, str]:
    return dict(zip(table["driver_id"], table["driver_name"], strict=True))


def _future_entrants(
    raw: dict, table: pd.DataFrame, state: RatingState, ref: RaceRef, params: ModelParams
) -> tuple[list[Entrant], dict[str, str], bool, str | None]:
    season_state = state_for_season(state, ref.season, params)
    quali = raw["qualifying"][raw["qualifying"]["raceId"] == ref.race_id]
    if not quali.empty:
        q = quali.merge(raw["drivers"][["driverId", "driverRef", "forename", "surname"]], on="driverId")
        q = q.merge(raw["constructors"][["constructorId", "constructorRef"]], on="constructorId")
        rows = [
            (r.driverRef, r.constructorRef, int(r.position), f"{r.forename} {r.surname}")
            for r in q.itertuples()
        ]
        use_grid, note = True, None
    else:
        races = table[~table["is_sprint"]]
        latest = races[races["race_id"] == races.sort_values("date")["race_id"].iloc[-1]]
        rows = [(r.driver_id, r.constructor_id, None, r.driver_name) for r in latest.itertuples()]
        use_grid, note = False, NO_QUALIFYING_NOTE
    entrants, names = [], {}
    for driver_id, constructor_id, grid, name in rows:
        entrants.append(
            Entrant(
                driver_id=driver_id,
                constructor_id=constructor_id,
                strength=strength(season_state, driver_id, constructor_id, params),
                grid=grid,
                p_dnf=dnf_probability(table, driver_id, constructor_id, ref.circuit_id, ref.date, params),
                low_confidence=season_state.driver_races.get(driver_id, 0) < params.min_races_for_confidence,
            )
        )
        names[driver_id] = name
    return entrants, names, use_grid, note


def build_prediction_inputs(
    raw: dict,
    table: pd.DataFrame,
    history: pd.DataFrame,
    state: RatingState,
    race: RaceRef,
    params: ModelParams,
    use_grid: bool = True,
) -> PredictionInputs:
    if race.has_results:
        rows = table[(table["race_id"] == race.race_id) & (~table["is_sprint"])]
        hist = history[(history["race_id"] == race.race_id) & (~history["is_sprint"])]
        dnf = {
            (race.race_id, r.driver_id): dnf_probability(
                table, r.driver_id, r.constructor_id, r.circuit_id, r.date, params
            )
            for r in rows.itertuples(index=False)
        }
        entrants = entrants_for_past_race(rows, hist, dnf, params)
        return PredictionInputs(race, entrants, _names_from_table(rows), use_grid, None)
    entrants, names, grid_ok, note = _future_entrants(raw, table, state, race, params)
    return PredictionInputs(race, entrants, names, use_grid and grid_ok, note)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_predict.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 7 passed. In `test_next_season_applies_regression`, 2025 is not a regulation season so constructors regress by 0.4 (factor 0.6).

- [ ] **Step 5: Commit**

```bash
git add f1pred/predict.py tests/test_predict.py
git commit -m "feat: resolve races and build entrants for past and future races"
```

---

### Task 11: Tables and charts

**Files:**
- Create: `f1pred/report/tables.py`, `f1pred/report/charts.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `RaceForecast` (Task 7), `RatingState` (Task 5), `BacktestResult` (Task 9).
- Produces:
  - `forecast_table(forecast: RaceForecast, names: Mapping[str, str], low_confidence: set[str], title: str) -> rich.table.Table`
  - `ratings_table(state: RatingState, names: Mapping[str, str], top: int = 10) -> tuple[Table, Table]` (drivers, constructors)
  - `backtest_table(seasons: pd.DataFrame) -> Table`
  - `win_chart(forecast, names, title, path: Path) -> Path`
  - `position_heatmap(forecast, names, title, path: Path) -> Path`
  - `calibration_chart(calibration: pd.DataFrame, path: Path) -> Path`

- [ ] **Step 1: Write the failing tests `tests/test_report.py`**

```python
import numpy as np
import pandas as pd
from rich.console import Console

from f1pred.backtest.run import calibration_table
from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import RatingState
from f1pred.report.charts import calibration_chart, position_heatmap, win_chart
from f1pred.report.tables import backtest_table, forecast_table, ratings_table
from f1pred.sim.race import Entrant, simulate_race

NAMES = {f"d{i}": f"Driver {i}" for i in range(6)}


def _forecast():
    ents = [Entrant(f"d{i}", f"t{i // 2}", 3000.0 + 20 * i, i + 1, 0.1) for i in range(6)]
    return simulate_race(ents, DEFAULT_PARAMS, n_runs=300, seed=0)


def _render(table) -> str:
    console = Console(record=True, width=120)
    console.print(table)
    return console.export_text()


def test_forecast_table_marks_low_confidence():
    text = _render(forecast_table(_forecast(), NAMES, {"d1"}, "Test GP"))
    assert "Driver 1" in text and "*" in text and "Test GP" in text
    assert "%" in text


def test_ratings_tables():
    state = RatingState({"d0": 1600.0, "d1": 1400.0}, {"t0": 1550.0}, {"d0": 5}, 2024)
    drivers, cons = ratings_table(state, NAMES, top=1)
    assert "Driver 0" in _render(drivers) and "Driver 1" not in _render(drivers)
    assert "t0" in _render(cons)


def test_backtest_table():
    seasons = pd.DataFrame(
        {
            "season": [2024],
            "n_races": [24],
            "model_logloss": [1.2],
            "pole_logloss": [1.5],
            "uniform_logloss": [3.0],
            "model_brier": [0.1],
            "pole_brier": [0.12],
            "uniform_brier": [0.2],
            "model_spearman": [0.7],
            "pole_spearman": [0.6],
        }
    )
    text = _render(backtest_table(seasons))
    assert "2024" in text and "1.20" in text and "3.00" in text


def test_charts_write_png(tmp_path):
    f = _forecast()
    assert win_chart(f, NAMES, "Test GP", tmp_path / "win.png").stat().st_size > 1000
    assert position_heatmap(f, NAMES, "Test GP", tmp_path / "pos.png").stat().st_size > 1000
    cal = calibration_table(np.linspace(0, 1, 50), (np.linspace(0, 1, 50) > 0.5).astype(float))
    assert calibration_chart(cal, tmp_path / "cal.png").stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/report/tables.py`**

```python
"""Rich tables for the terminal."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from rich.table import Table

from f1pred.ratings.history import RatingState
from f1pred.sim.race import RaceForecast


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def forecast_table(
    forecast: RaceForecast, names: Mapping[str, str], low_confidence: set[str], title: str
) -> Table:
    table = Table(title=title, caption="* fewer than 5 rated races: low confidence")
    for col, justify in [
        ("#", "right"),
        ("Driver", "left"),
        ("Win", "right"),
        ("Podium", "right"),
        ("Points", "right"),
        ("Exp. pos", "right"),
        ("P10-P90", "right"),
    ]:
        table.add_column(col, justify=justify)
    for i, r in enumerate(forecast.as_frame().itertuples(index=False), start=1):
        label = names.get(r.driver_id, r.driver_id) + (" *" if r.driver_id in low_confidence else "")
        table.add_row(
            str(i),
            label,
            _pct(r.p_win),
            _pct(r.p_podium),
            _pct(r.p_points),
            f"{r.expected_position:4.1f}",
            f"{int(r.position_p10)}-{int(r.position_p90)}",
        )
    return table


def ratings_table(state: RatingState, names: Mapping[str, str], top: int = 10) -> tuple[Table, Table]:
    drivers = Table(title=f"Top {top} drivers (Elo)")
    drivers.add_column("#", justify="right")
    drivers.add_column("Driver")
    drivers.add_column("Rating", justify="right")
    for i, (d, r) in enumerate(sorted(state.driver.items(), key=lambda kv: -kv[1])[:top], 1):
        drivers.add_row(str(i), names.get(d, d), f"{r:7.1f}")
    cons = Table(title=f"Top {top} constructors (Elo)")
    cons.add_column("#", justify="right")
    cons.add_column("Constructor")
    cons.add_column("Rating", justify="right")
    for i, (c, r) in enumerate(sorted(state.constructor.items(), key=lambda kv: -kv[1])[:top], 1):
        cons.add_row(str(i), c, f"{r:7.1f}")
    return drivers, cons


def backtest_table(seasons: pd.DataFrame) -> Table:
    table = Table(title="Backtest: model vs baselines (lower log loss / Brier is better)")
    cols = [
        ("Season", "season"),
        ("Races", "n_races"),
        ("LogLoss model", "model_logloss"),
        ("LogLoss pole", "pole_logloss"),
        ("LogLoss uniform", "uniform_logloss"),
        ("Brier model", "model_brier"),
        ("Brier pole", "pole_brier"),
        ("Brier uniform", "uniform_brier"),
        ("Spearman model", "model_spearman"),
        ("Spearman pole", "pole_spearman"),
    ]
    for label, _ in cols:
        table.add_column(label, justify="right")
    for r in seasons.itertuples(index=False):
        values = []
        for _, key in cols:
            v = getattr(r, key)
            values.append(str(int(v)) if key in ("season", "n_races") else f"{v:.2f}")
        table.add_row(*values)
    return table
```

- [ ] **Step 4: Write `f1pred/report/charts.py`**

```python
"""PNG charts. Uses the Agg backend so it works headless."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from f1pred.sim.race import RaceForecast  # noqa: E402


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def win_chart(forecast: RaceForecast, names: Mapping[str, str], title: str, path: Path) -> Path:
    df = forecast.as_frame()
    labels = [names.get(d, d) for d in df.driver_id]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1.5))
    ax.barh(labels[::-1], (100 * df.p_win)[::-1], color="#e10600")
    ax.set_xlabel("Win probability (%)")
    ax.set_title(f"{title}: win probability ({forecast.n_runs:,} runs)")
    for i, v in enumerate((100 * df.p_win)[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    return _save(fig, path)


def position_heatmap(
    forecast: RaceForecast, names: Mapping[str, str], title: str, path: Path
) -> Path:
    order = np.argsort(forecast.expected_position)
    matrix = forecast.position_matrix[order]
    labels = [names.get(forecast.driver_ids[i], forecast.driver_ids[i]) for i in order]
    n = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.35 * n + 2))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=max(matrix.max(), 1e-9))
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Finishing position")
    ax.set_title(f"{title}: P(driver finishes in position)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, path)


def calibration_chart(calibration: pd.DataFrame, path: Path) -> Path:
    cal = calibration.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.scatter(cal.predicted, cal.observed, s=10 + cal["count"] / cal["count"].max() * 200, color="#e10600")
    ax.plot(cal.predicted, cal.observed, color="#e10600", label="model")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration (bubble size = count)")
    ax.legend()
    return _save(fig, path)
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/test_report.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 4 passed. If ruff complains about the E402 imports after `matplotlib.use`, keep the `# noqa: E402` comments.

- [ ] **Step 6: Commit**

```bash
git add f1pred/report tests/test_report.py
git commit -m "feat: rich tables and PNG charts"
```

---

### Task 12: CLI

**Files:**
- Create: `f1pred/cli.py`, `tests/test_cli.py`
- Modify: `tests/conftest.py` (add `cache_dir` fixture)

**Interfaces:**
- Consumes: everything above.
- Produces: Typer app `f1pred.cli:app` with:
  - `f1pred data update [--cache-dir DIR]`: `load_tables(refresh=True)`, build the driver-race table, write `driver_race.parquet`, print row counts.
  - `f1pred ratings build [--cache-dir DIR]`: read `driver_race.parquet`, `replay`, write `ratings_history.parquet` and `ratings_state.json`, print top-10 tables.
  - `f1pred predict --season Y (--round N | --race NAME) [--runs 10000] [--seed 0] [--no-grid] [--cache-dir DIR] [--out DIR]`: resolve, build inputs, simulate, print `forecast_table`, save `outputs/<season>-<round:02d>-win.png` and `-positions.png`, print the note if any.
  - `f1pred backtest --seasons 2023-2025 [--runs] [--seed] [--cache-dir] [--out]`: run, print `backtest_table`, write `outputs/backtest_races.csv`, `outputs/backtest_seasons.csv`, `outputs/calibration.png`.
  - `f1pred tune [...]`: added in Task 13; leave a stub that prints "not implemented" for now? No. Do not add it in this task at all.
  - Helper `_load_cached(cache_dir) -> tuple[raw, table]`, `_load_ratings(cache_dir) -> tuple[history, state]`; both exit with code 1 and the spec's message when files are missing. `UnknownRaceError` exits 2 and prints the choices.
  - Exit codes: 1 data missing or unavailable, 2 unknown race.
  - `DRIVER_RACE_FILE = "driver_race.parquet"`, `HISTORY_FILE = "ratings_history.parquet"`, `STATE_FILE = "ratings_state.json"`.

- [ ] **Step 1: Add to `tests/conftest.py`**

```python
from f1pred.data.hub import TABLES


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory, raw_sample, driver_race) -> Path:
    """A cache directory pre-populated from the sample, as `f1pred data update` would leave it."""
    d = tmp_path_factory.mktemp("cache")
    for t in TABLES:
        raw_sample[t].to_parquet(d / f"{t}.parquet", index=False)
    driver_race.to_parquet(d / "driver_race.parquet", index=False)
    return d
```

- [ ] **Step 2: Write the failing tests `tests/test_cli.py`**

```python
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1pred.cli import app
from f1pred.data import hub

runner = CliRunner()
SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


def test_data_update_uses_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hub, "hf_hub_download", lambda repo_id, filename, repo_type: str(SAMPLE_DIR / Path(filename).name)
    )
    r = runner.invoke(app, ["data", "update", "--cache-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "driver_race.parquet").exists()
    assert "driver-race rows" in r.output


def test_ratings_build_then_predict_past(cache_dir, tmp_path):
    r = runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    assert r.exit_code == 0, r.output
    assert (cache_dir / "ratings_history.parquet").exists()
    assert "Max Verstappen" in r.output

    r = runner.invoke(
        app,
        [
            "predict",
            "--season",
            "2024",
            "--race",
            "monaco",
            "--runs",
            "300",
            "--seed",
            "1",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Monaco Grand Prix" in r.output and "Charles Leclerc" in r.output
    assert (tmp_path / "2024-08-win.png").exists()
    assert (tmp_path / "2024-08-positions.png").exists()


def test_predict_unknown_race_lists_choices(cache_dir):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(app, ["predict", "--season", "2024", "--race", "mars", "--cache-dir", str(cache_dir)])
    assert r.exit_code == 2
    assert "Monaco" in r.output


def test_predict_without_ratings_exits_1(tmp_path, raw_sample, driver_race):
    for t in hub.TABLES:
        raw_sample[t].to_parquet(tmp_path / f"{t}.parquet", index=False)
    driver_race.to_parquet(tmp_path / "driver_race.parquet", index=False)
    r = runner.invoke(app, ["predict", "--season", "2024", "--round", "1", "--cache-dir", str(tmp_path)])
    assert r.exit_code == 1
    assert "f1pred ratings build" in r.output


def test_backtest_writes_outputs(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        ["backtest", "--seasons", "2024", "--runs", "200", "--cache-dir", str(cache_dir), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    assert "2024" in r.output
    seasons = pd.read_csv(tmp_path / "backtest_seasons.csv")
    assert seasons.n_races.iloc[0] == 24
    assert (tmp_path / "calibration.png").exists()


def test_backtest_season_range_parsing(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        ["backtest", "--seasons", "2023-2024", "--runs", "100", "--cache-dir", str(cache_dir), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    assert len(pd.read_csv(tmp_path / "backtest_seasons.csv")) == 2
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Write `f1pred/cli.py`**

```python
"""The `f1pred` command line."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from f1pred.backtest.run import run_backtest
from f1pred.config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, load_params
from f1pred.data.frame import build_driver_race_table
from f1pred.data.hub import DataUnavailableError, load_cached_tables, load_tables
from f1pred.predict import UnknownRaceError, build_prediction_inputs, resolve_race
from f1pred.ratings.history import RatingState, replay
from f1pred.report.charts import calibration_chart, position_heatmap, win_chart
from f1pred.report.tables import backtest_table, forecast_table, ratings_table
from f1pred.sim.race import simulate_race

DRIVER_RACE_FILE = "driver_race.parquet"
HISTORY_FILE = "ratings_history.parquet"
STATE_FILE = "ratings_state.json"

app = typer.Typer(help="F1 race predictions with Elo ratings and Monte Carlo simulation.")
data_app = typer.Typer(help="Download and cache data.")
ratings_app = typer.Typer(help="Build ratings.")
app.add_typer(data_app, name="data")
app.add_typer(ratings_app, name="ratings")
console = Console()

CacheDir = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", help="Parquet cache directory")
OutDir = typer.Option(DEFAULT_OUTPUT_DIR, "--out", help="Where charts and CSVs go")


def _fail(message: str, code: int) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code)


def _load_cached(cache_dir: Path) -> tuple[dict, pd.DataFrame]:
    try:
        raw = load_cached_tables(cache_dir)
    except Exception as exc:
        _fail(f"{exc}\nRun `f1pred data update` first.", 1)
    path = cache_dir / DRIVER_RACE_FILE
    if not path.exists():
        _fail(f"{path} not found. Run `f1pred data update` first.", 1)
    return raw, pd.read_parquet(path)


def _load_ratings(cache_dir: Path) -> tuple[pd.DataFrame, RatingState]:
    hist, state = cache_dir / HISTORY_FILE, cache_dir / STATE_FILE
    if not hist.exists() or not state.exists():
        _fail("Ratings not built yet. Run `f1pred ratings build` first.", 1)
    return pd.read_parquet(hist), RatingState.from_json(state)


def _parse_seasons(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@data_app.command("update")
def data_update(cache_dir: Path = CacheDir) -> None:
    """Download the RaceData tables from Hugging Face and build the driver-race table."""
    params = load_params()
    try:
        raw = load_tables(cache_dir, refresh=True)
    except DataUnavailableError as exc:
        _fail(str(exc), 1)
    table = build_driver_race_table(raw, start_season=params.start_season)
    table.to_parquet(cache_dir / DRIVER_RACE_FILE, index=False)
    races = table[~table.is_sprint]
    console.print(
        f"{len(table):,} driver-race rows, {races.race_id.nunique()} races, "
        f"{races.season.min()}-{races.season.max()}, latest: {races.race_name.iloc[-1]} "
        f"{races.date.iloc[-1].date()}"
    )


@ratings_app.command("build")
def ratings_build(cache_dir: Path = CacheDir) -> None:
    """Replay history and write pre-race ratings plus the current rating state."""
    params = load_params()
    _, table = _load_cached(cache_dir)
    history, state = replay(table, params)
    history.to_parquet(cache_dir / HISTORY_FILE, index=False)
    state.to_json(cache_dir / STATE_FILE)
    names = dict(zip(table.driver_id, table.driver_name, strict=True))
    drivers, cons = ratings_table(state, names)
    console.print(drivers)
    console.print(cons)


@app.command()
def predict(
    season: int = typer.Option(..., "--season"),
    round: int | None = typer.Option(None, "--round"),
    race: str | None = typer.Option(None, "--race"),
    runs: int = typer.Option(10_000, "--runs"),
    seed: int = typer.Option(0, "--seed"),
    no_grid: bool = typer.Option(False, "--no-grid"),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Win, podium and points probabilities for one race."""
    params = load_params()
    raw, table = _load_cached(cache_dir)
    history, state = _load_ratings(cache_dir)
    try:
        ref = resolve_race(raw, table, season, round=round, race=race)
    except UnknownRaceError as exc:
        console.print(f"[red]{exc}[/red]\nRaces in {season}:")
        for c in exc.choices:
            console.print("  " + c)
        raise typer.Exit(2) from None
    inputs = build_prediction_inputs(raw, table, history, state, ref, params, use_grid=not no_grid)
    if inputs.note:
        console.print(f"[yellow]{inputs.note}[/yellow]")
    forecast = simulate_race(inputs.entrants, params, n_runs=runs, seed=seed, use_grid=inputs.use_grid)
    low = {e.driver_id for e in inputs.entrants if e.low_confidence}
    title = f"{ref.name} {ref.season} (round {ref.round})"
    console.print(forecast_table(forecast, inputs.names, low, title))
    stem = f"{ref.season}-{ref.round:02d}"
    win_path = win_chart(forecast, inputs.names, title, out / f"{stem}-win.png")
    pos_path = position_heatmap(forecast, inputs.names, title, out / f"{stem}-positions.png")
    console.print(f"Charts: {win_path}, {pos_path}")


@app.command()
def backtest(
    seasons: str = typer.Option("2023-2025", "--seasons", help="e.g. 2023-2025 or 2022,2024"),
    runs: int = typer.Option(10_000, "--runs"),
    seed: int = typer.Option(0, "--seed"),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Score the model on past seasons against the pole-wins and uniform baselines."""
    params = load_params()
    _, table = _load_cached(cache_dir)
    history, _ = _load_ratings(cache_dir)
    result = run_backtest(table, history, _parse_seasons(seasons), params, n_runs=runs, seed=seed)
    console.print(backtest_table(result.seasons))
    out.mkdir(parents=True, exist_ok=True)
    result.races.to_csv(out / "backtest_races.csv", index=False)
    result.seasons.to_csv(out / "backtest_seasons.csv", index=False)
    console.print(f"Calibration chart: {calibration_chart(result.calibration, out / 'calibration.png')}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check .`
Expected: whole suite green. Typer's `int | None` options need Typer >= 0.12 (pinned).

- [ ] **Step 6: Commit**

```bash
git add f1pred/cli.py tests/conftest.py tests/test_cli.py
git commit -m "feat: f1pred CLI with data, ratings, predict and backtest commands"
```

---

### Task 13: Tuning

**Files:**
- Create: `f1pred/tune.py`, `tests/test_tune.py`
- Modify: `f1pred/cli.py` (add `tune` command)

**Interfaces:**
- Produces:
  - `SEARCH_SPACE: dict[str, list[float]]` = `{"k_driver": [16, 24, 32], "k_constructor": [24, 32, 48], "sigma_team": [40, 60, 90], "sigma_driver": [60, 80, 110], "grid_bonus": [4, 8, 12], "regress_constructor": [0.3, 0.4, 0.6]}`
  - `objective(table, params, train_seasons, n_runs, seed) -> float` (mean winner log loss over train seasons; replays ratings for the given params)
  - `coordinate_descent(table, base: ModelParams, train_seasons, space=SEARCH_SPACE, passes: int = 2, n_runs: int = 2000, seed: int = 0, log=print) -> tuple[ModelParams, float]`
  - `write_tuned(params: ModelParams, path: Path, note: str) -> None` (JSON of only the fields that differ from `DEFAULT_PARAMS`, plus `"_note"`; `load_params` must ignore keys starting with `_`)
  - CLI: `f1pred tune [--train 2015-2022] [--test 2023-2025] [--passes 2] [--runs 2000] [--cache-dir]` prints the chosen params, then the held-out backtest table for both default and tuned params, and writes `f1pred/tuned.json`.

- [ ] **Step 1: Update `load_params` in `f1pred/config.py`** to drop keys starting with `_`:

```python
    overrides = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}
```
Add a test to `tests/test_config.py`:

```python
def test_load_params_ignores_underscore_keys(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"_note": "x", "k_driver": 30.0}))
    assert load_params(path).k_driver == 30.0
```

- [ ] **Step 2: Write the failing tests `tests/test_tune.py`**

```python
import json

from f1pred.config import DEFAULT_PARAMS, load_params
from f1pred.tune import coordinate_descent, objective, write_tuned


def test_objective_is_finite_and_params_matter(driver_race):
    a = objective(driver_race, DEFAULT_PARAMS, [2024], n_runs=200, seed=0)
    b = objective(driver_race, DEFAULT_PARAMS.replace(sigma_driver=1000.0), [2024], n_runs=200, seed=0)
    assert 0 < a < 14 and 0 < b < 14
    assert a != b


def test_coordinate_descent_small_space(driver_race):
    space = {"grid_bonus": [0.0, 8.0], "sigma_team": [60.0]}
    best, score = coordinate_descent(
        driver_race, DEFAULT_PARAMS, [2024], space=space, passes=1, n_runs=100, seed=0, log=lambda *_: None
    )
    assert best.grid_bonus in (0.0, 8.0)
    assert best.sigma_team == 60.0
    assert score > 0


def test_write_tuned_only_diffs(tmp_path):
    p = DEFAULT_PARAMS.replace(k_driver=30.0)
    path = tmp_path / "tuned.json"
    write_tuned(p, path, "test")
    data = json.loads(path.read_text())
    assert data == {"_note": "test", "k_driver": 30.0}
    assert load_params(path).k_driver == 30.0
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_tune.py tests/test_config.py -v`
Expected: `test_tune` fails with `ImportError`; the new config test fails until step 1 is applied (apply it now).

- [ ] **Step 4: Write `f1pred/tune.py`**

```python
"""Coordinate-descent search over ModelParams, scored by winner log loss. Spec 8.1."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from f1pred.backtest.run import run_backtest
from f1pred.config import DEFAULT_PARAMS, ModelParams
from f1pred.ratings.history import replay
from f1pred.sim.dnf import dnf_cache_for_races

SEARCH_SPACE: dict[str, list[float]] = {
    "k_driver": [16.0, 24.0, 32.0],
    "k_constructor": [24.0, 32.0, 48.0],
    "sigma_team": [40.0, 60.0, 90.0],
    "sigma_driver": [60.0, 80.0, 110.0],
    "grid_bonus": [4.0, 8.0, 12.0],
    "regress_constructor": [0.3, 0.4, 0.6],
}


def objective(
    table: pd.DataFrame,
    params: ModelParams,
    train_seasons: Iterable[int],
    n_runs: int = 2000,
    seed: int = 0,
    dnf_cache: dict | None = None,
) -> float:
    """Mean winner log loss over the training seasons for these params."""
    history, _ = replay(table, params)
    result = run_backtest(
        table, history, list(train_seasons), params, n_runs=n_runs, seed=seed, dnf_cache=dnf_cache
    )
    return float(result.races["model_logloss"].mean())


def coordinate_descent(
    table: pd.DataFrame,
    base: ModelParams,
    train_seasons: Iterable[int],
    space: dict[str, list[float]] = SEARCH_SPACE,
    passes: int = 2,
    n_runs: int = 2000,
    seed: int = 0,
    log: Callable[..., None] = print,
) -> tuple[ModelParams, float]:
    """Try each candidate value of one parameter at a time, keep the best, repeat."""
    train = list(train_seasons)
    race_ids = table[(~table.is_sprint) & (table.season.isin(train))].race_id.unique()
    dnf_cache = dnf_cache_for_races(table, race_ids, base)
    best = base
    best_score = objective(table, best, train, n_runs, seed, dnf_cache)
    log(f"start: {best_score:.4f}")
    for p in range(passes):
        for name, candidates in space.items():
            for value in candidates:
                if getattr(best, name) == value:
                    continue
                trial = best.replace(**{name: value})
                score = objective(table, trial, train, n_runs, seed, dnf_cache)
                log(f"pass {p + 1} {name}={value}: {score:.4f}")
                if score < best_score:
                    best, best_score = trial, score
                    log(f"  -> new best {best_score:.4f}")
    return best, best_score


def write_tuned(params: ModelParams, path: Path, note: str) -> None:
    diffs = {
        f.name: getattr(params, f.name)
        for f in dataclasses.fields(ModelParams)
        if getattr(params, f.name) != getattr(DEFAULT_PARAMS, f.name)
    }
    Path(path).write_text(json.dumps({"_note": note, **diffs}, indent=2))
```

- [ ] **Step 5: Add the `tune` command to `f1pred/cli.py`**

```python
from datetime import date

from f1pred import tune as tune_mod
from f1pred.config import DEFAULT_PARAMS, TUNED_PARAMS_PATH


@app.command()
def tune(
    train: str = typer.Option("2015-2022", "--train"),
    test: str = typer.Option("2023-2025", "--test"),
    passes: int = typer.Option(2, "--passes"),
    runs: int = typer.Option(2000, "--runs"),
    cache_dir: Path = CacheDir,
) -> None:
    """Search model parameters on the training seasons, report held-out seasons, save tuned.json."""
    _, table = _load_cached(cache_dir)
    train_seasons, test_seasons = _parse_seasons(train), _parse_seasons(test)
    best, score = tune_mod.coordinate_descent(
        table,
        DEFAULT_PARAMS,
        train_seasons,
        space=tune_mod.SEARCH_SPACE,
        passes=passes,
        n_runs=runs,
        log=console.print,
    )
    console.print(f"Best training log loss: {score:.4f}")
    for label, params in [("default", DEFAULT_PARAMS), ("tuned", best)]:
        history, _ = replay(table, params)
        result = run_backtest(table, history, test_seasons, params, n_runs=runs, seed=0)
        console.print(f"[bold]{label} params, held-out {test}[/bold]")
        console.print(backtest_table(result.seasons))
    note = f"tuned {date.today()} on {train}, held out {test}, train log loss {score:.4f}"
    tune_mod.write_tuned(best, TUNED_PARAMS_PATH, note)
    console.print(f"Wrote {TUNED_PARAMS_PATH}. Re-run `f1pred ratings build` to use it.")
```

Add a CLI smoke test to `tests/test_cli.py`:

```python
def test_tune_smoke(cache_dir, tmp_path, monkeypatch):
    import f1pred.cli as cli
    import f1pred.tune as tune_mod

    monkeypatch.setattr(cli, "TUNED_PARAMS_PATH", tmp_path / "tuned.json")
    monkeypatch.setattr(tune_mod, "SEARCH_SPACE", {"grid_bonus": [8.0, 12.0]})
    r = runner.invoke(
        app, ["tune", "--train", "2023", "--test", "2024", "--passes", "1", "--runs", "50", "--cache-dir", str(cache_dir)]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "tuned.json").exists()
```
The CLI goes through `tune_mod.SEARCH_SPACE` at call time (not the function's default argument) so the monkeypatch above takes effect.

- [ ] **Step 6: Run the whole suite and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check .`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add f1pred/tune.py f1pred/cli.py f1pred/config.py tests/test_tune.py tests/test_cli.py tests/test_config.py
git commit -m "feat: coordinate-descent tuning with held-out reporting"
```

---

### Task 14: Run the real pipeline, tune, write README

**Files:**
- Create: `README.md`, `f1pred/tuned.json`, `docs/results/` (committed PNGs: `backtest-calibration.png`, one `predict` win chart, one heatmap)
- Modify: `.gitignore` is unchanged (outputs/ stays ignored; results are copied into `docs/results/`)

This task needs network access to Hugging Face.

- [ ] **Step 1: Download and build**

```bash
uv run f1pred data update
uv run f1pred ratings build
```
Expected: a row count around 7,500 for 2010 to 2026, latest race the 2026 Spanish Grand Prix or later; top drivers look sane (Verstappen, Norris, Leclerc, Piastri, Hamilton near the top; a top constructor from McLaren, Red Bull, Ferrari, Mercedes).

- [ ] **Step 2: Baseline backtest with default params**

```bash
uv run f1pred backtest --seasons 2023-2025
```
Record the season table. The model must beat the uniform baseline on every score and should beat "pole wins" on log loss in at least two of three seasons. If it does not beat uniform, stop and debug before tuning (check `driver_rating_pre` values are not all 1500 and that grid values are 1 to 20).

- [ ] **Step 3: Tune**

```bash
uv run f1pred tune --train 2015-2022 --test 2023-2025 --passes 2 --runs 2000
uv run f1pred ratings build
uv run f1pred backtest --seasons 2023-2025
```
Expected runtime: 10 to 30 minutes. Keep the held-out table printed for the README. Commit `f1pred/tuned.json`.

- [ ] **Step 4: Produce the showcase charts**

```bash
uv run f1pred predict --season 2026 --round 15
mkdir -p docs/results
cp outputs/calibration.png docs/results/backtest-calibration.png
cp outputs/2026-15-win.png docs/results/example-win.png
cp outputs/2026-15-positions.png docs/results/example-positions.png
```
If round 15 (Azerbaijan, 2026-09-26) has no qualifying yet the command prints the no-grid note; that is fine for the example. If 2026 rounds have moved on, pick the next round that has not happened.

- [ ] **Step 5: Write `README.md`**

Sections, in order:
1. Title and one-paragraph pitch: what it predicts, how (Elo + Monte Carlo), data source with link to `https://huggingface.co/datasets/tracinginsights/RaceData`.
2. Example output: the `predict` table pasted as a code block, plus the two example PNGs.
3. How it works: four short paragraphs (ratings, one simulated race, Monte Carlo, DNF model), each linking to the matching `docs/*.md`.
4. Backtest results: the held-out season table for tuned params (markdown table), the calibration PNG, one sentence on what the baselines are and one honest sentence on where the model is weak.
5. Install and usage: `uv sync`, then the five commands with one line each.
6. Roadmap: Phases 3, 4, 5 from the spec, one line each.
7. Credits: RaceData / Ergast / Jolpica, and the prior art list from spec section 14.
8. License: MIT.

- [ ] **Step 6: Final check and commit**

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check .
git add README.md f1pred/tuned.json docs/results
git commit -m "docs: README with backtest results and example predictions"
git push origin main
```

---

## Self-review notes

- Spec coverage: 3.1 to 3.5 Task 2 and 3 (safety_cars not needed in Phase 1; circuit DNF rate is computed from results). 5.1 to 5.4 Tasks 4 and 5. 5.5 is Phase 4. 6.1 to 6.4 Tasks 6, 7, 10. 6.5 to 6.7 are Phases 4 and 5. 7 is Phase 3. 8 and 8.1 Tasks 8, 9, 13. 9 Task 12 (season command is Phase 3). 10 Tasks 2, 10, 12. 11 every task. 12 Tasks 1 and 14.
- Deviations from the spec, to be reflected in the spec after this plan lands: the loader uses `huggingface_hub.hf_hub_download` on the CSVs instead of `datasets.load_dataset` (same files, simpler null handling); tuned parameters live in `f1pred/tuned.json` read by `load_params` instead of being written into `config.py`; tuning is coordinate descent rather than a full grid; the driver-race table has an `is_sprint` column and sprint rows.
- Type consistency checked: `Entrant` fields, `RaceForecast` fields, history column names, `RACE_SCORE_COLUMNS`, `RatingState` fields are used identically across Tasks 5, 7, 9, 10, 11, 12, 13.
