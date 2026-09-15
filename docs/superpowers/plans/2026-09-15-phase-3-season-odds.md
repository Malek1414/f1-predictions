# F1 Predictions Phase 3: Season Championship Odds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `f1pred season --season 2026`: simulate every remaining race of a season 2,000 times, award points with that season's rules, and report drivers' and constructors' title probabilities, expected points and final-position distributions, with a chart.

**Architecture:** Extract the per-run finishing-position sampler out of `simulate_race` into `simulate_positions` so the season simulator can chain race draws. A new `sim/points.py` holds the points tables by season. A new `sim/season.py` assembles the remaining calendar, current standings (summed from the driver-race table), per-race entrants, and runs the chained simulation. Report and CLI layers gain a `season` command, table and chart.

**Tech Stack:** unchanged (Python 3.12, uv, pandas, numpy, typer, rich, matplotlib, pytest, ruff).

Spec: `docs/superpowers/specs/2026-09-15-f1-predictions-design.md` section 7. Existing code to read first: `f1pred/sim/race.py`, `f1pred/predict.py`, `f1pred/ratings/history.py`, `f1pred/cli.py`, `tests/conftest.py`.

## Global Constraints

- Same as the Phase 1+2 plan: `uv run` for everything, ruff line length 100, `uv run ruff check . && uv run ruff format --check .` clean before every commit, tests never touch the network, the sample fixture in `tests/fixtures/sample/` is the only real data in tests.
- Every commit message ends with:
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01YJeHtrUFY8CEqTesftaBuB
  ```
- Default season simulation runs: 2,000. Ratings are held fixed within a simulated season (spec 7).
- Points rules (spec 7, verified against the data): race 25-18-15-12-10-8-6-4-2-1 from 2010; a 1-point fastest-lap bonus for a top-10 finisher existed 2019 through 2024 only and is **not** simulated (documented); sprint points 3-2-1 in 2021, 8-7-6-5-4-3-2-1 from 2022.
- Existing public signatures must not change: `simulate_race(entrants, params, n_runs, seed, use_grid) -> RaceForecast` keeps working exactly as before.

## File Structure

```
f1pred/sim/points.py         race_points(season), sprint_points(season), points_for_positions(...)
f1pred/sim/race.py           MODIFY: extract simulate_positions(entrants, params, n_runs, rng, use_grid) -> np.ndarray
f1pred/sim/season.py         RemainingRace, remaining_calendar, current_standings, season_entrants,
                             SeasonForecast, simulate_season
f1pred/report/tables.py      MODIFY: add title_tables(forecast, names)
f1pred/report/charts.py      MODIFY: add title_chart(forecast, names, title, path)
f1pred/cli.py                MODIFY: add `season` command
tests/test_points.py
tests/test_race_sim.py       MODIFY: add simulate_positions tests
tests/test_season_sim.py
tests/test_report.py         MODIFY: add title table/chart tests
tests/test_cli.py            MODIFY: add season command test
docs/season-sim.md
README.md                    MODIFY: season section, example, usage line, roadmap
```

---

### Task 1: Points tables

**Files:**
- Create: `f1pred/sim/points.py`, `tests/test_points.py`

**Interfaces:**
- Produces:
  - `race_points(season: int) -> np.ndarray` length 10 (float), positions 1..10
  - `sprint_points(season: int) -> np.ndarray` (length 3 for 2021, 8 from 2022, empty before 2021)
  - `had_fastest_lap_bonus(season: int) -> bool` (True for 2019..2024)
  - `points_for_positions(positions: np.ndarray, table: np.ndarray) -> np.ndarray`: `positions` is an integer array of any shape with values ≥ 1; returns same shape of points, 0 beyond the table length.

- [ ] **Step 1: Write the failing tests `tests/test_points.py`**

```python
import numpy as np
import pytest

from f1pred.sim.points import (
    had_fastest_lap_bonus,
    points_for_positions,
    race_points,
    sprint_points,
)


def test_race_points_modern():
    np.testing.assert_array_equal(race_points(2026), [25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
    np.testing.assert_array_equal(race_points(2010), race_points(2026))


def test_sprint_points_by_season():
    assert sprint_points(2020).size == 0
    np.testing.assert_array_equal(sprint_points(2021), [3, 2, 1])
    np.testing.assert_array_equal(sprint_points(2024), [8, 7, 6, 5, 4, 3, 2, 1])


@pytest.mark.parametrize("season,expected", [(2018, False), (2019, True), (2024, True), (2025, False)])
def test_fastest_lap_bonus(season, expected):
    assert had_fastest_lap_bonus(season) is expected


def test_points_for_positions_shapes_and_zero_beyond_table():
    pos = np.array([[1, 2, 11, 20], [10, 3, 1, 5]])
    out = points_for_positions(pos, race_points(2025))
    np.testing.assert_array_equal(out, [[25, 18, 0, 0], [1, 15, 25, 10]])
    assert out.dtype == float


def test_points_for_positions_empty_table():
    out = points_for_positions(np.array([1, 2, 3]), sprint_points(2020))
    np.testing.assert_array_equal(out, [0, 0, 0])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_points.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/sim/points.py`**

```python
"""Points rules by season. Spec section 7."""

from __future__ import annotations

import numpy as np

RACE_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
SPRINT_POINTS_2021 = np.array([3, 2, 1], dtype=float)
SPRINT_POINTS_2022 = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)


def race_points(season: int) -> np.ndarray:
    return RACE_POINTS.copy()


def sprint_points(season: int) -> np.ndarray:
    if season < 2021:
        return np.array([], dtype=float)
    if season == 2021:
        return SPRINT_POINTS_2021.copy()
    return SPRINT_POINTS_2022.copy()


def had_fastest_lap_bonus(season: int) -> bool:
    """One bonus point for the fastest lap by a top-10 finisher, 2019 to 2024. Not simulated."""
    return 2019 <= season <= 2024


def points_for_positions(positions: np.ndarray, table: np.ndarray) -> np.ndarray:
    padded = np.concatenate([table.astype(float), np.zeros(1)])
    idx = np.clip(np.asarray(positions) - 1, 0, len(table))
    return padded[idx]
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_points.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add f1pred/sim/points.py tests/test_points.py
git commit -m "feat: points tables by season"
```

---

### Task 2: Extract `simulate_positions`

**Files:**
- Modify: `f1pred/sim/race.py`
- Modify: `tests/test_race_sim.py`

**Interfaces:**
- Produces: `simulate_positions(entrants: list[Entrant], params: ModelParams, n_runs: int, rng: np.random.Generator, use_grid: bool = True) -> tuple[np.ndarray, bool]` returning `(positions, used_grid)` where `positions` has shape `(n_runs, n_entrants)`, integer, each row a permutation of `1..n`.
- `simulate_race` keeps its signature and now calls `simulate_positions` internally; its outputs for a given seed are allowed to change bit-for-bit only if the RNG call order is preserved (it is, if you move the code verbatim).

- [ ] **Step 1: Add failing tests to `tests/test_race_sim.py`**

```python
from f1pred.sim.race import simulate_positions


def test_simulate_positions_rows_are_permutations():
    rng = np.random.default_rng(0)
    positions, used_grid = simulate_positions(_field(), P, 50, rng)
    assert positions.shape == (50, 20) and used_grid
    for row in positions:
        assert sorted(row.tolist()) == list(range(1, 21))


def test_simulate_race_matches_positions_with_same_seed():
    forecast = simulate_race(_field(), P, n_runs=300, seed=11)
    positions, _ = simulate_positions(_field(), P, 300, np.random.default_rng(11))
    p_win = (positions == 1).mean(axis=0)
    np.testing.assert_allclose(p_win, forecast.p_win)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_race_sim.py -v`
Expected: 2 failures with `ImportError` / `cannot import name 'simulate_positions'`.

- [ ] **Step 3: Refactor `f1pred/sim/race.py`**

Replace the body of `simulate_race` with the two functions below (keep `Entrant`, `RaceForecast`, constants unchanged):

```python
def simulate_positions(
    entrants: list[Entrant],
    params: ModelParams,
    n_runs: int,
    rng: np.random.Generator,
    use_grid: bool = True,
) -> tuple[np.ndarray, bool]:
    """One finishing order per run. Returns (positions of shape (n_runs, n), used_grid)."""
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
    return positions, used_grid


def simulate_race(
    entrants: list[Entrant],
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int | None = None,
    use_grid: bool = True,
) -> RaceForecast:
    rng = np.random.default_rng(seed)
    n = len(entrants)
    positions, used_grid = simulate_positions(entrants, params, n_runs, rng, use_grid)
    position_matrix = (
        np.stack([np.bincount(positions[:, i] - 1, minlength=n) for i in range(n)]).astype(float)
        / n_runs
    )
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

- [ ] **Step 4: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all green (existing race-sim tests still pass).

- [ ] **Step 5: Commit**

```bash
git add f1pred/sim/race.py tests/test_race_sim.py
git commit -m "refactor: extract simulate_positions from simulate_race"
```

---

### Task 3: Season simulation

**Files:**
- Create: `f1pred/sim/season.py`, `tests/test_season_sim.py`, `docs/season-sim.md`

**Interfaces:**
- Consumes: raw tables, driver-race table, `RatingState`, `state_for_season`, `strength`, `dnf_probability`, `Entrant`, `simulate_positions`, `race_points`, `sprint_points`, `points_for_positions`.
- Produces:
  - `@dataclass(frozen=True) RemainingRace(race_id: int, round: int, name: str, circuit_id: str, date: pd.Timestamp, has_sprint: bool)`
  - `remaining_calendar(raw: dict, table: pd.DataFrame, season: int) -> list[RemainingRace]`: every race of `season` in `raw["races"]` whose `raceId` has no non-sprint rows in `table`, ordered by round. `has_sprint` is `sprint_date` not null.
  - `current_standings(table: pd.DataFrame, season: int) -> tuple[dict[str, float], dict[str, float], dict[str, str]]`: `(driver_points, constructor_points, driver_to_constructor)` summed from race and sprint rows of that season; `driver_to_constructor` maps each driver to the constructor of their latest session that season. All three empty if the season has no rows.
  - `season_entrants(table: pd.DataFrame, state: RatingState, season: int, race: RemainingRace, params: ModelParams) -> list[Entrant]`: drivers from the latest completed non-sprint session in `table` (any season), `grid=None`, strength from `state_for_season(state, season, params)`, `p_dnf` from `dnf_probability(table, driver, constructor, race.circuit_id, race.date, params)`.
  - `@dataclass SeasonForecast(season: int, driver_ids: list[str], constructor_ids: list[str], p_driver_title: np.ndarray, p_constructor_title: np.ndarray, expected_driver_points: np.ndarray, expected_constructor_points: np.ndarray, driver_position_matrix: np.ndarray, current_driver_points: np.ndarray, current_constructor_points: np.ndarray, n_runs: int, n_remaining: int)` with `.drivers_frame() -> pd.DataFrame` (`driver_id, current_points, expected_points, p_title, p_top3`, sorted by `p_title` desc then `expected_points` desc) and `.constructors_frame()` (`constructor_id, current_points, expected_points, p_title`).
  - `simulate_season(season: int, remaining: list[RemainingRace], entrants_by_race: list[list[Entrant]], driver_points: Mapping[str, float], constructor_points: Mapping[str, float], driver_to_constructor: Mapping[str, str], params: ModelParams, n_runs: int = 2000, seed: int | None = None) -> SeasonForecast`

  Rules inside `simulate_season`:
  - `driver_ids` = union of drivers with current points and drivers in any entrant list, sorted by current points desc then id. `constructor_ids` = union of constructors with points and entrants' constructors, sorted likewise.
  - For each remaining race in order: `positions, _ = simulate_positions(entrants, params, n_runs, rng, use_grid=False)`; add `points_for_positions(positions, race_points(season))` to the driver totals matrix `(n_runs, n_drivers)` at those drivers' columns. If `has_sprint`, first draw a second, independent `simulate_positions` for the same entrants and add `points_for_positions(..., sprint_points(season))`.
  - Constructor totals per run: sum of the totals of the drivers currently mapped to that constructor (entrant constructor if the driver is an entrant, else `driver_to_constructor`) plus, for a constructor, any points held by drivers no longer mapped anywhere (guaranteed covered by `driver_to_constructor`).
    Implementation: build a `(n_drivers, n_constructors)` 0/1 matrix `M` from the mapping, then `constructor_totals = driver_totals @ M`.
  - Champion per run: `argmax` along axis 1 (ties go to the lower index, which is the driver higher in the current standings). `p_title = bincount / n_runs`.
  - `driver_position_matrix[i, k]` = fraction of runs where driver i finished the season in position k+1, using `argsort(-totals, kind="stable")`.
  - With zero remaining races, totals are just the current points and the leader gets probability 1.

- [ ] **Step 1: Write the failing tests `tests/test_season_sim.py`**

```python
import numpy as np
import pandas as pd
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import replay
from f1pred.sim.race import Entrant
from f1pred.sim.season import (
    RemainingRace,
    current_standings,
    remaining_calendar,
    season_entrants,
    simulate_season,
)

P = DEFAULT_PARAMS


def _race(i, has_sprint=False):
    return RemainingRace(1000 + i, 20 + i, f"Race {i}", "c", pd.Timestamp("2030-01-01"), has_sprint)


def _entrants():
    return [Entrant(f"d{i}", f"t{i // 2}", 3000.0, None, 0.0) for i in range(4)]


def test_unassailable_lead_gives_probability_one():
    remaining = [_race(0), _race(1)]
    pts = {"d0": 500.0, "d1": 100.0, "d2": 0.0, "d3": 0.0}
    cons = {"t0": 600.0, "t1": 0.0}
    mapping = {d: f"t{int(d[1]) // 2}" for d in pts}
    f = simulate_season(2025, remaining, [_entrants(), _entrants()], pts, cons, mapping, P, n_runs=200, seed=0)
    assert f.p_driver_title[f.driver_ids.index("d0")] == 1.0
    assert f.p_constructor_title[f.constructor_ids.index("t0")] == 1.0
    assert f.n_remaining == 2
    assert f.expected_driver_points[f.driver_ids.index("d0")] > 500.0


def test_no_remaining_races_returns_current_leader():
    pts = {"a": 10.0, "b": 20.0}
    f = simulate_season(2024, [], [], pts, {"x": 30.0}, {"a": "x", "b": "x"}, P, n_runs=10, seed=0)
    assert f.driver_ids[0] == "b" and f.p_driver_title[0] == 1.0
    assert f.p_constructor_title[0] == 1.0
    np.testing.assert_array_equal(f.expected_driver_points, [20.0, 10.0])


def test_points_awarded_per_race_and_sprint():
    # One driver far stronger than the rest: wins every race and sprint -> 25 + 8 per sprint weekend.
    ents = [Entrant("s", "t0", 6000.0, None, 0.0)] + [
        Entrant(f"d{i}", f"t{1 + i // 2}", 3000.0, None, 0.0) for i in range(3)
    ]
    remaining = [_race(0, has_sprint=True), _race(1)]
    pts = {e.driver_id: 0.0 for e in ents}
    mapping = {e.driver_id: e.constructor_id for e in ents}
    f = simulate_season(2025, remaining, [ents, ents], pts, {}, mapping, P, n_runs=50, seed=1)
    assert f.expected_driver_points[f.driver_ids.index("s")] == pytest.approx(25 + 8 + 25)
    assert f.expected_constructor_points[f.constructor_ids.index("t0")] == pytest.approx(58)


def test_position_matrix_and_frames():
    remaining = [_race(0)]
    pts = {"d0": 0.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}
    mapping = {d: f"t{int(d[1]) // 2}" for d in pts}
    f = simulate_season(2025, remaining, [_entrants()], pts, {}, mapping, P, n_runs=400, seed=2)
    np.testing.assert_allclose(f.driver_position_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(f.driver_position_matrix.sum(axis=0), 1.0)
    df = f.drivers_frame()
    assert list(df.columns) == ["driver_id", "current_points", "expected_points", "p_title", "p_top3"]
    assert df.p_title.is_monotonic_decreasing
    assert f.p_driver_title.sum() == pytest.approx(1.0)
    cdf = f.constructors_frame()
    assert list(cdf.columns) == ["constructor_id", "current_points", "expected_points", "p_title"]


def test_retired_driver_keeps_points_but_scores_no_more():
    remaining = [_race(0)]
    pts = {"old": 30.0, "d0": 0.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}
    mapping = {"old": "t9", **{d: f"t{int(d[1]) // 2}" for d in pts if d != "old"}}
    f = simulate_season(2025, remaining, [_entrants()], pts, {"t9": 30.0}, mapping, P, n_runs=100, seed=3)
    assert f.expected_driver_points[f.driver_ids.index("old")] == 30.0
    assert "t9" in f.constructor_ids
    assert f.expected_constructor_points[f.constructor_ids.index("t9")] == 30.0


def test_remaining_calendar_and_standings_on_sample(raw_sample, driver_race):
    last_id = int(driver_race[driver_race.season == 2024].race_id.max())
    table = driver_race[driver_race.race_id != last_id]
    remaining = remaining_calendar(raw_sample, table, 2024)
    assert [r.round for r in remaining] == [24]
    assert remaining[0].name == "Abu Dhabi Grand Prix" and not remaining[0].has_sprint
    assert remaining_calendar(raw_sample, driver_race, 2024) == []
    full = remaining_calendar(raw_sample, driver_race[driver_race.season == 2023], 2024)
    assert len(full) == 24 and sum(r.has_sprint for r in full) == 6

    drivers, cons, mapping = current_standings(driver_race, 2024)
    assert drivers["max_verstappen"] == pytest.approx(437.0)
    assert cons["mclaren"] == pytest.approx(666.0)
    assert mapping["max_verstappen"] == "red_bull"
    assert current_standings(driver_race, 1999) == ({}, {}, {})


def test_season_entrants_on_sample(driver_race):
    _, state = replay(driver_race, P)
    race = RemainingRace(999, 1, "Next", "bahrain", pd.Timestamp("2025-03-16"), False)
    ents = season_entrants(driver_race, state, 2025, race, P)
    assert len(ents) == 20 and all(e.grid is None for e in ents)
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    assert e.strength == pytest.approx(
        1500 + (state.driver["max_verstappen"] - 1500) * 0.9
        + 1500 + (state.constructor["red_bull"] - 1500) * 0.6
    )
    assert 0.01 <= e.p_dnf <= 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_season_sim.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `f1pred/sim/season.py`**

```python
"""Chain race simulations over the remaining calendar to get championship odds. Spec section 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.config import ModelParams
from f1pred.ratings.history import RatingState, state_for_season, strength
from f1pred.sim.dnf import dnf_probability
from f1pred.sim.points import points_for_positions, race_points, sprint_points
from f1pred.sim.race import Entrant, simulate_positions


@dataclass(frozen=True)
class RemainingRace:
    race_id: int
    round: int
    name: str
    circuit_id: str
    date: pd.Timestamp
    has_sprint: bool


@dataclass
class SeasonForecast:
    season: int
    driver_ids: list[str]
    constructor_ids: list[str]
    p_driver_title: np.ndarray
    p_constructor_title: np.ndarray
    expected_driver_points: np.ndarray
    expected_constructor_points: np.ndarray
    driver_position_matrix: np.ndarray
    current_driver_points: np.ndarray
    current_constructor_points: np.ndarray
    n_runs: int
    n_remaining: int

    def drivers_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "driver_id": self.driver_ids,
                "current_points": self.current_driver_points,
                "expected_points": self.expected_driver_points,
                "p_title": self.p_driver_title,
                "p_top3": self.driver_position_matrix[:, :3].sum(axis=1),
            }
        )
        return df.sort_values(
            ["p_title", "expected_points"], ascending=False, kind="stable"
        ).reset_index(drop=True)

    def constructors_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "constructor_id": self.constructor_ids,
                "current_points": self.current_constructor_points,
                "expected_points": self.expected_constructor_points,
                "p_title": self.p_constructor_title,
            }
        )
        return df.sort_values(
            ["p_title", "expected_points"], ascending=False, kind="stable"
        ).reset_index(drop=True)


def remaining_calendar(raw: dict, table: pd.DataFrame, season: int) -> list[RemainingRace]:
    races = raw["races"].merge(raw["circuits"][["circuitId", "circuitRef"]], on="circuitId")
    races = races[races["year"] == season].sort_values("round")
    done = set(table.loc[~table["is_sprint"], "race_id"].astype(int))
    out = []
    for r in races.itertuples():
        if int(r.raceId) in done:
            continue
        out.append(
            RemainingRace(
                race_id=int(r.raceId),
                round=int(r.round),
                name=str(r.name),
                circuit_id=str(r.circuitRef),
                date=pd.Timestamp(r.date),
                has_sprint=pd.notna(r.sprint_date),
            )
        )
    return out


def current_standings(
    table: pd.DataFrame, season: int
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    rows = table[table["season"] == season]
    if rows.empty:
        return {}, {}, {}
    driver_points = rows.groupby("driver_id")["points"].sum().to_dict()
    constructor_points = rows.groupby("constructor_id")["points"].sum().to_dict()
    latest = rows.sort_values("date").groupby("driver_id")["constructor_id"].last()
    return driver_points, constructor_points, latest.to_dict()


def season_entrants(
    table: pd.DataFrame,
    state: RatingState,
    season: int,
    race: RemainingRace,
    params: ModelParams,
) -> list[Entrant]:
    season_state = state_for_season(state, season, params)
    races = table[~table["is_sprint"]]
    latest_id = races.sort_values("date")["race_id"].iloc[-1]
    latest = races[races["race_id"] == latest_id]
    return [
        Entrant(
            driver_id=r.driver_id,
            constructor_id=r.constructor_id,
            strength=strength(season_state, r.driver_id, r.constructor_id, params),
            grid=None,
            p_dnf=dnf_probability(
                table, r.driver_id, r.constructor_id, race.circuit_id, race.date, params
            ),
            low_confidence=season_state.driver_races.get(r.driver_id, 0)
            < params.min_races_for_confidence,
        )
        for r in latest.itertuples(index=False)
    ]


def _ordered_ids(points: Mapping[str, float], extra: set[str]) -> list[str]:
    ids = set(points) | extra
    return sorted(ids, key=lambda i: (-points.get(i, 0.0), i))


def simulate_season(
    season: int,
    remaining: list[RemainingRace],
    entrants_by_race: list[list[Entrant]],
    driver_points: Mapping[str, float],
    constructor_points: Mapping[str, float],
    driver_to_constructor: Mapping[str, str],
    params: ModelParams,
    n_runs: int = 2000,
    seed: int | None = None,
) -> SeasonForecast:
    if len(remaining) != len(entrants_by_race):
        raise ValueError("remaining and entrants_by_race must have the same length")
    rng = np.random.default_rng(seed)

    entrant_drivers = {e.driver_id for ents in entrants_by_race for e in ents}
    entrant_teams = {e.constructor_id for ents in entrants_by_race for e in ents}
    driver_ids = _ordered_ids(driver_points, entrant_drivers)
    constructor_ids = _ordered_ids(constructor_points, entrant_teams)
    d_idx = {d: i for i, d in enumerate(driver_ids)}
    c_idx = {c: i for i, c in enumerate(constructor_ids)}

    mapping = dict(driver_to_constructor)
    for ents in entrants_by_race:
        for e in ents:
            mapping[e.driver_id] = e.constructor_id
    M = np.zeros((len(driver_ids), len(constructor_ids)))
    for d, c in mapping.items():
        if d in d_idx and c in c_idx:
            M[d_idx[d], c_idx[c]] = 1.0

    current_d = np.array([driver_points.get(d, 0.0) for d in driver_ids], dtype=float)
    totals = np.tile(current_d, (n_runs, 1))
    race_table, sprint_table = race_points(season), sprint_points(season)
    for race, ents in zip(remaining, entrants_by_race, strict=True):
        cols = np.array([d_idx[e.driver_id] for e in ents])
        if race.has_sprint and sprint_table.size:
            pos, _ = simulate_positions(ents, params, n_runs, rng, use_grid=False)
            totals[:, cols] += points_for_positions(pos, sprint_table)
        pos, _ = simulate_positions(ents, params, n_runs, rng, use_grid=False)
        totals[:, cols] += points_for_positions(pos, race_table)

    # Constructors: points already held (which may include drivers no longer mapped) plus
    # simulated points routed through the driver -> constructor map.
    current_c = np.array([constructor_points.get(c, 0.0) for c in constructor_ids], dtype=float)
    simulated = totals - current_d
    c_totals = current_c + simulated @ M

    champion = np.argmax(totals, axis=1)
    c_champion = np.argmax(c_totals, axis=1)
    order = np.argsort(-totals, axis=1, kind="stable")
    final_pos = np.empty_like(order)
    np.put_along_axis(
        final_pos, order, np.arange(1, len(driver_ids) + 1)[None, :].repeat(n_runs, 0), axis=1
    )
    n_d = len(driver_ids)
    position_matrix = (
        np.stack([np.bincount(final_pos[:, i] - 1, minlength=n_d) for i in range(n_d)]).astype(float)
        / n_runs
    )
    return SeasonForecast(
        season=season,
        driver_ids=driver_ids,
        constructor_ids=constructor_ids,
        p_driver_title=np.bincount(champion, minlength=n_d) / n_runs,
        p_constructor_title=np.bincount(c_champion, minlength=len(constructor_ids)) / n_runs,
        expected_driver_points=totals.mean(axis=0),
        expected_constructor_points=c_totals.mean(axis=0),
        driver_position_matrix=position_matrix,
        current_driver_points=current_d,
        current_constructor_points=current_c,
        n_runs=n_runs,
        n_remaining=len(remaining),
    )
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_season_sim.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: 7 passed. If the 2024 standings numbers differ, check the fixture: Verstappen 437 and McLaren 666 are the official 2024 totals including sprints; the fixture is real data.

- [ ] **Step 5: Write `docs/season-sim.md`** (about 250 words): why chaining races is the same Monte Carlo idea, why ratings are frozen inside a simulated season (and what that under-estimates: form swings, upgrades), how points are awarded and why fastest-lap bonus is skipped, why retired drivers keep points, how the constructor total is computed, and how to read the "final position" heatmap.

- [ ] **Step 6: Commit**

```bash
git add f1pred/sim/season.py tests/test_season_sim.py docs/season-sim.md
git commit -m "feat: chained season simulation for championship odds"
```

---

### Task 4: Title tables and chart

**Files:**
- Modify: `f1pred/report/tables.py`, `f1pred/report/charts.py`, `tests/test_report.py`

**Interfaces:**
- Produces:
  - `title_tables(forecast: SeasonForecast, names: Mapping[str, str], top: int = 10) -> tuple[Table, Table]`: drivers table columns `#, Driver, Points now, Exp. points, Title, Top 3`; constructors table `#, Constructor, Points now, Exp. points, Title`.
  - `title_chart(forecast: SeasonForecast, names: Mapping[str, str], title: str, path: Path) -> Path`: two horizontal bar panels side by side (drivers with `p_title > 0.5%`, constructors with `p_title > 0.5%`), percent labels.

- [ ] **Step 1: Add failing tests to `tests/test_report.py`**

```python
from f1pred.report.charts import title_chart
from f1pred.report.tables import title_tables
from f1pred.sim.season import RemainingRace, simulate_season


def _season_forecast():
    ents = [Entrant(f"d{i}", f"t{i // 2}", 3000.0 + 40 * i, None, 0.05) for i in range(6)]
    remaining = [RemainingRace(1, 20, "R", "c", pd.Timestamp("2030-01-01"), True)]
    pts = {e.driver_id: 10.0 * i for i, e in enumerate(ents)}
    mapping = {e.driver_id: e.constructor_id for e in ents}
    return simulate_season(2025, remaining, [ents], pts, {}, mapping, DEFAULT_PARAMS, n_runs=100, seed=0)


def test_title_tables():
    drivers, cons = title_tables(_season_forecast(), NAMES, top=3)
    text = _render(drivers)
    assert "Driver 5" in text and "Title" in text and "%" in text
    assert "t2" in _render(cons)


def test_title_chart_writes_png(tmp_path):
    assert title_chart(_season_forecast(), NAMES, "2025", tmp_path / "title.png").stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_report.py -v`
Expected: 2 failures with `ImportError`.

- [ ] **Step 3: Add to `f1pred/report/tables.py`**

```python
from f1pred.sim.season import SeasonForecast


def title_tables(
    forecast: SeasonForecast, names: Mapping[str, str], top: int = 10
) -> tuple[Table, Table]:
    drivers = Table(
        title=f"{forecast.season} drivers' championship "
        f"({forecast.n_remaining} races left, {forecast.n_runs:,} runs)"
    )
    for col, justify in [
        ("#", "right"),
        ("Driver", "left"),
        ("Points now", "right"),
        ("Exp. points", "right"),
        ("Title", "right"),
        ("Top 3", "right"),
    ]:
        drivers.add_column(col, justify=justify)
    for i, r in enumerate(forecast.drivers_frame().head(top).itertuples(index=False), start=1):
        drivers.add_row(
            str(i),
            names.get(r.driver_id, r.driver_id),
            f"{r.current_points:.0f}",
            f"{r.expected_points:.0f}",
            _pct(r.p_title),
            _pct(r.p_top3),
        )
    cons = Table(title=f"{forecast.season} constructors' championship")
    for col, justify in [
        ("#", "right"),
        ("Constructor", "left"),
        ("Points now", "right"),
        ("Exp. points", "right"),
        ("Title", "right"),
    ]:
        cons.add_column(col, justify=justify)
    for i, r in enumerate(forecast.constructors_frame().head(top).itertuples(index=False), start=1):
        cons.add_row(
            str(i),
            r.constructor_id,
            f"{r.current_points:.0f}",
            f"{r.expected_points:.0f}",
            _pct(r.p_title),
        )
    return drivers, cons
```

- [ ] **Step 4: Add to `f1pred/report/charts.py`**

```python
from f1pred.sim.season import SeasonForecast  # noqa: E402

MIN_SHOWN = 0.005


def title_chart(
    forecast: SeasonForecast, names: Mapping[str, str], title: str, path: Path
) -> Path:
    d = forecast.drivers_frame()
    d = d[d.p_title > MIN_SHOWN]
    c = forecast.constructors_frame()
    c = c[c.p_title > MIN_SHOWN]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 0.4 * max(len(d), len(c), 3) + 1.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    for ax, frame, key, label in [
        (ax1, d, "driver_id", "Drivers"),
        (ax2, c, "constructor_id", "Constructors"),
    ]:
        labels = [names.get(i, i) for i in frame[key]][::-1]
        values = (100 * frame.p_title)[::-1]
        ax.barh(labels, values, color="#e10600")
        for i, v in enumerate(values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Title probability (%)")
        ax.set_title(f"{label} ({forecast.n_remaining} races left)")
    fig.suptitle(title)
    return _save(fig, path)
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/test_report.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add f1pred/report tests/test_report.py
git commit -m "feat: championship tables and chart"
```

---

### Task 5: `season` command

**Files:**
- Modify: `f1pred/cli.py`, `tests/test_cli.py`

**Interfaces:**
- `f1pred season --season 2026 [--runs 2000] [--seed 0] [--cache-dir DIR] [--out DIR]`
  - Loads cached tables and ratings (same helpers as `predict`).
  - `remaining = remaining_calendar(raw, table, season)`; if empty, prints "Season {season} is complete" and still prints the tables (title probability 1 for the actual champion) with `n_remaining=0`.
  - `entrants_by_race = [season_entrants(table, state, season, r, params) for r in remaining]`.
  - `driver_points, constructor_points, mapping = current_standings(table, season)`.
  - Names: from the driver-race table (all seasons).
  - Prints both tables, writes `outputs/<season>-title.png`, `outputs/<season>-drivers.csv`, `outputs/<season>-constructors.csv`.
  - Exit 2 with the valid-seasons list if `raw["races"]` has no rows for the season.

- [ ] **Step 1: Add failing tests to `tests/test_cli.py`**

```python
def test_season_command_on_completed_season(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        ["season", "--season", "2024", "--runs", "50", "--cache-dir", str(cache_dir), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    assert "complete" in r.output.lower()
    assert "Max Verstappen" in r.output and "100.0%" in r.output
    assert (tmp_path / "2024-title.png").exists()
    drivers = pd.read_csv(tmp_path / "2024-drivers.csv")
    assert drivers.iloc[0].driver_id == "max_verstappen"


def test_season_command_with_remaining_races(cache_dir, tmp_path, driver_race):
    # Cache without the last two 2024 races so two rounds remain.
    partial = tmp_path / "cache"
    partial.mkdir()
    for p in cache_dir.glob("*.parquet"):
        (partial / p.name).write_bytes(p.read_bytes())
    ids = sorted(driver_race[driver_race.season == 2024].race_id.unique())[-2:]
    driver_race[~driver_race.race_id.isin(ids)].to_parquet(partial / "driver_race.parquet", index=False)
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(partial)])
    r = runner.invoke(
        app,
        ["season", "--season", "2024", "--runs", "100", "--cache-dir", str(partial), "--out", str(tmp_path)],
    )
    assert r.exit_code == 0, r.output
    assert "2 races left" in r.output
    cons = pd.read_csv(tmp_path / "2024-constructors.csv")
    assert set(cons.constructor_id) >= {"mclaren", "ferrari", "red_bull"}
    assert cons.p_title.sum() == pytest.approx(1.0)


def test_season_unknown_season_exits_2(cache_dir):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(app, ["season", "--season", "1980", "--cache-dir", str(cache_dir)])
    assert r.exit_code == 2
```
Add `import pytest` at the top of `tests/test_cli.py` if missing.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v -k season`
Expected: 3 failures (`No such command 'season'`).

- [ ] **Step 3: Add the command to `f1pred/cli.py`**

```python
from f1pred.report.charts import title_chart
from f1pred.report.tables import title_tables
from f1pred.sim.season import (
    current_standings,
    remaining_calendar,
    season_entrants,
    simulate_season,
)


@app.command()
def season(
    season: int = typer.Option(..., "--season"),
    runs: int = typer.Option(2000, "--runs"),
    seed: int = typer.Option(0, "--seed"),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Drivers' and constructors' title odds for the rest of a season."""
    params = load_params()
    raw, table = _load_cached(cache_dir)
    _, state = _load_ratings(cache_dir)
    seasons_available = sorted(raw["races"]["year"].unique().tolist())
    if season not in seasons_available:
        console.print(
            f"[red]No calendar for {season}.[/red] Seasons available: "
            f"{seasons_available[0]}-{seasons_available[-1]}"
        )
        raise typer.Exit(2)
    remaining = remaining_calendar(raw, table, season)
    if not remaining:
        console.print(f"[yellow]Season {season} is complete; showing final standings.[/yellow]")
    entrants_by_race = [season_entrants(table, state, season, r, params) for r in remaining]
    driver_points, constructor_points, mapping = current_standings(table, season)
    forecast = simulate_season(
        season,
        remaining,
        entrants_by_race,
        driver_points,
        constructor_points,
        mapping,
        params,
        n_runs=runs,
        seed=seed,
    )
    names = dict(zip(table.driver_id, table.driver_name, strict=True))
    drivers, cons = title_tables(forecast, names)
    console.print(drivers)
    console.print(cons)
    out.mkdir(parents=True, exist_ok=True)
    forecast.drivers_frame().to_csv(out / f"{season}-drivers.csv", index=False)
    forecast.constructors_frame().to_csv(out / f"{season}-constructors.csv", index=False)
    chart = title_chart(forecast, names, f"{season} championship odds", out / f"{season}-title.png")
    console.print(f"Chart: {chart}")
```

- [ ] **Step 4: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add f1pred/cli.py tests/test_cli.py
git commit -m "feat: f1pred season command for championship odds"
```

---

### Task 6: Run on real data and update README

Needs network access only if `data/cache/` is empty (`uv run f1pred data update`).

- [ ] **Step 1: Run**

```bash
uv run f1pred ratings build
uv run f1pred season --season 2026
mkdir -p docs/results && cp outputs/2026-title.png docs/results/example-title.png
```
Sanity checks: the drivers' title probabilities sum to about 1, the current points leader has the highest or second-highest probability, drivers with zero points and low ratings are near 0, constructors' table has 11 teams.

- [ ] **Step 2: Update `README.md`**

- Add a "Championship odds" subsection under "Example output" with the drivers' table (top 8 rows) as a code block and `docs/results/example-title.png`.
- Add a "Season simulation" paragraph under "How it works" linking `docs/season-sim.md`.
- Add `uv run f1pred season --season 2026` to the usage list.
- Move Phase 3 from the roadmap to done; keep Phases 4 and 5 listed.

- [ ] **Step 3: Final check, commit, push**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
git add README.md docs/results/example-title.png
git commit -m "docs: season odds in README"
git push origin HEAD:main
```

---

## Self-review notes

- Spec 7 coverage: remaining calendar (Task 3), current standings from the table (Task 3; the spec's `driver_standings` table is not needed because summing the driver-race table reproduces it), points rules keyed by season (Task 1), chained simulation with fixed ratings (Task 3), outputs: title probability, expected points, final-position distribution (Task 3), `season` command and chart (Tasks 4, 5), doc note (Task 3).
- `simulate_positions` is the only change to existing code paths; Task 2 keeps `simulate_race` results identical for a given seed.
- Constructor totals: `current_c + (totals - current_d) @ M` keeps points already scored by a constructor even when a driver has since moved teams, and routes future points through the current mapping.
