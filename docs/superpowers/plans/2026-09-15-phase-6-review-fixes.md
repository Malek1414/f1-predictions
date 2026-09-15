# F1 Predictions Phase 6: Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the defects found by the post-Phase-4 code review, then re-run the data build, tuning and backtests so the README numbers reflect the corrected model.

**Architecture:** No new modules. Targeted changes in `data/frame.py`, `sim/dnf.py`, `backtest/run.py`, `ratings/history.py`, `predict.py`, `sim/season.py`, `cli.py`, plus tests. Every fix has a regression test that fails before and passes after.

**Tech Stack:** unchanged.

Read first: the review findings below, then the current code of each file named.

## Global Constraints

- Same as earlier plans (uv, ruff 100, offline tests, commit trailer):
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01YJeHtrUFY8CEqTesftaBuB
  ```
- Work on the code as it exists after Phase 5 (profile features, `lap1_position`, `use_profile`), not as any earlier plan described it.
- The sample fixture (2023 and 2024) already has `position` null exactly where `positionText` is non-numeric, so existing fixture-based expectations do not change from fix 1.

## Review findings being fixed

1. **CONFIRMED, severe.** `data/frame.py`: "classified" is derived from `position`, but from 2025 the upstream CSV fills `position` with `positionOrder` for retirements (`positionText` = `R`, `W`, `D`, ...). Verified: 2025 has 0 null positions but 55 non-numeric `positionText`; 2026 has 0 vs 60. Effect: every 2025 to 2026 retirement is rated as a finish, `dnf` is False for all 2025 to 2026 rows, the global DNF rate collapses toward 0, and the current `p_dnf` values are about 1 to 3% instead of about 10 to 15%.
2. **CONFIRMED.** `sim/dnf.py`: the circuit factor divides the circuit's all-time DNF rate by the recent (last 400 rows) global rate. Reliability has improved over the years, so the ratio is inflated (median 1.7, 11 of 35 circuits pinned at the 2.0 clip) and circuit discrimination is lost.
3. **CONFIRMED.** Stale ratings history: after `f1pred data update` without `f1pred ratings build`, `predict` and `backtest` crash with a bare `KeyError` from `hist.loc[driver_id]`.
4. **CONFIRMED.** Missing Phase 4 or 5 columns (`is_wet`, `track_type`, `lap1_position`) make `replay`, `entrants_for_past_race` and `run_backtest` silently fall back to dry / mixed / no-profile behaviour instead of failing loudly.
5. **PLAUSIBLE, design.** The backtest uses the race's observed rainfall (`rain_probability = 1.0` on wet races), which is information not available before the race. The ablation's "weather" gain therefore overstates what a forecast could deliver.
6. **CONFIRMED, minor.** `sim/dnf.py` evidence count `n = (n_d + n_c) / 2` mixes a 20-row driver count with a 40-row constructor count.
7. **PLAUSIBLE, minor.** `data/frame.py`: unknown non-classified statuses default to `mechanical`; e.g. `Underweight` (a disqualification) counts as a mechanical failure.
8. **PLAUSIBLE, minor.** `predict.py`: with no qualifying data, the lineup is silently copied from the last completed race; the note should name that race.

---

### Task 1: Classification from `positionText`; unknown statuses are `other`

**Files:** `f1pred/data/frame.py`, `tests/test_frame.py`

**Rules:**
- `classified = pd.to_numeric(df["positionText"], errors="coerce").notna()`; `position = pd.to_numeric(df["positionText"], errors="coerce").astype("Int64")`. Ignore the `position` column entirely.
- `classify_dnf` becomes whitelist-based with three sets:
  - `ACCIDENT_STATUSES` (unchanged): `Accident, Collision, Spun off, Collision damage, Damage`.
  - `MECHANICAL_STATUSES`: `Engine, Gearbox, Transmission, Clutch, Hydraulics, Electrical, Radiator, Suspension, Brakes, Differential, Overheating, Mechanical, Tyre, Puncture, Driveshaft, Fuel pressure, Front wing, Water pressure, Refuelling, Wheel, Throttle, Steering, Technical, Electronics, Broken wing, Heat shield fire, Exhaust, Oil leak, Wheel rim, Water leak, Fuel pump, Track rod, Oil pressure, Engine fire, Engine misfire, Tyre puncture, Out of fuel, Wheel nut, Pneumatics, Handling, Rear wing, Fire, Wheel bearing, Fuel system, Oil line, Fuel rig, Launch control, Fuel, Power loss, Vibrations, Drivetrain, Ignition, Chassis, Battery, Stalled, Halfshaft, Crankshaft, Alternator, Oil pump, Fuel leak, Fuel pipe, Power Unit, ERS, Brake duct, Seat, Undertray, Cooling system, Spark plugs, Turbo, CV joint, Water pump, Debris`.
  - Everything else that is not classified (for example `Disqualified, Retired, Withdrew, Not classified, Did not qualify, Did not prequalify, Excluded, 107% Rule, Underweight, Injured, Injury, Illness, Fatal accident, Safety concerns, Not restarted, Driver unwell, Eye injury, Physical, Safety`) is `other`. A status in none of the lists is `other` and triggers a one-time `log.warning` naming it, so new upstream statuses surface.

- [ ] Failing tests:
```python
def test_position_comes_from_position_text():
    raw = {...}  # build a minimal raw dict with one race and two results rows:
    # row A: position=15, positionText="R", statusId=3 (Accident)
    # row B: position=1, positionText="1", statusId=1
    t = build_driver_race_table(raw)
    a = t[t.driver_id == "a"].iloc[0]
    assert pd.isna(a.position) and a.dnf and a.dnf_kind == "accident"
    assert t[t.driver_id == "b"].iloc[0].position == 1


@pytest.mark.parametrize("status,expected", [("Underweight", "other"), ("Excluded", "other"), ("Power Unit", "mechanical"), ("Some new thing", "other")])
def test_unknown_or_administrative_statuses_are_other(status, expected):
    assert classify_dnf(status, False) == expected
```
Build the minimal raw dict inline in the test (races, results, drivers, constructors, circuits, status, qualifying, sprint_results frames with the columns `frame.py` reads).

- [ ] Implement, full suite, lint, commit: `fix: classify finishers from positionText, administrative statuses are not DNFs`

---

### Task 2: DNF circuit factor on one time base; evidence count in races

**Files:** `f1pred/sim/dnf.py`, `tests/test_dnf.py`

**Rules:**
- `_circuit_factor` divides the circuit's rate by the **all-time** global rate over the same `past` frame (`past["dnf"].mean()`), not the recent-window rate. Signature becomes `_circuit_factor(past, circuit_id, params)`.
- Evidence count `n = (n_d + n_c / 2) / 2` (constructor rows are two per race).

- [ ] Failing tests:
```python
def test_circuit_factor_uses_all_time_global():
    # 400 recent rows with 5% DNF after 400 older rows with 25% DNF, all at circuit "c1";
    # the circuit rate equals the all-time global rate, so the factor must be 1.0 and p_dnf equals the recent rate.
    ...
    assert p == pytest.approx(recent_rate_after_shrink)  # not 1.7x it


def test_evidence_count_in_races():
    # driver with 20 races all DNF, constructor with 40 rows all DNF, global 25% -> n = 20 -> weight 20/30
    assert p == pytest.approx(0.25 + 0.75 * 20 / 30)
```
Update `test_crash_prone_driver_shrinks_toward_global` accordingly (its `n` is now 20, not 20; check the arithmetic: `n_d=20`, `n_c=20` rows in that fixture (one car per team), so `n = (20 + 10)/2 = 15` → weight `15/25`). Fix the expected value in the test with a comment.

- [ ] Implement, full suite, lint, commit: `fix: DNF circuit factor on a single time base, evidence in races`

---

### Task 3: Loud failures for stale ratings and missing columns

**Files:** `f1pred/backtest/run.py`, `f1pred/ratings/history.py`, `f1pred/cli.py`, `f1pred/predict.py`, `tests/test_backtest.py`, `tests/test_history.py`, `tests/test_cli.py`

**Rules:**
- New exception `f1pred.backtest.run.StaleRatingsError(Exception)`. `entrants_for_past_race` raises it (message names the race and the missing driver, and says to run `f1pred ratings build`) when a driver has no history row. `run_backtest` raises it when a race in range has no history rows at all.
- `cli._load_ratings(cache_dir)` also loads the driver-race table's latest date and the history's latest date; if the table is newer, `_fail("Ratings are older than the data (table through X, ratings through Y). Run `f1pred ratings build`.", 1)`.
- `history.replay` raises `ValueError("table is missing columns: ...")` if any of `is_wet, track_type, lap1_position` is absent. Remove the silent fallbacks in `replay`, `entrants_for_past_race` (conditional columns) and `run_backtest` (`is_wet`). Update `tests/test_history.py` fixtures (`tiny_table`, the sprint table in `test_sprint_uses_half_k`) to include the three columns; convert `test_replay_without_condition_columns_is_all_dry_mixed` into `test_replay_requires_condition_columns` expecting `ValueError`. Update any other test that builds a table without those columns (`tests/test_profile.py` already includes them; check `tests/test_dnf.py` builds only what `dnf_probability` reads and does not call `replay`).

- [ ] Failing tests:
```python
def test_stale_history_raises_clear_error(driver_race, sample_history):
    last = int(driver_race[driver_race.season == 2024].race_id.max())
    hist = sample_history[sample_history.race_id != last]
    with pytest.raises(StaleRatingsError, match="ratings build"):
        run_backtest(driver_race, hist, [2024], P, n_runs=10)


def test_cli_predict_with_stale_ratings_exits_1(cache_dir, tmp_path, driver_race):
    # build ratings on a table missing the last race, then restore the full table
    ...
    assert r.exit_code == 1 and "ratings build" in r.output
```

- [ ] Implement, full suite, lint, commit: `fix: fail loudly on stale ratings and missing feature columns`

---

### Task 4: Pre-race rain in the backtest

**Files:** `f1pred/backtest/run.py`, `f1pred/cli.py`, `tests/test_backtest.py`, `tests/test_cli.py`, `docs/backtesting.md`, `docs/weather-and-track.md`

**Rules:**
- `run_backtest(..., observed_rain: bool = False)`. Default: `rain_probability = circuit_wet_rate(table, circuit_id, race_date)` when `params.use_weather` (pre-race information only). `observed_rain=True` keeps the old behaviour (1.0 / 0.0 from `is_wet`) and is labelled as an upper bound.
- `ablation_backtest(..., observed_rain=False)` passes it through; the CLI gets `--observed-rain` on `backtest`. The ablation CSV gains a column `rain_mode` with `"historical"` or `"observed"`.
- Docs: one paragraph each explaining why the default is the honest one.

- [ ] Failing tests:
```python
def test_backtest_default_rain_is_pre_race(driver_race, sample_history, monkeypatch):
    import f1pred.backtest.run as run_mod
    seen = []
    orig = run_mod.simulate_race
    def spy(*a, **k):
        seen.append(k.get("rain_probability"))
        return orig(*a, **k)
    monkeypatch.setattr(run_mod, "simulate_race", spy)
    run_backtest(driver_race, sample_history, [2024], P, n_runs=20)
    assert all(0.0 <= r < 1.0 for r in seen) and any(0.0 < r for r in seen)
    seen.clear()
    run_backtest(driver_race, sample_history, [2024], P, n_runs=20, observed_rain=True)
    assert set(seen) <= {0.0, 1.0} and 1.0 in seen
```

- [ ] Implement, full suite, lint, commit: `fix: backtest uses pre-race rain probability by default`

---

### Task 5: Name the carried-over lineup

**Files:** `f1pred/predict.py`, `f1pred/sim/season.py`, `tests/test_predict.py`, `tests/test_season_sim.py`

**Rules:**
- `predict.NO_QUALIFYING_NOTE` becomes a format: `"No qualifying data yet; lineup carried over from {race_name} {season} and grid ignored"`; `_future_entrants` fills it with the source race's name and season.
- `season.season_entrants` uses the **modal driver per constructor seat over the last three completed races** of the same season when at least three exist, else the last race's lineup. Implementation: take the last three race_ids, count `(constructor_id, driver_id)` pairs, keep for each constructor the two most frequent drivers (ties broken by most recent appearance). Tests: a table where a stand-in drove only the last race must not appear in the lineup; the regular driver must.

- [ ] Failing tests as described, then implement, full suite, lint, commit: `fix: name carried-over lineups; season lineup from recent races`

---

### Task 6: Rebuild, re-tune, README

- [ ] Run:
```bash
uv run f1pred data update
uv run f1pred ratings build
uv run f1pred backtest --seasons 2023-2025 --ablate
uv run f1pred tune --train 2015-2022 --test 2023-2025 --passes 2 --runs 2000
uv run f1pred ratings build
uv run f1pred backtest --seasons 2023-2025 --ablate
uv run f1pred backtest --seasons 2023-2025 --ablate --observed-rain --out outputs/observed
uv run f1pred profile
uv run f1pred predict --season 2026 --round 15     # or next unraced round
uv run f1pred season --season 2026
cp outputs/2026-*-win.png docs/results/example-win.png
cp outputs/2026-*-positions.png docs/results/example-positions.png
cp outputs/2026-title.png docs/results/example-title.png
cp outputs/calibration.png docs/results/backtest-calibration.png
```
Sanity: 2025 to 2026 rows now have a DNF rate in the 8 to 16% range; the next-race `p_dnf` values are mostly 5 to 20%; circuit factors span roughly 0.6 to 1.6 with few at the clips.

- [ ] README: refresh every number and table (ratings top 5, prediction example, profile example, season odds, held-out backtest, ablation with both rain modes). Add a short "Changelog" section at the bottom listing the review fixes in one line each, including the upstream `position` quirk so future readers know why `positionText` is used.

- [ ] Final: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`, commit `docs: results after review fixes`, `git push origin HEAD:main`.

---

## Self-review notes

- Finding 1 fix changes ratings for 2025 to 2026 only (earlier seasons already had null positions); expect the top-5 driver order to shift.
- Findings 2 and 6 change `p_dnf` everywhere; re-tuning is mandatory, hence Task 6.
- Finding 4's loud failure means the `cache_dir` test fixture must produce a table with all feature columns; it already does after Phase 5.
