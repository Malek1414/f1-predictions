# F1 Predictions: Design

Date: 2026-09-15
Status: implemented (Phases 1 to 5 built 2026-09-15; Phase 6 review fixes in progress). Section 15 records where the build deviated from this text.

## 1. Goal

A Python project that predicts Formula 1 outcomes with an Elo rating system and Monte Carlo simulation, using Hugging Face data. It serves three purposes at once: a portfolio repo on GitHub, a vehicle for learning the maths by building it, and a tool that gives usable numbers before each race weekend.

Delivered in phases, each usable on its own:

| Phase | Output |
|-------|--------|
| 1 | Win probability per driver for a given race |
| 2 | Full finishing-order distribution (podium, points, position heatmap) |
| 3 | Drivers' and constructors' championship odds for the rest of a season |
| 4 | Weather (wet/dry) and track-type adjustments, measured against the backtest |
| 5 | Driver profile: aggression, risk and form ("mental edge"), measured against the backtest |

Phases 1 and 2 come from the same simulation loop and ship together. A backtest harness is built in Phase 1 and every later phase is judged by it.

## 2. Non-goals

- No lap-by-lap physics simulation (tyres, pit strategy, fuel).
- No web dashboard. CLI plus saved PNG charts. A dashboard can be added later on top of the library.
- No betting or bookmaker odds integration.
- No gradient-boosting or neural models. If wanted later, they plug in as an alternative strength model behind the same simulation and are compared with the backtest.

## 3. Data

### 3.1 Primary source: Hugging Face `tracinginsights/RaceData`

A live mirror of the Ergast schema, 1950 through the current 2026 round, updated frequently. Each table is a CSV at `data/<table>.csv` in the dataset repo, fetched with `huggingface_hub.hf_hub_download` (nulls are the literal `\N`). Tables used:

| Table | Used for |
|-------|----------|
| `races` | season, round, date, circuit, race name |
| `results` | grid, finish position, status, points per driver-race |
| `qualifying` | qualifying position and times (Phase 1 uses position only) |
| `sprint_results` | sprint points for season simulation |
| `drivers`, `constructors`, `circuits` | names, codes, circuit latitude/longitude |
| `status` | text for each status id, to classify DNFs |
| `driver_standings`, `constructor_standings` | current points for season simulation |
| `lap_times` | position at the end of lap 1, for the aggression measure (Phase 5) |

The dataset declares no license. It is derived from Ergast/Jolpica data (CC-BY style). Fine for a portfolio project; the README credits the source.

### 3.2 Secondary source (Phase 4): Open-Meteo

Free, no API key. Historical archive endpoint gives daily precipitation for any latitude/longitude and date back to 1940. Forecast endpoint gives precipitation probability up to 16 days ahead. Queried once per race using the circuit coordinates from the `circuits` table and the race date.

### 3.3 Hand-curated file (Phase 4): track types

`f1pred/data/track_types.yaml` maps each circuit id used since 2010 to one of `street`, `high_speed`, `high_downforce`, `mixed`. Around 40 entries. Any circuit missing from the file falls back to `mixed` and a warning is logged.

### 3.4 Cache and scope

- All downloaded tables are written to `data/cache/*.parquet`. Commands read the cache; `f1pred data update` refreshes it. If Hugging Face is unreachable and a cache exists, the cache is used with a warning. If no cache exists, the command fails with a clear message.
- The rating replay uses 2010 onward. Earlier history is not needed and complicates constructor identity.
- The `data/cache/` and `outputs/` folders are git-ignored.

### 3.5 The driver-race table

Everything downstream consumes one tidy table, one row per driver per race:

| Column | Meaning |
|--------|---------|
| `season`, `round`, `race_id`, `date`, `circuit_id` | race identity |
| `driver_id`, `driver_code`, `constructor_id` | who and in what car |
| `grid` | starting position (0 means pit lane start, mapped to last) |
| `position` | classified finishing position, null if not classified |
| `status` | raw status text |
| `dnf` | true if not classified due to accident, collision, or mechanical failure |
| `dnf_kind` | `accident`, `mechanical`, `other`, or null |
| `points` | points scored in this session |
| `is_sprint` | true for sprint-race rows; their `date` is the sprint date |
| `is_wet` (Phase 4) | race-day rainfall above threshold |
| `track_type` (Phase 4) | from the mapping file |

Drivers classified but many laps down still count as finishers for rating purposes.

## 4. Architecture

Package `f1pred`, five modules in a strict order, each depending only on the ones before it: data → ratings → sim → backtest → report.

```
f1pred/
  data/        Hugging Face download, Parquet cache, driver-race table, weather, track types
  ratings/     Elo engine and history replay
  sim/         race Monte Carlo, season Monte Carlo, points rules
  backtest/    replay past seasons, scoring, calibration
  report/      terminal tables and PNG charts
  cli.py       Typer entry point `f1pred`
tests/         pytest, one file per module
docs/          one short plain-English note per concept
data/cache/    git-ignored Parquet
outputs/       git-ignored charts and CSV results
```

Data flows one way: Hugging Face → Parquet cache → driver-race table → ratings history → simulation → tables and charts. The backtest sits beside the simulation and drives it with historical inputs.

Each layer exposes plain functions over pandas DataFrames and small dataclasses. No layer imports from a layer above it.

## 5. Ratings

### 5.1 Elo basics

Every driver and every constructor starts at 1500. A driver's race strength is `driver_rating + constructor_rating`. For any two finishers A and B in the same race, the expected chance that A beats B is

```
E_A = 1 / (1 + 10 ** ((S_B - S_A) / 400))
```

where `S` is race strength. After the race, if A finished ahead, the actual result is 1 for A and 0 for B. Each rating moves by `K * (actual - expected)`. Driver and constructor ratings use separate K values (`k_driver`, `k_constructor`). The per-race change for a driver is the sum over all their pairings, scaled by `1 / (n_finishers - 1)` so a race with 20 finishers does not move ratings 20 times harder than a race with 10.

### 5.2 What counts

- Only classified finishers take part in pairwise updates. DNFs neither gain nor lose rating; reliability is modelled separately (section 6.3).
- Teammate pairings are the cleanest signal of driver skill because the car cancels out. They are weighted by `teammate_weight` (default 2) in the driver update and excluded from the constructor update.
- Sprint results update ratings with `K` scaled by `sprint_weight` (default 0.5).

### 5.3 Season carry-over

At the start of each season, ratings regress toward 1500:

```
rating = 1500 + (rating - 1500) * (1 - regress)
```

with `regress_driver` (default 0.1) and `regress_constructor` (default 0.4). In regulation-change seasons (2014, 2017, 2022, 2026, listed in config) constructors use `regress_constructor_regulation` (default 0.7). A driver changing teams keeps their driver rating; the constructor rating follows the constructor id.

### 5.4 New entrants

A driver or constructor with no history starts at 1500. The report flags any driver with fewer than `min_races_for_confidence` (default 5) rated races as low-confidence.

### 5.5 Conditional ratings (Phase 4)

Each driver and constructor also carries a wet rating and one rating per track type. These are updated only on races of that condition, with the same rules, and are shrunk toward the overall rating when used:

```
effective = overall + (conditional - overall) * n / (n + shrink)
```

where `n` is the number of rated races in that condition and `shrink` (default 8) controls how much evidence is needed before the conditional rating matters. Two separate shrink constants exist, `shrink_wet` and `shrink_track`.

### 5.6 Ratings history

`f1pred ratings build` replays all races from 2010 in date order and writes `data/cache/ratings_history.parquet`: one row per driver-race and constructor-race with the rating *before* that race. This is the only table the simulation and backtest read from, which guarantees a prediction for race X never sees the result of race X or anything after it.

## 6. Race simulation

### 6.1 Inputs

For one race: the list of entrants (driver, constructor), their pre-race ratings, grid positions, DNF rates, and (Phase 4) rain probability and track type.

Grid positions for a future race come from the `qualifying` table if the session has happened, else the model can be run with `--no-grid`, in which case the grid term is dropped and its variance is folded into the driver noise.

### 6.2 One run

For each of `n_runs` (default 10,000) iterations:

1. Draw one team noise per constructor, `N(0, sigma_team)`, shared by both of its cars.
2. Draw one driver noise per driver, `N(0, sigma_driver)`.
3. Performance = `driver_rating + constructor_rating + grid_bonus * (n_entrants - grid) + team_noise + driver_noise`.
4. Draw DNF per driver with probability `p_dnf` (section 6.3). DNFs are removed from the classified order and placed at the back in random order.
5. Sort survivors by performance, descending. That is the finishing order for this run.

All runs are vectorised in NumPy: a `(n_runs, n_entrants)` matrix of performances, argsort along axis 1. A run of 10,000 iterations for 20 drivers takes well under a second.

### 6.3 DNF probability

```
p_dnf = shrink_toward(global_rate, blend(driver_recent_rate, constructor_recent_rate), n_recent) * circuit_factor
```

- Recent rates use the last `dnf_window` (default 20) races for the driver and constructor, blended 50/50.
- Shrunk toward the global rate over the same window with `shrink_dnf` (default 10).
- `circuit_factor` is the circuit's historical DNF rate (from results) divided by the global rate, clipped to [0.5, 2.0]; it is 1 until the circuit has 40 driver-race rows.
- Phase 4: multiplied by `wet_dnf_factor` (fitted, expected around 1.5) in wet runs.

### 6.4 Outputs

From the `(n_runs, n_entrants)` matrix of finishing positions:

- `p_win`, `p_podium`, `p_points` per driver
- expected finishing position and its 10th and 90th percentile
- a `(n_entrants, n_positions)` matrix of `P(driver finishes in position k)`

Probabilities across drivers for any single position sum to 1.

### 6.5 Driver profile in the simulation (Phase 5)

Three per-driver numbers from section 6.6 enter one run as follows:

- `aggression` adds `aggression_scale * aggression` to performance.
- `risk` multiplies that driver's `sigma_driver` by `(1 + risk_noise_scale * risk)` and multiplies the accident share of their `p_dnf` by `(1 + risk_dnf_scale * risk)`.
- `form` adds `form_scale * form` to performance.

All three scales are tuned (section 8.1) and default to 0, so the profile layer is inert until the backtest shows it helps.

### 6.6 Driver profile measures (Phase 5)

Computed from history before each race, each shrunk toward the field mean with `shrink_profile` (default 10) using the same `n / (n + shrink)` rule as section 5.5:

- **Aggression**: average of two signals, both in positions and both from finishers only: places gained on lap 1 (`grid` minus position after lap 1, from `lap_times`) and places gained over the race (`grid` minus `position`). Positive means the driver tends to move forward.
- **Risk**: the driver's rate of `accident` DNFs over the last `dnf_window` races, divided by the field rate over the same window, minus 1. Positive means crash-prone.
- **Form**: over the last `form_window` (default 6) races, mean of (expected position under the model minus actual position). Positive means the driver has recently beaten their rating.

Expected position for the form measure is the mean position from a small simulation (`n_runs` 1,000) with the profile layer switched off, so form does not feed back into itself.

### 6.7 Weather in the simulation (Phase 4)

For a historical race, `is_wet` is known and fixed. For a future race, each run first draws `wet ~ Bernoulli(rain_probability)`. In a wet run, effective ratings use the wet conditional rating, `sigma_team` and `sigma_driver` are multiplied by `wet_noise_factor`, and DNF probability by `wet_dnf_factor`. Track type always uses the effective track-type rating.

## 7. Season simulation

Inputs: current driver and constructor standings, the remaining calendar (with sprint flags), pre-race ratings as of today, and grid unknown for all remaining races (`--no-grid` mode).

For each of `n_runs` (default 2,000) iterations, simulate every remaining race with a single run of section 6.2, award points with the current rules (25-18-15-12-10-8-6-4-2-1 for the race, 8-7-6-5-4-3-2-1 for sprints, no fastest-lap point), add to the standings, and record the leaders. Ratings are held fixed within a simulated season; this is a known simplification and is documented.

Outputs: title probability per driver and per constructor, expected final points, and a distribution of final positions.

Points rules live in one table in `sim/points.py` keyed by season, so past seasons with the fastest-lap point backtest correctly.

## 8. Backtest and scoring

`f1pred backtest --seasons 2023-2025` does, for each race in range:

1. Read pre-race ratings from the ratings history.
2. Read the actual grid from the results table (the qualifying result was known before the race).
3. Run the race simulation.
4. Score against the actual result.

Scores per race, aggregated per season:

- **Winner log loss**: `-log(p_win of the actual winner)`.
- **Podium Brier score**: mean over drivers of `(p_podium - actually_on_podium)^2`.
- **Position Spearman**: rank correlation between expected finishing position and actual, finishers only.

Each score is printed next to two baselines computed on the same races:

- **Pole wins**: 90% on the pole sitter, remaining 10% spread evenly (log loss and Brier), expected position equals grid (Spearman).
- **Uniform**: equal probability for every driver.

A calibration chart bins all driver-race win probabilities into deciles and plots predicted versus observed frequency.

### 8.1 Tuning

`f1pred tune` runs coordinate descent (one parameter at a time over a short candidate list, two passes) over `k_driver`, `k_constructor`, `sigma_team`, `sigma_driver`, `grid_bonus`, `regress_constructor`, optimising winner log loss on the seasons 2015 through 2022 and reporting the held-out scores on 2023 through 2025. The chosen values are written to `f1pred/tuned.json` (only the fields that differ from the defaults, plus a note recording the date and scores); `load_params` reads that file on top of the defaults. Phase 4 adds the wet, track, and DNF factors to the grid and reports held-out scores with and without each. Phase 5 does the same for the three profile scales.

Tuning and evaluation seasons are kept separate so the reported numbers in the README are honest.

## 9. CLI and reports

Entry point `f1pred` built with Typer. Tables use Rich. Charts use matplotlib, saved to `outputs/`.

| Command | Does |
|---------|------|
| `f1pred data update` | download or refresh tables into the Parquet cache, build the driver-race table |
| `f1pred ratings build` | replay history, write ratings history, print current top 10 drivers and constructors |
| `f1pred predict --season 2026 --round 15` or `--race monza` | race probabilities table; saves `outputs/2026-15-win.png` (bar chart) and `outputs/2026-15-positions.png` (heatmap) |
| `f1pred season --season 2026` | title odds tables; saves `outputs/2026-title.png` |
| `f1pred backtest --seasons 2023-2025` | per-season score table versus baselines; saves `outputs/calibration.png` |
| `f1pred tune` | grid search, writes chosen parameters to config |

Options: `--runs N`, `--no-grid`, `--seed N` on the simulation commands, `--out DIR` everywhere. `--race` matches case-insensitively on circuit id or a substring of the race name; an unknown value lists the season's races.

## 10. Error handling

| Situation | Behaviour |
|-----------|-----------|
| Hugging Face unreachable, cache present | warn, use cache |
| Hugging Face unreachable, no cache | exit 1 with a message naming the command to run once online |
| Unknown season or round | exit 2, list valid rounds for that season |
| Race has no qualifying data yet | warn, run in `--no-grid` mode |
| Driver with fewer than 5 rated races | included, marked "low confidence" in the table |
| Circuit missing from track-type file (Phase 4) | fall back to `mixed`, warn |
| Open-Meteo unreachable (Phase 4) | treat rain probability as the circuit's historical wet rate, warn |
| Open-Meteo forecast horizon exceeded | same fallback, warn |

## 11. Testing

`pytest`, test-first for all maths. Fixtures are small hand-built DataFrames, never the live Hub.

- `test_elo.py`: expected-score formula on known values; a single pairwise update with hand-computed numbers; teammate weighting; DNFs unchanged; season regression; regulation-year regression.
- `test_history.py`: replay a fake three-race season and assert the pre-race rating table is right, including that race 2's row does not reflect race 2's result.
- `test_race_sim.py`: position probabilities sum to 1 for each position; a driver with much higher rating has the highest `p_win`; `p_dnf = 1` gives `p_win = 0`; identical drivers get equal probabilities within tolerance; fixed seed gives identical output.
- `test_dnf.py`: shrinkage moves toward the global rate as `n` shrinks; circuit factor clipping.
- `test_points.py`: race and sprint points tables per season; fastest-lap point only in seasons that had it.
- `test_season_sim.py`: a two-race remaining season where one driver already has an unassailable lead gives them probability 1.
- `test_backtest.py`: scores on a perfect prediction and on the uniform prediction match hand-computed values; the backtest never reads a row with a date on or after the race being predicted.
- `test_data.py`: driver-race table built from a saved sample matches expected columns, DNF classification for a list of status strings.
- Phase 4: `test_weather.py` (Open-Meteo responses mocked, wet threshold), `test_conditional.py` (shrinkage formula, unknown track type fallback).
- Phase 5: `test_profile.py`: aggression from a fake lap-times table with known lap-1 positions; risk equals 0 for a driver at the field rate; form is positive for a driver beating expectation; all three shrink to 0 with no history; with all scales at 0 the simulation output is identical to Phase 1.

CLI commands are smoke-tested with Typer's test runner on the sample data.

## 12. Repository

- GitHub: `Malek1414/f1-predictions`, public, MIT license.
- Tooling: `uv`, Python 3.12, `ruff` for lint and format, `pytest`.
- Dependencies: `datasets`, `pandas`, `pyarrow`, `numpy`, `typer`, `rich`, `matplotlib`, `pyyaml`, `requests` (Phase 4).
- README: what it does, the maths in one paragraph each, how to run, the backtest results table, the two charts, data credits.
- `docs/`: `elo.md`, `monte-carlo.md`, `dnf-model.md`, `backtesting.md`, `season-sim.md`, `weather-and-track.md`, `driver-profile.md`. Each is a short plain-English note written when the module is built.

## 13. Phase order

1. Data layer, Elo engine, ratings history, race simulation, backtest with baselines, `predict` and `backtest` commands, charts. Tune.
2. Full-order outputs (heatmap, podium and points probabilities) are produced by Phase 1's simulation; this phase is the report and README work plus the Spearman score.
3. Season simulation and `season` command.
4. Weather via Open-Meteo, track types via the mapping file, conditional ratings, re-tune, backtest comparison with and without.
5. Driver profile (aggression, risk, form) from lap-1 positions, DNF kinds and recent over-performance; re-tune; backtest comparison with and without.

## 14. Prior art consulted

- mar-antaya `2025_f1_predictions` and `2026_f1_predictions`: per-race gradient boosting on qualifying times. Inspiration for the "predict before each race" workflow; not the modelling approach.
- villekuosmanen `F1Predict`: driver, constructor, engine Elo with Monte Carlo perturbation. Closest to this design.
- van Kesteren and Bergkamp 2023 (JQAS): Bayesian model showing constructor explains most of the variance in results, which motivates the separate constructor rating and heavier constructor regression.
- Kevocado `F1_Predictor`, neevj2006 `F1_Race_Predictor`: separate reliability layer, shared team noise, leakage-safe backtests.

## 15. Deviations recorded during implementation

Each item names the section it amends.

- **3.1, 3.5**: `results.position` is not used. From the 2025 season the upstream CSV fills `position` with `positionOrder` for retired cars, so "classified" and `position` are derived from `positionText` (numeric means classified). Statuses are classified with explicit accident and mechanical whitelists; anything else unclassified is `other`.
- **3.1**: the `safety_cars` and `driver_standings` tables are not used. Circuit incident rates come from the results table; standings are summed from the driver-race table (race plus sprint points).
- **3.1**: `lap_times` is used only for lap-1 positions (Phase 5), stored as `lap1.parquet`.
- **3.2**: rain is the sum of hourly precipitation over the three hours from race start (UTC `time`, 13:00 if missing); a race is wet at 0.5 mm or more. Sprint rows share their race's `is_wet`.
- **5.5**: conditional (wet, track-type) updates are computed from the pre-race overall ratings and seed a new conditional entry at the driver's or constructor's current overall rating.
- **6.3**: the circuit factor is the circuit's all-time DNF rate divided by the all-time global rate (same time base); the baseline rate is over the last 400 driver-race rows. Evidence count is in races: `(n_driver + n_constructor / 2) / 2`.
- **6.5, 6.6**: the form measure's expected position is the driver's rank by pre-race strength among the race's entrants, not a nested simulation. Risk is `(accident_rate - field_rate) / field_rate` shrunk toward 0, so a driver with no accidents scores 0 rather than -1. Field baselines are taken once per race so no same-race row leaks into a driver's baseline.
- **7**: the 2019 to 2024 fastest-lap bonus point is not simulated. The remaining-season lineup is the modal driver per seat over the last three completed races of the season.
- **8**: the backtest uses the circuit's historical wet rate as the rain probability by default (pre-race information only); `--observed-rain` reports the upper bound with the observed race-day rainfall. Ablation variants: base, weather, track, full (weather + track), profile (everything).
- **8.1**: tuning is coordinate descent (two passes over short candidate lists) and writes `f1pred/tuned.json`, which `load_params` layers over the defaults.
- **9**: additional commands `season`, `profile`, and `tune`; `predict --rain` overrides the forecast; `backtest --ablate` and `--observed-rain`.
- **10**: missing feature columns (`is_wet`, `track_type`, `lap1_position`) and ratings older than the data are hard errors with a message naming the command to run, never silent fallbacks.
