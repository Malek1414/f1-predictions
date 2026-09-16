# F1 Predictions Phase 7: Accuracy Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the parts of the model that failed on 2026 (linear grid effect, slow driver-rating adaptation, no qualifying pace) with a concave grid term and a Bayesian hierarchical Plackett–Luce pace model fit on the DGX Spark, and judge everything with distribution-aware metrics that carry confidence intervals.

**Architecture:** Three sub-phases. **7a** stays inside the existing NumPy Monte Carlo: concave grid, new metrics, bootstrap intervals, rolling-origin evaluation. **7b** adds `f1pred/pace/` with the NumPyro model, a qualifying-gap table, posterior storage, and a `--model bayes` switch for `predict` and `season` that feeds posterior samples into the existing simulation. **7c** adds the walk-forward posterior backtest, the Spark `make` targets, and the comparison report. Phase 6 behaviour is the default until 7c shows the new model wins.

**Tech Stack:** adds `jax`, `jaxlib`, `numpyro`, `arviz` (optional, for diagnostics). On the Spark: `jax[cuda12]` aarch64 wheels. CPU fallback everywhere. Existing stack otherwise.

Spec: `docs/superpowers/specs/2026-09-16-accuracy-upgrade-design.md`. Existing code to read first: `f1pred/sim/race.py`, `f1pred/backtest/run.py`, `f1pred/backtest/scoring.py`, `f1pred/tune.py`, `f1pred/predict.py`, `f1pred/data/frame.py`, `f1pred/ratings/history.py`, `f1pred/cli.py`.

## Global Constraints

- Same as earlier plans: `uv run` everything, ruff line length 100, tests offline and GPU-free, commit trailer:
  ```
  Co-Authored-By: WOZCODE <contact@withwoz.com>
  Claude-Session: https://claude.ai/code/session_01YJeHtrUFY8CEqTesftaBuB
  ```
- `numpyro` tests run on CPU with `numpyro.set_platform("cpu")`, tiny synthetic data, few samples (`num_warmup=50, num_samples=50`), and assert recovery of planted effects within loose tolerances. Keep each such test under 20 seconds.
- Everything the Spark runs must also run on CPU with `--device cpu`. The Spark is an Arm64 Linux box (`aarch64`); do not assume x86 wheels.
- Every new metric has a hand-computed unit test.
- The Phase 6 model stays the default for `predict` and `season` until Task 12 flips it, and the flip is a one-line config change.

## Maths reference (read before Tasks 1, 5, 6)

**Concave grid.** With `n` entrants, `grid_term(grid) = grid_bonus * n / grid ** grid_shape`. For `grid_shape = 1`, pole gets `grid_bonus * n`, P2 half of that, P20 one twentieth. The tuner searches `grid_shape` in `[0.3, 0.5, 0.75, 1.0, 1.5]` and `grid_bonus` in `[2, 4, 6, 8, 12]` (the scale changes because of the `n /` factor).

**Plackett–Luce likelihood.** For one race with classified finishers in order `(1), (2), ..., (k)` and paces `mu`, the log likelihood is
`sum_{j=1..k} [ mu[(j)] - logsumexp(mu[(j)], mu[(j+1)], ..., mu[(k)]) ]`.
Implemented in JAX by sorting each race's `mu` into finishing order, computing a reverse cumulative logsumexp, and summing. Non-classified drivers do not enter the race likelihood; they enter the DNF hazard.

**Qualifying likelihood.** `gap[d, r]` = 100 × (best lap of d − pole lap) / pole lap, in percent. `gap ~ Normal(alpha[r] − kappa × mu_q[d, r], sigma_q)` where `alpha[r]` is a per-race intercept (absorbs track length and conditions), `mu_q` uses the same skill and car effects as `mu` but not the grid term, and `kappa > 0` converts pace units into percent. The pole sitter's gap is 0 by definition; the intercept is identified because the sum over the race of `mu_q` is free.

**DNF hazard.** `logit P(dnf) = h0 + h_driver[d] + h_car[c, s] + h_circuit[circuit] + h_wet × is_wet`. All `h_*` have `Normal(0, 0.5)` priors except `h0 ~ Normal(-2, 1)`.

**Hierarchy and priors.** Paces are on a unit scale where `sigma_race = 1` is fixed (the Plackett–Luce scale). `skill[d, s] ~ Normal(skill[d, s-1], tau_skill)`, first season `Normal(0, 1)`; `skill_season[d, s] ~ Normal(0, tau_season)`; `car[c, s] ~ Normal(car[c, s-1], tau_car_reg if regulation season else tau_car)`, first season `Normal(0, 1.5)`; `car_form[c, r] ~ Normal(car_form[c, r-1], tau_form)` within a season, reset to 0 at season start. Hyperpriors: `tau_skill ~ HalfNormal(0.3)`, `tau_season ~ HalfNormal(0.5)`, `tau_car ~ HalfNormal(0.5)`, `tau_car_reg ~ HalfNormal(1.5)`, `tau_form ~ HalfNormal(0.1)`, `beta_grid ~ Normal(0, 1)` on `g(grid) = 1 / grid ** 0.75` (fixed shape inside the Bayesian model; the tuner's shape only applies to the Elo path), `kappa ~ HalfNormal(2)`, `sigma_q ~ HalfNormal(1)`. Wet and track effects reuse Phase 4's `is_wet` and `track_type`: `wet[d] ~ Normal(0, tau_wet)`, `track[d, t] ~ Normal(0, tau_track)`, both `HalfNormal(0.3)`.

**From posterior to simulation.** For the target race, each posterior sample gives `mu[d]` for every entrant and `p_dnf[d]`. The simulation draws sample `i` for run `i` (cycling if runs exceed samples), sets `strength = mu × pace_scale` with `pace_scale` chosen so one pace unit equals `400 / ln(10)` Elo points (so `expected_score` semantics stay meaningful for reporting), uses `sigma_driver = pace_scale` (the Plackett–Luce noise is Gumbel with scale 1; a Normal with the same variance is `pi / sqrt(6)`, use that), `sigma_team = 0` (team correlation is already in the posterior), no separate grid term (it is inside `mu`), and `p_dnf` from the sample. Wet and track effects are inside `mu` as well.

**Metrics.**
- RPS for one driver-race with position distribution `p` over `1..n` and actual position `a`: `RPS = sum_{k=1..n-1} (F_p(k) − 1[a ≤ k])² / (n − 1)`. Averaged over classified drivers.
- Top-3 set log loss: `−log P(set of podium drivers)` where the set probability is the fraction of Monte Carlo runs in which the top three equal the observed set (floor `1e-4`).
- ECE over `B = 10` equal-width bins: `sum_b (n_b / N) × |mean_p_b − mean_y_b|`.
- Sharpness: mean over races of the maximum win probability.
- Skill score: `1 − model / baseline`.
- Bootstrap: resample races with replacement `R = 1000` times, recompute the season mean, report the 5th and 95th percentiles.

## File Structure

```
f1pred/config.py                 MODIFY: grid_shape, model switch, pace params, spark host
f1pred/sim/race.py               MODIFY: concave grid term
f1pred/backtest/scoring.py       MODIFY: rps, top3_set_log_loss, ece, sharpness, skill_score
f1pred/backtest/run.py           MODIFY: per-race RPS/top-3, bootstrap, rolling-origin helper
f1pred/backtest/bootstrap.py     bootstrap_intervals(races_df, columns, R, seed)
f1pred/backtest/rolling.py       rolling_origin(table, history, params, folds, tune=...)
f1pred/data/qualifying.py        qualifying gap table: race_id, driver_id, best_ms, gap_pct, quali_position
f1pred/pace/__init__.py
f1pred/pace/design.py            index tables (drivers, constructors, seasons, races, circuits) + JAX arrays
f1pred/pace/model.py             NumPyro model: plackett_luce_logp, model(), fit(), predict_mu()
f1pred/pace/posterior.py         save/load posterior samples (NetCDF or npz), summary tables
f1pred/pace/bridge.py            posterior -> list[Entrant] + per-run strength matrix for simulate_positions
f1pred/sim/race.py               MODIFY: simulate_positions accepts strength_samples (n_runs, n)
f1pred/backtest/walkforward.py   fit-per-race posterior backtest
f1pred/cli.py                    MODIFY: --model, pace fit / pace summary commands, backtest --bootstrap/--rolling
f1pred/tune.py                   MODIFY: grid_shape in the space; rolling objective option
spark/setup.sh                   Spark bootstrap (uv, python, jax cuda aarch64, numpyro)
spark/run.sh                     helper used by make targets
Makefile                         spark-sync, spark-fit, spark-backtest, spark-tune, spark-pull
tests/test_grid_shape.py
tests/test_scoring.py            MODIFY
tests/test_bootstrap.py
tests/test_rolling.py
tests/test_qualifying.py
tests/test_pace_design.py
tests/test_pace_model.py
tests/test_pace_bridge.py
tests/test_walkforward.py
tests/test_cli.py                MODIFY
docs/pace-model.md
docs/metrics.md
docs/spark.md
README.md                        MODIFY
```

---

## Sub-phase 7a: concave grid and honest metrics (CPU)

### Task 1: Concave grid term

**Files:** `f1pred/config.py`, `f1pred/sim/race.py`, `f1pred/tune.py`, `tests/test_grid_shape.py`

**Interfaces:**
- `ModelParams.grid_shape: float = 1.0` (new). `grid_term(grid: np.ndarray, n: int, params) -> np.ndarray` in `race.py` returns `params.grid_bonus * n / grid ** params.grid_shape`. `simulate_positions` uses it instead of the linear term. The no-grid fallback variance becomes the empirical variance of `grid_term` over `1..n`: `sigma_grid = std(grid_term(arange(1, n+1), n, params))`.
- Tune space: `grid_bonus: [2, 4, 6, 8, 12]`, `grid_shape: [0.3, 0.5, 0.75, 1.0, 1.5]`.

- [ ] **Failing tests** (`tests/test_grid_shape.py`):

```python
import numpy as np
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.sim.race import Entrant, grid_term, simulate_race

P = DEFAULT_PARAMS


def test_grid_term_is_concave_and_scaled():
    n = 20
    g = grid_term(np.arange(1, n + 1), n, P.replace(grid_bonus=10.0, grid_shape=1.0))
    assert g[0] == pytest.approx(200.0) and g[1] == pytest.approx(100.0) and g[-1] == pytest.approx(10.0)
    diffs = -np.diff(g)
    assert (np.diff(diffs) <= 1e-9).all()  # gaps shrink toward the back


def test_back_of_grid_still_wins_sometimes():
    # A dominant driver from P19 must not be priced at zero.
    field = [Entrant(f"d{i}", f"t{i // 2}", 3000.0, i + 1, 0.05) for i in range(20)]
    field[18] = Entrant("d18", "t9", 3400.0, 19, 0.05)
    f = simulate_race(field, P.replace(grid_bonus=8.0, grid_shape=1.0), n_runs=20000, seed=1)
    assert f.p_win[18] > 0.01


def test_no_grid_variance_matches_term_spread():
    from f1pred.sim.race import no_grid_sigma

    p = P.replace(grid_bonus=8.0, grid_shape=1.0)
    assert no_grid_sigma(20, p) == pytest.approx(np.std(grid_term(np.arange(1, 21), 20, p)))
```

- [ ] Implement, update `tests/test_race_sim.py::test_grid_bonus_helps_pole` only if its arithmetic assumed the linear term (it should still pass), full suite, lint, commit `feat: concave grid effect with tunable shape`.

- [ ] Quick check (not a test): `uv run python -c` backtest 2024 to 2026 with `grid_bonus=8, grid_shape=1.0` versus current; the 2026 Monza race must move from log loss 13.8 to below 6.

### Task 2: Distribution metrics

**Files:** `f1pred/backtest/scoring.py`, `f1pred/backtest/run.py`, `tests/test_scoring.py`, `tests/test_backtest.py`, `docs/metrics.md`

**Interfaces:**
- `rps(position_matrix_row: np.ndarray, actual: int) -> float`
- `mean_rps(position_matrix: np.ndarray, driver_ids: list[str], actual: Mapping[str, int]) -> float` (classified drivers only)
- `top3_set_log_loss(positions: np.ndarray, driver_ids: list[str], podium: set[str]) -> float` where `positions` is the `(n_runs, n)` matrix; needs `RaceForecast` to keep `positions` (add field `positions: np.ndarray | None`, filled by `simulate_race` when `keep_positions=True`; the backtest passes it).
- `ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float`, `sharpness(p_win_per_race: list[np.ndarray]) -> float`, `skill_score(model: float, baseline: float) -> float`.
- `RACE_SCORE_COLUMNS` gains `model_rps, pole_rps, uniform_rps, model_top3, pole_top3, uniform_top3`; `SEASON_SCORE_COLUMNS` follows; `BacktestResult` gains `ece_win: float`, `ece_podium: float`, `sharpness: float`.
- Pole and uniform baselines need position matrices: pole = each driver's position distribution concentrated 90% on their grid slot and 10% spread evenly; uniform = flat. `top3` for baselines uses the analogous set probability (pole: 0.9 if the podium set equals the top three grid slots, else `0.1 / C(n,3)`; uniform: `1 / C(n,3)`).

- [ ] **Failing tests** (append to `tests/test_scoring.py`):

```python
def test_rps_perfect_and_worst():
    p = np.zeros(5); p[2] = 1.0            # certain P3
    assert rps(p, 3) == 0.0
    assert rps(p, 5) == pytest.approx((1 + 1) / 4)   # F=1 at k=3,4 while 1[a<=k]=0 -> two unit terms
    flat = np.full(5, 0.2)
    assert rps(flat, 1) == pytest.approx(((0.2-1)**2 + (0.4-1)**2 + (0.6-1)**2 + (0.8-1)**2) / 4)


def test_top3_set_log_loss():
    positions = np.array([[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 4, 3], [4, 3, 2, 1]])
    ids = ["a", "b", "c", "d"]
    assert top3_set_log_loss(positions, ids, {"a", "b", "c"}) == pytest.approx(-math.log(0.5))
    assert top3_set_log_loss(positions, ids, {"b", "c", "d"}) == pytest.approx(-math.log(0.25))
    assert top3_set_log_loss(positions, ids, {"a", "b", "d"}) == pytest.approx(-math.log(1e-4))


def test_ece_and_sharpness_and_skill():
    p = np.array([0.1, 0.1, 0.9, 0.9]); y = np.array([0, 0, 1, 1])
    assert ece(p, y, bins=10) == pytest.approx(0.1)
    assert sharpness([np.array([0.5, 0.5]), np.array([0.9, 0.1])]) == pytest.approx(0.7)
    assert skill_score(1.0, 2.0) == pytest.approx(0.5)
```

- [ ] Implement; extend `run_backtest` (rows and season means, ECE over all driver-races for win and podium, sharpness); update `tests/test_backtest.py` column assertions; write `docs/metrics.md` (each metric in one paragraph with its formula and how to read it); full suite; commit `feat: RPS, top-3 set log loss, ECE, sharpness, skill scores`.

### Task 3: Bootstrap intervals and rolling-origin evaluation

**Files:** `f1pred/backtest/bootstrap.py`, `f1pred/backtest/rolling.py`, `f1pred/cli.py`, `f1pred/tune.py`, `tests/test_bootstrap.py`, `tests/test_rolling.py`, `tests/test_cli.py`

**Interfaces:**
- `bootstrap_intervals(races: pd.DataFrame, columns: list[str], R: int = 1000, seed: int = 0, level: float = 0.9) -> pd.DataFrame` with columns `metric, mean, lo, hi`; resamples rows with replacement.
- `rolling_origin(table, params, folds: list[int], tune_fn: Callable | None, n_runs, seed) -> pd.DataFrame`: for each fold season `f`, train seasons are `params.start_season .. f-1`; if `tune_fn` is given it is called with the train seasons and returns tuned params; ratings are replayed once (they only depend on params), the fold is scored with `run_backtest`, and the frame has one row per fold with every season score plus `lo/hi` for `model_logloss` and `model_rps` from `bootstrap_intervals`.
- CLI: `backtest --bootstrap R` prints the interval table; `backtest --rolling 2024-2026 [--tune]` prints the fold table and writes `outputs/rolling.csv`.
- `tune.objective` gains `metric: str = "model_logloss"` so it can optimise RPS.

- [ ] **Failing tests**:

```python
# tests/test_bootstrap.py
def test_bootstrap_interval_contains_mean_and_narrows():
    df = pd.DataFrame({"model_logloss": np.random.default_rng(0).normal(1.5, 0.5, 200)})
    a = bootstrap_intervals(df, ["model_logloss"], R=500, seed=1)
    assert a.lo.iloc[0] < a["mean"].iloc[0] < a.hi.iloc[0]
    b = bootstrap_intervals(df.head(20), ["model_logloss"], R=500, seed=1)
    assert (b.hi - b.lo).iloc[0] > (a.hi - a.lo).iloc[0]


# tests/test_rolling.py
def test_rolling_origin_never_scores_a_training_season(driver_race):
    seen = []
    def tune_fn(table, train_seasons):
        seen.append(tuple(train_seasons)); return DEFAULT_PARAMS
    out = rolling_origin(driver_race, DEFAULT_PARAMS, [2024], tune_fn, n_runs=50, seed=0)
    assert seen == [(2023,)] and list(out.season) == [2024] and {"lo", "hi"} <= set(out.columns)
```

- [ ] Implement; CLI tests for `--bootstrap 50` and `--rolling 2024` on the sample cache; commit `feat: bootstrap intervals and rolling-origin evaluation`.

### Task 4: Re-tune 7a and record

- [ ] `uv run f1pred tune --train 2015-2023 --test 2024-2026 --passes 2 --runs 1500` with the new space; `ratings build`; `backtest --seasons 2024-2026 --bootstrap 1000 --ablate`; `backtest --rolling 2024-2026`. Record in README under a new "Phase 7a" heading: the fold table with intervals, and the Monza 2026 race's new win probability. Commit `docs: 7a results`.

---

## Sub-phase 7b: Bayesian pace model

### Task 5: Qualifying gap table

**Files:** `f1pred/data/qualifying.py`, `f1pred/data/frame.py`, `f1pred/cli.py`, `tests/test_qualifying.py`, `tests/conftest.py`

**Interfaces:**
- `parse_lap_ms(text: str | float) -> float | None` (`"1:26.572"` → 86572.0; NaN → None).
- `qualifying_gaps(raw: dict) -> pd.DataFrame` with `race_id, driver_id (driverRef), quali_position, best_ms, gap_pct` where `best_ms = min(q1, q2, q3)` and `gap_pct = 100 * (best_ms - pole_ms) / pole_ms`, `pole_ms` = the minimum `best_ms` in that race. Rows with no time have `gap_pct` NaN.
- The driver-race table gains `quali_gap_pct` (float, NaN when unknown) via `build_driver_race_table(..., qualifying: pd.DataFrame | None)`; `data update` passes `qualifying_gaps(raw)`.

- [ ] **Failing tests**:

```python
def test_parse_lap_ms():
    assert parse_lap_ms("1:26.572") == 86572.0
    assert parse_lap_ms("59.999") == 59999.0
    assert parse_lap_ms(float("nan")) is None


def test_gaps_on_sample(raw_sample):
    g = qualifying_gaps(raw_sample)
    monaco = g[g.race_id == <2024 monaco raceId from the fixture>]
    assert monaco.set_index("driver_id").loc["leclerc", "gap_pct"] == 0.0
    assert (monaco.gap_pct.dropna() >= 0).all() and monaco.gap_pct.dropna().max() < 5
    assert g.gap_pct.notna().mean() > 0.95
```
(Look the race id up in the fixture inside the test with `raw_sample["races"]`.)

- [ ] Implement; conftest passes qualifying into the `driver_race` fixture; commit `feat: qualifying lap gaps in the driver-race table`.

### Task 6: Design matrices

**Files:** `f1pred/pace/__init__.py`, `f1pred/pace/design.py`, `tests/test_pace_design.py`

**Interfaces:**
- `@dataclass PaceDesign` holding integer index arrays (NumPy) for every non-sprint driver-race row up to a cutoff: `driver, driver_season, constructor_season, race, circuit, season_of_race, track_type, is_wet, grid_g (1/grid**0.75), position (0 = not classified), dnf, quali_gap (NaN where unknown)`, plus lookup tables (`driver_ids`, `constructor_season_keys` as `(constructor_id, season)`, `race_ids`, `circuit_ids`, `season_index`), per-race slices (`race_start, race_len` for a sorted layout), `previous_driver_season` and `previous_constructor_season` index arrays (−1 for none), `is_regulation_season` per constructor-season, and `n_*` counts.
- `build_design(table: pd.DataFrame, before_date: pd.Timestamp | None = None, params: ModelParams) -> PaceDesign`. Rows with `date < before_date` only; `before_date=None` uses everything.
- `target_design(design: PaceDesign, entrants: list[Entrant], race: RaceRef) -> dict` mapping each entrant to `(driver index or −1 if unseen, constructor_season index or −1, circuit index or −1)` so unseen drivers get prior draws.

- [ ] **Failing tests** on the sample: counts match (`n_driver == driver_race.driver_id.nunique()`), every race slice is contiguous and sums to the row count, `previous_driver_season` links 2023 → 2024 for Verstappen and is −1 for 2023, `is_regulation_season` is False for both sample seasons, `before_date` excludes rows on or after it.

- [ ] Implement; commit `feat: pace model design matrices`.

### Task 7: NumPyro model, fit, prediction

**Files:** `f1pred/pace/model.py`, `f1pred/pace/posterior.py`, `f1pred/config.py`, `tests/test_pace_model.py`, `docs/pace-model.md`

**Interfaces:**
- `plackett_luce_logp(mu_sorted: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray`: `mu_sorted` is a padded `(n_races, max_len)` array of paces in finishing order (padding `-inf`), returns per-race log likelihood via reverse `logcumsumexp`.
- `pace_model(design: PaceDesign, params: ModelParams)`: the NumPyro model exactly as in the maths reference, using `numpyro.plate` over drivers, constructor-seasons, races, with non-centred parameterisations for every random walk.
- `fit(design, params, num_warmup=500, num_samples=500, chains=2, seed=0, device="cpu"|"gpu") -> Posterior` (NUTS; `numpyro.set_platform(device)`; `chain_method="vectorized"` on GPU).
- `@dataclass Posterior(samples: dict[str, np.ndarray], design_meta: dict, diagnostics: dict)` with `save(path)`, `load(path)` (`.npz` plus a JSON sidecar), `summary() -> pd.DataFrame` (mean, sd, r_hat, ess for `skill`, `car`, `kappa`, `beta_grid`, taus).
- `predict_mu(posterior, design, entrants, race_index_info, grid: dict | None, is_wet_prob: float) -> tuple[np.ndarray, np.ndarray]`: `(mu_samples (S, n), p_dnf_samples (S, n))` for the target race; unseen driver → `skill` drawn from the hyperprior, unseen constructor-season → previous season's car plus the walk noise or the wide prior; `car_form` for the target race = last observed value for that constructor in the season.
- `ModelParams` gains `pace_num_warmup=500, pace_num_samples=500, pace_chains=2, pace_device="cpu"`.

- [ ] **Failing tests** (all on CPU, tiny synthetic data, seeds fixed):

```python
def test_plackett_luce_matches_brute_force():
    mu = jnp.array([[2.0, 1.0, 0.0, -jnp.inf]]); lengths = jnp.array([3])
    lp = plackett_luce_logp(mu, lengths)[0]
    expected = (2 - logsumexp([2, 1, 0])) + (1 - logsumexp([1, 0])) + (0 - 0)
    assert float(lp) == pytest.approx(expected, abs=1e-5)


def test_fit_recovers_planted_car_and_skill():
    # 3 seasons x 8 races x 6 drivers in 3 cars; car A +1.0, driver "ace" +0.8; sample finishing orders
    # from the true Plackett-Luce, no DNFs, quali gaps from the same paces with kappa=1.
    design = _synthetic_design(seed=0)
    post = fit(design, DEFAULT_PARAMS, num_warmup=80, num_samples=80, chains=1, seed=0)
    s = post.summary().set_index("param")
    assert s.loc["car[A,2]", "mean"] > s.loc["car[B,2]", "mean"] + 0.4
    assert s.loc["skill[ace,2]", "mean"] > 0.3
    assert post.diagnostics["divergences"] < 5


def test_breakout_season_is_credited_within_season():
    # Driver "kid" is average in seasons 1-2 and +1.2 in season 3 sharing car A with "vet".
    design = _synthetic_design(seed=1, breakout=("kid", 3, 1.2))
    post = fit(design, DEFAULT_PARAMS, num_warmup=80, num_samples=80, chains=1, seed=1)
    mu, _ = predict_mu(post, design, entrants_for_next_race(design), ..., grid=None, is_wet_prob=0.0)
    assert mu.mean(axis=0)[idx("kid")] > mu.mean(axis=0)[idx("vet")] + 0.5


def test_posterior_roundtrip(tmp_path):
    ...save/load equality on a tiny fit...
```
Write `_synthetic_design` in the test file: it builds a driver-race DataFrame with the same columns as the real table (use `tests/test_profile.py::_table` as a template), samples orders with Gumbel-max from planted paces, computes `quali_gap_pct` as `kappa*(max_mu - mu) + noise`, then calls `build_design`.

- [ ] Implement. Performance target on CPU for the sample (2 seasons): under 60 s with defaults; the tests use reduced settings. Write `docs/pace-model.md` (the maths reference above in plain English plus the priors table and how to read `summary()`). Commit `feat: Bayesian Plackett-Luce pace model with qualifying likelihood and DNF hazard`.

### Task 8: Posterior into the Monte Carlo

**Files:** `f1pred/pace/bridge.py`, `f1pred/sim/race.py`, `f1pred/predict.py`, `f1pred/cli.py`, `tests/test_pace_bridge.py`, `tests/test_cli.py`

**Interfaces:**
- `simulate_positions(..., strength_samples: np.ndarray | None = None, p_dnf_samples: np.ndarray | None = None)`: when given, run `i` uses row `i % S` of each; `sigma_team` is forced to 0 and the grid term is skipped (both live inside `mu`); `sigma_driver` is replaced by `params.pace_scale * pi / sqrt(6)`. `simulate_race` passes them through.
- `bridge.forecast_from_posterior(posterior, design, inputs: PredictionInputs, params, n_runs, seed) -> RaceForecast`.
- `ModelParams.model: str = "elo"` (`"elo"` or `"bayes"`), `pace_scale: float = 400 / ln(10)`.
- CLI: `predict --model bayes` loads `data/cache/posteriors/latest.npz` (error with the command to run if missing); new command group `pace`: `f1pred pace fit [--device cpu|gpu] [--warmup] [--samples] [--chains] [--cutoff DATE]` writes `posteriors/<cutoff or latest>.npz` and prints `summary()` top rows plus diagnostics; `f1pred pace summary` prints the driver skill and car pace tables for the current season with 90% intervals.

- [ ] **Failing tests**: `simulate_positions` with `strength_samples` reproduces exact per-run ordering for a hand-built 2-run, 3-driver case; `forecast_from_posterior` on the tiny synthetic posterior gives the planted best driver the highest `p_win`; CLI `pace fit --warmup 20 --samples 20 --chains 1` on the sample cache writes the file and `predict --model bayes --season 2024 --round 8` runs.

- [ ] Implement; commit `feat: posterior-driven race simulation and pace CLI`.

---

## Sub-phase 7c: walk-forward evaluation, Spark, comparison

> **Tabled until the DGX Spark is reachable** (decided 2026-09-16). Task 9's *code* ships with 7b
> and is tested with `fit` monkeypatched; the full 62-fit run does not. Measured on the M2 (8 cores,
> 24 GB): one fit is about 5 minutes on a single chain and 9 minutes on 4 parallel chains, so the
> walk-forward run is 4 to 9 hours here versus under an hour on the Spark. The workload is memory
> and latency bound, not FLOP bound: 1.11 MFLOP against 3.74 MB of traffic per gradient evaluation,
> so the Spark's value is running chains and fits side by side, not making one fit faster.
> To unblock: a `Host spark` entry in `~/.ssh/config`, then `make spark-check`.


### Task 9: Walk-forward posterior backtest

**Files:** `f1pred/backtest/walkforward.py`, `f1pred/cli.py`, `tests/test_walkforward.py`

**Interfaces:**
- `walkforward(table, seasons, params, fit_kwargs, cache_dir, n_runs, seed, refit_every: int = 1) -> BacktestResult`: for each race in the seasons in date order, `build_design(table, before_date=race_date)`, `fit` (or reuse the last posterior if fewer than `refit_every` races have passed), `forecast_from_posterior` with the actual grid and observed `is_wet` treated as the Phase 6 default (pre-race circuit rate) unless `observed_rain`, then score with the same row builder as `run_backtest`. Posteriors are cached at `cache_dir/posteriors/wf-<race_id>.npz` so a rerun skips finished fits.
- CLI: `backtest --model bayes [--refit-every 1] [--device gpu]`.

- [ ] **Failing test**: on a 3-race synthetic table, `walkforward` with `refit_every=1` and tiny sampler settings produces 3 scored rows, writes 3 posterior files, and a second call performs no fits (monkeypatch `fit` to count calls).

- [ ] Implement; commit `feat: walk-forward Bayesian backtest with posterior cache`.

### Task 10: Spark workflow

**Files:** `spark/setup.sh`, `spark/run.sh`, `Makefile`, `docs/spark.md`, `f1pred/config.py` (`SPARK_HOST` env read helper)

- `spark/setup.sh` (run once on the Spark over SSH): install `uv` if missing, `uv python install 3.12`, `uv sync`, then `uv pip install --upgrade "jax[cuda12]"` (aarch64 CUDA wheels), and verify with `uv run python -c "import jax; print(jax.devices())"` showing a `CudaDevice`. Print the CUDA driver version and free memory.
- `Makefile`:
  - `spark-sync`: `rsync -az --delete --exclude .venv --exclude outputs --exclude .git . $(SPARK_HOST):~/f1-predictions/` then `rsync -az data/cache/ $(SPARK_HOST):~/f1-predictions/data/cache/`.
  - `spark-fit`: `ssh $(SPARK_HOST) 'cd ~/f1-predictions && uv run f1pred pace fit --device gpu --warmup 1000 --samples 1000 --chains 4'` then pull `data/cache/posteriors/`.
  - `spark-backtest`: `ssh ... uv run f1pred backtest --model bayes --seasons 2024-2026 --device gpu --bootstrap 1000` then pull `outputs/`.
  - `spark-tune`: `ssh ... uv run f1pred tune --train 2015-2023 --test 2024-2026 --runs 3000 --passes 3` then pull `f1pred/tuned.json`.
  - `spark-pull`: rsync `outputs/` and `data/cache/posteriors/` back.
  - `SPARK_HOST ?= spark` (SSH config alias; document `~/.ssh/config` with `Host spark`, `HostName <ip>`, `User <user>`).
- `docs/spark.md`: one page: what runs where, the first-time setup, expected runtimes (single fit: minutes on GPU, an hour on a laptop; walk-forward 2024 to 2026 with `refit_every 1`: under an hour on GPU), how to check `nvidia-smi`, and the CPU fallback.

- [ ] No unit tests for shell; add a `make spark-check` target that only runs `ssh $(SPARK_HOST) nvidia-smi`. Commit `feat: DGX Spark make targets and setup`.

### Task 11: Run on the Spark and compare

- [ ] `make spark-sync && make spark-check`. If SSH is not configured, stop and tell the user which host and user to put in `~/.ssh/config`.
- [ ] `make spark-fit`; inspect `f1pred pace summary`: sanity checks are that Mercedes 2026 car pace is the highest car effect in 2026 with a clear interval, Antonelli's `skill_season[2026]` is positive with an interval excluding zero, `r_hat < 1.05` everywhere, divergences under 1% of samples.
- [ ] `make spark-backtest` (walk-forward 2024 to 2026, `refit_every 1`, bootstrap intervals) and, on the Mac, the Phase 7a Elo backtest with `--bootstrap 1000` on the same seasons. Build the comparison table: per fold, winner log loss, RPS, top-3 log loss, ECE, sharpness, each with 90% intervals, for Elo-7a, Bayes, and the pole baseline. Include the Monza 2026 race's win probability under both models.
- [ ] Decide the default: if Bayes beats Elo-7a on winner log loss and RPS in at least two of three folds with non-overlapping intervals in at least one, set `ModelParams.model = "bayes"`; otherwise keep `"elo"` and say why in the README.

### Task 12: README and wrap-up

- [ ] README: a "Phase 7" section with the maths in plain English (link `docs/pace-model.md`, `docs/metrics.md`, `docs/spark.md`), the comparison table with intervals, the default-model decision, the Spark quick start (`make spark-sync spark-fit`), and an updated usage list (`pace fit`, `pace summary`, `predict --model bayes`, `backtest --bootstrap/--rolling/--model bayes`).
- [ ] `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`; commit `docs: Phase 7 results and Spark quick start`; push.

---

## Self-review notes

- Spec coverage: 2.1 Task 1; 2.2 Tasks 5 to 9; 2.3 Tasks 2 and 3 (RPS, top-3, ECE, sharpness, skill, bootstrap, rolling-origin); 2.4 Task 10; success criteria checked in Task 11.
- The Elo path keeps working unchanged apart from the concave grid (Task 1), which is tuned before any Bayesian work so 7a is a shippable improvement on its own.
- Identifiability: the Plackett–Luce likelihood is invariant to adding a constant to all `mu` in a race; the per-race intercept `alpha[r]` in the qualifying likelihood absorbs the same freedom there. Car and driver effects are identified through teammates (same car, different drivers) and driver moves (same driver, different cars), as in van Kesteren and Bergkamp; the random-walk priors regularise seasons with few moves.
- GPU-free tests: every NumPyro test sets the CPU platform and tiny sampler settings; the walk-forward test monkeypatches `fit`.
