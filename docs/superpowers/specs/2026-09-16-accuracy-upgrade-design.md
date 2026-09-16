# Accuracy Upgrade: Bayesian Pace Model on the DGX Spark

Date: 2026-09-16
Status: approved in conversation (user asked for the plan directly); implementation plan at `docs/superpowers/plans/2026-09-16-phase-7-bayesian-pace-model.md`.

## 1. Why

The Elo + Monte Carlo model (Phases 1 to 6) is honest and beats the baselines on 2023 to 2025, but the 2026 season exposed three structural limits that tuning cannot fix:

1. **Breakout seasons are learned too slowly.** One race moves a driver's Elo about as much as one chess game. Antonelli won 8 of the first 14 races of 2026 and his driver rating rose by about 70 points. The car rating absorbed most of the signal, and because Russell shares that car the model could not tell the two apart until late in the season.
2. **The grid effect is linear.** After tuning, each grid slot is worth 25 rating points, so P19 is a 450-point penalty. Antonelli's Monza win from P19 was priced at 0.000%, a single race that costs a full point of season-average log loss. The true effect of grid position is strongly concave: the front rows matter a lot, the back rows barely at all.
3. **Qualifying pace is only used as a position.** The qualifying lap-time gap to pole is the cleanest pre-race measure of car pace and is 99% available since 2010, but the model ignores it.

The metrics are also thin: winner log loss, podium Brier and Spearman say little about the full finishing-order distribution, carry no confidence intervals, and calibration is judged by eye.

## 2. What we build

### 2.1 Concave grid effect (cheap, first)

Replace `grid_bonus * (n - grid)` with `grid_bonus * n * (1 / grid ** grid_shape)` where `grid_shape` in `[0.3, 1.5]` is tuned. At `grid_shape = 1`, pole versus P2 is worth half the full bonus and P19 versus P20 almost nothing. Stays inside the existing Monte Carlo and backtest.

### 2.2 Bayesian hierarchical pace model (the core)

A rank-ordered logit (Plackett–Luce) model, following van Kesteren and Bergkamp (2023), extended with season effects, qualifying pace, and a DNF hazard:

- Latent race pace for driver d in car c at race r of season s:
  `mu[d, r] = skill[d] + skill_season[d, s] + car[c, s] + car_form[c, r] + beta_grid * g(grid) + wet[d] * is_wet[r] + track[d, type[r]]`
- `skill[d]` is a driver's career skill (random walk across seasons: `skill[d, s] ~ Normal(skill[d, s-1], tau_skill)`), `skill_season[d, s]` is the season deviation with a tight prior (this is what credits a breakout year while it happens), `car[c, s]` is the constructor's season pace with a wide prior in regulation-change seasons, `car_form[c, r]` is a slow random walk within the season capturing upgrades.
- **Race likelihood**: Plackett–Luce over the classified finishing order: the probability the observed order happened is the product over positions of `softmax(mu)` restricted to the drivers not yet placed.
- **Qualifying likelihood**: the best qualifying lap gap to pole (in percent) is `Normal(alpha_r - kappa * mu_q[d, r], sigma_q)` where `mu_q` shares `skill`, `skill_season`, `car` with the race pace but has its own grid-free scale `kappa`. This is the second observation of the same latent pace and is what makes the model react to a new car's speed from the first qualifying of a season.
- **DNF hazard**: `P(dnf[d, r]) = logistic(h_driver[d] + h_car[c, s] + h_circuit[circuit[r]] + h_wet * is_wet[r])`, fit jointly.
- Output: posterior samples of `mu` for every entrant of the target race, plus DNF probabilities. The existing Monte Carlo draws one posterior sample per run instead of a point strength, so model uncertainty flows into the odds.

Inference: NumPyro NUTS on the Spark's GPU. A full fit on 2010 to 2026 (about 7,800 driver-races, roughly 2,000 parameters) is minutes on the GB10; a laptop CPU takes an hour. The walk-forward backtest needs one fit per race (about 70 fits for 2024 to 2026), which is why the Spark matters.

### 2.3 Metrics that judge the whole distribution

- **Ranked probability score (RPS)** per driver-race over the finishing-position distribution, averaged; the natural score for an ordinal outcome.
- **Top-3 set log loss**: the log probability the model assigned to the exact podium set.
- **Expected calibration error (ECE)** and reliability tables for win, podium and points probabilities, plus sharpness (mean max probability).
- **Skill score** versus the pole baseline: `1 - model / baseline` for log loss and RPS.
- **Bootstrap 90% confidence intervals** on every season and all-race metric (resampling races), so an "improvement" of 0.02 can be called noise when it is.
- **Nested rolling-origin evaluation**: tune on seasons before the fold, score the fold, roll forward. The reported number is the mean over folds, never a number the tuner saw.

### 2.4 Spark workflow

The Mac remains the day-to-day CLI. Heavy jobs run on the Spark over SSH:

- `spark/setup.sh` installs `uv`, Python 3.12, `jax[cuda12]` and `numpyro` for aarch64, and checks the GPU is visible.
- `make spark-sync` rsyncs the repo and `data/cache/` to `$SPARK_HOST`; `make spark-fit`, `make spark-backtest`, `make spark-tune` run the corresponding `f1pred` commands there and rsync `outputs/` and `data/cache/posteriors/` back.
- Everything has a CPU fallback with `--device cpu` so tests and small runs work anywhere; the test suite never needs a GPU.

## 3. Non-goals

- No neural networks; the hierarchical model is interpretable and the data is small.
- No lap-by-lap strategy simulation.
- No live-timing or telemetry sources; the data stays Hugging Face plus Open-Meteo.

## 4. Success criteria

Measured on the rolling-origin folds for 2024, 2025 and 2026 (through the latest completed round):

- Winner log loss beats the pole baseline in every fold, with the bootstrap interval excluding the baseline in at least two.
- Monza-2026-style upsets get at least 1% win probability from the back of the grid; no race scores worse than 7 (the `1e-3` floor) in log loss.
- RPS and top-3 log loss improve on the Phase 6 model in every fold.
- ECE below 0.03 for win probabilities over all folds combined.
- A full walk-forward backtest of 2024 to 2026 completes in under one hour on the Spark.

## 5. Phases

7a. Concave grid, RPS, ECE, bootstrap intervals, rolling-origin evaluation (CPU, one day).
7b. Qualifying-gap data and the Plackett–Luce model with qualifying likelihood and DNF hazard in NumPyro; single-fit prediction for the next race (Spark or CPU).
7c. Walk-forward posterior backtest, Spark make targets, comparison against Phase 6 with intervals, README.
