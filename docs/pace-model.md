# The Bayesian pace model

Phase 7b. Code: `f1pred/pace/`. Spec: `docs/superpowers/specs/2026-09-16-accuracy-upgrade-design.md`
section 2.2. The Elo model (`docs/elo.md`) stays the default until the walk-forward comparison
on the Spark says otherwise.

## The idea in one paragraph

Give every driver in every race one number, its **pace**. A faster car and a better driver both
raise it. Three things we observe are then explained by that one number at once: who finished
ahead of whom, how far off pole the driver qualified, and whether the car reached the flag. All
three likelihoods are fit jointly by NUTS, so a fast qualifying lap in the first race of a new
season moves a car's pace immediately instead of waiting for a dozen race results to drag an
Elo rating along. The output is not a point estimate: it is a few thousand posterior samples of
every entrant's pace, which the Monte Carlo then draws from run by run, so model uncertainty
ends up inside the odds.

## The pace

For driver `d` in car `c` at race `r` of season `s`:

```
mu[d, r] = skill[d, s] + skill_season[d, s] + car[c, s] + car_form[c, r]
           + beta_grid * g(grid) + wet[d] * is_wet[r] + track[d, type[r]]
```

- `skill[d, s]` is career skill. It is a **random walk across seasons**: this year starts where
  last year finished and drifts by `tau_skill`. It is deliberately slow.
- `skill_season[d, s]` is the season deviation, drawn fresh each year with a wider `tau_season`.
  This is the term that credits a breakout season *while it is happening* — the part of Phase 6
  that failed on Antonelli in 2026.
- `car[c, s]` is the constructor's season pace, also a random walk, but with a much wider step
  in a regulation season (`tau_car_reg`, 2014/2017/2022/2026) because the field reshuffles.
- `car_form[c, r]` is a slow walk over the constructor's races inside a season, reset to zero at
  each season start. It is the upgrade curve.
- `g(grid) = 1 / grid ** 0.75` is the concave grid term from Phase 7a, with the shape fixed
  inside the Bayesian model (`f1pred.pace.design.GRID_SHAPE`); the tuner's `grid_shape` only
  applies to the Elo path.
- `wet[d]` and `track[d, t]` reuse Phase 4's `is_wet` flag and four track types.

The pace is on the Plackett–Luce unit scale: the race noise is a standard Gumbel, so one pace
unit is a meaningful head-to-head edge and no separate `sigma_race` is estimated.

## The three likelihoods

**Finishing order (Plackett–Luce).** For one race with classified finishers in order
`(1), (2), ..., (k)`:

```
log P(order) = sum_{j=1..k} [ mu[(j)] - logsumexp(mu[(j)], mu[(j+1)], ..., mu[(k)]) ]
```

Read it as: the winner beat the whole field, the runner-up beat everyone still running, and so
on. It is implemented in `plackett_luce_logp` by laying each race's paces out in finishing order
into a padded matrix and taking a reverse cumulative `logsumexp` (`jax.lax.associative_scan` over
`logaddexp`). Non-classified drivers are not in this product at all; they only enter the hazard.
The likelihood is invariant to adding a constant to every pace in a race, which is why the
overall level is pinned only by the priors.

**Qualifying gap.** `gap[d, r]` is `100 * (best lap - pole lap) / pole lap`, in percent
(`f1pred/data/qualifying.py`), available for about 99% of driver-races since 2010.

```
gap[d, r] ~ StudentT(nu_q, alpha[r] - kappa * mu_q[d, r], sigma_q)
```

The gap has a long right tail: since 2010 the median is 1.8% but 8.4% of laps are more than 5%
off pole, 1.6% more than 10%, and the worst is 45.6%. Those are wet and red-flagged sessions,
not slow cars, and under a Normal likelihood they set `sigma_q` on their own and leave nothing
of the qualifying signal. The Student-t reads them as tail; `nu_q ~ Gamma(2, 0.1)` is the usual
weakly-informative degrees-of-freedom prior, with room to run high and behave like the Normal if
the tail is not there. Gaps above `MAX_PLAUSIBLE_GAP_PCT` (100%) are dropped before the model
sees them — a lap twice as slow as pole is a source error, not a lap.

`mu_q` is the same pace **without** the grid, wet and track terms — a car and driver measured on
an empty track. `alpha[r]` is a per-race intercept that absorbs track length and session
conditions, and `kappa > 0` converts pace units into percent. This is the second observation of
the same latent quantity and it is what makes the model react to a new car's speed from the
first Saturday of a season.

**DNF hazard.**

```
logit P(dnf[d, r]) = h0 + h_driver[d] + h_car[c, s] + h_circuit[circuit[r]] + h_wet * is_wet[r]
```

Fit jointly, so a fragile car is not mistaken for a slow one.

## Priors

| Parameter | Prior |
| --- | --- |
| `skill[d, s]` | `Normal(skill[d, s-1], tau_skill)`; first season `Normal(0, 1)` |
| `skill_season[d, s]` | `Normal(0, tau_season)` |
| `car[c, s]` | `Normal(car[c, s-1], tau_car_reg if regulation else tau_car)`; first season `Normal(0, 1.5)` |
| `car_form[c, r]` | `Normal(car_form[c, r-1], tau_form)`; 0 at each season start |
| `alpha[r]` | `Normal(0, 5)` (percent units) |
| `wet[d]`, `track[d, t]` | `Normal(0, tau_wet)`, `Normal(0, tau_track)` |
| `beta_grid` | `Normal(0, 1)` |
| `kappa`, `sigma_q` | `HalfNormal(2)`, `HalfNormal(1)` |
| `nu_q` | `Gamma(2, 0.1)` |
| `tau_skill`, `tau_season` | `HalfNormal(0.3)`, `HalfNormal(0.5)` |
| `tau_car`, `tau_car_reg`, `tau_form` | `HalfNormal(0.5)`, `HalfNormal(1.5)`, `HalfNormal(0.1)` |
| `tau_wet`, `tau_track` | `HalfNormal(0.3)` both |
| `h0` | `Normal(-2, 1)` |
| `h_driver`, `h_car`, `h_circuit`, `h_wet` | `Normal(0, 0.5)` |

Every random walk is non-centred: a standard-normal step is sampled and scaled by its `tau`, and
the walk itself is a segmented cumulative sum (`chain_layout` lays each chain out end to end, so
394 driver-seasons and 3,600 constructor-races cost one `cumsum` each rather than a Python loop).

## Identifiability

Driver and car effects are separated by teammates (same car, two drivers) and by driver moves
(same driver, two cars), as in van Kesteren and Bergkamp (2023). The random-walk priors carry
information across seasons where nobody moved. Both likelihoods are invariant to a per-race
constant, so the absolute level of `mu` is fixed only by the first-season priors; differences
within a race — the only thing the simulation uses — are what the data identifies.

## Fitting

```
uv run f1pred pace fit --device cpu            # writes data/cache/posteriors/latest.npz
uv run f1pred pace fit --cutoff 2026-07-01     # only races before that date
uv run f1pred pace summary                     # current-season skills and car pace
```

`fit()` is NUTS with `target_accept_prob=0.9`. Defaults are `pace_num_warmup=1000`,
`pace_num_samples=1000`, `pace_chains=4`, `pace_device="cpu"` in `ModelParams`. On CPU the chain
count is also passed to `numpyro.set_host_device_count`, so the chains really do run on separate
cores; on GPU the chains are vectorised instead. On an M2 the full 2010–2026 table (about 7,200
rows, 5,600 parameters) compiles in roughly two minutes and then samples four chains in under
ten. `--device gpu` is the same code on the Spark.

## Reading `summary()`

One row per parameter with `mean`, `sd`, the 5th and 95th percentiles (`lo`, `hi`, a 90%
credible interval), `r_hat` and `ess`.

- `skill[max_verstappen,2026]` is career skill at that season; `skill_season[...]` is the extra
  this year only. A driver having a genuine breakout shows a large positive `skill_season` with
  an interval clear of zero.
- `car[mercedes,2026]` is that constructor's season pace. Compare teams within one season; the
  level is not comparable across seasons in isolation.
- `kappa` should be positive: it is how many percent of lap time one pace unit buys.
- `beta_grid` is the value of track position on top of pace.
- Health check before believing any of it: `r_hat < 1.05` on the reported parameters and
  divergences under 1% of the draws. Both are printed by `f1pred pace fit`.

## From posterior to odds

`f1pred/pace/bridge.py` turns the posterior into simulation inputs: run `i` of the Monte Carlo
uses posterior sample `i % S`. The grid term, the team correlation and the wet and track effects
are already inside `mu`, so the simulation sets `sigma_team = 0`, skips its own grid term, and
uses a driver noise of `pace_scale * pi / sqrt(6)` — the Normal with the same variance as the
Gumbel the Plackett–Luce model assumes. `pace_scale = 400 / ln(10)` maps one pace unit onto the
Elo points the reports already speak in.
