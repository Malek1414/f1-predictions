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
  that failed on Antonelli in 2026. Its raw is a **Student-t**, not a Normal: a Normal pooled
  over 394 driver-seasons that genuinely deviate very little drove `tau_season` to 0.07 and left
  a real breakout nowhere to go. The heavy tail is the standard sparse construction — the bulk
  still shrinks to near zero, but one exceptional season is not forced down with it. The degrees
  of freedom `NU_SEASON` are fixed at 4 rather than sampled, so the non-centred funnel does not
  get a second global parameter on top of it.
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
of the qualifying signal. The Student-t reads them as tail. The degrees of freedom are
**bounded away from the Cauchy region**: `nu_q = 2 + Gamma(2, 0.5)`, sampled as `nu_q_raw` and
reported as the derived quantity, so `nu_q >= 2` and the likelihood always has a finite variance,
with the prior's mass around 4 to 8. Left free on a `Gamma(2, 0.1)` it went to 1.42 — effectively
a Cauchy, and a Cauchy plus a per-race intercept makes a session that was split wet/dry genuinely
bimodal, because there is a second mode over which half of the field to believe. Spa 2011,
Australia 2013 and Zandvoort 2023 are exactly that, and they took three `alpha` intercepts to
r_hat 12.7 with an ESS of 2. Gaps above `MAX_PLAUSIBLE_GAP_PCT` (100%) are dropped before the
model sees them — a lap twice as slow as pole is a source error, not a lap.

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

## Recent seasons count for more

All three likelihoods discount an observation by the age of its season. The reason is plain
enough: inside a single season the regulations, the car, the teammate and the calendar are all
held fixed, so "Antonelli beat Russell at Silverstone" is a far cleaner comparison than "Antonelli
in 2026 was quicker than Russell in 2022", where the car, the rules and the field have all moved
underneath the comparison. Older seasons still carry real information — they are most of what we
know about a driver who has changed teams — but they should not count the same as what is
happening now.

The discount is exponential in the age of the season:

```
w[d, r] = 0.5 ** ((latest_season - season[r]) / season_half_life)
```

`season_half_life` is a `ModelParams` field. It ships as `inf`, meaning the discount is off,
because at a half-life of 2 seasons it shrinks every variance component and freezes the driver
terms rather than sharpening them; the walk-forward searches it instead. With a finite half-life
of 2 the latest season counts
full, a season two years older counts half, four years older a quarter. The weight multiplies the
per-race Plackett-Luce log likelihood and scales the qualifying and DNF sites
(`numpyro.handlers.scale`). **The priors are not weighted** — the random walks still pool over
the whole history, which is what stops a discounted season from simply being forgotten.
`season_half_life = float("inf")` sets every weight to exactly 1.0 and recovers the unweighted
likelihood, which is how the two are compared.

One thing the discount does that is not obvious: it raises `tau_season`. The season deviation is
pooled over every driver-season in the table, and the overwhelming majority of them genuinely
deviate very little, so they drag the shared scale toward zero and take any real breakout with
it. Counting the old ones for less leaves them less able to do that. On synthetic data with a
planted breakout in the latest season, `tau_season` and the recovered effect both come up
(`tests/test_pace_model.py`). It only works when there is a run-up to discount: with only two
prior seasons the discount costs more contrast than it buys, because those prior seasons are the
only evidence separating a one-year deviation from a step in career skill.

## Priors

| Parameter | Prior |
| --- | --- |
| `skill[d, s]` | `Normal(skill[d, s-1], tau_skill)`; first season `Normal(0, 1)` |
| `skill_season[d, s]` | `tau_season * StudentT(NU_SEASON, 0, 1)`, `NU_SEASON` fixed at 4 |
| `car[c, s]` | `Normal(car[c, s-1], tau_car_reg if regulation else tau_car)`; first season `Normal(0, 1.5)` |
| `car_form[c, r]` | `Normal(car_form[c, r-1], tau_form)`; 0 at each season start |
| `alpha[r]` | `Normal(0, 5)` (percent units) |
| `wet[d]`, `track[d, t]` | `Normal(0, tau_wet)`, `Normal(0, tau_track)` |
| `beta_grid` | `Normal(0, 1)` |
| `kappa`, `sigma_q` | `HalfNormal(2)`, `HalfNormal(1)` |
| `nu_q` | `2 + Gamma(2, 0.5)`, sampled as `nu_q_raw` and reported as the sum |
| `tau_skill`, `tau_season` | `HalfNormal(0.3)`, `HalfNormal(1.0)` |
| `tau_car`, `tau_car_reg`, `tau_form` | `HalfNormal(0.5)`, `HalfNormal(1.5)`, `HalfNormal(0.1)` |
| `tau_wet`, `tau_track` | `HalfNormal(0.3)` both |
| `h0` | `Normal(-2, 1)` |
| `h_driver`, `h_car`, `h_circuit`, `h_wet` | `Normal(0, 0.5)` |

Every random walk is non-centred: a unit-scale step is sampled and scaled by its `tau` (standard
normal everywhere except the season deviation, whose raw is the Student-t above), and
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
cores; on GPU the chains are vectorised instead. On an M2 the full 2010–2026 table (7,223 rows,
5,657 parameters) takes about 11 minutes end to end, compilation included, for 1,000 warmup and
1,000 samples on each of 4 chains — 21 minutes with the season age discount off, which leaves
five times as much effective data to sample against. `--device gpu` is the same code on the
Spark.

## Reading `summary()`

One row per parameter with `mean`, `sd`, the 5th and 95th percentiles (`lo`, `hi`, a 90%
credible interval), `r_hat` and `ess`.

- `skill[max_verstappen,2026]` is career skill at that season; `skill_season[...]` is the extra
  this year only. A driver having a genuine breakout shows a large positive `skill_season` with
  an interval clear of zero. Calibrate your expectations: on the 2010–2026 table `tau_season`
  fits at 0.031 [0.003, 0.074] with a half-life of 2 and 0.063 [0.009, 0.111] with the discount off (the default),
  and no driver-season in the 394 reaches a standardised deviation of 3, so "large" here means a
  tenth or two of a pace unit. Antonelli in 2026 is the biggest on the current grid at
  +0.02 [-0.05, +0.12] and does not clear zero.
- `nu_q` fits at 2.09 [2.02, 2.20] — hard against its floor. The qualifying tail genuinely wants
  a Cauchy; the floor is a deliberate constraint for the sampler's sake, not a finding.
- The default fit's `max_r_hat` is 1.061, just over the 1.05 below. It is the per-race qualifying
  intercepts, which no prediction reads, but `car` is at 1.047 and worth watching. Without the
  season age discount the same fit is at 1.044 throughout.
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
