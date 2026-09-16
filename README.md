# f1-predictions

Race-by-race Formula 1 predictions from an Elo rating system and a Monte Carlo simulation.
Every driver and every constructor carries a rating learned from head-to-head results since
2010; a race is then simulated ten thousand times with shared team noise, individual driver
noise, a grid-position bonus and a per-driver retirement chance, and the finishing orders are
counted into win, podium, points and full position probabilities. A leakage-safe backtest
scores the model on past seasons against two baselines so the numbers below are honest.
Data comes from the Hugging Face dataset
[tracinginsights/RaceData](https://huggingface.co/datasets/tracinginsights/RaceData), a live
mirror of the Ergast schema.

## Example output

`f1pred predict --season 2026 --round 15`, run before qualifying (so without a grid):

```text
No qualifying data yet; lineup carried over from Spanish Grand Prix 2026 and grid ignored
Rain chance: 10% (forecast)
                    Azerbaijan Grand Prix 2026 (round 15)
 #   Driver                    Win   Podium   Points   Exp. pos   P10-P90
 1   Max Verstappen          20.3%    47.7%    79.1%        6.6      1-20
 2   Lando Norris            18.2%    46.2%    81.7%        6.3      1-19
 3   George Russell          15.5%    41.8%    80.8%        6.5      1-19
 4   Oscar Piastri           11.5%    35.6%    77.0%        7.5      1-20
 5   Andrea Kimi Antonelli   10.7%    35.7%    85.8%        6.1      1-13
 6   Charles Leclerc          8.6%    28.3%    69.5%        8.7      2-21
 7   Lewis Hamilton           6.4%    23.3%    74.0%        8.2      2-20
 8   Liam Lawson              2.1%    11.5%    66.3%        9.5      3-19
 9   Carlos Sainz             1.3%     4.4%    34.7%       13.0      6-21
10   Arvid Lindblad           1.1%     5.2%    53.5%       10.7      5-18
11   Alexander Albon          0.9%     2.6%    24.9%       14.2      7-21
12   Pierre Gasly             0.5%     3.2%    42.9%       11.6      6-18
13   Franco Colapinto         0.5%     2.7%    40.1%       11.7      6-17
14   Oliver Bearman           0.4%     1.9%    27.7%       13.5      7-20
15   Nico Hülkenberg          0.4%     1.8%    29.9%       13.5      7-21
16   Yuki Tsunoda             0.4%     1.8%    36.0%       12.2      7-18
17   Gabriel Bortoleto        0.4%     2.1%    32.7%       13.2      7-20
18   Esteban Ocon             0.4%     1.8%    27.6%       13.2      7-19
19   Fernando Alonso          0.3%     1.7%    21.9%       14.4      8-21
20   Lance Stroll             0.1%     0.4%     5.8%       17.5     12-22
21   Sergio Pérez             0.1%     0.3%     5.3%       17.3     12-21
22   Valtteri Bottas          0.0%     0.1%     2.6%       17.6     14-21
                  * fewer than 5 rated races: low confidence
```

The rain line is the Open-Meteo forecast for the three hours from the race start; it becomes the
share of simulated runs that are wet. Before the forecast horizon, or when Open-Meteo cannot be
reached, the circuit's historical wet rate is used instead and the line says so. The wide
P10-P90 bands at the top are the retirement chance at work: 2026 has been an 18% DNF season and
Baku's history multiplies that by 1.22, so even the favourites reach P20 in a tenth of runs.

The ratings behind it, after the 2026 Spanish Grand Prix (`f1pred ratings build`):

```text
 #   Driver             Rating       #   Constructor   Rating
 1   Max Verstappen     1754.4       1   mercedes      1740.2
 2   Lando Norris       1661.8       2   ferrari       1662.8
 3   Charles Leclerc    1651.0       3   mclaren       1661.5
 4   George Russell     1648.9
 5   Lewis Hamilton     1618.9
```

![Win probability](docs/results/example-win.png)

![Finishing position heatmap](docs/results/example-positions.png)

### Championship odds

`f1pred season --season 2026`, run with nine races left, chains the race simulation over the
rest of the calendar 2,000 times and counts who ends up on top:

```text
           2026 drivers' championship (9 races left, 2,000 runs)
 #   Driver                  Points now   Exp. points    Title    Top 3
 1   Andrea Kimi Antonelli          292           391    98.6%   100.0%
 2   George Russell                 211           321     1.2%    90.8%
 3   Lando Norris                   186           298     0.2%    66.0%
 4   Lewis Hamilton                 191           277     0.0%    27.9%
 5   Charles Leclerc                167           254     0.0%     7.2%
 6   Max Verstappen                 145           254     0.0%     8.0%
 7   Oscar Piastri                  120           210     0.0%     0.3%
 8   Liam Lawson                     59           101     0.0%     0.0%
```

The lineup for the remaining races is the modal driver per seat over the last three completed
races, so a one-off stand-in is not simulated for the rest of the year.

![Championship odds](docs/results/example-title.png)

### Driver profile

`f1pred profile` prints the three "mental edge" numbers for the drivers on the current grid,
computed over the last 20 races (6 for form) after the 2026 Spanish Grand Prix. The top and
bottom five by aggression:

```text
 Driver                  Aggression    Risk    Form   Races
 Fernando Alonso              +0.82   -0.67   +0.67      20
 Carlos Sainz                 +0.57   +0.67   +0.20      20
 Esteban Ocon                 +0.52   -0.67   +0.93      20
 Sergio Pérez                 +0.44   +0.67   +1.00      19
 Lance Stroll                 +0.37   +0.67   +0.92      19
 ...
 Andrea Kimi Antonelli        -0.33   -0.67   +0.94      20
 Valtteri Bottas              -0.72   +0.00   +1.14      20
 Gabriel Bortoleto            -0.79   +0.67   +0.88      18
 Nico Hülkenberg              -0.92   +0.00   +0.33      19
 George Russell               -0.92   +0.00   -0.93      19
```

Aggression is places gained relative to the field, so it rewards midfield drivers who start
behind and come through, and penalises front-row starters who have nowhere to go but back.
Risk is the accident-type retirement rate against the field's; it is populated again now that
retirements are classified from `positionText` (see the changelog), and over a 20-race window
it moves in steps because a driver has zero, one or two accidents in that span.

## How it works

**Ratings.** Each driver and each constructor starts at 1500. Every pair of classified
finishers in a race is one Elo matchup; the driver ahead scores 1, the driver behind 0, and both
ratings move by `K` times the surprise. Teammate matchups count double for the driver and not at
all for the constructor, because the car cancels out. Ratings regress toward 1500 at the start
of each season, constructors harder than drivers and harder still in rule-change years.
See [docs/elo.md](docs/elo.md).

**One simulated race.** A driver's performance is
`driver + constructor + grid_bonus * n / grid ** grid_shape` (concave since Phase 7a: pole
versus P2 is worth half the full bonus at `grid_shape = 1`, P19 versus P20 almost nothing)
plus one noise draw shared by both cars of the team and one private to the driver. Each driver
is retired with their own DNF probability and sent to the back; survivors are sorted by
performance. See [docs/monte-carlo.md](docs/monte-carlo.md).

**Monte Carlo.** That race is run 10,000 times as one NumPy matrix and the finishing orders
are counted: the share of runs a driver wins is their win probability, and the same counting
gives podium, points, expected position, its 10th and 90th percentiles, and a full
`P(driver finishes in position k)` matrix. Before qualifying the grid term is dropped and its
spread folded into the driver noise. See [docs/monte-carlo.md](docs/monte-carlo.md).

**DNF model.** Reliability is kept out of the Elo update. A driver's retirement chance blends
their recent DNF rate with their constructor's, shrinks it toward the field rate when the
evidence is thin, and scales it by the circuit's history, clipped to between half and double.
See [docs/dnf-model.md](docs/dnf-model.md).

**Season simulation.** Championship odds chain the same race simulation over every remaining
race and sprint, 2,000 times, adding points with that season's rules after each. Ratings are
held fixed within a simulated season, which under-estimates form swings and upgrades; the
fastest-lap bonus point (2019 to 2024) is not simulated. Drivers who have left the grid keep
their points, and constructor totals route each driver's simulated points to their current
team. See [docs/season-sim.md](docs/season-sim.md).

**Weather and track type.** `data update` asks [Open-Meteo](https://open-meteo.com/) how much
rain fell at each circuit in the three hours from the race start; 0.5 mm or more marks the race
wet. Every driver and constructor also carries a wet rating and one rating per hand-labelled
track type (street, high speed, high downforce, mixed), updated only on races of that kind and
shrunk toward the overall rating until there is enough evidence (eight races count for half).
For a future race the forecast rain probability becomes a coin flip inside every simulated run:
wet runs use the wet ratings, wider noise and higher DNF odds. When the forecast is out of
reach the circuit's historical wet rate stands in. See
[docs/weather-and-track.md](docs/weather-and-track.md).

**Driver profile (mental edge).** Three numbers per driver, each computed from the races before
the one being predicted and shrunk toward zero when the evidence is thin. *Aggression* is the
places a driver gains on lap 1 (from the lap-times table) and over the race, relative to the
field. *Risk* is their accident-type retirement rate relative to the field. *Form* is how far
their last six finishes beat the position their rating expected. Aggression and form enter the
simulation as pace, risk as wider driver noise and a higher DNF chance, all through scales that
default to zero and are set by the tuner; `f1pred profile` prints the current numbers. See
[docs/driver-profile.md](docs/driver-profile.md).

## Backtest results

> **Superseded.** This section records the Phase 6 model. The shipped model is Phase 7a, whose
> results are in [Phase 7a: concave grid and honest metrics](#phase-7a-concave-grid-and-honest-metrics)
> below. Kept for comparison.

Parameters were tuned by coordinate descent on 2015 to 2022 (`f1pred tune`) with the weather,
track and driver-profile knobs in the grid, after the review fixes listed in the changelog. It
chose `k_constructor = 48`, `sigma_team = 40`, `sigma_driver = 60`, `grid_bonus = 12`,
`regress_constructor = 0.3`, `shrink_wet = shrink_track = 16`, `wet_dnf_factor = 1.0`,
`aggression_scale = 5` and `risk_dnf_scale = 0.25`; everything else stayed at its default,
including `k_driver = 24`, `wet_noise_factor = 1.5` and the risk-noise and form scales at 0.
Training log loss was 1.3095. The seasons below were never seen by the tuner. Each race is
predicted from the ratings, DNF rates, grid, driver profile and the circuit's pre-race wet rate
that were known before it started; the rainfall that actually fell is not used. See
[docs/backtesting.md](docs/backtesting.md).

| Season | Races | Log loss model | Log loss pole | Log loss uniform | Brier model | Brier pole | Brier uniform | Spearman model | Spearman pole |
|-------:|------:|---------------:|--------------:|-----------------:|------------:|-----------:|--------------:|---------------:|--------------:|
| 2023 | 22 | 0.55 | 1.98 | 3.00 | 0.08 | 0.09 | 0.13 | 0.79 | 0.71 |
| 2024 | 24 | 1.74 | 2.89 | 2.99 | 0.08 | 0.09 | 0.13 | 0.80 | 0.78 |
| 2025 | 24 | 1.44 | 1.61 | 2.99 | 0.07 | 0.06 | 0.13 | 0.73 | 0.74 |

Log loss is the surprise at the actual winner (lower is better), Brier the squared error of the
podium probabilities (lower is better), Spearman the rank correlation of expected versus actual
finishing position (higher is better). The "pole" baseline gives the pole sitter a 90% win
chance and spreads the rest evenly; "uniform" gives every driver the same chance.

The model beats both baselines on winner log loss in every season and beats uniform on every
score. It is weakest where the grid already says most of what there is to say: in 2025 the
pole baseline edges it on podium Brier and Spearman.

### Did weather, track type and the driver profile help?

`f1pred backtest --seasons 2023-2025 --ablate` scores the same seasons with each addition
switched off and on, using the tuned parameters throughout (`base` is the Phase 3 model with
those parameters). `full` is weather plus track type with the profile off; `profile` is
everything, and is what `f1pred predict` runs. Log loss is the winner score, Brier the podium
score; lower is better for both. The first table is the default, honest mode: every race gets
the circuit's wet rate as known before it. The second (`--observed-rain`) tells the weather
variants which races were actually wet, an upper bound on what a perfect forecast could add.

Pre-race rain (`rain_mode = historical`, the default):

| Variant | Log loss 2023 | 2024 | 2025 | All 70 | Brier 2023 | 2024 | 2025 | All 70 |
|:--------|--------------:|-----:|-----:|-------:|-----------:|-----:|-----:|-------:|
| base (none) | 0.401 | 2.184 | 1.444 | 1.370 | 0.0832 | 0.0842 | 0.0698 | 0.0789 |
| weather | 0.430 | 1.886 | 1.429 | 1.271 | 0.0812 | 0.0823 | 0.0681 | 0.0771 |
| track | 0.476 | 1.827 | 1.386 | 1.251 | 0.0827 | 0.0817 | 0.0710 | 0.0783 |
| full (weather + track) | 0.560 | 1.724 | 1.427 | 1.256 | 0.0808 | 0.0808 | 0.0698 | 0.0770 |
| profile (everything, the default) | 0.552 | 1.738 | 1.443 | 1.264 | 0.0804 | 0.0811 | 0.0707 | 0.0773 |

Observed rain (`rain_mode = observed`, `--observed-rain`; `base` and `track` do not use rain
and are unchanged):

| Variant | Log loss 2023 | 2024 | 2025 | All 70 | Brier 2023 | 2024 | 2025 | All 70 |
|:--------|--------------:|-----:|-----:|-------:|-----------:|-----:|-----:|-------:|
| base (none) | 0.401 | 2.184 | 1.444 | 1.370 | 0.0832 | 0.0842 | 0.0698 | 0.0789 |
| weather | 0.416 | 2.063 | 1.478 | 1.345 | 0.0824 | 0.0817 | 0.0708 | 0.0782 |
| track | 0.476 | 1.827 | 1.386 | 1.251 | 0.0827 | 0.0817 | 0.0710 | 0.0783 |
| full (weather + track) | 0.518 | 1.873 | 1.428 | 1.295 | 0.0823 | 0.0827 | 0.0722 | 0.0790 |
| profile (everything, the default) | 0.512 | 1.894 | 1.442 | 1.304 | 0.0820 | 0.0830 | 0.0730 | 0.0792 |

Track type remains the largest single gain on the 70-race averages (winner log loss 1.370 to
1.251), and still unevenly: it cuts 2024 from 2.18 to 1.83 but makes 2023 worse.

The weather rows need a careful reading. With the pre-race rate, `weather` improves on `base`
by 0.10 of winner log loss, yet with the observed rainfall, which is strictly more information,
it improves by only 0.025. A layer that gains more from a vague 10 to 20% chance at every race
than from knowing exactly which races were wet is not extracting weather signal; the gain comes
from mixing a slice of wider-noise wet runs into every race, which softens the win
probabilities and is rewarded on seasons with surprise winners like 2024. Treat the weather
layer as a regulariser, not a forecast.

The driver profile still does not help. The tuner kept `aggression_scale = 5` and
`risk_dnf_scale = 0.25` above zero, but on the held-out seasons the `profile` row is 0.008
worse than `full` on winner log loss over 70 races and 0.0003 worse on podium Brier. The
profile stays on by default because that is what the tuner chose on the training seasons; the
honest summary is that the grid, the overall ratings and the track-type ratings already carry
the signal, and a 20-race window of places gained and accidents is too noisy to add to it.

![Calibration](docs/results/backtest-calibration.png)

### Phase 7a: concave grid and honest metrics

The 2026 season showed that a linear grid term cannot price an upset: with each slot worth 25
rating points, Antonelli's Monza win from P19 was given 0.01% (log loss 9.2 in one race). The
grid term is now `grid_bonus * n / grid ** grid_shape`, concave, so the front rows matter a lot
and the back rows barely at all. Re-tuning on 2015 to 2023 with `grid_shape` in the search space
(`f1pred tune --train 2015-2023 --test 2024-2026 --passes 2 --runs 1500`) chose
`grid_shape = 0.75`, `grid_bonus = 8`, `k_driver = 64`, `k_constructor = 48`,
`teammate_weight = 3`, `regress_driver = 0.2`, `regress_constructor = 0.1`,
`shrink_wet = shrink_track = 32`, `wet_dnf_factor = 1.0`, `shrink_profile = 2` and
`risk_dnf_scale = 0.25`; training log loss fell from 1.2034 (linear term) to 1.1522. Monza 2026
now gets a 6.0% win chance for Antonelli (log loss 2.8), and the worst race of 2024 to 2026 is
Miami 2024 at 5.6, under the spec's ceiling of 7.

Every score below carries a 90% bootstrap interval (1,000 resamples of the season's races;
[docs/metrics.md](docs/metrics.md)). The folds are rolling-origin: 2024 to 2026 were never seen
by the tuner, and `backtest --rolling 2024-2026` replays the ratings per fold. RPS is the ranked
probability score over each driver's full finishing-position distribution (lower is better;
flat is about 0.16); top-3 is the log loss of the exact podium set (uniform is 7.0).

| Fold | Races | Log loss model | Log loss pole | RPS model | RPS pole | Top-3 model | Top-3 pole |
|-----:|------:|:---------------|:--------------|:----------|:---------|------------:|-----------:|
| 2024 | 24 | 1.69 [1.24, 2.17] | 2.89 [2.03, 3.74] | 0.098 [0.089, 0.108] | 0.123 [0.110, 0.137] | 3.78 | 7.31 |
| 2025 | 24 | 1.20 [1.00, 1.42] | 1.61 [0.96, 2.46] | 0.109 [0.100, 0.118] | 0.132 [0.114, 0.149] | 3.33 | 5.79 |
| 2026 | 14 | 1.69 [1.32, 2.03] | 1.98 [0.85, 3.10] | 0.098 [0.087, 0.110] | 0.116 [0.102, 0.132] | 3.91 | 7.26 |

Over all 62 races (`backtest --seasons 2024-2026 --bootstrap 1000`): winner log loss 1.50
[1.27, 1.74] against the pole baseline's 2.19 [1.61, 2.69]; RPS 0.102 [0.096, 0.109] against
0.125 [0.115, 0.134]; expected calibration error 0.011 for win probabilities and 0.023 for
podium probabilities; sharpness (mean top win probability) 0.50. The model beats the pole
baseline on winner log loss in every fold, and its interval excludes the pole number in 2024 and
2025 but not in 2026 (14 races). It beats pole on RPS in every fold with non-overlapping
intervals, and on the podium set by 2.5 to 3.5 nats.

Read the intervals before the means: a full point of 2024 winner log loss is the width of the
interval, so the difference between this table and the Phase 6 one (1.74, 1.44 for 2024 and
2025) is inside the noise. What is not noise is the floor: no race can cost more than about 6
any more, and the same is true on the untuned defaults.

Ablation on the same 62 races with the Phase 7a parameters, winner log loss (pre-race rain):
base 1.594, weather 1.537, track 1.504, full 1.494, profile (the default) 1.500. The ordering
is the Phase 6 one and every gap is inside the intervals above.

The remaining structural limits, slow adaptation to a breakout season and no use of qualifying
lap times, are the subject of Phase 7b, the Bayesian pace model
([docs/superpowers/specs/2026-09-16-accuracy-upgrade-design.md](docs/superpowers/specs/2026-09-16-accuracy-upgrade-design.md)).

## Install and usage

```bash
uv sync
uv run f1pred data update                          # download tables into data/cache/, build the driver-race table
uv run f1pred ratings build                        # replay 2010 onward, write pre-race ratings, print the top 10
uv run f1pred predict --season 2026 --round 15     # or --race monza; saves outputs/2026-15-win.png and -positions.png
uv run f1pred season --season 2026                 # title odds for the rest of the season; saves outputs/2026-title.png and CSVs
uv run f1pred backtest --seasons 2023-2025         # score vs baselines; saves outputs/calibration.png and CSVs
uv run f1pred profile                              # aggression, risk and form for the drivers on the current grid
uv run f1pred backtest --ablate                    # also score with and without weather, track type and the driver profile; saves outputs/ablation.csv
uv run f1pred backtest --ablate --observed-rain    # same, but the weather variants see the race-day rainfall (an upper bound)
uv run f1pred backtest --seasons 2024-2026 --bootstrap 1000   # add 90% bootstrap intervals; saves outputs/backtest_intervals.csv
uv run f1pred backtest --rolling 2024-2026 [--tune]           # rolling-origin folds (optionally re-tuned per fold); saves outputs/rolling.csv
uv run f1pred tune --train 2015-2023 --test 2024-2026         # coordinate descent, held-out report, writes f1pred/tuned.json
```

Simulation commands take `--runs N`, `--seed N` and `--no-grid`; `predict` takes
`--rain 0.3` to override the rain probability; every command takes `--cache-dir DIR` and the
ones that write charts take `--out DIR`. `data update` fetches race-day rain from Open-Meteo
(no key needed) and caches it in `data/cache/weather.parquet`, so later updates only ask about
new races. It also downloads the 25 MB `lap_times.csv` and keeps only the lap-1 rows in
`data/cache/lap1.parquet`, for the aggression measure.

## Roadmap

Done: data and the driver-race table (1), Elo ratings (2), the race and season Monte Carlo with
the backtest and tuner (3), weather and track type (4), the driver profile (5), the review fixes
(6) and the concave grid with distribution-aware metrics and intervals (7a). Next, per the
Phase 7 plan in `docs/superpowers/plans/`:

- 7b: a Bayesian hierarchical Plackett–Luce pace model (NumPyro) with a qualifying lap-gap
  likelihood, season-level driver effects and a DNF hazard, feeding posterior samples into the
  existing simulation.
- 7c: the walk-forward posterior backtest on the DGX Spark and the comparison against Phase 6
  with intervals.

Unscheduled: ratings that update inside a simulated season, and a small Streamlit dashboard
over the cached outputs.

## Credits

Data: [tracinginsights/RaceData](https://huggingface.co/datasets/tracinginsights/RaceData),
derived from the Ergast and [Jolpica](https://github.com/jolpica/jolpica-f1) F1 databases.
Weather: [Open-Meteo](https://open-meteo.com/) historical archive and forecast APIs
(CC BY 4.0).

Prior art consulted: mar-antaya's `2025_f1_predictions` and `2026_f1_predictions`
(per-race gradient boosting on qualifying times); villekuosmanen's `F1Predict` (driver,
constructor and engine Elo with Monte Carlo perturbation, the closest relative of this design);
van Kesteren and Bergkamp 2023, JQAS (a Bayesian model showing constructors explain most of the
variance in results); Kevocado's `F1_Predictor` and neevj2006's `F1_Race_Predictor` (separate
reliability layer, shared team noise, leakage-safe backtests).

## Changelog

Phase 7a (concave grid and honest metrics):

- The grid term is `grid_bonus * n / grid ** grid_shape` instead of `grid_bonus * (n - grid)`;
  `grid_shape` is tuned (0.75) and the no-grid fallback folds the term's actual spread into the
  driver noise. Monza 2026 moves from a 0.01% to a 6% win chance for the winner.
- New scores per race: ranked probability score over the finishing-position distribution and
  the log loss of the exact podium set, for the model and both baselines; expected calibration
  error and sharpness per run. `backtest --bootstrap R` adds 90% intervals,
  `backtest --rolling 2024-2026 [--tune]` scores folds the tuner never saw
  ([docs/metrics.md](docs/metrics.md)).
- `tune` searches `grid_shape` and a smaller `grid_bonus` range; the objective can be any race
  score column.

Fixes from the post-Phase-5 code review (Phase 6); every number in the Phase 6 tables was
regenerated after them.

- Finishers are classified from `positionText`, not `position`. From 2025 the upstream CSV
  fills `position` with `positionOrder` for retirements (`positionText` is `R`, `W`, `D`, ...),
  so every 2025 and 2026 retirement had been rated as a finish and the DNF rate collapsed toward
  zero; it is now 10% for 2025 and 18% for 2026.
- Administrative statuses (`Disqualified`, `Withdrew`, `Not classified`, `Underweight`, ...)
  and any status not on the mechanical or accident lists are `other`, not `mechanical`; an
  unknown status is logged once so new upstream values surface.
- The DNF circuit factor divides the circuit's rate by the all-time global rate of the same
  rows instead of the recent-window rate, which had pinned 11 of 35 circuits at the 2.0 clip;
  the factors now span 0.5 to 1.6 with one at a clip. The evidence count is in races, not rows.
- `predict`, `backtest`, `season` and `profile` stop with a clear message when the ratings are
  older than the driver-race table, and the replay fails on a table missing the weather, track
  or lap-1 columns instead of silently running dry and mixed.
- The backtest gives each race the circuit's pre-race wet rate by default; `--observed-rain`
  restores the race-day rainfall as a labelled upper bound.
- The no-qualifying note names the race the lineup was carried over from, and season odds use
  the modal driver per seat over the last three completed races so a one-off stand-in is not
  simulated for the rest of the year.

## License

MIT. See [LICENSE](LICENSE).
