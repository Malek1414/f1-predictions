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
No qualifying data yet; running without grid
Rain chance: 10% (forecast)
                    Azerbaijan Grand Prix 2026 (round 15)
 #   Driver                    Win   Podium   Points   Exp. pos   P10-P90
 1   Andrea Kimi Antonelli   17.9%    45.7%    87.5%        5.3      1-11
 2   George Russell          17.9%    44.4%    85.3%        5.6      1-12
 3   Max Verstappen          16.2%    41.2%    85.9%        5.6      1-12
 4   Lando Norris            13.0%    35.3%    82.2%        6.2      1-13
 5   Oscar Piastri            7.8%    25.1%    74.7%        7.5      2-15
 6   Charles Leclerc          7.5%    25.8%    75.1%        7.5      2-15
 7   Lewis Hamilton           5.3%    19.0%    68.2%        8.5      2-16
 8   Carlos Sainz             2.5%     8.7%    44.0%       11.5      4-19
 9   Liam Lawson              2.0%     9.4%    50.5%       10.7      4-18
10   Arvid Lindblad           1.9%     7.9%    45.4%       11.2      4-19
11   Alexander Albon          1.6%     5.4%    30.3%       13.3      5-20
12   Franco Colapinto         1.2%     6.0%    40.7%       12.0      5-19
13   Fernando Alonso          1.0%     3.4%    26.5%       13.9      6-20
14   Gabriel Bortoleto        0.8%     4.1%    33.9%       12.9      5-20
15   Yuki Tsunoda             0.8%     3.2%    30.1%       13.4      6-20
16   Esteban Ocon             0.7%     3.8%    31.2%       13.2      6-20
17   Pierre Gasly             0.7%     4.0%    33.7%       13.0      6-20
18   Oliver Bearman           0.5%     3.0%    25.9%       14.1      7-21
19   Nico Hülkenberg          0.4%     2.6%    24.1%       14.4      7-21
20   Sergio Pérez             0.2%     1.1%    13.5%       16.3      9-22
21   Lance Stroll             0.1%     0.7%     7.7%       17.8     12-22
22   Valtteri Bottas          0.0%     0.2%     3.7%       19.0     14-22
                  * fewer than 5 rated races: low confidence
```

The rain line is the Open-Meteo forecast for the three hours from the race start; it becomes the
share of simulated runs that are wet. Before the forecast horizon, or when Open-Meteo cannot be
reached, the circuit's historical wet rate is used instead and the line says so.

![Win probability](docs/results/example-win.png)

![Finishing position heatmap](docs/results/example-positions.png)

### Championship odds

`f1pred season --season 2026`, run with nine races left, chains the race simulation over the
rest of the calendar 2,000 times and counts who ends up on top:

```text
           2026 drivers' championship (9 races left, 2,000 runs)
 #   Driver                  Points now   Exp. points    Title    Top 3
 1   Andrea Kimi Antonelli          292           393    98.5%   100.0%
 2   George Russell                 211           319     1.4%    92.0%
 3   Lando Norris                   186           283     0.1%    51.2%
 4   Lewis Hamilton                 191           273     0.0%    31.0%
 5   Max Verstappen                 145           255     0.0%    13.3%
 6   Charles Leclerc                167           255     0.0%    12.4%
 7   Oscar Piastri                  120           188     0.0%     0.0%
 8   Liam Lawson                     59            93     0.0%     0.0%
```

![Championship odds](docs/results/example-title.png)

### Driver profile

`f1pred profile` prints the three "mental edge" numbers for the drivers on the current grid,
computed over the last 20 races (6 for form) after the 2026 Spanish Grand Prix. The top and
bottom five by aggression:

```text
 Driver                  Aggression    Risk    Form   Races
 Fernando Alonso              +0.99   +0.00   +0.94      20
 Esteban Ocon                 +0.93   +0.00   +0.25      20
 Carlos Sainz                 +0.79   +0.00   -0.31      20
 Sergio Pérez                 +0.77   +6.67   +0.50      19
 Lance Stroll                 +0.59   +0.00   +0.94      20
 ...
 Gabriel Bortoleto            -0.52   +0.00   +1.00      20
 Lando Norris                 -0.72   +0.00   +0.50      20
 Nico Hülkenberg              -1.22   +0.00   +1.12      20
 George Russell               -1.34   +0.00   -2.19      20
 Oscar Piastri                -1.40   +0.00   -0.81      20
```

Aggression is places gained relative to the field, so it rewards midfield drivers who start
behind and come through, and penalises front-row starters who have nowhere to go but back.
Risk is nearly empty on this grid because the data source has recorded a classified finishing
position for every driver since the start of 2025, retirements included, so no 2025 or 2026
race carries an accident-type DNF; the two non-zero values (Pérez, Bottas) are drivers whose
20-race windows reach back into 2024, measured against a near-zero field rate. Treat the risk
column as unreliable until the source records retirements again.

## How it works

**Ratings.** Each driver and each constructor starts at 1500. Every pair of classified
finishers in a race is one Elo matchup; the driver ahead scores 1, the driver behind 0, and both
ratings move by `K` times the surprise. Teammate matchups count double for the driver and not at
all for the constructor, because the car cancels out. Ratings regress toward 1500 at the start
of each season, constructors harder than drivers and harder still in rule-change years.
See [docs/elo.md](docs/elo.md).

**One simulated race.** A driver's performance is `driver + constructor + grid_bonus * (n - grid)`
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

Parameters were tuned by coordinate descent on 2015 to 2022 (`f1pred tune`) with the weather,
track and driver-profile knobs in the grid. It chose `k_driver = 32`, `sigma_team = 40`,
`grid_bonus = 12`, `regress_constructor = 0.3`, `shrink_wet = shrink_track = 16`,
`wet_noise_factor = 1.0` (no extra noise in the wet) and `aggression_scale = 10`; the risk and
form scales stayed at 0 and everything else at its default, including `wet_dnf_factor = 1.5`,
which the Phase 4 tune had set to 1.0 (the two values differ by less than 0.001 in training
log loss, so the pick is noise). The seasons below were never seen by the tuner. Each race is
predicted from the ratings, DNF rates, grid, observed weather and driver profile that were
known before it started. See [docs/backtesting.md](docs/backtesting.md).

| Season | Races | Log loss model | Log loss pole | Log loss uniform | Brier model | Brier pole | Brier uniform | Spearman model | Spearman pole |
|-------:|------:|---------------:|--------------:|-----------------:|------------:|-----------:|--------------:|---------------:|--------------:|
| 2023 | 22 | 0.59 | 1.98 | 3.00 | 0.08 | 0.09 | 0.13 | 0.79 | 0.71 |
| 2024 | 24 | 1.76 | 2.89 | 2.99 | 0.08 | 0.09 | 0.13 | 0.80 | 0.78 |
| 2025 | 24 | 1.35 | 1.61 | 2.99 | 0.07 | 0.06 | 0.13 | 0.63 | 0.65 |

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
score; lower is better for both.

| Variant | Log loss 2023 | 2024 | 2025 | All 70 | Brier 2023 | 2024 | 2025 | All 70 |
|:--------|--------------:|-----:|-----:|-------:|-----------:|-----:|-----:|-------:|
| base (none) | 0.434 | 1.935 | 1.217 | 1.217 | 0.0806 | 0.0822 | 0.0682 | 0.0769 |
| weather | 0.442 | 1.908 | 1.240 | 1.218 | 0.0805 | 0.0806 | 0.0692 | 0.0767 |
| track | 0.568 | 1.679 | 1.280 | 1.193 | 0.0793 | 0.0802 | 0.0661 | 0.0751 |
| full (weather + track) | 0.594 | 1.741 | 1.326 | 1.238 | 0.0795 | 0.0811 | 0.0682 | 0.0762 |
| profile (everything, the default) | 0.587 | 1.760 | 1.347 | 1.250 | 0.0796 | 0.0813 | 0.0694 | 0.0767 |

Track type is still the only addition that improves the 70-race averages, and still unevenly:
it cuts 2024's winner log loss from 1.93 to 1.68 but makes 2023 and 2025 worse. Weather
remains a wash (the tuner keeps the wet noise factor at 1.0, and 52 wet races since 2010 move
the wet ratings very little).

The driver profile did not help. Of its four scales the tuner kept only `aggression_scale = 10`
above zero, and that bought 0.0013 of training log loss; on the held-out seasons the `profile`
row is 0.012 worse than `full` on winner log loss over 70 races (better in 2023, worse in 2024
and 2025) and 0.0005 worse on podium Brier. Risk and form were set to zero by the search, so
they are computed and printed but never enter the simulation. The profile stays on by default
because that is what the tuner chose on the training seasons, but the honest summary is that
the grid, the overall ratings and the track-type ratings already carry the signal, and a
20-race window of places gained is too noisy to add to it.

![Calibration](docs/results/backtest-calibration.png)

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
uv run f1pred tune                                 # coordinate descent on 2015-2022, held-out report, writes f1pred/tuned.json
```

Simulation commands take `--runs N`, `--seed N` and `--no-grid`; `predict` takes
`--rain 0.3` to override the rain probability; every command takes `--cache-dir DIR` and the
ones that write charts take `--out DIR`. `data update` fetches race-day rain from Open-Meteo
(no key needed) and caches it in `data/cache/weather.parquet`, so later updates only ask about
new races. It also downloads the 25 MB `lap_times.csv` and keeps only the lap-1 rows in
`data/cache/lap1.parquet`, for the aggression measure.

## Roadmap

All five planned phases are done: data and the driver-race table (1), Elo ratings (2), the
race and season Monte Carlo with the backtest and tuner (3), weather and track type (4), and
the driver profile (5). Candidate next steps, none scheduled:

- A qualifying-pace signal from FastF1 lap times, so the model sees car speed before the race
  rather than only where it started.
- Ratings that update inside a simulated season, so title odds reflect form swings and upgrades
  instead of holding every rating fixed to the end of the year.
- A small Streamlit dashboard over the cached outputs.

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

## License

MIT. See [LICENSE](LICENSE).
