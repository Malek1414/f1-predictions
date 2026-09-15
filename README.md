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
                    Azerbaijan Grand Prix 2026 (round 15)
 #   Driver                    Win   Podium   Points   Exp. pos   P10-P90
 1   George Russell          29.3%    63.1%    96.9%        3.5       1-7
 2   Max Verstappen          18.6%    48.6%    94.9%        4.4       1-9
 3   Lando Norris            14.3%    41.7%    93.0%        4.9       1-9
 4   Charles Leclerc         10.4%    34.0%    90.8%        5.5      1-10
 5   Andrea Kimi Antonelli   10.3%    37.5%    91.4%        5.3      1-10
 6   Lewis Hamilton           7.9%    28.7%    87.8%        6.1      2-11
 7   Oscar Piastri            6.1%    24.2%    84.3%        6.6      2-12
 8   Arvid Lindblad           0.8%     5.3%    53.8%       10.4      5-16
 9   Pierre Gasly             0.6%     4.4%    51.2%       10.7      5-17
10   Liam Lawson              0.5%     3.7%    49.6%       10.9      5-17
11   Franco Colapinto         0.4%     2.6%    40.0%       11.8      6-18
12   Gabriel Bortoleto        0.2%     1.6%    33.1%       12.6      7-19
13   Yuki Tsunoda             0.2%     1.9%    35.4%       12.3      7-18
14   Nico Hülkenberg          0.1%     1.2%    28.4%       13.2      7-19
15   Oliver Bearman           0.1%     0.5%    15.1%       15.0      9-20
16   Alexander Albon          0.0%     0.1%     7.6%       16.8     11-21
17   Carlos Sainz             0.0%     0.4%    14.9%       15.2      9-20
18   Esteban Ocon             0.0%     0.3%    15.7%       15.1      9-20
19   Sergio Pérez             0.0%     0.1%     4.8%       17.6     12-22
20   Fernando Alonso          0.0%     0.1%     7.4%       16.8     11-21
21   Valtteri Bottas          0.0%     0.0%     1.9%       19.1     15-22
22   Lance Stroll             0.0%     0.0%     1.8%       19.0     15-22
                  * fewer than 5 rated races: low confidence
```

![Win probability](docs/results/example-win.png)

![Finishing position heatmap](docs/results/example-positions.png)

### Championship odds

`f1pred season --season 2026`, run with nine races left, chains the race simulation over the
rest of the calendar 2,000 times and counts who ends up on top:

```text
           2026 drivers' championship (9 races left, 2,000 runs)
 #   Driver                  Points now   Exp. points    Title    Top 3
 1   Andrea Kimi Antonelli          292           398    89.8%   100.0%
 2   George Russell                 211           361    10.1%    99.1%
 3   Lando Norris                   186           300     0.1%    58.5%
 4   Lewis Hamilton                 191           283     0.0%    24.1%
 5   Max Verstappen                 145           268     0.0%    10.2%
 6   Charles Leclerc                167           268     0.0%     8.2%
 7   Oscar Piastri                  120           202     0.0%     0.0%
 8   Liam Lawson                     59            86     0.0%     0.0%
```

![Championship odds](docs/results/example-title.png)

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

## Backtest results

Parameters were tuned by coordinate descent on 2015 to 2022 (`f1pred tune`), which chose
`k_driver = 16` and `k_constructor = 24` and left the rest at their defaults. The seasons below
were never seen by the tuner. Each race is predicted from the ratings, DNF rates and grid that
were known before it started. See [docs/backtesting.md](docs/backtesting.md).

| Season | Races | Log loss model | Log loss pole | Log loss uniform | Brier model | Brier pole | Brier uniform | Spearman model | Spearman pole |
|-------:|------:|---------------:|--------------:|-----------------:|------------:|-----------:|--------------:|---------------:|--------------:|
| 2023 | 22 | 0.59 | 1.98 | 3.00 | 0.08 | 0.09 | 0.13 | 0.77 | 0.71 |
| 2024 | 24 | 1.76 | 2.89 | 2.99 | 0.08 | 0.09 | 0.13 | 0.79 | 0.78 |
| 2025 | 24 | 1.35 | 1.61 | 2.99 | 0.07 | 0.06 | 0.13 | 0.62 | 0.65 |

Log loss is the surprise at the actual winner (lower is better), Brier the squared error of the
podium probabilities (lower is better), Spearman the rank correlation of expected versus actual
finishing position (higher is better). The "pole" baseline gives the pole sitter a 90% win
chance and spreads the rest evenly; "uniform" gives every driver the same chance.

The model beats both baselines on winner log loss in every season and beats uniform on every
score. It is weakest where the grid already says most of what there is to say: in 2025 the
pole baseline edges it on podium Brier and Spearman, and tuning moved the held-out numbers only
marginally from the defaults (2024 improved, 2023 and 2025 got slightly worse). Weather, track
type and driver form are not yet modelled.

![Calibration](docs/results/backtest-calibration.png)

## Install and usage

```bash
uv sync
uv run f1pred data update                          # download tables into data/cache/, build the driver-race table
uv run f1pred ratings build                        # replay 2010 onward, write pre-race ratings, print the top 10
uv run f1pred predict --season 2026 --round 15     # or --race monza; saves outputs/2026-15-win.png and -positions.png
uv run f1pred season --season 2026                 # title odds for the rest of the season; saves outputs/2026-title.png and CSVs
uv run f1pred backtest --seasons 2023-2025         # score vs baselines; saves outputs/calibration.png and CSVs
uv run f1pred tune                                 # coordinate descent on 2015-2022, held-out report, writes f1pred/tuned.json
```

Simulation commands take `--runs N`, `--seed N` and `--no-grid`; every command takes
`--cache-dir DIR` and the ones that write charts take `--out DIR`.

## Roadmap

Done: Phase 3, season simulation with drivers' and constructors' championship odds
(`f1pred season`).

- Phase 4: wet/dry weather via Open-Meteo and track-type conditional ratings, judged by the backtest.
- Phase 5: driver profile (aggression, risk, form) from lap-1 positions, DNF kinds and recent over-performance.

## Credits

Data: [tracinginsights/RaceData](https://huggingface.co/datasets/tracinginsights/RaceData),
derived from the Ergast and [Jolpica](https://github.com/jolpica/jolpica-f1) F1 databases.

Prior art consulted: mar-antaya's `2025_f1_predictions` and `2026_f1_predictions`
(per-race gradient boosting on qualifying times); villekuosmanen's `F1Predict` (driver,
constructor and engine Elo with Monte Carlo perturbation, the closest relative of this design);
van Kesteren and Bergkamp 2023, JQAS (a Bayesian model showing constructors explain most of the
variance in results); Kevocado's `F1_Predictor` and neevj2006's `F1_Race_Predictor` (separate
reliability layer, shared team noise, leakage-safe backtests).

## License

MIT. See [LICENSE](LICENSE).
