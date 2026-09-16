# Backtest metrics

Every score is computed per race from pre-race information only, then averaged over the races
of a season (`f1pred backtest`). Lower is better for log loss, Brier, RPS and top-3 loss; higher
is better for Spearman, skill scores and sharpness. The same scores are computed for the two
baselines: `pole` (90% on the pole sitter, or on the grid slot for position distributions, the
rest spread evenly) and `uniform` (every driver and every position equally likely). Code:
`f1pred/backtest/scoring.py`; tests with hand-computed values: `tests/test_scoring.py`.

## Winner log loss

`-log p(winner)`, with `p` floored at `1e-6` (a loss of 13.8). It only judges the win column, so
a single upset priced near zero dominates a season average. Read it with the pole baseline next
to it: a season where the model is under the pole number is one where knowing the ratings beat
knowing the grid.

## Podium Brier

Mean over drivers of `(p(podium) - 1[on podium])^2`. Bounded in `[0, 1]`, forgiving of single
upsets, and mostly a measure of how well the model separates the front of the field from the
back.

## Position Spearman

Rank correlation between the expected finishing position and the actual one over classified
drivers. It ignores probabilities entirely, so it is a sanity check on the ordering, not a
proper score.

## Ranked probability score (RPS)

For one driver with finishing-position distribution `p` over `1..n` and actual position `a`:

```
RPS = sum_{k=1..n-1} (F_p(k) - 1[a <= k])^2 / (n - 1)
```

where `F_p` is the cumulative distribution. It is 0 when all the mass sits on `a` and 1 when it
sits at the opposite end of the order, and unlike log loss it credits a distribution for being
*close* (predicting P4 for a P3 finish costs little; predicting P18 costs a lot). The race value
is the mean over classified drivers, so it judges the whole finishing-order distribution, not
just the winner. A flat distribution over 20 places scores about 0.33 on average.

## Top-3 set log loss

`-log P(podium set)` where the set probability is the fraction of Monte Carlo runs whose top
three are exactly the observed podium drivers, floored at `1e-4` (a loss of 9.2). The uniform
baseline scores `log C(n, 3)` (7.04 for 20 cars) and the pole baseline `-log 0.9` when the front
row plus P3 finish on the podium in any order and `-log(0.1 / C(n, 3))` otherwise. It is a much
harder target than the podium Brier: it asks for the joint outcome, not three marginals.

## Expected calibration error (ECE)

Over every driver-race in the run, win (or podium) probabilities are put into `B = 10`
equal-width bins and

```
ECE = sum_b (n_b / N) * |mean predicted_b - mean observed_b|
```

A perfectly calibrated model has ECE 0; 0.03 means the average probability is off by three
points. It is reported once per backtest, not per season, because the top bins are sparse.
The reliability table behind it is `calibration.png`.

## Sharpness

Mean over races of the largest win probability. Calibration alone is easy to achieve with a
flat forecast (sharpness 0.05); a useful model is calibrated *and* sharp. Read it together with
ECE, never on its own.

## Skill score

`1 - model / baseline` for a loss-type metric (log loss, RPS, top-3). 0 matches the baseline,
1 would be perfect, negative is worse than the baseline.

## Bootstrap intervals

`backtest --bootstrap R` resamples the races of the run with replacement `R` times, recomputes
each mean, and reports the 5th and 95th percentiles (a 90% interval). With 24 races a season the
winner log loss interval is typically ±0.4, so a difference of 0.1 between two runs is noise.
Two runs whose intervals do not overlap are clearly different; overlapping intervals mean the
evidence is thin.

## Rolling-origin evaluation

`backtest --rolling 2024-2026 [--tune]` scores each fold season with parameters chosen on the
seasons before it (`start_season .. fold - 1`), then rolls forward. With `--tune` the search runs
once per fold on the training seasons only. The number reported per fold is one the tuner never
saw; the plain `backtest` number on a tuned season is not.
