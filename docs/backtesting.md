# Backtesting

A prediction is only worth something if it was made before the answer was known. The backtest
replays past seasons one race at a time and, for each race, hands the model exactly what it
would have had on the Saturday evening: the ratings built from every session *before* that
date, the DNF rates from the same history, and the actual grid. Then it scores the forecast
against what happened on Sunday.

The ratings history table makes this mechanical. It stores, for every driver in every race,
the rating they had going into that race. The backtest reads only that row, never the rating
state after the race. A test checks that every history row used for a race carries that race's
own id.

## The scores

**Winner log loss** is "how surprised were we by the winner". If we gave the winner a 50%
chance, the loss is `ln 2 = 0.69`; at 10% it is `2.3`; at 100% it is 0. Lower is better, and
a confident wrong answer is punished hard.

**Podium Brier** is the mean squared error of the podium probabilities: for each driver, the
gap between the probability we gave them of a top-three finish and what actually happened
(1 or 0), squared, averaged over the field. Lower is better; 0 is perfect.

**Position Spearman** is the rank correlation between the model's expected finishing position
and the real classified order, over finishers only. Higher is better; 1 is a perfect ordering.

## The baselines

Each score is printed next to two baselines on the same races. **Pole wins** gives 90% to the
pole sitter and spreads the rest evenly (for podium, 90% to the front row and third; for
position, expected equals grid). **Uniform** gives every driver the same chance. A model that
cannot beat "the pole sitter wins" is not adding anything beyond qualifying.

## Calibration

The calibration chart bins every driver-race win probability into deciles and plots the
average predicted probability against how often those drivers actually won. Points on the
diagonal mean the probabilities are honest; above it means the model is underconfident, below
means overconfident.

## Tuning versus reporting

Parameters are tuned on 2015 to 2022 and the numbers in the README come from 2023 to 2025,
which the tuner never saw. Scoring the seasons you tuned on would only tell you how well the
model memorised them.
