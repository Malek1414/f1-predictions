# Driver profile: aggression, risk and form

Phase 5 gives every driver three numbers that the ratings do not capture, each computed from
what had happened before the race being predicted, so the backtest stays leak-free.

## The three measures

**Aggression** is how many places a driver tends to gain, averaged over two signals: grid
position minus position at the end of lap 1 (from the lap-times table) and grid position minus
finishing position, both taken from races the driver finished. A driver who starts eighth, is
fifth after the first corner and finishes fourth scores `(3 + 4) / 2 = 3.5` for that race. The
number is the driver's average over the last 20 races minus the field's average over the same
period, so a lap-1 charger sits well above zero and a driver who tends to fall back sits below.

**Risk** is the driver's rate of accident-type retirements (accident, collision, spun off,
damage) over the last 20 races divided by the field's rate, minus one. A driver who crashes
twice as often as the field scores `+1`; one who never crashes scores `-1`. Mechanical
retirements do not count: they belong to the car, and the DNF model already handles them.

**Form** is how far a driver has beaten expectations lately: over the last six races, the mean
of expected finishing position minus actual position, finishers only. Expected position is the
driver's rank by pre-race strength (driver plus constructor rating) among that race's entrants.
A driver rated fifth-strongest who keeps finishing third is on a hot streak and scores `+2`.

The spec asked for expected position from a small simulation with the profile switched off.
The strength rank carries the same ordering information without a nested Monte Carlo inside
the ratings replay, so that is what is used.

## Centred and shrunk

All three are centred so that zero means "like the field", and each is multiplied by
`n / (n + 10)`, where `n` is the number of races behind it. A rookie with three races has a
quarter of the weight of a driver with a full window, and a driver's very first race gets zero.

## In the simulation

Aggression and form enter as pace: `aggression_scale * aggression + form_scale * form` is
added to the driver's performance in every run. Risk enters as spread and as accidents: the
driver's private noise is multiplied by `1 + risk_noise_scale * risk` (clipped to between 0.5
and 3), and their DNF probability by `1 + risk_dnf_scale * risk` (clipped to between 1% and
95%). No extra random draws are made, so with every scale at zero the Phase 4 results are
reproduced bit for bit.

## The honest caveat

These numbers add colour, and they may not add accuracy. The grid and the ratings already
carry most of what a race result is going to say, and 20-race windows are noisy. That is why
all four scales default to zero and are set by the tuner, and why `f1pred backtest --ablate`
reports a `profile` variant next to `full`: if the profile does not lower the held-out log loss
and Brier score, the scales stay at zero and the profile is only ever printed, never simulated.
