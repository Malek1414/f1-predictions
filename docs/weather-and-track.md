# Weather and track type

Phase 4 adds two pieces of context to every race: whether it was (or is likely to be) wet, and
what kind of circuit it is. Both feed conditional Elo ratings and the Monte Carlo.

## Where the rain data comes from

`f1pred data update` asks the free [Open-Meteo](https://open-meteo.com/) archive for hourly
precipitation at each circuit's coordinates on race day, sums the three hours from the race start
(UTC start time from the races table; 13:00 UTC if missing), and stores the result in
`data/cache/weather.parquet`. Races already in the file are not fetched again, so a routine
update only asks about new races. Rows whose fetch failed are kept with no rain figure, treated
as dry, and retried next time.

A race is **wet** when that three-hour total is at least 0.5 mm. The threshold is deliberately
low so that races run partly on intermediates count. It also produces false positives: a brief
shower before the start, or rain at the weather grid point but not on the track, will flag a race
that was effectively dry. Treat `is_wet` as "rain was around", not "the whole race was wet".

## Conditional ratings and shrinkage

Every driver and constructor keeps a wet rating and one rating per track type, updated with the
same pairwise Elo rules but only on races of that condition. When used, a conditional rating is
pulled toward the overall rating:

```
effective = overall + (conditional - overall) * n / (n + shrink)
```

`n` is the number of rated races in that condition and `shrink` defaults to 8, so after 8 wet
races the wet rating counts for exactly half of the difference; after 24 it counts for three
quarters. This keeps a driver with two lucky wet results from being crowned a rain specialist.
`shrink_wet` and `shrink_track` are separate knobs and are part of the tuning grid.

## Track types

`f1pred/data/track_types.yaml` labels each circuit raced since 2010 as `street`, `high_speed`,
`high_downforce`, or `mixed`. The labels are hand-made judgement calls, not derived from data;
several circuits could reasonably sit in two boxes. A circuit missing from the file is treated
as `mixed` with a warning.

## Rain inside the simulation

For a future race the model asks Open-Meteo's forecast endpoint for the highest hourly rain
probability in the race window. A 40% chance becomes a coin flip inside every simulated run:
40% of runs are wet. In a wet run each car uses its wet strength, both noise terms are scaled by
`wet_noise_factor` (default 1.5) and DNF odds by `wet_dnf_factor` (default 1.5, capped at 95%).
The printed probabilities are therefore a mixture of the dry and wet outcomes. `--rain 0.3`
overrides the forecast. For a past race the observed `is_wet` is used as a certainty.

## Reading the ablation table

`f1pred backtest --ablate` scores four variants: `base` (neither feature), `weather`, `track`,
and `full`. Lower log loss and Brier are better. If `weather` does not beat `base`, the wet
ratings are not adding information beyond the overall rating on those seasons; the same reading
applies to `track`.

## When Open-Meteo is unreachable

If the forecast call fails or the race is beyond the 16-day horizon, the model falls back to
the circuit's historical wet rate (or the global rate when the circuit has fewer than 3 past
races), prints which source it used, and continues. If the archive is unreachable during
`data update`, the affected races are marked dry and fetched on the next update.
