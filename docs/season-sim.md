# Season simulation

`f1pred season --season 2026` turns the per-race Monte Carlo into championship odds. It is the
same idea as [monte-carlo.md](monte-carlo.md), chained: one simulated season is one finishing
order drawn for every remaining race (and sprint) in calendar order, with points added to the
standings after each. Doing that 2,000 times gives a distribution of final standings, and the
share of seasons each driver or constructor ends on top is their title probability.

## What stays fixed

Ratings are frozen at their pre-season-adjusted values for the whole simulated season. Real
seasons are not like that: a team brings upgrades, a driver hits or loses form, a rookie
improves. Freezing ratings under-estimates that variance, so the odds are a little too
confident about the current order. The remaining races are also run without a grid (no
qualifying has happened yet), which folds the grid spread into the driver noise. Each race gets
its own DNF probability per driver from the circuit's history.

## Points

Points follow the rules of the season being simulated, from `sim/points.py`: 25-18-15-12-10-8-6-4-2-1
for a race, 8-7-6-5-4-3-2-1 for a sprint from 2022 (3-2-1 in 2021). The fastest-lap bonus point
(2019 to 2024) is not simulated: it needs a lap-time model and is worth at most one point a race.

## Standings and retired drivers

Current standings are summed from the driver-race table, race and sprint rows both. A driver
who has left the grid keeps their points and scores no more; they still appear in the table.
Constructor totals are the points a team already holds plus the simulated points of the drivers
currently mapped to it, so a mid-season driver swap keeps the earlier points with the old team.

## Reading the output

`p_title` is the share of runs ending with that driver first; ties go to the driver higher in
the current standings. The final-position matrix (`driver_position_matrix`) gives, for each
driver, the share of runs they finish the season in each position: a tall bar at 1 means a
likely champion, a wide spread means the title is open.
