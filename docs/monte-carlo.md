# Monte Carlo race simulation

A rating says who is stronger on average. A race weekend has luck in it: a safety car at the
wrong moment, a tyre that grains, a first-corner tangle. Monte Carlo is the plain way to turn
"stronger on average" into "how often does this driver actually win": pretend to run the race
many thousands of times with the luck rolled fresh each time, and count.

## One run

Each entrant gets a performance number for this run:

```text
performance = driver_rating + constructor_rating
            + grid_bonus * n_entrants / grid ** grid_shape
            + team_noise + driver_noise
```

The first two terms are the Elo strength. The grid term rewards starting near the front, since
track position is worth real time and overtaking is hard. Since Phase 7a it is concave rather
than linear: at `grid_shape = 1` pole is worth `grid_bonus * n`, P2 half of that and P20 one
twentieth, so the front rows matter a lot and the back rows barely at all. A linear term had
priced a win from P19 at zero (Monza 2026, log loss 9.2); the concave term gives it a few
percent. Then two random draws are added.

`team_noise` is one draw per constructor, shared by both of its cars. When a team brings an
upgrade that works, or misjudges the setup window, both cars move together. Giving teammates
the same draw makes the simulated results look like real ones, where the two cars of a team
tend to finish near each other.

`driver_noise` is one draw per driver and covers everything personal to that car on the day.

Next, each driver is retired with probability `p_dnf` (see `dnf-model.md`). Retired cars are
taken out of the classified order and placed at the back in random order, because a driver who
stopped on lap 3 and one who stopped on lap 50 are both "did not finish" in the results.

Finally the survivors are sorted by performance, highest first. That ordering is the finishing
order for this run.

## From runs to probabilities

After 10,000 runs, `P(driver wins)` is simply the share of runs in which they finished first.
The same counting gives podium and points probabilities, the average finishing position, its
10th and 90th percentiles, and a full matrix of `P(driver finishes in position k)`. Every column
of that matrix sums to one, because exactly one driver finishes in each position each run.

10,000 runs keep the sampling error on a win probability below about half a percentage point,
and the whole thing is one NumPy matrix of shape `(runs, entrants)` sorted along one axis, so
it takes well under a second.

## Running without a grid

Before qualifying the grid is unknown. In `--no-grid` mode the grid term is dropped and its
spread is folded into the driver noise: with the grid slot a uniform draw over `1..n`, the
grid term has standard deviation `sigma_grid = std(grid_term(1..n))`, so `sigma_driver` becomes
`sqrt(sigma_driver^2 + sigma_grid^2)`. The forecast gets wider, which is the honest thing to do
when you know less.
