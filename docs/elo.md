# Elo ratings for drivers and constructors

Elo was invented for chess. Every player carries a number; when two of them play, the
rating gap tells you how likely each one is to win, and the result moves both numbers.
A gap of 400 points means the stronger side is expected to win about ten times out of
eleven. A gap of zero means a coin flip.

We use the same idea for a race, treating it as a set of head-to-head matchups. Every pair
of classified finishers is one matchup: the driver who finished ahead "won" it. Each pair
moves both ratings by `K` times the surprise, where surprise is the actual result (1 or 0)
minus the expected score. The changes are scaled by `1 / (n - 1)`, so a twenty-car grid
does not move ratings ten times harder than a ten-car grid.

Every driver and every constructor has a rating, both starting at 1500. A driver's race
strength is the sum: `driver + constructor`. That is what enters the expected-score formula,
because a strong car makes a strong driver look even stronger, and the update is what the
result says about the *pair* of numbers.

Teammate matchups count double for the driver update and not at all for the constructor
update. Two drivers in the same car are the cleanest signal of driver skill there is, because
the car cancels out. For the same reason the constructor learns nothing from that pair.

Drivers who do not finish take no part in the updates. Reliability is modelled separately
(see `dnf-model.md`), so a blown engine is not treated as evidence of a slow driver.

Because each matchup gives one side what it takes from the other, the total change across
the field is zero. Ratings measure relative strength, not absolute pace.

At the start of a season every rating is pulled part of the way back toward 1500. Drivers
are pulled gently (10%), constructors harder (40%), and in rule-change seasons constructors
are pulled harder still (70%), because a new set of regulations reshuffles the cars far more
than it reshuffles the people driving them.

The formulas:

```text
E_A     = 1 / (1 + 10 ** ((S_B - S_A) / 400))
rating  = 1500 + (rating - 1500) * (1 - regress)      # at the start of each season
```
