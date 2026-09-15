# The DNF model

Pace and reliability are different things. A fast car with a fragile engine wins some races
and parks up in others; a slow but dependable car finishes every race in ninth. If we let a
blown engine count as a "loss" in the Elo update, the rating would say the driver is slow when
the truth is the car broke. So the Elo layer only looks at classified finishers, and a separate
number, `p_dnf`, says how likely each entrant is to not finish at all.

`p_dnf` starts from three recent rates: the driver's own DNF rate over their last 20 races,
the constructor's over its last 40 driver-races, and the whole field's rate over the same
period. The driver and constructor rates are averaged, then shrunk toward the field rate.

Shrinking means "do not overreact to a small sample". With `n` races of evidence the blended
rate gets weight `n / (n + 10)` and the field rate gets the rest. A rookie with one crash in
two races is not a 50% crasher; with `n = 2` they sit at 2/12 of the way from the average
toward 50%. After twenty races the weight is 20/30, and their own record mostly speaks for
itself.

Some circuits eat cars. The result is multiplied by the circuit's historical DNF rate divided
by the global rate, but only once the circuit has at least 40 driver-race rows, and clipped to
between 0.5 and 2.0. The clip stops a handful of chaotic races at a new venue from doubling or
halving everyone's chances by itself.

The final number is clipped to [0.01, 0.9]. In the simulation each driver's DNF is drawn
from a coin with this probability; the ones that "retire" are removed from the classified
order and placed at the back.

All numbers come from the driver-race table (`results` joined to `status`), and only rows
dated before the race being predicted are ever used. Sprint sessions are ignored here.
