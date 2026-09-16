"""A tiny planted-effect driver-race table used by the Bayesian pace-model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.config import DEFAULT_PARAMS
from f1pred.data.frame import DRIVER_RACE_COLUMNS
from f1pred.pace.design import build_design

SEASONS = (2021, 2022, 2023)
RACES_PER_SEASON = 10
CARS = {"A": 1.0, "B": 0.0, "C": -1.0}
# Two drivers per car; "ace" is the planted quick one.
LINEUP = {"A": ("ace", "vet"), "B": ("bee", "bo"), "C": ("cee", "co")}
SKILL = {"ace": 0.8, "vet": 0.0, "bee": 0.0, "bo": -0.2, "cee": 0.1, "co": 0.0}
CIRCUITS = ("alpha", "beta", "gamma", "delta", "epsilon")
KAPPA = 1.0


def true_pace(breakout: tuple[str, int, float] | None = None) -> dict:
    """(driver, season, delta) raises that driver's pace in that season only."""

    def pace(driver: str, car: str, season: int) -> float:
        mu = SKILL[driver] + CARS[car]
        if breakout is not None and driver == breakout[0] and season == breakout[1]:
            mu += breakout[2]
        return mu

    return {
        (d, car, s): pace(d, car, s)
        for car, drivers in LINEUP.items()
        for d in drivers
        for s in SEASONS
    }


def synthetic_table(seed: int = 0, breakout: tuple[str, int, float] | None = None) -> pd.DataFrame:
    """One row per driver-race with planted paces, Plackett-Luce orders and matching quali gaps."""
    rng = np.random.default_rng(seed)
    pace = true_pace(breakout)
    entries = [(d, car) for car, drivers in LINEUP.items() for d in drivers]
    rows, race_id = [], 1
    for season in SEASONS:
        for rnd in range(1, RACES_PER_SEASON + 1):
            mu = np.array([pace[(d, c, season)] for d, c in entries])
            # Qualifying first: the gap to pole is kappa * (pole pace - own pace) plus noise.
            q_noise = rng.normal(0.0, 0.15, size=len(entries))
            q_pace = mu + q_noise
            gap = KAPPA * (q_pace.max() - q_pace)
            grid = np.argsort(np.argsort(gap)) + 1
            # The race order is a Gumbel-max draw from the same paces (exactly Plackett-Luce).
            key = mu + rng.gumbel(0.0, 1.0, size=len(entries))
            finish = np.argsort(np.argsort(-key)) + 1
            circuit = CIRCUITS[(rnd - 1) % len(CIRCUITS)]
            for i, (driver, car) in enumerate(entries):
                rows.append(
                    {
                        "season": season,
                        "round": rnd,
                        "race_id": race_id,
                        "date": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=14 * rnd),
                        "circuit_id": circuit,
                        "race_name": f"{circuit} {season}",
                        "driver_id": driver,
                        "driver_code": driver[:3].upper(),
                        "driver_name": driver,
                        "constructor_id": car,
                        "constructor_name": car,
                        "grid": int(grid[i]),
                        "position": int(finish[i]),
                        "status": "Finished",
                        "dnf": False,
                        "dnf_kind": None,
                        "points": 0.0,
                        "is_sprint": False,
                        "is_wet": False,
                        "track_type": "mixed",
                        "lap1_position": pd.NA,
                        "quali_gap_pct": float(gap[i]),
                    }
                )
            race_id += 1
    table = pd.DataFrame(rows, columns=list(DRIVER_RACE_COLUMNS))
    table["position"] = table["position"].astype("Int64")
    table["lap1_position"] = table["lap1_position"].astype("Int64")
    return table


def synthetic_design(seed: int = 0, breakout: tuple[str, int, float] | None = None):
    return build_design(synthetic_table(seed, breakout), params=DEFAULT_PARAMS)


JUNK_FLOOR = 1.5  # a junk lap is at least this far off pole, then exponentially further
JUNK_SCALE = 1.0


def contaminated_table(seed: int = 0, fraction: float = 0.05, driver: str = "ace"):
    """The synthetic table with `fraction` of the qualifying gaps replaced by junk laps.

    A junk lap is what a wet or red-flagged session leaves behind: a large, one-sided gap that
    says nothing about pace. Two things about it are copied from the real data. It is
    positive-only — a disrupted session only ever makes a lap look slow. And it is not spread
    evenly: the gaps that reach 20% and beyond belong to whoever was on track at the wrong
    moment, so here they all land on one driver, which is what turns a heavy tail into a bias.
    """
    table = synthetic_table(seed)
    rng = np.random.default_rng(seed + 9_000)
    gaps = table["quali_gap_pct"].to_numpy(dtype=float, copy=True)
    rows = np.flatnonzero((table["driver_id"] == driver).to_numpy())
    hit = rng.choice(rows, size=max(int(round(fraction * len(table))), 1), replace=False)
    gaps[hit] = JUNK_FLOOR + rng.exponential(JUNK_SCALE, size=hit.size)
    table["quali_gap_pct"] = gaps
    return table


def contaminated_design(seed: int = 0, fraction: float = 0.05, driver: str = "ace"):
    return build_design(contaminated_table(seed, fraction, driver), params=DEFAULT_PARAMS)


def planted_teammate_gap(fast: str = "ace", slow: str = "vet") -> float:
    """The planted pace between two drivers in the same car, so the car term cancels."""
    return SKILL[fast] - SKILL[slow]


def last_race_entrants(table: pd.DataFrame):
    """Entrants copied from the final race of the synthetic table, with its grid."""
    from f1pred.sim.race import Entrant

    last = table[table.race_id == table.race_id.max()]
    return [
        Entrant(r.driver_id, r.constructor_id, 0.0, int(r.grid), 0.0)
        for r in last.itertuples(index=False)
    ]


class FakeRace:
    """The little bit of `RaceRef` that the pace model needs."""

    def __init__(self, season: int, circuit_id: str = "alpha", track_type: str = "mixed"):
        self.season = season
        self.circuit_id = circuit_id
        self.track_type = track_type
