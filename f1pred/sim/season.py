"""Chain race simulations over the remaining calendar to get championship odds. Spec section 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.config import ModelParams
from f1pred.ratings.history import RatingState, state_for_season, state_strengths
from f1pred.ratings.profile import Profile
from f1pred.sim.dnf import dnf_probability
from f1pred.sim.points import points_for_positions, race_points, sprint_points
from f1pred.sim.race import Entrant, simulate_positions


@dataclass(frozen=True)
class RemainingRace:
    race_id: int
    round: int
    name: str
    circuit_id: str
    date: pd.Timestamp
    has_sprint: bool
    track_type: str = "mixed"
    rain_probability: float = 0.0


@dataclass
class SeasonForecast:
    season: int
    driver_ids: list[str]
    constructor_ids: list[str]
    p_driver_title: np.ndarray
    p_constructor_title: np.ndarray
    expected_driver_points: np.ndarray
    expected_constructor_points: np.ndarray
    driver_position_matrix: np.ndarray
    current_driver_points: np.ndarray
    current_constructor_points: np.ndarray
    n_runs: int
    n_remaining: int

    def drivers_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "driver_id": self.driver_ids,
                "current_points": self.current_driver_points,
                "expected_points": self.expected_driver_points,
                "p_title": self.p_driver_title,
                "p_top3": self.driver_position_matrix[:, :3].sum(axis=1),
            }
        )
        return df.sort_values(
            ["p_title", "expected_points"], ascending=False, kind="stable"
        ).reset_index(drop=True)

    def constructors_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "constructor_id": self.constructor_ids,
                "current_points": self.current_constructor_points,
                "expected_points": self.expected_constructor_points,
                "p_title": self.p_constructor_title,
            }
        )
        return df.sort_values(
            ["p_title", "expected_points"], ascending=False, kind="stable"
        ).reset_index(drop=True)


def remaining_calendar(raw: dict, table: pd.DataFrame, season: int) -> list[RemainingRace]:
    races = raw["races"].merge(raw["circuits"][["circuitId", "circuitRef"]], on="circuitId")
    races = races[races["year"] == season].sort_values("round")
    done = set(table.loc[~table["is_sprint"], "race_id"].astype(int))
    out = []
    for r in races.itertuples():
        if int(r.raceId) in done:
            continue
        out.append(
            RemainingRace(
                race_id=int(r.raceId),
                round=int(r.round),
                name=str(r.name),
                circuit_id=str(r.circuitRef),
                date=pd.Timestamp(r.date),
                has_sprint=pd.notna(r.sprint_date),
            )
        )
    return out


def current_standings(
    table: pd.DataFrame, season: int
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    rows = table[table["season"] == season]
    if rows.empty:
        return {}, {}, {}
    driver_points = rows.groupby("driver_id")["points"].sum().to_dict()
    constructor_points = rows.groupby("constructor_id")["points"].sum().to_dict()
    latest = rows.sort_values("date").groupby("driver_id")["constructor_id"].last()
    return driver_points, constructor_points, latest.to_dict()


def season_entrants(
    table: pd.DataFrame,
    state: RatingState,
    season: int,
    race: RemainingRace,
    params: ModelParams,
    profiles: Mapping[str, Profile] | None = None,
) -> list[Entrant]:
    """The latest race's grid as entrants; `profiles` (spec 6.6) is zero when None or missing."""
    season_state = state_for_season(state, season, params)
    profiles = {} if profiles is None else profiles
    races = table[~table["is_sprint"]]
    latest_id = races.sort_values("date")["race_id"].iloc[-1]
    latest = races[races["race_id"] == latest_id]
    out = []
    for r in latest.itertuples(index=False):
        dry, wet = state_strengths(
            season_state, r.driver_id, r.constructor_id, race.track_type, params
        )
        profile = profiles.get(r.driver_id, Profile.zero())
        out.append(
            Entrant(
                driver_id=r.driver_id,
                constructor_id=r.constructor_id,
                strength=dry,
                grid=None,
                p_dnf=dnf_probability(
                    table, r.driver_id, r.constructor_id, race.circuit_id, race.date, params
                ),
                low_confidence=season_state.driver_races.get(r.driver_id, 0)
                < params.min_races_for_confidence,
                strength_wet=wet,
                aggression=profile.aggression,
                risk=profile.risk,
                form=profile.form,
            )
        )
    return out


def _ordered_ids(points: Mapping[str, float], extra: set[str]) -> list[str]:
    ids = set(points) | extra
    return sorted(ids, key=lambda i: (-points.get(i, 0.0), i))


def simulate_season(
    season: int,
    remaining: list[RemainingRace],
    entrants_by_race: list[list[Entrant]],
    driver_points: Mapping[str, float],
    constructor_points: Mapping[str, float],
    driver_to_constructor: Mapping[str, str],
    params: ModelParams,
    n_runs: int = 2000,
    seed: int | None = None,
    rain_by_race: list[float] | None = None,
) -> SeasonForecast:
    if len(remaining) != len(entrants_by_race):
        raise ValueError("remaining and entrants_by_race must have the same length")
    if rain_by_race is None:
        rain_by_race = [0.0] * len(remaining)
    if len(rain_by_race) != len(remaining):
        raise ValueError("rain_by_race must have one entry per remaining race")
    rng = np.random.default_rng(seed)

    entrant_drivers = {e.driver_id for ents in entrants_by_race for e in ents}
    entrant_teams = {e.constructor_id for ents in entrants_by_race for e in ents}
    driver_ids = _ordered_ids(driver_points, entrant_drivers)
    constructor_ids = _ordered_ids(constructor_points, entrant_teams)
    d_idx = {d: i for i, d in enumerate(driver_ids)}
    c_idx = {c: i for i, c in enumerate(constructor_ids)}

    mapping = dict(driver_to_constructor)
    for ents in entrants_by_race:
        for e in ents:
            mapping[e.driver_id] = e.constructor_id
    M = np.zeros((len(driver_ids), len(constructor_ids)))
    for d, c in mapping.items():
        if d in d_idx and c in c_idx:
            M[d_idx[d], c_idx[c]] = 1.0

    current_d = np.array([driver_points.get(d, 0.0) for d in driver_ids], dtype=float)
    totals = np.tile(current_d, (n_runs, 1))
    race_table, sprint_table = race_points(season), sprint_points(season)
    for race, ents, rain in zip(remaining, entrants_by_race, rain_by_race, strict=True):
        cols = np.array([d_idx[e.driver_id] for e in ents])
        if race.has_sprint and sprint_table.size:
            pos, _ = simulate_positions(
                ents, params, n_runs, rng, use_grid=False, rain_probability=rain
            )
            totals[:, cols] += points_for_positions(pos, sprint_table)
        pos, _ = simulate_positions(
            ents, params, n_runs, rng, use_grid=False, rain_probability=rain
        )
        totals[:, cols] += points_for_positions(pos, race_table)

    # Constructors: points already held (which may include drivers no longer mapped) plus
    # simulated points routed through the driver -> constructor map.
    current_c = np.array([constructor_points.get(c, 0.0) for c in constructor_ids], dtype=float)
    simulated = totals - current_d
    c_totals = current_c + simulated @ M

    champion = np.argmax(totals, axis=1)
    c_champion = np.argmax(c_totals, axis=1)
    order = np.argsort(-totals, axis=1, kind="stable")
    final_pos = np.empty_like(order)
    np.put_along_axis(
        final_pos, order, np.arange(1, len(driver_ids) + 1)[None, :].repeat(n_runs, 0), axis=1
    )
    n_d = len(driver_ids)
    position_matrix = (
        np.stack([np.bincount(final_pos[:, i] - 1, minlength=n_d) for i in range(n_d)]).astype(
            float
        )
        / n_runs
    )
    return SeasonForecast(
        season=season,
        driver_ids=driver_ids,
        constructor_ids=constructor_ids,
        p_driver_title=np.bincount(champion, minlength=n_d) / n_runs,
        p_constructor_title=np.bincount(c_champion, minlength=len(constructor_ids)) / n_runs,
        expected_driver_points=totals.mean(axis=0),
        expected_constructor_points=c_totals.mean(axis=0),
        driver_position_matrix=position_matrix,
        current_driver_points=current_d,
        current_constructor_points=current_c,
        n_runs=n_runs,
        n_remaining=len(remaining),
    )
