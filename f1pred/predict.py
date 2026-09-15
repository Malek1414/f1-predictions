"""Turn a season/round or race name into simulation inputs, for past or future races."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from f1pred.backtest.run import entrants_for_past_race
from f1pred.config import ModelParams
from f1pred.data.track_types import load_track_types, track_type_for
from f1pred.data.weather import circuit_wet_rate, fetch_rain_probability
from f1pred.ratings.history import RatingState, state_for_season, state_strengths
from f1pred.ratings.profile import (
    Profile,
    latest_profiles,
    profile_features,
    profile_lookup,
)
from f1pred.sim.dnf import dnf_probability
from f1pred.sim.race import Entrant

log = logging.getLogger(__name__)

NO_QUALIFYING_NOTE = "No qualifying data yet; running without grid"


class UnknownRaceError(Exception):
    def __init__(self, message: str, choices: list[str]):
        super().__init__(message)
        self.choices = choices


@dataclass
class RaceRef:
    race_id: int
    season: int
    round: int
    name: str
    circuit_id: str
    date: pd.Timestamp
    has_results: bool
    time: str | None
    lat: float
    lng: float
    track_type: str


@dataclass
class PredictionInputs:
    race: RaceRef
    entrants: list[Entrant]
    names: dict[str, str]
    use_grid: bool
    note: str | None
    rain_probability: float
    rain_source: str  # observed | forecast | historical | override | disabled


def _season_races(raw: dict, season: int) -> pd.DataFrame:
    circuits = raw["circuits"][["circuitId", "circuitRef", "lat", "lng"]]
    races = raw["races"].merge(circuits, on="circuitId")
    return races[races["year"] == season].sort_values("round")


def resolve_race(
    raw: dict,
    table: pd.DataFrame,
    season: int,
    round: int | None = None,
    race: str | None = None,
) -> RaceRef:
    races = _season_races(raw, season)
    choices = [f"{int(r.round):2d}  {r.name}  ({r.circuitRef})" for r in races.itertuples()]
    if races.empty:
        raise UnknownRaceError(f"No races found for season {season}", choices)
    if round is not None:
        match = races[races["round"] == round]
    elif race is not None:
        needle = race.lower()
        match = races[
            (races["circuitRef"].str.lower() == needle)
            | races["name"].str.lower().str.contains(needle, regex=False)
        ]
    else:
        raise UnknownRaceError("Give --round or --race", choices)
    if len(match) != 1:
        what = f"round {round}" if round is not None else f"race '{race}'"
        raise UnknownRaceError(f"{what} did not match exactly one race in {season}", choices)
    r = match.iloc[0]
    race_id = int(r.raceId)
    has_results = bool(((table["race_id"] == race_id) & (~table["is_sprint"])).any())
    circuit_id = str(r["circuitRef"])
    return RaceRef(
        race_id=race_id,
        season=season,
        round=int(r["round"]),
        name=str(r["name"]),
        circuit_id=circuit_id,
        date=pd.Timestamp(r["date"]),
        has_results=has_results,
        time=None if pd.isna(r["time"]) else str(r["time"]),
        lat=float(r["lat"]),
        lng=float(r["lng"]),
        track_type=track_type_for(circuit_id, load_track_types()),
    )


def _names_from_table(table: pd.DataFrame) -> dict[str, str]:
    return dict(zip(table["driver_id"], table["driver_name"], strict=True))


def _future_entrants(
    raw: dict,
    table: pd.DataFrame,
    state: RatingState,
    ref: RaceRef,
    params: ModelParams,
    profiles: Mapping[str, Profile] | None = None,
) -> tuple[list[Entrant], dict[str, str], bool, str | None]:
    season_state = state_for_season(state, ref.season, params)
    profiles = {} if profiles is None else profiles
    quali = raw["qualifying"][raw["qualifying"]["raceId"] == ref.race_id]
    if not quali.empty:
        q = quali.merge(
            raw["drivers"][["driverId", "driverRef", "forename", "surname"]], on="driverId"
        )
        q = q.merge(raw["constructors"][["constructorId", "constructorRef"]], on="constructorId")
        rows = [
            (r.driverRef, r.constructorRef, int(r.position), f"{r.forename} {r.surname}")
            for r in q.itertuples()
        ]
        use_grid, note = True, None
    else:
        races = table[~table["is_sprint"]]
        latest = races[races["race_id"] == races.sort_values("date")["race_id"].iloc[-1]]
        rows = [(r.driver_id, r.constructor_id, None, r.driver_name) for r in latest.itertuples()]
        use_grid, note = False, NO_QUALIFYING_NOTE
    entrants, names = [], {}
    for driver_id, constructor_id, grid, name in rows:
        dry, wet = state_strengths(season_state, driver_id, constructor_id, ref.track_type, params)
        profile = profiles.get(driver_id, Profile.zero())
        entrants.append(
            Entrant(
                driver_id=driver_id,
                constructor_id=constructor_id,
                strength=dry,
                grid=grid,
                p_dnf=dnf_probability(
                    table, driver_id, constructor_id, ref.circuit_id, ref.date, params
                ),
                low_confidence=season_state.driver_races.get(driver_id, 0)
                < params.min_races_for_confidence,
                strength_wet=wet,
                aggression=profile.aggression,
                risk=profile.risk,
                form=profile.form,
            )
        )
        names[driver_id] = name
    return entrants, names, use_grid, note


def _rain_for_race(
    table: pd.DataFrame, race: RaceRef, params: ModelParams, override: float | None
) -> tuple[float, str]:
    """Spec 6.7 and 10: observed for a past race; forecast, else the circuit's historical rate."""
    if override is not None:
        return float(override), "override"
    if not params.use_weather:
        return 0.0, "disabled"
    if race.has_results:
        rows = table[(table["race_id"] == race.race_id) & (~table["is_sprint"])]
        wet = bool(rows["is_wet"].iloc[0]) if "is_wet" in rows.columns else False
        return (1.0 if wet else 0.0), "observed"
    prob = fetch_rain_probability(
        race.lat, race.lng, race.date, race.time, params.race_window_hours
    )
    if prob is not None:
        return float(prob), "forecast"
    rate = circuit_wet_rate(table, race.circuit_id, race.date)
    log.warning(
        "no rain forecast for %s %s; using historical wet rate %.0f%%",
        race.name,
        race.date.date(),
        100 * rate,
    )
    return rate, "historical"


def build_prediction_inputs(
    raw: dict,
    table: pd.DataFrame,
    history: pd.DataFrame,
    state: RatingState,
    race: RaceRef,
    params: ModelParams,
    use_grid: bool = True,
    rain_probability: float | None = None,
) -> PredictionInputs:
    rain, source = _rain_for_race(table, race, params, rain_probability)
    if race.has_results:
        rows = table[(table["race_id"] == race.race_id) & (~table["is_sprint"])]
        hist = history[(history["race_id"] == race.race_id) & (~history["is_sprint"])]
        dnf = {
            (race.race_id, r.driver_id): dnf_probability(
                table, r.driver_id, r.constructor_id, r.circuit_id, r.date, params
            )
            for r in rows.itertuples(index=False)
        }
        past_profiles = (
            profile_lookup(profile_features(table, history, params)) if params.use_profile else None
        )
        entrants = entrants_for_past_race(rows, hist, dnf, params, past_profiles)
        return PredictionInputs(
            race, entrants, _names_from_table(rows), use_grid, None, rain, source
        )
    profiles = latest_profiles(table, history, params) if params.use_profile else None
    entrants, names, grid_ok, note = _future_entrants(raw, table, state, race, params, profiles)
    return PredictionInputs(race, entrants, names, use_grid and grid_ok, note, rain, source)
