"""Replay every session in date order and record the rating each entrant had going in."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from f1pred.config import ModelParams
from f1pred.ratings.conditional import strengths
from f1pred.ratings.elo import apply_update, race_update, regress

# Phase 4 and 5 feature columns; a table without them is stale and must be rebuilt.
REQUIRED_FEATURE_COLUMNS = ("is_wet", "track_type", "lap1_position")

HISTORY_COLUMNS = [
    "race_id",
    "is_sprint",
    "season",
    "round",
    "date",
    "driver_id",
    "constructor_id",
    "driver_rating_pre",
    "constructor_rating_pre",
    "driver_races_pre",
    "is_wet",
    "track_type",
    "driver_wet_pre",
    "constructor_wet_pre",
    "driver_wet_n_pre",
    "constructor_wet_n_pre",
    "driver_track_pre",
    "constructor_track_pre",
    "driver_track_n_pre",
    "constructor_track_n_pre",
]


@dataclass
class RatingState:
    driver: dict[str, float] = field(default_factory=dict)
    constructor: dict[str, float] = field(default_factory=dict)
    driver_races: dict[str, int] = field(default_factory=dict)
    season: int | None = None
    driver_wet: dict[str, float] = field(default_factory=dict)
    constructor_wet: dict[str, float] = field(default_factory=dict)
    driver_wet_n: dict[str, int] = field(default_factory=dict)
    constructor_wet_n: dict[str, int] = field(default_factory=dict)
    driver_track: dict[str, dict[str, float]] = field(default_factory=dict)
    constructor_track: dict[str, dict[str, float]] = field(default_factory=dict)
    driver_track_n: dict[str, dict[str, int]] = field(default_factory=dict)
    constructor_track_n: dict[str, dict[str, int]] = field(default_factory=dict)

    def copy(self) -> RatingState:
        return copy.deepcopy(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> RatingState:
        data = json.loads(Path(path).read_text())
        return cls(**data)


def _regress_all(
    state: RatingState, driver_fraction: float, constructor_fraction: float, initial: float
) -> None:
    regress(state.driver, driver_fraction, initial)
    regress(state.constructor, constructor_fraction, initial)
    regress(state.driver_wet, driver_fraction, initial)
    regress(state.constructor_wet, constructor_fraction, initial)
    for d in state.driver_track.values():
        regress(d, driver_fraction, initial)
    for c in state.constructor_track.values():
        regress(c, constructor_fraction, initial)


def start_season(state: RatingState, season: int, params: ModelParams) -> None:
    """Spec 5.3: when a new season begins, regress ratings toward the initial value."""
    if state.season is not None and season > state.season:
        fraction = (
            params.regress_constructor_regulation
            if season in params.regulation_seasons
            else params.regress_constructor
        )
        _regress_all(state, params.regress_driver, fraction, params.initial_rating)
    state.season = season


def state_for_season(state: RatingState, season: int, params: ModelParams) -> RatingState:
    """A copy of `state` as it would be at the start of `season`."""
    out = state.copy()
    start_season(out, season, params)
    return out


def strength(state: RatingState, driver_id: str, constructor_id: str, params: ModelParams) -> float:
    return state.driver.get(driver_id, params.initial_rating) + state.constructor.get(
        constructor_id, params.initial_rating
    )


def state_strengths(
    state: RatingState, driver_id: str, constructor_id: str, track_type: str, params: ModelParams
) -> tuple[float, float]:
    """(dry, wet) strength using the conditional ratings held in `state`."""
    d = state.driver.get(driver_id, params.initial_rating)
    c = state.constructor.get(constructor_id, params.initial_rating)
    return strengths(
        driver_overall=d,
        constructor_overall=c,
        driver_track=state.driver_track.get(track_type, {}).get(driver_id, d),
        constructor_track=state.constructor_track.get(track_type, {}).get(constructor_id, c),
        n_driver_track=state.driver_track_n.get(track_type, {}).get(driver_id, 0),
        n_constructor_track=state.constructor_track_n.get(track_type, {}).get(constructor_id, 0),
        driver_wet=state.driver_wet.get(driver_id, d),
        constructor_wet=state.constructor_wet.get(constructor_id, c),
        n_driver_wet=state.driver_wet_n.get(driver_id, 0),
        n_constructor_wet=state.constructor_wet_n.get(constructor_id, 0),
        params=params,
    )


def _apply_conditional(
    ratings_d: dict[str, float],
    ratings_c: dict[str, float],
    counts_d: dict[str, int],
    counts_c: dict[str, int],
    finishers: list[tuple[str, str, int]],
    state: RatingState,
    params: ModelParams,
    k_scale: float,
) -> None:
    for d, c, _ in finishers:
        ratings_d.setdefault(d, state.driver[d])
        ratings_c.setdefault(c, state.constructor[c])
    upd = race_update(finishers, ratings_d, ratings_c, params, k_scale=k_scale)
    apply_update(ratings_d, upd.driver_delta)
    apply_update(ratings_c, upd.constructor_delta)
    for d, c, _ in finishers:
        counts_d[d] = counts_d.get(d, 0) + 1
        counts_c[c] = counts_c.get(c, 0) + 1


def replay(table: pd.DataFrame, params: ModelParams) -> tuple[pd.DataFrame, RatingState]:
    """Walk the table in date order. Returns (history, final state).

    history has one row per table row with the ratings *before* that session.
    """
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"table is missing columns: {', '.join(missing)}. Rebuild it with `f1pred data update`."
        )
    state = RatingState()
    rows: list[dict] = []
    ordered = table.sort_values(["date", "race_id", "is_sprint"], kind="stable")
    for (date, race_id, is_sprint), grp in ordered.groupby(
        ["date", "race_id", "is_sprint"], sort=True
    ):
        season = int(grp["season"].iloc[0])
        is_wet = bool(grp["is_wet"].iloc[0])
        track_type = str(grp["track_type"].iloc[0])
        start_season(state, season, params)
        d_track = state.driver_track.setdefault(track_type, {})
        c_track = state.constructor_track.setdefault(track_type, {})
        d_track_n = state.driver_track_n.setdefault(track_type, {})
        c_track_n = state.constructor_track_n.setdefault(track_type, {})
        for r in grp.itertuples(index=False):
            d = state.driver.setdefault(r.driver_id, params.initial_rating)
            c = state.constructor.setdefault(r.constructor_id, params.initial_rating)
            rows.append(
                {
                    "race_id": int(race_id),
                    "is_sprint": bool(is_sprint),
                    "season": season,
                    "round": int(r.round),
                    "date": date,
                    "driver_id": r.driver_id,
                    "constructor_id": r.constructor_id,
                    "driver_rating_pre": d,
                    "constructor_rating_pre": c,
                    "driver_races_pre": state.driver_races.get(r.driver_id, 0),
                    "is_wet": is_wet,
                    "track_type": track_type,
                    "driver_wet_pre": state.driver_wet.get(r.driver_id, d),
                    "constructor_wet_pre": state.constructor_wet.get(r.constructor_id, c),
                    "driver_wet_n_pre": state.driver_wet_n.get(r.driver_id, 0),
                    "constructor_wet_n_pre": state.constructor_wet_n.get(r.constructor_id, 0),
                    "driver_track_pre": d_track.get(r.driver_id, d),
                    "constructor_track_pre": c_track.get(r.constructor_id, c),
                    "driver_track_n_pre": d_track_n.get(r.driver_id, 0),
                    "constructor_track_n_pre": c_track_n.get(r.constructor_id, 0),
                }
            )
        finishers = [
            (r.driver_id, r.constructor_id, int(r.position))
            for r in grp.itertuples(index=False)
            if pd.notna(r.position)
        ]
        k_scale = params.sprint_weight if is_sprint else 1.0
        upd = race_update(finishers, state.driver, state.constructor, params, k_scale=k_scale)
        _apply_conditional(
            d_track, c_track, d_track_n, c_track_n, finishers, state, params, k_scale
        )
        if is_wet:
            _apply_conditional(
                state.driver_wet,
                state.constructor_wet,
                state.driver_wet_n,
                state.constructor_wet_n,
                finishers,
                state,
                params,
                k_scale,
            )
        apply_update(state.driver, upd.driver_delta)
        apply_update(state.constructor, upd.constructor_delta)
        for driver_id, _, _ in finishers:
            state.driver_races[driver_id] = state.driver_races.get(driver_id, 0) + 1
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return history, state
