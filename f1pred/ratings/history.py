"""Replay every session in date order and record the rating each entrant had going in."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from f1pred.config import ModelParams
from f1pred.ratings.elo import apply_update, race_update, regress

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
]


@dataclass
class RatingState:
    driver: dict[str, float] = field(default_factory=dict)
    constructor: dict[str, float] = field(default_factory=dict)
    driver_races: dict[str, int] = field(default_factory=dict)
    season: int | None = None

    def copy(self) -> RatingState:
        return copy.deepcopy(self)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> RatingState:
        data = json.loads(Path(path).read_text())
        return cls(**data)


def start_season(state: RatingState, season: int, params: ModelParams) -> None:
    """Spec 5.3: when a new season begins, regress ratings toward the initial value."""
    if state.season is not None and season > state.season:
        regress(state.driver, params.regress_driver, params.initial_rating)
        fraction = (
            params.regress_constructor_regulation
            if season in params.regulation_seasons
            else params.regress_constructor
        )
        regress(state.constructor, fraction, params.initial_rating)
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


def replay(table: pd.DataFrame, params: ModelParams) -> tuple[pd.DataFrame, RatingState]:
    """Walk the table in date order. Returns (history, final state).

    history has one row per table row with the ratings *before* that session.
    """
    state = RatingState()
    rows: list[dict] = []
    ordered = table.sort_values(["date", "race_id", "is_sprint"], kind="stable")
    for (date, race_id, is_sprint), grp in ordered.groupby(
        ["date", "race_id", "is_sprint"], sort=True
    ):
        season = int(grp["season"].iloc[0])
        start_season(state, season, params)
        for r in grp.itertuples(index=False):
            state.driver.setdefault(r.driver_id, params.initial_rating)
            state.constructor.setdefault(r.constructor_id, params.initial_rating)
            rows.append(
                {
                    "race_id": int(race_id),
                    "is_sprint": bool(is_sprint),
                    "season": season,
                    "round": int(r.round),
                    "date": date,
                    "driver_id": r.driver_id,
                    "constructor_id": r.constructor_id,
                    "driver_rating_pre": state.driver[r.driver_id],
                    "constructor_rating_pre": state.constructor[r.constructor_id],
                    "driver_races_pre": state.driver_races.get(r.driver_id, 0),
                }
            )
        finishers = [
            (r.driver_id, r.constructor_id, int(r.position))
            for r in grp.itertuples(index=False)
            if pd.notna(r.position)
        ]
        k_scale = params.sprint_weight if is_sprint else 1.0
        upd = race_update(finishers, state.driver, state.constructor, params, k_scale=k_scale)
        apply_update(state.driver, upd.driver_delta)
        apply_update(state.constructor, upd.constructor_delta)
        for driver_id, _, _ in finishers:
            state.driver_races[driver_id] = state.driver_races.get(driver_id, 0) + 1
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return history, state
