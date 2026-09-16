"""Index tables and NumPy arrays that turn the driver-race table into model inputs.

Every array is either per *row* (one classified-or-not grand-prix entry), per *race*, or per
group (driver, driver-season, constructor-season, constructor-race, circuit, season). Rows are
laid out race by race in date order, and inside a race the classified finishers come first in
finishing order, so the Plackett-Luce likelihood can read each race as one contiguous slice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from f1pred.config import ModelParams
from f1pred.data.track_types import DEFAULT_TRACK_TYPE, TRACK_TYPES

GRID_SHAPE = 0.75  # fixed inside the Bayesian model; the tuner's grid_shape is the Elo path


@dataclass(frozen=True)
class PaceDesign:
    """Integer index arrays plus the lookup tables that name what the indices mean."""

    # Per row.
    driver: np.ndarray
    driver_season: np.ndarray
    constructor_season: np.ndarray
    constructor_race: np.ndarray
    race: np.ndarray
    circuit: np.ndarray
    track_type: np.ndarray
    is_wet: np.ndarray
    grid_g: np.ndarray
    position: np.ndarray  # 1-based finishing position, 0 = not classified
    dnf: np.ndarray
    quali_gap: np.ndarray  # percent gap to pole, NaN where unknown
    # Per race.
    race_start: np.ndarray
    race_len: np.ndarray
    race_n_classified: np.ndarray
    season_of_race: np.ndarray  # index into `seasons`
    # Per group.
    previous_driver_season: np.ndarray  # -1 for a driver's first season
    previous_constructor_season: np.ndarray  # -1 for a constructor's first season
    previous_constructor_race: np.ndarray  # -1 at the start of each season (the walk resets)
    constructor_race_season: np.ndarray  # constructor-season index of each constructor-race
    is_regulation_season: np.ndarray  # per constructor-season
    # Lookups.
    driver_ids: tuple[str, ...]
    driver_season_keys: tuple[tuple[str, int], ...]
    constructor_season_keys: tuple[tuple[str, int], ...]
    constructor_race_keys: tuple[tuple[str, int], ...]  # (constructor_id, race_id)
    race_ids: tuple[int, ...]
    circuit_ids: tuple[str, ...]
    seasons: tuple[int, ...]
    track_types: tuple[str, ...] = TRACK_TYPES

    @property
    def n_row(self) -> int:
        return int(self.driver.size)

    @property
    def n_driver(self) -> int:
        return len(self.driver_ids)

    @property
    def n_driver_season(self) -> int:
        return len(self.driver_season_keys)

    @property
    def n_constructor_season(self) -> int:
        return len(self.constructor_season_keys)

    @property
    def n_constructor_race(self) -> int:
        return len(self.constructor_race_keys)

    @property
    def n_race(self) -> int:
        return len(self.race_ids)

    @property
    def n_circuit(self) -> int:
        return len(self.circuit_ids)

    @property
    def n_season(self) -> int:
        return len(self.seasons)

    @property
    def n_track_type(self) -> int:
        return len(self.track_types)

    @property
    def max_classified(self) -> int:
        return int(self.race_n_classified.max()) if self.n_race else 0

    # Lookups used when building a target race that the design may not contain.
    def driver_index(self, driver_id: str) -> int:
        return _find(self.driver_ids, driver_id)

    def driver_season_index(self, driver_id: str, season: int) -> int:
        return _find(self.driver_season_keys, (driver_id, season))

    def constructor_season_index(self, constructor_id: str, season: int) -> int:
        return _find(self.constructor_season_keys, (constructor_id, season))

    def circuit_index(self, circuit_id: str) -> int:
        return _find(self.circuit_ids, circuit_id)

    def track_type_index(self, track_type: str) -> int:
        return _find(self.track_types, track_type or DEFAULT_TRACK_TYPE)

    def last_constructor_race(self, constructor_id: str, season: int) -> int:
        """Index of the constructor's most recent race in that season, or -1."""
        cs = self.constructor_season_index(constructor_id, season)
        if cs < 0:
            return -1
        matches = np.flatnonzero(self.constructor_race_season == cs)
        return int(matches[-1]) if matches.size else -1

    def meta(self) -> dict[str, Any]:
        """Everything a saved posterior needs to name its parameters again."""
        return {
            "driver_ids": [str(d) for d in self.driver_ids],
            "driver_season_keys": [[str(a), int(b)] for a, b in self.driver_season_keys],
            "constructor_season_keys": [[str(a), int(b)] for a, b in self.constructor_season_keys],
            "race_ids": [int(r) for r in self.race_ids],
            "circuit_ids": [str(c) for c in self.circuit_ids],
            "seasons": [int(s) for s in self.seasons],
            "track_types": [str(t) for t in self.track_types],
        }


def _find(values: Sequence[Any], needle: Any) -> int:
    try:
        return list(values).index(needle)
    except ValueError:
        return -1


def _index(values: Iterable[Any]) -> tuple[tuple[Any, ...], dict[Any, int]]:
    seen: dict[Any, int] = {}
    for v in values:
        if v not in seen:
            seen[v] = len(seen)
    return tuple(seen), seen


def build_design(
    table: pd.DataFrame,
    before_date: pd.Timestamp | None = None,
    params: ModelParams = None,  # type: ignore[assignment]
) -> PaceDesign:
    """Rows are the grand-prix entries strictly before `before_date` (all of them when None)."""
    from f1pred.config import DEFAULT_PARAMS

    params = DEFAULT_PARAMS if params is None else params
    df = table[~table["is_sprint"]].copy()
    if before_date is not None:
        df = df[df["date"] < pd.Timestamp(before_date)]
    if df.empty:
        raise ValueError("no grand-prix rows before the cutoff; nothing to fit")

    pos = pd.to_numeric(df["position"], errors="coerce")
    df["_pos"] = pos.fillna(0).astype(int)
    df["_unclassified"] = (df["_pos"] == 0).astype(int)
    # Race by race in date order; classified finishers first, in finishing order.
    df = df.sort_values(
        ["date", "race_id", "_unclassified", "_pos", "driver_id"], kind="stable"
    ).reset_index(drop=True)

    race_ids, race_of = _index(df["race_id"].astype(int))
    driver_ids, driver_of = _index(df["driver_id"])
    circuit_ids, circuit_of = _index(df["circuit_id"])
    seasons, season_of = _index(sorted(df["season"].astype(int).unique()))

    ds_keys, ds_of = _index(zip(df["driver_id"], df["season"].astype(int), strict=True))
    cs_keys, cs_of = _index(zip(df["constructor_id"], df["season"].astype(int), strict=True))
    cr_keys, cr_of = _index(zip(df["constructor_id"], df["race_id"].astype(int), strict=True))

    race = np.array([race_of[r] for r in df["race_id"].astype(int)], dtype=np.int32)
    counts = np.bincount(race, minlength=len(race_ids))
    race_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    classified = (df["_pos"].to_numpy() > 0).astype(int)
    n_classified = np.bincount(race, weights=classified, minlength=len(race_ids)).astype(np.int32)

    season_of_race = np.zeros(len(race_ids), dtype=np.int32)
    for r, s in zip(race, df["season"].astype(int), strict=True):
        season_of_race[r] = season_of[s]

    previous_driver_season = _previous_by_group(ds_keys)
    previous_constructor_season = _previous_by_group(cs_keys)

    # car_form is a random walk over each constructor's races inside a season, reset at the start.
    season_of_race_id = dict(zip(df["race_id"].astype(int), df["season"].astype(int), strict=True))
    cr_season = np.array([cs_of[(c, season_of_race_id[r])] for c, r in cr_keys], dtype=np.int32)
    previous_constructor_race = np.full(len(cr_keys), -1, dtype=np.int32)
    last_seen: dict[int, int] = {}
    for i, cs in enumerate(cr_season):
        previous_constructor_race[i] = last_seen.get(int(cs), -1)
        last_seen[int(cs)] = i

    grid = df["grid"].to_numpy(dtype=float)
    grid_g = 1.0 / np.maximum(grid, 1.0) ** GRID_SHAPE
    tt_of = {t: i for i, t in enumerate(TRACK_TYPES)}
    track_type = np.array(
        [tt_of.get(t, tt_of[DEFAULT_TRACK_TYPE]) for t in df["track_type"]], dtype=np.int32
    )
    quali = (
        df["quali_gap_pct"].to_numpy(dtype=float)
        if "quali_gap_pct" in df.columns
        else np.full(len(df), np.nan)
    )
    return PaceDesign(
        driver=np.array([driver_of[d] for d in df["driver_id"]], dtype=np.int32),
        driver_season=np.array(
            [ds_of[k] for k in zip(df["driver_id"], df["season"].astype(int), strict=True)],
            dtype=np.int32,
        ),
        constructor_season=np.array(
            [cs_of[k] for k in zip(df["constructor_id"], df["season"].astype(int), strict=True)],
            dtype=np.int32,
        ),
        constructor_race=np.array(
            [cr_of[k] for k in zip(df["constructor_id"], df["race_id"].astype(int), strict=True)],
            dtype=np.int32,
        ),
        race=race,
        circuit=np.array([circuit_of[c] for c in df["circuit_id"]], dtype=np.int32),
        track_type=track_type,
        is_wet=df["is_wet"].to_numpy(dtype=bool),
        grid_g=grid_g,
        position=df["_pos"].to_numpy(dtype=np.int32),
        dnf=df["dnf"].to_numpy(dtype=bool),
        quali_gap=quali,
        race_start=race_start,
        race_len=counts.astype(np.int32),
        race_n_classified=n_classified,
        season_of_race=season_of_race,
        previous_driver_season=previous_driver_season,
        previous_constructor_season=previous_constructor_season,
        previous_constructor_race=previous_constructor_race,
        constructor_race_season=cr_season,
        is_regulation_season=np.array(
            [s in params.regulation_seasons for _, s in cs_keys], dtype=bool
        ),
        driver_ids=driver_ids,
        driver_season_keys=ds_keys,
        constructor_season_keys=cs_keys,
        constructor_race_keys=cr_keys,
        race_ids=race_ids,
        circuit_ids=circuit_ids,
        seasons=seasons,
    )


def _previous_by_group(keys: Sequence[tuple[str, int]]) -> np.ndarray:
    """For each (id, season) key, the index of that id's previous season, else -1."""
    by_id: dict[str, list[tuple[int, int]]] = {}
    for i, (name, season) in enumerate(keys):
        by_id.setdefault(name, []).append((season, i))
    out = np.full(len(keys), -1, dtype=np.int32)
    for entries in by_id.values():
        entries.sort()
        for (_, prev), (_, cur) in zip(entries, entries[1:], strict=False):
            out[cur] = prev
    return out


def target_design(design: PaceDesign, entrants: Sequence[Any], race: Any) -> dict[str, tuple]:
    """Map each entrant to `(driver, constructor_season, circuit)` indices; -1 when unseen."""
    circuit = design.circuit_index(getattr(race, "circuit_id", ""))
    season = int(race.season)
    return {
        e.driver_id: (
            design.driver_index(e.driver_id),
            design.constructor_season_index(e.constructor_id, season),
            circuit,
        )
        for e in entrants
    }
