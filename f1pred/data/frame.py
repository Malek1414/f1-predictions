"""Build the one-row-per-driver-per-session table that every downstream module consumes."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from f1pred.data.track_types import DEFAULT_TRACK_TYPE, track_type_for

log = logging.getLogger(__name__)

DRIVER_RACE_COLUMNS = (
    "season",
    "round",
    "race_id",
    "date",
    "circuit_id",
    "race_name",
    "driver_id",
    "driver_code",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "grid",
    "position",
    "status",
    "dnf",
    "dnf_kind",
    "points",
    "is_sprint",
    "is_wet",
    "track_type",
    "lap1_position",
    "quali_gap_pct",
)
# Columns produced per session; the rest are added once for the whole table.
_WHOLE_TABLE_COLUMNS = ("is_wet", "track_type", "quali_gap_pct")
_SESSION_COLUMNS = tuple(c for c in DRIVER_RACE_COLUMNS if c not in _WHOLE_TABLE_COLUMNS)

ACCIDENT_STATUSES = {"Accident", "Collision", "Spun off", "Collision damage", "Damage"}
MECHANICAL_STATUSES = {
    "Engine",
    "Gearbox",
    "Transmission",
    "Clutch",
    "Hydraulics",
    "Electrical",
    "Radiator",
    "Suspension",
    "Brakes",
    "Differential",
    "Overheating",
    "Mechanical",
    "Tyre",
    "Puncture",
    "Driveshaft",
    "Fuel pressure",
    "Front wing",
    "Water pressure",
    "Refuelling",
    "Wheel",
    "Throttle",
    "Steering",
    "Technical",
    "Electronics",
    "Broken wing",
    "Heat shield fire",
    "Exhaust",
    "Oil leak",
    "Wheel rim",
    "Water leak",
    "Fuel pump",
    "Track rod",
    "Oil pressure",
    "Engine fire",
    "Engine misfire",
    "Tyre puncture",
    "Out of fuel",
    "Wheel nut",
    "Pneumatics",
    "Handling",
    "Rear wing",
    "Fire",
    "Wheel bearing",
    "Fuel system",
    "Oil line",
    "Fuel rig",
    "Launch control",
    "Fuel",
    "Power loss",
    "Vibrations",
    "Vibration",
    "Safety switch",
    "Drivetrain",
    "Ignition",
    "Chassis",
    "Battery",
    "Stalled",
    "Halfshaft",
    "Crankshaft",
    "Alternator",
    "Oil pump",
    "Fuel leak",
    "Fuel pipe",
    "Power Unit",
    "ERS",
    "Brake duct",
    "Seat",
    "Undertray",
    "Cooling system",
    "Spark plugs",
    "Turbo",
    "CV joint",
    "Water pump",
    "Debris",
}
# Administrative and non-mechanical reasons for not being classified: never a DNF.
OTHER_STATUSES = {
    "Disqualified",
    "Retired",
    "Withdrew",
    "Not classified",
    "Did not qualify",
    "Did not prequalify",
    "Excluded",
    "107% Rule",
    "Underweight",
    "Injured",
    "Injury",
    "Illness",
    "Fatal accident",
    "Safety concerns",
    "Not restarted",
    "Driver unwell",
    "Eye injury",
    "Physical",
    "Safety",
    "Finished",
}
_warned_statuses: set[str] = set()


def classify_dnf(status: str, classified: bool) -> str | None:
    """Spec 3.5: accident, mechanical, other, or None for a classified finisher.

    Whitelist-based: a status in none of the lists is `other` and is logged once, so that new
    upstream statuses surface instead of silently counting as mechanical failures.
    """
    if classified:
        return None
    if status in ACCIDENT_STATUSES:
        return "accident"
    if status in MECHANICAL_STATUSES:
        return "mechanical"
    if status not in OTHER_STATUSES and not status.startswith("+"):
        if status not in _warned_statuses:
            _warned_statuses.add(status)
            log.warning("unknown status %r for a non-classified row; treated as other", status)
    return "other"


def _session_rows(
    results: pd.DataFrame,
    races: pd.DataFrame,
    lookups: dict[str, pd.DataFrame],
    date_col: str,
    is_sprint: bool,
    lap1: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = results.merge(races, on="raceId", how="inner")
    df = df.merge(lookups["drivers"], on="driverId", how="left")
    df = df.merge(lookups["constructors"], on="constructorId", how="left")
    df = df.merge(lookups["circuits"], on="circuitId", how="left")
    df = df.merge(lookups["status"], on="statusId", how="left")
    df = df[df[date_col].notna()].copy()

    n_entrants = df.groupby("raceId")["driverId"].transform("count")
    grid = df["grid"].fillna(0).astype(int)
    df["grid_fixed"] = grid.where(grid >= 1, n_entrants)

    # From 2025 the upstream CSV fills `position` with positionOrder for retirements, so the
    # only reliable finisher marker is a numeric `positionText` (R, W, D, ... otherwise).
    position = pd.to_numeric(df["positionText"], errors="coerce").astype("Int64")
    classified = position.notna()
    dnf_kind = [
        classify_dnf(s, c) for s, c in zip(df["status"].fillna(""), classified, strict=True)
    ]
    if lap1 is not None and not is_sprint:
        # Lap-1 positions only exist for grands prix; one row per (race, driver).
        l1 = lap1[["raceId", "driverId", "lap1_position"]].drop_duplicates(["raceId", "driverId"])
        joined = df[["raceId", "driverId"]].merge(l1, on=["raceId", "driverId"], how="left")
        lap1_position = pd.array(
            pd.to_numeric(joined["lap1_position"], errors="coerce").to_numpy(), dtype="Int64"
        )
    else:
        lap1_position = pd.array([pd.NA] * len(df), dtype="Int64")

    out = pd.DataFrame(
        {
            "season": df["year"].astype(int),
            "round": df["round"].astype(int),
            "race_id": df["raceId"].astype(int),
            "date": pd.to_datetime(df[date_col]),
            "circuit_id": df["circuitRef"].astype(str),
            "race_name": df["name_race"].astype(str),
            "driver_id": df["driverRef"].astype(str),
            "driver_code": df["code"].fillna("").astype(str),
            "driver_name": (df["forename"] + " " + df["surname"]).astype(str),
            "constructor_id": df["constructorRef"].astype(str),
            "constructor_name": df["name_constructor"].astype(str),
            "grid": df["grid_fixed"].astype(int),
            "position": position,
            "status": df["status"].fillna("").astype(str),
            "dnf_kind": pd.Series(dnf_kind, index=df.index, dtype=object),
            "points": df["points"].fillna(0).astype(float),
            "is_sprint": is_sprint,
            "lap1_position": lap1_position,
        }
    )
    out["dnf"] = out["dnf_kind"].isin(["accident", "mechanical"])
    return out[list(_SESSION_COLUMNS)]


def build_driver_race_table(
    raw: dict[str, pd.DataFrame],
    start_season: int = 2010,
    weather: pd.DataFrame | None = None,
    track_types: Mapping[str, str] | None = None,
    lap1: pd.DataFrame | None = None,
    qualifying: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spec 3.5. `weather` is the race_id -> is_wet table; sprint rows share their race's is_wet.

    With `weather=None` every row is dry; with `track_types=None` every row is `mixed`;
    with `lap1=None` (the `raceId, driverId, lap1_position` table) `lap1_position` is all NA;
    with `qualifying=None` (the `qualifying_gaps` table) `quali_gap_pct` is all NaN.
    """
    races = raw["races"].rename(columns={"name": "name_race"})
    races = races[races["year"] >= start_season][
        ["raceId", "year", "round", "circuitId", "name_race", "date", "sprint_date"]
    ]
    lookups = {
        "drivers": raw["drivers"][["driverId", "driverRef", "code", "forename", "surname"]],
        "constructors": raw["constructors"][["constructorId", "constructorRef", "name"]].rename(
            columns={"name": "name_constructor"}
        ),
        "circuits": raw["circuits"][["circuitId", "circuitRef"]],
        "status": raw["status"][["statusId", "status"]],
    }
    race_rows = _session_rows(raw["results"], races, lookups, "date", is_sprint=False, lap1=lap1)
    sprint_rows = _session_rows(
        raw["sprint_results"], races, lookups, "sprint_date", is_sprint=True
    )
    table = pd.concat([race_rows, sprint_rows], ignore_index=True)
    if weather is not None and len(weather):
        wet = weather.set_index("race_id")["is_wet"].astype(bool)
        table["is_wet"] = table["race_id"].map(wet).fillna(False).astype(bool)
    else:
        table["is_wet"] = False
    mapping = track_types or {}
    table["track_type"] = table["circuit_id"].map(
        lambda c: track_type_for(c, mapping) if mapping else DEFAULT_TRACK_TYPE
    )
    if qualifying is not None and len(qualifying):
        gap = qualifying.set_index(["race_id", "driver_id"])["gap_pct"].astype(float)
        keys = pd.MultiIndex.from_arrays([table["race_id"], table["driver_id"]])
        table["quali_gap_pct"] = gap.reindex(keys).to_numpy(dtype=float)
    else:
        table["quali_gap_pct"] = float("nan")
    table = table.sort_values(["date", "race_id", "grid"], kind="stable").reset_index(drop=True)
    return table[list(DRIVER_RACE_COLUMNS)]
