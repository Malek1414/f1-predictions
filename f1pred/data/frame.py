"""Build the one-row-per-driver-per-session table that every downstream module consumes."""

from __future__ import annotations

import pandas as pd

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
)

ACCIDENT_STATUSES = {"Accident", "Collision", "Spun off", "Collision damage", "Damage"}
OTHER_STATUSES = {
    "Disqualified",
    "Retired",
    "Withdrew",
    "Did not qualify",
    "Did not prequalify",
    "Not classified",
    "Excluded",
    "107% Rule",
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


def classify_dnf(status: str, classified: bool) -> str | None:
    """Spec 3.5: accident, mechanical, other, or None for a classified finisher."""
    if classified:
        return None
    if status in ACCIDENT_STATUSES:
        return "accident"
    if status in OTHER_STATUSES or status.startswith("+"):
        return "other"
    return "mechanical"


def _session_rows(
    results: pd.DataFrame,
    races: pd.DataFrame,
    lookups: dict[str, pd.DataFrame],
    date_col: str,
    is_sprint: bool,
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

    position = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    classified = position.notna()
    dnf_kind = [
        classify_dnf(s, c) for s, c in zip(df["status"].fillna(""), classified, strict=True)
    ]

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
        }
    )
    out["dnf"] = out["dnf_kind"].isin(["accident", "mechanical"])
    return out[list(DRIVER_RACE_COLUMNS)]


def build_driver_race_table(raw: dict[str, pd.DataFrame], start_season: int = 2010) -> pd.DataFrame:
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
    race_rows = _session_rows(raw["results"], races, lookups, "date", is_sprint=False)
    sprint_rows = _session_rows(
        raw["sprint_results"], races, lookups, "sprint_date", is_sprint=True
    )
    table = pd.concat([race_rows, sprint_rows], ignore_index=True)
    table = table.sort_values(["date", "race_id", "grid"], kind="stable").reset_index(drop=True)
    return table
