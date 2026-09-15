"""Lap-1 positions from the lap_times table, for the aggression measure. Spec 6.6."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pred.data import hub
from f1pred.data.hub import REPO_ID, read_table_csv

LAP1_FILE = "lap1.parquet"
LAP1_COLUMNS = ["raceId", "driverId", "lap1_position"]


def _lap1_rows(lap_times: pd.DataFrame) -> pd.DataFrame:
    """Keep lap 1 only; a file that already lacks a `lap` column is taken as lap-1 rows."""
    df = lap_times
    if "lap" in df.columns:
        df = df[df["lap"] == 1]
    df = df[["raceId", "driverId", "position"]]
    return df.rename(columns={"position": "lap1_position"}).reset_index(drop=True)[LAP1_COLUMNS]


def load_lap1_csv(path: Path) -> pd.DataFrame:
    return _lap1_rows(read_table_csv(Path(path)))


def download_lap1(cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = hub.hf_hub_download(REPO_ID, "data/lap_times.csv", repo_type="dataset")
    df = _lap1_rows(read_table_csv(Path(local)))
    df.to_parquet(cache_dir / LAP1_FILE, index=False)
    return df


def load_cached_lap1(cache_dir: Path) -> pd.DataFrame | None:
    path = Path(cache_dir) / LAP1_FILE
    return pd.read_parquet(path) if path.exists() else None
