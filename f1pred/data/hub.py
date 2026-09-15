"""Download RaceData tables from the Hugging Face Hub and cache them as Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

REPO_ID = "tracinginsights/RaceData"
TABLES = (
    "races",
    "results",
    "qualifying",
    "sprint_results",
    "drivers",
    "constructors",
    "circuits",
    "status",
)
NA_VALUES = ["\\N"]


class CacheMissingError(Exception):
    """A required table is not in the local cache."""


class DataUnavailableError(Exception):
    """Neither the Hub nor the cache can supply the data."""


def read_table_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)


def load_raw_from_dir(directory: Path, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    return {t: read_table_csv(Path(directory) / f"{t}.csv") for t in tables}


def download_tables(cache_dir: Path, tables: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        local = hf_hub_download(REPO_ID, f"data/{table}.csv", repo_type="dataset")
        df = read_table_csv(Path(local))
        df.to_parquet(cache_dir / f"{table}.parquet", index=False)
        out[table] = df
        log.info("downloaded %s (%d rows)", table, len(df))
    return out


def load_cached_tables(
    cache_dir: Path, tables: tuple[str, ...] = TABLES
) -> dict[str, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for table in tables:
        path = cache_dir / f"{table}.parquet"
        if not path.exists():
            raise CacheMissingError(f"{path} not found; run `f1pred data update`")
        out[table] = pd.read_parquet(path)
    return out


def load_tables(cache_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Spec 3.4: download when asked or when the cache is incomplete; otherwise use the cache.

    If the download fails and a cache exists, warn and use the cache.
    If the download fails and there is no cache, raise DataUnavailableError.
    """
    cache_dir = Path(cache_dir)
    have_cache = all((cache_dir / f"{t}.parquet").exists() for t in TABLES)
    if have_cache and not refresh:
        return load_cached_tables(cache_dir)
    try:
        return download_tables(cache_dir)
    except Exception as exc:  # network, HTTP, disk
        if have_cache:
            log.warning("Hugging Face download failed (%s); using local cache", exc)
            return load_cached_tables(cache_dir)
        raise DataUnavailableError(
            "Could not reach Hugging Face and no local cache exists. "
            "Run `f1pred data update` once you are online."
        ) from exc
