from pathlib import Path

import pandas as pd
import pytest

from f1pred.data.frame import build_driver_race_table
from f1pred.data.hub import TABLES, load_raw_from_dir
from f1pred.data.weather import load_weather_csv

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


@pytest.fixture(scope="session")
def raw_sample() -> dict:
    return load_raw_from_dir(SAMPLE_DIR)


@pytest.fixture(scope="session")
def weather_sample() -> pd.DataFrame:
    return load_weather_csv(SAMPLE_DIR / "weather.csv")


@pytest.fixture(scope="session")
def driver_race(raw_sample) -> pd.DataFrame:
    return build_driver_race_table(raw_sample, start_season=2010)


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory, raw_sample, driver_race) -> Path:
    """A cache directory pre-populated from the sample, as `f1pred data update` would leave it."""
    d = tmp_path_factory.mktemp("cache")
    for t in TABLES:
        raw_sample[t].to_parquet(d / f"{t}.parquet", index=False)
    driver_race.to_parquet(d / "driver_race.parquet", index=False)
    return d
