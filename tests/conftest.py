from pathlib import Path

import pandas as pd
import pytest

from f1pred.data.frame import build_driver_race_table
from f1pred.data.hub import TABLES, load_raw_from_dir
from f1pred.data.track_types import load_track_types
from f1pred.data.weather import load_weather_csv

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Tests never touch the network: any Open-Meteo call a test did not patch fails fast."""

    def offline(url, params):
        raise OSError(f"network disabled in tests: {url}")

    monkeypatch.setattr("f1pred.data.weather.get_json", offline)


@pytest.fixture(scope="session")
def raw_sample() -> dict:
    return load_raw_from_dir(SAMPLE_DIR)


@pytest.fixture(scope="session")
def weather_sample() -> pd.DataFrame:
    return load_weather_csv(SAMPLE_DIR / "weather.csv")


@pytest.fixture(scope="session")
def driver_race(raw_sample, weather_sample) -> pd.DataFrame:
    return build_driver_race_table(
        raw_sample, start_season=2010, weather=weather_sample, track_types=load_track_types()
    )


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory, raw_sample, driver_race) -> Path:
    """A cache directory pre-populated from the sample, as `f1pred data update` would leave it."""
    d = tmp_path_factory.mktemp("cache")
    for t in TABLES:
        raw_sample[t].to_parquet(d / f"{t}.parquet", index=False)
    driver_race.to_parquet(d / "driver_race.parquet", index=False)
    return d
