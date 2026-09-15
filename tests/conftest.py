from pathlib import Path

import pandas as pd
import pytest

from f1pred.data.frame import build_driver_race_table
from f1pred.data.hub import load_raw_from_dir

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


@pytest.fixture(scope="session")
def raw_sample() -> dict:
    return load_raw_from_dir(SAMPLE_DIR)


@pytest.fixture(scope="session")
def driver_race(raw_sample) -> pd.DataFrame:
    return build_driver_race_table(raw_sample, start_season=2010)
