from pathlib import Path

import pytest

from f1pred.data.hub import load_raw_from_dir

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


@pytest.fixture(scope="session")
def raw_sample() -> dict:
    return load_raw_from_dir(SAMPLE_DIR)
