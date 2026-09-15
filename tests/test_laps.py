from pathlib import Path

import pandas as pd

from f1pred.data import hub
from f1pred.data.laps import LAP1_COLUMNS, download_lap1, load_cached_lap1, load_lap1_csv

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


def test_load_lap1_csv():
    df = load_lap1_csv(SAMPLE_DIR / "lap1.csv")
    assert list(df.columns) == LAP1_COLUMNS and len(df) == 895


def test_download_lap1_filters_and_caches(tmp_path, monkeypatch):
    src = tmp_path / "lap_times.csv"
    pd.DataFrame(
        {
            "raceId": [1, 1, 1],
            "driverId": [7, 7, 8],
            "lap": [1, 2, 1],
            "position": [3, 2, 1],
            "time": ["1:30.0"] * 3,
            "milliseconds": [90000] * 3,
        }
    ).to_csv(src, index=False)
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo_id, filename, repo_type: str(src))
    df = download_lap1(tmp_path)
    assert list(df.columns) == LAP1_COLUMNS and len(df) == 2
    assert load_cached_lap1(tmp_path).equals(df)
    assert load_cached_lap1(tmp_path / "empty") is None
