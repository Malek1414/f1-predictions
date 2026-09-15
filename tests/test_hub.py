import logging
from pathlib import Path

import pandas as pd
import pytest

from f1pred.data import hub
from f1pred.data.hub import (
    TABLES,
    CacheMissingError,
    DataUnavailableError,
    load_cached_tables,
    load_tables,
    read_table_csv,
)

SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


def test_read_table_csv_turns_backslash_n_into_null():
    df = read_table_csv(SAMPLE_DIR / "results.csv")
    assert df["position"].isna().any()
    assert pd.api.types.is_numeric_dtype(df["position"])
    assert df["statusId"].notna().all()


def test_raw_sample_has_all_tables(raw_sample):
    assert set(raw_sample) == set(TABLES)
    assert len(raw_sample["races"]) == 46


def _fake_download(repo_id, filename, repo_type):
    assert repo_id == hub.REPO_ID and repo_type == "dataset"
    return str(SAMPLE_DIR / Path(filename).name)


def test_load_tables_downloads_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "hf_hub_download", _fake_download)
    tables = load_tables(tmp_path)
    assert set(tables) == set(TABLES)
    assert (tmp_path / "results.parquet").exists()
    cached = load_cached_tables(tmp_path)
    assert len(cached["results"]) == len(tables["results"])


def test_load_tables_falls_back_to_cache_when_offline(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(hub, "hf_hub_download", _fake_download)
    load_tables(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(hub, "hf_hub_download", boom)
    with caplog.at_level(logging.WARNING):
        tables = load_tables(tmp_path, refresh=True)
    assert "cache" in caplog.text.lower()
    assert len(tables["races"]) == 46


def test_load_tables_raises_when_offline_and_no_cache(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(hub, "hf_hub_download", boom)
    with pytest.raises(DataUnavailableError, match="f1pred data update"):
        load_tables(tmp_path)


def test_load_cached_tables_missing_raises(tmp_path):
    with pytest.raises(CacheMissingError):
        load_cached_tables(tmp_path)
