from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from f1pred.cli import app
from f1pred.data import hub

runner = CliRunner()
SAMPLE_DIR = Path(__file__).parent / "fixtures" / "sample"


def test_data_update_uses_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda repo_id, filename, repo_type: str(SAMPLE_DIR / Path(filename).name),
    )
    r = runner.invoke(app, ["data", "update", "--cache-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "driver_race.parquet").exists()
    assert "driver-race rows" in r.output


def test_ratings_build_then_predict_past(cache_dir, tmp_path):
    r = runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    assert r.exit_code == 0, r.output
    assert (cache_dir / "ratings_history.parquet").exists()
    assert "Max Verstappen" in r.output

    r = runner.invoke(
        app,
        [
            "predict",
            "--season",
            "2024",
            "--race",
            "monaco",
            "--runs",
            "300",
            "--seed",
            "1",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Monaco Grand Prix" in r.output and "Charles Leclerc" in r.output
    assert (tmp_path / "2024-08-win.png").exists()
    assert (tmp_path / "2024-08-positions.png").exists()


def test_predict_unknown_race_lists_choices(cache_dir):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app, ["predict", "--season", "2024", "--race", "mars", "--cache-dir", str(cache_dir)]
    )
    assert r.exit_code == 2
    assert "Monaco" in r.output


def test_predict_without_ratings_exits_1(tmp_path, raw_sample, driver_race):
    for t in hub.TABLES:
        raw_sample[t].to_parquet(tmp_path / f"{t}.parquet", index=False)
    driver_race.to_parquet(tmp_path / "driver_race.parquet", index=False)
    r = runner.invoke(
        app, ["predict", "--season", "2024", "--round", "1", "--cache-dir", str(tmp_path)]
    )
    assert r.exit_code == 1
    assert "f1pred ratings build" in r.output


def test_backtest_writes_outputs(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        [
            "backtest",
            "--seasons",
            "2024",
            "--runs",
            "200",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "2024" in r.output
    seasons = pd.read_csv(tmp_path / "backtest_seasons.csv")
    assert seasons.n_races.iloc[0] == 24
    assert (tmp_path / "calibration.png").exists()


def test_backtest_season_range_parsing(cache_dir, tmp_path):
    runner.invoke(app, ["ratings", "build", "--cache-dir", str(cache_dir)])
    r = runner.invoke(
        app,
        [
            "backtest",
            "--seasons",
            "2023-2024",
            "--runs",
            "100",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert len(pd.read_csv(tmp_path / "backtest_seasons.csv")) == 2


def test_tune_smoke(cache_dir, tmp_path, monkeypatch):
    import f1pred.cli as cli
    import f1pred.tune as tune_mod

    monkeypatch.setattr(cli, "TUNED_PARAMS_PATH", tmp_path / "tuned.json")
    monkeypatch.setattr(tune_mod, "SEARCH_SPACE", {"grid_bonus": [8.0, 12.0]})
    r = runner.invoke(
        app,
        [
            "tune",
            "--train",
            "2023",
            "--test",
            "2024",
            "--passes",
            "1",
            "--runs",
            "50",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "tuned.json").exists()
