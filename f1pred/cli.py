"""The `f1pred` command line."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from f1pred.backtest.run import run_backtest
from f1pred.config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, load_params
from f1pred.data.frame import build_driver_race_table
from f1pred.data.hub import DataUnavailableError, load_cached_tables, load_tables
from f1pred.predict import UnknownRaceError, build_prediction_inputs, resolve_race
from f1pred.ratings.history import RatingState, replay
from f1pred.report.charts import calibration_chart, position_heatmap, win_chart
from f1pred.report.tables import backtest_table, forecast_table, ratings_table
from f1pred.sim.race import simulate_race

DRIVER_RACE_FILE = "driver_race.parquet"
HISTORY_FILE = "ratings_history.parquet"
STATE_FILE = "ratings_state.json"

app = typer.Typer(help="F1 race predictions with Elo ratings and Monte Carlo simulation.")
data_app = typer.Typer(help="Download and cache data.")
ratings_app = typer.Typer(help="Build ratings.")
app.add_typer(data_app, name="data")
app.add_typer(ratings_app, name="ratings")
console = Console()

CacheDir = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", help="Parquet cache directory")
OutDir = typer.Option(DEFAULT_OUTPUT_DIR, "--out", help="Where charts and CSVs go")


def _fail(message: str, code: int) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code)


def _load_cached(cache_dir: Path) -> tuple[dict, pd.DataFrame]:
    try:
        raw = load_cached_tables(cache_dir)
    except Exception as exc:
        _fail(f"{exc}\nRun `f1pred data update` first.", 1)
    path = cache_dir / DRIVER_RACE_FILE
    if not path.exists():
        _fail(f"{path} not found. Run `f1pred data update` first.", 1)
    return raw, pd.read_parquet(path)


def _load_ratings(cache_dir: Path) -> tuple[pd.DataFrame, RatingState]:
    hist, state = cache_dir / HISTORY_FILE, cache_dir / STATE_FILE
    if not hist.exists() or not state.exists():
        _fail("Ratings not built yet. Run `f1pred ratings build` first.", 1)
    return pd.read_parquet(hist), RatingState.from_json(state)


def _parse_seasons(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@data_app.command("update")
def data_update(cache_dir: Path = CacheDir) -> None:
    """Download the RaceData tables from Hugging Face and build the driver-race table."""
    params = load_params()
    try:
        raw = load_tables(cache_dir, refresh=True)
    except DataUnavailableError as exc:
        _fail(str(exc), 1)
    table = build_driver_race_table(raw, start_season=params.start_season)
    table.to_parquet(cache_dir / DRIVER_RACE_FILE, index=False)
    races = table[~table.is_sprint]
    console.print(
        f"{len(table):,} driver-race rows, {races.race_id.nunique()} races, "
        f"{races.season.min()}-{races.season.max()}, latest: {races.race_name.iloc[-1]} "
        f"{races.date.iloc[-1].date()}"
    )


@ratings_app.command("build")
def ratings_build(cache_dir: Path = CacheDir) -> None:
    """Replay history and write pre-race ratings plus the current rating state."""
    params = load_params()
    _, table = _load_cached(cache_dir)
    history, state = replay(table, params)
    history.to_parquet(cache_dir / HISTORY_FILE, index=False)
    state.to_json(cache_dir / STATE_FILE)
    names = dict(zip(table.driver_id, table.driver_name, strict=True))
    drivers, cons = ratings_table(state, names)
    console.print(drivers)
    console.print(cons)


@app.command()
def predict(
    season: int = typer.Option(..., "--season"),
    round: int | None = typer.Option(None, "--round"),
    race: str | None = typer.Option(None, "--race"),
    runs: int = typer.Option(10_000, "--runs"),
    seed: int = typer.Option(0, "--seed"),
    no_grid: bool = typer.Option(False, "--no-grid"),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Win, podium and points probabilities for one race."""
    params = load_params()
    raw, table = _load_cached(cache_dir)
    history, state = _load_ratings(cache_dir)
    try:
        ref = resolve_race(raw, table, season, round=round, race=race)
    except UnknownRaceError as exc:
        console.print(f"[red]{exc}[/red]\nRaces in {season}:")
        for c in exc.choices:
            console.print("  " + c)
        raise typer.Exit(2) from None
    inputs = build_prediction_inputs(raw, table, history, state, ref, params, use_grid=not no_grid)
    if inputs.note:
        console.print(f"[yellow]{inputs.note}[/yellow]")
    forecast = simulate_race(
        inputs.entrants, params, n_runs=runs, seed=seed, use_grid=inputs.use_grid
    )
    low = {e.driver_id for e in inputs.entrants if e.low_confidence}
    title = f"{ref.name} {ref.season} (round {ref.round})"
    console.print(forecast_table(forecast, inputs.names, low, title))
    stem = f"{ref.season}-{ref.round:02d}"
    win_path = win_chart(forecast, inputs.names, title, out / f"{stem}-win.png")
    pos_path = position_heatmap(forecast, inputs.names, title, out / f"{stem}-positions.png")
    console.print(f"Charts: {win_path}, {pos_path}")


@app.command()
def backtest(
    seasons: str = typer.Option("2023-2025", "--seasons", help="e.g. 2023-2025 or 2022,2024"),
    runs: int = typer.Option(10_000, "--runs"),
    seed: int = typer.Option(0, "--seed"),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Score the model on past seasons against the pole-wins and uniform baselines."""
    params = load_params()
    _, table = _load_cached(cache_dir)
    history, _ = _load_ratings(cache_dir)
    result = run_backtest(table, history, _parse_seasons(seasons), params, n_runs=runs, seed=seed)
    console.print(backtest_table(result.seasons))
    out.mkdir(parents=True, exist_ok=True)
    result.races.to_csv(out / "backtest_races.csv", index=False)
    result.seasons.to_csv(out / "backtest_seasons.csv", index=False)
    console.print(
        f"Calibration chart: {calibration_chart(result.calibration, out / 'calibration.png')}"
    )


if __name__ == "__main__":
    app()
