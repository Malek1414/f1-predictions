"""The `f1pred` command line."""

from __future__ import annotations

import dataclasses
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from f1pred import tune as tune_mod
from f1pred.backtest.run import ablation_backtest, run_backtest
from f1pred.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    TUNED_PARAMS_PATH,
    load_params,
)
from f1pred.data.frame import build_driver_race_table
from f1pred.data.hub import DataUnavailableError, load_cached_tables, load_tables
from f1pred.data.laps import download_lap1
from f1pred.data.track_types import load_track_types, track_type_for
from f1pred.data.weather import build_weather_table, circuit_wet_rate
from f1pred.predict import UnknownRaceError, build_prediction_inputs, resolve_race
from f1pred.ratings.history import RatingState, replay
from f1pred.ratings.profile import latest_profiles
from f1pred.report.charts import calibration_chart, position_heatmap, title_chart, win_chart
from f1pred.report.tables import (
    backtest_table,
    distribution_table,
    forecast_table,
    profile_table,
    ratings_table,
    title_tables,
)
from f1pred.sim.race import simulate_race
from f1pred.sim.season import (
    current_standings,
    remaining_calendar,
    season_entrants,
    simulate_season,
)

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
    history = pd.read_parquet(hist)
    table_through = pd.read_parquet(cache_dir / DRIVER_RACE_FILE, columns=["date"])["date"].max()
    ratings_through = history["date"].max()
    if table_through > ratings_through:
        _fail(
            f"Ratings are older than the data (table through {table_through.date()}, "
            f"ratings through {ratings_through.date()}). Run `f1pred ratings build`.",
            1,
        )
    return history, RatingState.from_json(state)


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
    try:
        weather = build_weather_table(raw, cache_dir, params)
    except Exception as exc:  # spec 10: Open-Meteo down entirely; keep going without rain
        console.print(f"[yellow]Weather unavailable ({exc}); all races treated as dry.[/yellow]")
        weather = None
    try:
        lap1 = download_lap1(cache_dir)
    except Exception as exc:  # lap_times.csv is only needed for the aggression measure
        console.print(
            f"[yellow]Lap-1 positions unavailable ({exc}); "
            "aggression uses race gains only.[/yellow]"
        )
        lap1 = None
    table = build_driver_race_table(
        raw,
        start_season=params.start_season,
        weather=weather,
        track_types=load_track_types(),
        lap1=lap1,
    )
    table.to_parquet(cache_dir / DRIVER_RACE_FILE, index=False)
    races = table[~table.is_sprint]
    n_wet = int(races.groupby("race_id").is_wet.first().sum())
    console.print(
        f"{len(table):,} driver-race rows, {races.race_id.nunique()} races, "
        f"{races.season.min()}-{races.season.max()}, latest: {races.race_name.iloc[-1]} "
        f"{races.date.iloc[-1].date()}"
    )
    console.print(f"{n_wet} wet races (>= {params.wet_threshold_mm} mm in the race window)")
    console.print(
        f"Lap-1 positions for {100 * races.lap1_position.notna().mean():.0f}% of race rows"
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
    rain: float | None = typer.Option(
        None, "--rain", min=0.0, max=1.0, help="Override the rain probability (0 to 1)"
    ),
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
    inputs = build_prediction_inputs(
        raw, table, history, state, ref, params, use_grid=not no_grid, rain_probability=rain
    )
    if inputs.note:
        console.print(f"[yellow]{inputs.note}[/yellow]")
    console.print(_rain_line(inputs.rain_probability, inputs.rain_source))
    forecast = simulate_race(
        inputs.entrants,
        params,
        n_runs=runs,
        seed=seed,
        use_grid=inputs.use_grid,
        rain_probability=inputs.rain_probability,
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
    ablate: bool = typer.Option(
        False,
        "--ablate",
        help="Also score with and without weather, track type and the driver profile",
    ),
    observed_rain: bool = typer.Option(
        False,
        "--observed-rain",
        help="Use each race's observed rainfall instead of the pre-race historical wet rate "
        "(an upper bound on what a forecast could deliver)",
    ),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Score the model on past seasons against the pole-wins and uniform baselines."""
    params = load_params()
    _, table = _load_cached(cache_dir)
    history, _ = _load_ratings(cache_dir)
    season_list = _parse_seasons(seasons)
    rain_mode = "observed" if observed_rain else "historical"
    console.print(f"Rain in the backtest: {rain_mode} ({_RAIN_MODE_LABEL[rain_mode]})")
    result = run_backtest(
        table, history, season_list, params, n_runs=runs, seed=seed, observed_rain=observed_rain
    )
    console.print(backtest_table(result.seasons))
    console.print(distribution_table(result.seasons))
    console.print(
        f"ECE win {result.ece_win:.4f}, ECE podium {result.ece_podium:.4f}, "
        f"sharpness {result.sharpness:.3f} over {len(result.races)} races"
    )
    out.mkdir(parents=True, exist_ok=True)
    result.races.to_csv(out / "backtest_races.csv", index=False)
    result.seasons.to_csv(out / "backtest_seasons.csv", index=False)
    console.print(
        f"Calibration chart: {calibration_chart(result.calibration, out / 'calibration.png')}"
    )
    if ablate:
        abl = ablation_backtest(
            table,
            history,
            season_list,
            params,
            n_runs=runs,
            seed=seed,
            observed_rain=observed_rain,
        )
        console.print(_ablation_table(abl))
        abl.to_csv(out / "ablation.csv", index=False)
        console.print(f"Ablation table: {out / 'ablation.csv'}")


def _rain_line(probability: float, source: str) -> str:
    label = {
        "forecast": "forecast",
        "historical": "historical rate; forecast unavailable",
        "observed": "observed",
        "override": "override",
        "disabled": "weather disabled",
    }.get(source, source)
    return f"Rain chance: {round(100 * probability):.0f}% ({label})"


_RAIN_MODE_LABEL = {
    "historical": "pre-race circuit wet rate",
    "observed": "race-day rainfall, an upper bound",
}


def _ablation_table(abl: pd.DataFrame) -> Table:
    modes = ", ".join(sorted(set(abl.rain_mode)))
    t = Table(
        title="Ablation: with and without weather, track type and the driver profile "
        f"(rain: {modes})"
    )
    for col in ["variant", "season", "races", "log loss", "Brier", "Spearman"]:
        t.add_column(col, justify="right" if col not in ("variant", "season") else "left")
    for r in abl.itertuples(index=False):
        t.add_row(
            r.variant,
            r.season,
            str(r.n_races),
            f"{r.model_logloss:.4f}",
            f"{r.model_brier:.4f}",
            f"{r.model_spearman:.3f}",
        )
    return t


@app.command()
def season(
    season: int = typer.Option(..., "--season"),
    runs: int = typer.Option(2000, "--runs"),
    seed: int = typer.Option(0, "--seed"),
    cache_dir: Path = CacheDir,
    out: Path = OutDir,
) -> None:
    """Drivers' and constructors' title odds for the rest of a season."""
    params = load_params()
    raw, table = _load_cached(cache_dir)
    history, state = _load_ratings(cache_dir)
    seasons_available = sorted(raw["races"]["year"].unique().tolist())
    if season not in seasons_available:
        console.print(
            f"[red]No calendar for {season}.[/red] Seasons available: "
            f"{seasons_available[0]}-{seasons_available[-1]}"
        )
        raise typer.Exit(2)
    track_types = load_track_types()
    remaining = [
        dataclasses.replace(
            r,
            track_type=track_type_for(r.circuit_id, track_types),
            rain_probability=circuit_wet_rate(table, r.circuit_id, r.date)
            if params.use_weather
            else 0.0,
        )
        for r in remaining_calendar(raw, table, season)
    ]
    if not remaining:
        console.print(f"[yellow]Season {season} is complete; showing final standings.[/yellow]")
    profiles = latest_profiles(table, history, params) if params.use_profile else None
    entrants_by_race = [
        season_entrants(table, state, season, r, params, profiles) for r in remaining
    ]
    driver_points, constructor_points, mapping = current_standings(table, season)
    forecast = simulate_season(
        season,
        remaining,
        entrants_by_race,
        driver_points,
        constructor_points,
        mapping,
        params,
        n_runs=runs,
        seed=seed,
        rain_by_race=[r.rain_probability for r in remaining],
    )
    names = dict(zip(table.driver_id, table.driver_name, strict=True))
    drivers, cons = title_tables(forecast, names)
    console.print(drivers)
    console.print(cons)
    out.mkdir(parents=True, exist_ok=True)
    forecast.drivers_frame().to_csv(out / f"{season}-drivers.csv", index=False)
    forecast.constructors_frame().to_csv(out / f"{season}-constructors.csv", index=False)
    chart = title_chart(forecast, names, f"{season} championship odds", out / f"{season}-title.png")
    console.print(f"Chart: {chart}")


@app.command()
def profile(
    season: int | None = typer.Option(
        None, "--season", help="Show the entrants of that season's latest race (default: latest)"
    ),
    cache_dir: Path = CacheDir,
) -> None:
    """Aggression, risk and form for the drivers on the current grid. Spec 6.6."""
    params = load_params()
    _, table = _load_cached(cache_dir)
    history, _ = _load_ratings(cache_dir)
    races = table[~table.is_sprint]
    if season is not None:
        races = races[races.season == season]
        if races.empty:
            _fail(f"No completed races for {season}.", 2)
    latest = races[races.race_id == races.sort_values("date").race_id.iloc[-1]]
    names = dict(zip(table.driver_id, table.driver_name, strict=True))
    profiles = latest_profiles(table, history, params)
    console.print(profile_table(profiles, names, latest.driver_id.tolist()))
    after = f"{latest.race_name.iloc[0]} {int(latest.season.iloc[0])}"
    console.print(
        f"Windows: {params.profile_window} races for aggression and risk, "
        f"{params.form_window} for form; after {after}."
    )


@app.command()
def tune(
    train: str = typer.Option("2015-2022", "--train"),
    test: str = typer.Option("2023-2025", "--test"),
    passes: int = typer.Option(2, "--passes"),
    runs: int = typer.Option(2000, "--runs"),
    cache_dir: Path = CacheDir,
) -> None:
    """Search model parameters on the training seasons, report held-out seasons, save tuned.json."""
    _, table = _load_cached(cache_dir)
    train_seasons, test_seasons = _parse_seasons(train), _parse_seasons(test)
    best, score = tune_mod.coordinate_descent(
        table,
        DEFAULT_PARAMS,
        train_seasons,
        space=tune_mod.SEARCH_SPACE,
        passes=passes,
        n_runs=runs,
        log=console.print,
    )
    console.print(f"Best training log loss: {score:.4f}")
    for label, params in [("default", DEFAULT_PARAMS), ("tuned", best)]:
        history, _ = replay(table, params)
        result = run_backtest(table, history, test_seasons, params, n_runs=runs, seed=0)
        console.print(f"[bold]{label} params, held-out {test}[/bold]")
        console.print(backtest_table(result.seasons))
    note = f"tuned {date.today()} on {train}, held out {test}, train log loss {score:.4f}"
    tune_mod.write_tuned(best, TUNED_PARAMS_PATH, note)
    console.print(f"Wrote {TUNED_PARAMS_PATH}. Re-run `f1pred ratings build` to use it.")


if __name__ == "__main__":
    app()
