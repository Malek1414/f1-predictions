"""Rich tables for the terminal."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from rich.table import Table

from f1pred.ratings.history import RatingState
from f1pred.sim.race import RaceForecast
from f1pred.sim.season import SeasonForecast


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def forecast_table(
    forecast: RaceForecast, names: Mapping[str, str], low_confidence: set[str], title: str
) -> Table:
    table = Table(title=title, caption="* fewer than 5 rated races: low confidence")
    for col, justify in [
        ("#", "right"),
        ("Driver", "left"),
        ("Win", "right"),
        ("Podium", "right"),
        ("Points", "right"),
        ("Exp. pos", "right"),
        ("P10-P90", "right"),
    ]:
        table.add_column(col, justify=justify)
    for i, r in enumerate(forecast.as_frame().itertuples(index=False), start=1):
        label = names.get(r.driver_id, r.driver_id) + (
            " *" if r.driver_id in low_confidence else ""
        )
        table.add_row(
            str(i),
            label,
            _pct(r.p_win),
            _pct(r.p_podium),
            _pct(r.p_points),
            f"{r.expected_position:4.1f}",
            f"{int(r.position_p10)}-{int(r.position_p90)}",
        )
    return table


def ratings_table(
    state: RatingState, names: Mapping[str, str], top: int = 10
) -> tuple[Table, Table]:
    drivers = Table(title=f"Top {top} drivers (Elo)")
    drivers.add_column("#", justify="right")
    drivers.add_column("Driver")
    drivers.add_column("Rating", justify="right")
    for i, (d, r) in enumerate(sorted(state.driver.items(), key=lambda kv: -kv[1])[:top], 1):
        drivers.add_row(str(i), names.get(d, d), f"{r:7.1f}")
    cons = Table(title=f"Top {top} constructors (Elo)")
    cons.add_column("#", justify="right")
    cons.add_column("Constructor")
    cons.add_column("Rating", justify="right")
    for i, (c, r) in enumerate(sorted(state.constructor.items(), key=lambda kv: -kv[1])[:top], 1):
        cons.add_row(str(i), c, f"{r:7.1f}")
    return drivers, cons


def backtest_table(seasons: pd.DataFrame) -> Table:
    table = Table(title="Backtest: model vs baselines (lower log loss / Brier is better)")
    cols = [
        ("Season", "season"),
        ("Races", "n_races"),
        ("LogLoss model", "model_logloss"),
        ("LogLoss pole", "pole_logloss"),
        ("LogLoss uniform", "uniform_logloss"),
        ("Brier model", "model_brier"),
        ("Brier pole", "pole_brier"),
        ("Brier uniform", "uniform_brier"),
        ("Spearman model", "model_spearman"),
        ("Spearman pole", "pole_spearman"),
    ]
    for label, _ in cols:
        table.add_column(label, justify="right")
    for r in seasons.itertuples(index=False):
        values = []
        for _, key in cols:
            v = getattr(r, key)
            values.append(str(int(v)) if key in ("season", "n_races") else f"{v:.2f}")
        table.add_row(*values)
    return table


def title_tables(
    forecast: SeasonForecast, names: Mapping[str, str], top: int = 10
) -> tuple[Table, Table]:
    drivers = Table(
        title=f"{forecast.season} drivers' championship "
        f"({forecast.n_remaining} races left, {forecast.n_runs:,} runs)"
    )
    for col, justify in [
        ("#", "right"),
        ("Driver", "left"),
        ("Points now", "right"),
        ("Exp. points", "right"),
        ("Title", "right"),
        ("Top 3", "right"),
    ]:
        drivers.add_column(col, justify=justify)
    for i, r in enumerate(forecast.drivers_frame().head(top).itertuples(index=False), start=1):
        drivers.add_row(
            str(i),
            names.get(r.driver_id, r.driver_id),
            f"{r.current_points:.0f}",
            f"{r.expected_points:.0f}",
            _pct(r.p_title),
            _pct(r.p_top3),
        )
    cons = Table(title=f"{forecast.season} constructors' championship")
    for col, justify in [
        ("#", "right"),
        ("Constructor", "left"),
        ("Points now", "right"),
        ("Exp. points", "right"),
        ("Title", "right"),
    ]:
        cons.add_column(col, justify=justify)
    for i, r in enumerate(forecast.constructors_frame().head(top).itertuples(index=False), start=1):
        cons.add_row(
            str(i),
            r.constructor_id,
            f"{r.current_points:.0f}",
            f"{r.expected_points:.0f}",
            _pct(r.p_title),
        )
    return drivers, cons
