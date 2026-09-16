"""Rich tables for the terminal."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd
from rich.table import Table

from f1pred.ratings.history import RatingState
from f1pred.ratings.profile import Profile
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


def distribution_table(seasons: pd.DataFrame) -> Table:
    """Phase 7a scores over the whole finishing order (docs/metrics.md); lower is better."""
    table = Table(title="Backtest: ranked probability score and top-3 set log loss")
    cols = [
        ("Season", "season"),
        ("Races", "n_races"),
        ("RPS model", "model_rps"),
        ("RPS pole", "pole_rps"),
        ("RPS uniform", "uniform_rps"),
        ("Top3 model", "model_top3"),
        ("Top3 pole", "pole_top3"),
        ("Top3 uniform", "uniform_top3"),
    ]
    for label, _ in cols:
        table.add_column(label, justify="right")
    for r in seasons.itertuples(index=False):
        values = []
        for _, key in cols:
            v = getattr(r, key)
            if key in ("season", "n_races"):
                values.append(str(int(v)))
            elif key.endswith("_rps"):
                values.append(f"{v:.3f}")
            else:
                values.append(f"{v:.2f}")
        table.add_row(*values)
    return table


def interval_table(intervals: pd.DataFrame, note: str = "") -> Table:
    """Bootstrap 90% intervals, one row per metric (`metric, mean, lo, hi`)."""
    title = "Bootstrap 90% intervals" + (f" ({note})" if note else "")
    table = Table(title=title)
    table.add_column("metric")
    for label in ("mean", "lo", "hi"):
        table.add_column(label, justify="right")
    for r in intervals.itertuples(index=False):
        table.add_row(r.metric, f"{r.mean:.4f}", f"{r.lo:.4f}", f"{r.hi:.4f}")
    return table


def rolling_table(folds: pd.DataFrame) -> Table:
    """One row per rolling-origin fold: model vs pole log loss and RPS with 90% intervals."""
    table = Table(title="Rolling-origin folds (tuned on earlier seasons only; 90% intervals)")
    labels = ("Fold", "Train", "Races", "LogLoss model", "LogLoss pole", "RPS model", "RPS pole")
    for label in labels:
        table.add_column(label, justify="left" if label == "Train" else "right")
    for r in folds.itertuples(index=False):
        table.add_row(
            str(int(r.season)),
            str(r.train_seasons),
            str(int(r.n_races)),
            f"{r.model_logloss:.2f} [{r.lo:.2f}, {r.hi:.2f}]",
            f"{r.pole_logloss:.2f} [{r.pole_lo:.2f}, {r.pole_hi:.2f}]",
            f"{r.model_rps:.3f} [{r.rps_lo:.3f}, {r.rps_hi:.3f}]",
            f"{r.pole_rps:.3f} [{r.pole_rps_lo:.3f}, {r.pole_rps_hi:.3f}]",
        )
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


def profile_table(
    profiles: Mapping[str, Profile], names: Mapping[str, str], drivers: Iterable[str]
) -> Table:
    """Spec 6.6 numbers for `drivers`, sorted by aggression; unknown drivers show as zero."""
    table = Table(
        title="Driver profile (aggression, risk, form)",
        caption="aggression: places gained vs the field; risk: accident rate vs the field; "
        "form: recent finishes vs expected. Positive is more aggressive, riskier, hotter.",
    )
    for col, justify in [
        ("Driver", "left"),
        ("Aggression", "right"),
        ("Risk", "right"),
        ("Form", "right"),
        ("Races", "right"),
    ]:
        table.add_column(col, justify=justify)
    rows = [(d, profiles.get(d, Profile.zero())) for d in drivers]
    for d, p in sorted(rows, key=lambda dp: (-dp[1].aggression, dp[0])):
        table.add_row(
            names.get(d, d), f"{p.aggression:+.2f}", f"{p.risk:+.2f}", f"{p.form:+.2f}", str(p.n)
        )
    return table


def pace_driver_table(
    drivers: pd.DataFrame, names: Mapping[str, str], season: int, top: int = 15
) -> Table:
    """Career skill, this season's deviation and their sum, in pace units, best first."""
    table = Table(title=f"{season} driver pace (posterior mean, 90% interval, pace units)")
    for label in ("#", "Driver", "Skill", "Season", "Total"):
        table.add_column(label, justify="left" if label in ("#", "Driver") else "right")
    for i, r in enumerate(drivers.head(top).itertuples(index=False), start=1):
        table.add_row(
            str(i),
            names.get(r.driver_id, r.driver_id),
            f"{r.skill:+.2f}",
            f"{r.season:+.2f} [{r.season_lo:+.2f}, {r.season_hi:+.2f}]",
            f"{r.total:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]",
        )
    return table


def pace_car_table(cars: pd.DataFrame, season: int) -> Table:
    """Constructor season pace with its 90% interval, fastest car first."""
    table = Table(title=f"{season} car pace (posterior mean, 90% interval, pace units)")
    for label in ("#", "Constructor", "Pace", "r_hat"):
        table.add_column(label, justify="left" if label in ("#", "Constructor") else "right")
    for i, r in enumerate(cars.itertuples(index=False), start=1):
        table.add_row(
            str(i),
            r.constructor_id,
            f"{r.mean:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]",
            "-" if pd.isna(r.r_hat) else f"{r.r_hat:.3f}",
        )
    return table


def pace_diagnostics_line(diagnostics: Mapping[str, object]) -> str:
    """One line a human can act on: sampler health first, then how long it took."""
    divergences = int(diagnostics.get("divergences", 0))
    draws = int(diagnostics.get("num_samples", 0)) or 1
    r_hat = float(diagnostics.get("max_r_hat", float("nan")))
    ess = float(diagnostics.get("min_ess", float("nan")))
    return (
        f"{draws} draws from {diagnostics.get('num_chains')} chain(s) on "
        f"{diagnostics.get('device')}; max r_hat {r_hat:.3f}, min ESS {ess:.0f}, "
        f"{divergences} divergences ({100 * divergences / draws:.2f}%), "
        f"{diagnostics.get('seconds')}s"
    )
