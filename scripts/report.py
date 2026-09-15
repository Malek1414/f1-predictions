"""Render the current f1pred simulation as a multi-page PDF report."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from f1pred.backtest.run import run_backtest  # noqa: E402
from f1pred.config import DEFAULT_CACHE_DIR, load_params  # noqa: E402
from f1pred.data.hub import load_cached_tables  # noqa: E402
from f1pred.data.track_types import load_track_types, track_type_for  # noqa: E402
from f1pred.data.weather import circuit_wet_rate  # noqa: E402
from f1pred.predict import build_prediction_inputs, resolve_race  # noqa: E402
from f1pred.ratings.history import RatingState  # noqa: E402
from f1pred.ratings.profile import latest_profiles  # noqa: E402
from f1pred.sim.race import simulate_race  # noqa: E402
from f1pred.sim.season import (  # noqa: E402
    current_standings,
    remaining_calendar,
    season_entrants,
    simulate_season,
)

SEASON, RACE = 2026, "baku"
OUT_PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/f1-predictions-report.pdf")
PNG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else None

# Palette (dataviz reference instance, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("f1blues", SEQ)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)
PAGE = (8.27, 11.69)  # A4 portrait


def clean(ax, x_grid=True):
    ax.tick_params(length=0)
    if x_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def header(fig, title, subtitle):
    import textwrap

    fig.text(0.06, 0.965, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(
        0.06,
        0.94,
        "\n".join(textwrap.wrap(subtitle, 118)),
        fontsize=9,
        color=INK2,
        va="top",
        linespacing=1.4,
    )


def footer(fig, page, note):
    fig.text(0.06, 0.03, note, fontsize=7.5, color=MUTED, va="bottom", wrap=True)
    fig.text(0.94, 0.03, f"{page}", fontsize=8, color=MUTED, ha="right", va="bottom")


def hbar(ax, labels, values, color, fmt="{:.1f}%", highlight=None):
    y = np.arange(len(labels))[::-1]
    colors = [color] * len(labels)
    if highlight:
        colors = [color if lab not in highlight else MUTED for lab in labels]
    ax.barh(y, values, height=0.62, color=colors, linewidth=0)
    ax.set_yticks(y, labels)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values, strict=True):
        ax.text(v + vmax * 0.012, yi, fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, vmax * 1.18)
    clean(ax)


def main():
    params = load_params()
    raw = load_cached_tables(DEFAULT_CACHE_DIR)
    table = pd.read_parquet(DEFAULT_CACHE_DIR / "driver_race.parquet")
    history = pd.read_parquet(DEFAULT_CACHE_DIR / "ratings_history.parquet")
    state = RatingState.from_json(DEFAULT_CACHE_DIR / "ratings_state.json")
    names = dict(zip(table.driver_id, table.driver_name, strict=True))
    team_names = dict(zip(table.constructor_id, table.constructor_name, strict=True))
    latest_race = table[~table.is_sprint].sort_values("date").iloc[-1]
    data_note = (
        f"Data: Hugging Face tracinginsights/RaceData through the {latest_race.race_name} "
        f"{int(latest_race.season)} ({latest_race.date.date()}), Open-Meteo rain. "
        "Model: driver + constructor Elo, Monte Carlo with team/driver noise, grid bonus, "
        "DNF layer, wet and track-type ratings, driver profile."
    )

    # ---- Race ------------------------------------------------------------
    ref = resolve_race(raw, table, SEASON, race=RACE)
    inputs = build_prediction_inputs(raw, table, history, state, ref, params)
    forecast = simulate_race(
        inputs.entrants,
        params,
        n_runs=10_000,
        seed=0,
        use_grid=inputs.use_grid,
        rain_probability=inputs.rain_probability,
    )
    fdf = forecast.as_frame()
    low = {e.driver_id for e in inputs.entrants if e.low_confidence}

    # ---- Season ----------------------------------------------------------
    track_types = load_track_types()
    remaining = [
        dataclasses.replace(
            r,
            track_type=track_type_for(r.circuit_id, track_types),
            rain_probability=circuit_wet_rate(table, r.circuit_id, r.date)
            if params.use_weather
            else 0.0,
        )
        for r in remaining_calendar(raw, table, SEASON)
    ]
    profiles = latest_profiles(table, history, params) if params.use_profile else None
    entrants_by_race = [
        season_entrants(table, state, SEASON, r, params, profiles) for r in remaining
    ]
    dpts, cpts, mapping = current_standings(table, SEASON)
    sf = simulate_season(
        SEASON,
        remaining,
        entrants_by_race,
        dpts,
        cpts,
        mapping,
        params,
        n_runs=4000,
        seed=0,
        rain_by_race=[r.rain_probability for r in remaining],
    )
    sdf, cdf = sf.drivers_frame(), sf.constructors_frame()

    # ---- Backtest --------------------------------------------------------
    bt = run_backtest(table, history, [2023, 2024, 2025], params, n_runs=3000, seed=0)

    pages = []

    # Page 1: race win probabilities
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        f"{ref.name} {ref.season}: win probability",
        f"Round {ref.round}, 10,000 simulated races. "
        + (f"{inputs.note}. " if inputs.note else "")
        + f"Rain chance {inputs.rain_probability:.0%} ({inputs.rain_source}).",
    )
    ax = fig.add_axes([0.24, 0.10, 0.68, 0.79])
    hbar(
        ax, [names.get(d, d) for d in fdf.driver_id], (100 * fdf.p_win).tolist(), S1, highlight=None
    )
    ax.set_xlabel("Share of simulations won (%)")
    for lab in ax.get_yticklabels():
        if lab.get_text() in {names.get(d, d) for d in low}:
            lab.set_color(MUTED)
    footer(fig, 1, data_note + (" Grey names: fewer than 5 rated races." if low else ""))
    pages.append(fig)

    # Page 2: podium/points + position heatmap
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        f"{ref.name} {ref.season}: where each driver finishes",
        "Left: podium and points probabilities. Right: probability of every finishing position, "
        "darker is more likely. Drivers ordered by expected finish.",
    )
    order = np.argsort(forecast.expected_position)
    ids = [forecast.driver_ids[i] for i in order]
    labs = [names.get(d, d) for d in ids]
    n = len(ids)
    ax1 = fig.add_axes([0.22, 0.10, 0.22, 0.78])
    y = np.arange(n)[::-1]
    pod = 100 * forecast.p_podium[order]
    pts = 100 * forecast.p_points[order]
    ax1.barh(y + 0.18, pts, height=0.34, color="#9ec5f4", linewidth=0, label="Points (top 10)")
    ax1.barh(y - 0.18, pod, height=0.34, color=S1, linewidth=0, label="Podium")
    ax1.set_yticks(y, labs)
    ax1.set_ylim(-0.5, n - 0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("%")
    ax1.legend(loc="lower right", frameon=False, fontsize=8)
    clean(ax1)
    ax2 = fig.add_axes([0.50, 0.10, 0.42, 0.78])
    im = ax2.imshow(
        forecast.position_matrix[order],
        cmap=BLUES,
        aspect="auto",
        vmin=0,
        vmax=forecast.position_matrix.max(),
    )
    ax2.set_yticks(range(n), [""] * n)
    ax2.set_xticks(range(n), [str(k + 1) for k in range(n)], fontsize=7)
    ax2.set_xlabel("Finishing position")
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    for i in range(n):
        k = int(np.argmax(forecast.position_matrix[order][i]))
        v = forecast.position_matrix[order][i, k]
        ax2.text(
            k,
            i,
            f"{100 * v:.0f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if v > 0.5 * forecast.position_matrix.max() else INK2,
        )
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label("probability", fontsize=7, color=INK2)
    footer(fig, 2, "Numbers in the heatmap mark each driver's most likely finishing position.")
    pages.append(fig)

    # Page 3: championship
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        f"{SEASON} championship odds",
        f"{sf.n_remaining} races left, {sf.n_runs:,} simulated seasons. Ratings held fixed; "
        "points rules of the season applied per race and sprint.",
    )
    top = sdf.head(8)
    ax = fig.add_axes([0.24, 0.62, 0.68, 0.28])
    hbar(ax, [names.get(d, d) for d in top.driver_id], (100 * top.p_title).tolist(), S1)
    ax.set_title("Drivers' title probability")
    ax.set_xlabel("%")
    ax = fig.add_axes([0.24, 0.30, 0.68, 0.26])
    yy = np.arange(len(top))[::-1]
    ax.barh(
        yy,
        top.expected_points,
        height=0.62,
        color="#9ec5f4",
        linewidth=0,
        label="Expected final points",
    )
    ax.barh(yy, top.current_points, height=0.62, color=S1, linewidth=0, label="Points now")
    for yi, cur, exp in zip(yy, top.current_points, top.expected_points, strict=True):
        ax.text(exp + 4, yi, f"{cur:.0f} → {exp:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(yy, [names.get(d, d) for d in top.driver_id])
    ax.set_xlim(0, top.expected_points.max() * 1.22)
    ax.set_title("Points now and expected at season end")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    clean(ax)
    ctop = cdf.head(6)
    ax = fig.add_axes([0.24, 0.07, 0.68, 0.17])
    yy = np.arange(len(ctop))[::-1]
    ax.barh(yy, ctop.expected_points, height=0.62, color="#9ec5f4", linewidth=0)
    ax.barh(yy, ctop.current_points, height=0.62, color=S1, linewidth=0)
    for yi, cur, exp, p in zip(
        yy, ctop.current_points, ctop.expected_points, ctop.p_title, strict=True
    ):
        ax.text(
            exp + 6,
            yi,
            f"{cur:.0f} → {exp:.0f}   title {100 * p:.0f}%",
            va="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_yticks(yy, [team_names.get(c, c) for c in ctop.constructor_id])
    ax.set_xlim(0, ctop.expected_points.max() * 1.35)
    ax.set_title("Constructors")
    clean(ax)
    footer(fig, 3, "A driver who has left the grid keeps their points but scores no more.")
    pages.append(fig)

    # Page 4: driver profile
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        "Driver profile: aggression, form and risk",
        f"Rolling {params.profile_window}-race windows ({params.form_window} for form), after the "
        f"{latest_race.race_name} {int(latest_race.season)}. "
        "Positive is more aggressive, hotter, riskier.",
    )
    grid_ids = [e.driver_id for e in inputs.entrants]
    prof = {d: profiles[d] for d in grid_ids if profiles and d in profiles}
    ax = fig.add_axes([0.12, 0.42, 0.80, 0.47])
    xs = np.array([p.aggression for p in prof.values()])
    ys = np.array([p.form for p in prof.values()])
    rs = np.array([p.risk for p in prof.values()])
    sizes = 40 + 60 * np.clip(rs, 0, None)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.scatter(xs, ys, s=sizes, color=S1, alpha=0.85, linewidths=1, edgecolors=SURFACE, zorder=3)
    placed = []
    for d, x_, y_ in zip(prof, xs, ys, strict=True):
        short = names.get(d, d).split()[-1]
        dy = 4
        for px, py in placed:
            if abs(px - x_) < 0.06 and abs(py - y_) < 0.08:
                dy -= 11
        placed.append((x_, y_))
        ax.annotate(
            short, (x_, y_), xytext=(5, dy), textcoords="offset points", fontsize=7.5, color=INK2
        )
    ax.set_xlabel("Aggression: places gained on lap 1 and over the race, versus the field")
    ax.set_ylabel("Form: recent finishes versus rating expectation")
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    fig.text(
        0.92,
        0.372,
        "Bigger dot = higher accident rate than the field",
        ha="right",
        fontsize=7.5,
        color=MUTED,
    )
    ax = fig.add_axes([0.30, 0.07, 0.62, 0.28])
    rdf = sorted(prof.items(), key=lambda kv: -kv[1].risk)[:10]
    hbar(ax, [names.get(d, d) for d, _ in rdf], [p.risk for _, p in rdf], S2, fmt="{:+.2f}")
    ax.set_title("Risk: accident-type retirements relative to the field (top 10)")
    ax.set_xlim(
        min(0, min(p.risk for _, p in rdf)) - 0.1, max(p.risk for _, p in rdf) * 1.25 + 0.05
    )
    footer(
        fig,
        4,
        "Tuned scales: aggression adds pace; risk raises the retirement chance; form is off.",
    )
    pages.append(fig)

    # Page 5: backtest
    fig = plt.figure(figsize=PAGE)
    header(
        fig,
        "Does it work? Backtest on 2023 to 2025",
        "Every race predicted from information available before it started, then scored. "
        "These seasons were never used for tuning.",
    )
    ax = fig.add_axes([0.10, 0.56, 0.82, 0.29])
    seasons = bt.seasons
    x = np.arange(len(seasons))
    w = 0.26
    for off, col, colr, lab in [
        (-w, "model_logloss", S1, "Model"),
        (0, "pole_logloss", S2, "Pole sitter wins (90%)"),
        (w, "uniform_logloss", S3, "Everyone equal"),
    ]:
        vals = seasons[col].to_numpy()
        ax.bar(x + off, vals, width=w - 0.03, color=colr, linewidth=0, label=lab)
        for xi, v in zip(x + off, vals, strict=True):
            ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [str(int(s)) for s in seasons.season])
    ax.set_ylabel("Winner log loss (lower is better)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(
        0, seasons[["model_logloss", "pole_logloss", "uniform_logloss"]].to_numpy().max() * 1.12
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3)
    ax.set_title("How surprised was each approach by the actual winner?", pad=22)
    ax = fig.add_axes([0.10, 0.10, 0.36, 0.38])
    cal = bt.calibration.dropna()
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
    ax.plot(cal.predicted, cal.observed, color=S1, linewidth=2)
    ax.scatter(
        cal.predicted,
        cal.observed,
        s=15 + 200 * cal["count"] / cal["count"].max(),
        color=S1,
        zorder=3,
    )
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Calibration (dot size = count)")
    ax = fig.add_axes([0.56, 0.10, 0.36, 0.38])
    ax.axis("off")
    rows = [["Season", "Races", "Log loss", "Brier", "Spearman"]]
    for r in seasons.itertuples(index=False):
        rows.append(
            [
                str(int(r.season)),
                str(int(r.n_races)),
                f"{r.model_logloss:.2f}",
                f"{r.model_brier:.3f}",
                f"{r.model_spearman:.2f}",
            ]
        )
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.5)
    for (i, _j), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.6)
        if i == 0:
            c.set_text_props(color=INK2, fontweight="bold")
            c.set_facecolor(SURFACE)
    ax.text(
        0.5,
        0.42,
        "Log loss: surprise at the actual winner (lower is better).\n"
        "Brier: squared error of podium probabilities (lower is better).\n"
        "Spearman: rank correlation of expected vs actual finish (higher is better).",
        ha="center",
        va="top",
        fontsize=8,
        color=INK2,
        transform=ax.transAxes,
    )
    ax.set_title("Model scores per season")
    footer(
        fig,
        5,
        "Calibration: when the model says 30%, that driver should win about 30% of the time.",
    )
    pages.append(fig)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        for i, fig in enumerate(pages, 1):
            pdf.savefig(fig)
            if PNG_DIR:
                PNG_DIR.mkdir(parents=True, exist_ok=True)
                fig.savefig(PNG_DIR / f"page{i}.png", dpi=110)
            plt.close(fig)
    print(f"wrote {OUT_PDF} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
