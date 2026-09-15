"""PNG charts. Uses the Agg backend so it works headless."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from f1pred.sim.race import RaceForecast  # noqa: E402
from f1pred.sim.season import SeasonForecast  # noqa: E402

MIN_SHOWN = 0.005


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def win_chart(forecast: RaceForecast, names: Mapping[str, str], title: str, path: Path) -> Path:
    df = forecast.as_frame()
    labels = [names.get(d, d) for d in df.driver_id]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1.5))
    ax.barh(labels[::-1], (100 * df.p_win)[::-1], color="#e10600")
    ax.set_xlabel("Win probability (%)")
    ax.set_title(f"{title}: win probability ({forecast.n_runs:,} runs)")
    for i, v in enumerate((100 * df.p_win)[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    return _save(fig, path)


def position_heatmap(
    forecast: RaceForecast, names: Mapping[str, str], title: str, path: Path
) -> Path:
    order = np.argsort(forecast.expected_position)
    matrix = forecast.position_matrix[order]
    labels = [names.get(forecast.driver_ids[i], forecast.driver_ids[i]) for i in order]
    n = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.35 * n + 2))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=max(matrix.max(), 1e-9))
    ax.set_xticks(range(n), [str(k + 1) for k in range(n)])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Finishing position")
    ax.set_title(f"{title}: P(driver finishes in position)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, path)


def calibration_chart(calibration: pd.DataFrame, path: Path) -> Path:
    cal = calibration.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.scatter(
        cal.predicted,
        cal.observed,
        s=10 + cal["count"] / cal["count"].max() * 200,
        color="#e10600",
    )
    ax.plot(cal.predicted, cal.observed, color="#e10600", label="model")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration (bubble size = count)")
    ax.legend()
    return _save(fig, path)


def title_chart(forecast: SeasonForecast, names: Mapping[str, str], title: str, path: Path) -> Path:
    d = forecast.drivers_frame()
    d = d[d.p_title > MIN_SHOWN]
    c = forecast.constructors_frame()
    c = c[c.p_title > MIN_SHOWN]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 0.4 * max(len(d), len(c), 3) + 1.5), gridspec_kw={"width_ratios": [3, 2]}
    )
    for ax, frame, key, label in [
        (ax1, d, "driver_id", "Drivers"),
        (ax2, c, "constructor_id", "Constructors"),
    ]:
        labels = [names.get(i, i) for i in frame[key]][::-1]
        values = (100 * frame.p_title)[::-1]
        ax.barh(labels, values, color="#e10600")
        for i, v in enumerate(values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Title probability (%)")
        ax.set_title(f"{label} ({forecast.n_remaining} races left)")
    fig.suptitle(title)
    return _save(fig, path)
