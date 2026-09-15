import numpy as np
import pandas as pd
from rich.console import Console

from f1pred.backtest.run import calibration_table
from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import RatingState
from f1pred.report.charts import calibration_chart, position_heatmap, title_chart, win_chart
from f1pred.report.tables import backtest_table, forecast_table, ratings_table, title_tables
from f1pred.sim.race import Entrant, simulate_race
from f1pred.sim.season import RemainingRace, simulate_season

NAMES = {f"d{i}": f"Driver {i}" for i in range(6)}


def _forecast():
    ents = [Entrant(f"d{i}", f"t{i // 2}", 3000.0 + 20 * i, i + 1, 0.1) for i in range(6)]
    return simulate_race(ents, DEFAULT_PARAMS, n_runs=300, seed=0)


def _render(table) -> str:
    console = Console(record=True, width=120)
    console.print(table)
    return console.export_text()


def test_forecast_table_marks_low_confidence():
    text = _render(forecast_table(_forecast(), NAMES, {"d1"}, "Test GP"))
    assert "Driver 1" in text and "*" in text and "Test GP" in text
    assert "%" in text


def test_ratings_tables():
    state = RatingState({"d0": 1600.0, "d1": 1400.0}, {"t0": 1550.0}, {"d0": 5}, 2024)
    drivers, cons = ratings_table(state, NAMES, top=1)
    assert "Driver 0" in _render(drivers) and "Driver 1" not in _render(drivers)
    assert "t0" in _render(cons)


def test_backtest_table():
    seasons = pd.DataFrame(
        {
            "season": [2024],
            "n_races": [24],
            "model_logloss": [1.2],
            "pole_logloss": [1.5],
            "uniform_logloss": [3.0],
            "model_brier": [0.1],
            "pole_brier": [0.12],
            "uniform_brier": [0.2],
            "model_spearman": [0.7],
            "pole_spearman": [0.6],
        }
    )
    text = _render(backtest_table(seasons))
    assert "2024" in text and "1.20" in text and "3.00" in text


def test_charts_write_png(tmp_path):
    f = _forecast()
    assert win_chart(f, NAMES, "Test GP", tmp_path / "win.png").stat().st_size > 1000
    assert position_heatmap(f, NAMES, "Test GP", tmp_path / "pos.png").stat().st_size > 1000
    cal = calibration_table(np.linspace(0, 1, 50), (np.linspace(0, 1, 50) > 0.5).astype(float))
    assert calibration_chart(cal, tmp_path / "cal.png").stat().st_size > 1000


def _season_forecast():
    ents = [Entrant(f"d{i}", f"t{i // 2}", 3000.0 + 40 * i, None, 0.05) for i in range(6)]
    remaining = [RemainingRace(1, 20, "R", "c", pd.Timestamp("2030-01-01"), True)]
    pts = {e.driver_id: 10.0 * i for i, e in enumerate(ents)}
    mapping = {e.driver_id: e.constructor_id for e in ents}
    return simulate_season(
        2025, remaining, [ents], pts, {}, mapping, DEFAULT_PARAMS, n_runs=100, seed=0
    )


def test_title_tables():
    drivers, cons = title_tables(_season_forecast(), NAMES, top=3)
    text = _render(drivers)
    assert "Driver 5" in text and "Title" in text and "%" in text
    assert "t2" in _render(cons)


def test_title_chart_writes_png(tmp_path):
    assert (
        title_chart(_season_forecast(), NAMES, "2025", tmp_path / "title.png").stat().st_size > 1000
    )
