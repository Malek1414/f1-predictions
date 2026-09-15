import math

import numpy as np
import pytest

from f1pred.backtest.run import (
    RACE_SCORE_COLUMNS,
    SEASON_SCORE_COLUMNS,
    calibration_table,
    entrants_for_past_race,
    run_backtest,
)
from f1pred.config import DEFAULT_PARAMS
from f1pred.ratings.history import replay

P = DEFAULT_PARAMS


@pytest.fixture(scope="module")
def sample_history(driver_race):
    history, _ = replay(driver_race, P)
    return history


def test_entrants_use_pre_race_ratings(driver_race, sample_history):
    race_id = int(
        driver_race[(driver_race.season == 2024) & (~driver_race.is_sprint)].race_id.iloc[-1]
    )
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    ents = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)
    assert len(ents) == len(rows)
    by_id = {e.driver_id: e for e in ents}
    h = hist.set_index("driver_id")
    for d, e in by_id.items():
        assert e.strength == pytest.approx(
            h.loc[d, "driver_rating_pre"] + h.loc[d, "constructor_rating_pre"]
        )
        assert e.grid == int(rows.set_index("driver_id").loc[d, "grid"])
        assert e.p_dnf == 0.1
        assert e.low_confidence == (h.loc[d, "driver_races_pre"] < P.min_races_for_confidence)


def test_run_backtest_shapes_and_baselines(driver_race, sample_history):
    result = run_backtest(driver_race, sample_history, [2024], P, n_runs=500, seed=1)
    assert list(result.races.columns) == RACE_SCORE_COLUMNS
    assert list(result.seasons.columns) == SEASON_SCORE_COLUMNS
    assert len(result.races) == 24
    assert result.seasons.n_races.iloc[0] == 24
    # pandas Series == pytest.approx is always False; compare the ndarray (one 2024 race: 19 cars).
    assert result.races.uniform_logloss.to_numpy() == pytest.approx(math.log(20), abs=0.06)
    assert result.races.model_logloss.between(0, -math.log(1e-6)).all()
    assert result.calibration.columns.tolist() == [
        "bin_low",
        "bin_high",
        "predicted",
        "observed",
        "count",
    ]
    # one entry per driver-race row; 24 * 20 is off because one 2024 race had 19 cars
    n_rows = int(((driver_race.season == 2024) & (~driver_race.is_sprint)).sum())
    assert result.calibration["count"].sum() == n_rows == 479


def test_backtest_never_reads_future(driver_race, sample_history, monkeypatch):
    """Every history row used for a race must carry that race's own id (pre-race snapshot)."""
    import f1pred.backtest.run as run_mod

    seen = []
    original = run_mod.entrants_for_past_race

    def spy(rows, hist, dnf, params):
        seen.append((set(rows.race_id), set(hist.race_id)))
        return original(rows, hist, dnf, params)

    monkeypatch.setattr(run_mod, "entrants_for_past_race", spy)
    run_backtest(driver_race, sample_history, [2023], P, n_runs=50, seed=0)
    assert seen and all(r == h and len(r) == 1 for r, h in seen)


def test_calibration_table_bins():
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.95])
    y = np.array([0, 0, 1, 1, 1])
    cal = calibration_table(p, y, bins=10)
    assert cal["count"].sum() == 5
    row = cal[cal.bin_low == 0.1].iloc[0]
    assert row.predicted == pytest.approx(0.15) and row.observed == pytest.approx(0.5)
    assert cal[cal.bin_low == 0.9].iloc[0].observed == 1.0
