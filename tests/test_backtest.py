import math

import numpy as np
import pytest

from f1pred.backtest.run import (
    RACE_SCORE_COLUMNS,
    SEASON_SCORE_COLUMNS,
    StaleRatingsError,
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
    # Phase 1 arithmetic holds with track ratings off (Phase 4 adds a track-type shrink).
    params = P.replace(use_track=False)
    ents = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, params)
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

    def spy(rows, hist, dnf, params, profiles=None):
        seen.append((set(rows.race_id), set(hist.race_id)))
        return original(rows, hist, dnf, params, profiles)

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


def test_entrants_use_conditional_strengths(driver_race, sample_history):
    from f1pred.ratings.conditional import strengths

    brazil = driver_race[(driver_race.season == 2024) & (driver_race.circuit_id == "interlagos")]
    race_id = int(brazil.race_id.iloc[0])
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    ents = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)
    h = hist.set_index("driver_id")
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    r = h.loc["max_verstappen"]
    dry, wet = strengths(
        r.driver_rating_pre,
        r.constructor_rating_pre,
        r.driver_track_pre,
        r.constructor_track_pre,
        r.driver_track_n_pre,
        r.constructor_track_n_pre,
        r.driver_wet_pre,
        r.constructor_wet_pre,
        r.driver_wet_n_pre,
        r.constructor_wet_n_pre,
        P,
    )
    assert e.strength == pytest.approx(dry) and e.strength_wet == pytest.approx(wet)


def test_base_variant_matches_flags_off(driver_race, sample_history):
    from f1pred.backtest.run import ablation_backtest

    off = P.replace(use_weather=False, use_track=False)
    direct = run_backtest(driver_race, sample_history, [2024], off, n_runs=200, seed=1)
    abl = ablation_backtest(driver_race, sample_history, [2024], P, n_runs=200, seed=1)
    base = abl[(abl.variant == "base") & (abl.season == "2024")].iloc[0]
    assert base.model_logloss == pytest.approx(direct.seasons.model_logloss.iloc[0])
    # Phase 5 adds the `profile` variant.
    assert set(abl.variant) == {"base", "weather", "track", "full", "profile"}
    assert (abl.season == "all").sum() == 5


def test_entrants_carry_profiles(driver_race, sample_history):
    from f1pred.ratings.profile import profile_features, profile_lookup

    race_id = int(
        driver_race[(driver_race.season == 2024) & (~driver_race.is_sprint)].race_id.iloc[-1]
    )
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    lookup = profile_lookup(profile_features(driver_race, sample_history, P))
    ents = entrants_for_past_race(
        rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P, lookup
    )
    e = next(x for x in ents if x.driver_id == "max_verstappen")
    p = lookup[(race_id, "max_verstappen")]
    assert (e.aggression, e.risk, e.form) == (p.aggression, p.risk, p.form)
    plain = entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)
    assert all(x.aggression == 0 and x.risk == 0 and x.form == 0 for x in plain)


def test_stale_history_raises_clear_error(driver_race, sample_history):
    # Ratings built before the last race: the backtest must say so, not KeyError.
    last = int(driver_race[driver_race.season == 2024].race_id.max())
    hist = sample_history[sample_history.race_id != last]
    with pytest.raises(StaleRatingsError, match="ratings build"):
        run_backtest(driver_race, hist, [2024], P, n_runs=10)


def test_entrants_missing_driver_raises_clear_error(driver_race, sample_history):
    race_id = int(
        driver_race[(driver_race.season == 2024) & (~driver_race.is_sprint)].race_id.iloc[-1]
    )
    rows = driver_race[(driver_race.race_id == race_id) & (~driver_race.is_sprint)]
    hist = sample_history[(sample_history.race_id == race_id) & (~sample_history.is_sprint)]
    hist = hist[hist.driver_id != "max_verstappen"]
    with pytest.raises(StaleRatingsError, match="max_verstappen.*ratings build"):
        entrants_for_past_race(rows, hist, {(race_id, d): 0.1 for d in rows.driver_id}, P)


def test_ablation_has_profile_variant(driver_race, sample_history):
    from f1pred.backtest.run import ablation_backtest

    abl = ablation_backtest(driver_race, sample_history, [2024], P, n_runs=100, seed=1)
    assert set(abl.variant) == {"base", "weather", "track", "full", "profile"}
