from f1pred.backtest.rolling import ROLLING_COLUMNS, rolling_origin
from f1pred.config import DEFAULT_PARAMS


def test_rolling_origin_never_scores_a_training_season(driver_race):
    seen = []

    def tune_fn(table, train_seasons):
        seen.append(tuple(train_seasons))
        return DEFAULT_PARAMS

    out = rolling_origin(driver_race, DEFAULT_PARAMS, [2024], tune_fn, n_runs=50, seed=0)
    assert seen == [(2023,)]
    assert list(out.season) == [2024]
    assert {"lo", "hi"} <= set(out.columns)
    assert list(out.columns) == ROLLING_COLUMNS
    row = out.iloc[0]
    assert row.n_races == 24 and row.train_seasons == "2023"
    assert row.lo <= row.model_logloss <= row.hi
    assert row.rps_lo <= row.model_rps <= row.rps_hi


def test_rolling_origin_without_tuner_uses_given_params(driver_race):
    out = rolling_origin(driver_race, DEFAULT_PARAMS, [2024], None, n_runs=50, seed=0, R=100)
    assert len(out) == 1 and out.season.iloc[0] == 2024
    assert out.model_logloss.iloc[0] > 0
