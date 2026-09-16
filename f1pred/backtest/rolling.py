"""Rolling-origin evaluation: score each fold season with parameters chosen on the seasons
before it, so the reported number is never one the tuner saw (docs/metrics.md)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from f1pred.backtest.bootstrap import bootstrap_intervals
from f1pred.backtest.run import SEASON_SCORE_COLUMNS, run_backtest
from f1pred.config import ModelParams
from f1pred.ratings.history import replay

TuneFn = Callable[[pd.DataFrame, list[int]], ModelParams]

INTERVAL_METRICS = ["model_logloss", "pole_logloss", "model_rps", "pole_rps"]
ROLLING_COLUMNS = (
    ["train_seasons"]
    + SEASON_SCORE_COLUMNS
    + ["lo", "hi", "pole_lo", "pole_hi", "rps_lo", "rps_hi", "pole_rps_lo", "pole_rps_hi"]
    + ["ece_win", "ece_podium", "sharpness"]
)


def rolling_origin(
    table: pd.DataFrame,
    params: ModelParams,
    folds: list[int],
    tune_fn: TuneFn | None = None,
    n_runs: int = 10_000,
    seed: int = 0,
    R: int = 1000,
) -> pd.DataFrame:
    """One row per fold season. Train seasons are those present in `table` from
    `params.start_season` to `fold - 1`; `tune_fn(table, train_seasons)` returns the params for
    the fold (or `params` when no tuner is given). Ratings are replayed once per distinct
    parameter set. `lo/hi` are the 90% bootstrap interval of the winner log loss, `rps_lo/hi` of
    the RPS, and the `pole_*` pairs the same for the pole baseline."""
    seasons_present = sorted(int(s) for s in table["season"].unique())
    histories: dict[ModelParams, pd.DataFrame] = {}
    rows = []
    for fold in folds:
        train = [s for s in seasons_present if params.start_season <= s < fold]
        fold_params = tune_fn(table, train) if tune_fn is not None else params
        if fold_params not in histories:
            histories[fold_params], _ = replay(table, fold_params)
        result = run_backtest(
            table, histories[fold_params], [fold], fold_params, n_runs=n_runs, seed=seed
        )
        row = {c: result.seasons.iloc[0][c] for c in SEASON_SCORE_COLUMNS}
        row["season"] = int(row["season"])
        row["n_races"] = int(row["n_races"])
        row["train_seasons"] = f"{train[0]}-{train[-1]}" if len(train) > 1 else str(train[0])
        ci = bootstrap_intervals(result.races, INTERVAL_METRICS, R=R, seed=seed).set_index("metric")
        row["lo"], row["hi"] = ci.loc["model_logloss", ["lo", "hi"]]
        row["pole_lo"], row["pole_hi"] = ci.loc["pole_logloss", ["lo", "hi"]]
        row["rps_lo"], row["rps_hi"] = ci.loc["model_rps", ["lo", "hi"]]
        row["pole_rps_lo"], row["pole_rps_hi"] = ci.loc["pole_rps", ["lo", "hi"]]
        row["ece_win"] = result.ece_win
        row["ece_podium"] = result.ece_podium
        row["sharpness"] = result.sharpness
        rows.append(row)
    return pd.DataFrame(rows, columns=ROLLING_COLUMNS)
