"""Coordinate-descent search over ModelParams, scored by winner log loss. Spec 8.1."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from f1pred.backtest.run import run_backtest
from f1pred.config import DEFAULT_PARAMS, ModelParams
from f1pred.ratings.history import replay
from f1pred.sim.dnf import dnf_cache_for_races

SEARCH_SPACE: dict[str, list[float]] = {
    "k_driver": [16.0, 24.0, 32.0],
    "k_constructor": [24.0, 32.0, 48.0],
    "sigma_team": [40.0, 60.0, 90.0],
    "sigma_driver": [60.0, 80.0, 110.0],
    "grid_bonus": [4.0, 8.0, 12.0],
    "regress_constructor": [0.3, 0.4, 0.6],
    # Phase 4 (spec 8.1)
    "shrink_wet": [4.0, 8.0, 16.0],
    "shrink_track": [4.0, 8.0, 16.0],
    "wet_noise_factor": [1.0, 1.5, 2.0],
    "wet_dnf_factor": [1.0, 1.5, 2.0],
}


def objective(
    table: pd.DataFrame,
    params: ModelParams,
    train_seasons: Iterable[int],
    n_runs: int = 2000,
    seed: int = 0,
    dnf_cache: dict | None = None,
) -> float:
    """Mean winner log loss over the training seasons for these params."""
    history, _ = replay(table, params)
    result = run_backtest(
        table, history, list(train_seasons), params, n_runs=n_runs, seed=seed, dnf_cache=dnf_cache
    )
    return float(result.races["model_logloss"].mean())


def coordinate_descent(
    table: pd.DataFrame,
    base: ModelParams,
    train_seasons: Iterable[int],
    space: dict[str, list[float]] = SEARCH_SPACE,
    passes: int = 2,
    n_runs: int = 2000,
    seed: int = 0,
    log: Callable[..., None] = print,
) -> tuple[ModelParams, float]:
    """Try each candidate value of one parameter at a time, keep the best, repeat."""
    train = list(train_seasons)
    race_ids = table[(~table.is_sprint) & (table.season.isin(train))].race_id.unique()
    dnf_cache = dnf_cache_for_races(table, race_ids, base)
    best = base
    best_score = objective(table, best, train, n_runs, seed, dnf_cache)
    log(f"start: {best_score:.4f}")
    for p in range(passes):
        for name, candidates in space.items():
            for value in candidates:
                if getattr(best, name) == value:
                    continue
                trial = best.replace(**{name: value})
                score = objective(table, trial, train, n_runs, seed, dnf_cache)
                log(f"pass {p + 1} {name}={value}: {score:.4f}")
                if score < best_score:
                    best, best_score = trial, score
                    log(f"  -> new best {best_score:.4f}")
    return best, best_score


def write_tuned(params: ModelParams, path: Path, note: str) -> None:
    diffs = {
        f.name: getattr(params, f.name)
        for f in dataclasses.fields(ModelParams)
        if getattr(params, f.name) != getattr(DEFAULT_PARAMS, f.name)
    }
    Path(path).write_text(json.dumps({"_note": note, **diffs}, indent=2))
