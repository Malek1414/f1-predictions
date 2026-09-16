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
from f1pred.ratings.profile import profile_features, profile_lookup
from f1pred.sim.dnf import dnf_cache_for_races

SEARCH_SPACE: dict[str, list[float]] = {
    # Widened after the first tune pinned several knobs at the edge of the old lists.
    "k_driver": [24.0, 32.0, 48.0, 64.0, 96.0],
    "k_constructor": [32.0, 48.0, 64.0, 96.0, 128.0],
    "teammate_weight": [1.0, 2.0, 3.0],
    "regress_driver": [0.05, 0.1, 0.2],
    "regress_constructor": [0.1, 0.2, 0.3, 0.4],
    "sigma_team": [15.0, 25.0, 40.0, 60.0],
    "sigma_driver": [25.0, 40.0, 60.0, 80.0],
    # Phase 7a: the concave term is grid_bonus * n / grid ** grid_shape, so the scale is smaller.
    "grid_bonus": [2.0, 4.0, 6.0, 8.0, 12.0],
    "grid_shape": [0.3, 0.5, 0.75, 1.0, 1.5],
    "shrink_wet": [8.0, 16.0, 32.0],
    "shrink_track": [8.0, 16.0, 32.0],
    "wet_noise_factor": [1.0, 1.5],
    "wet_dnf_factor": [1.0, 1.5],
    "aggression_scale": [0.0, 5.0, 10.0, 20.0],
    "risk_noise_scale": [0.0, 0.25],
    "risk_dnf_scale": [0.0, 0.25, 0.5],
    "form_scale": [0.0, 10.0, 20.0, 40.0],
    "shrink_profile": [2.0, 5.0, 10.0],
}


def objective(
    table: pd.DataFrame,
    params: ModelParams,
    train_seasons: Iterable[int],
    n_runs: int = 2000,
    seed: int = 0,
    dnf_cache: dict | None = None,
    metric: str = "model_logloss",
) -> float:
    """Mean of `metric` (a race-score column, default winner log loss) over the training
    seasons for these params. Lower is better for every column the tuner accepts."""
    history, _ = replay(table, params)
    # The profile lookup depends on `history` (and so on K), so it is rebuilt per candidate.
    profiles = (
        profile_lookup(profile_features(table, history, params)) if params.use_profile else None
    )
    result = run_backtest(
        table,
        history,
        list(train_seasons),
        params,
        n_runs=n_runs,
        seed=seed,
        dnf_cache=dnf_cache,
        profiles=profiles,
    )
    return float(result.races[metric].mean())


def coordinate_descent(
    table: pd.DataFrame,
    base: ModelParams,
    train_seasons: Iterable[int],
    space: dict[str, list[float]] = SEARCH_SPACE,
    passes: int = 2,
    n_runs: int = 2000,
    seed: int = 0,
    log: Callable[..., None] = print,
    metric: str = "model_logloss",
) -> tuple[ModelParams, float]:
    """Try each candidate value of one parameter at a time, keep the best, repeat."""
    train = list(train_seasons)
    race_ids = table[(~table.is_sprint) & (table.season.isin(train))].race_id.unique()
    dnf_cache = dnf_cache_for_races(table, race_ids, base)
    best = base
    best_score = objective(table, best, train, n_runs, seed, dnf_cache, metric)
    log(f"start: {best_score:.4f}")
    for p in range(passes):
        for name, candidates in space.items():
            for value in candidates:
                if getattr(best, name) == value:
                    continue
                trial = best.replace(**{name: value})
                score = objective(table, trial, train, n_runs, seed, dnf_cache, metric)
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
