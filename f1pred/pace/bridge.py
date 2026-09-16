"""Posterior samples into the existing Monte Carlo. Spec 2.2, "from posterior to simulation"."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from f1pred.config import ModelParams
from f1pred.pace.design import PaceDesign
from f1pred.pace.model import predict_mu
from f1pred.pace.posterior import Posterior
from f1pred.sim.race import RaceForecast, simulate_race

POSTERIOR_DIR = "posteriors"
LATEST_FILE = "latest.npz"
MISSING_POSTERIOR = (
    "No posterior at {path}. Run `f1pred pace fit` (about ten minutes on this machine, "
    "or `make spark-fit` on the DGX Spark) before predicting with --model bayes."
)


CHECKED_KEYS = ("driver_ids", "driver_season_keys", "constructor_season_keys", "circuit_ids")


class StalePosteriorError(Exception):
    """The posterior was fit on a different design, so its parameter indices do not line up."""


def check_design(posterior: Posterior, design: PaceDesign) -> None:
    live = design.meta()
    for key in CHECKED_KEYS:
        if live[key] != posterior.design_meta.get(key):
            raise StalePosteriorError(
                f"The saved posterior does not match the current data ({key} differs). "
                "Re-run `f1pred pace fit` after `f1pred data update`."
            )


def posterior_path(cache_dir: Path, name: str = LATEST_FILE) -> Path:
    return Path(cache_dir) / POSTERIOR_DIR / name


def load_posterior(cache_dir: Path, name: str = LATEST_FILE) -> Posterior:
    path = posterior_path(cache_dir, name)
    if not path.exists():
        raise FileNotFoundError(MISSING_POSTERIOR.format(path=path))
    return Posterior.load(path)


def strengths_from_posterior(
    posterior: Posterior,
    design: PaceDesign,
    entrants,
    race_info,
    params: ModelParams,
    use_grid: bool = True,
    rain_probability: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """`(strength_samples (S, n) in rating points, p_dnf_samples (S, n))` for one race."""
    check_design(posterior, design)
    mu, p_dnf = predict_mu(
        posterior,
        design,
        entrants,
        race_info,
        grid=None if use_grid else {},
        is_wet_prob=rain_probability,
        seed=seed,
    )
    return mu * params.pace_scale, p_dnf


def forecast_from_posterior(
    posterior: Posterior,
    design: PaceDesign,
    inputs,
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int = 0,
    keep_positions: bool = False,
) -> RaceForecast:
    """Simulate one race straight from the posterior: run `i` uses posterior sample `i % S`."""
    strength, p_dnf = strengths_from_posterior(
        posterior,
        design,
        inputs.entrants,
        inputs.race,
        params,
        use_grid=inputs.use_grid,
        rain_probability=inputs.rain_probability,
        seed=seed,
    )
    return simulate_race(
        inputs.entrants,
        params,
        n_runs=n_runs,
        seed=seed,
        use_grid=inputs.use_grid,
        rain_probability=inputs.rain_probability,
        keep_positions=keep_positions,
        strength_samples=strength,
        p_dnf_samples=p_dnf,
    )
