"""Walk-forward backtest of the Bayesian pace model: one posterior per race, fit on its past.

This is the honest version of `run_backtest` for the pace model. Refitting for every race is
what the DGX Spark is for; on a laptop use `refit_every` to trade accuracy for time. Every
posterior is cached under `<cache_dir>/posteriors/wf-<race_id>.npz`, so an interrupted run
picks up where it stopped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1pred.backtest.run import BacktestResult, assemble_result, score_race
from f1pred.config import ModelParams
from f1pred.data.weather import circuit_wet_rate
from f1pred.pace.bridge import POSTERIOR_DIR, forecast_from_posterior
from f1pred.pace.design import build_design
from f1pred.pace.model import fit
from f1pred.pace.posterior import Posterior
from f1pred.predict import PredictionInputs
from f1pred.sim.race import Entrant

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardRace:
    """The part of a `RaceRef` the pace model reads, built straight from the table."""

    race_id: int
    season: int
    round: int
    circuit_id: str
    track_type: str
    date: pd.Timestamp


def race_reference(race_rows: pd.DataFrame) -> WalkForwardRace:
    first = race_rows.iloc[0]
    return WalkForwardRace(
        race_id=int(first.race_id),
        season=int(first.season),
        round=int(first["round"]),
        circuit_id=str(first.circuit_id),
        track_type=str(first.track_type),
        date=pd.Timestamp(first.date),
    )


def entrants_for_posterior(race_rows: pd.DataFrame) -> list[Entrant]:
    """Only the identity and grid slot matter: pace and DNF odds come from the posterior."""
    return [
        Entrant(
            driver_id=r.driver_id,
            constructor_id=r.constructor_id,
            strength=0.0,
            grid=int(r.grid),
            p_dnf=0.0,
        )
        for r in race_rows.itertuples(index=False)
    ]


def walkforward(
    table: pd.DataFrame,
    seasons: Iterable[int],
    params: ModelParams,
    fit_kwargs: Mapping[str, object] | None = None,
    cache_dir: Path | str = "data/cache",
    n_runs: int = 10_000,
    seed: int = 0,
    refit_every: int = 1,
    observed_rain: bool = False,
    progress: bool = False,
) -> BacktestResult:
    """Score every race of `seasons` from a posterior fit only on the races before it."""
    fit_kwargs = dict(fit_kwargs or {})
    posteriors = Path(cache_dir) / POSTERIOR_DIR
    posteriors.mkdir(parents=True, exist_ok=True)
    seasons = list(seasons)
    races = table[(~table["is_sprint"]) & (table["season"].isin(seasons))]
    race_ids = races.sort_values("date")["race_id"].unique()

    rows, all_p, all_won, all_p_podium, all_on_podium, win_vectors = [], [], [], [], [], []
    posterior: Posterior | None = None
    design = None
    for i, race_id in enumerate(race_ids):
        race_rows = races[races["race_id"] == race_id]
        ref = race_reference(race_rows)
        if posterior is None or i % refit_every == 0:
            design = build_design(table, before_date=ref.date, params=params)
            path = posteriors / f"wf-{int(race_id)}.npz"
            if path.exists():
                posterior = Posterior.load(path)
            else:
                posterior = fit(design, params, **fit_kwargs)
                posterior.diagnostics["cutoff"] = str(ref.date.date())
                posterior.save(path)
            if progress:
                log.info(
                    "%s %d: %d rows, %d divergences",
                    ref.circuit_id,
                    ref.season,
                    design.n_row,
                    posterior.diagnostics.get("divergences", 0),
                )

        if not params.use_weather:
            rain = 0.0
        elif observed_rain:
            rain = 1.0 if bool(race_rows.iloc[0]["is_wet"]) else 0.0
        else:
            rain = circuit_wet_rate(table, ref.circuit_id, ref.date)
        inputs = PredictionInputs(
            race=ref,
            entrants=entrants_for_posterior(race_rows),
            names={},
            use_grid=True,
            note=None,
            rain_probability=rain,
            rain_source="observed" if observed_rain else "historical",
        )
        forecast = forecast_from_posterior(
            posterior, design, inputs, params, n_runs=n_runs, seed=seed + i, keep_positions=True
        )
        row, extras = score_race(race_rows, forecast)
        rows.append(row)
        all_p.extend(extras["p_win"])
        all_won.extend(extras["won"])
        all_p_podium.extend(extras["p_podium"])
        all_on_podium.extend(extras["on_podium"])
        win_vectors.append(forecast.p_win)

    return assemble_result(rows, all_p, all_won, all_p_podium, all_on_podium, win_vectors)
