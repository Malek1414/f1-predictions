"""Replay past seasons, predicting each race from pre-race data only. Spec section 8."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.backtest.scoring import (
    ece,
    mean_rps,
    podium_brier,
    pole_baseline,
    position_spearman,
    set_log_loss,
    sharpness,
    top3_set_log_loss,
    uniform_baseline,
    winner_log_loss,
)
from f1pred.config import ModelParams
from f1pred.data.weather import circuit_wet_rate
from f1pred.ratings.conditional import strengths
from f1pred.ratings.profile import Profile, profile_features, profile_lookup
from f1pred.sim.dnf import dnf_cache_for_races
from f1pred.sim.race import Entrant, simulate_race

RACE_SCORE_COLUMNS = [
    "season",
    "round",
    "race_id",
    "race_name",
    "winner",
    "winner_p",
    "model_logloss",
    "pole_logloss",
    "uniform_logloss",
    "model_brier",
    "pole_brier",
    "uniform_brier",
    "model_spearman",
    "pole_spearman",
    # Phase 7a distribution scores (docs/metrics.md).
    "model_rps",
    "pole_rps",
    "uniform_rps",
    "model_top3",
    "pole_top3",
    "uniform_top3",
]
SEASON_SCORE_COLUMNS = ["season", "n_races"] + RACE_SCORE_COLUMNS[6:]


@dataclass
class BacktestResult:
    races: pd.DataFrame
    seasons: pd.DataFrame
    calibration: pd.DataFrame
    # Over every driver-race in the run (Phase 7a).
    ece_win: float = float("nan")
    ece_podium: float = float("nan")
    sharpness: float = float("nan")


class StaleRatingsError(Exception):
    """The ratings history does not cover a race in the driver-race table."""


REBUILD_HINT = "Run `f1pred ratings build` to bring the ratings up to date with the data."


def entrants_for_past_race(
    race_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
    dnf_lookup: Mapping[tuple[int, str], float],
    params: ModelParams,
    profiles: Mapping[tuple[int, str], Profile] | None = None,
) -> list[Entrant]:
    """Entrants from pre-race rows; a missing profile (spec 6.6) is zero."""
    hist = history_rows.set_index("driver_id")
    profiles = {} if profiles is None else profiles
    out = []
    for r in race_rows.itertuples(index=False):
        if r.driver_id not in hist.index:
            raise StaleRatingsError(
                f"No pre-race rating for {r.driver_id} in {r.race_name} {int(r.season)} "
                f"(race_id {int(r.race_id)}). {REBUILD_HINT}"
            )
        h = hist.loc[r.driver_id]
        profile = profiles.get((int(r.race_id), r.driver_id), Profile.zero())
        dry, wet = strengths(
            float(h.driver_rating_pre),
            float(h.constructor_rating_pre),
            float(h.driver_track_pre),
            float(h.constructor_track_pre),
            int(h.driver_track_n_pre),
            int(h.constructor_track_n_pre),
            float(h.driver_wet_pre),
            float(h.constructor_wet_pre),
            int(h.driver_wet_n_pre),
            int(h.constructor_wet_n_pre),
            params,
        )
        out.append(
            Entrant(
                driver_id=r.driver_id,
                constructor_id=r.constructor_id,
                strength=dry,
                grid=int(r.grid),
                p_dnf=float(dnf_lookup[(int(r.race_id), r.driver_id)]),
                low_confidence=bool(h.driver_races_pre < params.min_races_for_confidence),
                strength_wet=wet,
                aggression=profile.aggression,
                risk=profile.risk,
                form=profile.form,
            )
        )
    return out


def calibration_table(p: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed": float(outcome[mask].mean()) if mask.any() else np.nan,
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_backtest(
    table: pd.DataFrame,
    history: pd.DataFrame,
    seasons: Iterable[int],
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int = 0,
    dnf_cache: dict[tuple[int, str], float] | None = None,
    profiles: Mapping[tuple[int, str], Profile] | None = None,
    observed_rain: bool = False,
) -> BacktestResult:
    """Score each race from pre-race information. Rain defaults to the circuit's historical
    wet rate as known before the race; `observed_rain=True` uses the race-day rainfall (1.0 or
    0.0), which is an upper bound on what a forecast could deliver, not a fair score."""
    seasons = list(seasons)
    races = table[(~table["is_sprint"]) & (table["season"].isin(seasons))]
    race_ids = races.sort_values("date")["race_id"].unique()
    if dnf_cache is None:
        dnf_cache = dnf_cache_for_races(table, race_ids, params)
    if profiles is None and params.use_profile:
        profiles = profile_lookup(profile_features(table, history, params))
    race_history = history[~history["is_sprint"]]

    rows, all_p, all_won, all_p_podium, all_on_podium, win_vectors = [], [], [], [], [], []
    for i, race_id in enumerate(race_ids):
        race_rows = races[races["race_id"] == race_id]
        hist_rows = race_history[race_history["race_id"] == race_id]
        if hist_rows.empty:
            first = race_rows.iloc[0]
            raise StaleRatingsError(
                f"No ratings history for {first.race_name} {int(first.season)} "
                f"(race_id {int(race_id)}). {REBUILD_HINT}"
            )
        entrants = entrants_for_past_race(race_rows, hist_rows, dnf_cache, params, profiles)
        first = race_rows.iloc[0]
        if not params.use_weather:
            rain = 0.0
        elif observed_rain:
            rain = 1.0 if bool(first["is_wet"]) else 0.0
        else:
            rain = circuit_wet_rate(table, first.circuit_id, first.date)
        forecast = simulate_race(
            entrants,
            params,
            n_runs=n_runs,
            seed=seed + i,
            rain_probability=rain,
            keep_positions=True,
        )

        classified = race_rows[race_rows["position"].notna()]
        actual = dict(zip(classified["driver_id"], classified["position"].astype(int), strict=True))
        winner = next(d for d, pos in actual.items() if pos == 1)
        podium = {d for d, pos in actual.items() if pos <= 3}
        grid = dict(zip(race_rows["driver_id"], race_rows["grid"].astype(int), strict=True))

        ids = forecast.driver_ids
        p_win = dict(zip(ids, forecast.p_win, strict=True))
        p_podium = dict(zip(ids, forecast.p_podium, strict=True))
        expected = dict(zip(ids, forecast.expected_position, strict=True))
        pole = pole_baseline(grid)
        uni = uniform_baseline(ids)

        rows.append(
            {
                "season": int(first.season),
                "round": int(first["round"]),
                "race_id": int(race_id),
                "race_name": first.race_name,
                "winner": winner,
                "winner_p": p_win[winner],
                "model_logloss": winner_log_loss(p_win, winner),
                "pole_logloss": winner_log_loss(pole.p_win, winner),
                "uniform_logloss": winner_log_loss(uni.p_win, winner),
                "model_brier": podium_brier(p_podium, podium),
                "pole_brier": podium_brier(pole.p_podium, podium),
                "uniform_brier": podium_brier(uni.p_podium, podium),
                "model_spearman": position_spearman(expected, actual),
                "pole_spearman": position_spearman(pole.expected_position, actual),
                "model_rps": mean_rps(forecast.position_matrix, ids, actual),
                "pole_rps": mean_rps(pole.position_matrix(ids), ids, actual),
                "uniform_rps": mean_rps(uni.position_matrix(ids), ids, actual),
                "model_top3": top3_set_log_loss(forecast.positions, ids, podium),
                "pole_top3": set_log_loss(pole.top3_set_prob(podium)),
                "uniform_top3": set_log_loss(uni.top3_set_prob(podium)),
            }
        )
        all_p.extend(forecast.p_win.tolist())
        all_won.extend([1.0 if d == winner else 0.0 for d in ids])
        all_p_podium.extend(forecast.p_podium.tolist())
        all_on_podium.extend([1.0 if d in podium else 0.0 for d in ids])
        win_vectors.append(forecast.p_win)

    race_df = pd.DataFrame(rows, columns=RACE_SCORE_COLUMNS)
    season_df = (
        race_df.groupby("season")
        .agg(n_races=("race_id", "count"), **{c: (c, "mean") for c in RACE_SCORE_COLUMNS[6:]})
        .reset_index()[SEASON_SCORE_COLUMNS]
    )
    calibration = calibration_table(np.array(all_p), np.array(all_won))
    return BacktestResult(
        race_df,
        season_df,
        calibration,
        ece_win=ece(np.array(all_p), np.array(all_won)),
        ece_podium=ece(np.array(all_p_podium), np.array(all_on_podium)),
        sharpness=sharpness(win_vectors),
    )


VARIANTS = {
    "base": {"use_weather": False, "use_track": False, "use_profile": False},
    "weather": {"use_weather": True, "use_track": False, "use_profile": False},
    "track": {"use_weather": False, "use_track": True, "use_profile": False},
    "full": {"use_weather": True, "use_track": True, "use_profile": False},
    "profile": {"use_weather": True, "use_track": True, "use_profile": True},
}
ABLATION_COLUMNS = [
    "variant",
    "season",
    "n_races",
    "model_logloss",
    "model_brier",
    "model_spearman",
    "rain_mode",
]


def ablation_backtest(
    table: pd.DataFrame,
    history: pd.DataFrame,
    seasons: Iterable[int],
    params: ModelParams,
    n_runs: int = 10_000,
    seed: int = 0,
    dnf_cache: dict[tuple[int, str], float] | None = None,
    observed_rain: bool = False,
) -> pd.DataFrame:
    """Spec 8.1: scores with and without weather, track type and the driver profile, per season
    plus an `all` row. `full` is weather and track with the profile off; `profile` is everything.
    `rain_mode` records whether the weather variants saw the pre-race historical rate
    (`historical`, the default) or the race-day rainfall (`observed`, an upper bound).
    """
    seasons = list(seasons)
    race_ids = table[(~table["is_sprint"]) & (table["season"].isin(seasons))]["race_id"].unique()
    if dnf_cache is None:
        dnf_cache = dnf_cache_for_races(table, race_ids, params)
    rain_mode = "observed" if observed_rain else "historical"
    rows = []
    for name, flags in VARIANTS.items():
        result = run_backtest(
            table,
            history,
            seasons,
            params.replace(**flags),
            n_runs,
            seed,
            dnf_cache,
            observed_rain=observed_rain,
        )
        for r in result.seasons.itertuples(index=False):
            rows.append(
                {
                    "variant": name,
                    "season": str(int(r.season)),
                    "n_races": int(r.n_races),
                    "model_logloss": r.model_logloss,
                    "model_brier": r.model_brier,
                    "model_spearman": r.model_spearman,
                    "rain_mode": rain_mode,
                }
            )
        rows.append(
            {
                "variant": name,
                "season": "all",
                "n_races": len(result.races),
                "model_logloss": result.races.model_logloss.mean(),
                "model_brier": result.races.model_brier.mean(),
                "model_spearman": result.races.model_spearman.mean(),
                "rain_mode": rain_mode,
            }
        )
    return pd.DataFrame(rows, columns=ABLATION_COLUMNS)
