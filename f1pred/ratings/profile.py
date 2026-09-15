"""Aggression, risk and form per driver, computed leak-free before each race. Spec 6.6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.config import ModelParams

PROFILE_COLUMNS = ["race_id", "driver_id", "aggression", "risk", "form", "profile_n"]
FIELD_ACC_FLOOR = 0.01


@dataclass(frozen=True)
class Profile:
    aggression: float
    risk: float
    form: float
    n: int

    @classmethod
    def zero(cls) -> Profile:
        return cls(0.0, 0.0, 0.0, 0)


def _raw_signals(table: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    races = table[~table["is_sprint"]].copy()
    hist = history[~history["is_sprint"]][
        ["race_id", "driver_id", "driver_rating_pre", "constructor_rating_pre"]
    ]
    df = races.merge(hist, on=["race_id", "driver_id"], how="left")
    df["strength_pre"] = df["driver_rating_pre"] + df["constructor_rating_pre"]
    df["expected_position"] = df.groupby("race_id")["strength_pre"].rank(
        ascending=False, method="average"
    )
    position = df["position"].astype(float)
    lap1 = df["lap1_position"].astype(float) if "lap1_position" in df.columns else np.nan
    gain_lap1 = df["grid"] - lap1
    gain_race = df["grid"] - position
    df["gain"] = pd.concat([gain_lap1, gain_race], axis=1).mean(axis=1, skipna=True)
    df["accident"] = (df["dnf_kind"] == "accident").astype(float)
    df["form_raw"] = df["expected_position"] - position
    return df.sort_values(["date", "race_id", "driver_id"], kind="stable").reset_index(drop=True)


def _rolling(df: pd.DataFrame, params: ModelParams, shift: int) -> pd.DataFrame:
    """Per-driver and field rolling statistics. shift=1 excludes the current row."""
    g = df.groupby("driver_id", sort=False)
    w, fw = params.profile_window, params.form_window

    def roll(col: str, window: int, func: str) -> pd.Series:
        return g[col].transform(
            lambda s: getattr(s.rolling(window, min_periods=1), func)().shift(shift)
        )

    out = pd.DataFrame(index=df.index)
    out["gain_mean"] = roll("gain", w, "mean")
    out["gain_n"] = roll("gain", w, "count").fillna(0)
    out["acc_rate"] = roll("accident", w, "mean")
    out["acc_n"] = roll("accident", w, "count").fillna(0)
    out["form_mean"] = roll("form_raw", fw, "mean")
    out["form_n"] = roll("form_raw", fw, "count").fillna(0)
    field_window = w * 20
    # The field baseline is taken once per race: at the first row when the current race is
    # excluded (so no row of the race leaks into it) and at the last row when it is included.
    by_race = df.groupby("race_id", sort=False)
    edge = "first" if shift else "last"
    field_gain = df["gain"].rolling(field_window, min_periods=1).mean().shift(shift)
    field_acc = df["accident"].rolling(field_window, min_periods=1).mean().shift(shift)
    out["field_gain"] = field_gain.groupby(by_race.ngroup()).transform(edge)
    out["field_acc"] = field_acc.groupby(by_race.ngroup()).transform(edge)
    return out


def _combine(stats: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    s = params.shrink_profile
    gain_w = stats["gain_n"] / (stats["gain_n"] + s)
    acc_w = stats["acc_n"] / (stats["acc_n"] + s)
    form_w = stats["form_n"] / (stats["form_n"] + s)
    out = pd.DataFrame(index=stats.index)
    out["aggression"] = ((stats["gain_mean"] - stats["field_gain"]) * gain_w).fillna(0.0)
    # Same as acc_rate / field_acc - 1 whenever the field rate is at or above the floor, but 0
    # (no relative signal) rather than -1 when nobody in the window has crashed.
    field_acc = stats["field_acc"].clip(lower=FIELD_ACC_FLOOR)
    out["risk"] = (((stats["acc_rate"] - stats["field_acc"]) / field_acc) * acc_w).fillna(0.0)
    out["form"] = (stats["form_mean"] * form_w).fillna(0.0)
    out["profile_n"] = stats["gain_n"].astype(int)
    return out


def profile_features(
    table: pd.DataFrame, history: pd.DataFrame, params: ModelParams
) -> pd.DataFrame:
    df = _raw_signals(table, history)
    if df.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    stats = _rolling(df, params, shift=1)
    # A driver's very first race: the field baseline is defined but their own stats are NaN -> 0.
    out = _combine(stats, params)
    out.insert(0, "driver_id", df["driver_id"].to_numpy())
    out.insert(0, "race_id", df["race_id"].astype(int).to_numpy())
    return out[PROFILE_COLUMNS].reset_index(drop=True)


def latest_profiles(
    table: pd.DataFrame, history: pd.DataFrame, params: ModelParams
) -> dict[str, Profile]:
    df = _raw_signals(table, history)
    if df.empty:
        return {}
    out = _combine(_rolling(df, params, shift=0), params)
    out["driver_id"] = df["driver_id"].to_numpy()
    last = out.groupby("driver_id").tail(1)
    return {
        r.driver_id: Profile(float(r.aggression), float(r.risk), float(r.form), int(r.profile_n))
        for r in last.itertuples(index=False)
    }


def profile_lookup(features: pd.DataFrame) -> dict[tuple[int, str], Profile]:
    return {
        (int(r.race_id), r.driver_id): Profile(
            float(r.aggression), float(r.risk), float(r.form), int(r.profile_n)
        )
        for r in features.itertuples(index=False)
    }
