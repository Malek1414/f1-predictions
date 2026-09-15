"""Shrink a conditional (wet / track-type) rating toward the overall rating. Spec 5.5."""

from __future__ import annotations

from f1pred.config import ModelParams


def effective_rating(overall: float, conditional: float, n: int, shrink: float) -> float:
    return overall + (conditional - overall) * n / (n + shrink)


def strengths(
    driver_overall: float,
    constructor_overall: float,
    driver_track: float,
    constructor_track: float,
    n_driver_track: int,
    n_constructor_track: int,
    driver_wet: float,
    constructor_wet: float,
    n_driver_wet: int,
    n_constructor_wet: int,
    params: ModelParams,
) -> tuple[float, float]:
    """(dry strength, wet strength) for one entrant on one circuit."""
    if params.use_track:
        dry = effective_rating(
            driver_overall, driver_track, n_driver_track, params.shrink_track
        ) + effective_rating(
            constructor_overall, constructor_track, n_constructor_track, params.shrink_track
        )
    else:
        dry = driver_overall + constructor_overall
    if not params.use_weather:
        return dry, dry
    wet = (
        dry
        + effective_rating(driver_overall, driver_wet, n_driver_wet, params.shrink_wet)
        - driver_overall
        + effective_rating(
            constructor_overall, constructor_wet, n_constructor_wet, params.shrink_wet
        )
        - constructor_overall
    )
    return dry, wet
