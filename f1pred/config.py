"""Model parameters and default paths."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_OUTPUT_DIR = Path("outputs")
TUNED_PARAMS_PATH = Path(__file__).parent / "tuned.json"


@dataclass(frozen=True)
class ModelParams:
    """All tunable knobs. See spec sections 5, 6 and 8."""

    # Elo
    initial_rating: float = 1500.0
    k_driver: float = 24.0
    k_constructor: float = 32.0
    teammate_weight: float = 2.0
    sprint_weight: float = 0.5
    regress_driver: float = 0.1
    regress_constructor: float = 0.4
    regress_constructor_regulation: float = 0.7
    regulation_seasons: tuple[int, ...] = (2014, 2017, 2022, 2026)
    start_season: int = 2010
    min_races_for_confidence: int = 5
    # Simulation
    sigma_team: float = 60.0
    sigma_driver: float = 80.0
    grid_bonus: float = 8.0
    grid_shape: float = 1.0  # Phase 7a: grid_term = grid_bonus * n / grid ** grid_shape
    # DNF
    dnf_window: int = 20
    shrink_dnf: float = 10.0
    circuit_min_rows: int = 40
    circuit_factor_min: float = 0.5
    circuit_factor_max: float = 2.0
    # Weather and track (Phase 4)
    use_weather: bool = True
    use_track: bool = True
    wet_threshold_mm: float = 0.5
    race_window_hours: int = 3
    shrink_wet: float = 8.0
    shrink_track: float = 8.0
    wet_noise_factor: float = 1.5
    wet_dnf_factor: float = 1.5
    # Driver profile (Phase 5)
    use_profile: bool = True
    profile_window: int = 20
    form_window: int = 6
    shrink_profile: float = 10.0
    aggression_scale: float = 0.0
    risk_noise_scale: float = 0.0
    risk_dnf_scale: float = 0.0
    form_scale: float = 0.0
    # Bayesian pace model (Phase 7b). `pace_chains` also sets the CPU host device count, so
    # four chains really do run in parallel; these defaults are one ~10 minute fit on an M2.
    pace_num_warmup: int = 1000
    pace_num_samples: int = 1000
    pace_chains: int = 4
    pace_device: str = "cpu"
    # Seasons of half-life for the likelihood's exponential age discount. Within one season the
    # regulations, the car, the teammate and the calendar are held fixed, so a within-season
    # comparison is far less confounded than a cross-season one; older seasons still carry
    # information but should not count the same as what is happening now. `inf` disables it.
    season_half_life: float = 2.0
    # "elo" (phases 1 to 6) or "bayes" (the posterior pace model). One pace unit is worth
    # `pace_scale` rating points, so the existing reports keep their Elo semantics.
    model: str = "elo"
    pace_scale: float = 400.0 / math.log(10.0)

    def replace(self, **changes: object) -> ModelParams:
        return dataclasses.replace(self, **changes)


DEFAULT_PARAMS = ModelParams()


def load_params(path: Path | None = None) -> ModelParams:
    """Defaults overridden by a JSON file of field values, if the file exists."""
    path = TUNED_PARAMS_PATH if path is None else path
    if not path.exists():
        return DEFAULT_PARAMS
    overrides = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}
    if "regulation_seasons" in overrides:
        overrides["regulation_seasons"] = tuple(overrides["regulation_seasons"])
    return DEFAULT_PARAMS.replace(**overrides)
