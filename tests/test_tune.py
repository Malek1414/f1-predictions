import json

from f1pred.config import DEFAULT_PARAMS, load_params
from f1pred.tune import coordinate_descent, objective, write_tuned


def test_objective_is_finite_and_params_matter(driver_race):
    a = objective(driver_race, DEFAULT_PARAMS, [2024], n_runs=200, seed=0)
    b = objective(
        driver_race, DEFAULT_PARAMS.replace(sigma_driver=1000.0), [2024], n_runs=200, seed=0
    )
    assert 0 < a < 14 and 0 < b < 14
    assert a != b


def test_coordinate_descent_small_space(driver_race):
    space = {"grid_bonus": [0.0, 8.0], "sigma_team": [60.0]}
    best, score = coordinate_descent(
        driver_race,
        DEFAULT_PARAMS,
        [2024],
        space=space,
        passes=1,
        n_runs=100,
        seed=0,
        log=lambda *_: None,
    )
    assert best.grid_bonus in (0.0, 8.0)
    assert best.sigma_team == 60.0
    assert score > 0


def test_write_tuned_only_diffs(tmp_path):
    p = DEFAULT_PARAMS.replace(k_driver=30.0)
    path = tmp_path / "tuned.json"
    write_tuned(p, path, "test")
    data = json.loads(path.read_text())
    assert data == {"_note": "test", "k_driver": 30.0}
    assert load_params(path).k_driver == 30.0


def test_search_space_has_phase4_knobs():
    from f1pred.tune import SEARCH_SPACE

    assert {"shrink_wet", "shrink_track", "wet_noise_factor", "wet_dnf_factor"} <= set(SEARCH_SPACE)


def test_search_space_has_profile_scales():
    from f1pred.tune import SEARCH_SPACE

    assert {"aggression_scale", "risk_noise_scale", "risk_dnf_scale", "form_scale"} <= set(
        SEARCH_SPACE
    )
