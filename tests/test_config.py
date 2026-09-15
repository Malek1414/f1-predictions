import json

from f1pred.config import DEFAULT_PARAMS, ModelParams, load_params


def test_defaults_match_spec():
    p = DEFAULT_PARAMS
    assert p.initial_rating == 1500.0
    assert p.start_season == 2010
    assert p.regulation_seasons == (2014, 2017, 2022, 2026)
    assert p.teammate_weight == 2.0
    assert p.sprint_weight == 0.5


def test_load_params_overrides_from_json(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"k_driver": 40.0, "sigma_team": 70.0}))
    p = load_params(path)
    assert p.k_driver == 40.0
    assert p.sigma_team == 70.0
    assert p.k_constructor == DEFAULT_PARAMS.k_constructor


def test_load_params_missing_file_returns_defaults(tmp_path):
    assert load_params(tmp_path / "nope.json") == DEFAULT_PARAMS


def test_replace_returns_new_params():
    p = DEFAULT_PARAMS.replace(k_driver=1.0)
    assert p.k_driver == 1.0
    assert isinstance(p, ModelParams)


def test_load_params_ignores_underscore_keys(tmp_path):
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({"_note": "x", "k_driver": 30.0}))
    assert load_params(path).k_driver == 30.0
