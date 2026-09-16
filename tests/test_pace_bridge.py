import numpy as np
import pytest

from f1pred.config import DEFAULT_PARAMS
from f1pred.pace.bridge import forecast_from_posterior
from f1pred.pace.model import fit
from f1pred.predict import PredictionInputs
from f1pred.sim.race import Entrant, simulate_positions, simulate_race
from tests.synthetic_pace import FakeRace, last_race_entrants, synthetic_design, synthetic_table

P = DEFAULT_PARAMS
TINY = {"num_warmup": 80, "num_samples": 80, "chains": 1}


def _entrants(n):
    return [Entrant(f"d{i}", f"t{i}", 0.0, i + 1, 0.0) for i in range(n)]


def test_strength_samples_drive_each_run_exactly():
    # Gaps of 5000 rating points dwarf the pace noise, so each run's order is forced.
    samples = np.array([[0.0, 5000.0, 10_000.0], [10_000.0, 5000.0, 0.0]])
    positions, used_grid = simulate_positions(
        _entrants(3),
        P,
        n_runs=2,
        rng=np.random.default_rng(0),
        strength_samples=samples,
        p_dnf_samples=np.zeros((2, 3)),
    )
    np.testing.assert_array_equal(positions, np.array([[3, 2, 1], [1, 2, 3]]))
    assert used_grid is True  # the grid was known; its effect lives inside the posterior pace


def test_runs_cycle_through_the_posterior_samples():
    samples = np.array([[0.0, 5000.0, 10_000.0], [10_000.0, 5000.0, 0.0]])
    positions, _ = simulate_positions(
        _entrants(3),
        P,
        n_runs=5,
        rng=np.random.default_rng(0),
        strength_samples=samples,
        p_dnf_samples=np.zeros((2, 3)),
    )
    np.testing.assert_array_equal(positions[0], positions[2])
    np.testing.assert_array_equal(positions[1], positions[3])
    np.testing.assert_array_equal(positions[0], positions[4])


def test_dnf_samples_are_used_per_run():
    samples = np.zeros((2, 2))
    p_dnf = np.array([[1.0, 0.0], [0.0, 1.0]])
    positions, _ = simulate_positions(
        _entrants(2),
        P,
        n_runs=2,
        rng=np.random.default_rng(3),
        strength_samples=samples,
        p_dnf_samples=p_dnf,
    )
    assert positions[0, 0] == 2 and positions[1, 1] == 2


def test_posterior_path_ignores_team_noise_and_the_grid_term():
    # Two entrants in the same team with identical posterior paces: only driver noise separates
    # them, so wins split evenly however big sigma_team and grid_bonus are.
    entrants = [Entrant("a", "same", 0.0, 1, 0.0), Entrant("b", "same", 0.0, 20, 0.0)]
    params = P.replace(sigma_team=5000.0, grid_bonus=500.0)
    forecast = simulate_race(
        entrants,
        params,
        n_runs=4000,
        seed=0,
        strength_samples=np.zeros((1, 2)),
        p_dnf_samples=np.zeros((1, 2)),
    )
    assert forecast.p_win[0] == pytest.approx(0.5, abs=0.05)


def test_simulate_race_without_samples_is_unchanged():
    entrants = _entrants(5)
    a = simulate_race(entrants, P, n_runs=200, seed=7)
    b = simulate_race(entrants, P, n_runs=200, seed=7, strength_samples=None)
    np.testing.assert_allclose(a.p_win, b.p_win)


@pytest.fixture(scope="module")
def planted():
    table = synthetic_table(seed=0)
    design = synthetic_design(seed=0)
    return table, design, fit(design, P, seed=0, **TINY)


def test_forecast_from_posterior_favours_the_planted_best_driver(planted):
    table, design, post = planted
    entrants = last_race_entrants(table)
    inputs = PredictionInputs(
        race=FakeRace(2023),
        entrants=entrants,
        names={e.driver_id: e.driver_id for e in entrants},
        use_grid=True,
        note=None,
        rain_probability=0.0,
        rain_source="disabled",
    )
    forecast = forecast_from_posterior(post, design, inputs, P, n_runs=2000, seed=0)
    best = forecast.driver_ids[int(np.argmax(forecast.p_win))]
    assert best == "ace"  # car A plus the planted +0.8 skill
    assert forecast.p_win.sum() == pytest.approx(1.0)
    assert forecast.prob("ace", "win") > forecast.prob("vet", "win")
    assert forecast.prob("co", "win") < forecast.prob("vet", "win")


def test_forecast_from_posterior_is_wider_than_a_point_estimate(planted):
    """Drawing a different posterior sample per run must spread the win odds, not sharpen them."""
    table, design, post = planted
    entrants = last_race_entrants(table)
    inputs = PredictionInputs(
        race=FakeRace(2023),
        entrants=entrants,
        names={},
        use_grid=True,
        note=None,
        rain_probability=0.0,
        rain_source="disabled",
    )
    spread = forecast_from_posterior(post, design, inputs, P, n_runs=2000, seed=0)
    from f1pred.pace.model import predict_mu

    mu, p_dnf = predict_mu(post, design, entrants, FakeRace(2023), grid=None, is_wet_prob=0.0)
    point = simulate_race(
        entrants,
        P,
        n_runs=2000,
        seed=0,
        strength_samples=mu.mean(axis=0)[None, :] * P.pace_scale,
        p_dnf_samples=p_dnf.mean(axis=0)[None, :],
    )
    assert spread.p_win.max() < point.p_win.max()
