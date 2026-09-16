import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from f1pred.config import DEFAULT_PARAMS
from f1pred.pace.model import fit, plackett_luce_logp, predict_mu
from f1pred.pace.posterior import Posterior
from tests.synthetic_pace import (
    SEASONS,
    FakeRace,
    contaminated_design,
    last_race_entrants,
    planted_teammate_gap,
    synthetic_design,
    synthetic_table,
)

P = DEFAULT_PARAMS
TINY = {"num_warmup": 80, "num_samples": 80, "chains": 1}


def test_plackett_luce_matches_brute_force():
    mu = jnp.array([[2.0, 1.0, 0.0, -jnp.inf]])
    lengths = jnp.array([3])
    lp = plackett_luce_logp(mu, lengths)[0]
    expected = (
        (2 - logsumexp(jnp.array([2.0, 1.0, 0.0])))
        + (1 - logsumexp(jnp.array([1.0, 0.0])))
        + (0 - 0)
    )
    assert float(lp) == pytest.approx(float(expected), abs=1e-5)


def test_plackett_luce_handles_ragged_races():
    mu = jnp.array([[1.0, 0.0, -1.0], [0.5, -0.5, -jnp.inf]])
    lp = plackett_luce_logp(mu, jnp.array([3, 2]))
    second = 0.5 - float(logsumexp(jnp.array([0.5, -0.5]))) + (-0.5 - -0.5)
    assert float(lp[1]) == pytest.approx(second, abs=1e-5)
    # A longer race is less likely to come out in exactly one order.
    assert float(lp[0]) < float(lp[1])


def test_plackett_luce_is_invariant_to_a_constant_shift():
    mu = jnp.array([[2.0, 1.0, 0.0]])
    a = plackett_luce_logp(mu, jnp.array([3]))
    b = plackett_luce_logp(mu + 3.7, jnp.array([3]))
    assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-5)


@pytest.fixture(scope="module")
def planted():
    design = synthetic_design(seed=0)
    return design, fit(design, P, seed=0, **TINY)


def test_fit_recovers_planted_car_and_skill(planted):
    _, post = planted
    s = post.summary().set_index("param")
    assert s.loc["car[A,2023]", "mean"] > s.loc["car[B,2023]", "mean"] + 0.4
    assert s.loc["car[B,2023]", "mean"] > s.loc["car[C,2023]", "mean"] + 0.4
    assert s.loc["skill[ace,2023]", "mean"] > 0.3
    assert post.diagnostics["divergences"] < 5


def test_fit_recovers_a_positive_qualifying_scale(planted):
    _, post = planted
    s = post.summary().set_index("param")
    assert s.loc["kappa", "mean"] > 0.3
    assert post.diagnostics["num_samples"] == TINY["num_samples"] * TINY["chains"]


def test_nu_q_is_sampled_and_reported(planted):
    _, post = planted
    s = post.summary().set_index("param")
    assert "nu_q" in s.index
    assert s.loc["nu_q", "mean"] > 0.0
    assert np.isfinite(s.loc["nu_q", ["lo", "hi", "r_hat", "ess"]].to_numpy(dtype=float)).all()


def _teammate_gap(post, season: int) -> float:
    """The recovered pace between the two drivers of car A, where the car term cancels.

    Only contrasts are identified: the qualifying intercept absorbs a per-race shift and the
    Plackett-Luce term is shift-invariant, so the overall level is pinned by the priors alone.
    """
    total = post.driver_table(season).set_index("driver_id")["total"]
    return float(total["ace"] - total["vet"])


def test_student_t_quali_survives_junk_laps_that_drag_the_normal():
    """5% of the gaps are junk laps on one driver: the Normal follows them, the Student-t does not.

    Both variants see exactly the same data and the same finishing orders, so anything that
    separates them comes from the qualifying likelihood alone.
    """
    design = contaminated_design(seed=0, fraction=0.05)
    season, planted = SEASONS[-1], planted_teammate_gap()
    robust = fit(design, P, seed=0, **TINY)
    normal = fit(design, P, seed=0, robust_quali=False, **TINY)
    robust_error = abs(_teammate_gap(robust, season) - planted)
    normal_error = abs(_teammate_gap(normal, season) - planted)
    assert robust_error < normal_error
    # Not a tie broken by noise: the Normal loses most of the planted gap to the junk laps.
    assert normal_error > 0.5 * planted > robust_error
    # The Normal has to stretch itself over the junk; the Student-t reads it as tail.
    robust_sigma = robust.summary().set_index("param").loc["sigma_q", "mean"]
    normal_sigma = normal.summary().set_index("param").loc["sigma_q", "mean"]
    assert robust_sigma < normal_sigma
    assert "nu_q" not in set(normal.summary()["param"])


def test_breakout_season_is_credited_within_season():
    table = synthetic_table(seed=1, breakout=("bo", 2023, 1.6))
    design = synthetic_design(seed=1, breakout=("bo", 2023, 1.6))
    post = fit(design, P, seed=1, **TINY)
    entrants = last_race_entrants(table)
    mu, p_dnf = predict_mu(post, design, entrants, FakeRace(2023), grid=None, is_wet_prob=0.0)
    ids = [e.driver_id for e in entrants]
    means = dict(zip(ids, mu.mean(axis=0), strict=True))
    # "bo" shares car B with "bee" and was 1.6 quicker all of 2023.
    assert means["bo"] > means["bee"] + 0.5
    assert mu.shape == (post.n_samples, len(ids))
    assert p_dnf.shape == mu.shape and ((p_dnf > 0) & (p_dnf < 1)).all()


def test_predict_mu_falls_back_to_priors_for_an_unseen_entrant(planted):
    from f1pred.sim.race import Entrant

    design, post = planted
    entrants = [
        Entrant("ace", "A", 0.0, 1, 0.0),
        Entrant("rookie", "newco", 0.0, 2, 0.0),
    ]
    mu, p_dnf = predict_mu(post, design, entrants, FakeRace(2023), grid=None, is_wet_prob=0.0)
    assert np.isfinite(mu).all() and np.isfinite(p_dnf).all()
    # The unseen driver in an unseen car is far more uncertain than the known pairing.
    assert mu[:, 1].std() > mu[:, 0].std()


def test_posterior_roundtrip(tmp_path, planted):
    _, post = planted
    path = post.save(tmp_path / "wf.npz")
    back = Posterior.load(path)
    assert set(back.samples) == set(post.samples)
    for k, v in post.samples.items():
        np.testing.assert_allclose(back.samples[k], v)
    assert back.design_meta == post.design_meta
    assert back.diagnostics["divergences"] == post.diagnostics["divergences"]
    assert back.summary().equals(post.summary())


def test_grid_slot_moves_mu_by_beta_grid(planted):
    # The synthetic grid is a deterministic function of the planted pace, so beta_grid itself is
    # not identified there; what is testable is that the grid term enters mu exactly as specified.
    design, post = planted
    from f1pred.sim.race import Entrant

    entrants = [Entrant("ace", "A", 0.0, None, 0.0)]
    pole, _ = predict_mu(post, design, entrants, FakeRace(2023), {"ace": 1}, 0.0)
    back, _ = predict_mu(post, design, entrants, FakeRace(2023), {"ace": 20}, 0.0)
    expected = post.samples["beta_grid"] * (1.0 - 1.0 / 20**0.75)
    np.testing.assert_allclose(pole[:, 0] - back[:, 0], expected, rtol=1e-5, atol=1e-6)


def test_grid_dict_overrides_the_entrant_grid(planted):
    design, post = planted
    from f1pred.sim.race import Entrant

    from_dict, _ = predict_mu(
        post, design, [Entrant("ace", "A", 0.0, 20, 0.0)], FakeRace(2023), {"ace": 1}, 0.0
    )
    from_entrant, _ = predict_mu(
        post, design, [Entrant("ace", "A", 0.0, 1, 0.0)], FakeRace(2023), None, 0.0
    )
    np.testing.assert_allclose(from_dict, from_entrant)
