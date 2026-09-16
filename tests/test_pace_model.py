import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from f1pred.config import DEFAULT_PARAMS
from f1pred.pace.model import fit, plackett_luce_logp, predict_mu
from f1pred.pace.posterior import Posterior
from tests.synthetic_pace import (
    LONG_SEASONS,
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
UNWEIGHTED = P.replace(season_half_life=float("inf"))
# "vet" is ordinary for five seasons and 2.0 quicker in the sixth, sharing car A with "ace".
BREAKOUT = ("vet", 2023, 2.0)


@pytest.fixture(scope="module")
def planted_pair():
    """The same planted breakout fit with the season age discount on and off."""
    kw = {"breakout": BREAKOUT, "seasons": LONG_SEASONS}
    weighted = fit(synthetic_design(seed=0, params=P, **kw), P, seed=0, **TINY)
    unweighted = fit(synthetic_design(seed=0, params=UNWEIGHTED, **kw), UNWEIGHTED, seed=0, **TINY)
    return weighted, unweighted


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


def _trace_sites(design) -> dict[str, str]:
    """Every site the model declares on one pass, by name, with its type."""
    from numpyro import handlers

    from f1pred.pace.model import model_arrays, pace_model

    arrays = model_arrays(design)
    traced = handlers.trace(handlers.seed(pace_model, rng_seed=0))
    return {name: site["type"] for name, site in traced.get_trace(design, P, arrays, True).items()}


def test_nu_season_is_a_module_constant_not_a_sampled_site(planted):
    """The Student-t tail on the season deviation is fixed, not learned.

    Sampling its degrees of freedom would add a second global parameter on top of an already
    funnel-shaped non-centred hierarchy. The heavy tail is there to let one exceptional
    driver-season escape the shrinkage, not to estimate how heavy the tail is.
    """
    from f1pred.pace import model as pace_module

    assert isinstance(pace_module.NU_SEASON, float)
    assert pace_module.NU_SEASON > 2.0
    design, _ = planted
    sites = _trace_sites(design)
    assert not any(name.startswith("nu_season") for name in sites)
    assert sites["skill_season_raw"] == "sample"


def test_nu_q_never_reaches_the_cauchy_region(planted):
    """`nu_q >= 2` on every draw, so the qualifying likelihood always has a finite variance.

    A near-Cauchy `nu_q` made the per-race intercepts of the wet/dry split sessions genuinely
    bimodal and took three of them to r_hat 12.7 with ESS 2.
    """
    design, post = planted
    nu_q = post.samples["nu_q"]
    assert nu_q.shape == (post.n_samples,)
    assert float(nu_q.min()) >= 2.0
    sites = _trace_sites(design)
    assert sites["nu_q"] == "deterministic"
    assert sites["nu_q_raw"] == "sample"


def test_a_planted_breakout_season_escapes_the_shrinkage():
    """`vet` is average for two seasons, then 2.0 quicker in the third beside a strong teammate.

    The Antonelli case in miniature: the breakout shares car A with `ace`, the quickest driver
    in the field, so the car term cancels between them and only the driver terms can explain
    the swap. The Normal season deviation this replaced recovered a mean of 0.4 to 0.9 on this
    data with the lower bound under zero — pooled over every driver-season it had to shrink the
    one real breakout with the bulk. The Student-t tail lets it through.
    """
    design = synthetic_design(seed=0, breakout=BREAKOUT, seasons=LONG_SEASONS)
    post = fit(design, P, seed=0, **TINY)
    table = post.driver_table(2023).set_index("driver_id")
    assert table.loc["vet", "season"] > 1.0
    assert table.loc["vet", "season_lo"] > 0.0  # the 90% interval is clear of zero
    assert table.loc["vet", "season"] > table.loc["ace", "season"]


def _log_density(design, params, latents):
    from numpyro.infer.util import log_density

    from f1pred.pace.model import model_arrays, pace_model

    lp, _ = log_density(pace_model, (design, params, model_arrays(design), True), {}, latents)
    return float(lp)


def test_the_season_weight_scales_every_likelihood_term_and_no_prior(planted):
    """Rescale every weight by a constant and the log-density must move linearly, from the prior.

    `log_density(c) = prior + c * loglik` if and only if every likelihood term is scaled and no
    prior is. Equal increments across c = 1, 2, 3 give the linearity; extrapolating back to
    c = 0 must land on the prior exactly, which it would not if any one of the three terms had
    been left unscaled. (numpyro's `scale` handler rejects a literal 0, hence the extrapolation.)
    """
    import dataclasses

    from numpyro import handlers

    from f1pred.pace.model import model_arrays, pace_model

    design, _ = planted
    trace = handlers.trace(handlers.seed(pace_model, rng_seed=0)).get_trace(
        design, P, model_arrays(design), True
    )
    latents = {
        k: s["value"]
        for k, s in trace.items()
        if s["type"] == "sample" and not s.get("is_observed")
    }
    prior = float(
        sum(
            np.sum(np.asarray(s["fn"].log_prob(s["value"])))
            for k, s in trace.items()
            if k in latents
        )
    )

    def at(scale: float) -> float:
        scaled = dataclasses.replace(
            design,
            row_weight=np.full(design.n_row, scale),
            race_weight=np.full(design.n_race, scale),
        )
        return _log_density(scaled, P, latents)

    one, two, three = at(1.0), at(2.0), at(3.0)
    assert (three - two) == pytest.approx(two - one, rel=1e-4)
    assert 2 * one - two == pytest.approx(prior, rel=1e-4)  # the intercept is the prior
    assert one < prior  # the likelihood of 30 races is not a rounding error


def test_discounting_older_seasons_credits_a_breakout_in_the_latest_one_more(planted_pair):
    """The same planted 2023 breakout, fit with and without the season age discount.

    Five ordinary seasons of `vet` pull `tau_season` toward zero and take the breakout with it.
    Discounting them by age leaves the pooling less able to do that, so `tau_season` comes up and
    more of the deviation survives on the season term where it belongs.
    """
    weighted, unweighted = planted_pair
    got = weighted.driver_table(2023).set_index("driver_id").loc["vet", "season"]
    base = unweighted.driver_table(2023).set_index("driver_id").loc["vet", "season"]
    assert got > base
    assert float(weighted.samples["tau_season"].mean()) > float(
        unweighted.samples["tau_season"].mean()
    )
