"""The Bayesian hierarchical pace model in NumPyro. Spec 2.2; maths in `docs/pace-model.md`.

One latent pace per driver-race explains three things at once: the finishing order (a
Plackett-Luce likelihood), the qualifying gap to pole (a Normal likelihood sharing the same
skill and car effects), and whether the car finished (a logistic DNF hazard).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS

from f1pred.config import ModelParams
from f1pred.pace.design import GRID_SHAPE, PaceDesign, target_design
from f1pred.pace.posterior import Posterior

NEG = -1e30  # padding for a race shorter than the longest one; finite so gradients stay clean
ALPHA_SD = 5.0  # weak prior on the per-race qualifying intercept (percent units)
FIRST_SKILL_SD = 1.0
FIRST_CAR_SD = 1.5
HAZARD_SD = 0.5
H0_LOC, H0_SD = -2.0, 1.0
# Saved from a fit; everything else is a non-centring helper and is dropped.
KEEP = (
    "skill",
    "skill_season",
    "car",
    "car_form",
    "alpha",
    "wet",
    "track",
    "beta_grid",
    "kappa",
    "sigma_q",
    "tau_skill",
    "tau_season",
    "tau_car",
    "tau_car_reg",
    "tau_form",
    "tau_wet",
    "tau_track",
    "h0",
    "h_driver",
    "h_car",
    "h_circuit",
    "h_wet",
)


def plackett_luce_logp(mu_sorted: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    """Log likelihood of each race's finishing order given the paces, in that order.

    `mu_sorted[r, j]` is the pace of the driver who finished `j + 1`-th in race `r`, padded
    beyond `lengths[r]`. The order's probability is the product over finishing positions of the
    softmax of the paces still in contention, so the log likelihood is a reverse cumulative
    logsumexp subtracted from the paces themselves.
    """
    mu_sorted = jnp.asarray(mu_sorted)
    k = mu_sorted.shape[-1]
    reverse = jax.lax.associative_scan(jnp.logaddexp, mu_sorted[:, ::-1], axis=1)[:, ::-1]
    mask = jnp.arange(k)[None, :] < jnp.asarray(lengths)[:, None]
    return jnp.where(mask, mu_sorted - reverse, 0.0).sum(axis=1)


def chain_layout(previous: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay out random-walk chains end to end so a walk is one cumulative sum.

    Returns `(order, segment_start, inverse)`: `order` visits every node root-first along its
    chain, `segment_start[p]` is the position of the chain that position `p` belongs to, and
    `inverse` puts a value laid out in `order` back into the original indexing.
    """
    previous = np.asarray(previous)
    successor = {int(p): i for i, p in enumerate(previous) if p >= 0}
    order, segment_start = [], []
    for root in np.flatnonzero(previous < 0):
        start, node = len(order), int(root)
        while True:
            order.append(node)
            segment_start.append(start)
            if node not in successor:
                break
            node = successor[node]
    order_arr = np.asarray(order, dtype=np.int32)
    inverse = np.argsort(order_arr).astype(np.int32)
    return order_arr, np.asarray(segment_start, dtype=np.int32), inverse


def _walk(steps: jnp.ndarray, order, segment_start, inverse) -> jnp.ndarray:
    """Cumulative sum of `steps` along each chain, returned in the original indexing."""
    laid = steps[order]
    running = jnp.cumsum(laid)
    return (running - (running - laid)[segment_start])[inverse]


@dataclass(frozen=True)
class ModelArrays:
    """Everything the model function needs, converted once instead of on every gradient."""

    design: PaceDesign
    skill_chain: tuple
    car_chain: tuple
    form_chain: tuple
    form_free: jnp.ndarray
    n_form_free: int
    car_root: jnp.ndarray
    car_reg: jnp.ndarray
    pl_index: jnp.ndarray
    pl_mask: jnp.ndarray
    pl_len: jnp.ndarray
    q_rows: jnp.ndarray
    q_obs: jnp.ndarray
    row: dict[str, jnp.ndarray]


def model_arrays(design: PaceDesign) -> ModelArrays:
    skill_chain = tuple(jnp.asarray(a) for a in chain_layout(design.previous_driver_season))
    car_chain = tuple(jnp.asarray(a) for a in chain_layout(design.previous_constructor_season))
    form_chain = tuple(jnp.asarray(a) for a in chain_layout(design.previous_constructor_race))
    form_free = np.flatnonzero(design.previous_constructor_race >= 0).astype(np.int32)

    k = max(design.max_classified, 1)
    offsets = np.arange(k)[None, :]
    pl_index = design.race_start[:, None] + offsets
    pl_mask = offsets < design.race_n_classified[:, None]
    pl_index = np.where(pl_mask, pl_index, 0)

    q_rows = np.flatnonzero(np.isfinite(design.quali_gap)).astype(np.int32)
    return ModelArrays(
        design=design,
        skill_chain=skill_chain,
        car_chain=car_chain,
        form_chain=form_chain,
        form_free=jnp.asarray(form_free),
        n_form_free=int(form_free.size),
        car_root=jnp.asarray(design.previous_constructor_season < 0),
        car_reg=jnp.asarray(design.is_regulation_season),
        pl_index=jnp.asarray(pl_index),
        pl_mask=jnp.asarray(pl_mask),
        pl_len=jnp.asarray(design.race_n_classified),
        q_rows=jnp.asarray(q_rows),
        q_obs=jnp.asarray(design.quali_gap[q_rows]),
        row={
            "driver": jnp.asarray(design.driver),
            "driver_season": jnp.asarray(design.driver_season),
            "constructor_season": jnp.asarray(design.constructor_season),
            "constructor_race": jnp.asarray(design.constructor_race),
            "race": jnp.asarray(design.race),
            "circuit": jnp.asarray(design.circuit),
            "track_type": jnp.asarray(design.track_type),
            "is_wet": jnp.asarray(design.is_wet.astype(float)),
            "grid_g": jnp.asarray(design.grid_g),
            "dnf": jnp.asarray(design.dnf.astype(float)),
        },
    )


def pace_model(design: PaceDesign, params: ModelParams, arrays: ModelArrays | None = None) -> None:
    a = model_arrays(design) if arrays is None else arrays
    d, row = a.design, a.row

    tau_skill = numpyro.sample("tau_skill", dist.HalfNormal(0.3))
    tau_season = numpyro.sample("tau_season", dist.HalfNormal(0.5))
    tau_car = numpyro.sample("tau_car", dist.HalfNormal(0.5))
    tau_car_reg = numpyro.sample("tau_car_reg", dist.HalfNormal(1.5))
    tau_form = numpyro.sample("tau_form", dist.HalfNormal(0.1))
    tau_wet = numpyro.sample("tau_wet", dist.HalfNormal(0.3))
    tau_track = numpyro.sample("tau_track", dist.HalfNormal(0.3))
    beta_grid = numpyro.sample("beta_grid", dist.Normal(0.0, 1.0))
    kappa = numpyro.sample("kappa", dist.HalfNormal(2.0))
    sigma_q = numpyro.sample("sigma_q", dist.HalfNormal(1.0))

    # Driver skill: a slow random walk across seasons plus a looser within-season deviation.
    with numpyro.plate("driver_seasons", d.n_driver_season):
        skill_raw = numpyro.sample("skill_raw", dist.Normal(0.0, 1.0))
        season_raw = numpyro.sample("skill_season_raw", dist.Normal(0.0, 1.0))
    skill_scale = jnp.where(jnp.asarray(d.previous_driver_season < 0), FIRST_SKILL_SD, tau_skill)
    skill = numpyro.deterministic("skill", _walk(skill_raw * skill_scale, *a.skill_chain))
    skill_season = numpyro.deterministic("skill_season", season_raw * tau_season)

    # Car pace: a random walk across seasons, wide in a regulation season.
    with numpyro.plate("constructor_seasons", d.n_constructor_season):
        car_raw = numpyro.sample("car_raw", dist.Normal(0.0, 1.0))
    step = jnp.where(a.car_reg, tau_car_reg, tau_car)
    car_scale = jnp.where(a.car_root, FIRST_CAR_SD, step)
    car = numpyro.deterministic("car", _walk(car_raw * car_scale, *a.car_chain))

    # Within-season upgrades: a slow walk over each constructor's races, reset to 0 each season.
    with numpyro.plate("constructor_race_steps", a.n_form_free):
        form_raw = numpyro.sample("car_form_raw", dist.Normal(0.0, 1.0))
    steps = jnp.zeros(d.n_constructor_race).at[a.form_free].set(form_raw * tau_form)
    car_form = numpyro.deterministic("car_form", _walk(steps, *a.form_chain))

    with numpyro.plate("drivers", d.n_driver):
        wet_raw = numpyro.sample("wet_raw", dist.Normal(0.0, 1.0))
    wet = numpyro.deterministic("wet", wet_raw * tau_wet)
    track_raw = numpyro.sample(
        "track_raw", dist.Normal(0.0, 1.0).expand([d.n_driver, d.n_track_type]).to_event(2)
    )
    track = numpyro.deterministic("track", track_raw * tau_track)

    shared = skill[row["driver_season"]] + skill_season[row["driver_season"]]
    car_pace = car[row["constructor_season"]] + car_form[row["constructor_race"]]
    mu = (
        shared
        + car_pace
        + beta_grid * row["grid_g"]
        + wet[row["driver"]] * row["is_wet"]
        + track[row["driver"], row["track_type"]]
    )

    mu_sorted = jnp.where(a.pl_mask, mu[a.pl_index], NEG)
    numpyro.factor("finishing_order", plackett_luce_logp(mu_sorted, a.pl_len).sum())

    # Qualifying: the same latent pace seen again, without the grid, wet or track terms.
    with numpyro.plate("races", d.n_race):
        alpha = numpyro.sample("alpha", dist.Normal(0.0, ALPHA_SD))
    mu_q = shared + car_pace
    loc = alpha[row["race"]] - kappa * mu_q
    numpyro.sample("quali", dist.Normal(loc[a.q_rows], sigma_q), obs=a.q_obs)

    h0 = numpyro.sample("h0", dist.Normal(H0_LOC, H0_SD))
    h_wet = numpyro.sample("h_wet", dist.Normal(0.0, HAZARD_SD))
    with numpyro.plate("hazard_drivers", d.n_driver):
        h_driver = numpyro.sample("h_driver", dist.Normal(0.0, HAZARD_SD))
    with numpyro.plate("hazard_cars", d.n_constructor_season):
        h_car = numpyro.sample("h_car", dist.Normal(0.0, HAZARD_SD))
    with numpyro.plate("hazard_circuits", d.n_circuit):
        h_circuit = numpyro.sample("h_circuit", dist.Normal(0.0, HAZARD_SD))
    logits = (
        h0
        + h_driver[row["driver"]]
        + h_car[row["constructor_season"]]
        + h_circuit[row["circuit"]]
        + h_wet * row["is_wet"]
    )
    numpyro.sample("retirement", dist.Bernoulli(logits=logits), obs=row["dnf"])


def _configure(device: str, chains: int) -> str:
    """Pick the JAX platform and the chain method. Must run before JAX touches a device."""
    numpyro.set_platform(device)
    if device == "cpu":
        numpyro.set_host_device_count(chains)
        return "parallel"
    return "vectorized"


def fit(
    design: PaceDesign,
    params: ModelParams,
    num_warmup: int | None = None,
    num_samples: int | None = None,
    chains: int | None = None,
    seed: int = 0,
    device: str | None = None,
    progress_bar: bool = False,
) -> Posterior:
    """Run NUTS and return the posterior. `None` arguments fall back to `params.pace_*`."""
    num_warmup = params.pace_num_warmup if num_warmup is None else num_warmup
    num_samples = params.pace_num_samples if num_samples is None else num_samples
    chains = params.pace_chains if chains is None else chains
    device = params.pace_device if device is None else device
    chain_method = _configure(device, chains)

    arrays = model_arrays(design)
    kernel = NUTS(pace_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    started = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(seed),
        design,
        params,
        arrays,
        extra_fields=("diverging",),
    )
    seconds = time.perf_counter() - started

    grouped = mcmc.get_samples(group_by_chain=True)
    stats = numpyro_summary(grouped, prob=0.9, group_by_chain=True)
    samples = {
        k: np.asarray(v, dtype=np.float32) for k, v in mcmc.get_samples().items() if k in KEEP
    }
    r_hat = {k: np.asarray(stats[k]["r_hat"]) for k in samples if k in stats}
    ess = {k: np.asarray(stats[k]["n_eff"]) for k in samples if k in stats}
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    reported = (
        np.concatenate([np.ravel(v) for v in r_hat.values()]) if r_hat else np.array([np.nan])
    )
    diagnostics = {
        "divergences": divergences,
        "num_samples": int(num_samples * chains),
        "num_chains": int(chains),
        "num_warmup": int(num_warmup),
        "seconds": round(seconds, 1),
        "device": device,
        "max_r_hat": float(np.nanmax(reported)),
        "min_ess": float(np.nanmin(np.concatenate([np.ravel(v) for v in ess.values()])))
        if ess
        else float("nan"),
        "n_row": design.n_row,
        "n_race": design.n_race,
        "last_race_id": int(design.race_ids[-1]),
        "last_season": int(design.seasons[-1]),
    }
    return Posterior(samples, design.meta(), diagnostics, r_hat, ess)


def _draws(rng: np.random.Generator, scale: np.ndarray | float, size: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=size) * np.asarray(scale)


def predict_mu(
    posterior: Posterior,
    design: PaceDesign,
    entrants,
    race_info,
    grid: dict[str, int] | None = None,
    is_wet_prob: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior paces and DNF probabilities for one race: `(mu (S, n), p_dnf (S, n))`.

    An entrant the design never saw falls back to the model's own priors, drawn once per
    posterior sample, so an unknown driver or a brand-new team is wide rather than average.
    """
    s = posterior.samples
    n_draw = posterior.n_samples
    rng = np.random.default_rng(seed)
    season = int(race_info.season)
    track_index = design.track_type_index(getattr(race_info, "track_type", ""))
    mapped = target_design(design, entrants, race_info)

    mu = np.zeros((n_draw, len(entrants)), dtype=float)
    logit = np.zeros((n_draw, len(entrants)), dtype=float)
    for i, e in enumerate(entrants):
        driver_i, cs_i, circuit_i = mapped[e.driver_id]
        ds_i = design.driver_season_index(e.driver_id, season)
        if ds_i >= 0:
            skill = s["skill"][:, ds_i]
            skill_season = s["skill_season"][:, ds_i]
        elif driver_i >= 0:
            last = _last_driver_season(design, e.driver_id)
            skill = s["skill"][:, last] + _draws(rng, s["tau_skill"], n_draw)
            skill_season = _draws(rng, s["tau_season"], n_draw)
        else:
            skill = _draws(rng, FIRST_SKILL_SD, n_draw)
            skill_season = _draws(rng, s["tau_season"], n_draw)

        if cs_i >= 0:
            car = s["car"][:, cs_i]
            form_i = design.last_constructor_race(e.constructor_id, season)
            car_form = s["car_form"][:, form_i] if form_i >= 0 else np.zeros(n_draw)
        else:
            previous = _last_constructor_season(design, e.constructor_id, season)
            walk = s["tau_car_reg"] if season in _regulation(design) else s["tau_car"]
            car = (
                s["car"][:, previous] + _draws(rng, walk, n_draw)
                if previous >= 0
                else _draws(rng, FIRST_CAR_SD, n_draw)
            )
            car_form = np.zeros(n_draw)

        wet = s["wet"][:, driver_i] if driver_i >= 0 else _draws(rng, s["tau_wet"], n_draw)
        track = (
            s["track"][:, driver_i, track_index]
            if driver_i >= 0 and track_index >= 0
            else _draws(rng, s["tau_track"], n_draw)
        )
        slot = (grid or {}).get(e.driver_id, e.grid)
        grid_g = 0.0 if slot is None else 1.0 / float(max(int(slot), 1)) ** GRID_SHAPE
        mu[:, i] = (
            skill
            + skill_season
            + car
            + car_form
            + s["beta_grid"] * grid_g
            + wet * is_wet_prob
            + track
        )

        h_driver = s["h_driver"][:, driver_i] if driver_i >= 0 else _draws(rng, HAZARD_SD, n_draw)
        h_car = s["h_car"][:, cs_i] if cs_i >= 0 else _draws(rng, HAZARD_SD, n_draw)
        h_circuit = (
            s["h_circuit"][:, circuit_i] if circuit_i >= 0 else _draws(rng, HAZARD_SD, n_draw)
        )
        logit[:, i] = s["h0"] + h_driver + h_car + h_circuit + s["h_wet"] * is_wet_prob
    return mu, 1.0 / (1.0 + np.exp(-logit))


def _last_driver_season(design: PaceDesign, driver_id: str) -> int:
    matches = [i for i, (d, _) in enumerate(design.driver_season_keys) if d == driver_id]
    return matches[-1] if matches else -1


def _last_constructor_season(design: PaceDesign, constructor_id: str, season: int) -> int:
    matches = [
        i
        for i, (c, s) in enumerate(design.constructor_season_keys)
        if c == constructor_id and s < season
    ]
    return matches[-1] if matches else -1


def _regulation(design: PaceDesign) -> set[int]:
    return {
        s
        for (_, s), flag in zip(
            design.constructor_season_keys, design.is_regulation_season, strict=True
        )
        if flag
    }
