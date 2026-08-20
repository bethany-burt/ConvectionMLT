"""JAX implementation of radiation_core for Stage 3 parity (point 34).

Requires JAX_ENABLE_X64=True before import.
All inputs/outputs are JAX arrays; no Python objects.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

STEFAN_BOLTZMANN = 5.670374419e-8


class JaxRadiationResult(NamedTuple):
    flux_up: jax.Array
    flux_down: jax.Array
    flux_net_band: jax.Array
    flux_net: jax.Array
    heating: jax.Array
    optical_depth: jax.Array
    transmissivity: jax.Array


def _thomas_solve_jax(lower, diag, upper, rhs):
    """Thomas algorithm in JAX using lax.scan."""
    n = diag.shape[0]

    def forward_step(carry, i):
        d_prev, b_prev = carry
        m = lower[i - 1] / d_prev
        d_new = diag[i] - m * upper[i - 1]
        b_new = rhs[i] - m * b_prev
        return (d_new, b_new), (d_new, b_new)

    init = (diag[0], rhs[0])
    _, (d_all, b_all) = jax.lax.scan(forward_step, init, jnp.arange(1, n))
    d_full = jnp.concatenate([diag[0:1], d_all])
    b_full = jnp.concatenate([rhs[0:1], b_all])

    def backward_step(carry, i):
        x_next = carry
        x_i = (b_full[i] - upper[i] * x_next) / d_full[i]
        return x_i, x_i

    x_last = b_full[n - 1] / d_full[n - 1]
    _, x_rev = jax.lax.scan(backward_step, x_last, jnp.arange(n - 2, -1, -1))
    x = jnp.concatenate([jnp.flip(x_rev), x_last[None]])
    return x


def _solve_down_jax(trans, emission_frac, source, f_down_top):
    n_layer = trans.shape[0]
    diag_d = jnp.ones(n_layer, dtype=jnp.float64)
    upper_d = -trans[:n_layer - 1]
    lower_d = jnp.zeros(n_layer - 1, dtype=jnp.float64)
    rhs_d = emission_frac * source
    rhs_d = rhs_d.at[n_layer - 1].add(trans[n_layer - 1] * f_down_top)
    x_down = _thomas_solve_jax(lower_d, diag_d, upper_d, rhs_d)
    return jnp.concatenate([x_down, f_down_top[None]])


def _solve_up_jax(trans, emission_frac, source, f_up_bot):
    n_layer = trans.shape[0]
    diag_u = jnp.ones(n_layer, dtype=jnp.float64)
    lower_u = -trans[1:]
    upper_u = jnp.zeros(n_layer - 1, dtype=jnp.float64)
    rhs_u = emission_frac * source
    rhs_u = rhs_u.at[0].add(trans[0] * f_up_bot)
    x_up = _thomas_solve_jax(lower_u, diag_u, upper_u, rhs_u)
    return jnp.concatenate([f_up_bot[None], x_up])


def _solve_band_jax(trans, emission_frac, source, f_down_top, f_up_bot):
    """Solve one band using Thomas (bidiagonal)."""
    fd = _solve_down_jax(trans, emission_frac, source, f_down_top)
    fu = _solve_up_jax(trans, emission_frac, source, f_up_bot)
    return fu, fd


def radiation_core_jax(
    temperature: jax.Array,
    mass_path: jax.Array,
    kappa: jax.Array,
    band_weights: jax.Array,
    top_down_flux_band: jax.Array,
    bottom_up_flux_band: jax.Array,
    diffusivity_factor: float,
) -> JaxRadiationResult:
    """JAX radiation_core: Δτ → 𝒯, B → solve → F↑, F↓, F_net → dh/dt."""
    dtau = kappa * mass_path[None, :]
    d_dtau = diffusivity_factor * dtau
    trans = jnp.exp(-d_dtau)
    emission_frac = -jnp.expm1(-d_dtau)
    planck_total = STEFAN_BOLTZMANN * temperature ** 4
    source = band_weights[:, None] * planck_total[None, :]

    def solve_one_band(b_idx):
        return _solve_band_jax(
            trans[b_idx], emission_frac[b_idx], source[b_idx],
            top_down_flux_band[b_idx], bottom_up_flux_band[b_idx],
        )

    n_band = kappa.shape[0]
    flux_up, flux_down = jax.vmap(
        lambda b: _solve_band_jax(
            trans[b], emission_frac[b], source[b],
            top_down_flux_band[b], bottom_up_flux_band[b],
        )
    )(jnp.arange(n_band))

    flux_net_band = flux_up - flux_down
    flux_net = jnp.sum(flux_net_band, axis=0)
    heating = (flux_net[:-1] - flux_net[1:]) / mass_path

    return JaxRadiationResult(
        flux_up=flux_up,
        flux_down=flux_down,
        flux_net_band=flux_net_band,
        flux_net=flux_net,
        heating=heating,
        optical_depth=dtau,
        transmissivity=trans,
    )


def radiation_core_jax_net_internal(
    temperature: jax.Array,
    mass_path: jax.Array,
    kappa: jax.Array,
    band_weights: jax.Array,
    top_down_flux_band: jax.Array,
    f_int: jax.Array,
    f_conv_bottom: jax.Array,
    diffusivity_factor: float,
) -> JaxRadiationResult:
    """JAX radiation_core with F_rad,net(0) + F_conv(0) = F_int."""
    dtau = kappa * mass_path[None, :]
    d_dtau = diffusivity_factor * dtau
    trans = jnp.exp(-d_dtau)
    emission_frac = -jnp.expm1(-d_dtau)
    planck_total = STEFAN_BOLTZMANN * temperature ** 4
    source = band_weights[:, None] * planck_total[None, :]
    excess = f_int - f_conv_bottom

    def _one_band(b):
        fd = _solve_down_jax(
            trans[b], emission_frac[b], source[b], top_down_flux_band[b]
        )
        f_up_bot = fd[0] + band_weights[b] * excess
        fu = _solve_up_jax(trans[b], emission_frac[b], source[b], f_up_bot)
        return fu, fd

    n_band = kappa.shape[0]
    flux_up, flux_down = jax.vmap(_one_band)(jnp.arange(n_band))
    flux_net_band = flux_up - flux_down
    flux_net = jnp.sum(flux_net_band, axis=0)
    heating = (flux_net[:-1] - flux_net[1:]) / mass_path
    return JaxRadiationResult(
        flux_up=flux_up,
        flux_down=flux_down,
        flux_net_band=flux_net_band,
        flux_net=flux_net,
        heating=heating,
        optical_depth=dtau,
        transmissivity=trans,
    )
