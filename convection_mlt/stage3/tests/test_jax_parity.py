"""Stage 3 point 34 — JAX float64 parity tests.

Requires: jax installed, JAX_ENABLE_X64=True.
Compares NumPy radiation_core vs JAX radiation_core_jax for:
  - eager JAX vs NumPy
  - JIT JAX vs NumPy
  - vmap batched vs independent per-band NumPy
  - batch-size-1 ≡ scalar route
  - column energy identity residual
All at float64 with parity gate ≤ 1e-12.
"""

from __future__ import annotations

import os
import time

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from convection_mlt.radiation import (
    DEFAULT_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    SolveRoute,
    radiation_core,
)
from convection_mlt.radiation_jax import radiation_core_jax

GATE = 1e-12


def _norm_diff(a, b, floor=1e-30):
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    scale = max(floor, float(np.max(np.abs(a_np))), float(np.max(np.abs(b_np))))
    return float(np.max(np.abs(a_np - b_np))) / scale


def _make_test_inputs(n_layer=10, n_band=1):
    T0 = 2000.0
    temp = np.full(n_layer, T0)
    mass_path = np.full(n_layer, 1000.0)
    if n_band == 1:
        kappa = np.full((1, n_layer), 0.05)
        weights = np.array([1.0])
    else:
        kappa = np.array([
            np.full(n_layer, 0.01),
            np.full(n_layer, 0.1),
            np.full(n_layer, 0.05),
        ])
        weights = np.array([0.6, 0.4, 0.0])
    B0 = STEFAN_BOLTZMANN * T0 ** 4
    top = weights * 100.0
    bot = weights * B0
    D = DEFAULT_DIFFUSIVITY
    return temp, mass_path, kappa, weights, top, bot, D


def _to_jax(*arrays):
    return tuple(jnp.array(a, dtype=jnp.float64) for a in arrays)


class TestEagerParity:
    """Eager JAX vs NumPy Thomas."""

    @pytest.mark.parametrize("n_band", [1, 3])
    def test_eager(self, n_band):
        temp, mass_path, kappa, w, top, bot, D = _make_test_inputs(n_band=n_band)
        r_np = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)

        j_temp, j_mp, j_kappa, j_w, j_top, j_bot = _to_jax(temp, mass_path, kappa, w, top, bot)
        r_jax = radiation_core_jax(j_temp, j_mp, j_kappa, j_w, j_top, j_bot, D)

        assert r_jax.flux_up.dtype == jnp.float64
        assert _norm_diff(r_np.flux_up, r_jax.flux_up) < GATE
        assert _norm_diff(r_np.flux_down, r_jax.flux_down) < GATE
        assert _norm_diff(r_np.flux_net, r_jax.flux_net) < GATE
        assert _norm_diff(r_np.heating, r_jax.heating) < GATE
        assert _norm_diff(r_np.optical_depth, r_jax.optical_depth) < GATE
        assert _norm_diff(r_np.transmissivity, r_jax.transmissivity) < GATE


class TestJITParity:
    """JIT-compiled JAX vs NumPy."""

    @pytest.mark.parametrize("n_band", [1, 3])
    def test_jit(self, n_band):
        temp, mass_path, kappa, w, top, bot, D = _make_test_inputs(n_band=n_band)
        r_np = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)

        j_temp, j_mp, j_kappa, j_w, j_top, j_bot = _to_jax(temp, mass_path, kappa, w, top, bot)

        jitted = jax.jit(radiation_core_jax, static_argnames=("diffusivity_factor",))

        t0 = time.perf_counter()
        r_jax = jitted(j_temp, j_mp, j_kappa, j_w, j_top, j_bot, diffusivity_factor=D)
        # force computation
        _ = r_jax.flux_up.block_until_ready()
        compile_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_jax2 = jitted(j_temp, j_mp, j_kappa, j_w, j_top, j_bot, diffusivity_factor=D)
        _ = r_jax2.flux_up.block_until_ready()
        exec_time = time.perf_counter() - t0

        assert r_jax.flux_up.dtype == jnp.float64
        assert _norm_diff(r_np.flux_up, r_jax.flux_up) < GATE
        assert _norm_diff(r_np.flux_down, r_jax.flux_down) < GATE
        assert _norm_diff(r_np.flux_net, r_jax.flux_net) < GATE
        assert _norm_diff(r_np.heating, r_jax.heating) < GATE

        # diagnostic: print timings (not assertions)
        print(f"\n  [JAX n_band={n_band}] compile={compile_time:.3f}s, exec={exec_time:.6f}s")


class TestBatchParity:
    """vmap batched JAX vs independent per-band NumPy calls."""

    def test_batch_vs_independent(self):
        temp, mass_path, kappa, w, top, bot, D = _make_test_inputs(n_band=3)
        r_np = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)

        j_temp, j_mp, j_kappa, j_w, j_top, j_bot = _to_jax(temp, mass_path, kappa, w, top, bot)
        r_jax = radiation_core_jax(j_temp, j_mp, j_kappa, j_w, j_top, j_bot, D)

        assert _norm_diff(r_np.flux_up, r_jax.flux_up) < GATE
        assert _norm_diff(r_np.flux_down, r_jax.flux_down) < GATE
        assert _norm_diff(r_np.heating, r_jax.heating) < GATE


class TestBatchSizeOne:
    """Batch size 1 must reproduce the scalar (grey) route."""

    def test_batch_one(self):
        temp, mass_path, kappa, w, top, bot, D = _make_test_inputs(n_band=1)
        r_np = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)

        j_temp, j_mp, j_kappa, j_w, j_top, j_bot = _to_jax(temp, mass_path, kappa, w, top, bot)
        r_jax = radiation_core_jax(j_temp, j_mp, j_kappa, j_w, j_top, j_bot, D)

        assert _norm_diff(r_np.flux_up, r_jax.flux_up) < GATE
        assert _norm_diff(r_np.flux_down, r_jax.flux_down) < GATE


class TestColumnEnergyParity:
    """Column energy identity residual matches between NumPy and JAX."""

    def test_energy_residual(self):
        temp, mass_path, kappa, w, top, bot, D = _make_test_inputs(n_band=3)

        r_np = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)
        n = temp.shape[0]
        lhs_np = float(np.sum(mass_path * r_np.heating))
        rhs_np = float(r_np.flux_net[0] - r_np.flux_net[n])
        resid_np = abs(lhs_np - rhs_np)

        j_temp, j_mp, j_kappa, j_w, j_top, j_bot = _to_jax(temp, mass_path, kappa, w, top, bot)
        r_jax = radiation_core_jax(j_temp, j_mp, j_kappa, j_w, j_top, j_bot, D)
        lhs_jax = float(jnp.sum(jnp.array(mass_path) * r_jax.heating))
        rhs_jax = float(r_jax.flux_net[0] - r_jax.flux_net[n])
        resid_jax = abs(lhs_jax - rhs_jax)

        scale = max(1e-30, abs(rhs_np))
        assert resid_np / scale < GATE
        assert resid_jax / scale < GATE
