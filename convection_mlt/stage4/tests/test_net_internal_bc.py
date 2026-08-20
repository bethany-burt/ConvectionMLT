"""Stage 4 LowerNetInternalFlux contract: F_rad,net(0) + F_conv(0) = F_int."""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt import (
    ConstantGravity,
    ConstantGreyOpacity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    PhysicsConfig,
    SolveRoute,
    TopIrradiation,
    build_grid,
    log_pressure_edges,
)
from convection_mlt.radiation import (
    DEFAULT_DIFFUSIVITY,
    net_internal_residual,
    radiation_core,
)
from convection_mlt.rce import RCEConfig, _run_unsplit
from convection_mlt.state import build_column_state


def _roundoff_scale(*values: float) -> float:
    mag = max(1.0, *(abs(v) for v in values))
    return mag * np.finfo(np.float64).eps * 64.0


@pytest.mark.parametrize("n_layers", [1, 7, 11, 33])
@pytest.mark.parametrize("route", list(SolveRoute))
def test_grey_net_internal_invariant_arbitrary_grid(n_layers, route):
    rng = np.random.default_rng(n_layers + 17)
    mass = 10.0 ** rng.uniform(1.0, 4.0, size=n_layers)
    temp = 400.0 + 800.0 * rng.random(n_layers)
    kappa = np.full((1, n_layers), 2.0e-4)
    weights = np.array([1.0])
    top = np.array([120.0])
    f_int = 300.0
    f_conv0 = 0.0 if n_layers == 1 else 40.0
    dummy_bot = np.array([0.0])
    r = radiation_core(
        temp, mass, kappa, weights, top, dummy_bot, DEFAULT_DIFFUSIVITY, route,
        net_internal_flux=f_int, bottom_convective_flux=f_conv0,
    )
    resid = net_internal_residual(r, f_int, f_conv0)
    scale = _roundoff_scale(f_int, float(r.flux_down[0, 0]), f_conv0)
    assert resid <= scale
    assert abs(float(r.flux_net[0]) + f_conv0 - f_int) <= scale


@pytest.mark.parametrize("route", list(SolveRoute))
def test_multiband_net_internal_uses_band_weights(route):
    n = 8
    temp = np.linspace(1200.0, 400.0, n)
    mass = np.full(n, 250.0)
    kappa = np.vstack([
        np.full(n, 1.0e-4),
        np.full(n, 5.0e-4),
        np.full(n, 1.0e-8),
    ])
    weights = np.array([0.7, 0.3, 0.0])
    top = weights * 80.0
    f_int = 300.0
    f_conv0 = 25.0
    r = radiation_core(
        temp, mass, kappa, weights, top, np.zeros(3), DEFAULT_DIFFUSIVITY, route,
        net_internal_flux=f_int, bottom_convective_flux=f_conv0,
    )
    excess = r.flux_up[:, 0] - r.flux_down[:, 0]
    expected = weights * (f_int - f_conv0)
    band_scale = _roundoff_scale(f_int, float(np.max(np.abs(r.flux_down[:, 0]))))
    assert np.max(np.abs(excess - expected)) <= band_scale
    assert net_internal_residual(r, f_int, f_conv0) <= band_scale
    assert abs(float(np.sum(excess[2]))) <= band_scale


def test_solve_radiation_wrapper_and_unsplit_total_flux():
    g = 15.0
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e2, 12), g)
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(gravity=g, alpha=1.0, closure_prefactor=0.5)
    opacity = ConstantGreyOpacity(2.0e-4)
    t = 800.0 * (grid.pressure_centres / grid.pressure_centres[0]) ** 0.28
    state = build_column_state(grid, t, thermo, ConstantGravity(g))
    f_int = 300.0
    closure, rad, f_conv, f_rad, f_total = _run_unsplit(
        grid, state, physics, thermo, opacity, grid.pressure_centres,
        TopIrradiation(120.0), LowerNetInternalFlux(f_int),
        RCEConfig(), None, ConstantGravity(g),
    )
    scale = _roundoff_scale(f_int, float(rad.flux_down[0, 0]), float(f_conv[0]))
    assert abs(float(f_total[0]) - f_int) <= scale
    assert abs(float(f_rad[0] + f_conv[0]) - f_int) <= scale


def test_numpy_routes_agree_on_net_internal():
    n = 9
    temp = np.linspace(1500.0, 600.0, n)
    mass = np.geomspace(1.0e3, 10.0, n)
    kappa = np.full((1, n), 3.0e-4)
    weights = np.array([1.0])
    top = np.array([50.0])
    kwargs = dict(
        net_internal_flux=300.0, bottom_convective_flux=12.0,
    )
    results = [
        radiation_core(temp, mass, kappa, weights, top, np.array([0.0]), DEFAULT_DIFFUSIVITY, route, **kwargs)
        for route in SolveRoute
    ]
    scale = max(1.0, float(np.max(np.abs(results[0].flux_net))))
    for other in results[1:]:
        assert np.max(np.abs(results[0].flux_net - other.flux_net)) / scale < 1e-12


def test_jax_net_internal_parity():
    jax = pytest.importorskip("jax")
    if not jax.config.x64_enabled:
        pytest.skip("JAX x64 not enabled")
    import jax.numpy as jnp
    from convection_mlt.radiation_jax import radiation_core_jax_net_internal

    n = 10
    temp = np.full(n, 1100.0)
    mass = np.full(n, 400.0)
    kappa = np.vstack([np.full(n, 1e-4), np.full(n, 4e-4)])
    weights = np.array([0.55, 0.45])
    top = weights * 90.0
    f_int = 300.0
    f_conv0 = 18.0
    r_np = radiation_core(
        temp, mass, kappa, weights, top, np.zeros(2), DEFAULT_DIFFUSIVITY, SolveRoute.THOMAS,
        net_internal_flux=f_int, bottom_convective_flux=f_conv0,
    )
    r_jax = radiation_core_jax_net_internal(
        jnp.array(temp), jnp.array(mass), jnp.array(kappa), jnp.array(weights),
        jnp.array(top), jnp.array(f_int), jnp.array(f_conv0), DEFAULT_DIFFUSIVITY,
    )
    scale = max(1.0, float(np.max(np.abs(r_np.flux_net))))
    assert float(np.max(np.abs(r_np.flux_net - np.asarray(r_jax.flux_net)))) / scale < 1e-12
    assert net_internal_residual(r_np, f_int, f_conv0) <= _roundoff_scale(f_int, float(r_np.flux_down[0, 0]))
