"""Hydrostatic reconstruction and analytic height tests."""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.gravity import ConstantGravity, InverseSquareGravity
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.hydrostatics import (
    HydrostaticDomainError,
    isothermal_constant_g_height,
    isothermal_inverse_square_height,
    pressure_from_height,
    reconstruct_hydrostatic,
)
from convection_mlt.thermodynamics import ConstantH2Thermo, NASAThermo


def test_analytic_isothermal_constant_g_height():
    gas = ConstantH2Thermo()
    g0 = 10.0
    temperature = 1000.0
    p_bottom, p_top = 1.0e7, 1.0e3
    analytic = isothermal_constant_g_height(
        p_bottom, p_top, temperature, gas.gas_constant, g0
    )
    grid = build_grid(log_pressure_edges(p_bottom, p_top, 100), g0)
    t = np.full(grid.n_layers, temperature)
    hydro = reconstruct_hydrostatic(grid, t, gas, ConstantGravity(g0))
    hp = gas.gas_constant * temperature / g0
    assert float(np.max(np.abs(hydro.z_edges[-1] - analytic) / hp)) <= 1.0e-12


def test_analytic_isothermal_inverse_square_height():
    gas = ConstantH2Thermo()
    g0 = 10.0
    rp = 1.0e8
    gravity = InverseSquareGravity(g0=g0, planet_radius=rp)
    temperature = 1000.0
    p_bottom, p_top = 1.0e7, 1.0e3
    analytic = isothermal_inverse_square_height(
        p_bottom, p_top, temperature, gas.gas_constant, gravity, 0.0
    )
    grid = build_grid(log_pressure_edges(p_bottom, p_top, 80), g0)
    t = np.full(grid.n_layers, temperature)
    hydro = reconstruct_hydrostatic(grid, t, gas, gravity)
    assert float(np.max(np.abs(hydro.z_edges[-1] / analytic - 1.0))) <= 1.0e-10
    assert hydro.max_z_over_rp > 0.0


def test_pressure_height_round_trip():
    gas = NASAThermo.from_json()
    g0 = 10.0
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 50), g0)
    temperature = 2000.0 * (grid.pressure_centres / grid.pressure_centres[0]) ** 0.12
    for gravity in (ConstantGravity(g0), InverseSquareGravity(g0=g0, planet_radius=1.0e8)):
        hydro = reconstruct_hydrostatic(grid, temperature, gas, gravity)
        recovered = pressure_from_height(
            grid, temperature, gas, gravity, hydro.z_edges, hydro=hydro
        )
        assert float(np.max(np.abs(recovered / grid.pressure_edges - 1.0))) <= 1.0e-10


def _continuous_temperature(pressure: float, p_bottom: float, p_top: float) -> float:
    xi = np.log(p_bottom / pressure) / np.log(p_bottom / p_top)
    # Mild curvature so the piecewise-constant-T scheme meets the N=100 gate.
    return 1500.0 + 0.5 * np.sin(np.pi * xi)


def _independent_continuous_integrator(
    pressure_edges: np.ndarray,
    gas_constant: float,
    gravity,
    n_sub: int = 64,
) -> np.ndarray:
    """High-order reference using continuous T(P) inside each pressure slab."""
    p_bottom = float(pressure_edges[0])
    p_top = float(pressure_edges[-1])
    z = np.zeros_like(pressure_edges)
    for i in range(len(pressure_edges) - 1):
        log_p = np.linspace(np.log(pressure_edges[i]), np.log(pressure_edges[i + 1]), n_sub + 1)
        z_local = float(z[i])
        for j in range(n_sub):
            p_mid = float(np.exp(0.5 * (log_p[j] + log_p[j + 1])))
            t_mid = _continuous_temperature(p_mid, p_bottom, p_top)
            g = float(gravity.gravity(np.asarray([z_local]))[0])
            dlogp = log_p[j + 1] - log_p[j]
            z_local = z_local + dlogp * (-gas_constant * t_mid / g)
        z[i + 1] = z_local
    return z


def test_manufactured_nonisothermal_vs_independent_and_refinement():
    gas = ConstantH2Thermo()
    g0 = 10.0
    gravity = ConstantGravity(g0)
    p_bottom, p_top = 1.0e7, 1.0e3
    errors = {}
    for n in (50, 100, 200):
        grid = build_grid(log_pressure_edges(p_bottom, p_top, n), g0)
        temperature = np.array(
            [_continuous_temperature(float(p), p_bottom, p_top) for p in grid.pressure_centres]
        )
        hydro = reconstruct_hydrostatic(grid, temperature, gas, gravity)
        z_ref = _independent_continuous_integrator(
            grid.pressure_edges, gas.gas_constant, gravity, n_sub=512
        )
        errors[n] = float(np.max(np.abs(hydro.z_edges - z_ref)) / np.max(np.abs(z_ref)))
    assert errors[100] <= 1.0e-8
    assert errors[50] / errors[100] >= 1.5
    assert errors[100] / errors[200] >= 1.5


def test_inverse_square_domain_error_for_unreachable_column():
    gas = ConstantH2Thermo()
    gravity = InverseSquareGravity(g0=10.0, planet_radius=1.0e5)
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e-2, 20), 10.0)
    temperature = np.full(grid.n_layers, 5000.0)
    with pytest.raises(HydrostaticDomainError):
        reconstruct_hydrostatic(grid, temperature, gas, gravity)


def test_approach_to_constant_g_as_rp_increases():
    gas = ConstantH2Thermo()
    g0 = 10.0
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 40), g0)
    temperature = np.full(grid.n_layers, 1000.0)
    z_const = reconstruct_hydrostatic(grid, temperature, gas, ConstantGravity(g0)).z_edges
    prev = None
    small_extent_ok = False
    for rp in (1.0e7, 1.0e8, 1.0e9, 1.0e10):
        hydro = reconstruct_hydrostatic(
            grid, temperature, gas, InverseSquareGravity(g0=g0, planet_radius=rp)
        )
        rel = float(
            np.sqrt(np.mean(((hydro.z_edges - z_const) / np.maximum(np.abs(z_const), 1.0)) ** 2))
        )
        if prev is not None:
            assert rel < prev
        prev = rel
        if hydro.max_z_over_rp < 0.02:
            if rel <= 1.0e-3:
                small_extent_ok = True
    assert small_extent_ok
