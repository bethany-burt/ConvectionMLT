"""Entropy-defined instability and manufactured isentrope tests."""

from __future__ import annotations

import numpy as np

from convection_mlt.closure import mixing_length_flux
from convection_mlt.diagnostics import numerical_isentrope
from convection_mlt.gravity import ConstantGravity
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.state import build_column_state
from convection_mlt.thermodynamics import ConstantH2Thermo, NASAThermo, analytic_h2_oracle


def test_constant_cp_entropy_delta_matches_legacy_nabla():
    gas = ConstantH2Thermo()
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 40), 10.0)
    temperature = 1500.0 * (grid.pressure_centres / grid.pressure_centres[0]) ** 0.40
    entropy_closure = mixing_length_flux(
        grid, temperature, 10.0, 1.0, gas, use_entropy_instability=True
    )
    legacy_closure = mixing_length_flux(
        grid, temperature, 10.0, 1.0, gas, use_entropy_instability=False
    )
    assert np.max(
        np.abs(entropy_closure.superadiabaticity - legacy_closure.superadiabaticity)
    ) <= 1.0e-14


def test_manufactured_isentrope_has_exact_zero_entropy_jump():
    for thermo in (analytic_h2_oracle(), NASAThermo.from_json()):
        grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 50), 10.0)
        # Superadiabatic seed with enthalpy inside the reachable isentrope domain.
        seed = 4000.0 * (grid.pressure_centres / grid.pressure_centres[0]) ** 0.30
        state = build_column_state(grid, seed, thermo, ConstantGravity(10.0))
        isentrope = numerical_isentrope(grid, seed, thermo, state.mass_path)
        entropy = thermo.entropy(isentrope, grid.pressure_centres)
        scale = max(float(np.max(np.abs(entropy))), 1.0)
        # Absolute 1e-12 is below one float64 ulp for |s|~2e4; use ulp-aware bound.
        assert float(np.max(np.abs(entropy - entropy[0]))) <= max(
            1.0e-12, 64.0 * np.finfo(float).eps * scale
        )

        closure = mixing_length_flux(
            grid,
            isentrope,
            state.g_edges,
            1.0,
            thermo,
            use_entropy_instability=True,
        )
        assert float(np.max(np.abs(closure.entropy_jump))) <= max(
            1.0e-12, 64.0 * np.finfo(float).eps * scale
        )
        assert float(np.max(np.abs(closure.superadiabaticity))) <= 1.0e-12


def test_potential_temperature_recovers_reference_pressure_entropy():
    thermo = NASAThermo.from_json()
    pressure = np.array([1.0e5, 1.0e6, 1.0e4])
    temperature = np.array([800.0, 1200.0, 600.0])
    theta = thermo.potential_temperature(temperature, pressure)
    s_state = thermo.entropy(temperature, pressure)
    s_theta = thermo.entropy(theta, np.full_like(theta, thermo.p_ref))
    assert np.max(np.abs(s_state - s_theta)) <= 1.0e-10
