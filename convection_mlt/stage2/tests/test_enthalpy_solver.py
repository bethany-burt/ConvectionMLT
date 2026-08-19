"""Enthalpy solver conservation, isentrope agreement, and rejected-state purity."""

from __future__ import annotations

import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import numerical_isentrope
from convection_mlt.energy import column_enthalpy_per_area
from convection_mlt.gravity import ConstantGravity
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import TerminalStatus
from convection_mlt.solvers_enthalpy import solve_adaptive_enthalpy, trial_enthalpy_step
from convection_mlt.state import build_column_state
from convection_mlt.thermodynamics import NASAThermo


def _superadiabatic_seed(grid, t_bottom: float = 4000.0, nabla: float = 0.30):
    return t_bottom * (grid.pressure_centres / grid.pressure_centres[0]) ** nabla


def test_constant_g_enthalpy_conservation_and_isentrope_agreement():
    nasa = NASAThermo.from_json()
    g0 = 10.0
    gravity = ConstantGravity(g0)
    physics = PhysicsConfig(gravity=g0, alpha=1.0)
    config = SolverConfig(
        max_steps=100_000,
        temperature_rms_tolerance=1.0e-6,
        theta_rms_tolerance=1.0e-6,
        flux_tolerance=5.0e-3,
        enthalpy_drift_tolerance=1.0e-12,
    )
    for n in (25, 50):
        grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, n), g0)
        seed = _superadiabatic_seed(grid)
        state0 = build_column_state(grid, seed, nasa, gravity)
        h0 = column_enthalpy_per_area(state0.mass_path, state0.enthalpy)
        reference = numerical_isentrope(grid, seed, nasa, state0.mass_path)
        result = solve_adaptive_enthalpy(
            grid, seed, physics, nasa, gravity, config
        )
        assert result.status is TerminalStatus.CONVERGED
        final = build_column_state(grid, result.temperature, nasa, gravity)
        h1 = column_enthalpy_per_area(final.mass_path, final.enthalpy)
        scale = max(abs(h0), 1.0)
        assert abs(h1 - h0) / scale <= 1.0e-12
        weights = final.mass_path
        relative = (final.temperature - reference) / reference
        rms = float(np.sqrt(np.sum(weights * relative**2) / np.sum(weights)))
        assert rms <= 1.0e-6


def test_rejected_state_purity_for_full_column():
    nasa = NASAThermo.from_json()
    g0 = 10.0
    gravity = ConstantGravity(g0)
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 25), g0)
    seed = _superadiabatic_seed(grid)
    state = build_column_state(grid, seed, nasa, gravity)
    prior = state.copy()
    physics = PhysicsConfig(gravity=g0, alpha=1.0)
    # Force fractional-temperature rejection with an oversized dt.
    config = SolverConfig(epsilon_temperature=1.0e-12)
    trial = trial_enthalpy_step(grid, state, 1.0e6, physics, config, nasa, gravity)
    assert trial.accepted is False
    assert np.array_equal(state.temperature, prior.temperature)
    assert np.array_equal(state.enthalpy, prior.enthalpy)
    assert np.array_equal(state.density_centres, prior.density_centres)
    assert np.array_equal(state.density_edges, prior.density_edges)
    assert np.array_equal(state.z_centres, prior.z_centres)
    assert np.array_equal(state.z_edges, prior.z_edges)
    assert np.array_equal(state.g_centres, prior.g_centres)
    assert np.array_equal(state.g_edges, prior.g_edges)
    assert np.array_equal(state.mass_path, prior.mass_path)
