"""Observational IntegrationTrace must not change enthalpy-solver physics."""

from __future__ import annotations

import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.gravity import ConstantGravity
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import TerminalStatus
from convection_mlt.solvers_enthalpy import solve_adaptive_enthalpy
from convection_mlt.thermodynamics import NASAThermo
from convection_mlt.trace import IntegrationTrace, TraceLevel


def _superadiabatic_seed(grid, t_bottom: float = 4000.0, nabla: float = 0.30):
    return t_bottom * (grid.pressure_centres / grid.pressure_centres[0]) ** nabla


def test_traced_and_untraced_enthalpy_solves_are_identical():
    nasa = NASAThermo.from_json()
    g0 = 10.0
    gravity = ConstantGravity(g0)
    physics = PhysicsConfig(gravity=g0, alpha=1.0)
    config = SolverConfig(
        max_steps=50_000,
        temperature_rms_tolerance=1.0e-6,
        theta_rms_tolerance=1.0e-6,
        flux_tolerance=5.0e-3,
        enthalpy_drift_tolerance=1.0e-12,
    )
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 25), g0)
    seed = _superadiabatic_seed(grid)

    untraced = solve_adaptive_enthalpy(grid, seed, physics, nasa, gravity, config)
    summary = IntegrationTrace(level=TraceLevel.SUMMARY, summary_stride=50)
    traced_summary = solve_adaptive_enthalpy(
        grid, seed, physics, nasa, gravity, config, trace=summary
    )
    profiles = IntegrationTrace(level=TraceLevel.PROFILES, summary_stride=50)
    traced_profiles = solve_adaptive_enthalpy(
        grid, seed, physics, nasa, gravity, config, trace=profiles
    )

    for traced in (traced_summary, traced_profiles):
        assert traced.status is untraced.status is TerminalStatus.CONVERGED
        assert traced.steps == untraced.steps
        assert np.array_equal(traced.temperature, untraced.temperature)
        assert traced.metrics.enthalpy_drift == untraced.metrics.enthalpy_drift
        assert traced.metrics.temperature_rms == untraced.metrics.temperature_rms
        assert traced.metrics.max_superadiabaticity == untraced.metrics.max_superadiabaticity
        assert traced.metrics.convective_flux_max == untraced.metrics.convective_flux_max

    assert summary.accepted_steps[0].accepted_step == 0
    assert summary.accepted_steps[-1].accepted_step == untraced.steps
    assert summary.final_flux is not None
    assert profiles.profiles[0].accepted_step == 0
    assert profiles.profiles[-1].accepted_step == untraced.steps
    assert len(profiles.profiles) == 2
