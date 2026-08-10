import numpy as np
 
from convection_mlt.config import PhysicsConfig
from convection_mlt.diagnostics import (
    column_enthalpy,
    enthalpy_normalized_adiabat,
    mixing_region_labels,
    piecewise_enthalpy_reference,
    reference_enthalpy_residuals,
)
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import TerminalStatus, fixed_step, solve_adaptive
from convection_mlt.thermodynamics import IdealH2


def superadiabatic_case(n_layers: int = 10):
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, n_layers), 15.0)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.35
    return grid, temperature


def test_connected_column_reaches_own_enthalpy_normalized_adiabat():
    gas = IdealH2()
    grid, initial = superadiabatic_case()
    reference = enthalpy_normalized_adiabat(
        grid, initial, gas.cp, gas.nabla_ad
    )
    result = solve_adaptive(grid, initial, PhysicsConfig(alpha=1.0))
    assert result.status is TerminalStatus.CONVERGED
    assert np.max(np.abs((result.temperature - reference) / reference)) < 1.0e-7
    assert result.metrics.enthalpy_drift < 1.0e-10


def test_positive_alpha_changes_rate_not_equilibrium():
    grid, initial = superadiabatic_case()
    slow = solve_adaptive(grid, initial, PhysicsConfig(alpha=0.5))
    fast = solve_adaptive(grid, initial, PhysicsConfig(alpha=2.0))
    assert slow.status is fast.status is TerminalStatus.CONVERGED
    assert np.allclose(slow.temperature, fast.temperature, rtol=1.0e-7)
    assert slow.simulated_time != fast.simulated_time


def test_fixed_steps_accumulate_negligible_enthalpy_drift():
    gas = IdealH2()
    grid, initial = superadiabatic_case()
    state = initial.copy()
    initial_enthalpy = column_enthalpy(grid, state, gas.cp)
    for _ in range(200):
        outcome = fixed_step(grid, state, 0.1, PhysicsConfig(alpha=1.0))
        assert outcome.accepted
        state = outcome.temperature
    drift = abs(
        column_enthalpy(grid, state, gas.cp) - initial_enthalpy
    ) / initial_enthalpy
    assert drift < 1.0e-12


def test_localized_case_uses_piecewise_enthalpy_references():
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 20), 15.0)
    initial = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** gas.nabla_ad
    initial[8:12] *= np.linspace(1.06, 0.94, 4)
    labels = mixing_region_labels(
        grid, initial, gas.nabla_ad, 1.0e-7
    )
    result = solve_adaptive(
        grid,
        initial,
        PhysicsConfig(alpha=1.0),
        region_labels=labels,
    )
    assert result.status is TerminalStatus.CONVERGED
    assert np.unique(result.region_labels).size < np.unique(labels).size
    assert result.metrics.temperature_max < 1.0e-7
    assert result.metrics.enthalpy_drift < 1.0e-10


def test_permanently_stable_barriers_remain_unmerged():
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 8), 15.0)
    initial = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.15
    labels = mixing_region_labels(
        grid, initial, gas.nabla_ad, 1.0e-7
    )
    result = solve_adaptive(
        grid,
        initial,
        PhysicsConfig(alpha=1.0),
        region_labels=labels,
    )
    assert result.status is TerminalStatus.NO_ACTIVE_CONVECTION
    assert np.array_equal(result.region_labels, labels)
    assert result.max_unmerged_transfer_fraction == 0.0


def test_piecewise_reference_conserves_each_merged_region_to_roundoff():
    gas = IdealH2()
    grid, initial = superadiabatic_case(8)
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    reference = piecewise_enthalpy_reference(
        grid, initial, gas.cp, gas.nabla_ad, labels
    )
    residuals = reference_enthalpy_residuals(
        grid, initial, reference, gas.cp, labels
    )
    assert max(residuals.values()) < 64.0 * np.finfo(float).eps
