import numpy as np
import pytest
 
from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import mixing_region_labels
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import (
    SolverFailure,
    TerminalStatus,
    fixed_step,
    solve_adaptive,
)
from convection_mlt.thermodynamics import IdealH2


def unstable_case():
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 10), 15.0)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.45
    return grid, temperature


def localized_case():
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 10), 15.0)
    temperature = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** gas.nabla_ad
    temperature[3:7] *= np.linspace(1.05, 0.95, 4)
    labels = mixing_region_labels(
        grid, temperature, gas.nabla_ad, 1.0e-7
    )
    return grid, temperature, labels


def test_failed_fixed_step_preserves_state_and_requested_dt():
    grid, temperature = unstable_case()
    requested_dt = 400.0
    result = fixed_step(
        grid,
        temperature,
        requested_dt,
        PhysicsConfig(alpha=1.0),
        SolverConfig(epsilon_temperature=1.0e300),
    )
    assert not result.accepted
    assert result.dt == requested_dt
    assert result.reason is not None and "crossed" in result.reason
    assert np.array_equal(result.temperature, temperature)
    assert not np.array_equal(result.trial_temperature, temperature)


def test_backtracking_limit_raises_typed_failure_with_unchanged_state():
    grid, temperature, labels = localized_case()
    assert np.unique(labels).size > 1
    settings = SolverConfig(
        c_diff=1.0e30,
        epsilon_temperature=1.0e300,
        max_rejections=0,
        max_steps=2,
    )
    with pytest.raises(SolverFailure) as caught:
        solve_adaptive(
            grid,
            temperature,
            PhysicsConfig(alpha=1.0),
            settings,
            region_labels=labels,
        )
    result = caught.value.result
    assert result.status is TerminalStatus.FAILED
    assert "backtracking failed" in result.reason
    assert result.rejections == 1
    assert np.array_equal(result.temperature, temperature)
    assert np.array_equal(result.region_labels, labels)
    assert np.count_nonzero(result.cumulative_unmerged_transfer) == 0


def test_rejected_active_interface_does_not_commit_candidate_merge():
    """Forced backtracking failure must retain original separated labels."""
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e4, 2), 15.0)
    temperature = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** (gas.nabla_ad + 0.1)
    labels = np.array([0, 1])
    settings = SolverConfig(
        c_diff=1.0e30,
        epsilon_temperature=1.0e300,
        max_rejections=0,
        max_steps=2,
    )
    with pytest.raises(SolverFailure) as caught:
        solve_adaptive(
            grid,
            temperature,
            PhysicsConfig(alpha=1.0),
            settings,
            gas,
            labels,
        )
    result = caught.value.result
    assert result.status is TerminalStatus.FAILED
    assert np.array_equal(result.temperature, temperature)
    assert np.array_equal(result.region_labels, labels)
    assert np.count_nonzero(result.cumulative_unmerged_transfer) == 0


def test_alpha_zero_superadiabatic_does_not_merge_separated_labels():
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e4, 2), 15.0)
    temperature = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** (gas.nabla_ad + 0.1)
    labels = np.array([0, 1])
    result = solve_adaptive(
        grid,
        temperature,
        PhysicsConfig(alpha=0.0),
        region_labels=labels,
    )
    assert result.status is TerminalStatus.NO_ACTIVE_CONVECTION
    assert np.array_equal(result.temperature, temperature)
    assert np.array_equal(result.region_labels, labels)
    assert np.count_nonzero(result.cumulative_unmerged_transfer) == 0


def test_backtracking_reduces_step_until_one_is_accepted():
    grid, temperature = unstable_case()
    settings = SolverConfig(
        c_diff=10.0,
        epsilon_temperature=0.5,
        max_rejections=50,
        max_steps=1,
    )
    # max_steps=1 deliberately terminates immediately after the accepted step,
    # exposing its state and rejection count in the typed terminal result.
    with pytest.raises(SolverFailure) as caught:
        solve_adaptive(
            grid, temperature, PhysicsConfig(alpha=1.0), settings
        )
    result = caught.value.result
    assert result.reason == "maximum accepted-step limit reached"
    assert result.rejections > 0
    assert not np.array_equal(result.temperature, temperature)
    assert result.final_dt is not None


def test_subthreshold_flux_triggers_merge_when_transfer_becomes_appreciable():
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e4, 2), 15.0)
    exponent = gas.nabla_ad + 5.0e-8
    initial = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** exponent
    labels = np.array([0, 1])
    settings = SolverConfig(
        transfer_merge_tolerance=1.0e-20,
        max_steps=100,
    )
    result = solve_adaptive(
        grid,
        initial,
        PhysicsConfig(alpha=1.0),
        settings,
        gas,
        labels,
    )
    assert np.array_equal(result.region_labels, np.array([0, 0]))
    assert result.cumulative_unmerged_transfer[1] > 0.0
    assert result.max_unmerged_transfer_fraction == 0.0


def test_pre_step_active_interface_merges_even_if_accepted_state_is_neutral():
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e4, 2), 15.0)
    initial = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** (gas.nabla_ad + 1.0e-5)
    settings = SolverConfig(
        c_diff=0.516,
        epsilon_temperature=0.001,
        max_steps=1,
    )
    try:
        result = solve_adaptive(
            grid,
            initial,
            PhysicsConfig(alpha=1.0),
            settings,
            gas,
            np.array([0, 1]),
        )
    except SolverFailure as error:
        result = error.result
    final_gradient = (
        np.log(result.temperature[0]) - np.log(result.temperature[1])
    ) / (
        np.log(grid.pressure_centres[0])
        - np.log(grid.pressure_centres[1])
    )
    assert final_gradient - gas.nabla_ad <= (
        settings.c_active * settings.epsilon_gradient
    )
    assert np.array_equal(result.region_labels, np.array([0, 0]))
