"""Trace API must be observational: no physics or status changes."""

import numpy as np
import pytest

from convection_mlt.closure import mixing_length_flux
from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import mixing_region_labels
from convection_mlt.energy import temperature_tendency
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import SolverFailure, TerminalStatus, solve_adaptive, _trial_step
from convection_mlt.thermodynamics import IdealH2
from convection_mlt.trace import TraceLevel, make_trace


def _superadiabatic_case(n_layers: int = 20, alpha: float = 1.0):
    physics = PhysicsConfig(alpha=alpha)
    grid = build_grid(
        log_pressure_edges(1.0e7, 1.0e3, n_layers), physics.gravity
    )
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.35
    return physics, grid, temperature


def test_trace_none_returns_none_and_matches_untraced_solution():
    physics, grid, temperature = _superadiabatic_case()
    solver = SolverConfig()
    plain = solve_adaptive(grid, temperature, physics, solver)
    traced = solve_adaptive(
        grid, temperature, physics, solver, trace=make_trace(TraceLevel.NONE)
    )
    assert make_trace(TraceLevel.NONE) is None
    assert plain.status == traced.status
    assert plain.steps == traced.steps
    assert plain.rejections == traced.rejections
    np.testing.assert_allclose(plain.temperature, traced.temperature)


def test_summary_trace_does_not_alter_solution_or_status():
    physics, grid, temperature = _superadiabatic_case()
    solver = SolverConfig()
    plain = solve_adaptive(grid, temperature, physics, solver)
    trace = make_trace(TraceLevel.SUMMARY)
    traced = solve_adaptive(grid, temperature, physics, solver, trace=trace)
    assert plain.status == traced.status
    assert plain.reason == traced.reason
    assert plain.steps == traced.steps
    assert plain.rejections == traced.rejections
    np.testing.assert_allclose(plain.temperature, traced.temperature)
    assert trace.initial_temperature is not None
    assert trace.final_temperature is not None
    assert trace.totals["accepted_steps"] == traced.steps
    assert "max_abs_enthalpy_drift" in trace.extrema
    assert len(trace.trials) == 0
    assert len(trace.profiles) == 0


def test_profiles_retain_initial_final_and_decimated_targets():
    physics, grid, temperature = _superadiabatic_case(n_layers=25)
    solver = SolverConfig()
    trace = make_trace(TraceLevel.PROFILES)
    result = solve_adaptive(grid, temperature, physics, solver, trace=trace)
    assert result.status is TerminalStatus.CONVERGED
    assert len(trace.theta_rms_targets) == 8
    assert trace.profiles[0].accepted_step == 0
    assert np.isclose(trace.profiles[0].simulated_time, 0.0)
    assert trace.profiles[-1].accepted_step == result.steps
    np.testing.assert_allclose(trace.final_temperature, result.temperature)
    hit_steps = [item.accepted_step for item in trace.profiles[1:-1]]
    assert hit_steps == sorted(hit_steps)


def test_rejected_trials_do_not_create_accepted_snapshots_or_mutate_labels():
    gas = IdealH2()
    physics = PhysicsConfig(alpha=1.0)
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 10), physics.gravity)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** gas.nabla_ad
    temperature = temperature.copy()
    temperature[3:7] *= np.linspace(1.05, 0.95, 4)
    labels = mixing_region_labels(
        grid, temperature, gas.nabla_ad, 1.0e-7
    )
    assert np.unique(labels).size > 1
    labels_before = labels.copy()
    temperature_before = temperature.copy()
    solver = SolverConfig(
        c_diff=1.0e30,
        epsilon_temperature=1.0e300,
        max_rejections=0,
        max_steps=2,
    )
    trace = make_trace(TraceLevel.TRIALS)
    with pytest.raises(SolverFailure) as caught:
        solve_adaptive(
            grid,
            temperature,
            physics,
            solver,
            region_labels=labels,
            trace=trace,
        )
    result = caught.value.result
    rejected = [item for item in trace.trials if not item.accepted]
    assert rejected
    assert result.steps == 0
    np.testing.assert_array_equal(result.region_labels, labels_before)
    np.testing.assert_allclose(result.temperature, temperature_before)
    assert np.all(result.cumulative_unmerged_transfer == 0.0)
    assert all(item.accepted_step == 0 for item in rejected)
    assert all(snap.accepted_step == 0 for snap in trace.profiles)


def test_trial_overshoot_margin_matches_hysteresis_calculation():
    physics, grid, temperature = _superadiabatic_case(n_layers=15)
    gas = IdealH2()
    solver = SolverConfig(epsilon_temperature=1.0e300)
    trial = _trial_step(grid, temperature, 50.0, physics, solver, gas)
    assert trial.min_active_trial_delta_over_epsilon is not None
    tendency = temperature_tendency(grid, trial.closure.flux, gas.cp)
    trial_temperature = temperature + 50.0 * tendency
    recomputed = mixing_length_flux(
        grid, trial_temperature, physics.gravity, physics.alpha, gas
    )
    old_delta = trial.closure.gradient[1:-1] - gas.nabla_ad
    trial_delta = recomputed.gradient[1:-1] - gas.nabla_ad
    active = old_delta > solver.c_active * solver.epsilon_gradient
    expected = float(np.min(trial_delta[active] / solver.epsilon_gradient))
    assert np.isclose(trial.min_active_trial_delta_over_epsilon, expected)
    crossed = expected < -solver.c_cross
    if crossed:
        assert trial.reason is not None
        assert "crossed the neutral hysteresis band" in trial.reason
    else:
        assert trial.accepted or (
            trial.reason is not None
            and "crossed" not in trial.reason
        )


def test_summary_stride_keeps_extrema_while_decimating_history():
    physics, grid, temperature = _superadiabatic_case(n_layers=20)
    solver = SolverConfig()
    trace = make_trace(TraceLevel.SUMMARY)
    assert trace is not None
    trace.summary_stride = 5
    result = solve_adaptive(grid, temperature, physics, solver, trace=trace)
    assert result.steps >= 1
    kept = {item.accepted_step for item in trace.accepted_steps}
    assert 1 in kept
    assert all(step % 5 == 0 or step == 1 for step in kept)
    assert np.isfinite(trace.extrema["max_abs_enthalpy_drift"])
    assert trace.totals["accepted_steps"] == result.steps


def test_summary_history_is_bounded_and_adaptively_decimated():
    physics, grid, temperature = _superadiabatic_case(n_layers=20)
    trace = make_trace(TraceLevel.SUMMARY)
    assert trace is not None
    trace.max_summary_records = 16
    result = solve_adaptive(
        grid, temperature, physics, SolverConfig(), trace=trace
    )
    assert result.steps > trace.max_summary_records
    assert len(trace.accepted_steps) <= trace.max_summary_records
    assert trace.summary_stride > 1
    assert trace.accepted_steps[0].accepted_step == 1
    assert trace.totals["accepted_steps"] == result.steps
