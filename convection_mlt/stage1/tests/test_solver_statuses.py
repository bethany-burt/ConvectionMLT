import numpy as np
 
from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import ConvergenceMetrics
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import TerminalStatus, adaptive_timestep, solve_adaptive
from convection_mlt.thermodynamics import IdealH2
from convection_mlt.closure import mixing_length_flux
from convection_mlt.energy import temperature_tendency


def grid_and_power(exponent: float, n_layers: int = 12):
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, n_layers), 15.0)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** exponent
    return grid, temperature


def test_exact_adiabat_is_converged_before_zero_shortcut():
    gas = IdealH2()
    grid, temperature = grid_and_power(gas.nabla_ad)
    closure = mixing_length_flux(grid, temperature, 15.0, 1.0, gas)
    # This grid produces a tiny positive roundoff-level flux at some edges.
    assert np.count_nonzero(closure.flux) > 0
    result = solve_adaptive(grid, temperature, PhysicsConfig(alpha=1.0))
    assert result.status is TerminalStatus.CONVERGED


def test_stable_nonadiabatic_profile_has_no_active_convection():
    grid, temperature = grid_and_power(0.15)
    result = solve_adaptive(grid, temperature, PhysicsConfig(alpha=1.0))
    assert result.status is TerminalStatus.NO_ACTIVE_CONVECTION
    assert np.array_equal(result.temperature, temperature)


def test_alpha_zero_is_stationary_not_falsely_converged():
    grid, temperature = grid_and_power(0.40)
    result = solve_adaptive(grid, temperature, PhysicsConfig(alpha=0.0))
    assert result.status is TerminalStatus.NO_ACTIVE_CONVECTION
    assert np.array_equal(result.temperature, temperature)


def test_inactive_adaptive_bounds_are_infinite():
    grid, temperature = grid_and_power(0.15)
    physics = PhysicsConfig(alpha=1.0)
    gas = IdealH2()
    closure = mixing_length_flux(
        grid, temperature, physics.gravity, physics.alpha, gas
    )
    tendency = temperature_tendency(grid, closure.flux, gas.cp)
    dt_diff, dt_temperature = adaptive_timestep(
        grid,
        temperature,
        closure,
        tendency,
        physics,
        SolverConfig(),
        gas,
    )
    assert np.isinf(dt_diff)
    assert np.isinf(dt_temperature)


def test_active_adaptive_bounds_match_defined_formulas():
    grid, temperature = grid_and_power(0.40)
    physics = PhysicsConfig(alpha=1.0)
    settings = SolverConfig()
    gas = IdealH2()
    closure = mixing_length_flux(
        grid, temperature, physics.gravity, physics.alpha, gas
    )
    tendency = temperature_tendency(grid, closure.flux, gas.cp)
    dt_diff, dt_temperature = adaptive_timestep(
        grid,
        temperature,
        closure,
        tendency,
        physics,
        settings,
        gas,
    )
    dz = (
        gas.gas_constant
        * temperature
        / physics.gravity
        * np.log(grid.pressure_edges[:-1] / grid.pressure_edges[1:])
    )
    adjacent = np.maximum(
        0.5 * closure.kzz[:-1], 0.5 * closure.kzz[1:]
    )
    expected_diff = settings.c_diff * np.min(dz**2 / adjacent)
    expected_temperature = settings.epsilon_temperature * np.min(
        temperature[np.abs(tendency) > 0.0]
        / np.abs(tendency[np.abs(tendency) > 0.0])
    )
    assert dt_diff == expected_diff
    assert dt_temperature == expected_temperature


def test_reference_agreement_cannot_override_large_convective_flux():
    settings = SolverConfig()
    metrics = ConvergenceMetrics(
        max_superadiabaticity=0.0,
        potential_temperature_rms=0.0,
        temperature_rms=0.0,
        temperature_max=0.0,
        normalized_tendency_max=0.0,
        convective_flux_max=10.0 * settings.flux_tolerance,
        enthalpy_drift=0.0,
    )
    assert not metrics.converged(settings)
