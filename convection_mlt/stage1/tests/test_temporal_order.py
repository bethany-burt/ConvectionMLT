import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers import fixed_step


def integrate(dt: float, final_time: float):
    physics = PhysicsConfig(alpha=1.0)
    settings = SolverConfig(epsilon_temperature=0.5)
    grid = build_grid(
        log_pressure_edges(1.0e7, 1.0e3, 16), physics.gravity
    )
    state = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.35
    for _ in range(round(final_time / dt)):
        outcome = fixed_step(grid, state, dt, physics, settings)
        assert outcome.accepted
        state = outcome.temperature
    return grid, state


def mass_weighted_relative_rms(grid, state, reference):
    relative = (state - reference) / reference
    return np.sqrt(
        np.sum(grid.layer_mass * relative**2) / np.sum(grid.layer_mass)
    )


def test_forward_euler_is_first_order_at_common_pre_equilibrium_time():
    final_time = 8.0
    grid, reference = integrate(1.0 / 64.0, final_time)
    errors = []
    for dt in (1.0, 0.5, 0.25, 0.125):
        _, state = integrate(dt, final_time)
        errors.append(mass_weighted_relative_rms(grid, state, reference))
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert np.all(ratios > 1.7)
    assert np.all(ratios < 2.5)
