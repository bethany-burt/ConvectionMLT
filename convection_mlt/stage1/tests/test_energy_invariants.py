import numpy as np

from convection_mlt.closure import mixing_length_flux
from convection_mlt.energy import telescoping_residual, temperature_tendency
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.thermodynamics import IdealH2


def test_single_interface_update_has_correct_sign_and_conserves_enthalpy():
    gravity = 15.0
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e4, 2), gravity)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.40
    closure = mixing_length_flux(grid, temperature, gravity, 1.0, gas)
    tendency = temperature_tendency(grid, closure.flux, gas.cp)
    assert closure.flux[1] > 0.0
    assert tendency[0] < 0.0  # lower layer cools
    assert tendency[1] > 0.0  # upper layer heats
    enthalpy_tendencies = gas.cp * grid.layer_mass * tendency
    assert np.isclose(
        enthalpy_tendencies[0],
        -enthalpy_tendencies[1],
        rtol=2.0e-15,
    )


def test_discrete_tendency_telescopes_to_boundary_flux():
    gravity = 15.0
    gas = IdealH2()
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 20), gravity)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.37
    closure = mixing_length_flux(grid, temperature, gravity, 0.7, gas)
    tendency = temperature_tendency(grid, closure.flux, gas.cp)
    residual = telescoping_residual(
        grid, tendency, gas.cp, closure.flux[0], closure.flux[-1]
    )
    scale = np.sum(np.abs(gas.cp * grid.layer_mass * tendency))
    assert abs(residual) <= 5.0 * np.finfo(float).eps * scale
