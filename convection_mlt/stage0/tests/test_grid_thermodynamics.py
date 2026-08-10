import numpy as np
import pytest

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.grid import (
    build_grid,
    hydrostatic_layer_thickness,
    interpolate_temperature_to_internal_edges,
)
from convection_mlt.thermodynamics import IdealH2, R_UNIVERSAL


def test_calorically_perfect_h2_properties():
    gas = IdealH2()
    assert gas.cp == pytest.approx(3.5 * R_UNIVERSAL / gas.molar_mass)
    assert gas.nabla_ad == pytest.approx(2.0 / 7.0)


def test_bottom_to_top_grid_geometry_mass_and_height():
    edges = np.array([1.0e6, 2.0e5, 1.0e4])
    gravity = 10.0
    grid = build_grid(edges, gravity)
    assert np.allclose(grid.pressure_centres, np.sqrt(edges[:-1] * edges[1:]))
    assert np.allclose(grid.layer_mass, (edges[:-1] - edges[1:]) / gravity)
    temperature = np.array([1000.0, 700.0])
    gas = IdealH2()
    expected = (
        gas.gas_constant
        * temperature
        / gravity
        * np.log(edges[:-1] / edges[1:])
    )
    assert np.allclose(
        hydrostatic_layer_thickness(
            grid, temperature, gas.gas_constant, gravity
        ),
        expected,
    )


def test_interface_interpolation_preserves_power_law_on_irregular_grid():
    edges = np.array([1.0e7, 3.0e6, 4.0e5, 8.0e4, 1.0e3])
    grid = build_grid(edges, 15.0)
    exponent = 0.217
    temperature = 900.0 * (grid.pressure_centres / 1.0e5) ** exponent
    expected = 900.0 * (edges[1:-1] / 1.0e5) ** exponent
    assert np.allclose(
        interpolate_temperature_to_internal_edges(grid, temperature),
        expected,
        rtol=2.0e-15,
    )


@pytest.mark.parametrize(
    "edges",
    [
        [1.0e5, 2.0e5],
        [1.0e5, 1.0e5],
        [1.0e5, -1.0],
        [1.0e5, np.nan],
    ],
)
def test_invalid_pressure_edges_raise(edges):
    with pytest.raises(ValueError):
        build_grid(edges, 10.0)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: PhysicsConfig(alpha=-1.0),
        lambda: PhysicsConfig(gravity=0.0),
        lambda: SolverConfig(f_back=1.0),
        lambda: SolverConfig(dt_min=0.0),
        lambda: SolverConfig(max_rejections=-1),
    ],
)
def test_invalid_configuration_raises(constructor):
    with pytest.raises(ValueError):
        constructor()
