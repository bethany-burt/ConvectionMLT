import numpy as np
import pytest

from convection_mlt.closure import mixing_length_flux
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.thermodynamics import IdealH2


def unstable_column():
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, 8), 15.0)
    temperature = 1200.0 * (grid.pressure_centres / 1.0e5) ** 0.35
    return grid, temperature


def test_shapes_boundaries_and_upward_flux():
    grid, temperature = unstable_column()
    result = mixing_length_flux(grid, temperature, 15.0, 1.0)
    for values in (
        result.gradient,
        result.flux,
        result.velocity,
        result.mixing_length,
        result.kzz,
    ):
        assert values.shape == (grid.n_layers + 1,)
    assert result.flux[0] == result.flux[-1] == 0.0
    assert result.velocity[0] == result.velocity[-1] == 0.0
    assert result.kzz[0] == result.kzz[-1] == 0.0
    assert result.mixing_length[0] == result.mixing_length[-1] == 0.0
    assert not result.mixing_length_applicable[[0, -1]].any()
    assert np.all(result.flux[1:-1] > 0.0)


def test_alpha_scaling_and_explicit_zero_case():
    grid, temperature = unstable_column()
    one = mixing_length_flux(grid, temperature, 15.0, 1.0)
    two = mixing_length_flux(grid, temperature, 15.0, 2.0)
    assert np.allclose(two.velocity[1:-1], 2.0 * one.velocity[1:-1])
    assert np.allclose(two.flux[1:-1], 4.0 * one.flux[1:-1])
    assert np.allclose(two.kzz[1:-1], 4.0 * one.kzz[1:-1])
    zero = mixing_length_flux(grid, temperature, 15.0, 0.0)
    assert np.count_nonzero(zero.flux) == 0
    assert np.count_nonzero(zero.velocity) == 0
    assert np.count_nonzero(zero.kzz) == 0


def test_manufactured_single_interface_matches_hand_calculation():
    gravity = 15.0
    alpha = 0.8
    prefactor = 0.5
    gas = IdealH2()
    grid = build_grid(np.array([1.0e6, 1.0e5, 1.0e4]), gravity)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** 0.4
    result = mixing_length_flux(
        grid, temperature, gravity, alpha, gas, prefactor
    )
    j = 1
    gradient = 0.4
    delta = gradient - gas.nabla_ad
    t_edge = 1000.0
    rho = grid.pressure_edges[j] / (gas.gas_constant * t_edge)
    hp = gas.gas_constant * t_edge / gravity
    ell = alpha * hp
    velocity = ell * np.sqrt(gravity / hp * delta)
    flux = (
        prefactor
        * rho
        * gas.cp
        * velocity
        * t_edge
        * alpha
        * delta
    )
    assert result.gradient[j] == pytest.approx(gradient)
    assert result.mixing_length[j] == pytest.approx(ell)
    assert result.velocity[j] == pytest.approx(velocity)
    assert result.flux[j] == pytest.approx(flux)
    assert result.kzz[j] == pytest.approx(velocity * ell)
    assert result.thermal_diffusivity[j] == pytest.approx(
        prefactor * velocity * ell
    )


def test_exact_adiabat_has_roundoff_level_flux_on_irregular_grid():
    gas = IdealH2()
    edges = np.array([1.0e7, 4.0e6, 7.0e5, 1.1e5, 2.0e4, 1.0e3])
    grid = build_grid(edges, 15.0)
    temperature = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** gas.nabla_ad
    result = mixing_length_flux(grid, temperature, 15.0, 1.0, gas)
    assert np.max(np.abs(result.gradient[1:-1] - gas.nabla_ad)) < 2.0e-15
    assert np.max(np.abs(result.flux)) < 1.0e-12


@pytest.mark.parametrize(
    ("gravity", "alpha"),
    [(0.0, 1.0), (-1.0, 1.0), (15.0, -0.1), (np.nan, 1.0)],
)
def test_invalid_physics_inputs_raise(gravity, alpha):
    grid, temperature = unstable_column()
    with pytest.raises(ValueError):
        mixing_length_flux(grid, temperature, gravity, alpha)
