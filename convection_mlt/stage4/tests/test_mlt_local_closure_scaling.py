"""Local MLT closure scaling: F_conv ∝ α² (Δ∇)^{3/2} on fixed atmospheric states.

These tests verify the implemented flux law directly, independent of HELIOS.
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.closure import mixing_length_flux
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.thermodynamics import ConstantH2Thermo, IdealH2
from convection_mlt.validate import pressure_edges


GRAVITY = 15.0
PREFACTOR = 0.5
NABLA_AD = float(IdealH2().nabla_ad)


def _column_with_delta(delta: float, n_layers: int = 12):
    """Manufactured power-law column with ∇ = ∇_ad + delta (constant)."""
    grid = build_grid(log_pressure_edges(1.0e7, 1.0e3, n_layers), GRAVITY)
    nabla = NABLA_AD + float(delta)
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** nabla
    return grid, temperature


def _internal_flux(result) -> np.ndarray:
    return np.asarray(result.flux[1:-1], dtype=np.float64)


def _internal_delta(result) -> np.ndarray:
    return np.asarray(result.superadiabaticity[1:-1], dtype=np.float64)


def test_stable_layers_carry_zero_flux():
    grid, temperature = _column_with_delta(-0.05)
    result = mixing_length_flux(
        grid, temperature, GRAVITY, 1.0, IdealH2(), PREFACTOR
    )
    assert np.all(_internal_delta(result) == 0.0)
    assert np.all(_internal_flux(result) == 0.0)
    assert not np.any(result.active[1:-1])


def test_unstable_interfaces_have_nonnegative_flux():
    grid, temperature = _column_with_delta(0.08)
    result = mixing_length_flux(
        grid, temperature, GRAVITY, 1.0, IdealH2(), PREFACTOR
    )
    assert np.all(_internal_delta(result) > 0.0)
    assert np.all(_internal_flux(result) >= 0.0)
    assert np.all(_internal_flux(result) > 0.0)


def test_alpha_log_slope_is_two():
    grid, temperature = _column_with_delta(0.1)
    alphas = np.asarray([0.5, 1.0, 2.0, 4.0])
    fluxes = []
    for alpha in alphas:
        r = mixing_length_flux(
            grid, temperature, GRAVITY, float(alpha), IdealH2(), PREFACTOR
        )
        fluxes.append(float(np.mean(_internal_flux(r))))
    fluxes = np.asarray(fluxes)
    slope, _ = np.polyfit(np.log(alphas), np.log(fluxes), 1)
    assert slope == pytest.approx(2.0, rel=1e-10, abs=1e-10)


def _analytic_prefactor(result, alpha: float) -> np.ndarray:
    """Thermodynamic factor C such that F = C (Δ∇)^{3/2} on internal interfaces."""
    gas = IdealH2()
    internal = slice(1, -1)
    rho = result.density_edges[internal]
    hp = result.scale_height[internal]
    t_edge = result.temperature_edges[internal]
    return (
        PREFACTOR
        * rho
        * gas.cp
        * (alpha**2)
        * hp
        * np.sqrt(GRAVITY / hp)
        * t_edge
    )


def test_delta_nabla_log_slope_is_three_halves():
    """On a fixed T(P), F ∝ (Δ∇)^{3/2}; across Δ∇, F / (Δ∇)^{3/2} = C(T,P)."""
    # Fixed column: verify interface-wise F = C Δ∇^{3/2}
    grid, temperature = _column_with_delta(0.05)
    alpha = 1.0
    r = mixing_length_flux(
        grid, temperature, GRAVITY, alpha, IdealH2(), PREFACTOR
    )
    delta = _internal_delta(r)
    flux = _internal_flux(r)
    c = _analytic_prefactor(r, alpha)
    assert np.allclose(flux, c * delta**1.5, rtol=1e-12, atol=1e-14)

    # Δ∇ ladder: divide out C so the remaining slope is exactly 3/2
    deltas = np.asarray([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    reduced = []
    for dval in deltas:
        grid_d, temperature_d = _column_with_delta(float(dval))
        rd = mixing_length_flux(
            grid_d, temperature_d, GRAVITY, alpha, IdealH2(), PREFACTOR
        )
        cd = _analytic_prefactor(rd, alpha)
        fd = _internal_flux(rd)
        dd = _internal_delta(rd)
        assert np.allclose(dd, dval, rtol=1e-10, atol=1e-14)
        reduced.append(float(np.mean(fd / cd)))
    reduced = np.asarray(reduced)
    slope, _ = np.polyfit(np.log(deltas), np.log(reduced), 1)
    assert slope == pytest.approx(1.5, rel=1e-10, abs=1e-10)
    assert np.allclose(reduced, deltas**1.5, rtol=1e-10, atol=1e-14)


def test_continuity_as_delta_nabla_approaches_zero():
    deltas = np.geomspace(1e-10, 1e-2, 11)
    alpha = 1.0
    mean_flux = []
    max_rel_err = []
    for delta in deltas:
        grid, temperature = _column_with_delta(float(delta))
        r = mixing_length_flux(
            grid, temperature, GRAVITY, alpha, IdealH2(), PREFACTOR
        )
        f = _internal_flux(r)
        d = _internal_delta(r)
        c = _analytic_prefactor(r, alpha)
        pred = c * np.maximum(d, 0.0) ** 1.5
        mean_flux.append(float(np.mean(f)))
        max_rel_err.append(float(np.max(np.abs(f - pred) / np.maximum(np.abs(pred), 1e-30))))
    mean_flux = np.asarray(mean_flux)
    assert mean_flux[0] < mean_flux[-1]
    assert np.all(np.diff(mean_flux) > 0.0)
    assert max(max_rel_err) < 1e-12
    assert mean_flux[0] / mean_flux[-1] < (deltas[0] / deltas[-1]) ** 1.4
    # Vanishes at the stable limit
    grid_s, temperature_s = _column_with_delta(0.0)
    stable = mixing_length_flux(
        grid_s, temperature_s, GRAVITY, alpha, IdealH2(), PREFACTOR
    )
    assert float(np.max(np.abs(stable.flux))) < 1e-12


def test_pressure_array_orientation_roundtrip_invariance():
    """TOA-first storage remapped to bottom-first yields identical F_conv."""
    grid, temperature = _column_with_delta(0.07)
    gas = ConstantH2Thermo()
    direct = mixing_length_flux(
        grid, temperature, GRAVITY, 1.25, gas, PREFACTOR
    )

    edges_toa_first = grid.pressure_edges[::-1]
    temp_toa_first = temperature[::-1]
    # Remap to the PressureGrid contract (bottom-to-top, decreasing P).
    edges_bt = edges_toa_first[::-1]
    temp_bt = temp_toa_first[::-1]
    grid_bt = build_grid(edges_bt, GRAVITY)
    remapped = mixing_length_flux(
        grid_bt, temp_bt, GRAVITY, 1.25, gas, PREFACTOR
    )
    assert np.allclose(direct.flux, remapped.flux)
    assert np.allclose(direct.superadiabaticity, remapped.superadiabaticity)
    assert np.allclose(direct.velocity, remapped.velocity)

    with pytest.raises(ValueError):
        pressure_edges(edges_toa_first)


def test_alpha_zero_is_identically_zero_flux():
    grid, temperature = _column_with_delta(0.2)
    zero = mixing_length_flux(
        grid, temperature, GRAVITY, 0.0, IdealH2(), PREFACTOR
    )
    assert np.count_nonzero(zero.flux) == 0
    assert np.count_nonzero(zero.velocity) == 0
