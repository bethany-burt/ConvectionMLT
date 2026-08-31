"""Targeted H2/He mixture thermodynamic consistency tests."""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.gravity import ConstantGravity
from convection_mlt.radiation import LowerNetInternalFlux, SolveRoute, TopIrradiation
from convection_mlt.rce import (
    _temperature_on_adiabat,
    nested_analytic_opacity_spec,
)
from convection_mlt.steady_rce import evaluate_trial
from convection_mlt.thermodynamics import ConstantH2Thermo, h2_he_mixture


@pytest.mark.parametrize("x_he", [0.1, 0.2])
def test_nabla_ad_matches_isentropic_log_derivative(x_he: float) -> None:
    """(d ln T / d ln P)_s = nabla_ad(T, P, x_He) along an isentrope."""
    thermo = h2_he_mixture(x_he)
    p0, t0 = 1.0e6, 1800.0
    pressures = np.geomspace(p0 * 0.01, p0, 40)
    t_ad = _temperature_on_adiabat(thermo, t0, p0, pressures)
    dlnT = np.diff(np.log(t_ad))
    dlnP = np.diff(np.log(pressures))
    nabla_num = dlnT / dlnP
    t_mid = 0.5 * (t_ad[1:] + t_ad[:-1])
    nabla_th = thermo.nabla_ad_at(t_mid)
    assert np.max(np.abs(nabla_num - nabla_th) / np.maximum(nabla_th, 1.0e-12)) <= 5.0e-4


@pytest.mark.parametrize("x_he", [0.1, 0.2])
def test_isentropic_integration_preserves_entropy(x_he: float) -> None:
    thermo = h2_he_mixture(x_he)
    p0, t0 = 1.0e6, 2000.0
    pressures = np.geomspace(1.0e4, p0, 60)
    t_ad = _temperature_on_adiabat(thermo, t0, p0, pressures)
    s = thermo.entropy(t_ad, pressures)
    s0 = float(s[-1])
    assert np.max(np.abs(s - s0) / max(abs(s0), 1.0)) <= 1.0e-10


@pytest.mark.parametrize("x_he", [0.1, 0.2])
def test_cp_equals_dh_dt_finite_difference(x_he: float) -> None:
    thermo = h2_he_mixture(x_he)
    # Avoid T=1000 K NASA interval breakpoint for H2.
    temps = np.array([400.0, 800.0, 1500.0, 2500.0])
    dT = 1.0e-3
    cp = thermo.specific_heat(temps)
    dh = (thermo.enthalpy(temps + dT) - thermo.enthalpy(temps - dT)) / (2.0 * dT)
    assert np.max(np.abs(dh - cp) / cp) <= 1.0e-8


@pytest.mark.parametrize("x_he", [0.1, 0.2])
def test_invert_enthalpy_matches_implicit_update_path(x_he: float) -> None:
    """Implicit convection uses invert_enthalpy(h); verify round-trip."""
    thermo = h2_he_mixture(x_he)
    temps = np.linspace(1200.0, 2500.0, 32)
    h_spec = thermo.enthalpy(temps)
    t_rec = thermo.invert_enthalpy(h_spec)
    assert np.max(np.abs(t_rec - temps) / temps) <= 1.0e-10


def test_radiative_flux_unchanged_for_composition_independent_opacity() -> None:
    """On identical T(P), radiative flux is unchanged if opacity is composition-independent."""
    from convection_mlt.radiation import solve_radiation

    spec = nested_analytic_opacity_spec(96, alpha=1.0, f_int=300.0, f_irr=120.0)
    grid = spec.grid()
    p = np.asarray(grid.pressure_centres, dtype=np.float64)
    mass = np.asarray(grid.layer_mass, dtype=np.float64)
    t = _temperature_on_adiabat(ConstantH2Thermo(), 2000.0, 1.0e6, p)
    fluxes = {}
    for label in ("h2", "he02", "he01"):
        rad = solve_radiation(
            t,
            mass,
            spec.opacity(),
            p,
            TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            1.0,
            SolveRoute.THOMAS,
            bottom_convective_flux=0.0,
        )
        fluxes[label] = np.asarray(rad.flux_net, dtype=np.float64)
    f_ref = fluxes["h2"]
    for label in ("he02", "he01"):
        rel = np.max(np.abs(fluxes[label] - f_ref) / np.maximum(np.abs(f_ref), 1.0))
        assert rel <= 1.0e-12, label
