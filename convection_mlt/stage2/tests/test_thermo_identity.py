"""Stage 2 thermodynamic identity and NASA regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from convection_mlt.thermodynamics import (
    AnalyticIdealGasThermo,
    ConstantH2Thermo,
    NASAThermo,
    ThermoDomainError,
    analytic_h2_oracle,
    h2_he_mixture,
    monatomic_helium,
)


def _nasa() -> NASAThermo:
    return NASAThermo.from_json()


def test_constant_h2_legacy_scalars():
    gas = ConstantH2Thermo()
    assert gas.cp == pytest.approx(3.5 * gas.gas_constant)
    assert gas.nabla_ad == pytest.approx(2.0 / 7.0)
    assert float(gas.enthalpy(gas.t_ref)) == pytest.approx(0.0)


@pytest.mark.parametrize("provider_factory", [_nasa, analytic_h2_oracle, monatomic_helium])
def test_t_h_t_roundtrip(provider_factory):
    thermo = provider_factory()
    temps = np.linspace(thermo.t_min * 1.01, thermo.t_max * 0.99, 40)
    # Stay away from hard endpoints for Newton/clipping margin.
    recovered = thermo.invert_enthalpy(thermo.enthalpy(temps))
    assert np.max(np.abs(recovered / temps - 1.0)) <= 1.0e-12


def test_nasa_interval_coverage_and_breakpoints():
    nasa = _nasa()
    assert nasa.t_min == pytest.approx(200.0)
    assert nasa.t_max == pytest.approx(6000.0)
    # Every checked-in interval interior plus breakpoint ±ε.
    samples = [250.0, 500.0, 999.0, 1000.0, 1000.0 + 1.0e-6, 2000.0, 5000.0]
    for temperature in samples:
        cp = float(nasa.specific_heat(temperature))
        h = float(nasa.enthalpy(temperature))
        assert cp > 0.0
        assert np.isfinite(h)

    eps = 1.0e-6
    for break_t in [1000.0]:
        cp_lo = float(nasa.specific_heat(break_t - eps))
        cp_hi = float(nasa.specific_heat(break_t + eps))
        h_lo = float(nasa.enthalpy(break_t - eps))
        h_hi = float(nasa.enthalpy(break_t + eps))
        assert abs(cp_hi - cp_lo) / max(abs(cp_lo), 1.0) <= 2.0e-9
        assert abs(h_hi - h_lo) / max(abs(h_lo), 1.0) <= 2.0e-9
        # Checked-in TPIS78 NASA7 intervals are continuous to ~2e-9 relative.


def test_nasa_golden_values_from_checked_in_source():
    nasa = _nasa()
    data_path = Path(__file__).resolve().parents[2] / "src" / "convection_mlt" / "thermo_data" / "h2_nasa7_tpis78.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    for point in payload["golden_values"]["points"]:
        temperature = float(point["T_k"])
        assert float(nasa.specific_heat(temperature)) == pytest.approx(
            float(point["cp_J_per_kg_K"]), rel=0.0, abs=0.0
        )
        assert float(nasa.enthalpy(temperature)) == pytest.approx(
            float(point["h_J_per_kg"]), rel=0.0, abs=0.0
        )


def test_strict_monotonicity_and_positive_cp():
    for thermo in (_nasa(), analytic_h2_oracle(), monatomic_helium()):
        temps = np.linspace(thermo.t_min * 1.01, thermo.t_max * 0.99, 80)
        enthalpy = thermo.enthalpy(temps)
        assert np.all(np.diff(enthalpy) > 0.0)
        assert np.all(thermo.specific_heat(temps) > 0.0)


def test_dh_dt_equals_cp_analytic_nasa_mixture():
    providers = [
        analytic_h2_oracle(),
        _nasa(),
        h2_he_mixture(0.10),
        h2_he_mixture(0.25),
        monatomic_helium(),
    ]
    dT = 1.0e-3
    for thermo in providers:
        temps = np.array([400.0, 800.0, 1500.0, 2500.0])
        cp = thermo.specific_heat(temps)
        dh = (thermo.enthalpy(temps + dT) - thermo.enthalpy(temps - dT)) / (2.0 * dT)
        assert np.max(np.abs(dh - cp) / cp) <= 1.0e-8


def test_mixture_recovers_pure_limits_and_helium_cp():
    nasa = _nasa()
    he = monatomic_helium()
    pure_h2 = h2_he_mixture(0.0)
    pure_he = h2_he_mixture(1.0)
    t = np.array([300.0, 1000.0, 2500.0])
    assert np.max(np.abs(pure_h2.specific_heat(t) - nasa.specific_heat(t))) <= 1.0e-12
    assert np.max(np.abs(pure_he.specific_heat(t) - he.specific_heat(t))) <= 1.0e-12
    assert float(he.specific_heat(1000.0)) == pytest.approx(2.5 * he.gas_constant)


def test_nasa_hard_fails_outside_domain():
    nasa = _nasa()
    with pytest.raises(ThermoDomainError):
        nasa.specific_heat(199.0)
    with pytest.raises(ThermoDomainError):
        nasa.enthalpy(6000.1)


def test_analytic_oracle_is_differentiable_away_from_vibration_limit():
    thermo = AnalyticIdealGasThermo(
        molar_mass=0.00201588,
        degrees_of_freedom_trans_rot=5.0,
        vibrational_temperature=None,
    )
    t = np.linspace(300.0, 3000.0, 20)
    assert np.allclose(thermo.specific_heat(t), 3.5 * thermo.gas_constant)
