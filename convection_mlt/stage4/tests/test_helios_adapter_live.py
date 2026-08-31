from __future__ import annotations

from pathlib import Path

import numpy as np

from convection_mlt import HeliosAdapter, load_integrated_flux, load_tp_profile, to_canonical_interfaces


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "helios"


def test_parse_helios_flux_and_tp_files():
    tp = load_tp_profile(FIXTURE_DIR / "sample_tp.dat")
    flux = load_integrated_flux(FIXTURE_DIR / "sample_integrated_flux.dat")

    assert tp.temperature_k.shape[0] >= 4
    assert flux.flux_net_cgs.shape[0] == 4
    assert np.isclose(flux.flux_net_cgs[0], 300.0)
    assert np.isclose(flux.flux_net_cgs[-1], 120.0)


def test_orientation_bottom_first_without_reversal():
    flux = load_integrated_flux(FIXTURE_DIR / "sample_integrated_flux.dat")
    canonical_net = to_canonical_interfaces(
        flux.flux_net_cgs,
        flux.pressure_microbar,
        n_layers=flux.flux_net_cgs.size - 1,
    )
    assert np.isclose(canonical_net[0], 300.0)
    assert np.isclose(canonical_net[-1], 120.0)


def test_orientation_asymmetric_flux_no_double_reversal():
    pressure = np.array([1.0e7, 1.0e6, 1.0e5, 1.0e4], dtype=np.float64)
    values = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float64)
    out = to_canonical_interfaces(values, pressure, n_layers=3)
    assert np.array_equal(out, values)


def test_legacy_pilot_reversal_isolated():
    flux = load_integrated_flux(FIXTURE_DIR / "sample_integrated_flux.dat")
    adapter = HeliosAdapter(legacy_reverse=True)
    reversed_net = adapter.to_canonical_interfaces(flux.flux_net_cgs)
    assert np.isclose(reversed_net[0], 120.0)
    assert np.isclose(reversed_net[-1], 300.0)
