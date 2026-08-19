from __future__ import annotations

from pathlib import Path

import numpy as np

from convection_mlt import HeliosAdapter, load_integrated_flux, load_tp_profile


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "helios"


def test_parse_helios_flux_and_tp_files():
    tp = load_tp_profile(FIXTURE_DIR / "sample_tp.dat")
    flux = load_integrated_flux(FIXTURE_DIR / "sample_integrated_flux.dat")

    assert tp.temperature_k.shape[0] >= 4
    assert flux.flux_net_cgs.shape[0] == 4
    assert np.isclose(flux.flux_net_cgs[0], 300.0)
    assert np.isclose(flux.flux_net_cgs[-1], 120.0)


def test_orientation_and_sign_mapping():
    flux = load_integrated_flux(FIXTURE_DIR / "sample_integrated_flux.dat")
    adapter = HeliosAdapter(helios_top_to_bottom=True)

    canonical_net = adapter.to_canonical_interfaces(flux.flux_net_cgs)
    # top-to-bottom HELIOS -> bottom-to-top canonical reversal
    assert np.isclose(canonical_net[0], 120.0)
    assert np.isclose(canonical_net[-1], 300.0)

    # roundtrip must be exact for this deterministic fixture
    rt = adapter.roundtrip_interfaces(flux.flux_net_cgs)
    assert np.array_equal(rt, flux.flux_net_cgs)
