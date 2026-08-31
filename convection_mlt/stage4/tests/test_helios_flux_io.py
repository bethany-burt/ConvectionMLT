"""HELIOS integrated-flux writer/parser contract."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
FIXTURES = ROOT / "fixtures" / "helios"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT.parent / "src"))

from apply_helios_write_precision import MARKER, apply
from convection_mlt.adapters.helios import (
    format_integrated_flux_row,
    load_integrated_flux,
    parse_flux_token,
)
from recover_glued_helios_flux import recover_fields, recover_file


def test_parse_flux_token_sentinels():
    assert np.isnan(parse_flux_token("not_avail."))
    assert np.isnan(parse_flux_token("not_avail"))
    assert np.isnan(parse_flux_token(""))
    assert parse_flux_token("-1.25e+05") == -1.25e5


def test_format_row_has_ten_space_fields_when_intern_present():
    line = format_integrated_flux_row(0, 1e7, 10.0, 310.0, 300.0, f_intern=3e5)
    fields = line.split()
    assert len(fields) == 10
    assert "  " not in line.replace("not_avail.", "X")


def test_writer_roundtrip_negative_and_exponents(tmp_path):
    line = format_integrated_flux_row(
        1, 1.234e105, -20.0, -1.2e2, -100.0, f_dir=0.0, delta_f_net=None, f_intern=None,
    )
    path = tmp_path / "flux.dat"
    path.write_text(
        "interface press.[10^-6bar] F_down F_up F_net F_dir delta_F_net F_net_conv F_add_heat F_intern\n"
        + line
        + "\n"
    )
    flux = load_integrated_flux(path)
    assert np.isclose(flux.pressure_microbar[0], 1.234e105)
    assert np.isclose(flux.flux_net_cgs[0], -100.0)
    assert np.isnan(flux.flux_intern_cgs[0])
    assert len(line.split()) == 9


def test_parser_fixtures():
    cases = {
        "flux_parser_not_avail.dat": dict(n=3, intern0=3e5, net0=300.0),
        "flux_parser_missing_intern.dat": dict(n=4, intern0=3e5, net0=300.0),
        "flux_parser_negatives.dat": dict(n=2, intern0=3e5, net0=300.0),
        "flux_parser_exp_two_digit.dat": dict(n=2, intern0=3e5, net0=9.19e5),
        "flux_parser_exp_three_digit.dat": dict(n=2, intern0=3e5, net0=9.19e5),
        "flux_parser_post_processing.dat": dict(n=4, intern0=3e5, net0=300.0),
        "flux_parser_iterative.dat": dict(n=4, intern0=3e5, net0=300.0),
        "sample_integrated_flux.dat": dict(n=4, intern0=300.0, net0=300.0),
    }
    for name, expect in cases.items():
        path = FIXTURES / name
        flux = load_integrated_flux(path)
        assert flux.flux_net_cgs.size == expect["n"], name
        assert np.isclose(flux.flux_net_cgs[0], expect["net0"]), name
        assert np.isclose(flux.flux_intern_cgs[0], expect["intern0"]), name
        assert np.isnan(flux.flux_intern_cgs[-1])
        for arr in (
            flux.interface_index,
            flux.pressure_microbar,
            flux.flux_down_cgs,
            flux.flux_up_cgs,
            flux.flux_net_cgs,
            flux.flux_conv_net_cgs,
            flux.flux_intern_cgs,
        ):
            assert arr.size == expect["n"], name


def test_tab_delimited_and_legacy_header(tmp_path):
    path = tmp_path / "tabs.dat"
    path.write_text(
        "interface\tpress.[10^-6bar]\tF_down\tF_up\tF_net\tF_dir\tdelta_F_net (layer quantity)\tF_net_conv\tF_add_heat\tF_intern\n"
        "0\t1e7\t10\t310\t300\t0\tnot_avail.\t100\t0\t300000\n"
    )
    flux = load_integrated_flux(path)
    assert np.isclose(flux.flux_net_cgs[0], 300.0)
    assert not np.isnan(flux.flux_conv_net_cgs[0])


def test_apply_replaces_fixed_width_writer(tmp_path):
    src = ROOT.parent / "external" / "HELIOS" / "source" / "write.py"
    dest = tmp_path / "write.py"
    dest.write_text(src.read_text())
    assert apply(dest) == "patched"
    text = dest.read_text()
    assert MARKER in text
    assert '"{:<23g}".format(quant.F_net[i])' not in text
    ast.parse(text)
    assert "    @staticmethod\n    def write_upward_spectral_flux" in text
    assert apply(dest) == "already_patched"


def test_versioned_write_py_patch_checksum(tmp_path):
    from apply_helios_write_precision import (
        PINNED_HELIOS_COMMIT,
        emit_patch,
        patch_provenance,
        unified_write_py_diff,
        verify_applied_diff,
        _replace_method,
    )

    src = ROOT.parent / "external" / "HELIOS" / "source" / "write.py"
    original = src.read_text()
    patch_path = tmp_path / "helios_write_integrated_flux_b0800f9.patch"
    sidecar = emit_patch(src, patch_path)
    assert sidecar["helios_commit"] == PINNED_HELIOS_COMMIT
    assert sidecar["sha256"] == patch_provenance(patch_path)["sha256"]
    assert len(sidecar["sha256"]) == 64
    pinned = FIXTURES / "helios_write_integrated_flux_b0800f9.patch"
    pinned_meta = FIXTURES / "helios_write_integrated_flux_b0800f9.patch.json"
    assert pinned.exists()
    assert pinned_meta.exists()
    pinned_sidecar = json.loads(pinned_meta.read_text())
    assert pinned_sidecar["helios_commit"] == PINNED_HELIOS_COMMIT
    assert pinned_sidecar["sha256"] == hashlib.sha256(pinned.read_bytes()).hexdigest()
    live = unified_write_py_diff(original, _replace_method(original))
    verify_applied_diff(live, pinned)



def test_recover_glued_fnet_fdir_zero():
    fields = [
        "0",
        "1.00000000000000000e+07",
        "1.14073361804592926e+07",
        "1.23263703720447160e+07",
        "9.19034191585423425e+050",
        "not_avail.",
        "0",
        "0",
        "300001",
    ]
    recovered = recover_fields(fields)
    assert recovered[4] == "9.19034191585423425e+05"
    assert recovered[5] == "0.0"
    assert recovered[-1] == "300001"


def test_recover_file_roundtrip(tmp_path):
    src = tmp_path / "glued.dat"
    src.write_text(
        "interface press.[10^-6bar] F_down F_up F_net F_dir delta_F_net F_net_conv F_add_heat F_intern\n"
        "0 1.00000000000000000e+07 1.14073361804592926e+07 1.23263703720447160e+07 "
        "9.19034191585423425e+050 not_avail. 0 0 300001\n"
    )
    dest = tmp_path / "recovered.dat"
    info = recover_file(src, dest)
    assert info["n_glued_split"] == 1
    flux = load_integrated_flux(dest)
    assert np.isclose(flux.flux_net_cgs[0], 9.19034191585423425e5)
    assert np.isclose(flux.flux_intern_cgs[0], 300001.0)
    assert len(dest.read_text().strip().splitlines()[-1].split()) == 10
