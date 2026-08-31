"""HELIOS adapter, opacity table, and frozen radiation-only parity (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "helios"
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT.parent / "src"))

from convection_mlt import nested_analytic_opacity_spec
from convection_mlt.adapters.helios import (
    flux_cgs_to_si,
    flux_si_to_cgs,
    heating_from_net_flux,
    layer_energy_increment,
    load_integrated_flux,
    load_tp_profile,
    opacity_cgs_to_si,
    opacity_si_to_cgs,
    pressure_microbar_to_pa,
    pressure_pa_to_microbar,
    simulate_helios_tp_read,
    to_canonical_interfaces,
    write_integrated_flux_stub,
    write_param_dat,
    write_tp_profile,
)
from convection_mlt.adapters.helios_contracts import (
    F_INT,
    OPACITY_SI_TO_CGS,
    PA_TO_MICROBAR,
    T_INT,
)
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid
from convection_mlt.adapters.helios_opacity_table import (
    analytic_kappa_cgs,
    bolometric_from_table_bands,
    build_constant_opacity_table,
    build_pressure_tagged_table,
    build_table_arrays,
    build_unique_axis_table,
    flatten_kpoints_helios_order,
    flatten_kpoints_numpy_c_order,
    helios_kpoints_flat_index,
    helios_logp_table_index,
    helios_meanmolmass_flat_index,
    helios_rayleigh_flat_index,
    interpolate_opacity_cgs,
    read_helios_opacity_hdf5,
    unique_kpoints_encoding,
    write_helios_opacity_hdf5,
)
from compare_helios_layer_opacity import load_helios_opacities_dat
from export_helios_grid_reference import export_helios_grid_reference
from compare_frozen_radiation import compare_frozen
from freeze_radiation_only_tolerances import main as freeze_tolerances


h5py = pytest.importorskip("h5py")


def test_unit_conversions():
    assert np.isclose(flux_si_to_cgs(np.array([1.0]))[0], 1000.0)
    assert np.isclose(flux_cgs_to_si(np.array([1000.0]))[0], 1.0)
    assert np.isclose(opacity_si_to_cgs(np.array([1.0]))[0], 10.0)
    assert np.isclose(opacity_cgs_to_si(np.array([10.0]))[0], 1.0)
    assert np.isclose(pressure_pa_to_microbar(np.array([1.0]))[0], PA_TO_MICROBAR)
    assert np.isclose(pressure_microbar_to_pa(np.array([10.0]))[0], 1.0)


def test_helios_gas_boa_microbar_covers_ten_bar():
    from convection_mlt.adapters.helios_contracts import helios_gas_boa_microbar

    toa = 10.0
    n = 96
    boa = helios_gas_boa_microbar(toa, n)
    exponent = 1.0 / (2 * n - 1)
    p0 = boa * (toa / boa) ** exponent
    assert p0 >= 1.0e7


def test_boa_temperature_not_internal_flux_temperature(tmp_path):
    grid = build_helios_pressure_grid(p_boa_microbar=1e9, p_toa_microbar=10.0, n_layers=4)
    t_boa = 650.0
    t_lay = np.linspace(640.0, 200.0, 4)
    path = tmp_path / "tp.dat"
    write_tp_profile(
        path,
        temperature_boa_k=t_boa,
        temperature_lay_k=t_lay,
        p_int_microbar=grid.p_int_microbar,
        p_lay_microbar=grid.p_lay_microbar,
    )
    loaded = load_tp_profile(path)
    boa_row = loaded.temperature_k[loaded.layer_index == -1][0]
    assert np.isclose(boa_row, t_boa)
    assert not np.isclose(boa_row, T_INT)


def test_tp_profile_identity_under_helios_read(tmp_path):
    grid = build_helios_pressure_grid(p_boa_microbar=1e9, p_toa_microbar=10.0, n_layers=8)
    t_boa = 620.0
    t_lay = np.linspace(600.0, 180.0, 8)
    path = tmp_path / "tp.dat"
    write_tp_profile(
        path,
        temperature_boa_k=t_boa,
        temperature_lay_k=t_lay,
        p_int_microbar=grid.p_int_microbar,
        p_lay_microbar=grid.p_lay_microbar,
    )
    loaded = load_tp_profile(path)
    file_p = loaded.pressure_microbar
    file_t = loaded.temperature_k
    recovered = simulate_helios_tp_read(file_p, file_t, grid)
    target = np.concatenate([[t_boa], t_lay])
    assert np.allclose(recovered, target, rtol=1e-5, atol=1e-3)


def test_integrated_flux_roundtrip(tmp_path):
    p = np.array([1e7, 1e6, 1e5, 1e4])
    fn = np.array([300_000.0, 250_000.0, 190_000.0, 120_000.0])
    fu = fn + np.array([10.0, 20.0, 30.0, 40.0])
    fd = fu - fn
    path = tmp_path / "flux.dat"
    write_integrated_flux_stub(path, pressure_microbar=p, flux_down_cgs=fd, flux_up_cgs=fu, flux_net_cgs=fn)
    loaded = load_integrated_flux(path)
    net_si = to_canonical_interfaces(flux_cgs_to_si(loaded.flux_net_cgs), loaded.pressure_microbar, n_layers=3)
    assert np.isclose(net_si[0], 300.0)
    assert np.isclose(net_si[-1], 120.0)


def test_helios_opacity_hdf5_schema(tmp_path):
    spec = nested_analytic_opacity_spec(96)
    table = build_table_arrays(spec.opacity(), t_min=300.0, t_max=500.0, p_min_bar=1.0, p_max_bar=2.0)
    path = tmp_path / "table.h5"
    write_helios_opacity_hdf5(path, table)
    with h5py.File(path, "r") as f:
        assert "interface wavelengths" in f
        assert "wavelength width of bins" in f
        nw, np_, nt = len(table.wavelengths_cm), len(table.pressures_bar), len(table.temperatures_k)
        assert f["meanmolmass"].shape == (np_ * nt,)
        assert f["weighted Rayleigh cross-sections"].shape == (nw * np_ * nt,)
        assert f["kpoints"].shape == (len(table.gauss_y) * nw * np_ * nt,)
        assert f.attrs["linear_index_order"] == "y_fastest"
        assert f.attrs["linear_index_formula"] == (
            "y + ny*x + ny*nx*p + ny*nx*npress*t"
        )
        assert f.attrs["kpoints_flatten"] == "helios"
        assert f.attrs["hdf5_pressure_unit"] == "microbar"
        assert np.allclose(f["pressures"][:], table.pressures_bar * 1.0e6)


def test_helios_table_pressure_covers_n96_layers_without_clamp(tmp_path):
    spec = nested_analytic_opacity_spec(96)
    table = build_table_arrays(
        spec.opacity(), t_min=200.0, t_max=800.0, p_min_bar=5.0e-6, p_max_bar=20.0,
    )
    path = tmp_path / "cover.h5"
    write_helios_opacity_hdf5(path, table)
    with h5py.File(path, "r") as f:
        kpress = np.asarray(f["pressures"][:], dtype=np.float64)
    grid = build_helios_pressure_grid(p_boa_microbar=1.0e7, p_toa_microbar=10.746, n_layers=96)
    npress = kpress.size
    for p_lay in (float(grid.p_lay_microbar[0]), float(grid.p_lay_microbar[-1])):
        idx = helios_logp_table_index(p_lay, kpress)
        assert 0.05 < idx < npress - 1.05


def test_opacity_table_write_read_and_off_node(tmp_path):
    spec = nested_analytic_opacity_spec(96)
    op = spec.opacity()
    table = build_table_arrays(op, t_min=200.0, t_max=1500.0, p_min_bar=1e-8, p_max_bar=10.0, n_temp=8, n_press=8)
    path = tmp_path / "table.h5"
    write_helios_opacity_hdf5(path, table)
    loaded = read_helios_opacity_hdf5(path)
    assert loaded.schema_version
    t, p_bar = float(table.temperatures_k[3]), float(table.pressures_bar[3])
    k_true = float(analytic_kappa_cgs(op, t, p_bar)[0])
    k_node = float(interpolate_opacity_cgs(loaded, t, p_bar))
    assert np.isclose(k_node, k_true, rtol=1e-10, atol=0.0)
    t_off = 0.5 * (table.temperatures_k[1] + table.temperatures_k[2])
    p_off = 0.5 * (table.pressures_bar[2] + table.pressures_bar[3])
    k_off = interpolate_opacity_cgs(loaded, t_off, p_off)
    assert np.isfinite(k_off)


def test_opacity_extrapolation_rejected(tmp_path):
    spec = nested_analytic_opacity_spec(96)
    table = build_table_arrays(spec.opacity(), t_min=300.0, t_max=400.0, p_min_bar=1.0, p_max_bar=2.0)
    path = tmp_path / "t.h5"
    write_helios_opacity_hdf5(path, table)
    loaded = read_helios_opacity_hdf5(path)
    with pytest.raises(ValueError):
        interpolate_opacity_cgs(loaded, 100.0, 1.5)


def test_planck_bolometric_closure(tmp_path):
    spec = nested_analytic_opacity_spec(96)
    table = build_table_arrays(spec.opacity(), t_min=200.0, t_max=2000.0, p_min_bar=1e-9, p_max_bar=100.0)
    t = np.array([300.0, 700.0, 1500.0])
    from convection_mlt import STEFAN_BOLTZMANN

    sigma_t4 = STEFAN_BOLTZMANN * t**4
    band = bolometric_from_table_bands(t, table.wavelengths_cm)
    rel = np.max(np.abs(band - sigma_t4) / sigma_t4)
    assert rel < 0.05


def test_export_helios_grid_reference_n96_thermal():
    payload = export_helios_grid_reference(96, thermal_only=True, diffusivity=2.0)
    assert payload["reference_grid"] == "helios_geometric"
    assert payload["mode"] == "thermal_only"
    assert payload["contracts"]["f_irr"] == 0.0
    assert payload["contracts"]["lower_bc"] == "LowerTemperature(T_boa)"
    assert len(payload["frozen"]["flux_net_W_m2"]) == 97
    assert payload["frozen"]["temperature_boa_k"] != T_INT


def test_freeze_tolerances_before_live(tmp_path, monkeypatch):
    table_path = tmp_path / "table.h5"
    spec = nested_analytic_opacity_spec(96)
    table = build_table_arrays(spec.opacity(), t_min=200.0, t_max=2000.0, p_min_bar=1e-10, p_max_bar=100.0)
    write_helios_opacity_hdf5(table_path, table)
    out = tmp_path / "tols.json"
    monkeypatch.setattr("freeze_radiation_only_tolerances.TABLE", table_path)
    monkeypatch.setattr("freeze_radiation_only_tolerances.OUT", out)
    payload = freeze_tolerances()
    assert payload["frozen_before_live"] is True
    assert payload["gates"]["flux_net_rel"] > 0.0
    assert payload["offline_blocking_gates"]["delta_tau_si_cgs"] < 1e-12


def test_compare_frozen_identical_flux(tmp_path):
    ref = export_helios_grid_reference(96, thermal_only=True, diffusivity=2.0)
    p_micro = np.asarray(ref["frozen"]["pressure_interfaces_helios_microbar"])
    fn_cgs = flux_si_to_cgs(np.asarray(ref["frozen"]["flux_net_W_m2"]))
    fu_cgs = flux_si_to_cgs(np.asarray(ref["frozen"]["flux_up_W_m2"]))
    fd_cgs = flux_si_to_cgs(np.asarray(ref["frozen"]["flux_down_W_m2"]))
    fi_cgs = np.full(fn_cgs.shape, 300_000.0)
    flux_path = tmp_path / "flux.dat"
    write_integrated_flux_stub(
        flux_path,
        pressure_microbar=p_micro,
        flux_down_cgs=fd_cgs,
        flux_up_cgs=fu_cgs,
        flux_net_cgs=fn_cgs,
        flux_intern_cgs=fi_cgs,
    )
    tols = {
        "frozen_before_live": True,
        "gates": {
            "pressure_grid_rel": 1.0,
            "f_intern_parameter_rel": 1.0,
            "flux_decomposition_rel": 1.0,
            "rocky_surface_up_rel": 1.0,
            "rocky_surface_net_rel": 1.0,
            "toa_flux_down_rel": 1.0,
            "flux_up_rel": 1.0,
            "flux_down_rel": 1.0,
            "flux_net_rel": 1.0,
            "heating_from_flux_rel": 1.0,
            "column_energy_closure_rel": 1.0,
        },
    }
    result = compare_frozen(ref, flux_path, tolerances=tols, case="n96_thermal")
    assert result["status"] == "PASS"
    assert result["helios_parity_headline"] is False
    assert result["helios_parity_headline_means"] == "coupled_helios_rce_parity"
    assert result["coupled_helios_rce_claimed"] is False
    assert result["helios_adapter_contract_status"] == "PASS"
    assert result["helios_radiation_only_n96_status"] == "PASS"
    assert result["helios_radiation_only_n192_status"] == "NOT_RUN"
    assert result["helios_radiation_only_parity_status"] == "NOT_RUN"
    assert result["helios_coupled_rce_status"] == "NOT_RUN"
    assert result["helios_coupled_rce_n96_status"] == "NOT_RUN"
    assert result["helios_coupled_rce_n192_status"] == "NOT_RUN"
    assert result["full_stage4_claim"] is False
    assert result["heating_units"] == "W kg^-1"
    assert "reference_column_energy_identity_rel" in result["metrics"]
    assert "helios_column_energy_closure_rel" in result["metrics"]
    assert "interface" in result["local"]["flux_net"]
    assert "pressure_microbar" in result["local"]["flux_net"]
    assert "layer_energy_increment_rel" in result["metrics"]


def test_param_dat_template_has_helios_brackets(tmp_path):
    path = tmp_path / "param.dat"
    write_param_dat(
        path,
        case_name="test_case",
        output_dir="./out/",
        toa_pressure_microbar=1.0,
        boa_pressure_microbar=1e9,
        opacity_path="./table.h5",
        tp_profile_path="./tp.dat",
        t_int_k=T_INT,
        diffusivity_factor=2.0,
        n_layers=96,
        planet_type="rocky",
    )
    text = path.read_text()
    assert "temperature file format" in text
    assert "[helios, TP (bar), PT (bar)]" in text
    assert "stellar spectral model =                              blackbody" in text
    assert "planet type =                                         rocky" in text
    assert "number of layers =                               96" in text


def test_frozen_orientation_fixture_metadata():
    tp = load_tp_profile(FIXTURES / "sample_tp.dat")
    flux = load_integrated_flux(FIXTURES / "sample_integrated_flux.dat")
    from convection_mlt.adapters.helios import make_fixture_metadata

    meta = make_fixture_metadata(
        helios_commit="b0800f9ea4366263241c13bb926e8ca68f266cc5",
        helios_config={
            "case": "stage4_orientation_fixture",
            "convection": 0,
            "lower_bc": "LowerNetInternalFlux",
            "top_irradiation": "disabled_for_n96a",
            "post_processing": True,
        },
        units={
            "temperature": "K",
            "tp_pressure": "microbar",
            "hdf5_pressure": "microbar",
            "flux": "erg s^-1 cm^-2",
            "opacity": "cm^2 g^-1",
            "gravity": "cm s^-2",
        },
        orientation="helios_bottom_first_same_as_canonical",
        arrays_for_checksum={
            "temperature": tp.temperature_k,
            "pressure_microbar": tp.pressure_microbar,
            "flux_net_cgs": flux.flux_net_cgs,
        },
    )
    fixture_path = FIXTURES / "frozen_orientation_fixture.json"
    payload = {
        "helios_commit": meta.helios_commit,
        "helios_config": meta.helios_config,
        "units": meta.units,
        "orientation": meta.orientation,
        "checksum_sha256": meta.checksum_sha256,
    }
    fixture_path.write_text(json.dumps(payload, indent=2) + "\n")
    loaded = json.loads(fixture_path.read_text())
    assert loaded["checksum_sha256"] != "pending_generated_by_test"
    assert loaded["orientation"] == "helios_bottom_first_same_as_canonical"

    beam = json.loads((FIXTURES / "beam_contract.json").read_text())
    assert beam["helios_irradiation"]["exact_map_to_TopIrradiation"] is False
    assert beam["n96b_status"] == "structural_comparison_only"


def test_heating_is_per_mass():
    flux = np.array([10.0, 7.0, 4.0])
    mass = np.array([2.0, 1.0])
    q = heating_from_net_flux(flux, mass)
    df = layer_energy_increment(flux)
    assert np.allclose(df, [3.0, 3.0])
    assert np.allclose(q, [1.5, 3.0])


def test_kpoints_helios_flatten_differs_from_numpy_c_for_p_law():
    table = build_pressure_tagged_table(
        t_min=200.0, t_max=400.0, p_min_bar=0.1, p_max_bar=10.0, n_temp=4, n_press=5, n_wave=3,
    )
    numpy_c = flatten_kpoints_numpy_c_order(table.kpoints_cgs)
    helios = flatten_kpoints_helios_order(table.kpoints_cgs)
    assert numpy_c.size == helios.size
    assert not np.array_equal(numpy_c, helios)
    transposed = np.ascontiguousarray(table.kpoints_cgs.transpose(3, 2, 1, 0)).ravel()
    assert np.array_equal(helios, transposed)


def test_kpoints_writer_matches_helios_linear_index(tmp_path):
    table = build_unique_axis_table()
    ny, nx, npress, ntemp = table.kpoints_cgs.shape
    path = tmp_path / "unique.h5"
    write_helios_opacity_hdf5(path, table)
    with h5py.File(path, "r") as f:
        flat = np.asarray(f["kpoints"][:], dtype=np.float64)
        mean_flat = np.asarray(f["meanmolmass"][:], dtype=np.float64)
        ray_flat = np.asarray(f["weighted Rayleigh cross-sections"][:], dtype=np.float64)
    for t in range(ntemp):
        for p in range(npress):
            assert mean_flat[helios_meanmolmass_flat_index(p, t, npress=npress)] == (
                1.0 + p + 1000.0 * t
            )
            for x in range(nx):
                assert ray_flat[helios_rayleigh_flat_index(x, p, t, nx=nx, npress=npress)] == (
                    x + 100.0 * p + 10000.0 * t
                )
                for y in range(ny):
                    i = helios_kpoints_flat_index(y, x, p, t, ny=ny, nx=nx, npress=npress)
                    assert flat[i] == unique_kpoints_encoding(y, x, p, t)
                    assert flat[i] == table.kpoints_cgs[y, x, p, t]
    loaded = read_helios_opacity_hdf5(path)
    assert np.array_equal(loaded.kpoints_cgs, table.kpoints_cgs)
    assert np.array_equal(loaded.mean_mol_mass_kg, table.mean_mol_mass_kg)
    assert np.array_equal(loaded.rayleigh_cross, table.rayleigh_cross)


def test_numpy_c_flatten_violates_helios_kpoints_index():
    table = build_unique_axis_table()
    ny, nx, npress, ntemp = table.kpoints_cgs.shape
    flat = flatten_kpoints_numpy_c_order(table.kpoints_cgs)
    mismatches = 0
    for t in range(ntemp):
        for p in range(npress):
            for x in range(nx):
                for y in range(ny):
                    i = helios_kpoints_flat_index(y, x, p, t, ny=ny, nx=nx, npress=npress)
                    if flat[i] != table.kpoints_cgs[y, x, p, t]:
                        mismatches += 1
    assert mismatches > 0


def test_constant_and_tagged_tables(tmp_path):
    constant = build_constant_opacity_table(
        0.00225, t_min=200.0, t_max=800.0, p_min_bar=1e-6, p_max_bar=10.0, n_temp=4, n_press=6,
    )
    assert np.allclose(constant.kpoints_cgs, constant.kpoints_cgs.flat[0])
    tagged = build_pressure_tagged_table(
        t_min=200.0, t_max=800.0, p_min_bar=1e-6, p_max_bar=10.0, n_temp=4, n_press=6,
    )
    assert np.allclose(tagged.kpoints_cgs[0, 0, :, 0], tagged.pressures_bar)
    assert np.allclose(tagged.kpoints_cgs[0, 0, :, 0], tagged.kpoints_cgs[0, 0, :, -1])
    path = tmp_path / "const.h5"
    write_helios_opacity_hdf5(path, constant)
    loaded = read_helios_opacity_hdf5(path)
    assert np.allclose(loaded.kpoints_cgs, constant.kpoints_cgs)
    tagged_path = tmp_path / "tagged.h5"
    write_helios_opacity_hdf5(tagged_path, tagged)
    loaded_tagged = read_helios_opacity_hdf5(tagged_path)
    assert np.allclose(loaded_tagged.kpoints_cgs, tagged.kpoints_cgs)


def test_parse_helios_opacities_dat(tmp_path):
    text = (
        "This file contains the bin integrated opacities at each layer center\n"
        "Opacity given in [cm^2 g^-1].\n"
        "bin     cent_lambda[um]   low_int_lambda[um]  delta_lambda[um] opac_lay[0] opac_lay[1]\n"
        "0       1.0               0.5                 1.0              0.02        0.04\n"
    )
    path = tmp_path / "opac.dat"
    path.write_text(text)
    dump = load_helios_opacities_dat(path)
    assert dump["n_bin"] == 1
    assert dump["n_layer"] == 2
    assert np.allclose(dump["opac_band_lay_cgs"][0], [0.02, 0.04])
