"""Coupled HELIOS benchmark: matched forcing, total flux, pilot vs headline."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
FIXTURES = ROOT / "fixtures" / "helios"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT.parent / "src"))

from compare_coupled_helios_rce import (
    flux_column_metrics,
    helios_total_flux_si,
    interpolate_temperature_common_domain,
)
from convection_mlt.adapters.helios import HeliosFluxProfile, flux_si_to_cgs
from convection_mlt.adapters.helios_contracts import helios_track_status
from verify_frozen_inputs import verify


def test_helios_track_status_pilot_is_not_headline():
    pilot = helios_track_status(coupled_n96="PASS")
    assert pilot["helios_coupled_rce_n96_status"] == "PASS"
    assert pilot["helios_coupled_rce_n192_status"] == "NOT_RUN"
    assert pilot["helios_coupled_rce_status"] == "PILOT_ONLY"
    assert pilot["helios_parity_headline"] is False
    assert pilot["coupled_helios_rce_claimed"] is False
    assert pilot["full_stage4_claim"] is False


def test_helios_track_status_n192_can_set_headline_but_not_full_claim():
    done = helios_track_status(coupled_n96="PASS", coupled_n192="PASS")
    assert done["helios_coupled_rce_status"] == "PASS"
    assert done["helios_parity_headline"] is True
    assert done["coupled_helios_rce_claimed"] is True
    assert done["full_stage4_claim"] is False


def test_helios_track_status_fail_blocks_headline():
    failed = helios_track_status(coupled_n96="FAIL")
    assert failed["helios_coupled_rce_status"] == "PILOT_FAILED"
    assert failed["helios_parity_headline"] is False
    assert failed["full_stage4_claim"] is False
    both = helios_track_status(coupled_n96="FAIL", coupled_n192="FAIL")
    assert both["helios_coupled_rce_status"] == "FAIL"


def test_detect_radconv_done_with_zero_radiative_counter(tmp_path):
    from compare_coupled_helios_rce import detect_helios_convergence

    log = tmp_path / "helios_stdout.log"
    log.write_text(
        "Number of radiative layers converged: 0 out of 75.\n"
        "Global energy imbalance is 2.721e-08 and should be less than 1.0e-08\n"
        "Total number of iterative steps: 843\n"
        "Done! Everything appears to have worked fine :-)\n"
        "Global energy imbalance: 0.010ppm\n"
    )
    conv = detect_helios_convergence(case_dir=None, case_name=None, helios_log=log)
    assert conv["converged"] is True
    assert conv["helios_native_convergence"] == "PASS_WITH_DIAGNOSTIC_WARNING"
    assert conv["termination_mode"] == "radconv_global_energy_imbalance"
    assert conv["radiative_layer_counter_final"] == "0/75"


def test_flux_column_metrics_are_total_flux_not_radiative_duplicate():
    total = np.array([300.0, 315.0, 291.0])
    metrics = flux_column_metrics(total, 300.0, intern_boa=300.0)
    assert metrics["toa_total_flux_rel"] == pytest.approx(9.0 / 300.0)
    assert metrics["boa_total_flux_rel"] == pytest.approx(0.0)
    assert metrics["max_column_flatness"] == pytest.approx(15.0 / 300.0)
    assert metrics["max_column_flatness_interface"] == 1
    assert metrics["column_closure_rel"] == pytest.approx(9.0 / 300.0)
    assert metrics["energy_closure_rel"] == metrics["column_closure_rel"]
    assert metrics["energy_closure_rel"] != metrics["boa_total_flux_rel"]
    assert metrics["f_intern_rel"] == pytest.approx(0.0)
    assert metrics["f_intern_W_m2"] == 300.0


def test_helios_total_flux_adds_convective_column():
    rad = np.array([250.0, 150.0, 50.0])
    conv = np.array([50.0, 150.0, 250.0])
    flux = HeliosFluxProfile(
        interface_index=np.array([0, 1, 2], dtype=np.int64),
        pressure_microbar=np.array([1.0e7, 1.0e5, 1.0e3]),
        flux_down_cgs=np.zeros(3),
        flux_up_cgs=np.zeros(3),
        flux_net_cgs=flux_si_to_cgs(rad),
        flux_conv_net_cgs=flux_si_to_cgs(conv),
        flux_intern_cgs=flux_si_to_cgs(np.array([300.0, np.nan, np.nan])),
    )
    out = helios_total_flux_si(flux)
    assert np.allclose(out["total"], [300.0, 300.0, 300.0])
    assert out["intern_boa"] == pytest.approx(300.0)


def test_helios_total_flux_rejects_nonfinite_convective_flux():
    flux = HeliosFluxProfile(
        interface_index=np.array([0, 1], dtype=np.int64),
        pressure_microbar=np.array([1.0e7, 1.0e3]),
        flux_down_cgs=np.zeros(2),
        flux_up_cgs=np.zeros(2),
        flux_net_cgs=flux_si_to_cgs(np.array([300.0, 300.0])),
        flux_conv_net_cgs=np.array([np.nan, np.nan]),
        flux_intern_cgs=flux_si_to_cgs(np.array([300.0, np.nan])),
    )
    with pytest.raises(ValueError, match="F_net_conv"):
        helios_total_flux_si(flux)


def test_coupled_wrapper_verifies_coupled_manifest_not_radiation_only():
    text = (ROOT / "cluster" / "run_stage4_helios_coupled_n96.sh").read_text()
    assert "coupled_input_manifest.json" in text
    verify_block = text.split("verify_frozen_inputs.py", 1)[1]
    assert "coupled_input_manifest.json" in verify_block
    assert "frozen_input_manifest.json" not in verify_block.split("export_coupled", 1)[0]


def test_verify_coupled_manifest_refuses_mismatch_or_missing(tmp_path):
    (tmp_path / "a.json").write_text("ok", encoding="utf-8")
    digest = hashlib.sha256(b"ok").hexdigest()
    passed = verify({"files": {"a.json": digest}}, fixtures=tmp_path)
    assert passed["status"] == "PASS"
    mismatched = verify({"files": {"a.json": "deadbeef"}}, fixtures=tmp_path)
    assert mismatched["status"] == "FAIL"
    assert mismatched["files"]["a.json"]["status"] == "MISMATCH"
    missing = verify({"files": {"missing.json": digest}}, fixtures=tmp_path)
    assert missing["status"] == "FAIL"
    assert missing["files"]["missing.json"]["status"] == "MISSING"


def test_sidecar_file_sha256_matches_cluster_h5_pin():
    sidecar = json.loads((FIXTURES / "analytic_grey_nested.json").read_text())
    expected = (
        "9505247e1104c9d11500944975a2d26b82d55c4e3c7c66f579a5a9c08334cd3c"
    )
    assert sidecar["checksum_sha256"] == expected
    assert sidecar["file_sha256"] == expected
    assert sidecar["table_checksum_sha256"] == (
        "bf4826e7889d9bad68f5ff7f6decf78290a39fa7b955d9c63af5bf08eedf35b9"
    )
    assert sidecar["checksum_sha256"] != sidecar["table_checksum_sha256"]


def test_frozen_firr0_mlt_is_gated_bottom_cz():
    path = FIXTURES / "mlt_nested_tau_n96_firr0.json"
    rec = json.loads(path.read_text())
    assert rec["status"] == "converged"
    assert float(rec["f_irr"]) == 0.0
    assert float(rec["f_int"]) == 300.0
    assert rec["flux_flatness"] < 1.0e-3
    assert rec["convective_regions"][0][0] == 0
    assert rec.get("detached_convective_regions") in ([], None)
    assert rec["profile_checksum_sha256"].startswith("b5eb3508")


def test_scorer_without_helios_stays_not_run():
    from compare_coupled_helios_rce import score

    tols = json.loads((FIXTURES / "coupled_rce_benchmark_tolerances.json").read_text())
    payload = score(n_layers=96, helios_tp=None, helios_flux=None, tolerances=tols)
    assert payload["status"] == "NOT_RUN"
    assert payload["helios_coupled_rce_n96_status"] == "NOT_RUN"
    assert payload["helios_coupled_rce_status"] == "NOT_RUN"
    assert payload["helios_parity_headline"] is False
    assert payload["full_stage4_claim"] is False
    assert payload["mlt_f_irr"] == 0.0
    assert payload["structural_irradiated"]["status"] == "STRUCTURAL_NOT_SCORED"


def test_structural_missing_nested_family_is_not_scored(monkeypatch, tmp_path):
    from compare_coupled_helios_rce import score_structural_irradiated
    import export_helios_grid_reference as egr

    monkeypatch.setattr(egr, "NESTED", tmp_path / "missing_nested_rce_family.json")
    rec = json.loads((FIXTURES / "mlt_nested_tau_n96_firr0.json").read_text())
    out = score_structural_irradiated(96, rec)
    assert out["status"] == "STRUCTURAL_NOT_SCORED"
    assert "unavailable" in out["note"]


def test_helios_abort_payload_skips_physical_gates(tmp_path):
    from compare_coupled_helios_rce import find_helios_abort, helios_abort_payload, score

    abort = tmp_path / "stage4_coupled_n96_ABORT.dat"
    abort.write_text("The run exceeded the maximum number of iteration steps and was aborted.\n")
    assert find_helios_abort(tmp_path, "stage4_coupled_n96") == abort
    tols = json.loads((FIXTURES / "coupled_rce_benchmark_tolerances.json").read_text())
    payload = score(
        n_layers=96,
        helios_tp=tmp_path / "missing_tp.dat",
        helios_flux=tmp_path / "missing_flux.dat",
        tolerances=tols,
        case_dir=tmp_path,
        case_name="stage4_coupled_n96",
    )
    assert payload["status"] == "HELIOS_ABORT"
    assert payload["execution_status"] == "HELIOS_ABORT"
    assert payload["failure_stage"] == "helios_convergence"
    assert payload["helios_coupled_rce_n96_status"] == "NOT_RUN"
    assert payload["full_stage4_claim"] is False
    assert "metrics" not in payload
    abort_only = helios_abort_payload(
        n_layers=96, abort_path=abort, helios_tp=None, helios_flux=None
    )
    assert abort_only["helios_coupled_rce_n96_status"] == "NOT_RUN"


def test_helios_crash_payload_detects_iso_convection_traceback(tmp_path):
    from compare_coupled_helios_rce import helios_abort_payload

    log = tmp_path / "helios_stdout.log"
    log.write_text(
        "Total number of iterative steps: 3872\n"
        "Traceback (most recent call last):\n"
        "  File \"computation.py\", line 1009, in convection_loop\n"
        "    condition = sum(quant.conv_unstable) > 0\n"
        "TypeError: 'NoneType' object is not iterable\n"
    )
    payload = helios_abort_payload(
        n_layers=96, abort_path=None, helios_tp=None, helios_flux=None, helios_log=log
    )
    assert payload["status"] == "HELIOS_CRASH"
    assert payload["failure_stage"] == "helios_iso_convection_incompatible"
    assert payload["helios_traceback"] is True
    assert payload["helios_coupled_rce_n96_status"] == "NOT_RUN"


def test_export_stock_iterative_uses_50k_iso_no_and_full_precision(tmp_path):
    from export_coupled_helios_case import export_coupled_case, PLANCK_TABLE_DIM, PLANCK_TABLE_STEP
    from convection_mlt.adapters.helios_contracts import T_INT, PROVENANCE_ONLY
    import re

    opacity = FIXTURES / "analytic_grey_nested.h5"
    if not opacity.exists():
        pytest.skip("analytic_grey_nested.h5 not available locally")
    runtime = export_coupled_case(96, case_dir=tmp_path, opacity_path=opacity)
    assert runtime["maximum_number_of_iterations"] == 50000
    assert runtime["physical_timestep"] == "no"
    assert runtime["isothermal_layers"] is False
    assert runtime["relax_radiative_criterion_at"] == [10000, 20000]
    assert runtime["planck_table_dimension_and_stepsize"] == [PLANCK_TABLE_DIM, PLANCK_TABLE_STEP]
    assert runtime["prior_job_16015568"] == "HELIOS_RUNTIME_BUG_ISO1"
    param = (tmp_path / "param.dat").read_text()
    m = re.search(r"^maximum number of iterations\s*=\s*(\S+)", param, re.M)
    assert m and int(m.group(1)) == 50000
    m = re.search(r"^isothermal layers\s*=\s*(\S+)", param, re.M)
    assert m and m.group(1) == "no"
    m = re.search(r"^internal temperature \[K\]\s*=\s*(\S+)", param, re.M)
    assert m and abs(float(m.group(1)) - float(T_INT)) < 1e-12
    m = re.search(r"^kappa value\s*=\s*(\S+)", param, re.M)
    assert m and abs(float(m.group(1)) - float(PROVENANCE_ONLY["nabla_ad"])) < 1e-15
    m = re.search(r"^physical timestep \[s\]\s*=\s*(\S+)", param, re.M)
    assert m and m.group(1) == "no"
    m = re.search(r"^plancktable dimension and stepsize\s*=\s*(\S+)\s+(\S+)", param, re.M)
    assert m and int(float(m.group(1))) == 8000 and int(float(m.group(2))) == 2


def test_runtime_fixture_isothermal_layers_is_false():
    runtime = json.loads((FIXTURES / "helios_coupled_n96_runtime_config.json").read_text())
    assert runtime["isothermal_layers"] is False
    assert runtime["maximum_number_of_iterations"] == 50000
    assert runtime["internal_flux_temperature_k"] == pytest.approx(269.6977849204774)
    assert runtime["kappa_nabla_ad"] == pytest.approx(2.0 / 7.0)
    assert runtime["prior_job_16015568"] == "HELIOS_RUNTIME_BUG_ISO1"


def test_job_16015568_classified_as_runtime_bug_iso1():
    path = ROOT / "results" / "helios_coupled_n96_job16015568_classification.json"
    rec = json.loads(path.read_text())
    assert rec["classification"] == "HELIOS_RUNTIME_BUG_ISO1"
    assert rec["not"] == "physical_nonconvergence"


def test_wrapper_abort_gate_before_physical_scorer():
    text = (ROOT / "cluster" / "run_stage4_helios_coupled_n96.sh").read_text()
    assert "find_helios_abort" in text
    assert "physical_scorer" in text and "SKIPPED" in text
    assert "maximum number of iterations" in text
    assert "50000" in text
    assert "isothermal layers\": \"no\"" in text or 'isothermal layers": "no"' in text
    assert "diagnose_coupled_frozen_iso_no" in text
    assert "8000" in text and "Planck table must stay" in text
    # Preflight must run before HELIOS launch.
    assert text.index("exported case preflight PASS") < text.index("helios.py")
    assert text.index("frozen_preflight") < text.index("compare_coupled_helios_rce.py")
    assert text.index("find_helios_abort") < text.index("compare_coupled_helios_rce.py")


def test_temperature_common_domain_rejects_extrapolation():
    src = np.log(np.array([1.0e2, 1.0e4, 1.0e6]))
    t = np.array([200.0, 300.0, 500.0])
    dst = np.log(np.array([10.0, 1.0e3, 1.0e5, 1.0e8]))
    out, mask = interpolate_temperature_common_domain(src, t, dst)
    assert mask.tolist() == [False, True, True, False]
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert out[1] == pytest.approx(250.0)
    assert out[2] == pytest.approx(400.0)


def test_coupled_tolerances_are_firr0_and_keep_numeric_gates():
    tols = json.loads((FIXTURES / "coupled_rce_benchmark_tolerances.json").read_text())
    assert tols["f_irr"] == 0.0
    assert tols["f_int"] == 300.0
    assert tols["mlt_grid"] == "nested_tau_interpolated_to_helios"
    assert tols["comparison_type"] == "independently_discretized_rce_matched_forcing"
    assert "independently discretized" in tols["benchmark_interpretation"]
    assert tols["irradiated_nested_mlt"] == "structural_diagnostic_only"
    assert tols["gates"]["toa_flux_rel"] == 0.05
    assert tols["gates"]["max_rel_T"] == 0.10
    assert tols["gates"]["rcb_dex"] == 0.15
    assert tols["gates"]["energy_closure_rel"] == 0.05
    assert "F_rad,net + F_conv" in tols["norms"]["toa_flux_rel"]
    assert "F_total,BOA" in tols["norms"]["energy_closure_rel"]
    assert "no endpoint extrapolation" in tols["norms"]["max_rel_T"]


def test_negative_grid_diagnostic_is_archived():
    index = json.loads(
        (ROOT / "results" / "helios_grid_mlt_negative_diagnostic" / "index.json").read_text()
    )
    assert index["status"] == "DIAGNOSTIC_NOT_SCORED"
    assert index["runs"]["rc_seed"]["primary_rcb_log10p"] is None
    assert index["runs"]["nested_seed_failed"]["flux_flatness"] > 1.0
    assert "independently discretized" in index["purpose"]
