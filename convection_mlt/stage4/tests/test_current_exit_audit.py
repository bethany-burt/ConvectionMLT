"""Current Stage 4 audit: every headline status is derived from rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(EXPERIMENTS))

from build_current_exit_audit import (  # noqa: E402
    ALGEBRAIC,
    ENERGY_GATE,
    SPATIAL_RCB_DEX,
    _helios_status_from_row,
    _row,
    assert_n192_audit_sync,
    energy_fields_from_record,
    main,
)


def test_richardson_in_range_rejects_upper_bound():
    lo_hi = [0.25, 3.0]
    assert _row("r", 1.825, lo_hi, "in_range", "p", "spatial", "t")["status"] == "PASS"
    assert _row("r", 0.25, lo_hi, "in_range", "p", "spatial", "t")["status"] == "FAIL"
    assert _row("r", 3.0, lo_hi, "in_range", "p", "spatial", "t")["status"] == "FAIL"
    assert _row("r", 3.1, lo_hi, "in_range", "p", "spatial", "t")["status"] == "FAIL"
    assert _row("r", 0.24, lo_hi, "in_range", "p", "spatial", "t")["status"] == "FAIL"


def test_helios_status_is_derived_from_row():
    assert _helios_status_from_row({"status": "NOT_RUN"}) == "NOT_RUN_OR_PILOT_ONLY"
    assert _helios_status_from_row({"status": "PASS"}) == "PASS"
    assert _helios_status_from_row({"status": "FAIL"}) == "NOT_PASSED"


def test_energy_fields_use_ulp_floor_not_relative_1e12():
    fields = energy_fields_from_record({
        "energy_committed_residual": 8.12e-5,
        "energy_ulp_floor": 3.95e-5,
        "energy_residual_rel": 1.08e-10,
    })
    assert fields is not None
    assert fields["energy_gate_ratio"] <= 1.0
    assert fields["energy_allowed"] >= 16.0 * 3.95e-5


def test_current_audit_enforces_rcb_timestep_operator_and_algebraic_rows():
    audit = main()
    names = {row["name"] for row in audit["rows"]}
    assert "spatial_n96_vs_n48_rcb_dex" in names
    assert "timestep_refinement_n96_800_vs_400" in names
    assert "operator_equilibrium_rad_then_implicit" in names
    assert "algebraic_n192_bottom_boundary_exactness_rel" in names
    assert "algebraic_n192_telescoping_column_energy_rel" in names
    assert "algebraic_n192_flux_split_identity_rel" in names
    assert "algebraic_n192_energy_gate_ratio" in names
    assert "physical_rce_n384_implicit" in names
    assert "spatial_nested_n384_vs_n192_max_rel_T" in names
    assert "richardson_nested_96_192_384" in names
    spatial_set = set(audit["headline_row_sets"]["spatial_and_operator"])
    assert "spatial_n96_vs_n48_rcb_dex" not in spatial_set
    assert "spatial_nested_n96_vs_n48_max_rel_T" not in spatial_set
    assert "physical_rce_n192_implicit" in spatial_set
    assert "physical_rce_n384_implicit" in spatial_set
    assert "spatial_nested_n384_vs_n192_max_rel_T" in spatial_set
    assert "spatial_nested_n384_vs_n192_rcb_dex" in spatial_set
    assert "richardson_nested_96_192_384" in spatial_set
    assert "timestep_refinement_n96_800_vs_400" in spatial_set
    assert "operator_equilibrium_rad_then_implicit" in spatial_set
    helios_row = next(r for r in audit["rows"] if r["name"] == "helios_parity")
    assert audit["helios_parity_status"] == _helios_status_from_row(helios_row)
    rcb = next(r for r in audit["rows"] if r["name"] == "spatial_n96_vs_n48_rcb_dex")
    assert rcb["tolerance"] == SPATIAL_RCB_DEX
    assert rcb["category"] == "spatial_diagnostic"
    algebraic = [r for r in audit["rows"] if r["category"] == "algebraic"]
    assert algebraic
    identities = [r for r in algebraic if r["name"].endswith("_rel")]
    assert all(r["tolerance"] == ALGEBRAIC for r in identities)
    energy = next(r for r in algebraic if r["name"] == "algebraic_n192_energy_gate_ratio")
    assert energy["tolerance"] == ENERGY_GATE
    assert "algebraic_n192_energy_gate_ratio" in audit["headline_row_sets"]["algebraic"]
    n192_path = RESULTS / "n192_implicit_rce.json"
    if n192_path.exists():
        n192 = json.loads(n192_path.read_text())
        assert_n192_audit_sync(audit, n192)
    assert audit["core_single_resolution_status"] == "PASS"
    assert audit["spatial_and_operator_convergence_status"] == "PASS"
    assert audit["algebraic_identity_status"] == "PASS"
    assert audit["internal_numerical_track_complete"] is True
    assert audit["full_stage4_claim"] is False
    assert audit["coupled_helios_rce_claimed"] is False
    assert audit["helios_parity_headline"] is False
    assert audit["helios_parity_headline_means"] == "coupled_helios_rce_parity"
    assert audit["helios_adapter_contract_status"] == "PASS"
    assert audit["helios_radiation_only_n96_status"] == "PASS"
    assert audit["helios_radiation_only_n192_status"] == "PASS"
    assert audit["helios_radiation_only_parity_status"] == "PASS"
    assert audit["helios_coupled_rce_status"] == "FAIL"
    assert audit["helios_coupled_rce_n96_status"] == "FAIL"
    assert audit["helios_coupled_rce_n192_status"] == "RESOLUTION_COMPLETE"
    assert audit["helios_parity_status"] != "PASS"
    spat = next(r for r in audit["rows"] if r["name"] == "spatial_nested_n384_vs_n192_max_rel_T")
    assert spat["status"] == "PASS"
    assert spat.get("n384_live_record") == "n384_implicit_rce.json"
    assert spat.get("n384_checksum", "").startswith("5e0bd359")
    coupled = next(r for r in audit["rows"] if r["name"] == "helios_coupled_rce_benchmark")
    assert coupled["status"] == "FAIL"
    assert coupled.get("frozen_before_live") is True
    assert coupled.get("comparison_type") in (
        "benchmark_not_solver_identity",
        "independently_discretized_rce_matched_forcing",
    )
    assert coupled.get("f_irr") == 0.0
    assert coupled.get("mlt_grid") == "nested_tau_interpolated_to_helios"
    assert coupled.get("irradiated_nested_mlt") == "structural_diagnostic_only"
    rich = next(r for r in audit["rows"] if r["name"] == "richardson_nested_96_192_384")
    assert rich["criterion"] == "in_range"
    assert rich["tolerance"] == [0.25, 3.0]
    assert rich["status"] == "PASS"
    assert 0.25 < float(rich["observed"]) < 3.0
