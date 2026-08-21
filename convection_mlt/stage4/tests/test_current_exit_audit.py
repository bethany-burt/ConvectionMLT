"""Current Stage 4 audit: every headline status is derived from rows."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from build_current_exit_audit import (  # noqa: E402
    ALGEBRAIC,
    SPATIAL_RCB_DEX,
    _helios_status_from_row,
    main,
)


def test_helios_status_is_derived_from_row():
    assert _helios_status_from_row({"status": "NOT_RUN"}) == "NOT_RUN_OR_PILOT_ONLY"
    assert _helios_status_from_row({"status": "PASS"}) == "PASS"
    assert _helios_status_from_row({"status": "FAIL"}) == "NOT_PASSED"


def test_current_audit_enforces_rcb_timestep_operator_and_algebraic_rows():
    audit = main()
    names = {row["name"] for row in audit["rows"]}
    assert "spatial_n96_vs_n48_rcb_dex" in names
    assert "timestep_refinement_n96_800_vs_400" in names
    assert "operator_equilibrium_rad_then_implicit" in names
    assert "algebraic_n192_bottom_boundary_exactness_rel" in names
    assert "algebraic_n192_telescoping_column_energy_rel" in names
    assert "algebraic_n192_flux_split_identity_rel" in names
    spatial_set = set(audit["headline_row_sets"]["spatial_and_operator"])
    assert "spatial_n96_vs_n48_rcb_dex" in spatial_set
    assert "timestep_refinement_n96_800_vs_400" in spatial_set
    assert "operator_equilibrium_rad_then_implicit" in spatial_set
    helios_row = next(r for r in audit["rows"] if r["name"] == "helios_parity")
    assert audit["helios_parity_status"] == _helios_status_from_row(helios_row)
    rcb = next(r for r in audit["rows"] if r["name"] == "spatial_n96_vs_n48_rcb_dex")
    assert rcb["tolerance"] == SPATIAL_RCB_DEX
    assert rcb["criterion"] == "<="
    assert rcb["status"] in {"PASS", "FAIL"}
    algebraic = [r for r in audit["rows"] if r["category"] == "algebraic"]
    assert algebraic
    assert all(r["tolerance"] == ALGEBRAIC for r in algebraic)
    assert audit["core_single_resolution_status"] == "PASS"
    assert audit["spatial_and_operator_convergence_status"] == "NOT_PASSED"
    assert audit["full_stage4_claim"] is False
