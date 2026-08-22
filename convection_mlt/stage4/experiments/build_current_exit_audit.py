"""Clean current Stage 4 audit. Historical pre-implicit rows live in the archive file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "plots" / "data"
ARCHIVE = RESULTS / "exit_gate_audit_preimplicit_2026-08-19.json"
OUT = RESULTS / "exit_gate_audit.json"

ALGEBRAIC = 1.0e-12
PHYSICAL_RCE = 1.0e-3
SPATIAL_MAX_REL_T = 0.02
SPATIAL_RCB_DEX = 0.05


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _row(name, observed, tolerance, criterion, scale, category, source, extra=None):
    finite = isinstance(observed, (int, float, bool, np.floating, np.integer)) and (
        isinstance(observed, bool) or np.isfinite(observed)
    )
    if criterion == "<=":
        status = "PASS" if finite and float(observed) <= tolerance else "FAIL"
    elif criterion == "==":
        status = "PASS" if observed == tolerance else "FAIL"
    elif criterion == "true":
        status = "PASS" if bool(observed) else "FAIL"
    elif criterion == "pending":
        status = "NOT_RUN"
    else:
        status = "FAIL"
    row = {
        "name": name,
        "observed": observed if finite or isinstance(observed, (str, bool)) else observed,
        "tolerance": tolerance,
        "criterion": criterion,
        "status": status,
        "scale": scale,
        "category": category,
        "source": source,
    }
    if extra:
        row.update(extra)
    return row


def _algebraic_from_fluxes(rec: dict | None) -> dict[str, float] | None:
    if rec is None:
        return None
    needed = ("flux_total", "flux_rad", "flux_conv", "mass_path", "f_int")
    if any(k not in rec or rec[k] is None for k in needed):
        return None
    f_tot = np.asarray(rec["flux_total"], dtype=np.float64)
    f_rad = np.asarray(rec["flux_rad"], dtype=np.float64)
    f_conv = np.asarray(rec["flux_conv"], dtype=np.float64)
    mass = np.asarray(rec["mass_path"], dtype=np.float64)
    f_int = float(rec["f_int"])
    scale = max(abs(f_int), 1.0)
    heating = (f_tot[:-1] - f_tot[1:]) / mass
    return {
        "flux_split_identity_rel": float(np.max(np.abs(f_rad + f_conv - f_tot)) / scale),
        "telescoping_column_energy_rel": float(
            abs(float(np.sum(mass * heating)) - float(f_tot[0] - f_tot[-1])) / scale
        ),
        "bottom_boundary_exactness_rel": float(abs(float(f_tot[0]) - f_int) / scale),
    }


def _helios_status_from_row(row: dict) -> str:
    if row["status"] == "PASS":
        return "PASS"
    if row["status"] in {"NOT_RUN", "NOT_RUN_OR_PILOT_ONLY"}:
        return "NOT_RUN_OR_PILOT_ONLY"
    return "NOT_PASSED"


def main() -> dict:
    analytic = _load(DATA / "analytic_opacity_rce.json") or {}
    spatial = _load(DATA / "equal_time_spatial.json") or {}
    operator = _load(DATA / "operator_order_point39.json") or {}
    n192 = _load(RESULTS / "n192_implicit_rce.json")
    nested = _load(RESULTS / "nested_rce_family.json") or {}
    n48 = analytic.get("48", {})
    n96 = analytic.get("96", {})
    n96_dt = (spatial.get("timestep_refinement_fixed_n") or {}).get("96") or {}
    spat_pair = ((spatial.get("spatial_comparison") or {}).get("pairs") or {}).get("96_vs_48") or {}
    nested_comp = (nested.get("comparisons") or {})
    nested_96_48 = nested_comp.get("96_vs_48") or {}
    op_fixed = (operator.get("operator_order_fixed_time") or operator.get("operator_order") or {}).get("routes") or {}
    op_eq = (operator.get("operator_order_equilibrium") or {}).get("routes") or {}
    p39 = operator.get("point39") or {}

    rows = []
    rows.append(_row(
        "physical_rce_n48_implicit", n48.get("flux_flatness"), PHYSICAL_RCE, "<=",
        "max |F-F_int|/F_scale", "physical_rce", "analytic_opacity_rce.json N=48",
        extra={"tendency_norm": n48.get("tendency_norm"), "status_run": n48.get("status"),
               "steps_accepted": n48.get("steps_accepted"), "rejections": n48.get("rejections"),
               "note": "independent-grid regression (n_phot=16), not nested Richardson"},
    ))
    if n48.get("status") != "converged":
        rows[-1]["status"] = "FAIL"
    rows.append(_row(
        "physical_rce_n96_implicit", n96.get("flux_flatness"), PHYSICAL_RCE, "<=",
        "max |F-F_int|/F_scale", "physical_rce", "analytic_opacity_rce.json N=96",
        extra={"tendency_norm": n96.get("tendency_norm"), "status_run": n96.get("status"),
               "steps_accepted": n96.get("steps_accepted"), "rejections": n96.get("rejections"),
               "note": "independent-grid regression (n_phot=24), not nested Richardson"},
    ))
    if n96.get("status") != "converged":
        rows[-1]["status"] = "FAIL"

    if n192 is None:
        rows.append(_row(
            "physical_rce_n192_implicit", None, PHYSICAL_RCE, "pending",
            "max |F-F_int|/F_scale", "physical_rce", "n192_implicit_rce.json missing",
        ))
        rows[-1]["status"] = "FAIL"
        rows[-1]["note"] = "complete N=192 record not stored"
    else:
        rows.append(_row(
            "physical_rce_n192_implicit", n192.get("flux_flatness"), PHYSICAL_RCE, "<=",
            "max |F-F_int|/F_scale", "physical_rce", "n192_implicit_rce.json",
            extra={
                "status_run": n192.get("status"),
                "steps_accepted": n192.get("steps_accepted"),
                "rejections": n192.get("rejections"),
                "simulated_time": n192.get("simulated_time"),
                "median_accepted_dt": n192.get("median_accepted_dt"),
                "profile_checksum_sha256": n192.get("profile_checksum_sha256") or n192.get("checksum_sha256"),
                "nested_master_layers": n192.get("nested_master_layers"),
                "tendency_norm": n192.get("tendency_norm"),
                "requested_route": n192.get("requested_route"),
                "actual_integrator": n192.get("actual_integrator"),
            },
        ))
        if n192.get("status") != "converged":
            rows[-1]["status"] = "FAIL"
        if n192.get("tendency_norm") is not None and float(n192["tendency_norm"]) > PHYSICAL_RCE:
            rows[-1]["status"] = "FAIL"

    rows.append(_row(
        "timestep_refinement_n96_800_vs_400",
        n96_dt.get("max_rel_T"), 1.0e-6, "<=",
        "max relative T", "timestep", "equal_time_spatial.json",
        extra={"valid": n96_dt.get("valid"), "dlog_rcb": n96_dt.get("dlog_rcb")},
    ))
    if not n96_dt.get("valid"):
        rows[-1]["status"] = "FAIL"

    rows.append(_row(
        "spatial_n96_vs_n48_max_rel_T",
        spat_pair.get("max_rel_T_on_ref_P"), SPATIAL_MAX_REL_T, "<=",
        "max relative T on N=48 P", "spatial", "equal_time_spatial.json",
        extra={"note": "independent grids; Richardson uses nested_rce_family.json"},
    ))
    rcb_obs = spat_pair.get("delta_log10_P_rcb")
    rows.append(_row(
        "spatial_n96_vs_n48_rcb_dex",
        None if rcb_obs is None else abs(float(rcb_obs)), SPATIAL_RCB_DEX, "<=",
        "abs Δ log10 P_RCB", "spatial", "equal_time_spatial.json",
        extra={"note": "independent 0.05-dex RCB gate; not nested Richardson"},
    ))

    if nested_96_48:
        rows.append(_row(
            "spatial_nested_n96_vs_n48_max_rel_T",
            nested_96_48.get("max_rel_T_on_coarse_P"), SPATIAL_MAX_REL_T, "<=",
            "max relative T on nested N=48 P", "spatial", "nested_rce_family.json",
        ))
        if not nested_96_48.get("both_converged"):
            rows[-1]["status"] = "FAIL"
        nested_rcb = nested_96_48.get("delta_log10_P_rcb")
        rows.append(_row(
            "spatial_nested_n96_vs_n48_rcb_dex",
            None if nested_rcb is None else abs(float(nested_rcb)), SPATIAL_RCB_DEX, "<=",
            "abs Δ log10 P_RCB", "spatial", "nested_rce_family.json",
        ))
        if not nested_96_48.get("both_converged"):
            rows[-1]["status"] = "FAIL"
    else:
        rows.append(_row(
            "spatial_nested_n96_vs_n48_max_rel_T", None, SPATIAL_MAX_REL_T, "pending",
            "max relative T on nested N=48 P", "spatial", "nested_rce_family.json missing",
        ))
        rows[-1]["status"] = "FAIL"
        rows.append(_row(
            "spatial_nested_n96_vs_n48_rcb_dex", None, SPATIAL_RCB_DEX, "pending",
            "abs Δ log10 P_RCB", "spatial", "nested_rce_family.json missing",
        ))
        rows[-1]["status"] = "FAIL"

    expl = (op_fixed.get("unsplit_explicit") or {}).get("refinement") or {}
    rows.append(_row(
        "operator_fixed_time_explicit_valid",
        bool(expl.get("valid")), True, "true",
        "all three explicit dts reached t_final", "operator",
        "operator_order_point39.json",
        extra={
            **expl,
            "note": (
                "Genuine operator-order evidence. Implicit reverse/forward/Strang "
                "with coupled_picard=True are route-alias parity of the same "
                "coupled_picard_backward_euler macrostep, not order evidence. "
                "Explicit error ratio 2.20 corresponds to observed order ≈ 1.14."
            ),
        },
    ))
    radc = (op_eq.get("rad_then_implicit_conv") or {})
    rows.append(_row(
        "operator_equilibrium_rad_then_implicit",
        bool(radc.get("reached_gate")), True, "true",
        "1e-3 gate", "operator", "operator_order_point39.json",
        extra={"flux_flatness": radc.get("flux_flatness")},
    ))

    p39_imp = ((p39.get("coupled_semi_implicit") or {}).get("stable_dt") or {})
    rows.append(_row(
        "point39_semi_implicit_has_failed_upper",
        p39_imp.get("failed_upper_dt") is not None, True, "true",
        "multi-step bracket with first failure at X", "39",
        "operator_order_point39.json", extra={"note": p39_imp.get("note")},
    ))

    def _add_algebraic(prefix, rec, source, energy_rel=None):
        ids = _algebraic_from_fluxes(rec)
        if ids is not None:
            rows.append(_row(
                f"algebraic_{prefix}_bottom_boundary_exactness_rel",
                ids["bottom_boundary_exactness_rel"], ALGEBRAIC, "<=",
                "|F_bottom - F_int| / F_int", "algebraic", source,
            ))
            rows.append(_row(
                f"algebraic_{prefix}_telescoping_column_energy_rel",
                ids["telescoping_column_energy_rel"], ALGEBRAIC, "<=",
                "|Σ Δm Q - (F_bot - F_top)| / F_int", "algebraic", source,
            ))
            rows.append(_row(
                f"algebraic_{prefix}_flux_split_identity_rel",
                ids["flux_split_identity_rel"], ALGEBRAIC, "<=",
                "max |F_rad + F_conv - F_total| / F_int", "algebraic", source,
            ))
        if energy_rel is not None:
            rows.append(_row(
                f"algebraic_{prefix}_energy_residual_rel",
                energy_rel, ALGEBRAIC, "<=",
                "last accepted ΣΔm Δh - dt(F_bot-F_top)", "algebraic", source,
                extra={
                    "note": (
                        "Committed-step energy |ΣΔm Δh - Δt(F_bot-F_top)| / E_scale. "
                        "Picard acceptance now requires this 1e-12 (or ULP-floor) gate."
                    ),
                },
            ))

    def _energy_rel(rec):
        if rec is None:
            return None
        committed = rec.get("energy_committed_residual_rel")
        if committed is not None:
            return committed
        return rec.get("energy_residual_rel")

    _add_algebraic("n48", n48, "analytic_opacity_rce.json N=48", _energy_rel(n48))
    _add_algebraic("n96", n96, "analytic_opacity_rce.json N=96", _energy_rel(n96))
    _add_algebraic("n192", n192, "n192_implicit_rce.json", _energy_rel(n192))

    rows.append(_row(
        "helios_parity", False, True, "pending",
        "matched grid/opacity/EOS/g/F_int", "helios", "not run",
    ))
    rows[-1]["status"] = "NOT_RUN"

    by_name = {r["name"]: r for r in rows}
    core_rows = {"physical_rce_n48_implicit", "physical_rce_n96_implicit"}
    spatial_rows = {
        "spatial_n96_vs_n48_max_rel_T",
        "spatial_n96_vs_n48_rcb_dex",
        "spatial_nested_n96_vs_n48_max_rel_T",
        "spatial_nested_n96_vs_n48_rcb_dex",
        "timestep_refinement_n96_800_vs_400",
        "operator_fixed_time_explicit_valid",
        "operator_equilibrium_rad_then_implicit",
        "physical_rce_n192_implicit",
    }
    algebraic_rows = {r["name"] for r in rows if r["category"] == "algebraic"}
    core = all(by_name[n]["status"] == "PASS" for n in core_rows)
    spatial_ok = all(by_name[n]["status"] == "PASS" for n in spatial_rows)
    algebraic_ok = bool(algebraic_rows) and all(by_name[n]["status"] == "PASS" for n in algebraic_rows)
    helios = _helios_status_from_row(by_name["helios_parity"])
    full = (
        core
        and spatial_ok
        and algebraic_ok
        and by_name["helios_parity"]["status"] == "PASS"
    )

    audit = {
        "stage": "4",
        "full_stage4_claim": full,
        "core_single_resolution_status": "PASS" if core else "NOT_PASSED",
        "spatial_and_operator_convergence_status": "PASS" if spatial_ok else "NOT_PASSED",
        "algebraic_identity_status": "PASS" if algebraic_ok else "NOT_PASSED",
        "helios_parity_status": helios,
        "claim_text": (
            "Analytic-opacity N=48 and N=96 implicit RCE reach the physical 1e-3 gate. "
            "Spatial/operator convergence is not passed. Algebraic 1e-12 identities "
            "include committed-step energy closure. HELIOS parity is derived from "
            "the HELIOS row. full_stage4_claim is false."
        ),
        "gates": {
            "algebraic": ALGEBRAIC,
            "physical_rce": PHYSICAL_RCE,
            "spatial_max_rel_T": SPATIAL_MAX_REL_T,
            "spatial_rcb_dex": SPATIAL_RCB_DEX,
        },
        "headline_row_sets": {
            "core_single_resolution": sorted(core_rows),
            "spatial_and_operator": sorted(spatial_rows),
            "algebraic": sorted(algebraic_rows),
            "helios": ["helios_parity"],
        },
        "archive": ARCHIVE.name,
        "rows": rows,
        "n192_record": None if n192 is None else {
            "path": "n192_implicit_rce.json",
            "status": n192.get("status"),
            "flux_flatness": n192.get("flux_flatness"),
            "tendency_norm": n192.get("tendency_norm"),
            "steps_accepted": n192.get("steps_accepted"),
            "rejections": n192.get("rejections"),
            "simulated_time": n192.get("simulated_time"),
            "profile_checksum_sha256": n192.get("profile_checksum_sha256") or n192.get("checksum_sha256"),
            "nested_master_layers": n192.get("nested_master_layers"),
            "requested_route": n192.get("requested_route"),
            "actual_integrator": n192.get("actual_integrator"),
        },
    }
    OUT.write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "core_single_resolution_status": audit["core_single_resolution_status"],
        "spatial_and_operator_convergence_status": audit["spatial_and_operator_convergence_status"],
        "algebraic_identity_status": audit["algebraic_identity_status"],
        "helios_parity_status": audit["helios_parity_status"],
        "full_stage4_claim": audit["full_stage4_claim"],
        "n192_steps": None if n192 is None else n192.get("steps_accepted"),
        "n192_flatness": None if n192 is None else n192.get("flux_flatness"),
        "n192_checksum": None if n192 is None else (
            n192.get("profile_checksum_sha256") or n192.get("checksum_sha256")
        ),
        "out": str(OUT),
    }, indent=2))
    return audit


def assert_n192_audit_sync(audit: dict, record: dict) -> None:
    """Fail if the audit's N=192 profile is not the record just written."""
    rec_hash = record.get("profile_checksum_sha256") or record.get("checksum_sha256")
    audit_hash = (audit.get("n192_record") or {}).get("profile_checksum_sha256")
    rec_steps = record.get("steps_accepted")
    audit_steps = (audit.get("n192_record") or {}).get("steps_accepted")
    if rec_hash != audit_hash or rec_steps != audit_steps:
        raise RuntimeError(
            "stale exit-gate audit: "
            f"record checksum={rec_hash} steps={rec_steps}; "
            f"audit checksum={audit_hash} steps={audit_steps}"
        )


if __name__ == "__main__":
    main()
