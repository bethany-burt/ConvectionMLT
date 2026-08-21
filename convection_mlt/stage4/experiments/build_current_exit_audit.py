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


def main() -> dict:
    analytic = _load(DATA / "analytic_opacity_rce.json") or {}
    spatial = _load(DATA / "equal_time_spatial.json") or {}
    operator = _load(DATA / "operator_order_point39.json") or {}
    n192 = _load(RESULTS / "n192_implicit_rce.json")
    n48 = analytic.get("48", {})
    n96 = analytic.get("96", {})
    n96_dt = (spatial.get("timestep_refinement_fixed_n") or {}).get("96") or {}
    spat_pair = ((spatial.get("spatial_comparison") or {}).get("pairs") or {}).get("96_vs_48") or {}
    op_fixed = (operator.get("operator_order_fixed_time") or operator.get("operator_order") or {}).get("routes") or {}
    op_eq = (operator.get("operator_order_equilibrium") or {}).get("routes") or {}
    p39 = operator.get("point39") or {}

    rows = []
    rows.append(_row(
        "physical_rce_n48_implicit", n48.get("flux_flatness"), PHYSICAL_RCE, "<=",
        "max |F-F_int|/F_scale", "physical_rce", "analytic_opacity_rce.json N=48",
        extra={"tendency_norm": n48.get("tendency_norm"), "status_run": n48.get("status"),
               "steps_accepted": n48.get("steps_accepted"), "rejections": n48.get("rejections")},
    ))
    if n48.get("status") != "converged":
        rows[-1]["status"] = "FAIL"
    rows.append(_row(
        "physical_rce_n96_implicit", n96.get("flux_flatness"), PHYSICAL_RCE, "<=",
        "max |F-F_int|/F_scale", "physical_rce", "analytic_opacity_rce.json N=96",
        extra={"tendency_norm": n96.get("tendency_norm"), "status_run": n96.get("status"),
               "steps_accepted": n96.get("steps_accepted"), "rejections": n96.get("rejections")},
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
                "checksum_sha256": n192.get("checksum_sha256"),
                "nested_master_layers": n192.get("nested_master_layers"),
            },
        ))
        if n192.get("status") != "converged":
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
        extra={"delta_log10_P_rcb": spat_pair.get("delta_log10_P_rcb")},
    ))

    expl = (op_fixed.get("unsplit_explicit") or {}).get("refinement") or {}
    rows.append(_row(
        "operator_fixed_time_explicit_valid",
        bool(expl.get("valid")), True, "true",
        "all three explicit dts reached t_final", "operator",
        "operator_order_point39.json", extra=expl,
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

    rows.append(_row(
        "helios_parity", False, True, "pending",
        "matched grid/opacity/EOS/g/F_int", "helios", "not run",
    ))
    rows[-1]["status"] = "NOT_RUN"

    core_rows = {"physical_rce_n48_implicit", "physical_rce_n96_implicit"}
    spatial_rows = {
        "spatial_n96_vs_n48_max_rel_T",
        "operator_fixed_time_explicit_valid",
        "physical_rce_n192_implicit",
    }
    by_name = {r["name"]: r for r in rows}
    core = all(by_name[n]["status"] == "PASS" for n in core_rows)
    spatial = all(by_name[n]["status"] == "PASS" for n in spatial_rows)
    helios = "NOT_RUN_OR_PILOT_ONLY"
    full = core and spatial and by_name["helios_parity"]["status"] == "PASS"

    audit = {
        "stage": "4",
        "full_stage4_claim": full,
        "core_single_resolution_status": "PASS" if core else "NOT_PASSED",
        "spatial_and_operator_convergence_status": "PASS" if spatial else "NOT_PASSED",
        "helios_parity_status": helios,
        "claim_text": (
            "Analytic-opacity N=48 and N=96 implicit RCE reach the physical 1e-3 gate. "
            "Spatial/operator convergence is not passed. HELIOS parity is not run. "
            "full_stage4_claim is false."
        ),
        "gates": {
            "algebraic": ALGEBRAIC,
            "physical_rce": PHYSICAL_RCE,
            "spatial_max_rel_T": SPATIAL_MAX_REL_T,
            "spatial_rcb_dex": SPATIAL_RCB_DEX,
        },
        "archive": str(ARCHIVE.relative_to(ROOT.parent.parent) if False else ARCHIVE.name),
        "rows": rows,
        "n192_record": None if n192 is None else {
            "path": "n192_implicit_rce.json",
            "status": n192.get("status"),
            "flux_flatness": n192.get("flux_flatness"),
            "steps_accepted": n192.get("steps_accepted"),
            "rejections": n192.get("rejections"),
            "simulated_time": n192.get("simulated_time"),
            "checksum_sha256": n192.get("checksum_sha256"),
            "nested_master_layers": n192.get("nested_master_layers"),
        },
    }
    OUT.write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "core_single_resolution_status": audit["core_single_resolution_status"],
        "spatial_and_operator_convergence_status": audit["spatial_and_operator_convergence_status"],
        "helios_parity_status": audit["helios_parity_status"],
        "full_stage4_claim": audit["full_stage4_claim"],
        "out": str(OUT),
    }, indent=2))
    return audit


if __name__ == "__main__":
    main()
