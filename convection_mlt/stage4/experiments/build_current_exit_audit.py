"""Clean current Stage 4 audit. Historical pre-implicit rows live in the archive file."""

from __future__ import annotations

import hashlib
import json
import shutil
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
ENERGY_GATE = 1.0
ENERGY_ULP_FACTOR = 16.0
RICHARDSON_MIN = 0.25
RICHARDSON_MAX = 3.0


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_helios_smoke():
    """Prefer a scored PASS record. Cluster fixture survives rsync excluding results/."""
    results = _load(RESULTS / "helios_contract_smoke_n8.json")
    cluster = _load(ROOT / "fixtures" / "helios" / "helios_contract_smoke_n8.cluster.json")
    if results is not None and results.get("status") == "PASS":
        return results
    if cluster is not None and cluster.get("status") == "PASS":
        return cluster
    return results


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
    elif criterion == "in_range":
        lo, hi = float(tolerance[0]), float(tolerance[1])
        status = "PASS" if finite and lo < float(observed) < hi else "FAIL"
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


def energy_fields_from_record(rec: dict | None) -> dict[str, float] | None:
    """ULP-aware committed-energy quantities. PASS when energy_gate_ratio ≤ 1."""
    if rec is None:
        return None
    resid = rec.get("energy_committed_residual")
    ulp = rec.get("energy_ulp_floor")
    e_scale = rec.get("energy_scale")
    if e_scale is None and resid is not None and rec.get("energy_residual_rel") not in (None, 0, 0.0):
        rel = abs(float(rec["energy_residual_rel"]))
        if rel > 0.0 and np.isfinite(rel):
            e_scale = abs(float(resid)) / rel
    if resid is None:
        return None
    resid = float(resid)
    if rec.get("energy_allowed") is not None:
        allowed = float(rec["energy_allowed"])
    else:
        if ulp is None and e_scale is None:
            return None
        allowed = max(
            ALGEBRAIC * float(e_scale or 0.0),
            ENERGY_ULP_FACTOR * float(ulp or 0.0),
        )
    if allowed <= 0.0 or not np.isfinite(allowed):
        return None
    ratio = rec.get("energy_gate_ratio")
    if ratio is None:
        ratio = abs(resid) / allowed
    return {
        "energy_committed_residual": resid,
        "energy_scale": None if e_scale is None else float(e_scale),
        "energy_ulp_floor": None if ulp is None else float(ulp),
        "energy_allowed": float(allowed),
        "energy_gate_ratio": float(ratio),
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
    n384 = _load(RESULTS / "n384_implicit_rce.json")
    nested = _load(RESULTS / "nested_rce_family.json") or {}
    n48 = analytic.get("48", {})
    n96 = analytic.get("96", {})
    n96_dt = (spatial.get("timestep_refinement_fixed_n") or {}).get("96") or {}
    spat_pair = ((spatial.get("spatial_comparison") or {}).get("pairs") or {}).get("96_vs_48") or {}
    nested_comp = (nested.get("comparisons") or {})
    nested_96_48 = nested_comp.get("96_vs_48") or {}
    nested_192_96 = nested_comp.get("192_vs_96") or {}
    nested_384_192 = nested_comp.get("384_vs_192") or {}
    if n384 is None:
        n384 = (nested.get("members") or {}).get("384")
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

    if n384 is None:
        rows.append(_row(
            "physical_rce_n384_implicit", None, PHYSICAL_RCE, "pending",
            "max |F-F_int|/F_scale", "physical_rce", "n384_implicit_rce.json missing",
        ))
        rows[-1]["status"] = "FAIL"
        rows[-1]["note"] = "nested N=384 not stored"
    else:
        rows.append(_row(
            "physical_rce_n384_implicit", n384.get("flux_flatness"), PHYSICAL_RCE, "<=",
            "max |F-F_int|/F_scale", "physical_rce", "n384_implicit_rce.json",
            extra={
                "status_run": n384.get("status"),
                "steps_accepted": n384.get("steps_accepted"),
                "tendency_norm": n384.get("tendency_norm"),
                "primary_rcb_log10p": n384.get("primary_rcb_log10p"),
                "convective_regions": n384.get("convective_regions"),
                "detached_convective_regions": n384.get("detached_convective_regions") or [],
                "actual_integrator": n384.get("actual_integrator"),
                "profile_checksum_sha256": n384.get("profile_checksum_sha256") or n384.get("checksum_sha256"),
                "record_checksum_sha256": n384.get("record_checksum_sha256"),
                "source_profile_checksum_sha256": n384.get("source_profile_checksum_sha256"),
            },
        ))
        if n384.get("status") != "converged":
            rows[-1]["status"] = "FAIL"
        if n384.get("tendency_norm") is not None and float(n384["tendency_norm"]) > PHYSICAL_RCE:
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
        "max relative T on N=48 P", "spatial_diagnostic", "equal_time_spatial.json",
        extra={"note": "independent-grid diagnostic; not the spatial exit gate"},
    ))
    rcb_obs = spat_pair.get("delta_log10_P_rcb")
    rows.append(_row(
        "spatial_n96_vs_n48_rcb_dex",
        None if rcb_obs is None else abs(float(rcb_obs)), SPATIAL_RCB_DEX, "<=",
        "abs Δ log10 P_RCB", "spatial_diagnostic", "equal_time_spatial.json",
        extra={"note": "independent-grid diagnostic; not the spatial exit gate"},
    ))

    def _diag_pair(name_t, name_r, pair, source, missing):
        if pair:
            rows.append(_row(
                name_t, pair.get("max_rel_T_on_coarse_P"), SPATIAL_MAX_REL_T, "<=",
                "max relative T on coarse P", "spatial_diagnostic", source,
                extra={"note": "coarse-family diagnostic; not the spatial exit gate"},
            ))
            rcb = pair.get("delta_log10_P_rcb")
            rows.append(_row(
                name_r, None if rcb is None else abs(float(rcb)), SPATIAL_RCB_DEX, "<=",
                "abs Δ log10 P_RCB", "spatial_diagnostic", source,
                extra={"note": "coarse-family diagnostic; not the spatial exit gate"},
            ))
        else:
            rows.append(_row(
                name_t, None, SPATIAL_MAX_REL_T, "pending",
                "max relative T on coarse P", "spatial_diagnostic", missing,
            ))
            rows[-1]["status"] = "NOT_RUN"
            rows.append(_row(
                name_r, None, SPATIAL_RCB_DEX, "pending",
                "abs Δ log10 P_RCB", "spatial_diagnostic", missing,
            ))
            rows[-1]["status"] = "NOT_RUN"

    _diag_pair(
        "spatial_nested_n96_vs_n48_max_rel_T",
        "spatial_nested_n96_vs_n48_rcb_dex",
        nested_96_48, "nested_rce_family.json", "nested_rce_family.json missing 96_vs_48",
    )
    _diag_pair(
        "spatial_nested_n192_vs_n96_max_rel_T",
        "spatial_nested_n192_vs_n96_rcb_dex",
        nested_192_96, "nested_rce_family.json", "nested_rce_family.json missing 192_vs_96",
    )

    probe = _load(RESULTS / "n384_spatial_probe.json")
    probe_latest = (probe or {}).get("latest_comparison") or {}

    if nested_384_192:
        rows.append(_row(
            "spatial_nested_n384_vs_n192_max_rel_T",
            nested_384_192.get("max_rel_T_on_coarse_P"), SPATIAL_MAX_REL_T, "<=",
            "max relative T on nested N=192 P", "spatial", "nested_rce_family.json",
            extra={
                "max_rel_T_index": nested_384_192.get("max_rel_T_index"),
                "max_rel_T_pressure": nested_384_192.get("max_rel_T_pressure"),
                "max_rel_T_log10p": nested_384_192.get("max_rel_T_log10p"),
                "column_enthalpy_rel": nested_384_192.get("column_enthalpy_rel"),
                "topology_agree": nested_384_192.get("topology_agree"),
                "both_physically_gated": nested_384_192.get("both_physically_gated"),
                "n192_checksum": ((nested_384_192.get("coarse") or {}).get("profile_checksum_sha256")),
                "n384_checksum": ((nested_384_192.get("fine") or {}).get("profile_checksum_sha256")),
                "n384_live_record": "n384_implicit_rce.json",
            },
        ))
        if (
            not nested_384_192.get("both_converged")
            or nested_384_192.get("both_physically_gated") is False
            or nested_384_192.get("topology_agree") is False
        ):
            rows[-1]["status"] = "FAIL"
        rcb384 = nested_384_192.get("delta_log10_P_rcb")
        rows.append(_row(
            "spatial_nested_n384_vs_n192_rcb_dex",
            None if rcb384 is None else abs(float(rcb384)), SPATIAL_RCB_DEX, "<=",
            "abs Δ log10 P_RCB", "spatial", "nested_rce_family.json",
            extra={
                "n192_rcb": nested_384_192.get("coarse_rcb"),
                "n384_rcb": nested_384_192.get("fine_rcb"),
                "column_enthalpy_rel": nested_384_192.get("column_enthalpy_rel"),
            },
        ))
        if (
            not nested_384_192.get("both_converged")
            or nested_384_192.get("both_physically_gated") is False
            or nested_384_192.get("topology_agree") is False
        ):
            rows[-1]["status"] = "FAIL"
    elif probe_latest:
        note = (
            "Informal 192→384 from n384_spatial_probe.json. Formal spatial PASS "
            "still requires physically gated N=384 (flatness and tendency ≤ 1e-3)."
        )
        rows.append(_row(
            "spatial_nested_n384_vs_n192_max_rel_T",
            probe_latest.get("max_rel_T_on_coarse_P"), SPATIAL_MAX_REL_T, "<=",
            "max relative T on nested N=192 P", "spatial", "n384_spatial_probe.json",
            extra={
                "note": note,
                "formal_spatial_pass": False,
                "within_T_gate": probe_latest.get("within_T_gate"),
                "n384_physically_gated": False,
            },
        ))
        rows[-1]["status"] = "FAIL"
        rcb384 = probe_latest.get("abs_delta_log10_P_rcb")
        rows.append(_row(
            "spatial_nested_n384_vs_n192_rcb_dex",
            rcb384, SPATIAL_RCB_DEX, "<=",
            "abs Δ log10 P_RCB", "spatial", "n384_spatial_probe.json",
            extra={
                "note": note,
                "formal_spatial_pass": False,
                "within_rcb_gate": probe_latest.get("within_rcb_gate"),
                "n384_physically_gated": False,
            },
        ))
        rows[-1]["status"] = "FAIL"
    else:
        rows.append(_row(
            "spatial_nested_n384_vs_n192_max_rel_T", None, SPATIAL_MAX_REL_T, "pending",
            "max relative T on nested N=192 P", "spatial", "nested_rce_family.json missing 384_vs_192",
        ))
        rows[-1]["status"] = "FAIL"
        rows.append(_row(
            "spatial_nested_n384_vs_n192_rcb_dex", None, SPATIAL_RCB_DEX, "pending",
            "abs Δ log10 P_RCB", "spatial", "nested_rce_family.json missing 384_vs_192",
        ))
        rows[-1]["status"] = "FAIL"

    order_96_192_384 = nested.get("richardson_order_96_192_384")
    if order_96_192_384 is None and nested_192_96 and nested_384_192:
        e_c = nested_192_96.get("max_rel_T_on_coarse_P")
        e_f = nested_384_192.get("max_rel_T_on_coarse_P")
        if (
            nested_192_96.get("both_converged")
            and nested_384_192.get("both_converged")
            and e_c
            and e_f
            and float(e_c) > 0.0
            and float(e_f) > 0.0
        ):
            order_96_192_384 = float(np.log2(float(e_c) / float(e_f)))
    if order_96_192_384 is None:
        rows.append(_row(
            "richardson_nested_96_192_384", None, [RICHARDSON_MIN, RICHARDSON_MAX], "pending",
            f"{RICHARDSON_MIN} < p < {RICHARDSON_MAX}", "spatial",
            "nested_rce_family.json missing 96/192/384",
        ))
        rows[-1]["status"] = "FAIL"
    else:
        rows.append(_row(
            "richardson_nested_96_192_384", float(order_96_192_384),
            [RICHARDSON_MIN, RICHARDSON_MAX], "in_range",
            f"{RICHARDSON_MIN} < p < {RICHARDSON_MAX}", "spatial", "nested_rce_family.json",
            extra={
                "norm": (nested.get("richardson_norm") or {}).get("norm", "max_rel_T"),
                "grid": (nested.get("richardson_norm") or {}).get("grid"),
                "e_192_vs_96": (nested.get("richardson_norm") or {}).get("e_192_vs_96"),
                "e_384_vs_192": (nested.get("richardson_norm") or {}).get("e_384_vs_192"),
                "pairwise_coarse": nested.get("richardson_order_96_192_384_pairwise_coarse"),
            },
        ))

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

    def _add_algebraic(prefix, rec, source):
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
        fields = energy_fields_from_record(rec)
        if fields is None and prefix == "n192" and rec is not None:
            rows.append(_row(
                f"algebraic_{prefix}_energy_gate_ratio", None, ENERGY_GATE, "pending",
                "|R_E| / max(1e-12 E_scale, C_ulp E_ulp)", "algebraic", source,
                extra={"note": "raw committed-energy quantities not serialized; run a restart step"},
            ))
            rows[-1]["status"] = "FAIL"
        if fields is not None:
            rows.append(_row(
                f"algebraic_{prefix}_energy_gate_ratio",
                fields["energy_gate_ratio"], ENERGY_GATE, "<=",
                "|R_E| / max(1e-12 E_scale, C_ulp E_ulp)", "algebraic", source,
                extra={
                    "energy_committed_residual": fields["energy_committed_residual"],
                    "energy_scale": fields["energy_scale"],
                    "energy_ulp_floor": fields["energy_ulp_floor"],
                    "energy_allowed": fields["energy_allowed"],
                    "note": (
                        "Committed-step energy gate: |R_E| ≤ max(1e-12 E_scale, "
                        f"{ENERGY_ULP_FACTOR:g} E_ulp). PASS when energy_gate_ratio ≤ 1."
                    ),
                },
            ))

    _add_algebraic("n48", n48, "analytic_opacity_rce.json N=48")
    _add_algebraic("n96", n96, "analytic_opacity_rce.json N=96")
    _add_algebraic("n192", n192, "n192_implicit_rce.json")
    _add_algebraic("n384", n384, "n384_implicit_rce.json")

    def _helios_rad_row(name: str, path: Path, note: str, *, requires_smoke: bool = True):
        rec = _load(path)
        smoke = _load_helios_smoke()
        smoke_pass = smoke is not None and smoke.get("status") == "PASS"
        if rec is None:
            rows.append(_row(
                name, None, 1.0, "pending",
                "max relative flux/heating residual", "helios_radiation_only", path.name,
                extra={
                    "note": note,
                    "helios_parity_headline": False,
                    "reference_source": f"helios_grid_{path.stem.split('_')[-2]}_reference.json",
                    "smoke_n8_status": (smoke or {}).get("status", "NOT_RUN"),
                },
            ))
            rows[-1]["status"] = "NOT_RUN"
            return
        tol_path = ROOT / "fixtures" / "helios" / "radiation_only_tolerances.json"
        tol = _load(tol_path) or {}
        gate = float((tol.get("gates") or {}).get("flux_net_rel", 1.0))
        observed = (rec.get("metrics") or {}).get("flux_net_rel")
        rows.append(_row(
            name, observed, gate, "<=",
            "max relative flux_net", "helios_radiation_only", path.name,
            extra={
                "comparison_status": rec.get("status"),
                "comparison_type": rec.get("comparison_type"),
                "reference_grid": rec.get("reference_grid", "helios_geometric"),
                "grid_checksum_sha256": rec.get("grid_checksum_sha256"),
                "tolerances_frozen_before_live": rec.get("tolerances_frozen_before_live"),
                "helios_parity_headline": False,
                "smoke_n8_status": (smoke or {}).get("status", "NOT_RUN"),
                "note": note,
            },
        ))
        if requires_smoke and not smoke_pass:
            rows[-1]["status"] = "NOT_RUN"
            rows[-1]["note"] = (rows[-1].get("note") or "") + " blocked until N=8 smoke PASS."
        elif rec.get("status") not in ("PASS",):
            rows[-1]["status"] = "FAIL" if rec.get("status") == "FAIL" else "NOT_RUN"
        if rec.get("comparison_type") == "structural_not_parity":
            rows[-1]["status"] = "NOT_RUN"
            rows[-1]["note"] = (rows[-1].get("note") or "") + " structural; not parity."

    _helios_smoke = _load_helios_smoke()
    smoke_status = (_helios_smoke or {}).get("status", "NOT_RUN")
    smoke_contracts = (_helios_smoke or {}).get("contracts") or {}
    rows.append(_row(
        "helios_radiation_only_smoke_n8",
        smoke_status,
        "PASS",
        "==",
        "infrastructure smoke status",
        "helios_radiation_only",
        "helios_contract_smoke_n8.json",
        extra={
            "contracts": smoke_contracts,
            "radiation_parity_scored": (_helios_smoke or {}).get("radiation_parity_scored", False),
            "helios_runtime_config": (_helios_smoke or {}).get("helios_runtime_config"),
            "helios_parity_headline": False,
            "note": (
                "N=8 infrastructure smoke: grid/tp/orientation/F_intern parameter/"
                "rocky-surface BC/stage3 LowerTemperature(T_boa). Not formal N96 parity."
            ),
        },
    ))
    if _helios_smoke is None:
        rows[-1]["status"] = "NOT_RUN"

    _helios_rad_row(
        "helios_radiation_only_n96_thermal",
        RESULTS / "helios_frozen_rad_n96_thermal.json",
        "N=96 analytic radiation-only on HELIOS geometric grid after HDF5 flatten and microbar pressure fix.",
    )
    _helios_rad_row(
        "helios_radiation_only_n96_irradiated",
        RESULTS / "helios_frozen_rad_n96_irradiated.json",
        "N96-B structural unless beam contract exact",
    )
    _helios_rad_row(
        "helios_radiation_only_n192_thermal",
        RESULTS / "helios_frozen_rad_n192_thermal.json",
        "N=192 radiation-only; opacity-layer gate then flux/heating/energy-increment. Coupled RCE still blocked.",
    )

    rows.append(_row(
        "helios_parity", False, True, "pending",
        "matched grid/opacity/EOS/g/F_int", "helios", "not run",
        extra={
            "note": (
                "Coupled HELIOS RCE benchmark, not solver identity. "
                "Tolerances frozen in coupled_rce_benchmark_tolerances.json "
                "before any live coupled run. Pilot N=96, then headline N=192."
            ),
            "tolerances_path": "fixtures/helios/coupled_rce_benchmark_tolerances.json",
            "helios_parity_headline": False,
        },
    ))
    rows[-1]["status"] = "NOT_RUN"

    coupled_tol = _load(ROOT / "fixtures" / "helios" / "coupled_rce_benchmark_tolerances.json") or {}
    coupled_n96_file = _load(RESULTS / "helios_coupled_rce_n96.json") or {}
    coupled_n192_file = _load(RESULTS / "helios_coupled_rce_n192.json") or {}
    if not coupled_n192_file:
        # Fall back to labelled resolution artifact if present.
        coupled_n192_file = _load(RESULTS / "helios_coupled_n192_resolution_rcb.json") or {}
        if coupled_n192_file and coupled_n192_file.get("status") == "COMPLETE":
            coupled_n192_file = {
                **coupled_n192_file,
                "helios_coupled_rce_n192_status": "RESOLUTION_COMPLETE",
                "status": "RESOLUTION_COMPLETE",
                "source": "helios_coupled_n192_resolution_rcb.json",
                "helios_parity_headline": False,
                "full_stage4_claim": False,
            }
    coupled_n96_st = (
        coupled_n96_file.get("helios_coupled_rce_n96_status")
        or coupled_n96_file.get("status")
        or "NOT_RUN"
    )
    coupled_n192_st = (
        coupled_n192_file.get("helios_coupled_rce_n192_status")
        or coupled_n192_file.get("status")
        or "NOT_RUN"
    )
    # RESOLUTION_COMPLETE is not a parity PASS; overall stays FAIL if N=96 FAIL.
    if coupled_n192_st == "PASS" and coupled_n96_st == "PASS":
        coupled_overall, coupled_headline = "PASS", True
    elif coupled_n96_st == "FAIL" or coupled_n192_st == "FAIL":
        coupled_overall, coupled_headline = "FAIL", False
    elif coupled_n96_st == "PASS":
        coupled_overall, coupled_headline = "PILOT_ONLY", False
    else:
        coupled_overall, coupled_headline = "NOT_RUN", False
    rows.append(_row(
        "helios_coupled_rce_benchmark",
        None,
        (coupled_tol.get("gates") or {}).get("max_rel_T"),
        "pending",
        "predeclared coupled-HELIOS benchmark",
        "helios",
        "coupled_rce_benchmark_tolerances.json",
        extra={
            "status_run": coupled_overall,
            "frozen_before_live": coupled_tol.get("frozen_before_live"),
            "comparison_type": coupled_tol.get("comparison_type"),
            "benchmark_interpretation": coupled_tol.get("benchmark_interpretation"),
            "forcing": coupled_tol.get("forcing"),
            "f_irr": coupled_tol.get("f_irr"),
            "mlt_grid": coupled_tol.get("mlt_grid"),
            "helios_grid": coupled_tol.get("helios_grid"),
            "irradiated_nested_mlt": coupled_tol.get("irradiated_nested_mlt"),
            "pilot_resolution": coupled_tol.get("pilot_resolution"),
            "headline_resolution": coupled_tol.get("headline_resolution"),
            "gates": coupled_tol.get("gates"),
            "helios_coupled_rce_n96_status": coupled_n96_st,
            "helios_coupled_rce_n192_status": coupled_n192_st,
            "helios_coupled_rce_status": coupled_overall,
            "helios_parity_headline": coupled_headline,
            "note": coupled_tol.get("note"),
        },
    ))
    rows[-1]["status"] = coupled_overall

    by_name = {r["name"]: r for r in rows}
    core_rows = {"physical_rce_n48_implicit", "physical_rce_n96_implicit"}
    spatial_rows = {
        "physical_rce_n192_implicit",
        "physical_rce_n384_implicit",
        "spatial_nested_n384_vs_n192_max_rel_T",
        "spatial_nested_n384_vs_n192_rcb_dex",
        "richardson_nested_96_192_384",
        "timestep_refinement_n96_800_vs_400",
        "operator_fixed_time_explicit_valid",
        "operator_equilibrium_rad_then_implicit",
    }
    algebraic_rows = {r["name"] for r in rows if r["category"] == "algebraic"}
    core = all(by_name[n]["status"] == "PASS" for n in core_rows)
    spatial_ok = all(by_name[n]["status"] == "PASS" for n in spatial_rows)
    algebraic_ok = bool(algebraic_rows) and all(by_name[n]["status"] == "PASS" for n in algebraic_rows)
    internal_complete = bool(core and spatial_ok and algebraic_ok)
    helios = _helios_status_from_row(by_name["helios_parity"])
    n96_st = by_name["helios_radiation_only_n96_thermal"]["status"]
    n192_st = by_name["helios_radiation_only_n192_thermal"]["status"]
    adapter_st = by_name["helios_radiation_only_smoke_n8"]["status"]
    rad_parity = (
        "PASS" if n96_st == "PASS" and n192_st == "PASS"
        else "FAIL" if n96_st == "FAIL" or n192_st == "FAIL"
        else "NOT_RUN"
    )

    audit = {
        "stage": "4",
        "full_stage4_claim": bool(internal_complete and coupled_headline),
        "coupled_helios_rce_claimed": coupled_headline,
        "helios_parity_headline": coupled_headline,
        "helios_parity_headline_means": "coupled_helios_rce_parity",
        "helios_radiation_only_parity_status": rad_parity,
        "helios_adapter_contract_status": adapter_st,
        "helios_radiation_only_n96_status": n96_st,
        "helios_radiation_only_n192_status": n192_st,
        "helios_coupled_rce_n96_status": coupled_n96_st,
        "helios_coupled_rce_n192_status": coupled_n192_st,
        "helios_coupled_rce_status": coupled_overall,
        "internal_numerical_track_complete": internal_complete,
        "core_single_resolution_status": "PASS" if core else "NOT_PASSED",
        "spatial_and_operator_convergence_status": "PASS" if spatial_ok else "NOT_PASSED",
        "algebraic_identity_status": "PASS" if algebraic_ok else "NOT_PASSED",
        "helios_parity_status": helios,
        "claim_text": (
            "Independent-grid N=48/96 are the core single-resolution gate. "
            "Spatial exit evidence is nested N=192 and N=384 at 1e-3, the "
            "192→384 2%/0.05-dex pair, and a positive 96/192/384 Richardson "
            "order. Coarse 48→96 and 96→192 rows are diagnostics only. "
            "Committed energy uses |R_E|/allowed ≤ 1, not residual_rel ≤ 1e-12. "
            "helios_radiation_only_parity_status is N=96 and N=192 radiation-only. "
            "helios_parity_headline is coupled HELIOS RCE only. It becomes true "
            "only after N=192 coupled PASS. N=96 coupled PASS is PILOT_ONLY. "
            "full_stage4_claim requires the internal nested 192→384 track and "
            "the N=192 coupled headline. Irradiated nested MLT is structural only; "
            "the coupled benchmark is F_irr=0 on the HELIOS geometric grid."
        ),
        "gates": {
            "algebraic": ALGEBRAIC,
            "energy_gate_ratio": ENERGY_GATE,
            "energy_ulp_factor": ENERGY_ULP_FACTOR,
            "physical_rce": PHYSICAL_RCE,
            "spatial_max_rel_T": SPATIAL_MAX_REL_T,
            "spatial_rcb_dex": SPATIAL_RCB_DEX,
            "richardson_min": RICHARDSON_MIN,
            "richardson_max": RICHARDSON_MAX,
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
            "record_checksum_sha256": n192.get("record_checksum_sha256"),
            "nested_master_layers": n192.get("nested_master_layers"),
            "requested_route": n192.get("requested_route"),
            "actual_integrator": n192.get("actual_integrator"),
        },
        "n384_record": None if n384 is None else {
            "path": "n384_implicit_rce.json",
            "status": n384.get("status"),
            "flux_flatness": n384.get("flux_flatness"),
            "tendency_norm": n384.get("tendency_norm"),
            "steps_accepted": n384.get("steps_accepted"),
            "primary_rcb_log10p": n384.get("primary_rcb_log10p"),
            "convective_regions": n384.get("convective_regions"),
            "actual_integrator": n384.get("actual_integrator"),
            "profile_checksum_sha256": n384.get("profile_checksum_sha256") or n384.get("checksum_sha256"),
            "record_checksum_sha256": n384.get("record_checksum_sha256"),
            "source_profile_checksum_sha256": n384.get("source_profile_checksum_sha256"),
        },
        "nested_spatial_384_vs_192": nested_384_192 or None,
        "richardson_order_96_192_384": order_96_192_384,
        "coupled_helios_benchmark_tolerances": coupled_tol.get("gates"),
    }
    OUT.write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n")
    _write_status_report(audit, n192, n384)
    bundle = _write_gated_spatial_bundle(audit)
    print(json.dumps({
        "core_single_resolution_status": audit["core_single_resolution_status"],
        "spatial_and_operator_convergence_status": audit["spatial_and_operator_convergence_status"],
        "algebraic_identity_status": audit["algebraic_identity_status"],
        "internal_numerical_track_complete": audit["internal_numerical_track_complete"],
        "helios_parity_status": audit["helios_parity_status"],
        "helios_radiation_only_parity_status": audit["helios_radiation_only_parity_status"],
        "helios_adapter_contract_status": audit["helios_adapter_contract_status"],
        "helios_radiation_only_n96_status": audit["helios_radiation_only_n96_status"],
        "helios_radiation_only_n192_status": audit["helios_radiation_only_n192_status"],
        "helios_coupled_rce_n96_status": audit["helios_coupled_rce_n96_status"],
        "helios_coupled_rce_n192_status": audit["helios_coupled_rce_n192_status"],
        "helios_coupled_rce_status": audit["helios_coupled_rce_status"],
        "helios_parity_headline": audit["helios_parity_headline"],
        "full_stage4_claim": audit["full_stage4_claim"],
        "n192_steps": None if n192 is None else n192.get("steps_accepted"),
        "n192_flatness": None if n192 is None else n192.get("flux_flatness"),
        "n192_checksum": None if n192 is None else (
            n192.get("profile_checksum_sha256") or n192.get("checksum_sha256")
        ),
        "n384_checksum": None if n384 is None else (
            n384.get("profile_checksum_sha256") or n384.get("checksum_sha256")
        ),
        "bundle": None if bundle is None else str(bundle),
        "out": str(OUT),
    }, indent=2))
    return audit


def _row_from_audit(audit: dict, name: str) -> dict:
    return next((r for r in audit["rows"] if r["name"] == name), {})


def _write_status_report(audit: dict, n192: dict | None, n384: dict | None) -> None:
    spat_t = _row_from_audit(audit, "spatial_nested_n384_vs_n192_max_rel_T")
    spat_r = _row_from_audit(audit, "spatial_nested_n384_vs_n192_rcb_dex")
    rich = _row_from_audit(audit, "richardson_nested_96_192_384")
    e192 = _row_from_audit(audit, "algebraic_n192_energy_gate_ratio")
    e384 = _row_from_audit(audit, "algebraic_n384_energy_gate_ratio")
    phys384 = _row_from_audit(audit, "physical_rce_n384_implicit")
    coupled = _row_from_audit(audit, "helios_coupled_rce_benchmark")
    pair = audit.get("nested_spatial_384_vs_192") or {}
    coarse = pair.get("coarse") or {}
    fine = pair.get("fine") or {}
    nested = _load(RESULTS / "nested_rce_family.json") or {}
    checksums = nested.get("member_checksums") or {}
    n96_meta = checksums.get("96") or {}
    n192_meta = checksums.get("192") or {}
    n384_meta = checksums.get("384") or {}
    if audit.get("helios_parity_headline"):
        next_step = (
            "N=192 coupled HELIOS headline passed. full_stage4_claim follows "
            "if the internal numerical track is also complete."
        )
    elif audit.get("helios_coupled_rce_status") == "PILOT_ONLY":
        next_step = (
            "N=96 coupled HELIOS pilot passed. Repeat the matched F_irr=0 "
            "HELIOS-grid MLT construction at N=192, then run the headline."
        )
    elif audit.get("internal_numerical_track_complete"):
        next_step = (
            "Internal Stage-4 numerical track is complete. Next: freeze the "
            "F_irr=0 HELIOS-grid MLT N=96 reference, then the N=96 coupled "
            "HELIOS pilot. Do not submit GPU until the coupled manifest verifies. "
            "full_stage4_claim stays false until N=192 coupled PASS."
        )
    else:
        next_step = "Coupled HELIOS RCE waits on the 192→384 spatial and N=384 energy rows."
    (ROOT / "STAGE4_STATUS_REPORT.txt").write_text(
        "Stage 4 status report\n"
        "Fixed-composition H2 radiative-convective equilibrium (handbook points 35-40)\n"
        "\n"
        "Current claim (auto-rebuilt after nested 192→384 scoring)\n"
        "--------------------------------------------------------------------\n"
        f"core_single_resolution_status: {audit['core_single_resolution_status']}\n"
        f"spatial_and_operator_convergence_status: {audit['spatial_and_operator_convergence_status']}\n"
        f"algebraic_identity_status: {audit['algebraic_identity_status']}\n"
        f"internal_numerical_track_complete: {str(audit.get('internal_numerical_track_complete')).lower()}\n"
        f"helios_parity_status: {audit['helios_parity_status']}\n"
        f"helios_radiation_only_parity_status: {audit['helios_radiation_only_parity_status']}\n"
        f"helios_coupled_rce_n96_status: {audit.get('helios_coupled_rce_n96_status')}\n"
        f"helios_coupled_rce_n192_status: {audit.get('helios_coupled_rce_n192_status')}\n"
        f"helios_coupled_rce_status: {audit['helios_coupled_rce_status']}\n"
        f"full_stage4_claim: {str(audit['full_stage4_claim']).lower()}\n"
        "\n"
        "HELIOS N=192 note: labelled resolution study is RESOLUTION_COMPLETE\n"
        "(ΔRCB vs N=96 ≈ 0.019 dex). That is not a 0.15-dex parity PASS; the\n"
        "frozen coupled RCB gate remains FAIL.\n"
        "\n"
        "Statuses are computed from current rows in stage4/results/exit_gate_audit.json.\n"
        "The live N=384 source of truth is n384_implicit_rce.json (five-check polish),\n"
        "not the discrete-RZ accelerator snapshot.\n"
        "\n"
        "Gates\n"
        "  algebraic identities: 1e-12, including committed-step energy\n"
        "  physical RCE: 1e-3 (unchanged; not relaxed)\n"
        "  spatial: max rel T 0.02 and 0.05 dex RCB; Richardson in (0.25, 3)\n"
        "  coupled HELIOS: predeclared benchmark tolerances; not solver identity\n"
        "\n"
        "N=96 nested member\n"
        f"  physically_gated: {n96_meta.get('physically_gated')}\n"
        f"  flux_flatness: {n96_meta.get('flux_flatness')}\n"
        f"  tendency_norm: {n96_meta.get('tendency_norm')}\n"
        f"  primary_rcb_log10p: {n96_meta.get('primary_rcb_log10p')}\n"
        f"  convective_regions: {n96_meta.get('convective_regions')}\n"
        f"  profile_checksum_sha256: {n96_meta.get('profile_checksum_sha256')}\n"
        "\n"
        "N=192 live record\n"
        f"  status: {None if n192 is None else n192.get('status')}\n"
        f"  physically_gated: {n192_meta.get('physically_gated')}\n"
        f"  steps_accepted: {None if n192 is None else n192.get('steps_accepted')}\n"
        f"  flux_flatness: {None if n192 is None else n192.get('flux_flatness')}\n"
        f"  tendency_norm: {None if n192 is None else n192.get('tendency_norm')}\n"
        f"  primary_rcb_log10p: {None if n192 is None else n192.get('primary_rcb_log10p')}\n"
        f"  convective_regions: {None if n192 is None else n192.get('convective_regions')}\n"
        f"  energy_gate_ratio: {None if n192 is None else n192.get('energy_gate_ratio')} "
        f"({e192.get('status')})\n"
        f"  profile_checksum_sha256: {None if n192 is None else (n192.get('profile_checksum_sha256') or n192.get('checksum_sha256'))}\n"
        f"  record_checksum_sha256: {None if n192 is None else n192.get('record_checksum_sha256')}\n"
        "\n"
        "N=384 live record (five-check polish)\n"
        f"  status: {None if n384 is None else n384.get('status')}\n"
        f"  physically_gated: {n384_meta.get('physically_gated')}\n"
        f"  steps_accepted: {None if n384 is None else n384.get('steps_accepted')}\n"
        f"  actual_integrator: {None if n384 is None else n384.get('actual_integrator')}\n"
        f"  flux_flatness: {None if n384 is None else n384.get('flux_flatness')}\n"
        f"  tendency_norm: {None if n384 is None else n384.get('tendency_norm')}\n"
        f"  primary_rcb_log10p: {None if n384 is None else n384.get('primary_rcb_log10p')}\n"
        f"  convective_regions: {None if n384 is None else n384.get('convective_regions')}\n"
        f"  energy_gate_ratio: {None if n384 is None else n384.get('energy_gate_ratio')} "
        f"({e384.get('status')})\n"
        f"  physical_rce_n384_implicit: {phys384.get('status')} "
        f"(observed {phys384.get('observed')})\n"
        f"  profile_checksum_sha256: {None if n384 is None else (n384.get('profile_checksum_sha256') or n384.get('checksum_sha256'))}\n"
        f"  record_checksum_sha256: {None if n384 is None else n384.get('record_checksum_sha256')}\n"
        "\n"
        "Nested 192→384 spatial exit pair\n"
        f"  max_rel_T: {spat_t.get('status')} (observed {spat_t.get('observed')})\n"
        f"  location_index: {spat_t.get('max_rel_T_index')}\n"
        f"  location_P: {spat_t.get('max_rel_T_pressure')} Pa\n"
        f"  location_log10P: {spat_t.get('max_rel_T_log10p')}\n"
        f"  rcb_dex: {spat_r.get('status')} (observed {spat_r.get('observed')})\n"
        f"  n192_rcb: {spat_r.get('n192_rcb')}\n"
        f"  n384_rcb: {spat_r.get('n384_rcb')}\n"
        f"  column_enthalpy_rel: {pair.get('column_enthalpy_rel')}\n"
        f"  topology_agree: {pair.get('topology_agree')}\n"
        f"  n192_single_bottom_cz: {coarse.get('single_bottom_cz')}\n"
        f"  n384_single_bottom_cz: {fine.get('single_bottom_cz')}\n"
        f"  both_physically_gated: {pair.get('both_physically_gated')}\n"
        f"  richardson_nested_96_192_384: {rich.get('status')} "
        f"(observed {rich.get('observed')}; common-grid max_rel_T)\n"
        f"  e_192_vs_96: {rich.get('e_192_vs_96')}\n"
        f"  e_384_vs_192: {rich.get('e_384_vs_192')}\n"
        "\n"
        "Coupled HELIOS RCE (next, not claimed until N=192)\n"
        f"  n96_status: {audit.get('helios_coupled_rce_n96_status')}\n"
        f"  n192_status: {audit.get('helios_coupled_rce_n192_status')}\n"
        f"  overall: {audit['helios_coupled_rce_status']}\n"
        f"  forcing: {coupled.get('forcing')}\n"
        f"  mlt_grid: {coupled.get('mlt_grid')}\n"
        f"  irradiated_nested_mlt: {coupled.get('irradiated_nested_mlt')}\n"
        f"  comparison_type: {coupled.get('comparison_type')}\n"
        f"  frozen_before_live: {coupled.get('frozen_before_live')}\n"
        f"  pilot: N={coupled.get('pilot_resolution')}; headline: N={coupled.get('headline_resolution')}\n"
        f"  gates: {coupled.get('gates')}\n"
        f"  {next_step}\n"
        "\n"
        "Radiation scope (isothermal cell-centred source)\n"
        "--------------------------------------------------------------------\n"
        "Radiation uses a piecewise-isothermal, cell-centred source\n"
        "representation. This is a consistent finite-volume discretization\n"
        "whose adequacy is established through optical-depth-adapted grids\n"
        "and demonstrated spatial convergence. It is not equivalent at finite\n"
        "resolution to HELIOS non-isothermal within-layer source reconstruction.\n"
        "The HELIOS N=96 iso=yes vs iso=no+adj counterfactual (different CZ\n"
        "topology, ~145 K deep-T shift) warns that the radiative-convective\n"
        "transition is sensitive to within-layer source treatment on a coarse\n"
        "geometric grid; that result does not transfer directly to the nested-τ\n"
        "internal track.\n"
        "\n"
        "Stage-5 radiation source sensitivity (future requirement)\n"
        "--------------------------------------------------------------------\n"
        "Implement or construct a linear-in-optical-depth source diagnostic;\n"
        "compare it with the constant-source calculation on the same frozen\n"
        "T(P); repeat at N=96, 192 and 384; measure changes in flux, T(P), and\n"
        "RCB. Adequacy criterion: whether the two reconstructions converge as\n"
        "max Δτ decreases. If the RCB difference falls below the spatial gate\n"
        "with refinement, the present scheme is fully adequate; if a persistent\n"
        "nonzero offset remains, production radiation should gain a\n"
        "non-isothermal source option.\n"
        "\n"
        "MLT sensitivity campaign (internal track evidence)\n"
        "--------------------------------------------------------------------\n"
        "See stage4/results/mlt_sensitivity/mlt_sensitivity_summary.json and\n"
        "mlt_sensitivity_completion.json.\n"
        "Local closure PASS; N=96 α-sweep PASS with Δ∇∝α^{-4/3}; prescribed-Δt\n"
        "ladder PASS (order ~1.59, constant histories); broad IC basin PASS\n"
        "(RC / RE / ±5%); N=192 α=0.5,1,2 formally gated after discrete-RZ +\n"
        "five-check polish. Prior dt_hold_init ladder SUPERSEDED (adaptive grew\n"
        "Δt). Exact HELIOS RCB agreement is neither expected nor required; keep\n"
        "the 0.15-dex HELIOS row FAIL. User-facing runner:\n"
        "examples/rce/run_rce.py + cfg_demo.py "
        "(production discrete-RZ + five-check; frozen 1e-3 gate).\n"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_gated_spatial_bundle(audit: dict) -> Path:
    bundle = RESULTS / "n384_gated_spatial_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    copies = {
        "n384_implicit_rce.json": RESULTS / "n384_implicit_rce.json",
        "nested_rce_family.json": RESULTS / "nested_rce_family.json",
        "exit_gate_audit.json": OUT,
        "STAGE4_STATUS_REPORT.txt": ROOT / "STAGE4_STATUS_REPORT.txt",
        "n384_discrete_rz_rce.json": RESULTS / "n384_discrete_rz_rce.json",
        "n384_discrete_rz_flux_profiles.json": RESULTS / "n384_discrete_rz_flux_profiles.json",
        "nested_spatial_1e-3.png": ROOT / "plots" / "generated" / "nested_spatial_1e-3.png",
        "coupled_rce_benchmark_tolerances.json": (
            ROOT / "fixtures" / "helios" / "coupled_rce_benchmark_tolerances.json"
        ),
    }
    files = {}
    for name, src in copies.items():
        if not src.exists():
            continue
        dest = bundle / name
        shutil.copy2(src, dest)
        files[name] = {"bytes": dest.stat().st_size, "sha256": _sha256_file(dest)}
    n192 = RESULTS / "n192_implicit_rce.json"
    n9500 = RESULTS / "n384_implicit_rce_9500.json"
    index = {
        "purpose": (
            "Gated N=384 spatial source of truth plus nested 96/192/384 scoring. "
            "The live record is the five-check polish n384_implicit_rce.json "
            "(profile 5e0bd…), not the discrete-RZ accelerator snapshot."
        ),
        "n384_live_record": "n384_implicit_rce.json",
        "n384_profile_checksum_sha256": (audit.get("n384_record") or {}).get(
            "profile_checksum_sha256"
        ),
        "n384_record_checksum_sha256": (audit.get("n384_record") or {}).get(
            "record_checksum_sha256"
        ),
        "n192_profile_checksum_sha256": (audit.get("n192_record") or {}).get(
            "profile_checksum_sha256"
        ),
        "n192_implicit_record_excluded": True,
        "n192_implicit_record_bytes": n192.stat().st_size if n192.exists() else None,
        "n384_9500_archive_excluded": True,
        "n384_9500_archive_sha256": _sha256_file(n9500) if n9500.exists() else None,
        "accelerator_snapshot": "n384_discrete_rz_rce.json",
        "internal_numerical_track_complete": audit.get("internal_numerical_track_complete"),
        "core_single_resolution_status": audit.get("core_single_resolution_status"),
        "spatial_and_operator_convergence_status": audit.get(
            "spatial_and_operator_convergence_status"
        ),
        "algebraic_identity_status": audit.get("algebraic_identity_status"),
        "helios_radiation_only_parity_status": audit.get("helios_radiation_only_parity_status"),
        "helios_coupled_rce_n96_status": audit.get("helios_coupled_rce_n96_status"),
        "helios_coupled_rce_n192_status": audit.get("helios_coupled_rce_n192_status"),
        "helios_coupled_rce_status": audit.get("helios_coupled_rce_status"),
        "helios_parity_headline": audit.get("helios_parity_headline"),
        "full_stage4_claim": audit.get("full_stage4_claim"),
        "files": files,
    }
    (bundle / "bundle_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return bundle


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
