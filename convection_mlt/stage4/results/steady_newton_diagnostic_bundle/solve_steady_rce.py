"""Steady flux-defect Newton–Krylov: invariance, N=384 solve, five-check polish.

Does not launch coupled HELIOS. Does not restart N=384 from N=192 interpolation.
Does not overwrite n384_implicit_rce.json unless the five-check polish converges.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    RCERoute,
    SteadyRCEConfig,
    TopIrradiation,
    nested_analytic_opacity_spec,
    solve_adaptive_rce,
    solve_steady_rce,
)
from convection_mlt.rce import _primary_rcb_log10p
from convection_mlt.config import SolverConfig

from rce_record import (
    PHYSICAL_GATE,
    algebraic_identities,
    dumps,
    finalize_record,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
    verify_record_checksums,
    _sha256_arrays,
)

RESULTS = ROOT / "results"
N192 = RESULTS / "n192_implicit_rce.json"
N384 = RESULTS / "n384_implicit_rce.json"
N384_ARCHIVE = RESULTS / "n384_implicit_rce_9500.json"
NESTED = RESULTS / "nested_rce_family.json"
OUT = RESULTS / "steady_rce_solve.json"
N384_STEADY = RESULTS / "n384_steady_rce.json"
N384_RESIDUAL = RESULTS / "n384_steady_residual.json"
LIVE_CHECKSUM = "1e23c0a90bf8125db3a61720e8f05b5f5d7cf80680597dde9fa289b5cbf70511"


def _max_rel(a, b, floor=1.0) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(aa), floor)
    return float(np.max(np.abs(bb - aa) / scale))


def _progress(event: dict) -> None:
    kind = event.get("event")
    if kind == "newton":
        print(
            f"  newton outer={event['outer']} it={event['newton']} "
            f"flatness={event['flux_flatness']:.4e} "
            f"(was {event.get('flux_flatness_before', float('nan')):.4e}) "
            f"tendency={event['tendency_norm']:.4e} "
            f"alpha={event['alpha']:.3g} reason={event.get('line_search_reason')} "
            f"merit={event.get('merit_after', float('nan')):.4e} "
            f"dT={event.get('step_rel_T_accepted', float('nan')):.3g} "
            f"dir={event.get('direction')} "
            f"gmres={event['gmres_iters']}/{event.get('gmres_success')} "
            f"evals={event['n_evals']}",
            flush=True,
        )
    elif kind == "mask":
        print(
            f"  mask outer={event['outer']} changed={event['mask_changed']} "
            f"inner_ok={event['inner_ok']} regions={event['regions']} "
            f"detached={event['detached']} flatness={event['flux_flatness']:.4e}",
            flush=True,
        )
    else:
        print(f"  {kind}: {event}", flush=True)


def _history(res) -> list[dict]:
    return [asdict(rec) for rec in res.history]


def _identities(res, f_int: float) -> dict[str, float]:
    return algebraic_identities(
        {
            "flux_total": res.flux_total,
            "flux_rad": res.flux_rad,
            "flux_conv": res.flux_conv,
            "mass_path": res.state.mass_path,
            "f_int": f_int,
        }
    )


def _config_dict(cfg: SteadyRCEConfig) -> dict:
    payload = asdict(cfg)
    payload["radiation_route"] = cfg.radiation_route.value
    payload["jv_formula"] = "eps = fd_rel * ||h|| / ||direction||; subspace uses centred live MLT at subspace_fd_rel"
    payload["gmres_unknown_scaling"] = "dh = h_scale * v, h_scale = max(|h|, h_floor)"
    payload["subspace_jv_uses_live_mlt"] = True
    payload["gmres_jv_uses_lagged_f_conv"] = True
    payload["line_search"] = "Armijo on Φ=½||r||₂² with tendency safeguard"
    payload["line_search_uses_live_mlt"] = True
    return payload


def _complete_payload(
    res, spec, solver: SolverConfig, source: dict, summary: dict, cfg: SteadyRCEConfig
) -> dict:
    payload = {
        "n_layers": spec.n_layers,
        "solver": "steady_flux_defect_newton_krylov",
        "source_steps_accepted": source.get("steps_accepted"),
        "source_profile_checksum_sha256": source.get("profile_checksum_sha256"),
        "source_flux_flatness": source.get("flux_flatness"),
        "source_tendency_norm": source.get("tendency_norm"),
        "status": summary["status"],
        "reason": summary["reason"],
        "flux_flatness": res.flux_flatness,
        "tendency_norm": res.tendency_norm,
        "primary_rcb_log10p": summary["primary_rcb_log10p"],
        "convective_regions": res.convective_regions,
        "detached_convective_regions": res.detached_convective_regions,
        "frozen_support": np.asarray(res.frozen_support, dtype=bool).tolist(),
        "temperature": res.state.temperature.tolist(),
        "enthalpy": res.state.enthalpy.tolist(),
        "mass_path": res.state.mass_path.tolist(),
        "pressure_centres": spec.grid().pressure_centres.tolist(),
        "pressure_edges": spec.grid().pressure_edges.tolist(),
        "flux_total": res.flux_total.tolist(),
        "flux_rad": res.flux_rad.tolist(),
        "flux_conv": res.flux_conv.tolist(),
        "residual": res.residual.tolist(),
        "f_int": spec.f_int,
        "f_irr": spec.f_irr,
        "newton_iterations": res.newton_iterations,
        "line_search_backtracks": res.line_search_backtracks,
        "mask_outer_iterations": res.mask_outer_iterations,
        "n_evals": res.n_evals,
        "history": summary["history"],
        "identities": summary["identities"],
        "max_rel_T_from_source": summary["max_rel_T"],
        "physical_gate": PHYSICAL_GATE,
        "steady_config": _config_dict(cfg),
        "wall_time_s": summary.get("wall_time_s"),
    }
    payload["profile_checksum_sha256"] = _sha256_arrays(
        np.asarray(payload["temperature"], dtype=np.float64),
        np.asarray(payload["pressure_centres"], dtype=np.float64),
        np.asarray(payload["flux_total"], dtype=np.float64),
        np.asarray(payload["flux_rad"], dtype=np.float64),
        np.asarray(payload["flux_conv"], dtype=np.float64),
        np.asarray(payload["enthalpy"], dtype=np.float64),
        np.asarray(payload["mass_path"], dtype=np.float64),
        np.asarray(payload["pressure_edges"], dtype=np.float64),
    )
    payload.update(algebraic_identities(payload))
    return payload


def _summarise(res, spec, t0, solver: SolverConfig) -> dict:
    rcb = _primary_rcb_log10p(spec.grid(), res.closure, solver)
    return {
        "status": res.status.value,
        "reason": res.reason,
        "flux_flatness": res.flux_flatness,
        "tendency_norm": res.tendency_norm,
        "newton_iterations": res.newton_iterations,
        "line_search_backtracks": res.line_search_backtracks,
        "mask_outer_iterations": res.mask_outer_iterations,
        "n_evals": res.n_evals,
        "convective_regions": res.convective_regions,
        "detached_convective_regions": res.detached_convective_regions,
        "primary_rcb_log10p": rcb,
        "max_rel_T": _max_rel(t0, res.state.temperature),
        "bottom_flux_exactness_rel": abs(float(res.flux_total[0]) - spec.f_int) / spec.f_int,
        "identities": _identities(res, spec.f_int),
        "history": _history(res),
    }


def _load_n96() -> dict:
    members = (json.loads(NESTED.read_text()).get("members") or {})
    if "96" not in members:
        raise FileNotFoundError("nested_rce_family.json has no N=96 member")
    return members["96"]


def _solve_record(n_layers: int, record: dict, cfg: SteadyRCEConfig) -> tuple:
    spec = nested_analytic_opacity_spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    t0 = np.asarray(record["temperature"], dtype=np.float64)
    h0 = np.asarray(record["enthalpy"], dtype=np.float64) if record.get("enthalpy") is not None else None
    wall0 = time.perf_counter()
    res = solve_steady_rce(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        config=cfg,
        initial_enthalpy=h0,
        progress=_progress,
    )
    wall = time.perf_counter() - wall0
    summary = _summarise(res, spec, t0, solver)
    summary["wall_time_s"] = wall
    summary["initial_flux_flatness"] = float(record.get("flux_flatness") or np.nan)
    summary["initial_profile_checksum"] = record.get("profile_checksum_sha256")
    return res, spec, solver, summary


def _invariance(n_layers: int, record: dict) -> dict:
    print(f"\n=== invariance N={n_layers} (physical gate {PHYSICAL_GATE:g}) ===", flush=True)
    cfg_gate = SteadyRCEConfig(
        flux_flatness_tolerance=PHYSICAL_GATE,
        tendency_tolerance=PHYSICAL_GATE,
        max_newton=12,
        max_mask_outer=4,
    )
    _, _, _, at_gate = _solve_record(n_layers, record, cfg_gate)
    current = float(record.get("flux_flatness") or PHYSICAL_GATE)
    refine_tol = min(PHYSICAL_GATE, max(3.0e-4, 0.5 * current))
    print(f"\n=== invariance N={n_layers} refine to {refine_tol:g} ===", flush=True)
    cfg_refine = SteadyRCEConfig(
        flux_flatness_tolerance=refine_tol,
        tendency_tolerance=refine_tol,
        max_newton=20,
        max_mask_outer=4,
    )
    _, _, _, refined = _solve_record(n_layers, record, cfg_refine)
    unchanged = (
        at_gate["max_rel_T"] < 1.0e-5
        and at_gate["flux_flatness"] <= PHYSICAL_GATE + 1.0e-12
        and not at_gate["detached_convective_regions"]
        and refined["max_rel_T"] < 1.0e-3
    )
    return {
        "n_layers": n_layers,
        "at_physical_gate": at_gate,
        "refined": refined,
        "refine_tolerance": refine_tol,
        "essentially_unchanged": unchanged,
    }


def _write_n384_snapshot(
    res, spec, solver, source: dict, summary: dict, cfg: SteadyRCEConfig | None = None
) -> dict:
    payload = _complete_payload(
        res, spec, solver, source, summary, cfg or SteadyRCEConfig()
    )
    N384_STEADY.write_text(dumps(finalize_record(payload)))
    compact = {
        "status": payload["status"],
        "flux_flatness": payload["flux_flatness"],
        "tendency_norm": payload["tendency_norm"],
        "primary_rcb_log10p": payload["primary_rcb_log10p"],
        "convective_regions": payload["convective_regions"],
        "detached_convective_regions": payload["detached_convective_regions"],
        "pressure_log10": np.log10(spec.grid().pressure_edges).tolist(),
        "flux_total": payload["flux_total"],
        "residual": payload["residual"],
        "f_int": spec.f_int,
        "source_profile_checksum_sha256": payload["source_profile_checksum_sha256"],
    }
    N384_RESIDUAL.write_text(json.dumps(compact, indent=2) + "\n")
    return payload


def _polish(res, spec, source: dict) -> dict:
    print("\n=== five-check pseudo-time polish ===", flush=True)
    solver = production_solver_config()
    rcb = _primary_rcb_log10p(spec.grid(), res.closure, solver)
    cfg = production_rce_config(
        max_steps=80,
        dt_accuracy=50000.0,
        dt_hold_init=float(source.get("last_accepted_dt") or 18415.0),
        previous_rcb_init=rcb,
        simulated_time_init=float(source.get("simulated_time") or 0.0),
        gate=PHYSICAL_GATE,
    )
    wall0 = time.perf_counter()
    polished = solve_adaptive_rce(
        spec.grid(),
        res.state.temperature,
        spec.physics(),
        solver,
        ConstantH2Thermo(),
        spec.opacity(),
        spec.grid().pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    wall = time.perf_counter() - wall0
    payload = serialize_rce_result(
        polished,
        spec,
        pressure_centres=spec.grid().pressure_centres,
        pressure_edges=spec.grid().pressure_edges,
        solver=solver,
        rce_config=cfg,
        extra={
            "wall_time_s": wall,
            "physical_gate": PHYSICAL_GATE,
            "initial_from": "steady_flux_defect_newton_krylov",
            "source_profile_checksum_sha256": source.get("profile_checksum_sha256"),
            "source_steps_accepted": source.get("steps_accepted"),
            "actual_integrator": "steady_newton_then_five_check_pseudotime",
            "coupled_helios_rce_status": "NOT_RUN",
        },
    )
    return payload


def _richardson_indicative(n96: dict, n192: dict, n384: dict) -> dict:
    def rel_t(coarse, fine):
        from run_nested_family import interpolate_temperature

        p_c = np.log(np.asarray(coarse["pressure_centres"], dtype=np.float64))
        p_f = np.log(np.asarray(fine["pressure_centres"], dtype=np.float64))
        t_c = np.asarray(coarse["temperature"], dtype=np.float64)
        t_f = interpolate_temperature(p_f, fine["temperature"], p_c)
        scale = np.maximum(np.abs(t_c), 1.0)
        return float(np.max(np.abs(t_f - t_c) / scale))

    e96 = rel_t(n96, n192)
    e192 = rel_t(n192, n384)
    rcb96 = n96.get("primary_rcb_log10p")
    rcb192 = n192.get("primary_rcb_log10p")
    rcb384 = n384.get("primary_rcb_log10p")
    drcb_c = None if rcb96 is None or rcb192 is None else abs(float(rcb192) - float(rcb96))
    drcb_f = None if rcb192 is None or rcb384 is None else abs(float(rcb384) - float(rcb192))
    p_t = None if e96 <= 0.0 or e192 <= 0.0 else float(np.log2(e96 / e192))
    p_rcb = None if not drcb_c or not drcb_f else float(np.log2(drcb_c / drcb_f))
    gated = (
        n384.get("status") == "converged"
        and float(n384.get("flux_flatness") or 1.0) <= PHYSICAL_GATE
        and float(n384.get("tendency_norm") or 1.0) <= PHYSICAL_GATE
    )
    return {
        "max_rel_T_192_vs_96": e96,
        "max_rel_T_384_vs_192": e192,
        "p_T": p_t,
        "abs_delta_rcb_192_vs_96": drcb_c,
        "abs_delta_rcb_384_vs_192": drcb_f,
        "p_RCB": p_rcb,
        "order_interval": [0.25, 3.0],
        "formal": gated,
        "note": (
            "Formal Richardson requires physically gated N=384. "
            "Coupled HELIOS RCE is not run here."
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-n384", action="store_true")
    p.add_argument("--invariance-only", action="store_true")
    p.add_argument("--skip-polish", action="store_true")
    p.add_argument("--skip-invariance", action="store_true")
    args = p.parse_args(argv)

    n384 = json.loads(N384.read_text())
    verify_record_checksums(n384)
    stored = n384.get("profile_checksum_sha256")
    if stored != LIVE_CHECKSUM:
        raise AssertionError(
            f"N=384 live checksum {stored} != expected 9500-step checksum {LIVE_CHECKSUM}"
        )
    print(
        f"N=384 live record checksum ok: {stored} "
        f"steps={n384.get('steps_accepted')} flatness={n384.get('flux_flatness')}",
        flush=True,
    )

    report: dict = {
        "physical_gate": PHYSICAL_GATE,
        "n384_source_checksum": stored,
        "n384_source_steps": n384.get("steps_accepted"),
        "coupled_helios_rce_status": "NOT_RUN",
    }

    if not args.skip_invariance:
        n192 = json.loads(N192.read_text())
        verify_record_checksums(n192)
        n96 = _load_n96()
        report["invariance_n96"] = _invariance(96, n96)
        report["invariance_n192"] = _invariance(192, n192)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "n96_unchanged": report["invariance_n96"]["essentially_unchanged"],
                    "n96_dT": report["invariance_n96"]["at_physical_gate"]["max_rel_T"],
                    "n192_unchanged": report["invariance_n192"]["essentially_unchanged"],
                    "n192_dT": report["invariance_n192"]["at_physical_gate"]["max_rel_T"],
                },
                indent=2,
            ),
            flush=True,
        )

    if args.invariance_only or args.skip_n384:
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        return report

    print("\n=== N=384 steady flux-defect solve ===", flush=True)
    cfg384 = SteadyRCEConfig(
        flux_flatness_tolerance=PHYSICAL_GATE,
        tendency_tolerance=PHYSICAL_GATE,
        max_newton=8,
        max_mask_outer=2,
        gmres_maxiter=48,
        gmres_restart=32,
        fd_rel=1.0e-6,
        subspace_fd_rel=1.0e-8,
        max_step_rel=0.05,
        use_subspace=True,
        use_gmres=True,
        reject_branch_crossing=True,
    )
    res384, spec384, solver384, summary384 = _solve_record(384, n384, cfg384)
    report["n384_steady"] = summary384
    _write_n384_snapshot(res384, spec384, solver384, n384, summary384, cfg384)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": summary384["status"],
                "flatness": summary384["flux_flatness"],
                "tendency": summary384["tendency_norm"],
                "newton": summary384["newton_iterations"],
                "evals": summary384["n_evals"],
                "dT": summary384["max_rel_T"],
                "regions": summary384["convective_regions"],
                "detached": summary384["detached_convective_regions"],
                "identities": summary384["identities"],
            },
            indent=2,
        ),
        flush=True,
    )

    polish = None
    if not args.skip_polish and summary384["status"] == "converged":
        polish = _polish(res384, spec384, n384)
        report["n384_polish"] = {
            "status": polish.get("status"),
            "reason": polish.get("reason"),
            "steps_accepted": polish.get("steps_accepted"),
            "rejections": polish.get("rejections"),
            "flux_flatness": polish.get("flux_flatness"),
            "tendency_norm": polish.get("tendency_norm"),
            "temp_change": polish.get("temp_change"),
            "rcb_stable": polish.get("rcb_stable"),
            "primary_rcb_log10p": polish.get("primary_rcb_log10p"),
            "convective_regions": polish.get("convective_regions"),
            "detached_convective_regions": polish.get("detached_convective_regions"),
            "energy_gate_ratio": polish.get("energy_gate_ratio"),
            "coupled_defect": polish.get("coupled_defect"),
            "identities": algebraic_identities(polish),
            "profile_checksum_sha256": polish.get("profile_checksum_sha256"),
        }
        gated = (
            polish.get("status") == "converged"
            and float(polish.get("flux_flatness") or 1.0) <= PHYSICAL_GATE
            and float(polish.get("tendency_norm") or 1.0) <= PHYSICAL_GATE
            and not (polish.get("detached_convective_regions") or [])
        )
        report["n384_physically_gated"] = gated
        if gated:
            if not N384_ARCHIVE.exists():
                shutil.copy2(N384, N384_ARCHIVE)
            N384.write_text(dumps(polish))
            print(f"wrote gated N=384 live record to {N384}", flush=True)
            from n384_spatial_probe import main as rebuild_probe
            from build_current_exit_audit import main as rebuild_audit

            probe = rebuild_probe()
            report["spatial_probe_latest"] = probe.get("latest_comparison")
            audit = rebuild_audit()
            report["audit"] = {
                "core_single_resolution_status": audit.get("core_single_resolution_status"),
                "spatial_and_operator_convergence_status": audit.get(
                    "spatial_and_operator_convergence_status"
                ),
                "full_stage4_claim": audit.get("full_stage4_claim"),
                "helios_parity_status": audit.get("helios_parity_status"),
            }
        n96 = _load_n96() if "invariance_n96" not in report else None
        if n96 is None:
            n96 = _load_n96()
        n192 = json.loads(N192.read_text())
        report["richardson"] = _richardson_indicative(n96, n192, polish)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
    elif summary384["status"] != "converged":
        report["n384_physically_gated"] = False
        report["n384_polish"] = None
        OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in report if k != "n384_steady"}, indent=2)[:4000], flush=True)
    return report


if __name__ == "__main__":
    main()
