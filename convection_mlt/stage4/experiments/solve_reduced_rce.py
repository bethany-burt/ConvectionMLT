"""Reduced radiative-matching accelerator, then live-MLT five-check polish.

Does not overwrite n384_implicit_rce.json unless polish reaches the physical
gate. Does not launch coupled HELIOS. Does not enforce an adiabatic CZ.
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
    ReducedRCEConfig,
    TopIrradiation,
    nested_analytic_opacity_spec,
    reduced_config_as_dict,
    solve_adaptive_rce,
    solve_reduced_radiative_matching,
)
from convection_mlt.rce import _primary_rcb_log10p

from rce_record import (
    PHYSICAL_GATE,
    algebraic_identities,
    dumps,
    finalize_record,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
    verify_record_checksums,
)
from solve_steady_rce import LIVE_CHECKSUM, N384, RESULTS

OUT = RESULTS / "reduced_rce_solve.json"
LAGGED_SNAP = RESULTS / "n384_reduced_rce.json"
COUPLED_SNAP = RESULTS / "n384_coupled_rce.json"
DISCRETE_SNAP = RESULTS / "n384_discrete_rz_rce.json"
PROFILES = RESULTS / "n384_discrete_rz_flux_profiles.json"
NEGATIVE = RESULTS / "subspace_newton_negative_result.json"


def _progress(event: dict) -> None:
    kind = event.get("event")
    if kind in {"secant", "rcb_outer"}:
        print(
            f"  {kind} it={event['picard']} accepted={event['accepted']} "
            f"reason={event['reason']} T_RCB={event['t_rcb']:.6g} "
            f"F_top={event['f_top']:.6g} defect={event['f_top_defect']:.4e} "
            f"flatness={event['flatness']:.4e} tendency={event['tendency']:.4e} "
            f"worst={event['worst_gate']:.3f} rcb={event['rcb_layer']} "
            f"inner={event['inner_picard']} conv={event.get('inner_converged')} "
            f"dΔ∇={event.get('delta_abs')} cz_mlt={event.get('cz_mlt_mismatch')} "
            f"rz_div={event.get('rz_div')} min_excess={event['min_excess']:.3e}",
            flush=True,
        )
    elif kind == "reduced_picard":
        print(
            f"  reduced picard={event['picard']} accepted={event['accepted']} "
            f"reason={event['reason']} flatness={event['flatness']:.4e} "
            f"tendency={event['tendency']:.4e} worst={event['worst_gate']:.3f} "
            f"damp={event['damping']:.3g} dlnT={event['logt_shift']:.3g} "
            f"rcb={event['rcb_layer']} min_excess={event['min_excess']:.3e}",
            flush=True,
        )
    else:
        print(f"  {event}", flush=True)


def _history(res) -> list[dict]:
    return [asdict(rec) for rec in res.history]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-polish", action="store_true")
    args = parser.parse_args(argv)

    n384 = json.loads(N384.read_text())
    verify_record_checksums(n384)
    stored = n384.get("profile_checksum_sha256")
    if stored != LIVE_CHECKSUM:
        raise AssertionError(
            f"N=384 live checksum {stored} != expected gated checksum {LIVE_CHECKSUM}"
        )
    print(
        f"N=384 live record checksum ok: {stored} "
        f"steps={n384.get('steps_accepted')} flatness={n384.get('flux_flatness')}",
        flush=True,
    )
    print(f"subspace negative result: {NEGATIVE}", flush=True)

    spec = nested_analytic_opacity_spec(384)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    t0 = np.asarray(n384["temperature"], dtype=np.float64)
    cfg = ReducedRCEConfig(
        coupling="consistent",
        rz_mode="discrete",
        flux_flatness_tolerance=PHYSICAL_GATE,
        tendency_tolerance=PHYSICAL_GATE,
        f_top_tolerance=PHYSICAL_GATE,
        max_secant=16,
        max_inner_picard=8,
        inner_damping=0.5,
        t_rcb_rel_bracket=0.03,
        max_rcb_outer=4,
        match_rz_to_grey_re=False,
        rz_blend=0.0,
        max_rz_kappa_picard=4,
    )
    print("\n=== N=384 discrete-RZ + T_RCB matching solve ===", flush=True)
    wall0 = time.perf_counter()
    reduced = solve_reduced_radiative_matching(
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
        progress=_progress,
    )
    wall = time.perf_counter() - wall0
    scale = np.maximum(np.abs(t0), 1.0)
    dT = float(np.max(np.abs(reduced.temperature - t0) / scale))
    f_int = float(spec.f_int)
    f_top_before = float(n384["flux_total"][-1])
    snapshot = {
        "solver": "discrete_rz_t_rcb_finite_mlt",
        "exactly_adiabatic_cz": False,
        "lagged_picard_archive": str(LAGGED_SNAP),
        "t_rcb_live_scale_archive": str(COUPLED_SNAP),
        "reduced_config": reduced_config_as_dict(cfg),
        "status": reduced.status.value,
        "reason": reduced.reason,
        "flux_flatness": reduced.flux_flatness,
        "tendency_norm": reduced.tendency_norm,
        "worst_gate": reduced.worst_gate,
        "improved": reduced.improved,
        "n_picard": reduced.n_picard,
        "n_evals": reduced.n_evals,
        "n_inner_picard": reduced.n_inner_picard,
        "n_rcb_outer": reduced.n_rcb_outer,
        "inner_converged": reduced.inner_converged,
        "inner_delta_abs_mismatch": reduced.inner_delta_abs_mismatch,
        "inner_delta_rel_mismatch": reduced.inner_delta_rel_mismatch,
        "max_cz_mlt_flux_mismatch": reduced.max_cz_mlt_flux_mismatch,
        "rz_max_flux_divergence": reduced.rz_max_flux_divergence,
        "t_rcb": reduced.t_rcb,
        "f_top": reduced.f_top,
        "f_top_defect": reduced.f_top_defect,
        "f_top_before": f_top_before,
        "f_top_defect_before": f_top_before - f_int,
        "max_rel_T": dT,
        "convective_regions": reduced.convective_regions,
        "rcb_layer": reduced.rcb_layer,
        "min_superadiabatic_excess_active": reduced.min_superadiabatic_excess_active,
        "source_profile_checksum_sha256": stored,
        "temperature": reduced.temperature.tolist(),
        "history": _history(reduced),
        "physical_gate": PHYSICAL_GATE,
        "wall_time_s": wall,
        "coupled_helios_rce_status": "NOT_RUN",
    }
    if reduced.trial is not None:
        snapshot["flux_total"] = reduced.trial.flux_total.tolist()
        snapshot["flux_rad"] = reduced.trial.flux_rad.tolist()
        snapshot["flux_conv"] = reduced.trial.flux_conv.tolist()
        snapshot["residual"] = reduced.trial.residual.tolist()
        snapshot["mass_path"] = reduced.trial.state.mass_path.tolist()
        snapshot["f_int"] = f_int
        snapshot.update(algebraic_identities(snapshot))
    DISCRETE_SNAP.write_text(dumps(finalize_record(snapshot)))
    profiles = {
        "f_int": f_int,
        "pressure_centres": spec.grid().pressure_centres.tolist(),
        "pressure_edges": spec.grid().pressure_edges.tolist(),
        "before": {
            "temperature": n384["temperature"],
            "flux_rad": n384["flux_rad"],
            "flux_conv": n384["flux_conv"],
            "flux_total": n384["flux_total"],
            "f_top": f_top_before,
            "flux_flatness": n384.get("flux_flatness"),
            "tendency_norm": n384.get("tendency_norm"),
        },
        "after": {
            "temperature": reduced.temperature.tolist(),
            "flux_rad": snapshot.get("flux_rad"),
            "flux_conv": snapshot.get("flux_conv"),
            "flux_total": snapshot.get("flux_total"),
            "f_top": reduced.f_top,
            "flux_flatness": reduced.flux_flatness,
            "tendency_norm": reduced.tendency_norm,
            "t_rcb": reduced.t_rcb,
            "rcb_layer": reduced.rcb_layer,
        },
        "history": snapshot["history"],
        "status": reduced.status.value,
        "reason": reduced.reason,
        "source_profile_checksum_sha256": stored,
    }
    PROFILES.write_text(json.dumps(profiles, indent=2) + "\n")

    report = {
        "physical_gate": PHYSICAL_GATE,
        "n384_source_checksum": stored,
        "subspace_newton": "NEGATIVE",
        "subspace_newton_record": str(NEGATIVE),
        "lagged_picard_archive": str(LAGGED_SNAP),
        "t_rcb_live_scale_archive": str(COUPLED_SNAP),
        "coupled_snapshot": str(DISCRETE_SNAP),
        "flux_profiles": str(PROFILES),
        "reduced_config": reduced_config_as_dict(cfg),
        "reduced": {
            "solver": "discrete_rz_t_rcb_finite_mlt",
            "status": reduced.status.value,
            "reason": reduced.reason,
            "flux_flatness": reduced.flux_flatness,
            "tendency_norm": reduced.tendency_norm,
            "worst_gate": reduced.worst_gate,
            "improved": reduced.improved,
            "n_picard": reduced.n_picard,
            "n_evals": reduced.n_evals,
            "n_inner_picard": reduced.n_inner_picard,
            "n_rcb_outer": reduced.n_rcb_outer,
            "inner_converged": reduced.inner_converged,
            "inner_delta_abs_mismatch": reduced.inner_delta_abs_mismatch,
            "inner_delta_rel_mismatch": reduced.inner_delta_rel_mismatch,
            "max_cz_mlt_flux_mismatch": reduced.max_cz_mlt_flux_mismatch,
            "rz_max_flux_divergence": reduced.rz_max_flux_divergence,
            "t_rcb": reduced.t_rcb,
            "f_top": reduced.f_top,
            "f_top_defect": reduced.f_top_defect,
            "f_top_before": f_top_before,
            "f_top_defect_before": f_top_before - f_int,
            "max_rel_T": dT,
            "convective_regions": reduced.convective_regions,
            "min_superadiabatic_excess_active": reduced.min_superadiabatic_excess_active,
            "exactly_adiabatic_cz": False,
            "wall_time_s": wall,
        },
        "coupled_helios_rce_status": "NOT_RUN",
        "n384_live_overwritten": False,
        "spatial_headline": "WITHHELD",
    }
    print(json.dumps(report["reduced"], indent=2), flush=True)

    polish_from = "reduced" if reduced.improved else "live_checkpoint"
    t_polish = reduced.temperature if reduced.improved else t0
    report["polish_initial"] = polish_from
    if args.skip_polish:
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        return report

    print(f"\n=== five-check live-MLT polish from {polish_from} ===", flush=True)
    rcb = _primary_rcb_log10p(spec.grid(), reduced.trial.closure, solver) if reduced.trial is not None else n384.get("primary_rcb_log10p")
    polish_cfg = production_rce_config(
        max_steps=80,
        dt_accuracy=50000.0,
        dt_hold_init=float(n384.get("last_accepted_dt") or 18415.0),
        previous_rcb_init=rcb,
        simulated_time_init=float(n384.get("simulated_time") or 0.0),
        gate=PHYSICAL_GATE,
    )
    wall0 = time.perf_counter()
    polished = solve_adaptive_rce(
        spec.grid(),
        t_polish,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        spec.grid().pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=polish_cfg,
    )
    wall_p = time.perf_counter() - wall0
    payload = serialize_rce_result(
        polished,
        spec,
        pressure_centres=spec.grid().pressure_centres,
        pressure_edges=spec.grid().pressure_edges,
        solver=solver,
        rce_config=polish_cfg,
        extra={
            "wall_time_s": wall_p,
            "physical_gate": PHYSICAL_GATE,
            "initial_from": polish_from,
            "reduced_status": reduced.status.value,
            "source_profile_checksum_sha256": stored,
            "source_steps_accepted": n384.get("steps_accepted"),
            "actual_integrator": "discrete_rz_t_rcb_finite_mlt_then_five_check_pseudotime",
            "coupled_helios_rce_status": "NOT_RUN",
            "exactly_adiabatic_cz": False,
        },
    )
    gated = (
        payload.get("status") == "converged"
        and float(payload.get("flux_flatness") or 1.0) <= PHYSICAL_GATE
        and float(payload.get("tendency_norm") or 1.0) <= PHYSICAL_GATE
        and not (payload.get("detached_convective_regions") or [])
    )
    report["n384_polish"] = {
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "steps_accepted": payload.get("steps_accepted"),
        "flux_flatness": payload.get("flux_flatness"),
        "tendency_norm": payload.get("tendency_norm"),
        "primary_rcb_log10p": payload.get("primary_rcb_log10p"),
        "convective_regions": payload.get("convective_regions"),
        "detached_convective_regions": payload.get("detached_convective_regions"),
        "profile_checksum_sha256": payload.get("profile_checksum_sha256"),
        "wall_time_s": wall_p,
        "gated": gated,
    }
    report["n384_physically_gated"] = gated
    if gated:
        archive = RESULTS / "n384_implicit_rce_9500.json"
        if not archive.exists():
            shutil.copy2(N384, archive)
        N384.write_text(dumps(payload))
        report["n384_live_overwritten"] = True
        print(f"wrote gated N=384 live record to {N384}", flush=True)
    else:
        print("polish did not gate N=384; live record unchanged", flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "n384_polish"}, indent=2)[:3000], flush=True)
    return report


if __name__ == "__main__":
    main()
