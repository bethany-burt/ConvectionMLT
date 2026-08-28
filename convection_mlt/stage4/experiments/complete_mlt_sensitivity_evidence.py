#!/usr/bin/env python3
"""Complete MLT sensitivity evidence: prescribed-Δt, broad ICs, N=192 polish.

Corrects the earlier dt_hold_init ladder (adaptive growth erased the held Δt).
Does not reopen HELIOS debugging or run N=384 α sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    RCERoute,
    TopIrradiation,
    grey_radiative_equilibrium_temperature,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from rce_record import (
    PHYSICAL_GATE,
    dumps,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)

OUT = ROOT / "results" / "mlt_sensitivity"
SUMMARY = OUT / "mlt_sensitivity_completion.json"
GATE = PHYSICAL_GATE
DT_LADDER = (10000.0, 5000.0, 2500.0)
T_FINAL = 1.0e5
IC_T_TOL = 1.0e-3
IC_RCB_TOL = 0.05


def _spec(n: int, alpha: float = 1.0):
    return nested_analytic_opacity_spec(n, alpha=float(alpha))


def _gated(rec: dict) -> bool:
    return (
        rec.get("status") == "converged"
        and float(rec.get("flux_flatness") or 1.0) <= GATE
        and float(rec.get("tendency_norm") or 1.0) <= GATE
        and not (rec.get("detached_convective_regions") or [])
    )


def _topo_ok(rec: dict) -> bool:
    regs = rec.get("convective_regions") or []
    det = rec.get("detached_convective_regions") or []
    return len(regs) == 1 and regs[0][0] == 0 and not det


def _run(
    *,
    n_layers: int,
    alpha: float,
    t0,
    label: str,
    max_steps: int = 20000,
    dt_accuracy: float = 2500.0,
    dt_hold: float | None = None,
    prescribed_dt: float | None = None,
    t_final: float | None = None,
    n_consec: int | None = None,
    gate: float = GATE,
) -> dict:
    spec = _spec(n_layers, alpha)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    cfg = production_rce_config(
        max_steps=max_steps,
        dt_accuracy=dt_accuracy,
        dt_hold_init=dt_hold,
        gate=gate,
    )
    kwargs = {}
    if prescribed_dt is not None:
        kwargs["prescribed_dt"] = float(prescribed_dt)
    if t_final is not None:
        kwargs["t_final"] = float(t_final)
    if n_consec is not None:
        kwargs["n_consec"] = int(n_consec)
        kwargs["stall_window"] = 10**9
    if kwargs:
        cfg = replace(cfg, **kwargs)
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
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
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    wall = time.perf_counter() - wall0
    payload = serialize_rce_result(
        res,
        spec,
        pressure_centres=grid.pressure_centres,
        pressure_edges=grid.pressure_edges,
        solver=solver,
        rce_config=cfg,
        extra={
            "wall_time_s": wall,
            "campaign_label": label,
            "alpha": float(alpha),
            "n_layers": int(n_layers),
            "prescribed_dt": prescribed_dt,
            "t_final_target": t_final,
        },
    )
    payload["physically_gated"] = _gated(payload)
    # Timestep history diagnostics
    raw = (payload.get("history") or {}).get("dt") or []
    dts = [float(x) for x in raw if x is not None]
    payload["accepted_dt_unique"] = sorted({round(x, 12) for x in dts})
    payload["accepted_dt_constant"] = (
        len(payload["accepted_dt_unique"]) <= 1 and bool(dts)
    )
    print(
        f"[{label}] status={payload.get('status')} flat={payload.get('flux_flatness')} "
        f"rcb={payload.get('primary_rcb_log10p')} t={payload.get('simulated_time')} "
        f"steps={payload.get('steps_accepted')} wall={wall:.1f}s "
        f"dt_const={payload['accepted_dt_constant']} dts={payload['accepted_dt_unique'][:5]}",
        flush=True,
    )
    return payload


def phase_prescribed_dt() -> dict:
    """Genuine fixed-Δt equal-time ladder at N=96, α=1."""
    ref_path = OUT / "alpha_sweep_n96_alpha1.0.json"
    ref = json.loads(ref_path.read_text())
    t_eq = np.asarray(ref["temperature"], dtype=np.float64)
    t_pert = t_eq * (1.0 + 2.0e-2)
    arms = {}
    profiles = {}
    for dt in DT_LADDER:
        n_steps = int(np.ceil(T_FINAL / dt)) + 5
        rec = _run(
            n_layers=96,
            alpha=1.0,
            t0=t_pert,
            label=f"prescribed_dt_{dt}",
            max_steps=n_steps,
            dt_accuracy=1.0e12,
            prescribed_dt=float(dt),
            t_final=T_FINAL,
            n_consec=10**9,
            gate=1.0e-30,
        )
        path = OUT / f"prescribed_dt_n96_alpha1_dt{int(dt)}.json"
        path.write_text(dumps(rec))
        key = str(dt)
        profiles[key] = np.asarray(rec["temperature"], dtype=np.float64)
        arms[key] = {
            "status": rec.get("status"),
            "reason": rec.get("reason"),
            "simulated_time": rec.get("simulated_time"),
            "steps_accepted": rec.get("steps_accepted"),
            "flux_flatness": rec.get("flux_flatness"),
            "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
            "accepted_dt_constant": rec.get("accepted_dt_constant"),
            "accepted_dt_unique": rec.get("accepted_dt_unique"),
            "energy_gate_ratio": rec.get("energy_gate_ratio"),
            "convective_regions": rec.get("convective_regions"),
            "record": str(path),
        }

    dts = sorted(float(k) for k in arms)
    finest = str(dts[0])  # 2500
    t_ref = profiles[finest]
    errors = {}
    for dt in dts[1:]:
        errors[str(dt)] = float(
            np.max(np.abs(profiles[str(dt)] - t_ref) / np.maximum(t_ref, 1.0))
        )
    # temporal order between successive halvings: e(10000)/e(5000)
    order = None
    if "10000.0" in errors and "5000.0" in errors and errors["5000.0"] > 0:
        order = float(np.log(errors["10000.0"] / errors["5000.0"]) / np.log(2.0))

    # Equilibrium independence: tiny kick, prescribed_dt fixed, gate to 1e-3
    eq = {}
    t_tiny = t_eq * (1.0 + 1.0e-4)
    for dt in (dts[0], dts[-1]):
        rec = _run(
            n_layers=96,
            alpha=1.0,
            t0=t_tiny,
            label=f"prescribed_eq_dt{dt}",
            max_steps=20000,
            dt_accuracy=1.0e12,
            prescribed_dt=float(dt),
            gate=GATE,
        )
        eq[str(dt)] = {
            "physically_gated": _gated(rec),
            "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
            "flux_flatness": rec.get("flux_flatness"),
            "temperature": rec["temperature"],
            "topo_ok": _topo_ok(rec),
        }
    keys = list(eq.keys())
    ta = np.asarray(eq[keys[0]]["temperature"], dtype=np.float64)
    tb = np.asarray(eq[keys[1]]["temperature"], dtype=np.float64)
    eq_rel = float(np.max(np.abs(ta - tb) / np.maximum(ta, 1.0)))
    for v in eq.values():
        v.pop("temperature", None)

    times_ok = all(
        abs(float(arms[str(dt)]["simulated_time"]) - T_FINAL) / T_FINAL < 0.02
        for dt in dts
    )
    const_ok = all(arms[str(dt)]["accepted_dt_constant"] for dt in dts)
    mono = True
    if len(dts) >= 3:
        e_vals = [errors[str(dt)] for dt in dts[1:]]
        # coarser should have larger error vs finest
        mono = e_vals[-1] >= e_vals[0]  # 10000 err >= 5000 err
    no_topo_jump = all(_topo_ok({"convective_regions": arms[str(dt)]["convective_regions"], "detached_convective_regions": []}) or True for dt in dts)
    # use reference topology from finest trajectory end / eq
    pass_ = (
        times_ok
        and const_ok
        and mono
        and order is not None
        and 0.6 <= order <= 1.6
        and eq_rel < GATE
        and all(eq[k]["physically_gated"] for k in eq)
        and all(eq[k]["topo_ok"] for k in eq)
    )
    return {
        "status": "PASS" if pass_ else "FAIL",
        "method": "prescribed_dt_fixed_ladder",
        "dt_values_s": list(dts),
        "t_final": T_FINAL,
        "arms": arms,
        "equal_time_max_rel_T_vs_finest": errors,
        "observed_temporal_order": order,
        "errors_monotone_with_coarser_dt": mono,
        "constant_timestep_histories": const_ok,
        "identical_final_physical_time": times_ok,
        "equilibrium_independence": eq,
        "equilibrium_max_rel_T": eq_rel,
        "prior_dt_hold_flaw_note": (
            "Earlier campaign used dt_hold_init only; adaptive growth erased the "
            "held Δt. This phase uses RCEConfig.prescribed_dt."
        ),
    }


def phase_broad_ic() -> dict:
    """Broader basin tests at N=96, α=1 with production procedure (+ polish).

    Cases match the meeting brief: production RC seed; RE seed; ±5% hot/cold.
    Far seeds use discrete-RZ + five-check (the user-facing complete procedure)
    rather than tens of thousands of adaptive pseudo-time steps alone.
    """
    spec = _spec(96, 1.0)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opac = spec.opacity()

    path_rc = OUT / "broad_ic_n96_alpha1_radiative_convective.json"
    if path_rc.exists():
        ref = json.loads(path_rc.read_text())
        if not _gated(ref):
            ref = None
    else:
        ref = None
    if ref is None:
        ref = _run(
            n_layers=96,
            alpha=1.0,
            t0=radiative_convective_initial_temperature(
                grid, opac, thermo, spec.f_int, spec.f_irr
            ),
            label="broad_ic_radiative_convective",
            max_steps=20000,
            dt_accuracy=2500.0,
        )
        if not _gated(ref):
            ref = _polish_via_reduced(
                96,
                1.0,
                np.asarray(ref["temperature"], dtype=np.float64),
                label="broad_ic_radiative_convective_polish",
            )
        path_rc.write_text(dumps(ref))

    t_ref = np.asarray(ref["temperature"], dtype=np.float64)
    rcb_ref = ref.get("primary_rcb_log10p")
    seeds = {
        "radiative_equilibrium": grey_radiative_equilibrium_temperature(
            grid, opac, spec.f_int, spec.f_irr
        ),
        "hot_5pct_from_rc": t_ref * 1.05,
        "cold_5pct_from_rc": t_ref * 0.95,
    }

    cases = {
        "radiative_convective": {
            "physically_gated": _gated(ref),
            "topo_ok": _topo_ok(ref),
            "max_rel_T": 0.0,
            "rcb_dex": 0.0,
            "energy_gate_ratio": ref.get("energy_gate_ratio"),
            "primary_rcb_log10p": rcb_ref,
            "flux_flatness": ref.get("flux_flatness"),
            "record": str(path_rc),
            "pass": _gated(ref) and _topo_ok(ref),
            "procedure": "adaptive_pseudotime",
        }
    }

    for name, t0 in seeds.items():
        # Complete production path: reduced accelerator then live five-check.
        rec = _polish_via_reduced(96, 1.0, np.asarray(t0, dtype=np.float64), label=f"broad_ic_{name}")
        if not _gated(rec):
            # Five-check continuation: keep accuracy ceiling high (do not cap at 2500).
            rec = _run(
                n_layers=96,
                alpha=1.0,
                t0=np.asarray(rec["temperature"], dtype=np.float64),
                label=f"broad_ic_{name}_continue",
                max_steps=500,
                dt_accuracy=50000.0,
                dt_hold=float(rec.get("last_accepted_dt") or 18415.0),
            )
        if not _gated(rec):
            # Second discrete-RZ pass from the near-gate state.
            rec = _polish_via_reduced(
                96,
                1.0,
                np.asarray(rec["temperature"], dtype=np.float64),
                label=f"broad_ic_{name}_repolish",
            )
        outp = OUT / f"broad_ic_n96_alpha1_{name}.json"
        outp.write_text(dumps(rec))
        t_f = np.asarray(rec["temperature"], dtype=np.float64)
        max_rel = float(np.max(np.abs(t_f - t_ref) / np.maximum(t_ref, 1.0)))
        rcb = rec.get("primary_rcb_log10p")
        rcb_dex = None if rcb is None or rcb_ref is None else abs(float(rcb) - float(rcb_ref))
        ok = (
            _gated(rec)
            and _topo_ok(rec)
            and max_rel <= IC_T_TOL
            and (rcb_dex is None or rcb_dex <= IC_RCB_TOL)
            and bool(np.all(np.isfinite(t_f)))
            and bool(np.all(t_f > 0.0))
        )
        cases[name] = {
            "physically_gated": _gated(rec),
            "topo_ok": _topo_ok(rec),
            "max_rel_T": max_rel,
            "rcb_dex": rcb_dex,
            "energy_gate_ratio": rec.get("energy_gate_ratio"),
            "primary_rcb_log10p": rcb,
            "flux_flatness": rec.get("flux_flatness"),
            "record": str(outp),
            "pass": ok,
            "procedure": "discrete_rz_plus_five_check",
        }

    return {
        "status": "PASS" if all(c["pass"] for c in cases.values()) else "FAIL",
        "reference": "radiative_convective_gated",
        "tolerances": {"max_rel_T": IC_T_TOL, "rcb_dex": IC_RCB_TOL},
        "cases": cases,
        "note": (
            "RC seed uses adaptive pseudo-time; RE and ±5% use the complete "
            "discrete-RZ + five-check procedure (allowed for user-facing solver)."
        ),
    }


def _polish_via_reduced(n_layers: int, alpha: float, t0: np.ndarray, *, label: str) -> dict:
    """Shared production procedure (discrete-RZ + five-check + recovery)."""
    from convection_mlt.production_rce import ProductionControls, run_production_rce

    wall0 = time.perf_counter()
    run = run_production_rce(
        n_layers=int(n_layers),
        alpha=float(alpha),
        temperature_initial=np.asarray(t0, dtype=np.float64),
        procedure="production",
        controls=ProductionControls(gate=GATE),
        log=lambda m: print(f"[{label}] {m}", flush=True),
    )
    wall = time.perf_counter() - wall0
    payload = serialize_rce_result(
        run.result,
        run.spec,
        pressure_centres=run.pressure_centres,
        pressure_edges=run.pressure_edges,
        solver=run.solver,
        rce_config=run.rce_config_last,
        extra={
            "wall_time_s": wall,
            "campaign_label": label,
            "actual_integrator": "discrete_rz_t_rcb_finite_mlt_then_five_check_pseudotime",
            "phases": run.phases,
            "alpha": float(alpha),
            "n_layers": int(n_layers),
        },
    )
    payload["physically_gated"] = _gated(payload)
    print(
        f"[{label}] polish gated={payload['physically_gated']} "
        f"flat={payload.get('flux_flatness')} rcb={payload.get('primary_rcb_log10p')}",
        flush=True,
    )
    return payload


def phase_polish_n192() -> dict:
    members = {}
    for alpha in (0.5, 1.0, 2.0):
        src = OUT / f"alpha_sweep_n192_alpha{alpha}.json"
        rec0 = json.loads(src.read_text())
        t0 = np.asarray(rec0["temperature"], dtype=np.float64)
        # Prefer five-check continuation first (already near gate)
        polish = _run(
            n_layers=192,
            alpha=alpha,
            t0=t0,
            label=f"n192_fivecheck_a{alpha}",
            max_steps=200,
            dt_accuracy=50000.0,
            dt_hold=float(rec0.get("last_accepted_dt") or 18415.0),
            gate=GATE,
        )
        if not _gated(polish):
            polish = _polish_via_reduced(
                192, alpha, t0, label=f"n192_reduced_polish_a{alpha}"
            )
        path = OUT / f"n192_polished_alpha{alpha}.json"
        path.write_text(dumps(polish))
        members[str(alpha)] = {
            "physically_gated": _gated(polish),
            "status": polish.get("status"),
            "flux_flatness": polish.get("flux_flatness"),
            "tendency_norm": polish.get("tendency_norm"),
            "primary_rcb_log10p": polish.get("primary_rcb_log10p"),
            "convective_regions": polish.get("convective_regions"),
            "detached_convective_regions": polish.get("detached_convective_regions"),
            "topo_ok": _topo_ok(polish),
            "pre_polish_flatness": rec0.get("flux_flatness"),
            "record": str(path),
            "integrator": polish.get("actual_integrator"),
        }
    # Δ∇ scaling on polished
    xs, ys = [], []
    for a, m in members.items():
        if not m["physically_gated"]:
            continue
        rec = json.loads(Path(m["record"]).read_text())
        # compute delta from closure quickly via campaign helper if present
        from run_mlt_sensitivity_campaign import _closure_diagnostics

        diag = _closure_diagnostics(rec, _spec(192, float(a)))
        d = diag.get("delta_nabla_rms_active")
        m["delta_nabla_rms_active"] = d
        if d and float(d) > 0:
            xs.append(float(a))
            ys.append(float(d))
    slope = None
    if len(xs) >= 2:
        slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    all_ok = all(m["physically_gated"] and m["topo_ok"] for m in members.values())
    return {
        "status": "PASS" if all_ok else "FAIL",
        "members": members,
        "alpha_delta_scaling_slope": slope,
        "expected_slope": -4.0 / 3.0,
        "note": "Formal gate via five-check and/or discrete-RZ + five-check polish.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        default="all",
        choices=["prescribed_dt", "broad_ic", "polish_n192", "all"],
    )
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    if SUMMARY.exists():
        report = json.loads(SUMMARY.read_text())

    order = (
        ["prescribed_dt", "broad_ic", "polish_n192"]
        if args.phase == "all"
        else [args.phase]
    )
    for name in order:
        print(f"\n=== {name} ===", flush=True)
        if name == "prescribed_dt":
            report["prescribed_dt"] = phase_prescribed_dt()
        elif name == "broad_ic":
            report["broad_ic"] = phase_broad_ic()
        elif name == "polish_n192":
            report["polish_n192"] = phase_polish_n192()
        SUMMARY.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: (v or {}).get("status") for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
