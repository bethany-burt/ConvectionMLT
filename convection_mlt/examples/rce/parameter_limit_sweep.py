#!/usr/bin/env python3
"""Map production-RCE convergence limits in (x_He, F_irr) and related knobs.

Expands beyond the realistic sweep to find failure boundaries. Each case uses
the default cfg_demo production budget unless noted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from convection_mlt.production_rce import (
    ProductionControls,
    _gates_from_result,
    run_production_rce,
)

OUT = Path(__file__).resolve().parent / "runs" / "parameter_limit_sweep.json"
SUMMARY = Path(__file__).resolve().parent / "runs" / "parameter_limit_summary.json"

CONTROLS = ProductionControls(
    max_steps_live_polish=200,
    max_steps_continuation=500,
    max_recovery_cycles=2,
    dt_accuracy_s=50000.0,
    dt_hold_init_s=18415.0,
    continuation_dt_accuracy_s=2500.0,
)

# --- sweep design ---
X_HE_THERMAL = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
F_IRR_AT_XHE = (0.4, 0.5, 0.6, 0.7)  # irradiation ladder at fixed x_He values
F_IRR_LADDER = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6000.0)
X_HE_AT_FIRR = (0.4, 0.5, 0.6, 0.7, 0.8)  # He ladder at high F_irr

# Extra knobs on hardest-looking thermal corners
KNOB_CASES = (
    {"x_he": 0.9, "f_irr": 0.0, "alpha": 0.5},
    {"x_he": 0.9, "f_irr": 0.0, "alpha": 2.0},
    {"x_he": 0.9, "f_irr": 0.0, "alpha": 4.0},
    {"x_he": 1.0, "f_irr": 0.0, "alpha": 0.5},
    {"x_he": 0.7, "f_irr": 4000.0, "alpha": 0.5},
    {"x_he": 0.7, "f_irr": 4000.0, "alpha": 2.0},
)


def _case_id(**kw) -> str:
    parts = []
    for k in sorted(kw):
        v = kw[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def run_case(
    *,
    x_he: float,
    f_irr: float,
    alpha: float = 1.0,
    f_int: float = 300.0,
    n_layers: int = 96,
    seed: str = "radiative_convective",
    wall_limit_s: float = 900.0,
) -> dict:
    meta = {
        "x_he": float(x_he),
        "f_irr": float(f_irr),
        "alpha": float(alpha),
        "f_int": float(f_int),
        "n_layers": int(n_layers),
        "seed": seed,
    }
    cid = _case_id(**meta)
    t0 = time.perf_counter()
    err = None
    run = None
    try:
        run = run_production_rce(
            n_layers=n_layers,
            alpha=alpha,
            f_int=f_int,
            f_irr=f_irr,
            x_he=x_he,
            seed=seed,
            procedure="production",
            controls=CONTROLS,
        )
    except Exception as exc:  # noqa: BLE001 — boundary survey
        err = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    if wall > wall_limit_s:
        return {
            **meta,
            "id": cid,
            "verdict": "TIMEOUT",
            "wall_s": wall,
            "error": f"exceeded {wall_limit_s}s wall limit",
        }
    if err is not None:
        return {**meta, "id": cid, "verdict": "ERROR", "wall_s": wall, "error": err}
    assert run is not None
    require_topo = abs(f_irr) <= 1e-15
    gates = _gates_from_result(
        run.result,
        run.spec,
        gate=0.001,
        require_bottom_connected_cz=require_topo,
    )
    passed = gates.convergence_ok and (gates.topology_ok or not require_topo)
    accepted = [d for d in run.result.diagnostics if d.accepted]
    failing = []
    if not gates.flux_flatness_ok:
        failing.append("flux_flatness")
    if not gates.tendency_ok:
        failing.append("tendency")
    if not gates.energy_ok:
        failing.append("energy")
    if not gates.algebraic_ok:
        failing.append("algebraic")
    if require_topo and not gates.topology_ok:
        failing.append("topology")
    return {
        **meta,
        "id": cid,
        "verdict": "CONVERGED" if passed else "NOT CONVERGED",
        "wall_s": wall,
        "phases": run.phases,
        "n_accepted_steps": len(accepted),
        "flux_flatness": gates.flux_flatness,
        "tendency_norm": gates.tendency_norm,
        "flux_flatness_ok": gates.flux_flatness_ok,
        "tendency_ok": gates.tendency_ok,
        "energy_ok": gates.energy_ok,
        "algebraic_ok": gates.algebraic_ok,
        "topology_ok": gates.topology_ok,
        "failing_gates": failing,
        "primary_rcb_log10p": run.result.primary_rcb_log10p,
        "cz_layers": sum(hi - lo + 1 for lo, hi in run.result.convective_regions),
        "detached": run.result.detached_convective_regions,
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    seen: set[str] = set()

    def add(**kw) -> None:
        cid = _case_id(**{k: kw[k] for k in sorted(kw)})
        if cid in seen:
            return
        seen.add(cid)
        cases.append(kw)

    # Thermal: high He
    for x_he in X_HE_THERMAL:
        add(x_he=x_he, f_irr=0.0)

    # Irradiation ladder at moderate-high He
    for x_he in F_IRR_AT_XHE:
        for f_irr in F_IRR_LADDER:
            add(x_he=x_he, f_irr=f_irr)

    # He ladder at extreme irradiation
    for f_irr in (3000.0, 4000.0, 5000.0, 6000.0):
        for x_he in X_HE_AT_FIRR:
            add(x_he=x_he, f_irr=f_irr)

    # Knob variations
    for kw in KNOB_CASES:
        add(**kw)

    # RE seed on a hot corner
    add(x_he=0.6, f_irr=3000.0, seed="radiative_equilibrium")
    add(x_he=0.8, f_irr=2000.0, seed="radiative_equilibrium")

    return cases


def summarize(results: list[dict]) -> dict:
    converged = [r for r in results if r["verdict"] == "CONVERGED"]
    failed = [r for r in results if r["verdict"] == "NOT CONVERGED"]
    errors = [r for r in results if r["verdict"] in {"ERROR", "TIMEOUT"}]

    def _max_firr_ok(x_he: float) -> float | None:
        ok = [r["f_irr"] for r in converged if r["x_he"] == x_he and r.get("alpha", 1.0) == 1.0]
        return max(ok) if ok else None

    def _max_xhe_ok(f_irr: float) -> float | None:
        ok = [r["x_he"] for r in converged if r["f_irr"] == f_irr and r.get("alpha", 1.0) == 1.0]
        return max(ok) if ok else None

    thermal = [r for r in results if r["f_irr"] == 0.0 and r.get("alpha", 1.0) == 1.0]
    thermal_fail = [r for r in thermal if r["verdict"] != "CONVERGED"]

    return {
        "n_total": len(results),
        "n_converged": len(converged),
        "n_failed": len(failed),
        "n_error": len(errors),
        "thermal_x_he_limit": {
            "last_converged": max((r["x_he"] for r in thermal if r["verdict"] == "CONVERGED"), default=None),
            "first_failed": min((r["x_he"] for r in thermal_fail), default=None) if thermal_fail else None,
        },
        "max_f_irr_converged_by_x_he": {str(x): _max_firr_ok(x) for x in F_IRR_AT_XHE},
        "max_x_he_converged_by_f_irr": {str(int(f)): _max_xhe_ok(f) for f in (2000, 3000, 4000, 5000, 6000)},
        "hardest_converged": min(converged, key=lambda r: r["flux_flatness"], default=None),
        "worst_failures": sorted(
            failed,
            key=lambda r: r.get("flux_flatness") or 0.0,
            reverse=True,
        )[:10],
    }


def main() -> None:
    cases = build_cases()
    results = []
    print(f"Running {len(cases)} boundary cases...", flush=True)
    for i, kw in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {kw}", flush=True)
        row = run_case(**kw)
        print(json.dumps({k: row[k] for k in ("id", "verdict", "wall_s", "flux_flatness", "failing_gates") if k in row}), flush=True)
        results.append(row)
    summary = summarize(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "summary": str(SUMMARY), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
