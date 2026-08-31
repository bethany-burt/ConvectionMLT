#!/usr/bin/env python3
"""Stiffness sweep: higher He abundances and F_irr up to 2000 W m^-2 (N=96)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from convection_mlt.production_rce import (
    ProductionControls,
    _gates_from_result,
    run_production_rce,
)

OUT = Path(__file__).resolve().parent / "runs" / "realistic_he_irr_sweep.json"

CONTROLS = ProductionControls(
    max_steps_live_polish=200,
    max_steps_continuation=500,
    max_recovery_cycles=2,
    dt_accuracy_s=50000.0,
    dt_hold_init_s=18415.0,
    continuation_dt_accuracy_s=2500.0,
)

X_HE = (0.2, 0.25, 0.3, 0.35, 0.4)
F_IRR = (0.0, 500.0, 1000.0, 1500.0, 2000.0)


def run_one(x_he: float, f_irr: float) -> dict:
    t0 = time.perf_counter()
    run = run_production_rce(
        n_layers=96,
        alpha=1.0,
        f_int=300.0,
        f_irr=f_irr,
        gravity=15.0,
        p_bottom=1e6,
        p_top=1.0,
        x_he=x_he,
        seed="radiative_convective",
        procedure="production",
        controls=CONTROLS,
    )
    wall = time.perf_counter() - t0
    require_topo = abs(f_irr) <= 1e-15
    gates = _gates_from_result(
        run.result,
        run.spec,
        gate=0.001,
        require_bottom_connected_cz=require_topo,
    )
    passed = gates.convergence_ok and (gates.topology_ok or not require_topo)
    accepted = [d for d in run.result.diagnostics if d.accepted]
    return {
        "x_he": x_he,
        "f_irr": f_irr,
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
        "primary_rcb_log10p": run.result.primary_rcb_log10p,
        "convective_regions": run.result.convective_regions,
        "cz_layers": sum(hi - lo + 1 for lo, hi in run.result.convective_regions),
    }


def main() -> None:
    results = []
    for x_he in X_HE:
        for f_irr in F_IRR:
            label = f"x_he={x_he} f_irr={f_irr}"
            print(f"--- {label} ---", flush=True)
            row = run_one(x_he, f_irr)
            print(json.dumps(row, indent=2), flush=True)
            results.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    n_ok = sum(r["verdict"] == "CONVERGED" for r in results)
    print(
        json.dumps(
            {
                "out": str(OUT),
                "converged": n_ok,
                "total": len(results),
                "max_flatness": max(r["flux_flatness"] for r in results),
                "max_steps": max(r["n_accepted_steps"] for r in results),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
