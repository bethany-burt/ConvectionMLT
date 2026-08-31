#!/usr/bin/env python3
"""Isolation sweep: x_he x f_irr for advisor diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path

from convection_mlt.production_rce import (
    ProductionControls,
    _gates_from_result,
    run_production_rce,
)

OUT = Path(__file__).resolve().parent / "runs" / "isolation_sweep.json"

# Match cfg_demo.py solver budget
CONTROLS = ProductionControls(
    max_steps_live_polish=200,
    max_steps_continuation=500,
    max_recovery_cycles=2,
    dt_accuracy_s=50000.0,
    dt_hold_init_s=18415.0,
    continuation_dt_accuracy_s=2500.0,
)

CASES = [
    {"id": "A_baseline", "x_he": 0.0, "f_irr": 0.0, "note": "Validated thermal (pure H2)"},
    {"id": "B_he_only", "x_he": 0.2, "f_irr": 0.0, "note": "He isolation (20% He, no irradiation)"},
    {"id": "C_irr_only", "x_he": 0.0, "f_irr": 500.0, "note": "Irradiation isolation (pure H2, F_irr=500)"},
    {"id": "D_irr120_pure", "x_he": 0.0, "f_irr": 120.0, "note": "Stage-4 default nested F_irr=120, pure H2"},
    {"id": "E_he_irr120", "x_he": 0.2, "f_irr": 120.0, "note": "He + validated irradiation level"},
    {"id": "F_user_case", "x_he": 0.2, "f_irr": 500.0, "note": "Current cfg_demo.py combination"},
    {"id": "G_he01_irr500", "x_he": 0.1, "f_irr": 500.0, "note": "Demonstrated He + strong irradiation"},
]


def run_one(case: dict) -> dict:
    x_he = float(case["x_he"])
    f_irr = float(case["f_irr"])
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
    return {
        **case,
        "verdict": "CONVERGED" if passed else "NOT CONVERGED",
        "wall_s": wall,
        "phases": run.phases,
        "n_accepted_steps": len([d for d in run.result.diagnostics if d.accepted]),
        "flux_flatness": gates.flux_flatness,
        "tendency_norm": gates.tendency_norm,
        "energy_gate_ratio": gates.energy_gate_ratio,
        "flux_flatness_ok": gates.flux_flatness_ok,
        "tendency_ok": gates.tendency_ok,
        "energy_ok": gates.energy_ok,
        "algebraic_ok": gates.algebraic_ok,
        "topology_ok": gates.topology_ok,
        "primary_rcb_log10p": run.result.primary_rcb_log10p,
        "convective_regions": run.result.convective_regions,
        "detached_convective_regions": run.result.detached_convective_regions,
        "cz_layers": (
            None
            if not run.result.convective_regions
            else run.result.convective_regions[0][1] - run.result.convective_regions[0][0] + 1
        ),
    }


def main() -> None:
    results = []
    for case in CASES:
        print(f"RUN {case['id']} x_he={case['x_he']} f_irr={case['f_irr']}", flush=True)
        try:
            row = run_one(case)
        except Exception as exc:
            row = {**case, "verdict": "ERROR", "error": str(exc)}
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
