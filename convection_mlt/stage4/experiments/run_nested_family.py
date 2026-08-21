"""Gate-converged nested N=48 and N=96 from the same 384-layer master as N=192.

Independent-grid N=48/96 remain the cheap regression pair. Richardson analysis
uses this nested family only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    RCERoute,
    TopIrradiation,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.energy import column_enthalpy_per_area

from rce_record import (
    PHYSICAL_GATE,
    dumps,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)

OUT = ROOT / "results" / "nested_rce_family.json"
MAX_STEPS = {48: 5000, 96: 20000}


def _run_one(n_layers: int) -> dict:
    spec = nested_analytic_opacity_spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    t0 = radiative_convective_initial_temperature(
        grid, spec.opacity(), thermo, spec.f_int, spec.f_irr
    )
    cfg = production_rce_config(max_steps=MAX_STEPS[n_layers], dt_accuracy=2500.0)
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
            "max_steps_budget": MAX_STEPS[n_layers],
            "physical_gate": PHYSICAL_GATE,
        },
    )
    print(
        f"N={n_layers}",
        res.status.value,
        "flat", res.convergence.flux_flatness,
        "tend", res.convergence.tendency_norm,
        "acc", res.steps_accepted,
        "rej", res.rejections,
        "rcb", res.primary_rcb_log10p,
        "wall", round(wall, 1),
        flush=True,
    )
    return payload


def interpolate_temperature(log_p_src, t_src, log_p_dst):
    import numpy as np

    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def _compare(coarse: dict, fine: dict) -> dict:
    import numpy as np

    p_c = np.asarray(coarse["pressure_centres"], dtype=np.float64)
    p_f = np.asarray(fine["pressure_centres"], dtype=np.float64)
    t_c = np.asarray(coarse["temperature"], dtype=np.float64)
    t_f_on_c = interpolate_temperature(np.log(p_f), fine["temperature"], np.log(p_c))
    scale = np.maximum(np.abs(t_c), 1.0)
    rcb_c = coarse.get("primary_rcb_log10p")
    rcb_f = fine.get("primary_rcb_log10p")
    h_c = float(coarse.get("column_enthalpy") or column_enthalpy_per_area(
        coarse["mass_path"], coarse["enthalpy"]
    ))
    h_f = float(fine.get("column_enthalpy") or column_enthalpy_per_area(
        fine["mass_path"], fine["enthalpy"]
    ))
    return {
        "coarse_n": coarse["n_layers"],
        "fine_n": fine["n_layers"],
        "both_converged": coarse.get("status") == "converged" and fine.get("status") == "converged",
        "max_rel_T_on_coarse_P": float(np.max(np.abs(t_f_on_c - t_c) / scale)),
        "delta_log10_P_rcb": (
            None if rcb_c is None or rcb_f is None else abs(float(rcb_f) - float(rcb_c))
        ),
        "column_enthalpy_rel": abs(h_f - h_c) / max(abs(h_c), abs(h_f), 1.0),
        "coarse_rcb": rcb_c,
        "fine_rcb": rcb_f,
    }


def richardson_order(e_coarse_vs_mid: float, e_mid_vs_fine: float) -> float | None:
    import numpy as np

    if e_coarse_vs_mid <= 0.0 or e_mid_vs_fine <= 0.0:
        return None
    return float(np.log2(e_coarse_vs_mid / e_mid_vs_fine))


def main(layers=(48, 96)) -> dict:
    import numpy as np

    cases = {}
    if OUT.exists():
        cases = json.loads(OUT.read_text())
        cases = dict(cases.get("members") or cases)
    for n in layers:
        key = str(n)
        existing = cases.get(key)
        if existing and existing.get("status") == "converged":
            print(f"N={n} already converged; skipping", flush=True)
            continue
        cases[key] = _run_one(n)
        OUT.write_text(dumps({"members": cases}))

    n192_path = ROOT / "results" / "n192_implicit_rce.json"
    n192 = json.loads(n192_path.read_text()) if n192_path.exists() else None
    comparisons = {}
    if "48" in cases and "96" in cases:
        comparisons["96_vs_48"] = _compare(cases["48"], cases["96"])
    if n192 is not None and "96" in cases:
        comparisons["192_vs_96"] = _compare(cases["96"], n192)
    if n192 is not None and "48" in cases:
        comparisons["192_vs_48"] = _compare(cases["48"], n192)

    order = None
    pair_c = comparisons.get("96_vs_48") or {}
    pair_f = comparisons.get("192_vs_96") or {}
    if pair_c.get("both_converged") and pair_f.get("both_converged"):
        order = richardson_order(
            float(pair_c["max_rel_T_on_coarse_P"]),
            float(pair_f["max_rel_T_on_coarse_P"]),
        )

    payload = {
        "members": cases,
        "comparisons": comparisons,
        "richardson_order_from_max_rel_T": order,
        "note": (
            "Nested τ-family from master N=384, n_phot=64. Independent-grid "
            "N=48/96 are not used here."
        ),
    }
    OUT.write_text(dumps(payload))
    print(json.dumps({
        "out": str(OUT),
        "comparisons": comparisons,
        "richardson_order_from_max_rel_T": order,
    }, indent=2, default=str), flush=True)
    return payload


if __name__ == "__main__":
    main()
