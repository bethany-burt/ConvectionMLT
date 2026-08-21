"""Run nested N=192 implicit RCE and store the complete result."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    RCEConfig,
    RCERoute,
    SolverConfig,
    TopIrradiation,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)

from rce_record import dumps, serialize_rce_result

OUT = ROOT / "results" / "n192_implicit_rce.json"
GATE = 1.0e-3


def main(max_steps: int = 12000) -> dict:
    spec = nested_analytic_opacity_spec(192)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    cfg = RCEConfig(
        max_steps=max_steps,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=GATE,
        tendency_tolerance=GATE,
        temp_change_tolerance=GATE,
        dt_accuracy=2500.0,
        coupled_picard=True,
        use_coupled_tendency_dt=True,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10,
            step_tolerance=1e-10,
            newton_residual_tolerance=1e-12,
            newton_step_tolerance=1e-12,
        ),
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid, t0, spec.physics(),
        SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14),
        thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    wall = time.perf_counter() - wall0
    payload = serialize_rce_result(
        res, spec, pressure_centres=grid.pressure_centres,
        extra={"wall_time_s": wall, "max_steps_budget": max_steps, "physical_gate": GATE},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dumps(payload))
    print(
        res.status.value,
        "flat", res.convergence.flux_flatness,
        "tend", res.convergence.tendency_norm,
        "acc", res.steps_accepted,
        "rej", res.rejections,
        "median_dt", payload["median_accepted_dt"],
        "wall", round(wall, 1),
        "out", OUT,
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
