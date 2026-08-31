"""Dedicated F_irr=0 MLT RCE on the HELIOS geometric interface grid.

This is the matched-forcing coupled-HELIOS benchmark reference. It is not the
irradiated nested family. Freeze the profile checksum before any live HELIOS run.

A radiative-convective seed on the log-P HELIOS grid stalls with a detached
mid-column cell. The production seed is therefore a nested τ-grid F_irr=0
solve interpolated onto HELIOS interfaces, then relaxed on that grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    TopIrradiation,
    build_grid,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.adapters.helios_contracts import HELIOS_DEFAULT_DIFFUSIVITY
from convection_mlt.adapters.helios_grid import build_helios_grid_from_nested_edges

from export_helios_grid_reference import _load_record
from rce_record import (
    PHYSICAL_GATE,
    dumps,
    finalize_record,
    merge_continuation,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)

RESULTS = ROOT / "results"
FIXTURES = ROOT / "fixtures" / "helios"
OUT_N = {96: RESULTS / "mlt_nested_tau_n96_firr0.json", 192: RESULTS / "mlt_nested_tau_n192_firr0.json"}
FIXTURE_N = {
    96: FIXTURES / "mlt_nested_tau_n96_firr0.json",
    192: FIXTURES / "mlt_nested_tau_n192_firr0.json",
}
MAX_STEPS = {96: 20000, 192: 40000}
NESTED_SEED_STEPS = {96: 5000, 192: 12000}


def interpolate_temperature(log_p_src, t_src, log_p_dst):
    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def helios_interface_mlt_grid(n_layers: int):
    nested = _load_record(n_layers)
    edges = np.asarray(nested["pressure_edges"], dtype=np.float64)
    helios = build_helios_grid_from_nested_edges(edges, n_layers)
    mlt = build_grid(helios.p_int_Pa, nested.get("gravity") or 15.0)
    return nested, helios, mlt


def _spec(n_layers: int):
    return nested_analytic_opacity_spec(
        n_layers, f_irr=0.0, diffusivity_factor=HELIOS_DEFAULT_DIFFUSIVITY
    )


def _run_mlt(grid, t0, spec, *, max_steps: int, dt_accuracy: float, dt_hold=None,
             previous_rcb=None, simulated_time: float = 0.0):
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    cfg = production_rce_config(
        max_steps=max_steps,
        dt_accuracy=dt_accuracy,
        dt_hold_init=dt_hold,
        previous_rcb_init=previous_rcb,
        simulated_time_init=simulated_time,
        diffusivity_factor=HELIOS_DEFAULT_DIFFUSIVITY,
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(0.0),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    return res, cfg, solver, time.perf_counter() - wall0


def solve_nested_tau_firr0(n_layers: int) -> dict:
    spec = _spec(n_layers)
    t0 = radiative_convective_initial_temperature(
        spec.grid(), spec.opacity(), ConstantH2Thermo(), spec.f_int, 0.0,
        diffusivity_factor=HELIOS_DEFAULT_DIFFUSIVITY,
    )
    res, cfg, solver, wall = _run_mlt(
        spec.grid(), t0, spec,
        max_steps=NESTED_SEED_STEPS[n_layers],
        dt_accuracy=2500.0,
    )
    payload = serialize_rce_result(
        res, spec, pressure_centres=spec.grid().pressure_centres,
        pressure_edges=spec.grid().pressure_edges, solver=solver, rce_config=cfg,
        extra={
            "wall_time_s": wall,
            "physical_gate": PHYSICAL_GATE,
            "f_irr": 0.0,
            "mlt_grid": "nested_tau",
            "seed_for_helios_grid": True,
        },
    )
    finalize_record(payload)
    return payload


def _payload(res, spec, grid, helios, nested, cfg, solver, wall, extra):
    payload = serialize_rce_result(
        res,
        spec,
        pressure_centres=grid.pressure_centres,
        pressure_edges=grid.pressure_edges,
        solver=solver,
        rce_config=cfg,
        extra={
            "wall_time_s": wall,
            "physical_gate": PHYSICAL_GATE,
            "f_irr": 0.0,
            "forcing": "F_int=300, F_irr=0",
            "mlt_grid": "helios_geometric_interfaces",
            "helios_p_lay_Pa": helios.p_lay_Pa.tolist(),
            "helios_p_int_Pa": helios.p_int_Pa.tolist(),
            "helios_p_boa_microbar": helios.p_boa_microbar,
            "helios_p_toa_microbar": helios.p_toa_microbar,
            "diffusivity_factor": HELIOS_DEFAULT_DIFFUSIVITY,
            "nested_irradiated_checksum": nested.get("profile_checksum_sha256"),
            "nested_irradiated_is_structural_only": True,
            "comparison_type": "matched_forcing_helios_grid_mlt_reference",
            "coupled_helios_rce_status": "NOT_RUN",
            **extra,
        },
    )
    finalize_record(payload)
    return payload


def solve_firr0(
    n_layers: int,
    *,
    max_steps: int | None = None,
    seed: str = "nested",
    dt_accuracy: float = 2500.0,
) -> dict:
    spec = _spec(n_layers)
    nested, helios, grid = helios_interface_mlt_grid(n_layers)
    out = OUT_N[n_layers]
    extra = {"seed": seed}

    if seed == "continue":
        if not out.exists():
            raise FileNotFoundError(f"no HELIOS-grid record to continue: {out}")
        base = json.loads(out.read_text())
        t0 = np.asarray(base["temperature"], dtype=np.float64)
        res, cfg, solver, wall = _run_mlt(
            grid, t0, spec,
            max_steps=max_steps if max_steps is not None else MAX_STEPS[n_layers],
            dt_accuracy=dt_accuracy,
            dt_hold=base.get("last_accepted_dt") or base.get("dt_hold"),
            previous_rcb=base.get("primary_rcb_log10p"),
            simulated_time=float(base.get("simulated_time") or 0.0),
        )
        chunk = _payload(res, spec, grid, helios, nested, cfg, solver, wall, extra)
        merged = merge_continuation(base, chunk)
        finalize_record(merged)
        return merged

    if seed == "nested":
        nested_seed = solve_nested_tau_firr0(n_layers)
        extra["nested_firr0_checksum"] = nested_seed.get("profile_checksum_sha256")
        extra["nested_firr0_status"] = nested_seed.get("status")
        extra["nested_firr0_rcb_log10p"] = nested_seed.get("primary_rcb_log10p")
        extra["nested_firr0_regions"] = nested_seed.get("convective_regions")
        t0 = interpolate_temperature(
            np.log(np.asarray(nested_seed["pressure_centres"], dtype=np.float64)),
            nested_seed["temperature"],
            np.log(grid.pressure_centres),
        )
        (RESULTS / f"mlt_nested_tau_n{n_layers}_firr0.json").write_text(dumps(nested_seed))
    else:
        t0 = radiative_convective_initial_temperature(
            grid, spec.opacity(), ConstantH2Thermo(), spec.f_int, 0.0,
            diffusivity_factor=HELIOS_DEFAULT_DIFFUSIVITY,
        )

    res, cfg, solver, wall = _run_mlt(
        grid, t0, spec,
        max_steps=max_steps if max_steps is not None else MAX_STEPS[n_layers],
        dt_accuracy=dt_accuracy,
    )
    return _payload(res, spec, grid, helios, nested, cfg, solver, wall, extra)


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96, choices=(96, 192))
    parser.add_argument("--seed", choices=("nested", "rc", "continue"), default="nested")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dt-accuracy", type=float, default=2500.0)
    parser.add_argument("--freeze-fixture", action="store_true")
    args = parser.parse_args()
    out = OUT_N[args.layers]
    payload = solve_firr0(
        args.layers,
        max_steps=args.max_steps,
        seed=args.seed,
        dt_accuracy=args.dt_accuracy,
    )
    out.write_text(dumps(payload))
    if args.freeze_fixture:
        if payload.get("status") != "converged":
            raise SystemExit(
                f"refusing to freeze: status={payload.get('status')} "
                f"flatness={payload.get('flux_flatness')}"
            )
        dest = FIXTURE_N[args.layers]
        dest.write_text(dumps(payload))
        print(json.dumps({"fixture": str(dest)}, indent=2), flush=True)
    print(json.dumps({
        "out": str(out),
        "status": payload.get("status"),
        "flux_flatness": payload.get("flux_flatness"),
        "tendency_norm": payload.get("tendency_norm"),
        "f_irr": payload.get("f_irr"),
        "primary_rcb_log10p": payload.get("primary_rcb_log10p"),
        "convective_regions": payload.get("convective_regions"),
        "profile_checksum_sha256": payload.get("profile_checksum_sha256"),
        "record_checksum_sha256": payload.get("record_checksum_sha256"),
        "steps_accepted": payload.get("steps_accepted"),
        "mlt_grid": payload.get("mlt_grid"),
        "seed": payload.get("seed"),
        "nested_firr0_status": payload.get("nested_firr0_status"),
    }, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
