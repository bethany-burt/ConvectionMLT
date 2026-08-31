"""Checkpointed nested N=384 from the gated N=192 column."""

from __future__ import annotations

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
    nested_analytic_opacity_spec,
    solve_adaptive_rce,
)

from build_current_exit_audit import main as rebuild_audit
from rce_record import (
    PHYSICAL_GATE,
    dumps,
    finalize_record,
    merge_continuation,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)
from run_nested_family import interpolate_temperature, main as rebuild_family

OUT = ROOT / "results" / "n384_implicit_rce.json"
CHECKPOINT = ROOT / "results" / "n384_implicit_rce.checkpoint.json"
N192 = ROOT / "results" / "n192_implicit_rce.json"
STATUS = ROOT / "STAGE4_STATUS_REPORT.txt"
CHUNK_ACCEPTED = 500
MAX_STEPS = 20000
DT_ACCURACY = 50000.0
TIGHTEN_STEPS = 8
TIGHTEN_DT = 500.0


def _write(record: dict) -> None:
    finalize_record(record)
    text = dumps(record)
    OUT.write_text(text)
    CHECKPOINT.write_text(text)


def _run_chunk(temperature, *, max_steps, dt_hold, rcb, simulated_time, dt_accuracy) -> dict:
    spec = nested_analytic_opacity_spec(384)
    grid = spec.grid()
    solver = production_solver_config()
    cfg = production_rce_config(
        max_steps=max_steps,
        dt_accuracy=dt_accuracy,
        dt_hold_init=float(dt_hold),
        previous_rcb_init=rcb,
        simulated_time_init=float(simulated_time),
        gate=PHYSICAL_GATE,
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        temperature,
        spec.physics(),
        solver,
        ConstantH2Thermo(),
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    wall = time.perf_counter() - wall0
    return serialize_rce_result(
        res,
        spec,
        pressure_centres=grid.pressure_centres,
        pressure_edges=grid.pressure_edges,
        solver=solver,
        rce_config=cfg,
        extra={
            "wall_time_s": wall,
            "max_steps_budget": max_steps,
            "physical_gate": PHYSICAL_GATE,
            "initial_from": "interpolated_n192",
            "continuation_dt_accuracy": dt_accuracy,
        },
    )


def _seed_from_n192() -> dict:
    n192 = json.loads(N192.read_text())
    spec = nested_analytic_opacity_spec(384)
    grid = spec.grid()
    t0 = interpolate_temperature(
        np.log(np.asarray(n192["pressure_centres"], dtype=np.float64)),
        n192["temperature"],
        np.log(grid.pressure_centres),
    )
    last_dt = float(n192.get("last_accepted_dt") or 2500.0)
    return {
        "temperature": t0.tolist(),
        "primary_rcb_log10p": n192.get("primary_rcb_log10p"),
        "last_accepted_dt": min(last_dt, DT_ACCURACY),
        "simulated_time": 0.0,
        "steps_accepted": 0,
        "rejections": 0,
        "status": "seeded",
        "continuation": {"extra_accepted": 0, "phase": "n384_from_n192"},
    }


def _write_status_report(record: dict, audit: dict) -> None:
    from build_current_exit_audit import _write_status_report as write_status

    n192 = json.loads(N192.read_text()) if N192.exists() else {}
    write_status(audit, n192, record)


def _refresh_artifacts(record: dict) -> dict:
    rebuild_family(layers=(), force=False)
    return rebuild_audit()


def main(max_steps: int = MAX_STEPS, chunk: int = CHUNK_ACCEPTED) -> dict:
    if OUT.exists():
        record = json.loads(OUT.read_text())
    elif CHECKPOINT.exists():
        record = json.loads(CHECKPOINT.read_text())
    else:
        record = _seed_from_n192()
    extra_done = int((record.get("continuation") or {}).get("extra_accepted") or record.get("steps_accepted") or 0)
    remaining = max(0, max_steps - extra_done)
    print(
        "continue_n384 start",
        "status", record.get("status"),
        "flat", record.get("flux_flatness"),
        "acc", record.get("steps_accepted"),
        "remaining", remaining,
        flush=True,
    )
    while remaining > 0 and record.get("status") != "converged":
        this = min(chunk, remaining)
        payload = _run_chunk(
            record["temperature"],
            max_steps=this,
            dt_hold=float(record.get("last_accepted_dt") or DT_ACCURACY),
            rcb=record.get("primary_rcb_log10p"),
            simulated_time=float(record.get("simulated_time") or 0.0),
            dt_accuracy=DT_ACCURACY,
        )
        if record.get("status") == "seeded":
            record = payload
            record["continuation"] = {
                "extra_accepted": int(payload.get("steps_accepted") or 0),
                "phase": "n384_from_n192",
                "max_steps": max_steps,
                "chunk_accepted": chunk,
            }
        else:
            record = merge_continuation(record, payload)
            record["continuation"]["extra_accepted"] = int(
                (record.get("continuation") or {}).get("extra_accepted") or 0
            ) + int(payload.get("steps_accepted") or 0)
        extra_done = int(record["continuation"]["extra_accepted"])
        remaining = max(0, max_steps - extra_done)
        _write(record)
        print(
            record.get("status"),
            "flat", record.get("flux_flatness"),
            "tend", record.get("tendency_norm"),
            "acc", record.get("steps_accepted"),
            "t_sim", record.get("simulated_time"),
            "chunk_wall", round(float(payload.get("wall_time_s") or 0.0), 1),
            flush=True,
        )
        if int(payload.get("steps_accepted") or 0) == 0:
            print("chunk accepted zero steps; stopping", flush=True)
            break

    if record.get("status") == "converged":
        if record.get("tighten_status") is None:
            print("N=384 gated; tightening", TIGHTEN_STEPS, "steps at", TIGHTEN_DT, flush=True)
            hold = _run_chunk(
                record["temperature"],
                max_steps=TIGHTEN_STEPS,
                dt_hold=TIGHTEN_DT,
                rcb=record.get("primary_rcb_log10p"),
                simulated_time=float(record.get("simulated_time") or 0.0),
                dt_accuracy=TIGHTEN_DT,
            )
            record = merge_continuation(record, hold)
            record["tighten_status"] = hold.get("status")
            record["tighten_steps"] = hold.get("steps_accepted")
            record["tighten_dt_accuracy"] = TIGHTEN_DT
            _write(record)
            print(
                "tighten",
                hold.get("status"),
                "flat", record.get("flux_flatness"),
                "energy_ratio", record.get("energy_gate_ratio"),
                flush=True,
            )
        else:
            print("N=384 already tightened; skipping tighten", flush=True)
        _refresh_artifacts(record)
    return record


if __name__ == "__main__":
    main()
