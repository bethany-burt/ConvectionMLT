"""Restart the stored nested N=192 state and continue toward the 1e-3 gate.

Timestep policy for continuation is dt_accuracy = 100 s: a direct restart
probe from the stored state passed at 100 s (residual 3.7e-11) and failed at
180 s. Gates are unchanged (flatness, tendency, temperature change, RCB
stability, five consecutive accepted steps).
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
    solve_adaptive_rce,
)

from rce_record import (
    PHYSICAL_GATE,
    _record_checksum,
    dumps,
    enrich_stored_record,
    merge_continuation,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)

OUT = ROOT / "results" / "n192_implicit_rce.json"
CHECKPOINT = ROOT / "results" / "n192_implicit_rce.checkpoint.json"
CHUNK_ACCEPTED = 2500
MAX_EXTRA_ACCEPTED = 50000
# Largest probed Δt that still meets the projection residual from the stored state.
CONTINUATION_DT_ACCURACY = 100.0


def _load() -> dict:
    return json.loads(OUT.read_text())


def _run_chunk(temperature, *, max_steps: int, dt_hold: float, rcb) -> tuple[dict, object]:
    spec = nested_analytic_opacity_spec(192)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    cfg = production_rce_config(
        max_steps=max_steps,
        dt_accuracy=CONTINUATION_DT_ACCURACY,
        dt_hold_init=min(float(dt_hold), CONTINUATION_DT_ACCURACY),
        previous_rcb_init=rcb,
        gate=PHYSICAL_GATE,
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        temperature,
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
            "max_steps_budget": max_steps,
            "physical_gate": PHYSICAL_GATE,
            "continuation_dt_accuracy": CONTINUATION_DT_ACCURACY,
        },
    )
    return payload, res


def main(max_extra: int = MAX_EXTRA_ACCEPTED, chunk: int = CHUNK_ACCEPTED) -> dict:
    record = enrich_stored_record(_load())
    OUT.write_text(dumps(record))
    extra_done = int((record.get("continuation") or {}).get("extra_accepted") or 0)
    remaining = max(0, max_extra - extra_done)
    print(
        "continue_n192 start",
        "status", record.get("status"),
        "flat", record.get("flux_flatness"),
        "tend", record.get("tendency_norm"),
        "acc", record.get("steps_accepted"),
        "extra_done", extra_done,
        "remaining", remaining,
        flush=True,
    )
    if record.get("status") == "converged":
        print("already converged; nothing to do", flush=True)
        return record
    if remaining <= 0:
        print("extra-accepted budget exhausted", flush=True)
        return record

    while remaining > 0 and record.get("status") != "converged":
        this_chunk = min(chunk, remaining)
        t = record["temperature"]
        last_dt = record.get("last_accepted_dt")
        if last_dt is None:
            hist = record.get("history") or {}
            dts = hist.get("dt") or [CONTINUATION_DT_ACCURACY]
            last_dt = dts[-1]
        payload, res = _run_chunk(
            t,
            max_steps=this_chunk,
            dt_hold=float(last_dt),
            rcb=record.get("primary_rcb_log10p"),
        )
        record = merge_continuation(record, payload)
        extra_done += int(payload["steps_accepted"])
        remaining = max(0, max_extra - extra_done)
        record["continuation"] = {
            "extra_accepted": extra_done,
            "max_extra_accepted": max_extra,
            "chunk_accepted": chunk,
            "continuation_dt_accuracy": CONTINUATION_DT_ACCURACY,
            "note": (
                "Restart from stored N=192 state. Δt held at 100 s because a "
                "direct probe passed at 100 s and failed at 180 s. Original "
                "1e-3 gates and n_consec=5 are unchanged."
            ),
        }
        record["record_checksum_sha256"] = _record_checksum(record)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        text = dumps(record)
        OUT.write_text(text)
        CHECKPOINT.write_text(text)
        print(
            res.status.value,
            "flat", record["flux_flatness"],
            "tend", record["tendency_norm"],
            "acc", record["steps_accepted"],
            "rej", record["rejections"],
            "t_sim", record["simulated_time"],
            "extra", extra_done,
            "chunk_wall", round(payload["wall_time_s"], 1),
            "median_dt", record.get("median_accepted_dt"),
            flush=True,
        )
        if payload["steps_accepted"] == 0:
            print("chunk accepted zero steps; stopping", flush=True)
            break
    return record


if __name__ == "__main__":
    main()
