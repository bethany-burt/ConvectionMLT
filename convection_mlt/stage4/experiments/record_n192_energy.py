"""Short N=192 restart that serializes the ULP-aware committed-energy fields."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from continue_n192 import (
    CHECKPOINT,
    OUT,
    _load,
    _merge_chunk,
    _refresh_audit,
    _run_chunk,
    _write_record,
)
from rce_record import dumps, enrich_stored_record, finalize_record

ENERGY_STEPS = 8
ENERGY_DT = 500.0


def main() -> dict:
    record = enrich_stored_record(_load())
    last_dt = float(record.get("last_accepted_dt") or ENERGY_DT)
    payload, res = _run_chunk(
        record["temperature"],
        max_steps=ENERGY_STEPS,
        dt_hold=min(last_dt, ENERGY_DT),
        rcb=record.get("primary_rcb_log10p"),
        simulated_time=float(record.get("simulated_time") or 0.0),
        dt_accuracy=ENERGY_DT,
    )
    extra_done = int((record.get("continuation") or {}).get("energy_closed_extra_accepted") or 0)
    record, extra_done = _merge_chunk(record, payload, extra_done, dict(record.get("continuation") or {}))
    record["continuation"]["energy_diagnostics_steps"] = int(payload.get("steps_accepted") or 0)
    record["continuation"]["energy_diagnostics_status"] = payload.get("status")
    finalize_record(record)
    _write_record(record)
    audit = _refresh_audit(record)
    print(
        dumps({
            "status": record.get("status"),
            "steps_accepted": record.get("steps_accepted"),
            "energy_committed_residual": record.get("energy_committed_residual"),
            "energy_scale": record.get("energy_scale"),
            "energy_ulp_floor": record.get("energy_ulp_floor"),
            "energy_allowed": record.get("energy_allowed"),
            "energy_gate_ratio": record.get("energy_gate_ratio"),
            "chunk_status": res.status.value,
            "chunk_accepted": payload.get("steps_accepted"),
            "algebraic": audit["algebraic_identity_status"],
            "out": str(OUT),
            "checkpoint": str(CHECKPOINT),
        }),
        flush=True,
    )
    return record


if __name__ == "__main__":
    main()
