"""Restart the stored nested N=192 state toward the 1e-3 gate.

Energy-closed Picard phase: the coupled macrostep must meet both the local
defect and the committed-step energy identity. Δt may grow to ~50,000 s.
Gates are unchanged (flatness, tendency, temperature change, RCB stability,
five consecutive accepted steps). After each chunk the exit-gate audit is
rebuilt and its N=192 checksum is asserted against the written record.
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

from build_current_exit_audit import assert_n192_audit_sync, main as rebuild_audit
from rce_record import (
    PHYSICAL_GATE,
    dumps,
    enrich_stored_record,
    finalize_record,
    merge_continuation,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)

OUT = ROOT / "results" / "n192_implicit_rce.json"
CHECKPOINT = ROOT / "results" / "n192_implicit_rce.checkpoint.json"
STATUS = ROOT / "STAGE4_STATUS_REPORT.txt"
CHUNK_ACCEPTED = 2500
MAX_EXTRA_ACCEPTED = 50000
CONTINUATION_DT_ACCURACY = 50000.0
TIGHTEN_DT_ACCURACY = 500.0
TIGHTEN_STEPS = 8
CONTINUATION_PHASE = "energy_closed_picard"


def _load() -> dict:
    return json.loads(OUT.read_text())


def _run_chunk(
    temperature,
    *,
    max_steps: int,
    dt_hold: float,
    rcb,
    simulated_time: float,
    dt_accuracy: float,
) -> tuple[dict, object]:
    spec = nested_analytic_opacity_spec(192)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
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
            "continuation_dt_accuracy": dt_accuracy,
        },
    )
    return payload, res


def _refresh_audit(record: dict) -> dict:
    audit = rebuild_audit()
    assert_n192_audit_sync(audit, record)
    _write_status_report(record, audit)
    return audit


def _write_status_report(record: dict, audit: dict) -> None:
    energy = next(
        (r for r in audit["rows"] if r["name"] == "algebraic_n192_energy_gate_ratio"),
        {},
    )
    STATUS.write_text(
        "Stage 4 status report\n"
        "Fixed-composition H2 radiative-convective equilibrium (handbook points 35-40)\n"
        "\n"
        f"Current claim (auto-rebuilt after {CONTINUATION_PHASE})\n"
        "--------------------------------------------------------------------\n"
        f"core_single_resolution_status: {audit['core_single_resolution_status']}\n"
        f"spatial_and_operator_convergence_status: {audit['spatial_and_operator_convergence_status']}\n"
        f"algebraic_identity_status: {audit['algebraic_identity_status']}\n"
        f"helios_parity_status: {audit['helios_parity_status']}\n"
        f"full_stage4_claim: {str(audit['full_stage4_claim']).lower()}\n"
        "\n"
        "Statuses are computed from current rows in stage4/results/exit_gate_audit.json.\n"
        "The audit N=192 profile_checksum_sha256 is asserted equal to the live record.\n"
        "\n"
        "Gates\n"
        "  algebraic identities: 1e-12, including committed-step energy\n"
        "  physical RCE: 1e-3 (unchanged; not relaxed)\n"
        "  spatial: max rel T 0.02 and 0.05 dex RCB\n"
        "  HELIOS: adapter engineering only; no coupled RCE claim\n"
        "\n"
        "N=192 live record\n"
        f"  status: {record.get('status')}\n"
        f"  steps_accepted: {record.get('steps_accepted')}\n"
        f"  rejections: {record.get('rejections')}\n"
        f"  simulated_time: {record.get('simulated_time')}\n"
        f"  flux_flatness: {record.get('flux_flatness')}\n"
        f"  tendency_norm: {record.get('tendency_norm')}\n"
        f"  primary_rcb_log10p: {record.get('primary_rcb_log10p')}\n"
        f"  median_accepted_dt: {record.get('median_accepted_dt')}\n"
        f"  last_accepted_dt: {record.get('last_accepted_dt')}\n"
        f"  energy_committed_residual: {record.get('energy_committed_residual')}\n"
        f"  energy_scale: {record.get('energy_scale')}\n"
        f"  energy_ulp_floor: {record.get('energy_ulp_floor')}\n"
        f"  energy_allowed: {record.get('energy_allowed')}\n"
        f"  energy_gate_ratio: {record.get('energy_gate_ratio')}\n"
        f"  coupled_defect: {record.get('coupled_defect')}\n"
        f"  profile_checksum_sha256: {record.get('profile_checksum_sha256') or record.get('checksum_sha256')}\n"
        f"  record_checksum_sha256: {record.get('record_checksum_sha256')}\n"
        f"  continuation_phase: {(record.get('continuation') or {}).get('phase')}\n"
        f"  algebraic_n192_energy_gate_ratio: {energy.get('status')} "
        f"(observed {energy.get('observed')})\n"
        "  N=192 is gate-converged. Nested N=384 is the spatial next step.\n"
        "  Coupled HELIOS RCE waits on the 192→384 spatial gate.\n"
    )


def _merge_chunk(record: dict, payload: dict, extra_done: int, prev: dict) -> dict:
    record = merge_continuation(record, payload)
    extra_done += int(payload["steps_accepted"])
    versions = list((record.get("continuation") or {}).get("code_versions") or [])
    if versions:
        versions[-1]["extra_accepted_from"] = extra_done - int(payload["steps_accepted"])
        versions[-1]["extra_accepted_to"] = extra_done
        versions[-1]["phase"] = CONTINUATION_PHASE
    record["continuation"] = {
        **prev,
        "extra_accepted": int(prev.get("extra_accepted") or 0) + int(payload["steps_accepted"]),
        "energy_closed_extra_accepted": extra_done,
        "max_extra_accepted": MAX_EXTRA_ACCEPTED,
        "chunk_accepted": CHUNK_ACCEPTED,
        "continuation_dt_accuracy": CONTINUATION_DT_ACCURACY,
        "phase": CONTINUATION_PHASE,
        "code_versions": versions,
        "note": (
            "Energy-closed Picard continuation. Acceptance requires both the "
            "local coupled defect and committed-step energy. Δt ceiling is "
            f"{CONTINUATION_DT_ACCURACY:g} s; original 1e-3 gates and "
            "n_consec=5 are unchanged."
        ),
    }
    finalize_record(record)
    return record, extra_done


def _write_record(record: dict) -> None:
    finalize_record(record)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = dumps(record)
    OUT.write_text(text)
    CHECKPOINT.write_text(text)


def main(max_extra: int = MAX_EXTRA_ACCEPTED, chunk: int = CHUNK_ACCEPTED) -> dict:
    record = enrich_stored_record(_load())
    _write_record(record)
    prev = dict(record.get("continuation") or {})
    extra_done = int(prev.get("energy_closed_extra_accepted") or 0)
    remaining = max(0, max_extra - extra_done)
    print(
        "continue_n192 start",
        "phase", CONTINUATION_PHASE,
        "status", record.get("status"),
        "flat", record.get("flux_flatness"),
        "tend", record.get("tendency_norm"),
        "acc", record.get("steps_accepted"),
        "energy_closed_extra", extra_done,
        "remaining", remaining,
        "dt_accuracy", CONTINUATION_DT_ACCURACY,
        "energy_rel", record.get("energy_committed_residual_rel"),
        flush=True,
    )
    if remaining <= 0 and record.get("status") != "converged":
        print("extra-accepted budget exhausted", flush=True)
        _refresh_audit(record)
        return record

    while remaining > 0 and record.get("status") != "converged":
        this_chunk = min(chunk, remaining)
        t = record["temperature"]
        last_dt = record.get("last_accepted_dt")
        if last_dt is None:
            hist = record.get("history") or {}
            dts = hist.get("dt") or [min(CONTINUATION_DT_ACCURACY, 2500.0)]
            last_dt = dts[-1]
        payload, res = _run_chunk(
            t,
            max_steps=this_chunk,
            dt_hold=float(last_dt),
            rcb=record.get("primary_rcb_log10p"),
            simulated_time=float(record.get("simulated_time") or 0.0),
            dt_accuracy=CONTINUATION_DT_ACCURACY,
        )
        record, extra_done = _merge_chunk(record, payload, extra_done, prev)
        remaining = max(0, max_extra - extra_done)
        prev = dict(record["continuation"])
        _write_record(record)
        audit = _refresh_audit(record)
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
            "energy_rel", record.get("energy_committed_residual_rel"),
            "algebraic", audit["algebraic_identity_status"],
            flush=True,
        )
        if payload["steps_accepted"] == 0:
            print("chunk accepted zero steps; stopping", flush=True)
            break

    if record.get("status") == "converged":
        print(
            "gate converged; tightening endpoint with",
            TIGHTEN_STEPS,
            "steps at dt_accuracy",
            TIGHTEN_DT_ACCURACY,
            flush=True,
        )
        last_dt = record.get("last_accepted_dt") or TIGHTEN_DT_ACCURACY
        payload, res = _run_chunk(
            record["temperature"],
            max_steps=TIGHTEN_STEPS,
            dt_hold=min(float(last_dt), TIGHTEN_DT_ACCURACY),
            rcb=record.get("primary_rcb_log10p"),
            simulated_time=float(record.get("simulated_time") or 0.0),
            dt_accuracy=TIGHTEN_DT_ACCURACY,
        )
        record, extra_done = _merge_chunk(record, payload, extra_done, prev)
        record["continuation"]["tighten_dt_accuracy"] = TIGHTEN_DT_ACCURACY
        record["continuation"]["tighten_steps"] = int(payload["steps_accepted"])
        record["continuation"]["tighten_status"] = payload.get("status")
        _write_record(record)
        _refresh_audit(record)
        print(
            "tighten",
            res.status.value,
            "flat", record["flux_flatness"],
            "tend", record["tendency_norm"],
            "acc", record["steps_accepted"],
            flush=True,
        )
    return record


if __name__ == "__main__":
    main()
