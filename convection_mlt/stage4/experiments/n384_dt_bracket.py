"""Short N=384 timestep/acceleration bracket from a cloned checkpoint.

Does not write n384_implicit_rce.json. Identifies which controller is
holding Δt near 1.7e4 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
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
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)

RESULTS = ROOT / "results"
SOURCE = RESULTS / "n384_implicit_rce.json"
OUT = RESULTS / "n384_dt_bracket.json"
STEPS = 50


def _run(record: dict, *, name: str, steps: int, solver, cfg) -> dict:
    spec = nested_analytic_opacity_spec(384)
    grid = spec.grid()
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        record["temperature"],
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
    payload = serialize_rce_result(
        res,
        spec,
        pressure_centres=grid.pressure_centres,
        pressure_edges=grid.pressure_edges,
        solver=solver,
        rce_config=cfg,
        extra={"wall_time_s": wall, "bracket_arm": name},
    )
    flat0 = float(record["flux_flatness"])
    flat1 = float(payload["flux_flatness"])
    dflat = flat0 - flat1
    accepted = int(payload.get("steps_accepted") or 0)
    rejected = int(payload.get("rejections") or 0)
    return {
        "name": name,
        "status": payload.get("status"),
        "steps_accepted": accepted,
        "rejections": rejected,
        "rejection_fraction": rejected / max(accepted + rejected, 1),
        "wall_time_s": wall,
        "dt_start": record.get("last_accepted_dt"),
        "dt_end": payload.get("last_accepted_dt"),
        "median_accepted_dt": payload.get("median_accepted_dt"),
        "flatness_start": flat0,
        "flatness_end": flat1,
        "delta_flatness": dflat,
        "flatness_per_wall_s": dflat / wall if wall > 0.0 else None,
        "tendency_start": record.get("tendency_norm"),
        "tendency_end": payload.get("tendency_norm"),
        "rcb_start": record.get("primary_rcb_log10p"),
        "rcb_end": payload.get("primary_rcb_log10p"),
        "abs_drcb": abs(float(payload["primary_rcb_log10p"]) - float(record["primary_rcb_log10p"])),
        "energy_gate_ratio": payload.get("energy_gate_ratio"),
        "energy_ok": payload.get("energy_gate_ratio") is not None
        and float(payload["energy_gate_ratio"]) <= 1.0,
        "coupled_defect": payload.get("coupled_defect"),
    }


def _arms(record: dict, steps: int):
    last_dt = float(record.get("last_accepted_dt") or 17000.0)
    rcb = record.get("primary_rcb_log10p")
    t_sim = float(record.get("simulated_time") or 0.0)
    solver = production_solver_config()

    def cfg(**kwargs):
        base = production_rce_config(
            max_steps=steps,
            dt_accuracy=kwargs.pop("dt_accuracy", 50000.0),
            dt_hold_init=kwargs.pop("dt_hold_init", last_dt),
            previous_rcb_init=rcb,
            simulated_time_init=t_sim,
            gate=PHYSICAL_GATE,
        )
        return replace(base, **kwargs) if kwargs else base

    return [
        (
            "current_controller",
            solver,
            cfg(dt_accuracy=50000.0, dt_hold_init=last_dt),
        ),
        (
            "release_hold_to_50ks",
            solver,
            cfg(dt_accuracy=50000.0, dt_hold_init=50000.0),
        ),
        (
            "larger_implicit_100ks",
            solver,
            cfg(dt_accuracy=100000.0, dt_hold_init=100000.0),
        ),
        (
            "permissive_epsilon_1e-2",
            replace(solver, epsilon_temperature=1.0e-2),
            replace(
                cfg(dt_accuracy=50000.0, dt_hold_init=last_dt),
                use_coupled_tendency_dt=False,
            ),
        ),
        (
            "prescribed_25000",
            solver,
            replace(cfg(dt_accuracy=25000.0, dt_hold_init=25000.0), prescribed_dt=25000.0),
        ),
        (
            "prescribed_50000",
            solver,
            replace(cfg(dt_accuracy=50000.0, dt_hold_init=50000.0), prescribed_dt=50000.0),
        ),
        (
            "prescribed_100000",
            solver,
            replace(cfg(dt_accuracy=100000.0, dt_hold_init=100000.0), prescribed_dt=100000.0),
        ),
    ]


def main(steps: int = STEPS, source: Path = SOURCE) -> dict:
    record = json.loads(source.read_text())
    print(
        "n384_dt_bracket source",
        "steps", record.get("steps_accepted"),
        "flat", record.get("flux_flatness"),
        "dt", record.get("last_accepted_dt"),
        flush=True,
    )
    arms = []
    for name, solver, cfg in _arms(record, steps):
        print("arm start", name, flush=True)
        row = _run(record, name=name, steps=steps, solver=solver, cfg=cfg)
        arms.append(row)
        print(
            "arm",
            name,
            "acc", row["steps_accepted"],
            "rej", row["rejections"],
            "dflat", row["delta_flatness"],
            "per_s", row["flatness_per_wall_s"],
            "dt_end", row["dt_end"],
            "energy_ok", row["energy_ok"],
            "drcb", row["abs_drcb"],
            "wall", round(row["wall_time_s"], 1),
            flush=True,
        )
    payload = {
        "source": str(source),
        "source_steps_accepted": record.get("steps_accepted"),
        "source_flux_flatness": record.get("flux_flatness"),
        "source_last_accepted_dt": record.get("last_accepted_dt"),
        "bracket_accepted_steps": steps,
        "note": (
            "Cloned-checkpoint probe only. Does not continue the production N=384 job. "
            "current_controller keeps the sticky dt_hold. release_hold tests whether "
            "that hold, not dt_accuracy, is the 1.7e4 s cap."
        ),
        "arms": arms,
        "ranking": rank_arms(arms, source_flatness=float(record["flux_flatness"])),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    print(json.dumps(payload["ranking"], indent=2), flush=True)
    return payload


def rank_arms(arms: list[dict], *, source_flatness: float) -> dict:
    """Rank cloned arms by −Δ(flatness)/wall time under safety constraints."""
    ranked = []
    for arm in arms:
        defect = arm.get("coupled_defect")
        tend0 = arm.get("tendency_start")
        tend1 = arm.get("tendency_end")
        energy_ok = bool(arm.get("energy_ok"))
        defect_ok = defect is not None and float(defect) <= 1.0e-10
        rcb_ok = float(arm.get("abs_drcb") or 0.0) < 0.01
        accepted = int(arm.get("steps_accepted") or 0)
        rej_frac = float(arm.get("rejection_fraction") or 1.0)
        tend_ok = (
            tend0 is not None and tend1 is not None and float(tend1) <= 1.01 * float(tend0)
        )
        dflat = float(arm.get("delta_flatness") or 0.0)
        wall = float(arm.get("wall_time_s") or 0.0)
        safe = energy_ok and defect_ok and rcb_ok and accepted > 0 and tend_ok and rej_frac < 0.5
        ranked.append({
            "name": arm["name"],
            "safe": safe,
            "flatness_per_wall_s": arm.get("flatness_per_wall_s"),
            "delta_flatness": dflat,
            "steps_accepted": accepted,
            "rejection_fraction": rej_frac,
            "energy_gate_ratio": arm.get("energy_gate_ratio"),
            "coupled_defect": defect,
            "abs_drcb": arm.get("abs_drcb"),
            "median_accepted_dt": arm.get("median_accepted_dt"),
            "status": arm.get("status"),
            "safety_flags": {
                "energy_ok": energy_ok,
                "coupled_defect_le_1e-10": defect_ok,
                "no_material_rcb_jump": rcb_ok,
                "accepted_steps": accepted > 0,
                "tendency_declining": tend_ok,
                "rejection_fraction_lt_0.5": rej_frac < 0.5,
            },
        })
    safe = [r for r in ranked if r["safe"] and r["flatness_per_wall_s"] is not None]
    safe.sort(key=lambda r: -float(r["flatness_per_wall_s"]))
    best = safe[0] if safe else None
    rates = [float(r["flatness_per_wall_s"]) for r in safe]
    same_tail = (
        len(rates) >= 2
        and (max(rates) - min(rates)) / max(abs(max(rates)), 1e-30) < 0.5
    )
    steps_to_gate = None
    if best is not None and best["delta_flatness"] > 0.0:
        remain = max(source_flatness - 1.0e-3, 0.0)
        steps_to_gate = remain / max(best["delta_flatness"] / max(best["steps_accepted"], 1), 1e-30)
    return {
        "safe_ranked_by_minus_dflat_per_wall": safe,
        "best_safe_arm": None if best is None else best["name"],
        "same_linear_tail": same_tail,
        "estimated_accepted_steps_to_1e-3_at_best_rate": steps_to_gate,
        "recommendation": (
            "continue_best_safe_arm"
            if best is not None and not same_tail and steps_to_gate is not None and steps_to_gate < 2000
            else "stop_pseudotime_implement_steady_defect_solve"
        ),
        "all_arms": ranked,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Re-score an existing n384_dt_bracket.json without rerunning HELIOS-free clones.",
    )
    args = parser.parse_args()
    if args.rank_only:
        existing = json.loads(OUT.read_text())
        existing["ranking"] = rank_arms(
            existing["arms"],
            source_flatness=float(existing["source_flux_flatness"]),
        )
        OUT.write_text(json.dumps(existing, indent=2) + "\n")
        print(json.dumps(existing["ranking"], indent=2))
    else:
        main(steps=args.steps, source=args.source)
