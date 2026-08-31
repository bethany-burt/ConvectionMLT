"""Long continuation from x_he=0.2, f_irr=0 checkpoint to test budget vs asymptote."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convection_mlt.production_rce import (
    ProductionControls,
    _gates_from_result,
    _live_solve,
    build_spec,
    production_solver_config,
    production_thermo,
    run_production_rce,
)

OUT_DIR = ROOT.parent / "examples" / "rce" / "runs" / "mixture_diagnostics"
CHECKPOINT = OUT_DIR / "residual_localize_he_only.json"
EXTRA_STEPS = 800  # 200 default + 800 = 1000 total accepted target


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_he = 0.2
    f_irr = 0.0

    if CHECKPOINT.exists():
        ckpt = json.loads(CHECKPOINT.read_text())
        t_init = np.asarray(ckpt["checkpoint_temperature"], dtype=np.float64)
        print(f"Restarting from checkpoint {CHECKPOINT}", flush=True)
    else:
        print("No checkpoint found; running default production first...", flush=True)
        run = run_production_rce(
            n_layers=96,
            alpha=1.0,
            f_int=300.0,
            f_irr=f_irr,
            x_he=x_he,
            controls=ProductionControls(max_recovery_cycles=2),
        )
        t_init = np.asarray(run.result.final_state.temperature, dtype=np.float64)
        ckpt = {"checkpoint_temperature": t_init.tolist()}
        CHECKPOINT.write_text(json.dumps(ckpt, indent=2) + "\n")

    spec = build_spec(n_layers=96, alpha=1.0, f_int=300.0, f_irr=f_irr)
    grid = spec.grid()
    thermo = production_thermo(x_he)
    solver = production_solver_config()
    ctrl = ProductionControls(max_recovery_cycles=0)

    t0 = time.perf_counter()
    res, _cfg = _live_solve(
        grid=grid,
        t0=t_init,
        spec=spec,
        solver=solver,
        thermo=thermo,
        max_steps=EXTRA_STEPS,
        dt_accuracy=ctrl.continuation_dt_accuracy_s,
        dt_hold_init=min(ctrl.dt_hold_init_s, ctrl.continuation_dt_accuracy_s),
        previous_rcb=None,
        gate=ctrl.gate,
        prescribed_dt=None,
    )
    wall_s = time.perf_counter() - t0
    gates = _gates_from_result(res, spec, gate=ctrl.gate, require_bottom_connected_cz=True)

    accepted = [d for d in res.diagnostics if d.accepted]
    steps = np.arange(1, len(accepted) + 1)
    flatness = np.array([float(d.flux_flatness) for d in accepted], dtype=np.float64)

    payload = {
        "x_he": x_he,
        "f_irr": f_irr,
        "extra_steps_requested": EXTRA_STEPS,
        "extra_steps_accepted": len(accepted),
        "wall_s": wall_s,
        "verdict_after_continuation": "CONVERGED" if gates.convergence_ok else "NOT CONVERGED",
        "final_flux_flatness": float(gates.flux_flatness),
        "flatness_history": flatness.tolist(),
        "step_numbers": steps.tolist(),
    }

    # Simple trend classification on the tail.
    if flatness.size >= 20:
        tail = flatness[-20:]
        slope = float((tail[0] - tail[-1]) / max(len(tail) - 1, 1))
        payload["tail_slope_per_step"] = slope
        payload["tail_mean"] = float(np.mean(tail))
        payload["tail_std"] = float(np.std(tail))
        if slope > 1.0e-6 and tail[-1] < tail[0]:
            payload["budget_interpretation"] = "continued_reduction_suggests_budget_too_short"
        elif float(np.std(tail)) / max(float(np.mean(tail)), 1.0e-12) < 0.05:
            payload["budget_interpretation"] = "asymptote_plateau_additional_steps_unlikely_to_help"
        else:
            payload["budget_interpretation"] = "oscillatory_or_transitional"

    out_json = OUT_DIR / "long_continuation_he_only.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, flatness, "b.-", lw=1.2)
    ax.axhline(1.0e-3, color="k", ls="--", lw=0.8, label="gate 1e-3")
    ax.set_xlabel("Accepted step (continuation segment)")
    ax.set_ylabel("Flux flatness")
    ax.set_title(f"x_he=0.2, f_irr=0 — {len(accepted)} extra steps")
    ax.legend()
    fig.tight_layout()
    fig_path = OUT_DIR / "flatness_vs_step_he_only.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(json.dumps({"out_json": str(out_json), "fig": str(fig_path), **payload}, indent=2))


if __name__ == "__main__":
    main()
