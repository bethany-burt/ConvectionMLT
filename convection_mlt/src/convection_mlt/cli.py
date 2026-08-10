"""Small deterministic commands for R0 reproducibility."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from .config import PhysicsConfig, SolverConfig
from .grid import build_grid, log_pressure_edges
from .metadata import json_safe, run_metadata
from .solvers import solve_adaptive
from .thermodynamics import IdealH2


def baseline() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic R0 baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    physics = PhysicsConfig()
    solver = SolverConfig()
    gas = IdealH2()
    grid = build_grid(
        log_pressure_edges(1.0e7, 1.0e3, 8), physics.gravity
    )
    initial_temperature = 1000.0 * (
        grid.pressure_centres / 1.0e5
    ) ** gas.nabla_ad
    result = solve_adaptive(grid, initial_temperature, physics, solver, gas)
    payload = {
        "metadata": run_metadata(
            {"physics": asdict(physics), "solver": asdict(solver)}
        ),
        "case": {
            "name": "stage0_exact_adiabat",
            "n_layers": grid.n_layers,
            "p_bottom_pa": grid.pressure_edges[0],
            "p_top_pa": grid.pressure_edges[-1],
        },
        "outcome": {
            "status": result.status.value,
            "reason": result.reason,
            "steps": result.steps,
            "rejections": result.rejections,
            "simulated_time_s": result.simulated_time,
            "metrics": result.metrics,
            "region_labels": result.region_labels,
            "cumulative_unmerged_transfer_j_m2": (
                result.cumulative_unmerged_transfer
            ),
            "max_unmerged_transfer_fraction": (
                result.max_unmerged_transfer_fraction
            ),
        },
    }
    text = json.dumps(
        json_safe(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
