"""Shared deterministic helpers for Stage 1 campaigns."""

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.metadata import dump_json, run_metadata
from convection_mlt.solvers import SolverFailure, solve_adaptive


def power_law_case(
    n_layers: int,
    exponent: float,
    alpha: float,
    p_bottom: float = 1.0e7,
    p_top: float = 1.0e3,
):
    physics = PhysicsConfig(alpha=alpha)
    grid = build_grid(
        log_pressure_edges(p_bottom, p_top, n_layers), physics.gravity
    )
    temperature = 1000.0 * (grid.pressure_centres / 1.0e5) ** exponent
    return physics, grid, temperature


def run_adaptive_case(
    name: str,
    n_layers: int,
    alpha: float,
    exponent: float,
    solver: SolverConfig,
) -> dict:
    physics, grid, temperature = power_law_case(
        n_layers, exponent, alpha
    )
    started = time.perf_counter()
    try:
        result = solve_adaptive(grid, temperature, physics, solver)
    except SolverFailure as error:
        result = error.result
    wall_time = time.perf_counter() - started
    return {
        "identity": run_metadata(
            {"physics": asdict(physics), "solver": asdict(solver)}
        ),
        "case": {
            "name": name,
            "n_layers": n_layers,
            "alpha": alpha,
            "initial_exponent": exponent,
            "p_bottom_pa": float(grid.pressure_edges[0]),
            "p_top_pa": float(grid.pressure_edges[-1]),
        },
        "outcome": {
            "status": result.status.value,
            "reason": result.reason,
            "steps": result.steps,
            "rejections": result.rejections,
            "simulated_time_s": result.simulated_time,
            "wall_time_s": wall_time,
            "final_dt_s": result.final_dt,
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


def write_campaign(path: Path, campaign: str, records: list[dict]) -> None:
    dump_json(path, {"campaign": campaign, "records": records})
