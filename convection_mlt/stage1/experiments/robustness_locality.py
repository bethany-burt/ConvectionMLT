"""Campaign 3: grid, profile, stable-barrier, and locality cases."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import mixing_region_labels
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.metadata import run_metadata
from convection_mlt.solvers import SolverFailure, solve_adaptive
from convection_mlt.thermodynamics import IdealH2

from common import write_campaign


def perturbed_edges(n_layers: int) -> np.ndarray:
    log_edges = np.linspace(np.log(1.0e7), np.log(1.0e3), n_layers + 1)
    phase = np.linspace(0.0, np.pi, n_layers + 1)
    perturbation = 0.12 * np.sin(phase) * abs(np.diff(log_edges).mean())
    values = np.exp(log_edges + perturbation)
    if not np.all(np.diff(values) < 0.0):
        raise AssertionError("constructed irregular grid is not monotonic")
    return values


def run_case(
    name: str,
    grid,
    temperature,
    physics,
    solver,
    region_labels=None,
) -> dict:
    try:
        result = solve_adaptive(
            grid,
            temperature,
            physics,
            solver,
            region_labels=region_labels,
        )
    except SolverFailure as error:
        result = error.result
    return {
        "identity": run_metadata(
            {"physics": asdict(physics), "solver": asdict(solver)}
        ),
        "case": {"name": name, "n_layers": grid.n_layers},
        "outcome": {
            "status": result.status.value,
            "reason": result.reason,
            "steps": result.steps,
            "rejections": result.rejections,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stage1/results/robustness_locality.json"),
    )
    args = parser.parse_args()
    n_layers = 25
    physics = PhysicsConfig(alpha=1.0)
    solver = SolverConfig()
    gas = IdealH2()
    uniform = build_grid(
        log_pressure_edges(1.0e7, 1.0e3, n_layers), physics.gravity
    )
    irregular = build_grid(perturbed_edges(n_layers), physics.gravity)

    cases = []
    for name, grid, exponent in (
        ("uniform_stable", uniform, 0.15),
        ("irregular_exact_adiabat", irregular, gas.nabla_ad),
        ("irregular_superadiabatic", irregular, 0.35),
    ):
        temperature = 1000.0 * (
            grid.pressure_centres / 1.0e5
        ) ** exponent
        cases.append(run_case(name, grid, temperature, physics, solver))

    localized = 1000.0 * (
        uniform.pressure_centres / 1.0e5
    ) ** gas.nabla_ad
    localized = localized.copy()
    localized[10:15] *= np.linspace(1.08, 0.92, 5)
    labels = mixing_region_labels(
        uniform,
        localized,
        gas.nabla_ad,
        solver.c_active * solver.epsilon_gradient,
    )
    cases.append(
        run_case(
            "localized_unstable_region",
            uniform,
            localized,
            physics,
            solver,
            labels,
        )
    )
    write_campaign(args.output, "robustness_locality", cases)


if __name__ == "__main__":
    main()
