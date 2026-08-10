"""Campaign 2: fixed-step forward-Euler convergence at common finite time."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from convection_mlt.config import SolverConfig
from convection_mlt.metadata import dump_json
from convection_mlt.solvers import fixed_step

from common import power_law_case


def integrate_fixed(dt: float, final_time: float):
    physics, grid, temperature = power_law_case(25, 0.35, 1.0)
    settings = SolverConfig(epsilon_temperature=0.5)
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time):
        raise ValueError("final_time must be an integer multiple of dt")
    state = temperature.copy()
    for step in range(steps):
        outcome = fixed_step(grid, state, dt, physics, settings)
        if not outcome.accepted:
            return grid, state, {
                "status": "failed",
                "reason": outcome.reason,
                "failed_step": step,
            }
        state = outcome.temperature
    return grid, state, {"status": "completed", "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dt", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=16.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stage1/results/temporal_convergence.json"),
    )
    args = parser.parse_args()
    reference_dt = args.base_dt / 32.0
    grid, reference, reference_info = integrate_fixed(
        reference_dt, args.final_time
    )
    records = []
    for divisor in (1, 2, 4, 8):
        dt = args.base_dt / divisor
        _, state, info = integrate_fixed(dt, args.final_time)
        relative = (state - reference) / reference
        rms = np.sqrt(
            np.sum(grid.layer_mass * relative**2)
            / np.sum(grid.layer_mass)
        )
        records.append(
            {
                "dt_s": dt,
                "relative_temperature_rms": rms,
                **info,
            }
        )
    observed_orders = [
        float(
            np.log(
                records[index]["relative_temperature_rms"]
                / records[index + 1]["relative_temperature_rms"]
            )
            / np.log(2.0)
        )
        for index in range(len(records) - 1)
    ]
    dump_json(
        args.output,
        {
            "campaign": "temporal_convergence",
            "configuration": {
                "n_layers": 25,
                "alpha": 1.0,
                "initial_exponent": 0.35,
                "final_time_s": args.final_time,
                "reference_dt_s": reference_dt,
                "solver": asdict(SolverConfig(epsilon_temperature=0.5)),
            },
            "reference": reference_info,
            "records": records,
            "observed_orders": observed_orders,
        },
    )


if __name__ == "__main__":
    main()
