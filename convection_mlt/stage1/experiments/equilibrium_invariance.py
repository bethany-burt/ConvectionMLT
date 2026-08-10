"""Campaign 1: adaptive N-by-alpha equilibrium invariance matrix."""

import argparse
from pathlib import Path

from convection_mlt.config import SolverConfig

from common import run_adaptive_case, write_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stage1/results/equilibrium_invariance.json"),
    )
    args = parser.parse_args()
    resolutions = [25, 50] if args.quick else [25, 50, 100, 200, 400]
    alphas = [0.5, 1.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 4.0]
    solver = SolverConfig()
    records = [
        run_adaptive_case(
            "globally_superadiabatic",
            n_layers,
            alpha,
            0.35,
            solver,
        )
        for n_layers in resolutions
        for alpha in alphas
    ]
    write_campaign(args.output, "equilibrium_invariance", records)


if __name__ == "__main__":
    main()
