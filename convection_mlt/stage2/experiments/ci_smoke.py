"""CI smoke campaign for Stage 2 constant-g NASA + one inverse-square case."""

from __future__ import annotations

import json
from pathlib import Path

from common import RESULTS_DIR, run_case


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for n in (25, 50):
        cases.append(run_case(n_layers=n, x_he=0.0, gravity_mode="constant"))
    cases.append(
        run_case(
            n_layers=25,
            x_he=0.0,
            gravity_mode="inverse_square",
            planet_radius=1.0e8,
        )
    )
    for case in cases:
        assert case["status"] == "converged", case
        if case["gravity_mode"] == "constant":
            assert case["enthalpy_drift"] <= 1.0e-12, case
            assert case["temperature_rms_vs_isentrope"] <= 1.0e-6, case
        else:
            assert case["max_z_over_rp"] > 0.0, case
            assert case["temperature_rms_vs_isentrope"] <= 1.0e-4, case

    out = RESULTS_DIR / "ci_smoke.json"
    out.write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
