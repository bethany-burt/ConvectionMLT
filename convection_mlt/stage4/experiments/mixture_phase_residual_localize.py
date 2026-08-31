"""Phase-resolved flux residual localization for H2/He mixture cases (N=96).

Runs two representative cases requested for advisor diagnosis:
  - x_he=0.2, f_irr=0   (helium isolation)
  - x_he=0.2, f_irr=500 (helium + irradiation)

Writes JSON with r_i = (F_total,i - F_int)/F_scale vs pressure for each solver phase.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mixture_residual_utils import run_instrumented_production

OUT_DIR = ROOT.parent / "examples" / "rce" / "runs" / "mixture_diagnostics"
CASES = (
    {"id": "he_only", "x_he": 0.2, "f_irr": 0.0},
    {"id": "he_irr500", "x_he": 0.2, "f_irr": 500.0},
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        t0 = time.perf_counter()
        print(f"=== {case['id']}: x_he={case['x_he']}, f_irr={case['f_irr']} ===", flush=True)
        payload = run_instrumented_production(
            x_he=float(case["x_he"]),
            f_irr=float(case["f_irr"]),
        )
        payload["id"] = case["id"]
        payload["wall_s"] = time.perf_counter() - t0
        out = OUT_DIR / f"residual_localize_{case['id']}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "verdict": payload["verdict"],
                    "wall_s": payload["wall_s"],
                    "out": str(out),
                    "phase_classifications": [
                        {"phase": p["phase"], "class": p.get("classification")}
                        for p in payload["phases"]
                        if "classification" in p
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        results.append(payload)

    summary = OUT_DIR / "residual_localize_summary.json"
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
