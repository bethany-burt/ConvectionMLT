"""Production Stage 2 campaign matrix (local / scheduled CI)."""

from __future__ import annotations

import json
from pathlib import Path

from common import RESULTS_DIR, run_case

# Irregular N=25 burned 2e5 steps without converging; production uses 10× that.
PRODUCTION_MAX_STEPS = 5_000_000
CHECKPOINT_PATH = RESULTS_DIR / "production_campaign.json"


def _classify_failures(cases: list[dict]) -> list[dict]:
    failures = []
    for case in cases:
        if case["status"] != "converged":
            failures.append(case)
            continue
        if case["gravity_mode"] == "constant":
            if case["enthalpy_drift"] > 1.0e-12:
                failures.append(case)
            if case["temperature_rms_vs_isentrope"] > 1.0e-6:
                failures.append(case)
        else:
            if case["max_z_over_rp"] <= 0.0:
                failures.append(case)
            if case["temperature_rms_vs_isentrope"] > 1.0e-6:
                failures.append(case)
    return failures


def _write_checkpoint(
    path: Path,
    *,
    cases: list[dict],
    specs_total: int,
    completed_index: int,
    failures: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "in_progress" if completed_index < specs_total else "complete",
        "completed": completed_index,
        "total": specs_total,
        "max_steps": PRODUCTION_MAX_STEPS,
        "cases": cases,
        "failures": failures,
        "n_failures": len(failures),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(
        f"  checkpoint {path.name}: {completed_index}/{specs_total} "
        f"({len(failures)} gate failures so far)",
        flush=True,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs: list[dict] = []
    resolutions = (25, 50, 100, 200)
    x_he_values = (0.0, 0.10, 0.25)
    for n in resolutions:
        for x_he in x_he_values:
            for irregular in (False, True):
                specs.append(
                    dict(
                        n_layers=n,
                        x_he=x_he,
                        gravity_mode="constant",
                        irregular=irregular,
                        max_steps=PRODUCTION_MAX_STEPS,
                        campaign_role="parameter_matrix",
                    )
                )
    for rp in (1.0e7, 1.0e8):
        specs.append(
            dict(
                n_layers=50,
                x_he=0.0,
                gravity_mode="inverse_square",
                planet_radius=rp,
                max_steps=PRODUCTION_MAX_STEPS,
                campaign_role="gravity_stress",
            )
        )
    # Secondary pressure range (production only): domain coverage, not a repeat.
    specs.append(
        dict(
            n_layers=50,
            x_he=0.0,
            gravity_mode="constant",
            p_bottom=1.0e6,
            p_top=1.0e2,
            max_steps=PRODUCTION_MAX_STEPS,
            campaign_role="pressure_range_check",
        )
    )

    cases: list[dict] = []
    total = len(specs)
    print(
        f"production campaign: {total} cases, max_steps={PRODUCTION_MAX_STEPS}",
        flush=True,
    )
    for index, spec in enumerate(specs, start=1):
        label = (
            f"[{index}/{total}] N={spec['n_layers']} x_He={spec['x_he']} "
            f"g={spec['gravity_mode']} irregular={spec.get('irregular', False)}"
        )
        print(label, flush=True)
        spec = dict(spec)
        spec["case_id"] = index
        case = run_case(**spec)
        cases.append(case)
        reason = case.get("reason")
        extra = f" reason={reason}" if reason and case["status"] != "converged" else ""
        print(
            f"  -> status={case['status']} steps={case['steps']} "
            f"drift={case['enthalpy_drift']:.3e} "
            f"T_rms={case['temperature_rms_vs_isentrope']:.3e}{extra}",
            flush=True,
        )
        failures = _classify_failures(cases)
        _write_checkpoint(
            CHECKPOINT_PATH,
            cases=cases,
            specs_total=total,
            completed_index=index,
            failures=failures,
        )

    failures = _classify_failures(cases)
    _write_checkpoint(
        CHECKPOINT_PATH,
        cases=cases,
        specs_total=total,
        completed_index=total,
        failures=failures,
    )
    print(
        f"wrote {CHECKPOINT_PATH} complete with {len(failures)} failures",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
