"""Orchestrate Stage 1 validation data generation and figure production."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Plot generation is always noninteractive.  On macOS, allowing Matplotlib to
# select MacOSX makes child processes register an AppKit application and abort
# when run headlessly (for example from pytest, CI, or Cursor).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "convection-mlt-matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PLOTS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PLOTS_ROOT.parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTS_ROOT))

from common import (  # noqa: E402
    DATA_DIR,
    GENERATED_DIR,
    ensure_dirs,
    machine_identity,
    read_json,
    write_json,
)


def _plot_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(SRC_ROOT), str(PLOTS_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    return env

PLOT_SCRIPTS = (
    "plot_global_column.py",
    "plot_locality.py",
    "plot_robustness.py",
    "plot_enthalpy.py",
    "plot_equilibrium_invariance.py",
    "plot_alpha_relaxation.py",
    "plot_resolution_scaling.py",
    "plot_temporal_stability.py",
    "plot_validation_table.py",
)

REQUIRED_OUTPUTS = (
    "01_global_column.png",
    "02_localized_barrier.png",
    "02b_robustness_summary.png",
    "robustness_summary.csv",
    "03_enthalpy_conservation.png",
    "enthalpy_telescoping.csv",
    "04_equilibrium_invariance.png",
    "05_alpha_relaxation.png",
    "05b_closure_scaling.png",
    "06_resolution_scaling.png",
    "07a_stability_controller.png",
    "07b_temporal_convergence.png",
    "08_invariant_table.png",
    "validation_table.csv",
)

EXPECTED_UNSTABLE_OUTCOMES = frozenset({"fixed_step_failure", "adaptive_failure"})
UNEXPECTED_FAILURE_OUTCOMES = frozenset(
    {
        "fixed_step_failure",
        "adaptive_failure",
        "max_steps_exhausted",
        "unexpected_success",
    }
)


def _run_generate_data(smoke: bool) -> None:
    cmd = [sys.executable, str(PLOTS_ROOT / "generate_data.py")]
    if smoke:
        cmd.append("--smoke")
    print(f"[make_all] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PLOTS_ROOT, check=True, env=_plot_env())


def _run_plot_script(name: str) -> None:
    path = PLOTS_ROOT / name
    print(f"[make_all] running: {path.name}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load plot script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{name} has no main()")
    module.main()


def _check_outputs() -> list[str]:
    missing = []
    for name in REQUIRED_OUTPUTS:
        path = GENERATED_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(name)
    return missing


def _validate_robustness() -> list[str]:
    failures = []
    path = DATA_DIR / "robustness.json"
    if not path.exists():
        return ["robustness.json missing"]
    data = read_json(path)
    for record in data.get("records", []):
        name = record["name"]
        expected = record.get("expected_status")
        status = record["outcome"]["status"]
        if expected is not None and status != expected:
            failures.append(f"robustness/{name}: status {status!r} != {expected!r}")
        if not record.get("status_ok", True):
            failures.append(f"robustness/{name}: status_ok false")
        for key, value in record.get("metrics_for_score", {}).items():
            if not np.isfinite(float(value)):
                failures.append(f"robustness/{name}: nonfinite metric {key}")
    return failures


def _validate_equilibrium() -> list[str]:
    failures = []
    path = DATA_DIR / "equilibrium_matrix.json"
    if not path.exists():
        return ["equilibrium_matrix.json missing"]
    data = read_json(path)
    for record in data.get("records", []):
        alpha = float(record["alpha"])
        if alpha <= 0.0:
            continue
        n = record["n_layers"]
        status = record["outcome"]["status"]
        if status not in ("converged", "no_active_convection"):
            failures.append(f"equilibrium N={n} α={alpha}: unexpected status {status!r}")
        score = record.get("score", {})
        s = float(score.get("score", float("nan")))
        if not np.isfinite(s):
            failures.append(f"equilibrium N={n} α={alpha}: nonfinite score")
        elif s >= 1.0 and status == "converged" and not record.get("failed"):
            failures.append(f"equilibrium N={n} α={alpha}: score {s:.3g} >= 1")
        for key, value in record.get("metrics_for_score", {}).items():
            if not np.isfinite(float(value)):
                failures.append(f"equilibrium N={n} α={alpha}: nonfinite {key}")
    return failures


def _validate_temporal_safety() -> list[str]:
    failures = []
    path = DATA_DIR / "temporal_stability.json"
    if not path.exists():
        return ["temporal_stability.json missing"]
    data = read_json(path)
    for case in data.get("safety_cases", []):
        expected_unstable = bool(case.get("expected_unstable", False))
        outcome = case["outcome_class"]
        label = (
            f"safety c_diff={case.get('c_diff')} dt={case.get('dt_s')} "
            f"mode={case['mode']}"
        )
        if expected_unstable:
            if outcome not in EXPECTED_UNSTABLE_OUTCOMES:
                failures.append(f"{label}: expected unstable but got {outcome!r}")
            continue
        if outcome in UNEXPECTED_FAILURE_OUTCOMES:
            failures.append(f"{label}: unexpected failure class {outcome!r}")
    return failures


def _validate_invariant_audit() -> tuple[list[str], bool]:
    failures = []
    path = DATA_DIR / "invariant_audit.json"
    if not path.exists():
        return ["invariant_audit.json missing"], False
    data = read_json(path)
    all_pass = bool(data.get("pass", False))
    for row in data.get("rows", []):
        for key in ("expected", "observed", "error", "tolerance"):
            if not np.isfinite(float(row[key])):
                failures.append(f"invariant {row['name']}: nonfinite {key}")
        if not row.get("pass", False):
            all_pass = False
    return failures, all_pass


def _campaign_metrics_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    eq_path = DATA_DIR / "equilibrium_matrix.json"
    if eq_path.exists():
        data = read_json(eq_path)
        scores = []
        for record in data.get("records", []):
            if float(record["alpha"]) <= 0.0:
                continue
            scores.append(
                {
                    "n_layers": record["n_layers"],
                    "alpha": record["alpha"],
                    "status": record["outcome"]["status"],
                    "score": record.get("score", {}).get("score"),
                    "pass": record.get("score", {}).get("pass"),
                }
            )
        summary["equilibrium_scores"] = scores
    alpha_path = DATA_DIR / "alpha_trajectories.json"
    if alpha_path.exists():
        data = read_json(alpha_path)
        closure = data.get("closure_scaling", [])
        if len(closure) >= 2:
            alphas = np.asarray([c["alpha"] for c in closure], dtype=float)
            summary["closure_scaling_slopes"] = {
                "velocity_vs_alpha": _safe_slope(
                    alphas, np.asarray([c["mean_velocity"] for c in closure])
                ),
                "flux_vs_alpha2": _safe_slope(
                    alphas**2, np.asarray([c["mean_flux"] for c in closure])
                ),
                "kzz_vs_alpha2": _safe_slope(
                    alphas**2, np.asarray([c["mean_kzz"] for c in closure])
                ),
            }
    temporal_path = DATA_DIR / "temporal_stability.json"
    if temporal_path.exists():
        data = read_json(temporal_path)
        summary["temporal_fitted_slopes"] = [
            {
                "n_layers": item["n_layers"],
                "alpha": item["alpha"],
                "fitted_slope": item.get("fitted_slope"),
            }
            for item in data.get("order_records", [])
        ]
    scaling_path = DATA_DIR / "resolution_scaling.json"
    if scaling_path.exists():
        data = read_json(scaling_path)
        ns = np.asarray([r["n_layers"] for r in data.get("records", [])], dtype=float)
        times = np.asarray(
            [r["timing"]["median_s"] for r in data.get("records", [])], dtype=float
        )
        summary["resolution_timing_exponent"] = _safe_slope(ns, times)
        summary["resolution_timing_note"] = "provisional fitted wall-time vs N exponent"
    robust_path = DATA_DIR / "robustness.json"
    if robust_path.exists():
        data = read_json(robust_path)
        summary["robustness_pass_fail"] = [
            {
                "name": item["name"],
                "status": item["outcome"]["status"],
                "pass": item.get("pass"),
            }
            for item in data.get("records", [])
        ]
    return summary


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return None
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def _build_manifest(
    smoke: bool,
    missing: list[str],
    validation_failures: list[str],
    invariant_pass: bool,
) -> dict[str, Any]:
    files = []
    for name in sorted(REQUIRED_OUTPUTS):
        path = GENERATED_DIR / name
        files.append(
            {
                "name": name,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    overall_pass = not missing and not validation_failures and invariant_pass
    return {
        "smoke": smoke,
        "machine": machine_identity(),
        "files": files,
        "missing_outputs": missing,
        "validation_failures": validation_failures,
        "invariant_audit_pass": invariant_pass,
        "campaign_metrics": _campaign_metrics_summary(),
        "pass": overall_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="reduced campaigns for CI")
    parser.add_argument("--skip-data", action="store_true", help="skip generate_data.py")
    args = parser.parse_args()

    ensure_dirs()
    if str(PLOTS_ROOT) not in sys.path:
        sys.path.insert(0, str(PLOTS_ROOT))

    if not args.skip_data:
        _run_generate_data(args.smoke)

    for script in PLOT_SCRIPTS:
        _run_plot_script(script)

    missing = _check_outputs()
    validation_failures: list[str] = []
    validation_failures.extend(_validate_robustness())
    validation_failures.extend(_validate_equilibrium())
    validation_failures.extend(_validate_temporal_safety())
    inv_failures, invariant_pass = _validate_invariant_audit()
    validation_failures.extend(inv_failures)

    manifest = _build_manifest(args.smoke, missing, validation_failures, invariant_pass)
    manifest_path = GENERATED_DIR / "evidence_manifest.json"
    write_json(manifest_path, manifest)
    print(f"wrote {manifest_path}")

    if missing:
        raise SystemExit(f"missing or empty outputs: {', '.join(missing)}")
    if validation_failures:
        raise SystemExit(
            "validation failures:\n  " + "\n  ".join(validation_failures)
        )
    if not invariant_pass:
        raise SystemExit("invariant audit did not pass")

    print("[make_all] all figures and validation checks passed")


if __name__ == "__main__":
    main()
