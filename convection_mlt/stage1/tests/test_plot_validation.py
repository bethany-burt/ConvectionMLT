"""Schema and contract tests for Stage 1 validation plot data."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from convection_mlt.config import SolverConfig

PLOTS = Path(__file__).resolve().parents[1] / "plots"
DATA = PLOTS / "data"
GENERATED = PLOTS / "generated"
SRC = Path(__file__).resolve().parents[2] / "src"


def _env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), str(PLOTS), env.get("PYTHONPATH", "")]
    )
    env["MPLBACKEND"] = "Agg"
    mpl_cache = Path(tempfile.gettempdir()) / "convection-mlt-test-matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cache)
    return env


@pytest.fixture(scope="module")
def smoke_bundle():
    """Generate reduced data once without replacing production evidence."""
    backup_root = Path(tempfile.mkdtemp(prefix="convection-mlt-evidence-"))
    data_backup = backup_root / "data"
    generated_backup = backup_root / "generated"
    shutil.copytree(DATA, data_backup)
    shutil.copytree(GENERATED, generated_backup)
    try:
        subprocess.run(
            [sys.executable, str(PLOTS / "generate_data.py"), "--smoke"],
            cwd=PLOTS,
            check=True,
            env=_env(),
        )
        yield DATA
    finally:
        shutil.rmtree(DATA)
        shutil.rmtree(GENERATED)
        shutil.copytree(data_backup, DATA)
        shutil.copytree(generated_backup, GENERATED)
        shutil.rmtree(backup_root)


def test_common_exact_zero_display_never_invents_positive():
    sys.path.insert(0, str(PLOTS))
    from common import exact_zero_display

    value, flagged = exact_zero_display(0.0, 1.0e-12)
    assert flagged is True
    assert value == 1.0e-12
    with pytest.raises(ValueError):
        exact_zero_display(-1.0, 1.0e-12)
    with pytest.raises(ValueError):
        exact_zero_display(float("nan"), 1.0e-12)


def test_acceptance_tolerances_come_from_solver_metadata():
    sys.path.insert(0, str(PLOTS))
    from common import acceptance_tolerances

    solver = SolverConfig(theta_rms_tolerance=3.0e-7, flux_tolerance=0.01)
    tol = acceptance_tolerances(solver)
    assert tol["potential_temperature_rms"] == 3.0e-7
    assert tol["convective_flux_max"] == 0.01
    assert "enthalpy_drift" in tol


def test_score_rejects_missing_and_nonfinite_metrics():
    sys.path.insert(0, str(PLOTS))
    from common import format_score_ratio, score_against_tolerances

    with pytest.raises(KeyError):
        score_against_tolerances({}, {"temperature_rms": 1.0e-8})
    with pytest.raises(ValueError):
        score_against_tolerances(
            {"temperature_rms": float("nan")},
            {"temperature_rms": 1.0e-8},
        )
    scored = score_against_tolerances(
        {"temperature_rms": 9.97e-9},
        {"temperature_rms": 1.0e-8},
    )
    assert scored["controlling_metric"] == "temperature_rms"
    assert format_score_ratio(0.9978295800738834, decimals=3) == "0.998"
    assert format_score_ratio(0.9999996, decimals=3) == "<1.000"


def test_pressure_axis_is_logarithmic_and_increases_downward():
    sys.path.insert(0, str(PLOTS))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import pressure_axis

    fig, ax = plt.subplots()
    ax.set_ylim(1.0e3, 1.0e7)
    pressure_axis(ax)
    assert ax.get_yscale() == "log"
    lower, upper = ax.get_ylim()
    assert lower > upper
    plt.close(fig)


def test_smoke_campaign_schemas(smoke_bundle):
    sys.path.insert(0, str(PLOTS))
    from common import read_json

    global_profile = read_json(smoke_bundle / "global_profile.json")
    assert global_profile["profiles"]
    assert global_profile["profiles"][0]["accepted_step"] == 0
    assert len(global_profile["reference_temperature_k"]) == global_profile["case"][
        "n_layers"
    ]

    locality = read_json(smoke_bundle / "locality.json")
    assert "global_reference" not in locality
    assert locality["initial_piecewise_reference_k"]
    assert locality["outcome"]["cumulative_unmerged_transfer_j_m2"]
    assert "initial_piecewise_reference_potential_temperature_k" in locality
    assert "normalized_unmerged_transfer" in locality
    assert "pressure_edges_pa" in locality["case"]

    matrix = read_json(smoke_bundle / "equilibrium_matrix.json")
    for record in matrix["records"]:
        assert float(record["alpha"]) > 0.0
        assert "tolerances" in record
        assert "score" in record
        assert "max_abs_enthalpy_drift" in record
        # Score uses own-run tolerances, not hard-coded plot values.
        assert record["tolerances"]["potential_temperature_rms"] > 0.0

    enthalpy = read_json(smoke_bundle / "enthalpy.json")
    for record in enthalpy["records"]:
        assert record["history"]
        assert "signed_enthalpy_drift" in record["history"][0]
        audit = record["conservation_audit"]
        assert audit["bottom_boundary_flux_w_m2"] == 0.0
        assert audit["top_boundary_flux_w_m2"] == 0.0
        assert np.isfinite(audit["telescoping_residual_w_m2"])

    alpha = read_json(smoke_bundle / "alpha_trajectories.json")
    assert alpha["closure_scaling"]
    assert all(item["alpha"] > 0.0 for item in alpha["trajectories"])

    temporal = read_json(smoke_bundle / "temporal_stability.json")
    for order in temporal["order_records"]:
        fit_points = [p for p in order["points"] if p.get("used_in_fit")]
        assert fit_points
        assert all(p["status"] == "completed" for p in fit_points)
        assert all(np.isfinite(p["relative_temperature_rms"]) for p in fit_points)
    for case in temporal["safety_cases"]:
        assert "outcome_class" in case
        assert "expected_unstable" in case
        assert "min_accepted_trial_delta_over_epsilon" in case or case[
            "outcome_class"
        ] in {"adaptive_failure", "fixed_step_failure", "max_steps_exhausted"}
        if case["expected_unstable"]:
            assert case["outcome_class"] in {
                "fixed_step_failure",
                "adaptive_failure",
            }

    robustness = read_json(smoke_bundle / "robustness.json")
    for record in robustness["records"]:
        assert record["status_ok"] is True
        assert all(math.isfinite(v) for v in record["metrics_for_score"].values())
        assert "applicable_metrics" in record
        if record["outcome"]["status"] == "no_active_convection":
            assert "temperature_rms" not in record["applicable_metrics"]

    audit = read_json(smoke_bundle / "invariant_audit.json")
    names = {row["name"] for row in audit["rows"]}
    required = {
        "stable_profile_zero_flux",
        "exact_adiabat_zero_flux",
        "manufactured_mixing_length",
        "manufactured_velocity",
        "manufactured_flux",
        "manufactured_kzz",
        "manufactured_positive_flux",
        "manufactured_positive_velocity",
        "manufactured_positive_mixing_length",
        "manufactured_positive_kzz",
        "update_sign_lower_cools",
        "boundary_flux_bottom_zero",
        "one_step_telescoping_residual",
        "alpha_zero_status_no_active",
        "rejected_state_purity",
        "regional_reference_enthalpy_identity",
        "status_stable_no_active",
        "status_superadiabatic_converged",
    }
    assert required.issubset(names)
    assert audit["pass"] is True
    assert all(row["pass"] for row in audit["rows"])


def test_smoke_make_all_writes_nonempty_outputs(smoke_bundle):
    del smoke_bundle
    subprocess.run(
        [sys.executable, str(PLOTS / "make_all.py"), "--smoke", "--skip-data"],
        cwd=PLOTS,
        check=True,
        env=_env(),
    )
    required = [
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
        "evidence_manifest.json",
    ]
    for name in required:
        path = GENERATED / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    with (GENERATED / "validation_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row.get("pass", "").lower() in {"true", "1", "yes"} for row in rows)
