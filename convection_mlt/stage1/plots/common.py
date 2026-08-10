"""Shared helpers for Stage 1 validation data and Matplotlib figures."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import enthalpy_normalized_adiabat, potential_temperature
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.metadata import dump_json, git_commit, git_dirty, json_safe, run_metadata
from convection_mlt.thermodynamics import IdealH2

PLOTS_ROOT = Path(__file__).resolve().parent
DATA_DIR = PLOTS_ROOT / "data"
GENERATED_DIR = PLOTS_ROOT / "generated"
PACKAGE_ROOT = PLOTS_ROOT.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

RESOLUTIONS = (25, 50, 100, 200, 400)
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
PRODUCTION_EXPONENT = 0.35
DEFAULT_P_BOTTOM = 1.0e7
DEFAULT_P_TOP = 1.0e3


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def machine_identity() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": git_commit(REPO_ROOT),
        "git_dirty": git_dirty(REPO_ROOT),
    }


def campaign_identity(physics: PhysicsConfig, solver: SolverConfig) -> dict[str, Any]:
    payload = run_metadata({"physics": asdict(physics), "solver": asdict(solver)}, REPO_ROOT)
    payload["machine"] = machine_identity()
    return payload


def power_law_temperature(grid, exponent: float, t_ref: float = 1000.0) -> np.ndarray:
    return t_ref * (grid.pressure_centres / 1.0e5) ** exponent


def standard_grid(
    n_layers: int,
    gravity: float = 15.0,
    p_bottom: float = DEFAULT_P_BOTTOM,
    p_top: float = DEFAULT_P_TOP,
):
    return build_grid(log_pressure_edges(p_bottom, p_top, n_layers), gravity)


def perturbed_edges(
    n_layers: int,
    p_bottom: float = DEFAULT_P_BOTTOM,
    p_top: float = DEFAULT_P_TOP,
    amplitude: float = 0.12,
) -> np.ndarray:
    log_edges = np.linspace(np.log(p_bottom), np.log(p_top), n_layers + 1)
    phase = np.linspace(0.0, np.pi, n_layers + 1)
    perturbation = amplitude * np.sin(phase) * abs(np.diff(log_edges).mean())
    values = np.exp(log_edges + perturbation)
    if not np.all(np.diff(values) < 0.0):
        raise AssertionError("constructed irregular grid is not monotonic")
    return values


def reference_temperature(grid, temperature, gas: IdealH2 | None = None) -> np.ndarray:
    thermo = gas or IdealH2()
    return enthalpy_normalized_adiabat(
        grid, temperature, thermo.cp, thermo.nabla_ad
    )


def potential_temperature_profile(
    grid, temperature, gas: IdealH2 | None = None, p0: float | None = None
) -> np.ndarray:
    thermo = gas or IdealH2()
    reference = float(grid.pressure_centres[0] if p0 is None else p0)
    return potential_temperature(
        grid.pressure_centres, temperature, thermo.nabla_ad, reference
    )


def piecewise_potential_temperature(
    grid,
    temperature,
    region_labels,
    gas: IdealH2 | None = None,
) -> np.ndarray:
    """Potential temperature using each region's first-layer pressure as p0.

    For an enthalpy-normalized adiabat within a region this is constant in
    pressure, so a piecewise θ reference appears as vertical segments.
    """
    thermo = gas or IdealH2()
    labels = np.asarray(region_labels)
    temperature = np.asarray(temperature, dtype=float)
    theta = np.empty(grid.n_layers, dtype=float)
    for label in np.unique(labels):
        region = labels == label
        p0 = float(grid.pressure_centres[np.flatnonzero(region)[0]])
        theta[region] = potential_temperature(
            grid.pressure_centres[region],
            temperature[region],
            thermo.nabla_ad,
            p0,
        )
    return theta


def write_json(path: Path, payload: Any) -> None:
    dump_json(path, payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_finite(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def exact_zero_display(value: float, floor: float) -> tuple[float, bool]:
    """Map exact zeros to a labelled floor for log axes; never invent positives."""
    if value == 0.0:
        return floor, True
    if not np.isfinite(value):
        raise ValueError(f"nonfinite metric cannot be plotted: {value!r}")
    if value < 0.0:
        raise ValueError(f"log-axis metric must be nonnegative: {value!r}")
    return value, False


def pressure_axis(ax) -> None:
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("P [Pa]")


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(fig, name: str) -> Path:
    ensure_dirs()
    path = GENERATED_DIR / name
    fig.savefig(path)
    return path


def repeated_timing(callable_fn, repetitions: int = 5, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        callable_fn()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        callable_fn()
        samples.append(time.perf_counter() - started)
    array = np.asarray(samples, dtype=float)
    return {
        "samples_s": array.tolist(),
        "median_s": float(np.median(array)),
        "q25_s": float(np.percentile(array, 25)),
        "q75_s": float(np.percentile(array, 75)),
        "min_s": float(np.min(array)),
        "max_s": float(np.max(array)),
    }


def fit_log_log_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n_points": 0}
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": int(np.count_nonzero(mask)),
    }


def score_against_tolerances(metrics: dict[str, float], tolerances: dict[str, float]) -> dict[str, Any]:
    ratios = {}
    for key, tolerance in tolerances.items():
        if key not in metrics:
            raise KeyError(f"missing metric for score: {key}")
        value = require_finite(key, metrics[key])
        tol = require_finite(f"tol:{key}", tolerance)
        ratios[key] = value / tol
    controlling = max(ratios, key=ratios.get)
    score = float(ratios[controlling])
    return {
        "ratios": ratios,
        "score": score,
        "controlling_metric": controlling,
        "pass": score < 1.0,
    }


def format_score_ratio(value: float, *, decimals: int = 3) -> str:
    """Format a pass-score ratio where acceptance requires S < 1.

    Values that would round to 1.000 under fixed decimals are shown as ``<1.000``
    so a passing cell is never misread as a failure.
    """
    number = require_finite("score_ratio", value)
    if number < 1.0:
        text = f"{number:.{decimals}f}"
        if float(text) >= 1.0:
            return f"<1.{'0' * decimals}"
        return text
    return f"{number:.{decimals}f}"


def format_significant(value: float, *, digits: int = 4) -> str:
    """Format with significant digits, avoiding a rounded '1' for S < 1."""
    number = require_finite("significant_value", value)
    if 0.0 < number < 1.0:
        text = f"{number:.{digits}g}"
        if float(text) >= 1.0:
            return f"<1.{'0' * max(digits - 1, 3)}"
        return text
    return f"{number:.{digits}g}"


def acceptance_tolerances(solver: SolverConfig | dict[str, Any]) -> dict[str, float]:
    if isinstance(solver, SolverConfig):
        payload = solver.as_metadata()
    else:
        payload = solver
    return {
        "max_superadiabaticity": float(payload["epsilon_gradient"]),
        "potential_temperature_rms": float(payload["theta_rms_tolerance"]),
        "temperature_rms": float(payload["temperature_rms_tolerance"]),
        "temperature_max": float(payload["temperature_max_tolerance"]),
        "normalized_tendency_max": float(payload["tendency_tolerance"]),
        "convective_flux_max": float(payload["flux_tolerance"]),
        "enthalpy_drift": float(payload["enthalpy_drift_tolerance"]),
    }


def metrics_for_score(metrics: dict[str, float], max_abs_enthalpy_drift: float | None = None) -> dict[str, float]:
    payload = {
        "max_superadiabaticity": float(metrics["max_superadiabaticity"]),
        "potential_temperature_rms": float(metrics["potential_temperature_rms"]),
        "temperature_rms": float(metrics["temperature_rms"]),
        "temperature_max": float(metrics["temperature_max"]),
        "normalized_tendency_max": float(metrics["normalized_tendency_max"]),
        "convective_flux_max": float(metrics["convective_flux_max"]),
        "enthalpy_drift": float(
            metrics["enthalpy_drift"]
            if max_abs_enthalpy_drift is None
            else max_abs_enthalpy_drift
        ),
    }
    for key, value in payload.items():
        require_finite(key, value)
    return payload
