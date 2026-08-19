"""Shared Stage 2 experiment helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import numerical_isentrope
from convection_mlt.energy import column_enthalpy_per_area
from convection_mlt.gravity import ConstantGravity, InverseSquareGravity, GravityLaw
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.solvers_enthalpy import solve_adaptive_enthalpy
from convection_mlt.state import build_column_state
from convection_mlt.thermodynamics import ThermoProvider, h2_he_mixture, NASAThermo


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def irregular_log_pressure_edges(
    p_bottom: float, p_top: float, n_layers: int, seed: int = 42
) -> np.ndarray:
    """Irregular log-P edges matching Stage 1 locality/robustness.

    Uses a mild sinusoidal perturbation of uniform log-P (not an unconstrained
    random scramble). A pure RNG sort can create layers thin enough that the
    diffusion CFL collapses dt by ~100× and production cases stall for hours.
    ``seed`` is retained for API compatibility; the Stage 1 construction is
    deterministic in ``n_layers`` and does not use it.
    """
    del seed  # API compatibility with campaign kwargs
    log_edges = np.linspace(np.log(p_bottom), np.log(p_top), n_layers + 1)
    phase = np.linspace(0.0, np.pi, n_layers + 1)
    perturbation = 0.12 * np.sin(phase) * abs(np.diff(log_edges).mean())
    values = np.exp(log_edges + perturbation)
    if not np.all(np.diff(values) < 0.0):
        raise AssertionError("constructed irregular grid is not monotonic")
    return values


def superadiabatic_seed(grid, t_bottom: float = 4000.0, nabla: float = 0.30):
    return t_bottom * (grid.pressure_centres / grid.pressure_centres[0]) ** nabla


def make_thermo(x_he: float = 0.0) -> ThermoProvider:
    if x_he == 0.0:
        return NASAThermo.from_json()
    return h2_he_mixture(x_he)


def make_gravity(mode: str, g0: float = 10.0, planet_radius: float = 1.0e8) -> GravityLaw:
    if mode == "constant":
        return ConstantGravity(g0)
    if mode == "inverse_square":
        return InverseSquareGravity(g0=g0, planet_radius=planet_radius)
    raise ValueError(f"unknown gravity mode {mode}")


def run_case(
    *,
    n_layers: int,
    x_he: float = 0.0,
    gravity_mode: str = "constant",
    planet_radius: float = 1.0e8,
    p_bottom: float = 1.0e7,
    p_top: float = 1.0e3,
    irregular: bool = False,
    seed: int = 42,
    max_steps: int = 2_000_000,
    case_id: int | None = None,
    campaign_role: str | None = None,
    temperature_case: str = "superadiabatic_nabla_0.30_Tbot_4000",
) -> dict:
    from convection_mlt.solvers import SolverFailure

    g0 = 10.0
    edges = (
        irregular_log_pressure_edges(p_bottom, p_top, n_layers, seed=seed)
        if irregular
        else log_pressure_edges(p_bottom, p_top, n_layers)
    )
    grid = build_grid(edges, g0)
    thermo = make_thermo(x_he)
    gravity = make_gravity(gravity_mode, g0=g0, planet_radius=planet_radius)
    seed_t = superadiabatic_seed(grid)
    state0 = build_column_state(grid, seed_t, thermo, gravity)
    h0 = column_enthalpy_per_area(state0.mass_path, state0.enthalpy)
    reference = numerical_isentrope(grid, seed_t, thermo, state0.mass_path)
    physics = PhysicsConfig(gravity=g0, alpha=1.0)
    config = SolverConfig(
        max_steps=max_steps,
        temperature_rms_tolerance=1.0e-6,
        theta_rms_tolerance=1.0e-6,
        flux_tolerance=5.0e-3,
        enthalpy_drift_tolerance=1.0e-12,
    )
    try:
        result = solve_adaptive_enthalpy(grid, seed_t, physics, thermo, gravity, config)
    except SolverFailure as exc:
        failed = exc.result
        payload = {
            "n_layers": n_layers,
            "x_he": x_he,
            "gravity_mode": gravity_mode,
            "planet_radius": planet_radius if gravity_mode == "inverse_square" else None,
            "irregular_grid": irregular,
            "status": failed.status.value,
            "steps": failed.steps,
            "enthalpy_drift": failed.metrics.enthalpy_drift,
            "temperature_rms_vs_isentrope": failed.metrics.temperature_rms,
            "max_z_over_rp": state0.max_z_over_rp,
            "max_superadiabaticity": failed.metrics.max_superadiabaticity,
            "reason": failed.reason,
        }
        return _with_campaign_metadata(
            payload,
            case_id=case_id,
            campaign_role=campaign_role,
            temperature_case=temperature_case,
            p_bottom=p_bottom,
            p_top=p_top,
        )
    final = build_column_state(grid, result.temperature, thermo, gravity)
    h1 = column_enthalpy_per_area(final.mass_path, final.enthalpy)
    weights = final.mass_path
    # Constant-g: compare to initial-mass enthalpy-normalized isentrope.
    # Variable-g: rebuild with current mass path (exit gate is entropy/theta based).
    if gravity_mode == "constant":
        relative = (final.temperature - reference) / reference
        rms = float(np.sqrt(np.sum(weights * relative**2) / np.sum(weights)))
    else:
        reference_var = numerical_isentrope(
            grid, final.temperature, thermo, final.mass_path
        )
        relative = (final.temperature - reference_var) / reference_var
        rms = float(np.sqrt(np.sum(weights * relative**2) / np.sum(weights)))
    payload = {
        "n_layers": n_layers,
        "x_he": x_he,
        "gravity_mode": gravity_mode,
        "planet_radius": planet_radius if gravity_mode == "inverse_square" else None,
        "irregular_grid": irregular,
        "status": result.status.value,
        "steps": result.steps,
        "enthalpy_drift": abs(h1 - h0) / max(abs(h0), 1.0),
        "temperature_rms_vs_isentrope": rms,
        "max_z_over_rp": final.max_z_over_rp,
        "max_superadiabaticity": result.metrics.max_superadiabaticity,
        "reason": result.reason,
    }
    return _with_campaign_metadata(
        payload,
        case_id=case_id,
        campaign_role=campaign_role,
        temperature_case=temperature_case,
        p_bottom=p_bottom,
        p_top=p_top,
    )


def _with_campaign_metadata(
    payload: dict,
    *,
    case_id: int | None,
    campaign_role: str | None,
    temperature_case: str,
    p_bottom: float,
    p_top: float,
) -> dict:
    payload["pressure_bottom"] = p_bottom
    payload["pressure_top"] = p_top
    payload["temperature_case"] = temperature_case
    if case_id is not None:
        payload["case_id"] = case_id
    if campaign_role is not None:
        payload["campaign_role"] = campaign_role
    return payload
