"""Stage 2 diagnostic exporters. Never overwrites production_campaign.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PLOTS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PLOTS_ROOT.parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTS_ROOT))
STAGE2_ROOT = PLOTS_ROOT.parent
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

from convection_mlt.closure import mixing_length_flux
from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import numerical_isentrope
from convection_mlt.energy import column_enthalpy_per_area
from convection_mlt.gravity import ConstantGravity, InverseSquareGravity
from convection_mlt.grid import build_grid, log_pressure_edges
from convection_mlt.hydrostatics import HydrostaticDomainError, reconstruct_hydrostatic
from convection_mlt.solvers_enthalpy import solve_adaptive_enthalpy, trial_enthalpy_step
from convection_mlt.state import build_column_state
from convection_mlt.thermodynamics import (
    ConstantH2Thermo,
    NASAThermo,
    ThermoDomainError,
    h2_he_mixture,
    monatomic_helium,
)
from convection_mlt.trace import IntegrationTrace, TraceLevel

from campaign_spec import (
    CAMPAIGN_CONFIG,
    CANONICAL_P_BOTTOM,
    CANONICAL_P_TOP,
    G0,
    TEMPERATURE_CASE,
    enrich_campaign_payload,
)
from common import (
    DATA_DIR,
    ENRICHED_CAMPAIGN_PATH,
    RAW_CAMPAIGN_PATH,
    code_revision,
    ensure_dirs,
    read_json,
    require_source,
    write_json,
)
from hydro_references import (
    analytic_isothermal_constant_g_edges,
    analytic_isothermal_inverse_square_edges,
    column_scale_height_error,
    integrate_z_of_pressure,
)
try:
    from experiments.common import superadiabatic_seed
except ImportError:
    from stage2.experiments.common import superadiabatic_seed

SOLVER = SolverConfig(
    max_steps=500_000,
    temperature_rms_tolerance=1.0e-6,
    theta_rms_tolerance=1.0e-6,
    flux_tolerance=5.0e-3,
    enthalpy_drift_tolerance=1.0e-12,
)
PHYSICS = PhysicsConfig(gravity=G0, alpha=1.0)


def _campaign_solver_config() -> dict[str, float]:
    return dict(CAMPAIGN_CONFIG)


def enrich_campaign() -> Path:
    require_source(RAW_CAMPAIGN_PATH, description="raw production campaign JSON")
    raw = read_json(RAW_CAMPAIGN_PATH)
    payload = enrich_campaign_payload(raw)
    payload["code_revision"] = code_revision()
    write_json(ENRICHED_CAMPAIGN_PATH, payload)
    print(f"wrote {ENRICHED_CAMPAIGN_PATH}")
    return ENRICHED_CAMPAIGN_PATH


def _breakpoints(thermo) -> list[float]:
    points = {float(thermo.t_min), float(thermo.t_max)}
    intervals = getattr(thermo, "intervals", None)
    if intervals:
        for t_min, t_max, _coeffs in intervals:
            points.add(float(t_min))
            points.add(float(t_max))
    return sorted(points)


def _interval_aware_dhdT(thermo, temperature: np.ndarray, dt: float = 1.0e-4) -> np.ndarray:
    """One-sided h(T) differences that do not straddle NASA piecewise breaks."""
    t = np.asarray(temperature, dtype=float)
    breaks = np.asarray(_breakpoints(thermo), dtype=float)
    interior = breaks[1:-1]
    deriv = np.empty_like(t)
    for i, ti in enumerate(t):
        lo = float(np.max(breaks[breaks <= ti], initial=breaks[0]))
        hi_candidates = breaks[breaks > ti]
        hi = float(np.min(hi_candidates, initial=breaks[-1]))
        # Stay inside the assigned piecewise interval.
        forward_ok = ti + dt < hi
        backward_ok = ti - dt > lo
        near_join = np.any(np.abs(interior - ti) <= 2.0 * dt) if interior.size else False
        if near_join:
            # Prefer the side that remains in the same interval.
            if ti >= interior[0]:
                t1, t2 = min(ti + dt, hi - 1.0e-12), ti
            else:
                t1, t2 = ti, max(ti - dt, lo + 1.0e-12)
        elif forward_ok and backward_ok:
            t1, t2 = ti + dt, ti - dt
        elif forward_ok:
            t1, t2 = ti + dt, ti
        elif backward_ok:
            t1, t2 = ti, ti - dt
        else:
            deriv[i] = float("nan")
            continue
        h1 = float(thermo.enthalpy(np.asarray([t1]))[0])
        h2 = float(thermo.enthalpy(np.asarray([t2]))[0])
        deriv[i] = (h1 - h2) / (t1 - t2)
    return deriv


def _provider_curve(thermo, temperatures: np.ndarray) -> dict[str, Any]:
    t = np.asarray(temperatures, dtype=float)
    cp = thermo.specific_heat(t)
    h = thermo.enthalpy(t)
    psi = thermo.psi(t)
    nabla = thermo.nabla_ad_at(t)
    t_from_h = thermo.invert_enthalpy(h)
    from convection_mlt.thermodynamics import invert_psi_newton

    t_psi = invert_psi_newton(thermo, psi, t_min=thermo.t_min, t_max=thermo.t_max)
    breaks = _breakpoints(thermo)
    dhdT = _interval_aware_dhdT(thermo, t)
    residual = np.abs(dhdT - cp) / np.maximum(np.abs(cp), 1.0)
    residual[~np.isfinite(dhdT)] = np.nan
    h_err = np.abs(t_from_h / t - 1.0)
    psi_err = np.abs(t_psi / t - 1.0)
    # Exact piecewise joins are excluded from derivative/inversion curves.
    # Continuity of cp, h, and Psi across the join is audited separately.
    join_mask = np.zeros(t.shape, dtype=bool)
    for break_t in breaks[1:-1]:
        join_mask |= np.abs(t - break_t) < 1.0e-8
    residual[join_mask] = np.nan
    h_err[join_mask] = np.nan
    psi_err[join_mask] = np.nan
    dhdT[join_mask] = np.nan
    return {
        "temperature_k": t.tolist(),
        "cp_J_per_kg_K": cp.tolist(),
        "enthalpy_J_per_kg": h.tolist(),
        "nabla_ad": nabla.tolist(),
        "dh_dT_J_per_kg_K": dhdT.tolist(),
        "dh_dT_relative_residual": residual.tolist(),
        "T_from_h_relative_error": h_err.tolist(),
        "T_from_psi_relative_error": psi_err.tolist(),
        "breakpoints_k": breaks,
        "t_ref_k": float(thermo.t_ref),
        "t_min_k": float(thermo.t_min),
        "t_max_k": float(thermo.t_max),
        "exact_join_samples_excluded_from_residuals": True,
    }


def export_thermo_audit() -> Path:
    ensure_dirs()
    nasa = NASAThermo.from_json()
    helium = monatomic_helium()
    mixtures = {0.0: nasa, 0.10: h2_he_mixture(0.10), 0.25: h2_he_mixture(0.25)}
    t_lo = 250.0
    t_hi = 5000.0
    # Dense samples hugging the 1000 K break from each side, excluding the exact join.
    base = np.unique(
        np.concatenate(
            [
                np.geomspace(t_lo, t_hi, 240),
                np.array([999.0, 999.5, 999.9, 999.99, 1000.01, 1000.1, 1000.5, 1001.0]),
            ]
        )
    )
    base = base[np.abs(base - 1000.0) > 1.0e-8]
    providers = {
        "nasa_h2": _provider_curve(nasa, base),
        "monatomic_he": _provider_curve(helium, base[(base >= helium.t_min) & (base <= helium.t_max)]),
    }
    mixture_curves = {}
    for x_he, thermo in mixtures.items():
        mixture_curves[str(x_he)] = _provider_curve(thermo, base)
    payload = {
        "title": "Stage 2 thermodynamic provider audit",
        "code_revision": code_revision(),
        "providers": providers,
        "mixtures": mixture_curves,
        "notes": (
            "dh/dT uses interval-aware one-sided differences that do not "
            "straddle the NASA 1000 K piecewise breakpoint. Residual and "
            "inversion curves omit the exact join sample; cp/h/Psi continuity "
            "at that join is reported in the audit table."
        ),
    }
    path = DATA_DIR / "thermo_audit.json"
    write_json(path, payload)
    print(f"wrote {path}")
    return path


def _history_from_trace(trace: IntegrationTrace) -> list[dict[str, Any]]:
    rows = []
    for item in trace.accepted_steps:
        rows.append(
            {
                "accepted_step": item.accepted_step,
                "simulated_time": item.simulated_time,
                "dt_accepted": item.dt_accepted,
                "max_superadiabaticity": item.metrics.max_superadiabaticity,
                "convective_flux_max": item.metrics.convective_flux_max,
                "enthalpy_drift": item.metrics.enthalpy_drift,
                "signed_enthalpy_drift": item.signed_enthalpy_drift,
                "entropy_span": item.entropy_span,
                "temperature_rms": item.metrics.temperature_rms,
            }
        )
    return rows


def export_representative_column() -> Path:
    ensure_dirs()
    nasa = NASAThermo.from_json()
    n_layers = 100
    grid = build_grid(log_pressure_edges(CANONICAL_P_BOTTOM, CANONICAL_P_TOP, n_layers), G0)
    seed = superadiabatic_seed(grid)
    gravity = ConstantGravity(G0)
    state0 = build_column_state(grid, seed, nasa, gravity)
    h0 = column_enthalpy_per_area(state0.mass_path, state0.enthalpy)
    reference = numerical_isentrope(grid, seed, nasa, state0.mass_path)
    trace = IntegrationTrace(level=TraceLevel.PROFILES, summary_stride=500)
    result = solve_adaptive_enthalpy(
        grid, seed, PHYSICS, nasa, gravity, SOLVER, trace=trace
    )
    final = build_column_state(grid, result.temperature, nasa, gravity, enthalpy=None)
    # Final flux from the trace is the post-commit closure on the accepted state.
    if trace.final_flux is None:
        raise RuntimeError("representative solve did not record post-commit final flux")
    h1 = column_enthalpy_per_area(final.mass_path, final.enthalpy)
    s0 = nasa.entropy(state0.temperature, grid.pressure_centres)
    s1 = nasa.entropy(final.temperature, grid.pressure_centres)
    s_ref = nasa.entropy(reference, grid.pressure_centres)
    payload = {
        "title": "Representative constant-g NASA H2 column",
        "n_layers": n_layers,
        "x_he": 0.0,
        "gravity_mode": "constant",
        "irregular_grid": False,
        "pressure_bottom": CANONICAL_P_BOTTOM,
        "pressure_top": CANONICAL_P_TOP,
        "temperature_case": TEMPERATURE_CASE,
        "status": result.status.value,
        "steps": result.steps,
        "reason": result.reason,
        "pressure_centres_pa": grid.pressure_centres.tolist(),
        "pressure_edges_pa": grid.pressure_edges.tolist(),
        "temperature_initial_k": state0.temperature.tolist(),
        "temperature_final_k": result.temperature.tolist(),
        "temperature_isentrope_k": reference.tolist(),
        "entropy_initial": s0.tolist(),
        "entropy_final": s1.tolist(),
        "entropy_isentrope": s_ref.tolist(),
        "flux_final": np.asarray(trace.final_flux).tolist(),
        "column_enthalpy_initial": h0,
        "column_enthalpy_final": h1,
        "enthalpy_drift": abs(h1 - h0) / max(abs(h0), 1.0),
        "metrics": result.metrics.as_dict(),
        "history": _history_from_trace(trace),
        "code_revision": code_revision(),
        "flux_semantics": "post_commit_closure_on_accepted_state",
    }
    path = DATA_DIR / "representative_column.json"
    write_json(path, payload)
    print(f"wrote {path}")
    return path


def _mild_temperature(pressure: float, p_bottom: float, p_top: float) -> float:
    xi = np.log(p_bottom / pressure) / np.log(p_bottom / p_top)
    return 1500.0 + 0.5 * np.sin(np.pi * xi)


def _strong_temperature(pressure: float, p_bottom: float, p_top: float) -> float:
    xi = np.log(p_bottom / pressure) / np.log(p_bottom / p_top)
    return 1500.0 + 400.0 * np.sin(np.pi * xi)


def export_hydro_references() -> Path:
    ensure_dirs()
    from convection_mlt.thermodynamics import ConstantH2Thermo

    gas = ConstantH2Thermo()
    g0 = G0
    p_bottom, p_top = CANONICAL_P_BOTTOM, CANONICAL_P_TOP
    t_iso = 1000.0
    n_iso = 100
    grid_iso = build_grid(log_pressure_edges(p_bottom, p_top, n_iso), g0)
    t_iso_prof = np.full(n_iso, t_iso)
    model_const = reconstruct_hydrostatic(
        grid_iso, t_iso_prof, gas, ConstantGravity(g0)
    )
    analytic_const = analytic_isothermal_constant_g_edges(
        grid_iso.pressure_edges, t_iso, gas.gas_constant, g0
    )
    rp = 1.0e8
    grav_isq = InverseSquareGravity(g0=g0, planet_radius=rp)
    model_isq = reconstruct_hydrostatic(grid_iso, t_iso_prof, gas, grav_isq)
    analytic_isq = analytic_isothermal_inverse_square_edges(
        grid_iso.pressure_edges, t_iso, gas.gas_constant, grav_isq
    )

    rtol, atol, max_step = 1.0e-12, 1.0e-10, 0.005
    mild_errors = {}
    strong_errors = {}
    refinement = {}
    last_stats = None
    for n in (50, 100, 200):
        grid = build_grid(log_pressure_edges(p_bottom, p_top, n), g0)
        t_mild = np.array([_mild_temperature(float(p), p_bottom, p_top) for p in grid.pressure_centres])
        t_strong = np.array(
            [_strong_temperature(float(p), p_bottom, p_top) for p in grid.pressure_centres]
        )
        model_mild = reconstruct_hydrostatic(grid, t_mild, gas, ConstantGravity(g0))
        model_strong = reconstruct_hydrostatic(grid, t_strong, gas, ConstantGravity(g0))
        z_mild_ref, stats_mild = integrate_z_of_pressure(
            grid.pressure_edges,
            gas_constant=gas.gas_constant,
            temperature_of_p=lambda p, pb=p_bottom, pt=p_top: _mild_temperature(p, pb, pt),
            gravity=ConstantGravity(g0),
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        z_strong_ref, stats_strong = integrate_z_of_pressure(
            grid.pressure_edges,
            gas_constant=gas.gas_constant,
            temperature_of_p=lambda p, pb=p_bottom, pt=p_top: _strong_temperature(p, pb, pt),
            gravity=ConstantGravity(g0),
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        last_stats = stats_strong
        mild_errors[str(n)] = float(
            np.max(np.abs(model_mild.z_edges - z_mild_ref)) / np.max(np.abs(z_mild_ref))
        )
        strong_errors[str(n)] = float(
            np.max(np.abs(model_strong.z_edges - z_strong_ref)) / np.max(np.abs(z_strong_ref))
        )
        refinement[str(n)] = {
            "pressure_edges_pa": grid.pressure_edges.tolist(),
            "z_model_mild_m": model_mild.z_edges.tolist(),
            "z_ref_mild_m": z_mild_ref.tolist(),
            "z_model_strong_m": model_strong.z_edges.tolist(),
            "z_ref_strong_m": z_strong_ref.tolist(),
        }

    # Reference refinement check: tighten ODE tolerances and compare.
    grid_check = build_grid(log_pressure_edges(p_bottom, p_top, 100), g0)
    z_loose, _ = integrate_z_of_pressure(
        grid_check.pressure_edges,
        gas_constant=gas.gas_constant,
        temperature_of_p=lambda p: _strong_temperature(p, p_bottom, p_top),
        gravity=ConstantGravity(g0),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    z_tight, stats_tight = integrate_z_of_pressure(
        grid_check.pressure_edges,
        gas_constant=gas.gas_constant,
        temperature_of_p=lambda p: _strong_temperature(p, p_bottom, p_top),
        gravity=ConstantGravity(g0),
        rtol=rtol * 0.1,
        atol=atol * 0.1,
        max_step=max_step * 0.5,
    )
    ref_change = float(np.max(np.abs(z_tight - z_loose)) / np.max(np.abs(z_tight)))

    # Round-trip using the *model* inverse, reported as a diagnostic of the scheme.
    nasa = NASAThermo.from_json()
    grid_rt = build_grid(log_pressure_edges(p_bottom, p_top, 50), g0)
    t_rt = 2000.0 * (grid_rt.pressure_centres / grid_rt.pressure_centres[0]) ** 0.12
    hydro_rt = reconstruct_hydrostatic(grid_rt, t_rt, nasa, ConstantGravity(g0))
    from convection_mlt.hydrostatics import pressure_from_height

    recovered = pressure_from_height(
        grid_rt, t_rt, nasa, ConstantGravity(g0), hydro_rt.z_edges, hydro=hydro_rt
    )
    round_trip = float(np.max(np.abs(recovered / grid_rt.pressure_edges - 1.0)))

    payload = {
        "title": "Independent hydrostatic references",
        "code_revision": code_revision(),
        "isothermal_constant_g": {
            "n_layers": n_iso,
            "temperature_k": t_iso,
            "pressure_edges_pa": grid_iso.pressure_edges.tolist(),
            "z_model_m": model_const.z_edges.tolist(),
            "z_analytic_m": analytic_const.tolist(),
            "max_abs_z_error_over_Hp": float(
                np.max(np.abs(model_const.z_edges - analytic_const))
                / (gas.gas_constant * t_iso / g0)
            ),
        },
        "isothermal_inverse_square": {
            "n_layers": n_iso,
            "temperature_k": t_iso,
            "planet_radius_m": rp,
            "pressure_edges_pa": grid_iso.pressure_edges.tolist(),
            "z_model_m": model_isq.z_edges.tolist(),
            "z_analytic_m": analytic_isq.tolist(),
            "max_relative_top_error": float(
                np.max(np.abs(model_isq.z_edges[-1] / analytic_isq[-1] - 1.0))
            ),
        },
        "nonisothermal": {
            "mild_relative_error_vs_ode": mild_errors,
            "strong_relative_error_vs_ode": strong_errors,
            "profiles": refinement,
        },
        "round_trip_relative_pressure_error": round_trip,
        "ode_reference": last_stats,
        "reference_refinement_check": {
            "n_layers": 100,
            "loose_rtol": rtol,
            "tight_rtol": rtol * 0.1,
            "max_relative_change": ref_change,
            "tight_stats": stats_tight,
        },
    }
    path = DATA_DIR / "hydro_references.json"
    write_json(path, payload)
    print(f"wrote {path}")
    return path


def export_gravity_limit() -> Path:
    ensure_dirs()
    nasa = NASAThermo.from_json()
    n_layers = 50
    grid = build_grid(log_pressure_edges(CANONICAL_P_BOTTOM, CANONICAL_P_TOP, n_layers), G0)
    seed = superadiabatic_seed(grid)
    const_result = solve_adaptive_enthalpy(
        grid, seed, PHYSICS, nasa, ConstantGravity(G0), SOLVER
    )
    const_state = build_column_state(grid, const_result.temperature, nasa, ConstantGravity(G0))
    radii = (1.0e7, 1.0e8, 1.0e9, 1.0e10)
    cases = []
    thin_profile = None
    for rp in radii:
        gravity = InverseSquareGravity(g0=G0, planet_radius=rp)
        state0 = build_column_state(grid, seed, nasa, gravity)
        h0 = column_enthalpy_per_area(state0.mass_path, state0.enthalpy)
        result = solve_adaptive_enthalpy(grid, seed, PHYSICS, nasa, gravity, SOLVER)
        state = build_column_state(grid, result.temperature, nasa, gravity)
        h1 = column_enthalpy_per_area(state.mass_path, state.enthalpy)
        apparent = abs(h1 - h0) / max(abs(h0), 1.0)
        heights = column_scale_height_error(state.z_edges, const_state.z_edges)
        dg = (state.g_edges - G0) / G0
        pred_signed = -2.0 * state.z_edges / rp
        pred_mag = 2.0 * state.z_edges / rp
        max_z_over_rp = float(np.max(state.z_edges) / rp)
        weights = state.mass_path
        reference_var = numerical_isentrope(
            grid, state.temperature, nasa, state.mass_path
        )
        relative = (state.temperature - reference_var) / reference_var
        t_rms_vs_isentrope = float(
            np.sqrt(np.sum(weights * relative**2) / np.sum(weights))
        )
        record = {
            "planet_radius_m": rp,
            "status": result.status.value,
            "steps": result.steps,
            "temperature_rms": result.metrics.temperature_rms,
            "temperature_rms_vs_isentrope": t_rms_vs_isentrope,
            "max_superadiabaticity": result.metrics.max_superadiabaticity,
            "apparent_enthalpy_drift": apparent,
            "max_z_over_rp": max_z_over_rp,
            "max_abs_dg_over_g0": float(np.max(np.abs(dg))),
            "E_z": heights["E_z"],
            "E_z_top": heights["E_z_top"],
            "extreme_stress": rp == 1.0e7,
            "thin_atmosphere": max_z_over_rp < 0.1 and rp != 1.0e7,
        }
        if record["thin_atmosphere"] and thin_profile is None:
            thin_profile = {
                "planet_radius_m": rp,
                "z_edges_m": state.z_edges.tolist(),
                "g_edges": state.g_edges.tolist(),
                "dg_over_g0_signed": dg.tolist(),
                "approx_signed_minus_2z_over_rp": pred_signed.tolist(),
                "dg_over_g0_magnitude": np.abs(dg).tolist(),
                "approx_magnitude_2z_over_rp": pred_mag.tolist(),
            }
        cases.append(record)

    payload = {
        "title": "Coupled inverse-square gravity limit sweep",
        "n_layers": n_layers,
        "x_he": 0.0,
        "g0": G0,
        "gm_definition": "g0 * Rp**2",
        "planet_radii_m": list(radii),
        "constant_g_status": const_result.status.value,
        "constant_g_steps": const_result.steps,
        "cases": cases,
        "thin_atmosphere_profile": thin_profile,
        "notes": (
            "Rp=1e7 m is an extreme hydrostatic stress test and is excluded "
            "from thin-atmosphere Delta g / g fits. Apparent enthalpy drift "
            "is diagnostic only. E_z uses column-scale normalization. "
            "temperature_rms_vs_isentrope matches the campaign definition "
            "(mass-weighted relative T vs a rebuilt numerical isentrope). "
            "temperature_rms is the solver's variable-g entropy-span proxy "
            "and is not comparable to the campaign T RMS."
        ),
        "code_revision": code_revision(),
    }
    path = DATA_DIR / "gravity_limit.json"
    write_json(path, payload)
    print(f"wrote {path}")
    return path


def _audit_row(
    metric: str,
    observed: float | str,
    tolerance: float | None,
    comparison: str,
    units: str,
    source_case: str,
    notes: str = "",
) -> dict[str, Any]:
    if comparison == "N/A":
        status = "N/A"
    elif not isinstance(observed, (int, float)):
        raise TypeError(f"{metric}: numeric comparison requires a numeric observed value")
    elif tolerance is None:
        raise TypeError(f"{metric}: numeric comparison requires a numeric tolerance")
    elif comparison == "<=":
        status = "PASS" if observed <= tolerance else "FAIL"
    elif comparison == ">=":
        status = "PASS" if observed >= tolerance else "FAIL"
    elif comparison == ">":
        status = "PASS" if observed > tolerance else "FAIL"
    elif comparison == "==":
        status = "PASS" if observed == tolerance else "FAIL"
    else:
        raise ValueError(f"unsupported audit comparison {comparison!r}")
    return {
        "metric": metric,
        "observed": observed,
        "tolerance": tolerance,
        "comparison": comparison,
        "status": status,
        "units": units,
        "source_case": source_case,
        "notes": notes,
    }


def export_audit() -> Path:
    ensure_dirs()
    thermo = NASAThermo.from_json()
    he = monatomic_helium()
    mixtures = {0.0: thermo, 0.10: h2_he_mixture(0.10), 0.25: h2_he_mixture(0.25)}

    # Use exactly the same temperature samples and join-exclusion logic as
    # _provider_curve / export_thermo_audit so that the audit maxima match
    # the plotted curves in Figure 01.
    t_lo, t_hi = 250.0, 5000.0
    t_all = np.unique(
        np.concatenate(
            [
                np.geomspace(t_lo, t_hi, 240),
                np.array([999.0, 999.5, 999.9, 999.99, 1000.01, 1000.1, 1000.5, 1001.0]),
            ]
        )
    )
    t_all = t_all[np.abs(t_all - 1000.0) > 1.0e-8]

    # Compute residuals across all plotted providers, excluding the join.
    dhdt_res = 0.0
    t_h_err = 0.0
    psi_err = 0.0
    from convection_mlt.thermodynamics import invert_psi_newton
    for _xhe, mix_thermo in mixtures.items():
        t_valid = t_all[(t_all >= mix_thermo.t_min) & (t_all <= mix_thermo.t_max)]
        cp_v = mix_thermo.specific_heat(t_valid)
        dhdT_v = _interval_aware_dhdT(mix_thermo, t_valid)
        res_v = np.abs(dhdT_v - cp_v) / np.maximum(np.abs(cp_v), 1.0)
        res_v[~np.isfinite(dhdT_v)] = np.nan
        breaks = _breakpoints(mix_thermo)
        for bk in breaks[1:-1]:
            res_v[np.abs(t_valid - bk) < 1.0e-8] = np.nan
        dhdt_res = max(dhdt_res, float(np.nanmax(res_v)))

        h_v = mix_thermo.enthalpy(t_valid)
        t_from_h = mix_thermo.invert_enthalpy(h_v)
        t_h_err = max(t_h_err, float(np.max(np.abs(t_from_h / t_valid - 1.0))))

        psi_v = mix_thermo.psi(t_valid)
        t_from_psi = invert_psi_newton(
            mix_thermo, psi_v, t_min=mix_thermo.t_min, t_max=mix_thermo.t_max
        )
        psi_err = max(psi_err, float(np.max(np.abs(t_from_psi / t_valid - 1.0))))

    t = np.array([300.0, 800.0, 1200.0, 2500.0])
    rho = thermo.density(np.full_like(t, 1.0e5), t)
    eos_res = float(np.max(np.abs(rho * thermo.gas_constant * t / 1.0e5 - 1.0)))
    pure_h2 = h2_he_mixture(0.0)
    pure_h2_err = float(np.max(np.abs(pure_h2.specific_heat(t) - thermo.specific_heat(t))))
    he_exact = float(np.max(np.abs(he.specific_heat(t) / (2.5 * he.gas_constant) - 1.0)))

    hydro = read_json(DATA_DIR / "hydro_references.json")
    gravity = read_json(DATA_DIR / "gravity_limit.json")
    campaign = read_json(ENRICHED_CAMPAIGN_PATH)
    const_cases = [
        c for c in campaign["cases"] if c["campaign_role"] == "parameter_matrix"
    ]
    gravity_cases = [
        c for c in campaign["cases"] if c["campaign_role"] == "gravity_stress"
    ]
    max_drift = max(c["enthalpy_drift"] for c in const_cases)
    n_complete = int(campaign["completed"])
    n_converged = sum(1 for c in campaign["cases"] if c["status"] == "converged")
    n_fail = int(campaign.get("n_failures", 0))
    var_drifts = [c["enthalpy_drift"] for c in gravity_cases]
    var_drift_text = f"{min(var_drifts):.3e} – {max(var_drifts):.3e}"

    grid = build_grid(log_pressure_edges(CANONICAL_P_BOTTOM, CANONICAL_P_TOP, 50), G0)
    seed = superadiabatic_seed(grid)
    state = build_column_state(grid, seed, thermo, ConstantGravity(G0))
    isentrope = numerical_isentrope(grid, seed, thermo, state.mass_path)
    s = thermo.entropy(isentrope, grid.pressure_centres)
    s_span = float(np.max(np.abs(s - s[0])))
    s_scale = max(abs(float(np.max(np.abs(s)))), 1.0)
    s_tol = max(1.0e-12, 64.0 * np.finfo(float).eps * s_scale)
    closure_s = mixing_length_flux(
        grid, isentrope, state.g_edges, 1.0, thermo, use_entropy_instability=True
    )
    s_jump = float(np.max(np.abs(closure_s.entropy_jump)))

    const_cp = ConstantH2Thermo()
    grid_id = build_grid(log_pressure_edges(1.0e7, 1.0e3, 40), 10.0)
    t_id = 1500.0 * (grid_id.pressure_centres / grid_id.pressure_centres[0]) ** 0.40
    entropy_closure = mixing_length_flux(
        grid_id, t_id, 10.0, 1.0, const_cp, use_entropy_instability=True
    )
    legacy_closure = mixing_length_flux(
        grid_id, t_id, 10.0, 1.0, const_cp, use_entropy_instability=False
    )
    entropy_id = float(
        np.max(np.abs(entropy_closure.superadiabaticity - legacy_closure.superadiabaticity))
    )

    temps_mono = np.linspace(thermo.t_min * 1.01, thermo.t_max * 0.99, 80)
    # min cp across all plotted providers (mixtures have lower cp than pure H2)
    min_cp_all = float(np.min(thermo.specific_heat(temps_mono)))
    for _xhe, mix_thermo in mixtures.items():
        t_valid = temps_mono[(temps_mono >= mix_thermo.t_min) & (temps_mono <= mix_thermo.t_max)]
        min_cp_all = min(min_cp_all, float(np.min(mix_thermo.specific_heat(t_valid))))
    # min(dh/dT) via the same interval-aware stencil (units: J kg^-1 K^-1)
    dhdT_mono = _interval_aware_dhdT(thermo, temps_mono)
    min_dhdT = float(np.nanmin(dhdT_mono))

    eps_join = 1.0e-6
    cp_join = abs(
        float(thermo.specific_heat(1000.0 - eps_join)) - float(thermo.specific_heat(1000.0 + eps_join))
    ) / abs(float(thermo.specific_heat(1000.0)))
    h_join = abs(
        float(thermo.enthalpy(1000.0 - eps_join)) - float(thermo.enthalpy(1000.0 + eps_join))
    ) / max(abs(float(thermo.enthalpy(1000.0))), 1.0)
    psi_join = abs(
        float(thermo.psi(np.asarray([1000.0 - eps_join]))[0])
        - float(thermo.psi(np.asarray([1000.0 + eps_join]))[0])
    ) / max(abs(float(thermo.psi(np.asarray([1000.0]))[0])), 1.0)

    prior = state.copy()
    physics = PhysicsConfig(gravity=G0, alpha=1.0)
    reject_cfg = SolverConfig(epsilon_temperature=1.0e-12)
    trial = trial_enthalpy_step(grid, state, 1.0e6, physics, reject_cfg, thermo, ConstantGravity(G0))
    purity_ok = (
        trial.accepted is False
        and np.array_equal(state.temperature, prior.temperature)
        and np.array_equal(state.enthalpy, prior.enthalpy)
        and np.array_equal(state.density_centres, prior.density_centres)
        and np.array_equal(state.density_edges, prior.density_edges)
        and np.array_equal(state.z_centres, prior.z_centres)
        and np.array_equal(state.z_edges, prior.z_edges)
        and np.array_equal(state.g_centres, prior.g_centres)
        and np.array_equal(state.g_edges, prior.g_edges)
        and np.array_equal(state.mass_path, prior.mass_path)
    )

    thermo_domain_ok = False
    try:
        thermo.specific_heat(199.0)
    except ThermoDomainError:
        try:
            thermo.enthalpy(6000.1)
        except ThermoDomainError:
            thermo_domain_ok = True

    hydro_domain_ok = False
    try:
        reconstruct_hydrostatic(
            build_grid(log_pressure_edges(1.0e7, 1.0e-2, 20), 10.0),
            np.full(20, 5000.0),
            ConstantH2Thermo(),
            InverseSquareGravity(g0=10.0, planet_radius=1.0e5),
        )
    except HydrostaticDomainError:
        hydro_domain_ok = True

    ez_limit = next(
        float(c["E_z"])
        for c in gravity["cases"]
        if abs(float(c["planet_radius_m"]) - 1.0e10) < 1.0
    )

    rows = [
        _audit_row("dh/dT = cp identity (all plotted providers)", dhdt_res, 1.0e-8, "<=", "relative", "thermo_audit", "max over NASA H2 + H2-He mixtures, 250-5000 K, exact 1000 K join excluded"),
        _audit_row("equation-of-state identity", eos_res, 1.0e-12, "<=", "relative", "thermo_audit", "rho R T / P"),
        _audit_row("pure-H2 mixture limit", pure_h2_err, 1.0e-12, "<=", "relative", "thermo_audit"),
        _audit_row("exact monatomic He cp", he_exact, 1.0e-12, "<=", "relative", "thermo_audit", "cp = 5/2 R"),
        _audit_row("enthalpy inversion T->h->T (all plotted providers)", t_h_err, 1.0e-12, "<=", "relative", "thermo_audit", "max over NASA H2 + H2-He mixtures, 250-5000 K, exact 1000 K join excluded"),
        _audit_row("entropy-function inversion T->Psi->T (all plotted providers)", psi_err, 1.0e-12, "<=", "relative", "thermo_audit", "max over NASA H2 + H2-He mixtures, 250-5000 K, exact 1000 K join excluded"),
        _audit_row("NASA breakpoint continuity (cp)", cp_join, 2.0e-9, "<=", "relative", "thermo_audit", "checked-in TPIS78 coefficients"),
        _audit_row("NASA breakpoint continuity (h)", h_join, 2.0e-9, "<=", "relative", "thermo_audit", "T=1000 K ± 1e-6 K"),
        _audit_row("NASA breakpoint continuity (Psi)", psi_join, 2.0e-9, "<=", "relative", "thermo_audit", "T=1000 K ± 1e-6 K"),
        _audit_row("cp > 0 (all providers, min cp)", min_cp_all, 0.0, ">", "J kg^-1 K^-1", "thermo_audit", "min over NASA H2 + H2-He mixtures on 80 interior samples"),
        _audit_row("enthalpy monotonicity min(dh/dT) > 0", min_dhdT, 0.0, ">", "J kg^-1 K^-1", "thermo_audit", "interval-aware stencil on NASA H2, 80 interior samples"),
        _audit_row("manufactured-isentrope entropy flatness", s_span, s_tol, "<=", "J kg^-1 K^-1", "N=50 NASA seed"),
        _audit_row("finite-layer entropy-instability identity", entropy_id, 1.0e-14, "<=", "relative", "constant-cp closure", "entropy Δ∇ matches legacy nabla for ConstantH2Thermo"),
        _audit_row("manufactured-isentrope entropy jump", s_jump, s_tol, "<=", "J kg^-1 K^-1", "N=50 NASA seed"),
        _audit_row("rejected-state purity", 1.0 if purity_ok else 0.0, 1.0, "==", "boolean", "trial_enthalpy_step", "T, h, density, z, g, and mass paths unchanged after rejection"),
        _audit_row("thermodynamic domain failure", 1.0 if thermo_domain_ok else 0.0, 1.0, "==", "boolean", "NASAThermo", "ThermoDomainError below t_min and above t_max"),
        _audit_row("hydrostatic domain failure", 1.0 if hydro_domain_ok else 0.0, 1.0, "==", "boolean", "reconstruct_hydrostatic", "HydrostaticDomainError for unreachable inverse-square column"),
        _audit_row(
            "constant-g isothermal hydrostatic error",
            hydro["isothermal_constant_g"]["max_abs_z_error_over_Hp"],
            1.0e-12,
            "<=",
            "relative (z/H_p)",
            "hydro_references",
            "max|z_model − z_analytic| / scale height",
        ),
        _audit_row(
            "inverse-square isothermal hydrostatic error",
            hydro["isothermal_inverse_square"]["max_relative_top_error"],
            1.0e-10,
            "<=",
            "relative (z_top)",
            "hydro_references",
            "max|z_model/z_analytic − 1| at column top",
        ),
        _audit_row(
            "nonisothermal ODE-reference error at N=100 (mild)",
            hydro["nonisothermal"]["mild_relative_error_vs_ode"]["100"],
            2.0e-8,
            "<=",
            "relative",
            "hydro_references",
            "max|z − z_ODE|/max|z_ODE|; mild T(P) profile",
        ),
        _audit_row(
            "pressure round-trip error",
            hydro["round_trip_relative_pressure_error"],
            1.0e-10,
            "<=",
            "relative",
            "hydro_references",
            "max|P_recovered/P_edges − 1|",
        ),
        _audit_row(
            "ODE reference refinement change",
            hydro["reference_refinement_check"]["max_relative_change"],
            1.0e-6,
            "<=",
            "relative",
            "hydro_references",
            "max|z_tight − z_loose|/max|z_loose|",
        ),
        _audit_row(
            "constant-g enthalpy conservation (campaign max)",
            max_drift,
            CAMPAIGN_CONFIG["enthalpy_drift_tolerance"],
            "<=",
            "relative",
            "production_campaign_enriched",
        ),
        _audit_row(
            "constant-gravity limit E_z (Rp=1e10 m)",
            ez_limit,
            1.0e-3,
            "<=",
            "relative",
            "gravity_limit",
            "coupled inverse-square column vs constant-g height",
        ),
        _audit_row(
            "production completion",
            float(n_complete),
            27.0,
            ">=",
            "count",
            "production_campaign_enriched",
            f"{n_converged}/27 converged, {n_fail} gate failures",
        ),
        _audit_row(
            "variable-g apparent enthalpy drift",
            var_drift_text,
            None,
            "N/A",
            "relative",
            "gravity_stress",
            "Pressure-layer mass paths change under inverse-square gravity; strict conservation is not claimed.",
        ),
    ]
    payload = {
        "title": "Stage 2 deterministic audit",
        "code_revision": code_revision(),
        "rows": rows,
    }
    path = DATA_DIR / "audit.json"
    write_json(path, payload)
    print(f"wrote {path}")
    return path


TARGETS = {
    "enrich": enrich_campaign,
    "thermo": export_thermo_audit,
    "representative": export_representative_column,
    "hydro": export_hydro_references,
    "gravity_limit": export_gravity_limit,
    "audit": export_audit,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 2 validation diagnostic data")
    parser.add_argument(
        "targets",
        nargs="*",
        default=["all"],
        help="enrich, thermo, representative, hydro, gravity_limit, audit, or all",
    )
    args = parser.parse_args()
    names = list(TARGETS) if "all" in args.targets else args.targets
    # Audit reads saved hydro/gravity/campaign files; regenerate them only if absent.
    if "audit" in names:
        deps = (
            ("enrich", ENRICHED_CAMPAIGN_PATH),
            ("hydro", DATA_DIR / "hydro_references.json"),
            ("gravity_limit", DATA_DIR / "gravity_limit.json"),
        )
        for dep, path in reversed(deps):
            if dep not in names and not path.exists():
                names.insert(0, dep)
    seen = set()
    ordered = []
    for name in names:
        if name not in TARGETS:
            raise SystemExit(f"unknown target {name!r}")
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in ordered:
        print(f"==> {name}", flush=True)
        TARGETS[name]()


if __name__ == "__main__":
    main()
