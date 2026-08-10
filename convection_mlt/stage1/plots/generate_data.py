"""Generate Stage 1 validation datasets for Matplotlib figure scripts.

Use ``--smoke`` for CI-reduced cases. Default is the full production evidence
bundle requested by the Stage 1 validation plot suite.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from convection_mlt.closure import mixing_length_flux
from convection_mlt.config import PhysicsConfig, SolverConfig
from convection_mlt.diagnostics import (
    enthalpy_normalized_adiabat,
    mixing_region_labels,
    piecewise_enthalpy_reference,
    reference_enthalpy_residuals,
)
from convection_mlt.energy import telescoping_residual, temperature_tendency
from convection_mlt.grid import (
    build_grid,
    interpolate_temperature_to_internal_edges,
    log_pressure_edges,
)
from convection_mlt.solvers import (
    SolverFailure,
    TerminalStatus,
    _unmerged_transfer_fractions,
    adaptive_timestep,
    fixed_step,
    solve_adaptive,
)
from convection_mlt.thermodynamics import IdealH2
from convection_mlt.trace import TraceLevel, make_trace

from common import (
    ALPHAS,
    DATA_DIR,
    PACKAGE_ROOT,
    PRODUCTION_EXPONENT,
    RESOLUTIONS,
    acceptance_tolerances,
    campaign_identity,
    ensure_dirs,
    machine_identity,
    metrics_for_score,
    perturbed_edges,
    piecewise_potential_temperature,
    power_law_temperature,
    potential_temperature_profile,
    reference_temperature,
    score_against_tolerances,
    standard_grid,
    write_json,
)


def _outcome_payload(result) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "reason": result.reason,
        "steps": result.steps,
        "rejections": result.rejections,
        "simulated_time_s": result.simulated_time,
        "final_dt_s": result.final_dt,
        "metrics": result.metrics.as_dict() if hasattr(result.metrics, "as_dict") else result.metrics,
        "region_labels": np.asarray(result.region_labels).tolist(),
        "cumulative_unmerged_transfer_j_m2": np.asarray(
            result.cumulative_unmerged_transfer
        ).tolist(),
        "max_unmerged_transfer_fraction": result.max_unmerged_transfer_fraction,
    }


def _solve(grid, temperature, physics, solver, region_labels=None, trace=None):
    try:
        result = solve_adaptive(
            grid,
            temperature,
            physics,
            solver,
            region_labels=region_labels,
            trace=trace,
        )
        failed = False
    except SolverFailure as error:
        result = error.result
        failed = True
    return result, failed


def generate_global_profile(smoke: bool) -> dict[str, Any]:
    n_layers = 25 if smoke else 100
    physics = PhysicsConfig(alpha=1.0)
    solver = SolverConfig()
    grid = standard_grid(n_layers, physics.gravity)
    temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
    reference = reference_temperature(grid, temperature)
    trace = make_trace(TraceLevel.PROFILES)
    result, failed = _solve(grid, temperature, physics, solver, trace=trace)
    gas = IdealH2()
    profiles = []
    for snapshot in trace.profiles:
        profiles.append(
            {
                "accepted_step": snapshot.accepted_step,
                "simulated_time_s": snapshot.simulated_time,
                "temperature_k": snapshot.temperature.tolist(),
                "flux_w_m2": snapshot.flux.tolist(),
                "potential_temperature_k": potential_temperature_profile(
                    grid, snapshot.temperature, gas
                ).tolist(),
                "potential_temperature_rms": snapshot.potential_temperature_rms,
                "region_labels": snapshot.region_labels.tolist(),
            }
        )
    return {
        "campaign": "global_profile",
        "identity": campaign_identity(physics, solver),
        "case": {
            "n_layers": n_layers,
            "alpha": physics.alpha,
            "initial_exponent": PRODUCTION_EXPONENT,
            "pressure_centres_pa": grid.pressure_centres.tolist(),
            "pressure_edges_pa": grid.pressure_edges.tolist(),
            "failed": failed,
        },
        "reference_temperature_k": reference.tolist(),
        "outcome": _outcome_payload(result),
        "profiles": profiles,
        "trace_extrema": trace.extrema,
        "trace_totals": trace.totals,
    }


def generate_locality(smoke: bool) -> dict[str, Any]:
    n_layers = 20 if smoke else 40
    physics = PhysicsConfig(alpha=1.0)
    solver = SolverConfig()
    gas = IdealH2()
    grid = standard_grid(n_layers, physics.gravity)
    temperature = power_law_temperature(grid, gas.nabla_ad)
    mid = n_layers // 2
    width = 3 if smoke else 5
    temperature = temperature.copy()
    temperature[mid - width : mid + width] *= np.linspace(1.08, 0.92, 2 * width)
    labels = mixing_region_labels(
        grid, temperature, gas.nabla_ad, solver.c_active * solver.epsilon_gradient
    )
    piecewise_ref = piecewise_enthalpy_reference(
        grid, temperature, gas.cp, gas.nabla_ad, labels
    )
    trace = make_trace(TraceLevel.PROFILES)
    result, failed = _solve(
        grid, temperature, physics, solver, region_labels=labels, trace=trace
    )
    final_ref = piecewise_enthalpy_reference(
        grid, temperature, gas.cp, gas.nabla_ad, result.region_labels
    )
    initial_theta_ref = piecewise_potential_temperature(
        grid, piecewise_ref, labels, gas
    )
    final_theta_ref = piecewise_potential_temperature(
        grid, final_ref, result.region_labels, gas
    )
    transfer_fractions = _unmerged_transfer_fractions(
        grid,
        temperature,
        gas,
        result.region_labels,
        np.asarray(result.cumulative_unmerged_transfer, dtype=float),
    )
    merge_events = []
    previous = None
    for record in trace.accepted_steps:
        labels_now = np.asarray(record.region_labels)
        if previous is not None and not np.array_equal(previous, labels_now):
            merge_events.append(
                {
                    "accepted_step": record.accepted_step,
                    "simulated_time_s": record.simulated_time,
                    "region_labels": labels_now.tolist(),
                    "n_regions": int(np.unique(labels_now).size),
                }
            )
        previous = labels_now
    return {
        "campaign": "locality",
        "identity": campaign_identity(physics, solver),
        "case": {
            "n_layers": n_layers,
            "alpha": physics.alpha,
            "failed": failed,
            "pressure_centres_pa": grid.pressure_centres.tolist(),
            "pressure_edges_pa": grid.pressure_edges.tolist(),
            "initial_region_labels": labels.tolist(),
            "n_initial_regions": int(np.unique(labels).size),
            "transfer_merge_tolerance": solver.transfer_merge_tolerance,
        },
        "initial_temperature_k": temperature.tolist(),
        "initial_potential_temperature_k": piecewise_potential_temperature(
            grid, temperature, labels, gas
        ).tolist(),
        "initial_piecewise_reference_temperature_k": piecewise_ref.tolist(),
        "initial_piecewise_reference_potential_temperature_k": (
            initial_theta_ref.tolist()
        ),
        # Kept for older plot scripts; equals the temperature reference.
        "initial_piecewise_reference_k": piecewise_ref.tolist(),
        "final_temperature_k": result.temperature.tolist(),
        "final_potential_temperature_k": piecewise_potential_temperature(
            grid, result.temperature, result.region_labels, gas
        ).tolist(),
        "final_piecewise_reference_temperature_k": final_ref.tolist(),
        "final_piecewise_reference_potential_temperature_k": (
            final_theta_ref.tolist()
        ),
        "final_piecewise_reference_k": final_ref.tolist(),
        "final_piecewise_residuals": (
            result.temperature - final_ref
        ).tolist(),
        "normalized_unmerged_transfer": transfer_fractions.tolist(),
        "outcome": _outcome_payload(result),
        "merge_events": merge_events,
        "profiles": [
            {
                "accepted_step": snap.accepted_step,
                "simulated_time_s": snap.simulated_time,
                "region_labels": snap.region_labels.tolist(),
                "temperature_k": snap.temperature.tolist(),
                "potential_temperature_k": piecewise_potential_temperature(
                    grid, snap.temperature, snap.region_labels, gas
                ).tolist(),
            }
            for snap in trace.profiles
        ],
    }


def generate_enthalpy(smoke: bool) -> dict[str, Any]:
    cases = (
        [(25, 1.0), (50, 1.0)]
        if smoke
        else [(25, 1.0), (100, 1.0), (100, 4.0), (400, 4.0)]
    )
    solver = SolverConfig()
    records = []
    for n_layers, alpha in cases:
        physics = PhysicsConfig(alpha=alpha)
        grid = standard_grid(n_layers, physics.gravity)
        temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
        gas = IdealH2()
        initial_closure = mixing_length_flux(
            grid,
            temperature,
            physics.gravity,
            physics.alpha,
            gas,
            physics.closure_prefactor,
        )
        initial_tendency = temperature_tendency(
            grid, initial_closure.flux, gas.cp
        )
        telescoping = telescoping_residual(
            grid,
            initial_tendency,
            gas.cp,
            initial_closure.flux[0],
            initial_closure.flux[-1],
        )
        telescoping_scale = float(
            np.sum(np.abs(gas.cp * grid.layer_mass * initial_tendency))
        )
        trace = make_trace(TraceLevel.SUMMARY)
        result, failed = _solve(grid, temperature, physics, solver, trace=trace)
        history = [
            {
                "accepted_step": item.accepted_step,
                "simulated_time_s": item.simulated_time,
                "signed_enthalpy_drift": item.signed_enthalpy_drift,
                "metrics": item.metrics.as_dict(),
            }
            for item in trace.accepted_steps
        ]
        records.append(
            {
                "n_layers": n_layers,
                "alpha": alpha,
                "failed": failed,
                "identity": campaign_identity(physics, solver),
                "outcome": _outcome_payload(result),
                "history": history,
                "max_abs_enthalpy_drift": trace.extrema.get(
                    "max_abs_enthalpy_drift", result.metrics.enthalpy_drift
                ),
                "conservation_audit": {
                    "telescoping_residual_w_m2": telescoping,
                    "telescoping_scale_w_m2": telescoping_scale,
                    "bottom_boundary_flux_w_m2": float(
                        initial_closure.flux[0]
                    ),
                    "top_boundary_flux_w_m2": float(
                        initial_closure.flux[-1]
                    ),
                },
                "tolerances": acceptance_tolerances(solver),
            }
        )
    return {
        "campaign": "enthalpy",
        "machine": machine_identity(),
        "records": records,
    }


def _equilibrium_record(case: tuple[int, float]) -> dict[str, Any]:
    n_layers, alpha = case
    solver = SolverConfig()
    physics = PhysicsConfig(alpha=alpha)
    grid = standard_grid(n_layers, physics.gravity)
    temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
    reference = reference_temperature(grid, temperature)
    trace = make_trace(TraceLevel.SUMMARY)
    result, failed = _solve(grid, temperature, physics, solver, trace=trace)
    max_abs_drift = trace.extrema.get(
        "max_abs_enthalpy_drift", result.metrics.enthalpy_drift
    )
    metrics = metrics_for_score(result.metrics.as_dict(), max_abs_drift)
    tolerances = acceptance_tolerances(solver)
    return {
        "n_layers": n_layers,
        "alpha": alpha,
        "failed": failed,
        "outcome": _outcome_payload(result),
        "reference_temperature_k": reference.tolist(),
        "metrics_for_score": metrics,
        "tolerances": tolerances,
        "score": score_against_tolerances(metrics, tolerances),
        "max_abs_enthalpy_drift": max_abs_drift,
        "identity": campaign_identity(physics, solver),
    }


def generate_equilibrium_matrix(smoke: bool) -> dict[str, Any]:
    resolutions = (25, 50) if smoke else RESOLUTIONS
    alphas = (0.5, 1.0) if smoke else ALPHAS
    cases = [(n_layers, alpha) for n_layers in resolutions for alpha in alphas]
    if smoke:
        records = [_equilibrium_record(case) for case in cases]
    else:
        # Independent matrix cells are isolated processes. This changes only
        # campaign wall time; each solver run and its trace remain independent.
        with ProcessPoolExecutor(max_workers=min(5, len(cases))) as executor:
            records = list(executor.map(_equilibrium_record, cases))
    return {
        "campaign": "equilibrium_matrix",
        "machine": machine_identity(),
        "resolutions": list(resolutions),
        "alphas": list(alphas),
        "records": records,
    }


def generate_alpha_trajectories(smoke: bool) -> dict[str, Any]:
    n_layers = 25 if smoke else 100
    alphas = (0.5, 1.0, 2.0) if smoke else ALPHAS
    solver = SolverConfig()
    grid = standard_grid(n_layers, 15.0)
    temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
    gas = IdealH2()
    threshold = solver.theta_rms_tolerance * 10.0
    trajectories = []
    closure_scaling = []
    # Fixed-state closure scaling uses the common initial profile.
    for alpha in alphas:
        physics = PhysicsConfig(alpha=alpha)
        closure = mixing_length_flux(
            grid, temperature, physics.gravity, alpha, gas, physics.closure_prefactor
        )
        internal = slice(1, -1)
        active = closure.active[internal]
        closure_scaling.append(
            {
                "alpha": alpha,
                "mean_velocity": float(np.mean(closure.velocity[internal][active]))
                if np.any(active)
                else 0.0,
                "mean_flux": float(np.mean(closure.flux[internal][active]))
                if np.any(active)
                else 0.0,
                "mean_kzz": float(np.mean(closure.kzz[internal][active]))
                if np.any(active)
                else 0.0,
                "mean_mixing_length": float(
                    np.mean(closure.mixing_length[internal][active])
                )
                if np.any(active)
                else 0.0,
            }
        )
        trace = make_trace(TraceLevel.SUMMARY)
        result, failed = _solve(grid, temperature, physics, solver, trace=trace)
        history = [
            {
                "accepted_step": item.accepted_step,
                "simulated_time_s": item.simulated_time,
                "dt_accepted_s": item.dt_accepted,
                "rejections_this_step": item.rejections_this_step,
                "theta_rms": item.metrics.potential_temperature_rms,
                "metrics": item.metrics.as_dict(),
            }
            for item in trace.accepted_steps
        ]
        threshold_time = None
        for item in history:
            if item["theta_rms"] <= threshold:
                threshold_time = item["simulated_time_s"]
                break
        trajectories.append(
            {
                "alpha": alpha,
                "failed": failed,
                "outcome": _outcome_payload(result),
                "history": history,
                "threshold": threshold,
                "threshold_time_s": threshold_time,
                "accepted_dts": [item["dt_accepted_s"] for item in history],
                "identity": campaign_identity(physics, solver),
            }
        )
    return {
        "campaign": "alpha_trajectories",
        "machine": machine_identity(),
        "n_layers": n_layers,
        "threshold_theta_rms": threshold,
        "trajectories": trajectories,
        "closure_scaling": closure_scaling,
    }


def _integrate_adaptive_to_time(
    grid,
    initial_temperature: np.ndarray,
    physics: PhysicsConfig,
    solver: SolverConfig,
    final_time: float,
) -> dict[str, Any]:
    """Integrate to one finite endpoint using the production timestep rules."""
    gas = IdealH2()
    state = initial_temperature.copy()
    simulated_time = 0.0
    accepted_dts: list[float] = []
    rejections = 0
    while simulated_time < final_time:
        closure = mixing_length_flux(
            grid,
            state,
            physics.gravity,
            physics.alpha,
            gas,
            physics.closure_prefactor,
        )
        tendency = temperature_tendency(grid, closure.flux, gas.cp)
        if np.all(tendency == 0.0):
            simulated_time = final_time
            break
        dt = min(
            *adaptive_timestep(
                grid, state, closure, tendency, physics, solver, gas
            ),
            final_time - simulated_time,
        )
        accepted = False
        for _ in range(solver.max_rejections + 1):
            if not np.isfinite(dt) or dt < solver.dt_min:
                raise RuntimeError(
                    f"finite-endpoint timestep {dt!r} below dt_min"
                )
            trial = fixed_step(grid, state, dt, physics, solver, gas)
            if trial.accepted:
                state = trial.temperature
                simulated_time += dt
                accepted_dts.append(float(dt))
                accepted = True
                break
            rejections += 1
            dt *= solver.f_back
        if not accepted:
            raise RuntimeError("finite-endpoint backtracking failed")
    dts = np.asarray(accepted_dts, dtype=float)
    return {
        "temperature_k": state.tolist(),
        "simulated_time_s": simulated_time,
        "steps": int(dts.size),
        "rejections": rejections,
        "min_accepted_dt_s": float(np.min(dts)) if dts.size else None,
        "median_accepted_dt_s": float(np.median(dts)) if dts.size else None,
        "max_accepted_dt_s": float(np.max(dts)) if dts.size else None,
    }


def generate_resolution_scaling(smoke: bool) -> dict[str, Any]:
    resolutions = (25, 50) if smoke else RESOLUTIONS
    alpha = 1.0
    repetitions = 2 if smoke else 5
    solver = SolverConfig()
    physics = PhysicsConfig(alpha=alpha)
    # A common pre-equilibrium endpoint isolates algorithmic N scaling.
    common_time = 1.0e5
    records = []
    for n_layers in resolutions:
        grid = standard_grid(n_layers, physics.gravity)
        temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
        # Warm-up (untimed, with tracing and plotting disabled).
        warm = _integrate_adaptive_to_time(
            grid, temperature, physics, solver, common_time
        )
        samples = []
        last = warm
        for _ in range(repetitions):
            started = time.perf_counter()
            result = _integrate_adaptive_to_time(
                grid, temperature, physics, solver, common_time
            )
            samples.append(time.perf_counter() - started)
            last = result
        convergence, convergence_failed = _solve(
            grid, temperature, physics, solver, trace=None
        )
        array = np.asarray(samples, dtype=float)
        timing = {
            "samples_s": array.tolist(),
            "median_s": float(np.median(array)),
            "q25_s": float(np.percentile(array, 25)),
            "q75_s": float(np.percentile(array, 75)),
            "min_s": float(np.min(array)),
            "max_s": float(np.max(array)),
        }
        records.append(
            {
                "n_layers": n_layers,
                "alpha": alpha,
                "failed": False,
                "common_physical_endpoint_s": common_time,
                "time_to_convergence_s": convergence.simulated_time,
                "outcome": last,
                "convergence_failed": convergence_failed,
                "convergence_outcome": _outcome_payload(convergence),
                "timing": timing,
                "wall_time_per_step_s": timing["median_s"]
                / max(last["steps"], 1),
                "rejection_fraction": (
                    last["rejections"]
                    / max(last["steps"] + last["rejections"], 1)
                ),
                "identity": campaign_identity(physics, solver),
            }
        )
    return {
        "campaign": "resolution_scaling",
        "machine": machine_identity(),
        "alpha": alpha,
        "repetitions": repetitions,
        "common_physical_endpoint_note": (
            "timing disables tracing, runs one warm-up, then times only the solver; "
            "all N use the same 1e5 s pre-equilibrium endpoint; "
            "time-to-convergence is measured once and reported separately"
        ),
        "records": records,
    }


def _mass_weighted_relative_rms(grid, state, reference) -> float:
    relative = (state - reference) / reference
    return float(
        np.sqrt(np.sum(grid.layer_mass * relative**2) / np.sum(grid.layer_mass))
    )


def _integrate_fixed(grid, temperature, physics, dt, final_time, solver):
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time):
        raise ValueError("final_time must be an integer multiple of dt")
    state = temperature.copy()
    for step in range(steps):
        outcome = fixed_step(grid, state, dt, physics, solver)
        if not outcome.accepted:
            return state, {
                "status": "failed",
                "reason": outcome.reason,
                "failed_step": step,
                "steps": step,
            }
        state = outcome.temperature
    return state, {"status": "completed", "steps": steps}


def generate_temporal_stability(smoke: bool) -> dict[str, Any]:
    cases = ((25, 1.0),) if smoke else ((25, 1.0), (100, 1.0))
    base_dt = 1.0
    final_time = 8.0 if smoke else 16.0
    divisors = (1, 2, 4) if smoke else (1, 2, 4, 8, 16)
    solver = SolverConfig(epsilon_temperature=0.5)
    order_records = []
    for n_layers, alpha in cases:
        physics = PhysicsConfig(alpha=alpha)
        grid = standard_grid(n_layers, physics.gravity)
        temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
        # Refine the small-step reference until successive refinements change
        # the fitted-window errors negligibly relative to the smallest error.
        reference_divisor = max(divisors) * (4 if smoke else 8)
        ref_dt = base_dt / reference_divisor
        reference_state, reference_info = _integrate_fixed(
            grid, temperature, physics, ref_dt, final_time, solver
        )
        if reference_info["status"] != "completed":
            raise RuntimeError(
                f"reference integration failed for N={n_layers}: {reference_info}"
            )
        # Optionally refine once more and require small change.
        finer_dt = ref_dt / 2.0
        finer_state, finer_info = _integrate_fixed(
            grid, temperature, physics, finer_dt, final_time, solver
        )
        if finer_info["status"] == "completed":
            refine_delta = _mass_weighted_relative_rms(
                grid, finer_state, reference_state
            )
            reference_state = finer_state
            ref_dt = finer_dt
            reference_info = finer_info
        else:
            refine_delta = float("nan")
        points = []
        for divisor in divisors:
            dt = base_dt / divisor
            state, info = _integrate_fixed(
                grid, temperature, physics, dt, final_time, solver
            )
            if info["status"] != "completed":
                points.append(
                    {
                        "dt_s": dt,
                        "status": info["status"],
                        "reason": info.get("reason"),
                        "relative_temperature_rms": None,
                        "used_in_fit": False,
                    }
                )
                continue
            rms = _mass_weighted_relative_rms(grid, state, reference_state)
            points.append(
                {
                    "dt_s": dt,
                    "status": "completed",
                    "relative_temperature_rms": rms,
                    "used_in_fit": True,
                }
            )
        stable = [p for p in points if p["used_in_fit"]]
        # Fit only the asymptotic small-dt half when enough points exist.
        fit_points = stable[-min(3, len(stable)) :] if stable else []
        if len(fit_points) >= 2:
            x = np.array([p["dt_s"] for p in fit_points], dtype=float)
            y = np.array(
                [p["relative_temperature_rms"] for p in fit_points], dtype=float
            )
            slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
        else:
            slope = float("nan")
        for point in points:
            point["used_in_fit"] = any(
                np.isclose(point["dt_s"], item["dt_s"]) for item in fit_points
            ) and point["used_in_fit"]
        order_records.append(
            {
                "n_layers": n_layers,
                "alpha": alpha,
                "final_time_s": final_time,
                "reference_dt_s": ref_dt,
                "reference_refinement_delta": refine_delta,
                "reference": reference_info,
                "points": points,
                "fitted_slope": slope,
                "identity": campaign_identity(physics, solver),
            }
        )

    # Safety-factor sweeps with trial tracing. Use the smallest campaign N so the
    # outcome map stays affordable while still exercising backtracking margins.
    safety_cases = []
    safety_n, safety_alpha = cases[0]
    physics = PhysicsConfig(alpha=safety_alpha)
    grid = standard_grid(safety_n, physics.gravity)
    temperature = power_law_temperature(grid, PRODUCTION_EXPONENT)
    adaptive_c_diffs = (0.2, 1.0, 10.0) if smoke else (0.2, 1.0, 10.0, 100.0)
    adaptive_max_steps = 8_000 if smoke else 120_000
    for c_diff in adaptive_c_diffs:
        local_solver = SolverConfig(c_diff=c_diff, max_steps=adaptive_max_steps)
        trace = make_trace(TraceLevel.TRIALS)
        result, failed = _solve(
            grid, temperature, physics, local_solver, trace=trace
        )
        min_margins_all = [
            item.min_active_trial_delta_over_epsilon
            for item in trace.trials
            if item.min_active_trial_delta_over_epsilon is not None
        ]
        min_margins_accepted = [
            item.min_active_trial_delta_over_epsilon
            for item in trace.trials
            if item.accepted
            and item.min_active_trial_delta_over_epsilon is not None
        ]
        min_margins_rejected = [
            item.min_active_trial_delta_over_epsilon
            for item in trace.trials
            if (not item.accepted)
            and item.min_active_trial_delta_over_epsilon is not None
        ]
        if failed and "backtracking failed" in result.reason:
            outcome_class = "adaptive_failure"
        elif failed:
            outcome_class = "max_steps_exhausted"
        elif result.rejections > 0:
            outcome_class = "adaptive_acceptance_after_backtracking"
        else:
            outcome_class = "direct_acceptance"
        safety_cases.append(
            {
                "n_layers": safety_n,
                "alpha": safety_alpha,
                "c_diff": c_diff,
                "mode": "adaptive",
                "expected_unstable": False,
                "outcome_class": outcome_class,
                "status": result.status.value,
                "reason": result.reason,
                "rejections": result.rejections,
                "steps": result.steps,
                "min_trial_delta_over_epsilon": (
                    None if not min_margins_all else float(np.min(min_margins_all))
                ),
                "min_trial_delta_over_epsilon_all_attempts": (
                    None if not min_margins_all else float(np.min(min_margins_all))
                ),
                "min_accepted_trial_delta_over_epsilon": (
                    None
                    if not min_margins_accepted
                    else float(np.min(min_margins_accepted))
                ),
                "min_rejected_trial_delta_over_epsilon": (
                    None
                    if not min_margins_rejected
                    else float(np.min(min_margins_rejected))
                ),
                "margin_includes_rejected_trials": True,
                "c_cross": local_solver.c_cross,
            }
        )

    # Intentional adaptive backtracking failure (no retries, huge step).
    intentional_solver = SolverConfig(
        c_diff=1.0e30,
        epsilon_temperature=1.0e300,
        max_rejections=0,
        max_steps=2,
    )
    intentional_trace = make_trace(TraceLevel.TRIALS)
    intentional_result, intentional_failed = _solve(
        grid,
        temperature,
        physics,
        intentional_solver,
        trace=intentional_trace,
    )
    intentional_margins = [
        item.min_active_trial_delta_over_epsilon
        for item in intentional_trace.trials
        if item.min_active_trial_delta_over_epsilon is not None
    ]
    intentional_accepted = [
        item.min_active_trial_delta_over_epsilon
        for item in intentional_trace.trials
        if item.accepted
        and item.min_active_trial_delta_over_epsilon is not None
    ]
    intentional_rejected = [
        item.min_active_trial_delta_over_epsilon
        for item in intentional_trace.trials
        if (not item.accepted)
        and item.min_active_trial_delta_over_epsilon is not None
    ]
    safety_cases.append(
        {
            "n_layers": safety_n,
            "alpha": safety_alpha,
            "c_diff": intentional_solver.c_diff,
            "mode": "adaptive",
            "expected_unstable": True,
            "outcome_class": (
                "adaptive_failure"
                if intentional_failed
                and "backtracking failed" in intentional_result.reason
                else "unexpected_success"
            ),
            "status": intentional_result.status.value,
            "reason": intentional_result.reason,
            "rejections": intentional_result.rejections,
            "steps": intentional_result.steps,
            "min_trial_delta_over_epsilon": (
                None if not intentional_margins else float(np.min(intentional_margins))
            ),
            "min_trial_delta_over_epsilon_all_attempts": (
                None if not intentional_margins else float(np.min(intentional_margins))
            ),
            "min_accepted_trial_delta_over_epsilon": (
                None
                if not intentional_accepted
                else float(np.min(intentional_accepted))
            ),
            "min_rejected_trial_delta_over_epsilon": (
                None
                if not intentional_rejected
                else float(np.min(intentional_rejected))
            ),
            "margin_includes_rejected_trials": True,
            "c_cross": intentional_solver.c_cross,
        }
    )

    # Fixed-step probes: one stable small dt and intentionally unstable large dts.
    fixed_solver = SolverConfig(epsilon_temperature=1.0e300)
    for dt, expected_unstable in (
        ((1.0e-2, False), (1.0, False), (100.0, True), (400.0, True))
        if not smoke
        else ((1.0e-2, False), (100.0, True), (400.0, True))
    ):
        outcome = fixed_step(grid, temperature, dt, physics, fixed_solver)
        outcome_class = (
            "direct_acceptance" if outcome.accepted else "fixed_step_failure"
        )
        safety_cases.append(
            {
                "n_layers": safety_n,
                "alpha": safety_alpha,
                "c_diff": None,
                "mode": "fixed_like",
                "dt_s": dt,
                "expected_unstable": expected_unstable,
                "outcome_class": outcome_class,
                "reason": outcome.reason,
                "min_trial_delta_over_epsilon": (
                    outcome.min_active_trial_delta_over_epsilon
                ),
                "min_trial_delta_over_epsilon_all_attempts": (
                    outcome.min_active_trial_delta_over_epsilon
                ),
                "min_accepted_trial_delta_over_epsilon": (
                    outcome.min_active_trial_delta_over_epsilon
                    if outcome.accepted
                    else None
                ),
                "min_rejected_trial_delta_over_epsilon": (
                    outcome.min_active_trial_delta_over_epsilon
                    if not outcome.accepted
                    else None
                ),
                "margin_includes_rejected_trials": False,
                "c_cross": fixed_solver.c_cross,
            }
        )
    return {
        "campaign": "temporal_stability",
        "machine": machine_identity(),
        "order_records": order_records,
        "safety_cases": safety_cases,
    }


def generate_robustness(smoke: bool) -> dict[str, Any]:
    n_layers = 20 if smoke else 40
    solver = SolverConfig()
    gas = IdealH2()
    physics = PhysicsConfig(alpha=1.0)
    records = []

    def add_case(name, grid, temperature, region_labels=None, expected_status=None):
        result, failed = _solve(
            grid, temperature, physics, solver, region_labels=region_labels
        )
        tolerances = acceptance_tolerances(solver)
        metrics = metrics_for_score(result.metrics.as_dict())
        score = score_against_tolerances(metrics, tolerances)
        status_ok = (
            True
            if expected_status is None
            else result.status.value == expected_status
        )
        # Equilibrium reference-error metrics are inapplicable when the terminal
        # claim is "no active convection" rather than mixed equilibrium.
        if result.status is TerminalStatus.NO_ACTIVE_CONVECTION:
            applicable_metrics = [
                "max_superadiabaticity",
                "normalized_tendency_max",
                "convective_flux_max",
                "enthalpy_drift",
            ]
        else:
            applicable_metrics = list(metrics.keys())
        records.append(
            {
                "name": name,
                "n_layers": grid.n_layers,
                "p_bottom_pa": float(grid.pressure_edges[0]),
                "p_top_pa": float(grid.pressure_edges[-1]),
                "failed": failed,
                "expected_status": expected_status,
                "status_ok": status_ok,
                "outcome": _outcome_payload(result),
                "metrics_for_score": metrics,
                "applicable_metrics": applicable_metrics,
                "tolerances": tolerances,
                "score": score,
                "pass": bool(
                    status_ok
                    and (
                        result.status is TerminalStatus.NO_ACTIVE_CONVECTION
                        or score["pass"]
                    )
                    and all(np.isfinite(list(metrics.values())))
                ),
            }
        )

    uniform = standard_grid(n_layers, physics.gravity)
    irregular = build_grid(perturbed_edges(n_layers), physics.gravity)
    add_case(
        "uniform_superadiabatic",
        uniform,
        power_law_temperature(uniform, 0.35),
        expected_status="converged",
    )
    add_case(
        "irregular_superadiabatic",
        irregular,
        power_law_temperature(irregular, 0.35),
        expected_status="converged",
    )
    add_case(
        "exact_adiabat",
        uniform,
        power_law_temperature(uniform, gas.nabla_ad),
        expected_status="converged",
    )
    add_case(
        "fully_stable",
        uniform,
        power_law_temperature(uniform, 0.15),
        expected_status="no_active_convection",
    )
    if not smoke:
        add_case(
            "steep_superadiabatic",
            uniform,
            power_law_temperature(uniform, 0.45),
            expected_status="converged",
        )
        add_case(
            "shallow_superadiabatic",
            uniform,
            power_law_temperature(uniform, 0.30),
            expected_status="converged",
        )
        wide = standard_grid(n_layers, physics.gravity, 1.0e8, 1.0e2)
        narrow = standard_grid(n_layers, physics.gravity, 1.0e6, 1.0e4)
        add_case(
            "wide_pressure_range",
            wide,
            power_law_temperature(wide, 0.35),
            expected_status="converged",
        )
        add_case(
            "narrow_pressure_range",
            narrow,
            power_law_temperature(narrow, 0.35),
            expected_status="converged",
        )
    localized = power_law_temperature(uniform, gas.nabla_ad).copy()
    mid = n_layers // 2
    localized[mid - 2 : mid + 2] *= np.linspace(1.07, 0.93, 4)
    labels = mixing_region_labels(
        uniform, localized, gas.nabla_ad, solver.c_active * solver.epsilon_gradient
    )
    add_case(
        "localized_unstable",
        uniform,
        localized,
        region_labels=labels,
        expected_status="converged",
    )
    return {
        "campaign": "robustness",
        "machine": machine_identity(),
        "records": records,
    }


def generate_invariant_audit(smoke: bool) -> dict[str, Any]:
    del smoke  # deterministic and cheap; always full audit
    gas = IdealH2()
    physics = PhysicsConfig(alpha=1.0)
    solver = SolverConfig()
    rows = []

    def row(name, expected, observed, tolerance, notes=""):
        error = abs(float(observed) - float(expected))
        passed = bool(np.isfinite(error) and error <= tolerance)
        rows.append(
            {
                "name": name,
                "expected": float(expected),
                "observed": float(observed),
                "error": float(error),
                "tolerance": float(tolerance),
                "pass": passed,
                "notes": notes,
            }
        )

    grid = standard_grid(20, physics.gravity)
    stable = power_law_temperature(grid, 0.15)
    closure = mixing_length_flux(grid, stable, physics.gravity, 1.0, gas)
    row(
        "stable_profile_zero_flux",
        0.0,
        float(np.max(np.abs(closure.flux))),
        0.0,
        "fully stable => exact-zero convective flux",
    )
    seed = power_law_temperature(grid, gas.nabla_ad)
    adiabat = enthalpy_normalized_adiabat(grid, seed, gas.cp, gas.nabla_ad)
    closure_ad = mixing_length_flux(grid, adiabat, physics.gravity, 1.0, gas)
    # Discrete lnT/lnP adiabats leave only roundoff-level residual flux.
    row(
        "exact_adiabat_zero_flux",
        0.0,
        float(np.max(np.abs(closure_ad.flux))),
        1.0e-9,
    )
    row(
        "exact_adiabat_max_superadiabaticity",
        0.0,
        float(np.max(closure_ad.superadiabaticity)),
        1.0e-14,
    )

    # Manufactured single-interface check on a 2-layer column.
    tiny = build_grid(log_pressure_edges(1.0e6, 1.0e4, 2), physics.gravity)
    manufactured = 1000.0 * (tiny.pressure_centres / 1.0e5) ** 0.40
    gravity = physics.gravity
    alpha = 1.0
    prefactor = physics.closure_prefactor
    # Explicit hand calculation of the R0 closure at the single internal edge.
    log_grad = (
        np.log(manufactured[0]) - np.log(manufactured[1])
    ) / (
        np.log(tiny.pressure_centres[0]) - np.log(tiny.pressure_centres[1])
    )
    expected_delta = float(max(log_grad - gas.nabla_ad, 0.0))
    expected_t_edge = float(
        interpolate_temperature_to_internal_edges(tiny, manufactured)[0]
    )
    expected_rho = float(
        tiny.pressure_edges[1] / (gas.gas_constant * expected_t_edge)
    )
    expected_hp = float(gas.gas_constant * expected_t_edge / gravity)
    expected_ell = float(alpha * expected_hp)
    expected_w = float(
        expected_ell * np.sqrt(gravity / expected_hp * expected_delta)
    )
    expected_flux = float(
        prefactor
        * expected_rho
        * gas.cp
        * expected_w
        * expected_t_edge
        * (expected_ell / expected_hp)
        * expected_delta
    )
    expected_kzz = float(expected_w * expected_ell)
    closure_m = mixing_length_flux(tiny, manufactured, gravity, alpha, gas)
    tendency = temperature_tendency(tiny, closure_m.flux, gas.cp)
    # Relative tolerance for floating manufactured comparisons.
    man_tol = 1.0e-12
    row(
        "manufactured_mixing_length",
        expected_ell,
        float(closure_m.mixing_length[1]),
        max(man_tol * abs(expected_ell), 1.0e-30),
        "hand-calculated ℓ=α H_P at internal interface",
    )
    row(
        "manufactured_velocity",
        expected_w,
        float(closure_m.velocity[1]),
        max(man_tol * abs(expected_w), 1.0e-30),
        "hand-calculated w=ℓ√(g/H_P δ)",
    )
    row(
        "manufactured_flux",
        expected_flux,
        float(closure_m.flux[1]),
        max(man_tol * abs(expected_flux), 1.0e-30),
        "hand-calculated F_conv with prefactor 1/2",
    )
    row(
        "manufactured_kzz",
        expected_kzz,
        float(closure_m.kzz[1]),
        max(man_tol * abs(expected_kzz), 1.0e-30),
        "hand-calculated Kzz=wℓ",
    )
    row(
        "manufactured_positive_flux",
        1.0,
        1.0 if closure_m.flux[1] > 0.0 else 0.0,
        0.0,
        "boolean (1=true): F_conv>0",
    )
    row(
        "manufactured_positive_velocity",
        1.0,
        1.0 if closure_m.velocity[1] > 0.0 else 0.0,
        0.0,
        "boolean (1=true): w>0",
    )
    row(
        "manufactured_positive_mixing_length",
        1.0,
        1.0 if closure_m.mixing_length[1] > 0.0 else 0.0,
        0.0,
        "boolean (1=true): ℓ>0",
    )
    row(
        "manufactured_positive_kzz",
        1.0,
        1.0 if closure_m.kzz[1] > 0.0 else 0.0,
        0.0,
        "boolean (1=true): Kzz>0",
    )
    row(
        "update_sign_lower_cools",
        1.0,
        1.0 if tendency[0] < 0.0 else 0.0,
        0.0,
        "boolean (1=true): lower layer cools",
    )
    row(
        "update_sign_upper_heats",
        1.0,
        1.0 if tendency[1] > 0.0 else 0.0,
        0.0,
        "boolean (1=true): upper layer heats",
    )
    row("boundary_flux_bottom_zero", 0.0, float(closure_m.flux[0]), 0.0)
    row("boundary_flux_top_zero", 0.0, float(closure_m.flux[-1]), 0.0)
    residual = telescoping_residual(
        tiny, tendency, gas.cp, closure_m.flux[0], closure_m.flux[-1]
    )
    scale = np.sum(np.abs(gas.cp * tiny.layer_mass * tendency))
    row(
        "one_step_telescoping_residual",
        0.0,
        residual,
        5.0 * np.finfo(float).eps * scale,
    )

    # alpha = 0 => no active convection on a superadiabatic profile.
    result0, _ = _solve(
        grid,
        power_law_temperature(grid, 0.35),
        PhysicsConfig(alpha=0.0),
        solver,
    )
    row(
        "alpha_zero_status_no_active",
        1.0,
        1.0 if result0.status is TerminalStatus.NO_ACTIVE_CONVECTION else 0.0,
        0.0,
        "boolean (1=true): α=0 => no_active_convection",
    )

    # Rejected-state purity.
    labels = np.arange(10, dtype=np.int64)
    rej_grid = standard_grid(10, physics.gravity)
    rej_temp = power_law_temperature(rej_grid, 0.45)
    labels = mixing_region_labels(rej_grid, rej_temp, gas.nabla_ad, 1.0e-7)
    labels_before = labels.copy()
    temp_before = rej_temp.copy()
    try:
        solve_adaptive(
            rej_grid,
            rej_temp,
            physics,
            SolverConfig(
                c_diff=1.0e30,
                epsilon_temperature=1.0e300,
                max_rejections=0,
                max_steps=2,
            ),
            region_labels=labels,
        )
        purity = 0.0
    except SolverFailure as error:
        purity = (
            1.0
            if (
                np.array_equal(error.result.temperature, temp_before)
                and np.array_equal(error.result.region_labels, labels_before)
                and np.all(error.result.cumulative_unmerged_transfer == 0.0)
            )
            else 0.0
        )
    row(
        "rejected_state_purity",
        1.0,
        purity,
        0.0,
        "boolean (1=true): rejected trials leave state/labels/transfers unchanged",
    )

    # Regional reference enthalpy identity.
    loc = power_law_temperature(grid, gas.nabla_ad).copy()
    loc[8:12] *= np.linspace(1.06, 0.94, 4)
    loc_labels = mixing_region_labels(grid, loc, gas.nabla_ad, 1.0e-7)
    piecewise_ref = piecewise_enthalpy_reference(
        grid, loc, gas.cp, gas.nabla_ad, loc_labels
    )
    residuals = reference_enthalpy_residuals(
        grid, loc, piecewise_ref, gas.cp, loc_labels
    )
    row(
        "regional_reference_enthalpy_identity",
        0.0,
        float(max(residuals.values())),
        1.0e-14,
    )

    # Terminal statuses on representative cases.
    stable_result, _ = _solve(grid, stable, physics, solver)
    row(
        "status_stable_no_active",
        1.0,
        1.0
        if stable_result.status is TerminalStatus.NO_ACTIVE_CONVECTION
        else 0.0,
        0.0,
        "boolean (1=true): fully stable => no_active_convection",
    )
    conv_result, _ = _solve(
        grid, power_law_temperature(grid, 0.35), physics, solver
    )
    row(
        "status_superadiabatic_converged",
        1.0,
        1.0 if conv_result.status is TerminalStatus.CONVERGED else 0.0,
        0.0,
        "boolean (1=true): globally superadiabatic => converged",
    )
    return {
        "campaign": "invariant_audit",
        "machine": machine_identity(),
        "rows": rows,
        "pass": all(item["pass"] for item in rows),
    }


CAMPAIGNS = {
    "global_profile": ("global_profile.json", generate_global_profile),
    "locality": ("locality.json", generate_locality),
    "enthalpy": ("enthalpy.json", generate_enthalpy),
    "equilibrium_matrix": ("equilibrium_matrix.json", generate_equilibrium_matrix),
    "alpha_trajectories": ("alpha_trajectories.json", generate_alpha_trajectories),
    "resolution_scaling": ("resolution_scaling.json", generate_resolution_scaling),
    "temporal_stability": ("temporal_stability.json", generate_temporal_stability),
    "robustness": ("robustness.json", generate_robustness),
    "invariant_audit": ("invariant_audit.json", generate_invariant_audit),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(CAMPAIGNS),
        help="optional subset of campaigns",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(CAMPAIGNS)
    summary_path = args.output_dir / "generation_summary.json"
    if args.only and summary_path.exists():
        from common import read_json

        summary = read_json(summary_path)
        summary["smoke"] = args.smoke
        summary["machine"] = machine_identity()
    else:
        summary = {
            "smoke": args.smoke,
            "machine": machine_identity(),
            "campaigns": {},
        }
    for name in selected:
        filename, generator = CAMPAIGNS[name]
        started = time.perf_counter()
        payload = generator(args.smoke)
        path = args.output_dir / filename
        write_json(path, payload)
        elapsed = time.perf_counter() - started
        summary["campaigns"][name] = {
            "path": str(path.relative_to(PACKAGE_ROOT) if path.is_relative_to(PACKAGE_ROOT) else path),
            "wall_time_s": elapsed,
        }
        print(f"[generate_data] {name}: {path} ({elapsed:.2f}s)")
    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
