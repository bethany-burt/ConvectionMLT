"""Shared flux-residual localization helpers for mixture diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from convection_mlt.gravity import ConstantGravity
from convection_mlt.production_rce import (
    algebraic_identity_residuals,
    build_seed_temperature,
    build_spec,
    production_solver_config,
    production_thermo,
    _gates_from_result,
    _live_solve,
    _reduced,
)
from convection_mlt.radiation import LowerNetInternalFlux, SolveRoute, TopIrradiation
from convection_mlt.rce import _primary_rcb_log10p
from convection_mlt.steady_rce import TrialFluxes, evaluate_trial, flux_flatness_residual


RCB_HALF_WIDTH_DEX = 0.15
UPPER_N_LAYERS = 5


def f_scale_from(f_int: float, floor: float = 1.0e-30) -> float:
    return max(float(floor), abs(float(f_int)))


def layer_mask(n_layers: int, regions: list[tuple[int, int]]) -> NDArray[np.bool_]:
    mask = np.zeros(n_layers, dtype=bool)
    for lo, hi in regions or []:
        mask[int(lo) : int(hi) + 1] = True
    return mask


def interface_residual(
    flux_total: NDArray[np.float64], f_int: float, f_scale: float
) -> NDArray[np.float64]:
    """r_i = (F_total,i - F_int) / F_scale on all N+1 interfaces."""
    f_tot = np.asarray(flux_total, dtype=np.float64)
    return (f_tot - float(f_int)) / float(f_scale)


def zone_residual_stats(
    residual: NDArray[np.float64],
    pressure_centres: NDArray[np.float64],
    pressure_edges: NDArray[np.float64],
    convective_regions: list[tuple[int, int]],
    rcb_log10p: float | None,
    *,
    rcb_half_width: float = RCB_HALF_WIDTH_DEX,
) -> dict[str, Any]:
    """Max and RMS |r| in CZ, RZ, RCB band, and upper boundary."""
    n = int(pressure_centres.size)
    mask_cz = layer_mask(n, convective_regions)
    mask_rz = ~mask_cz
    logp = np.log10(np.maximum(pressure_centres, 1.0e-30))
    if rcb_log10p is not None and np.isfinite(rcb_log10p):
        near_rcb = np.abs(logp - float(rcb_log10p)) <= float(rcb_half_width)
    else:
        near_rcb = np.zeros(n, dtype=bool)
    upper = np.arange(n) >= max(n - UPPER_N_LAYERS, 0)

    # Map layer masks to interior interface indices 1..N-1 (exclude bottom BC i=0).
    idx = np.arange(1, n)
    r_int = np.asarray(residual, dtype=np.float64)
    if r_int.size == n + 1:
        r_layers = r_int[1:]
    elif r_int.size == n:
        r_layers = r_int
    else:
        raise ValueError(f"unexpected residual length {r_int.size} for n={n}")

    def _stats(sel: NDArray[np.bool_]) -> dict[str, float | None]:
        if not np.any(sel):
            return {"max_abs": None, "rms": None, "n": 0}
        vals = r_layers[sel[: r_layers.size]]
        return {
            "max_abs": float(np.max(np.abs(vals))),
            "rms": float(np.sqrt(np.mean(vals**2))),
            "n": int(vals.size),
        }

    return {
        "convective_zone": _stats(mask_cz),
        "radiative_zone": _stats(mask_rz),
        "rcb_vicinity": _stats(near_rcb),
        "upper_boundary": _stats(upper),
        "column": {
            "max_abs": float(np.max(np.abs(r_int))),
            "rms": float(np.sqrt(np.mean(r_int**2))),
            "n": int(r_int.size),
        },
    }


def classify_residual_location(stats: dict[str, Any]) -> str:
    """Heuristic mapping from zone stats to likely cause (advisor table)."""
    cz = stats["convective_zone"]["max_abs"] or 0.0
    rz = stats["radiative_zone"]["max_abs"] or 0.0
    rcb = stats["rcb_vicinity"]["max_abs"] or 0.0
    upper = stats["upper_boundary"]["max_abs"] or 0.0
    col = stats["column"]["max_abs"] or 0.0
    if col <= 0.0:
        return "unknown"
    if upper >= 0.85 * col and upper > max(cz, rz, rcb):
        return "upper_boundary_irradiation_or_discretization"
    if rcb >= 0.85 * col and rcb > max(cz, rz):
        return "rcb_accelerator_active_region_mismatch"
    if cz >= 0.85 * col and cz > rz:
        return "deep_convective_mixture_adiabat_or_mlt"
    if rz >= 0.85 * col and rz > cz:
        return "radiative_zone_slow_mode_or_seed"
    if abs(cz - rz) / max(col, 1.0e-30) < 0.15:
        return "column_wide_preconditioning_or_inconsistent_residual"
    return "mixed_or_transitional"


def irradiation_flux_audit(
    trial: TrialFluxes,
    *,
    f_int: float,
    f_irr: float,
    mass_path: NDArray[np.float64],
) -> dict[str, Any]:
    """Confirm gate uses F_total = F_rad + F_conv; F_irr enters via radiation BC."""
    f_tot = np.asarray(trial.flux_total, dtype=np.float64)
    f_rad = np.asarray(trial.flux_rad, dtype=np.float64)
    f_conv = np.asarray(trial.flux_conv, dtype=np.float64)
    scale = max(abs(float(f_int)), 1.0)
    alg = algebraic_identity_residuals(
        flux_total=f_tot,
        flux_rad=f_rad,
        flux_conv=f_conv,
        mass_path=mass_path,
        f_int=f_int,
    )
    return {
        "f_int_target_W_m2": float(f_int),
        "f_irr_downward_W_m2": float(f_irr),
        "F_total_bottom_W_m2": float(f_tot[0]),
        "F_total_top_W_m2": float(f_tot[-1]),
        "F_rad_top_W_m2": float(f_rad[-1]),
        "F_conv_top_W_m2": float(f_conv[-1]),
        "conserved_net_flux_is_F_int": True,
        "note": (
            "F_irr is imposed as downward irradiation at the top radiative BC; "
            "the discrete gate flatness uses F_total = F_rad + F_conv against F_int "
            "on all interfaces (complete discrete energy flux, not F_int+F_irr)."
        ),
        "residual_definition": "r_i = (F_total,i - F_int) / max(|F_int|, flux_scale_floor)",
        "algebraic_identities": alg,
    }


def snapshot_from_trial(
    phase: str,
    trial: TrialFluxes,
    *,
    f_int: float,
    f_scale: float,
    f_irr: float,
    convective_regions: list[tuple[int, int]],
    rcb_log10p: float | None,
    pressure_centres: NDArray[np.float64],
    pressure_edges: NDArray[np.float64],
) -> dict[str, Any]:
    residual = interface_residual(trial.flux_total, f_int, f_scale)
    stats = zone_residual_stats(
        residual,
        pressure_centres,
        pressure_edges,
        convective_regions,
        rcb_log10p,
    )
    p_edges = np.asarray(pressure_edges, dtype=np.float64)
    if p_edges.size != residual.size:
        p_edges = np.concatenate(
            [np.asarray(pressure_centres, dtype=np.float64), [pressure_centres[-1]]]
        )
    return {
        "phase": phase,
        "flux_flatness": float(trial.flux_flatness),
        "tendency_norm": float(trial.tendency_norm),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "zone_stats": stats,
        "classification": classify_residual_location(stats),
        "irradiation_audit": irradiation_flux_audit(
            trial,
            f_int=f_int,
            f_irr=f_irr,
            mass_path=trial.state.mass_path,
        ),
        "convective_regions": convective_regions,
        "primary_rcb_log10p": rcb_log10p,
        "profile": {
            "log10p_interface": np.log10(np.maximum(p_edges, 1.0e-30)).tolist(),
            "r_i": residual.tolist(),
            "F_total_W_m2": np.asarray(trial.flux_total, dtype=np.float64).tolist(),
            "F_rad_W_m2": np.asarray(trial.flux_rad, dtype=np.float64).tolist(),
            "F_conv_W_m2": np.asarray(trial.flux_conv, dtype=np.float64).tolist(),
        },
    }


def evaluate_temperature(
    *,
    grid,
    temperature: NDArray[np.float64],
    spec,
    solver,
    thermo,
    f_scale: float,
) -> TrialFluxes | None:
    h = np.asarray(thermo.enthalpy(temperature), dtype=np.float64)
    return evaluate_trial(
        grid,
        h,
        spec.physics(),
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        ConstantGravity(spec.gravity),
        f_int=spec.f_int,
        f_scale=f_scale,
        frozen_support=None,
        diffusivity_factor=1.0,
        radiation_route=SolveRoute.THOMAS,
    )


def run_instrumented_production(
    *,
    x_he: float,
    f_irr: float,
    seed: str = "radiative_convective",
    max_recovery_cycles: int = 2,
) -> dict[str, Any]:
    """Mirror production phases and capture flux residuals at each stage."""
    from convection_mlt.production_rce import ProductionControls

    spec = build_spec(n_layers=96, alpha=1.0, f_int=300.0, f_irr=f_irr)
    grid = spec.grid()
    thermo = production_thermo(x_he)
    solver = production_solver_config()
    f_scale = f_scale_from(spec.f_int)
    ctrl = ProductionControls(max_recovery_cycles=max_recovery_cycles)

    t0 = build_seed_temperature(spec, seed, thermo=thermo)
    require_topo = abs(float(f_irr)) <= 1.0e-15
    p_centres = np.asarray(grid.pressure_centres, dtype=np.float64)
    p_edges = np.asarray(grid.pressure_edges, dtype=np.float64)
    phases: list[dict[str, Any]] = []

    def _capture(phase: str, trial: TrialFluxes | None, regions: list[tuple[int, int]] | None):
        if trial is None:
            phases.append({"phase": phase, "error": "trial evaluation failed"})
            return
        regs = regions if regions is not None else []
        rcb = _primary_rcb_log10p(grid, trial.closure, solver)
        phases.append(
            snapshot_from_trial(
                phase,
                trial,
                f_int=spec.f_int,
                f_scale=f_scale,
                f_irr=spec.f_irr,
                convective_regions=regs,
                rcb_log10p=rcb,
                pressure_centres=p_centres,
                pressure_edges=p_edges,
            )
        )

    seed_trial = evaluate_temperature(
        grid=grid, temperature=t0, spec=spec, solver=solver, thermo=thermo, f_scale=f_scale
    )
    _capture("initial_seed", seed_trial, [])

    reduced = _reduced(
        grid=grid,
        t0=t0,
        spec=spec,
        solver=solver,
        thermo=thermo,
        log=None,
        label="diagnostic",
    )
    t_work = (
        np.asarray(reduced.temperature, dtype=np.float64).copy()
        if reduced.improved
        else t0.copy()
    )
    _capture("reduced_rz", reduced.trial, reduced.convective_regions)

    rcb = None
    if reduced.trial is not None:
        rcb = _primary_rcb_log10p(grid, reduced.trial.closure, solver)

    res_first, _cfg = _live_solve(
        grid=grid,
        t0=t_work,
        spec=spec,
        solver=solver,
        thermo=thermo,
        max_steps=1,
        dt_accuracy=ctrl.dt_accuracy_s,
        dt_hold_init=ctrl.dt_hold_init_s,
        previous_rcb=rcb,
        gate=ctrl.gate,
        prescribed_dt=None,
    )
    first_trial = evaluate_temperature(
        grid=grid,
        temperature=res_first.final_state.temperature,
        spec=spec,
        solver=solver,
        thermo=thermo,
        f_scale=f_scale,
    )
    _capture("live_polish_first_accepted", first_trial, res_first.convective_regions)

    res, _cfg = _live_solve(
        grid=grid,
        t0=np.asarray(res_first.final_state.temperature, dtype=np.float64),
        spec=spec,
        solver=solver,
        thermo=thermo,
        max_steps=max(ctrl.max_steps_live_polish - 1, 0),
        dt_accuracy=ctrl.dt_accuracy_s,
        dt_hold_init=ctrl.dt_hold_init_s,
        previous_rcb=res_first.primary_rcb_log10p,
        gate=ctrl.gate,
        prescribed_dt=None,
        simulated_time_init=float(res_first.simulated_time),
    )
    final_trial = evaluate_temperature(
        grid=grid,
        temperature=res.final_state.temperature,
        spec=spec,
        solver=solver,
        thermo=thermo,
        f_scale=f_scale,
    )
    _capture("live_polish_final", final_trial, res.convective_regions)

    gates = _gates_from_result(
        res, spec, gate=ctrl.gate, require_bottom_connected_cz=require_topo
    )

    for cycle in range(int(ctrl.max_recovery_cycles)):
        if gates.convergence_ok and (gates.topology_ok or not require_topo):
            break
        res, _cfg = _live_solve(
            grid=grid,
            t0=np.asarray(res.final_state.temperature, dtype=np.float64),
            spec=spec,
            solver=solver,
            thermo=thermo,
            max_steps=ctrl.max_steps_continuation,
            dt_accuracy=ctrl.continuation_dt_accuracy_s,
            dt_hold_init=min(ctrl.dt_hold_init_s, ctrl.continuation_dt_accuracy_s),
            previous_rcb=res.primary_rcb_log10p,
            gate=ctrl.gate,
            prescribed_dt=None,
            simulated_time_init=float(res.simulated_time),
        )
        cont_trial = evaluate_temperature(
            grid=grid,
            temperature=res.final_state.temperature,
            spec=spec,
            solver=solver,
            thermo=thermo,
            f_scale=f_scale,
        )
        _capture(f"continuation[{cycle}]_final", cont_trial, res.convective_regions)
        gates = _gates_from_result(
            res, spec, gate=ctrl.gate, require_bottom_connected_cz=require_topo
        )
        if gates.convergence_ok and (gates.topology_ok or not require_topo):
            break

        reduced = _reduced(
            grid=grid,
            t0=np.asarray(res.final_state.temperature, dtype=np.float64),
            spec=spec,
            solver=solver,
            thermo=thermo,
            log=None,
            label=f"repolish[{cycle}]",
        )
        _capture(f"repolish[{cycle}]_reduced_rz", reduced.trial, reduced.convective_regions)
        t_work = (
            np.asarray(reduced.temperature, dtype=np.float64).copy()
            if reduced.improved
            else np.asarray(res.final_state.temperature, dtype=np.float64).copy()
        )
        rcb = None
        if reduced.trial is not None:
            rcb = _primary_rcb_log10p(grid, reduced.trial.closure, solver)
        res, _cfg = _live_solve(
            grid=grid,
            t0=t_work,
            spec=spec,
            solver=solver,
            thermo=thermo,
            max_steps=ctrl.max_steps_live_polish,
            dt_accuracy=ctrl.dt_accuracy_s,
            dt_hold_init=ctrl.dt_hold_init_s,
            previous_rcb=rcb,
            gate=ctrl.gate,
            prescribed_dt=None,
            simulated_time_init=float(res.simulated_time),
        )
        rep_trial = evaluate_temperature(
            grid=grid,
            temperature=res.final_state.temperature,
            spec=spec,
            solver=solver,
            thermo=thermo,
            f_scale=f_scale,
        )
        _capture(f"repolish[{cycle}]_final", rep_trial, res.convective_regions)
        gates = _gates_from_result(
            res, spec, gate=ctrl.gate, require_bottom_connected_cz=require_topo
        )

    return {
        "x_he": float(x_he),
        "f_irr": float(f_irr),
        "seed": seed,
        "verdict": "CONVERGED" if gates.convergence_ok and (gates.topology_ok or not require_topo) else "NOT CONVERGED",
        "gates": gates.as_dict,
        "phases": phases,
        "checkpoint_temperature": np.asarray(res.final_state.temperature, dtype=np.float64).tolist(),
    }
