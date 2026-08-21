"""Operator-order Δt refinement and rebuilt point-39 stability/cost bracket."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    PhysicsConfig,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
    solve_adaptive_rce_with_prescribed_external_flux,
)
from convection_mlt.rce import _run_unsplit
from convection_mlt.state import build_column_state


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "plots" / "data"
GATE = 1.0e-3


def _spec(n_layers: int = 24) -> AnalyticOpacityRCESpec:
    n_phot = 8 if n_layers <= 24 else 16 if n_layers <= 48 else 24
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=n_phot)


def _solver() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)


def _base_cfg(**kwargs) -> RCEConfig:
    cfg = dict(
        max_steps=4000,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=GATE,
        tendency_tolerance=GATE,
        temp_change_tolerance=GATE,
        dt_accuracy=2500.0,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10,
            step_tolerance=1e-10,
            newton_residual_tolerance=1e-12,
            newton_step_tolerance=1e-12,
        ),
    )
    cfg.update(kwargs)
    return RCEConfig(**cfg)


def _initial(spec: AnalyticOpacityRCESpec):
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    return grid, thermo, opacity, t0


def _summarize(res, wall: float) -> dict:
    accepted = [d for d in res.diagnostics if d.accepted]
    return {
        "status": res.status.value,
        "reason": res.reason,
        "steps_accepted": res.steps_accepted,
        "rejections": res.rejections,
        "simulated_time": res.simulated_time,
        "wall_time_s": wall,
        "flux_flatness": res.convergence.flux_flatness,
        "tendency_norm": res.convergence.tendency_norm,
        "newton_iterations": int(sum(d.newton_iterations for d in accepted)),
        "mlt_evals": int(sum(d.mlt_evals for d in accepted)),
        "line_search_backtracks": int(sum(d.line_search_backtracks for d in accepted)),
        "primary_rcb_log10p": res.primary_rcb_log10p,
        "detached": res.detached_convective_regions,
    }


def operator_order_refinement(
    *,
    n_layers: int = 24,
    t_final: float = 2.0e4,
    dt0: float = 500.0,
) -> dict:
    """Same t_final with Δt, Δt/2, Δt/4 for four operator routes."""
    spec = _spec(n_layers)
    grid, thermo, opacity, t0 = _initial(spec)
    routes = {
        "unsplit_explicit": RCERoute.UNSPLIT,
        "rad_then_implicit_conv": RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        "implicit_conv_then_rad": RCERoute.SPLIT_IMPLICIT_CONV_THEN_RAD,
        "strang_rad_implicit_conv": RCERoute.SPLIT_STRANG_RAD_IMPLICIT_CONV,
    }
    out: dict = {"t_final": t_final, "dt0": dt0, "routes": {}}
    for name, route in routes.items():
        out["routes"][name] = {}
        for factor in (1.0, 0.5, 0.25):
            dt = dt0 * factor
            cfg = _base_cfg(
                max_steps=int(t_final / dt) + 50,
                t_final=t_final,
                prescribed_dt=dt,
                n_consec=10**9,
                flux_flatness_tolerance=1e-12,
                tendency_tolerance=1e-12,
                temp_change_tolerance=1e-12,
            )
            wall0 = time.perf_counter()
            res = solve_adaptive_rce(
                grid,
                t0,
                spec.physics(),
                _solver(),
                thermo,
                opacity,
                grid.pressure_centres,
                TopIrradiation(spec.f_irr),
                LowerNetInternalFlux(spec.f_int),
                gravity=ConstantGravity(spec.gravity),
                route=route,
                config=cfg,
            )
            wall = time.perf_counter() - wall0
            summary = _summarize(res, wall)
            summary["prescribed_dt"] = dt
            summary["temperature"] = res.final_state.temperature.tolist()
            reached = (
                res.steps_accepted > 0
                and res.simulated_time >= 0.99 * t_final
                and res.status != RCETerminalStatus.PRESCRIBED_DT_REJECTED
                and res.status != RCETerminalStatus.DT_MIN_FAILURE
            )
            summary["reached_t_final"] = reached
            summary["contract"] = "fixed_physical_time"
            if not reached:
                summary["failed"] = True
                summary["failure_reason"] = (
                    f"truncated/failed trajectory: status={res.status.value} "
                    f"t={res.simulated_time} < t_final={t_final}"
                )
            out["routes"][name][str(factor)] = summary
        completed = [
            out["routes"][name][k]
            for k in ("1.0", "0.5", "0.25")
            if out["routes"][name][k].get("reached_t_final")
        ]
        if len(completed) < 3:
            out["routes"][name]["refinement"] = {
                "valid": False,
                "reason": "truncated or failed trajectories excluded from fitted orders",
                "n_completed": len(completed),
            }
        else:
            t1 = np.asarray(out["routes"][name]["1.0"]["temperature"])
            t2 = np.asarray(out["routes"][name]["0.5"]["temperature"])
            t4 = np.asarray(out["routes"][name]["0.25"]["temperature"])
            scale = np.maximum(np.abs(t4), 1.0)
            e_coarse = float(np.max(np.abs(t1 - t2) / scale))
            e_fine = float(np.max(np.abs(t2 - t4) / scale))
            out["routes"][name]["refinement"] = {
                "valid": True,
                "err_dt_vs_dt2": e_coarse,
                "err_dt2_vs_dt4": e_fine,
                "ratio": e_coarse / max(e_fine, 1e-30),
            }
    return out


def _largest_stable_dt(
    *,
    route: RCERoute,
    physics: PhysicsConfig,
    grid,
    t0,
    thermo,
    opacity,
    spec: AnalyticOpacityRCESpec,
    prescribed_f_ext=None,
    t_probe: float = 5.0e3,
    dt_candidates: list[float] | None = None,
) -> dict:
    if dt_candidates is None:
        dt_candidates = [
            1.0e6, 2.0e5, 5.0e4, 2.0e4, 1.0e4, 5.0e3, 2.5e3, 1.0e3, 200.0, 50.0, 10.0, 1.0
        ]
    trials = []
    largest_success = None
    smallest_fail_above = None
    for dt in dt_candidates:
        cfg = _base_cfg(
            max_steps=int(t_probe / dt) + 20,
            t_final=t_probe,
            prescribed_dt=dt,
            n_consec=10**9,
            flux_flatness_tolerance=1e-12,
            tendency_tolerance=1e-12,
            temp_change_tolerance=1e-12,
        )
        wall0 = time.perf_counter()
        if prescribed_f_ext is not None:
            res = solve_adaptive_rce_with_prescribed_external_flux(
                grid,
                t0,
                physics,
                _solver(),
                thermo,
                prescribed_f_ext,
                gravity=ConstantGravity(spec.gravity),
                config=cfg,
            )
        else:
            res = solve_adaptive_rce(
                grid,
                t0,
                physics,
                _solver(),
                thermo,
                opacity,
                grid.pressure_centres,
                TopIrradiation(spec.f_irr),
                LowerNetInternalFlux(spec.f_int),
                gravity=ConstantGravity(spec.gravity),
                route=route,
                config=cfg,
            )
        wall = time.perf_counter() - wall0
        ok = (
            res.status
            in (RCETerminalStatus.CONVERGED, RCETerminalStatus.MAX_STEPS)
            and res.steps_accepted > 0
            and res.simulated_time >= 0.9 * t_probe
            and np.all(np.isfinite(res.final_state.temperature))
            and res.convergence.finite_state
        )
        if res.status in (
            RCETerminalStatus.PRESCRIBED_DT_REJECTED,
            RCETerminalStatus.DT_MIN_FAILURE,
        ):
            ok = False
        entry = {
            "dt": dt,
            "ok": ok,
            "status": res.status.value,
            "reason": res.reason,
            "steps_accepted": res.steps_accepted,
            "rejections": res.rejections,
            "wall_time_s": wall,
            "flux_flatness": res.convergence.flux_flatness,
        }
        trials.append(entry)
        if ok:
            largest_success = entry
            break
        smallest_fail_above = entry
    if largest_success is None:
        note = "no successful prescribed_dt in the tested set"
        dt_stable_report = None
    elif smallest_fail_above is None:
        note = (
            f"dt_stable >= {largest_success['dt']}: largest timestep tested succeeded; "
            "not a measured upper threshold"
        )
        dt_stable_report = largest_success["dt"]
    else:
        note = (
            f"bracket: last success {largest_success['dt']}, "
            f"first failure above {smallest_fail_above['dt']}"
        )
        dt_stable_report = largest_success["dt"]
    return {
        "largest_stable_dt": dt_stable_report,
        "largest_stable_dt_is_lower_bound_only": smallest_fail_above is None
        and largest_success is not None,
        "failed_upper_dt": None if smallest_fail_above is None else smallest_fail_above["dt"],
        "note": note,
        "probe": largest_success,
        "failed_upper": smallest_fail_above,
        "t_probe": t_probe,
        "trials": trials,
    }


def _residual_drop_timing(
    *,
    route: RCERoute,
    physics: PhysicsConfig,
    grid,
    t0,
    thermo,
    opacity,
    spec: AnalyticOpacityRCESpec,
    target_flatness: float = 0.05,
    max_steps: int = 2000,
    prescribed_f_ext=None,
    dt_accuracy: float | None = 2500.0,
) -> dict:
    cfg = _base_cfg(
        max_steps=max_steps,
        flux_flatness_tolerance=target_flatness,
        tendency_tolerance=target_flatness,
        temp_change_tolerance=target_flatness,
        dt_accuracy=dt_accuracy,
        n_consec=3,
    )
    wall0 = time.perf_counter()
    if prescribed_f_ext is not None:
        res = solve_adaptive_rce_with_prescribed_external_flux(
            grid,
            t0,
            physics,
            _solver(),
            thermo,
            prescribed_f_ext,
            gravity=ConstantGravity(spec.gravity),
            config=cfg,
        )
    else:
        res = solve_adaptive_rce(
            grid,
            t0,
            physics,
            _solver(),
            thermo,
            opacity,
            grid.pressure_centres,
            TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            gravity=ConstantGravity(spec.gravity),
            route=route,
            config=cfg,
        )
    wall = time.perf_counter() - wall0
    summary = _summarize(res, wall)
    summary["target_flatness"] = target_flatness
    summary["hit_target"] = (
        res.status == RCETerminalStatus.CONVERGED
        and res.convergence.flux_flatness <= target_flatness
    )
    return summary


def operator_order_equilibrium(*, n_layers: int = 24, max_steps: int = 800) -> dict:
    """Compare operator orders only after each independently hits the 1e-3 gate."""
    spec = _spec(n_layers)
    grid, thermo, opacity, t0 = _initial(spec)
    routes = {
        "unsplit_explicit": RCERoute.UNSPLIT,
        "rad_then_implicit_conv": RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        "implicit_conv_then_rad": RCERoute.SPLIT_IMPLICIT_CONV_THEN_RAD,
        "strang_rad_implicit_conv": RCERoute.SPLIT_STRANG_RAD_IMPLICIT_CONV,
    }
    out: dict = {"gate": GATE, "routes": {}}
    for name, route in routes.items():
        cfg = _base_cfg(max_steps=max_steps, prescribed_dt=None, t_final=None)
        if route == RCERoute.UNSPLIT:
            cfg = _base_cfg(
                max_steps=max_steps,
                prescribed_dt=None,
                t_final=None,
                dt_accuracy=None,
            )
        wall0 = time.perf_counter()
        res = solve_adaptive_rce(
            grid,
            t0,
            spec.physics(),
            _solver(),
            thermo,
            opacity,
            grid.pressure_centres,
            TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            gravity=ConstantGravity(spec.gravity),
            route=route,
            config=cfg,
        )
        wall = time.perf_counter() - wall0
        summary = _summarize(res, wall)
        hit = (
            res.status == RCETerminalStatus.CONVERGED
            and res.convergence.flux_flatness <= GATE
            and res.convergence.tendency_norm <= GATE
        )
        summary["reached_gate"] = hit
        summary["contract"] = "equilibrium_1e-3"
        if not hit:
            summary["failed"] = True
            summary["failure_reason"] = (
                f"did not reach 1e-3 gate: status={res.status.value} "
                f"flatness={res.convergence.flux_flatness}"
            )
        out["routes"][name] = summary
    completed = [k for k, v in out["routes"].items() if v.get("reached_gate")]
    out["n_gate_converged"] = len(completed)
    out["comparison_valid"] = len(completed) >= 2
    if out["comparison_valid"]:
        names = completed
        out["gate_rcb"] = {
            name: out["routes"][name]["primary_rcb_log10p"] for name in names
        }
        out["gate_flatness"] = {
            name: out["routes"][name]["flux_flatness"] for name in names
        }
    return out


def point39_bracket(*, n_layers: int = 24) -> dict:
    """Four brackets: rad-only; conv-only+frozen rad; coupled explicit; coupled semi-implicit."""
    spec = _spec(n_layers)
    grid, thermo, opacity, t0 = _initial(spec)
    grav = ConstantGravity(spec.gravity)

    state0 = build_column_state(grid, t0, thermo, grav)
    _c, _r, _fc, f_rad0, _ft = _run_unsplit(
        grid,
        state0,
        spec.physics(),
        thermo,
        opacity,
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        _base_cfg(),
        None,
        grav,
    )

    brackets: dict = {}

    phys_rad = PhysicsConfig(
        gravity=spec.gravity,
        alpha=0.0,
        closure_prefactor=spec.physics().closure_prefactor,
    )
    brackets["radiation_only_explicit"] = {
        "stable_dt": _largest_stable_dt(
            route=RCERoute.UNSPLIT,
            physics=phys_rad,
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
        ),
        "residual_drop": _residual_drop_timing(
            route=RCERoute.UNSPLIT,
            physics=phys_rad,
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
            dt_accuracy=None,
        ),
    }

    brackets["convection_only_frozen_rad"] = {
        "stable_dt": _largest_stable_dt(
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            physics=spec.physics(),
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
            prescribed_f_ext=f_rad0,
        ),
        "residual_drop": _residual_drop_timing(
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            physics=spec.physics(),
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
            prescribed_f_ext=f_rad0,
        ),
    }

    brackets["coupled_explicit"] = {
        "stable_dt": _largest_stable_dt(
            route=RCERoute.UNSPLIT,
            physics=spec.physics(),
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
        ),
        "residual_drop": _residual_drop_timing(
            route=RCERoute.UNSPLIT,
            physics=spec.physics(),
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
            dt_accuracy=None,
        ),
    }

    brackets["coupled_semi_implicit"] = {
        "stable_dt": _largest_stable_dt(
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            physics=spec.physics(),
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
        ),
        "residual_drop": _residual_drop_timing(
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            physics=spec.physics(),
            grid=grid,
            t0=t0,
            thermo=thermo,
            opacity=opacity,
            spec=spec,
        ),
    }

    wall0 = time.perf_counter()
    relax = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        _solver(),
        thermo,
        opacity,
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=grav,
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=_base_cfg(
            max_steps=800,
            flux_flatness_tolerance=GATE,
            tendency_tolerance=GATE,
            temp_change_tolerance=GATE,
        ),
    )
    brackets["thermal_relaxation_semi_implicit_to_1e-3"] = _summarize(
        relax, time.perf_counter() - wall0
    )
    return brackets


def main() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "operator_order_fixed_time": operator_order_refinement(),
        "operator_order_equilibrium": operator_order_equilibrium(),
        "point39": point39_bracket(),
    }
    path = DATA_DIR / "operator_order_point39.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    oo = {
        name: payload["operator_order_fixed_time"]["routes"][name]["refinement"]
        for name in payload["operator_order_fixed_time"]["routes"]
    }
    eq = {
        name: {
            "reached_gate": payload["operator_order_equilibrium"]["routes"][name]["reached_gate"],
            "flux_flatness": payload["operator_order_equilibrium"]["routes"][name]["flux_flatness"],
            "status": payload["operator_order_equilibrium"]["routes"][name]["status"],
        }
        for name in payload["operator_order_equilibrium"]["routes"]
    }
    p39 = {
        name: {
            "largest_stable_dt": payload["point39"][name]["stable_dt"]["largest_stable_dt"],
            "largest_stable_dt_is_lower_bound_only": payload["point39"][name]["stable_dt"][
                "largest_stable_dt_is_lower_bound_only"
            ],
            "failed_upper_dt": payload["point39"][name]["stable_dt"]["failed_upper_dt"],
            "note": payload["point39"][name]["stable_dt"]["note"],
            "drop_time": payload["point39"][name]["residual_drop"]["simulated_time"],
            "drop_wall": payload["point39"][name]["residual_drop"]["wall_time_s"],
            "hit": payload["point39"][name]["residual_drop"]["hit_target"],
        }
        for name in (
            "radiation_only_explicit",
            "convection_only_frozen_rad",
            "coupled_explicit",
            "coupled_semi_implicit",
        )
    }
    print(json.dumps({
        "operator_refinement": oo,
        "operator_equilibrium": eq,
        "point39": p39,
        "data": str(path),
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()
