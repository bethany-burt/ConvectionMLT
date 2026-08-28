"""Shared production RCE procedure and physical gate evaluation.

Used by the Stage-4 user runner and MLT validation scripts so the
validated discrete-RZ + five-check path cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .config import SolverConfig
from .gravity import ConstantGravity
from .implicit_convection import ImplicitConvectionConfig
from .rce import (
    AnalyticOpacityRCESpec,
    RCEConfig,
    RCEResult,
    RCERoute,
    grey_radiative_equilibrium_temperature,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
    _primary_rcb_log10p,
)
from .radiation import LowerNetInternalFlux, TopIrradiation
from .reduced_rce import ReducedRCEConfig, solve_reduced_radiative_matching
from .thermodynamics import ConstantH2Thermo

PHYSICAL_GATE = 1.0e-3
ALGEBRAIC_GATE = 1.0e-12
ENERGY_GATE_RATIO_MAX = 1.0

DEFAULT_GRAVITY = 15.0
DEFAULT_P_BOTTOM = 1.0e6
DEFAULT_P_TOP = 1.0
DEFAULT_F_INT = 300.0
DEFAULT_F_IRR = 0.0
DEFAULT_ALPHA = 1.0

ENVELOPE_N_LAYERS = frozenset({96, 192, 384})
ENVELOPE_ALPHA_MIN = 0.5
ENVELOPE_ALPHA_MAX = 4.0

PhaseName = Literal[
    "reduced_rz", "live_polish", "continuation", "repolish", "adaptive_only"
]


def production_solver_config() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)


def production_implicit_config() -> ImplicitConvectionConfig:
    return ImplicitConvectionConfig(
        residual_tolerance=1e-10,
        step_tolerance=1e-10,
        newton_residual_tolerance=1e-12,
        newton_step_tolerance=1e-12,
    )


def production_rce_config(
    *,
    max_steps: int,
    dt_accuracy: float = 2500.0,
    dt_hold_init: float | None = None,
    previous_rcb_init: float | None = None,
    simulated_time_init: float = 0.0,
    gate: float = PHYSICAL_GATE,
    prescribed_dt: float | None = None,
    diffusivity_factor: float | None = None,
) -> RCEConfig:
    kwargs: dict[str, Any] = dict(
        max_steps=max_steps,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=gate,
        tendency_tolerance=gate,
        temp_change_tolerance=gate,
        dt_accuracy=dt_accuracy,
        coupled_picard=True,
        use_coupled_tendency_dt=True,
        dt_hold_init=dt_hold_init,
        previous_rcb_init=previous_rcb_init,
        simulated_time_init=simulated_time_init,
        implicit_convection=production_implicit_config(),
    )
    if prescribed_dt is not None:
        kwargs["prescribed_dt"] = float(prescribed_dt)
    if diffusivity_factor is not None:
        kwargs["diffusivity_factor"] = float(diffusivity_factor)
    return RCEConfig(**kwargs)


@dataclass(frozen=True)
class GateEvaluation:
    flux_flatness_ok: bool
    tendency_ok: bool
    energy_ok: bool
    finite_state_ok: bool
    algebraic_ok: bool
    topology_ok: bool
    convergence_ok: bool
    flux_flatness: float
    tendency_norm: float
    energy_gate_ratio: float | None
    energy_committed_residual_rel: float | None
    flux_split_identity_rel: float | None
    telescoping_column_energy_rel: float | None
    bottom_boundary_exactness_rel: float | None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "flux_flatness_ok": self.flux_flatness_ok,
            "tendency_ok": self.tendency_ok,
            "energy_ok": self.energy_ok,
            "finite_state_ok": self.finite_state_ok,
            "algebraic_ok": self.algebraic_ok,
            "topology_ok": self.topology_ok,
            "convergence_ok": self.convergence_ok,
            "flux_flatness": self.flux_flatness,
            "tendency_norm": self.tendency_norm,
            "energy_gate_ratio": self.energy_gate_ratio,
            "energy_committed_residual_rel": self.energy_committed_residual_rel,
            "flux_split_identity_rel": self.flux_split_identity_rel,
            "telescoping_column_energy_rel": self.telescoping_column_energy_rel,
            "bottom_boundary_exactness_rel": self.bottom_boundary_exactness_rel,
            "details": self.details,
        }


def _topo_ok(regions: Any, detached: Any) -> bool:
    regs = list(regions or [])
    det = list(detached or [])
    if det:
        return False
    if len(regs) != 1:
        return False
    return int(regs[0][0]) == 0


def algebraic_identity_residuals(
    *,
    flux_total: NDArray[np.float64],
    flux_rad: NDArray[np.float64],
    flux_conv: NDArray[np.float64],
    mass_path: NDArray[np.float64],
    f_int: float,
) -> dict[str, float]:
    f_tot = np.asarray(flux_total, dtype=np.float64)
    f_rad = np.asarray(flux_rad, dtype=np.float64)
    f_conv = np.asarray(flux_conv, dtype=np.float64)
    mass = np.asarray(mass_path, dtype=np.float64)
    scale = max(abs(float(f_int)), 1.0)
    heating = (f_tot[:-1] - f_tot[1:]) / mass
    return {
        "flux_split_identity_rel": float(np.max(np.abs(f_rad + f_conv - f_tot)) / scale),
        "telescoping_column_energy_rel": float(
            abs(float(np.sum(mass * heating)) - float(f_tot[0] - f_tot[-1])) / scale
        ),
        "bottom_boundary_exactness_rel": float(abs(float(f_tot[0]) - float(f_int)) / scale),
    }


def evaluate_physical_gates(
    record: dict[str, Any],
    *,
    gate: float = PHYSICAL_GATE,
    require_bottom_connected_cz: bool = True,
    algebraic_gate: float = ALGEBRAIC_GATE,
) -> GateEvaluation:
    """Shared physical-gate evaluator (convergence vs topology kept separate)."""
    flat = float(record.get("flux_flatness") or np.inf)
    tend = float(record.get("tendency_norm") or np.inf)
    flat_ok = bool(np.isfinite(flat) and flat <= gate)
    tend_ok = bool(np.isfinite(tend) and tend <= gate)

    e_ratio = record.get("energy_gate_ratio")
    e_res = record.get("energy_committed_residual_rel")
    e_ratio_f = None if e_ratio is None else float(e_ratio)
    e_res_f = None if e_res is None else float(e_res)
    if e_ratio_f is not None and np.isfinite(e_ratio_f):
        energy_ok = e_ratio_f <= ENERGY_GATE_RATIO_MAX
    elif e_res_f is not None and np.isfinite(e_res_f):
        energy_ok = e_res_f <= gate
    else:
        energy_ok = False

    t_raw = record.get("temperature")
    if t_raw is None:
        t = np.asarray([], dtype=np.float64)
    else:
        t = np.asarray(t_raw, dtype=np.float64)
    finite_ok = bool(t.size > 0 and np.all(np.isfinite(t)) and np.all(t > 0.0))
    for key in ("flux_total", "flux_rad", "flux_conv"):
        arr_raw = record.get(key)
        if arr_raw is None:
            continue
        arr = np.asarray(arr_raw, dtype=np.float64)
        if arr.size and not np.all(np.isfinite(arr)):
            finite_ok = False

    split = record.get("flux_split_identity_rel")
    tele = record.get("telescoping_column_energy_rel")
    bot = record.get("bottom_boundary_exactness_rel")
    if split is None or tele is None or bot is None:
        try:
            ids = algebraic_identity_residuals(
                flux_total=np.asarray(record["flux_total"], dtype=np.float64),
                flux_rad=np.asarray(record["flux_rad"], dtype=np.float64),
                flux_conv=np.asarray(record["flux_conv"], dtype=np.float64),
                mass_path=np.asarray(record["mass_path"], dtype=np.float64),
                f_int=float(record["f_int"]),
            )
            split = ids["flux_split_identity_rel"]
            tele = ids["telescoping_column_energy_rel"]
            bot = ids["bottom_boundary_exactness_rel"]
        except Exception:
            split = tele = bot = None
    split_f = None if split is None else float(split)
    tele_f = None if tele is None else float(tele)
    bot_f = None if bot is None else float(bot)
    algebraic_ok = all(
        v is not None and np.isfinite(v) and v <= algebraic_gate
        for v in (split_f, tele_f, bot_f)
    )

    topo = _topo_ok(
        record.get("convective_regions"),
        record.get("detached_convective_regions"),
    )
    if not require_bottom_connected_cz:
        topo = True

    convergence_ok = flat_ok and tend_ok and energy_ok and finite_ok and algebraic_ok
    return GateEvaluation(
        flux_flatness_ok=flat_ok,
        tendency_ok=tend_ok,
        energy_ok=energy_ok,
        finite_state_ok=finite_ok,
        algebraic_ok=algebraic_ok,
        topology_ok=bool(topo),
        convergence_ok=convergence_ok,
        flux_flatness=flat,
        tendency_norm=tend,
        energy_gate_ratio=e_ratio_f,
        energy_committed_residual_rel=e_res_f,
        flux_split_identity_rel=split_f,
        telescoping_column_energy_rel=tele_f,
        bottom_boundary_exactness_rel=bot_f,
        details={
            "gate": gate,
            "require_bottom_connected_cz": require_bottom_connected_cz,
            "algebraic_gate": algebraic_gate,
        },
    )


def validation_envelope(
    *,
    n_layers: int,
    alpha: float,
    f_int: float,
    f_irr: float,
    gravity: float,
    p_bottom: float,
    p_top: float,
    composition: str,
    opacity_model: str,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if int(n_layers) not in ENVELOPE_N_LAYERS:
        warnings.append(
            f"n_layers={n_layers} outside demonstrated set {sorted(ENVELOPE_N_LAYERS)}"
        )
    if not (ENVELOPE_ALPHA_MIN <= float(alpha) <= ENVELOPE_ALPHA_MAX):
        warnings.append(
            f"alpha={alpha} outside tested range "
            f"[{ENVELOPE_ALPHA_MIN}, {ENVELOPE_ALPHA_MAX}]"
        )
    if composition != "constant_h2":
        warnings.append(f"composition={composition!r} is not the validated constant_h2 EOS")
    if opacity_model != "analytic_grey_powerlaw":
        warnings.append(
            f"opacity_model={opacity_model!r} is not the validated grey power law"
        )
    if abs(float(f_int) - DEFAULT_F_INT) > 1.0e-12 * max(abs(DEFAULT_F_INT), 1.0):
        warnings.append(f"f_int={f_int} differs from validated default {DEFAULT_F_INT}")
    if abs(float(f_irr) - DEFAULT_F_IRR) > 1.0e-12:
        warnings.append(f"f_irr={f_irr} differs from validated default {DEFAULT_F_IRR}")
    if abs(float(gravity) - DEFAULT_GRAVITY) > 1.0e-12 * DEFAULT_GRAVITY:
        warnings.append(f"gravity={gravity} differs from validated default {DEFAULT_GRAVITY}")
    if abs(float(p_bottom) - DEFAULT_P_BOTTOM) > 1.0e-12 * DEFAULT_P_BOTTOM:
        warnings.append(
            f"p_bottom={p_bottom} differs from validated default {DEFAULT_P_BOTTOM}"
        )
    if abs(float(p_top) - DEFAULT_P_TOP) > 1.0e-12 * DEFAULT_P_TOP:
        warnings.append(f"p_top={p_top} differs from validated default {DEFAULT_P_TOP}")
    return ("INSIDE" if not warnings else "OUTSIDE"), warnings


@dataclass
class ConvergenceRow:
    step: int
    phase: PhaseName
    time_s: float
    dt: float
    flux_flatness: float
    tendency_norm: float
    time_is_physical: bool


@dataclass
class ProductionRCERun:
    result: RCEResult
    temperature_initial: NDArray[np.float64]
    spec: AnalyticOpacityRCESpec
    solver: SolverConfig
    rce_config_last: RCEConfig
    convergence_log: list[ConvergenceRow]
    phases: list[str]
    gate: float
    procedure: str
    prescribed_dt: float | None
    nabla: NDArray[np.float64]
    nabla_ad: NDArray[np.float64]
    delta_nabla: NDArray[np.float64]
    pressure_centres: NDArray[np.float64]
    pressure_edges: NDArray[np.float64]


@dataclass(frozen=True)
class ProductionControls:
    max_steps_live_polish: int = 200
    max_steps_continuation: int = 500
    max_recovery_cycles: int = 2
    dt_accuracy_s: float = 50000.0
    dt_hold_init_s: float = 18415.0
    continuation_dt_accuracy_s: float = 2500.0
    prescribed_dt_s: float | None = None
    max_steps_adaptive_only: int = 20000
    gate: float = PHYSICAL_GATE


def build_spec(
    *,
    n_layers: int,
    alpha: float = DEFAULT_ALPHA,
    f_int: float = DEFAULT_F_INT,
    f_irr: float = DEFAULT_F_IRR,
    gravity: float = DEFAULT_GRAVITY,
    p_bottom: float = DEFAULT_P_BOTTOM,
    p_top: float = DEFAULT_P_TOP,
) -> AnalyticOpacityRCESpec:
    return nested_analytic_opacity_spec(
        int(n_layers),
        alpha=float(alpha),
        f_int=float(f_int),
        f_irr=float(f_irr),
        gravity=float(gravity),
        p_bottom=float(p_bottom),
        p_top=float(p_top),
    )


def build_seed_temperature(
    spec: AnalyticOpacityRCESpec,
    seed: str,
) -> NDArray[np.float64]:
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opac = spec.opacity()
    if seed == "radiative_convective":
        return radiative_convective_initial_temperature(
            grid, opac, thermo, spec.f_int, spec.f_irr
        )
    if seed == "radiative_equilibrium":
        return grey_radiative_equilibrium_temperature(
            grid, opac, spec.f_int, spec.f_irr
        )
    raise ValueError(
        f"unsupported seed {seed!r}; v1 supports "
        "radiative_convective | radiative_equilibrium"
    )


def _closure_gradients(
    result: RCEResult, thermo: ConstantH2Thermo
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    closure = result.final_closure
    nabla = np.asarray(closure.gradient, dtype=np.float64).copy()
    nabla_ad = np.full_like(nabla, float(thermo.nabla_ad))
    delta = np.asarray(closure.superadiabaticity, dtype=np.float64).copy()
    return nabla, nabla_ad, delta


def _append_history(
    rows: list[ConvergenceRow],
    result: RCEResult,
    phase: PhaseName,
    *,
    time_is_physical: bool,
    step_offset: int,
) -> int:
    t_cum = 0.0
    accepted = [d for d in result.diagnostics if d.accepted]
    for i, d in enumerate(accepted):
        sim = getattr(d, "simulated_time", None)
        if sim is not None and np.isfinite(float(sim)):
            t_cum = float(sim)
        else:
            t_cum += float(d.dt)
        rows.append(
            ConvergenceRow(
                step=step_offset + i,
                phase=phase,
                time_s=t_cum,
                dt=float(d.dt),
                flux_flatness=float(d.flux_flatness),
                tendency_norm=float(d.tendency_norm),
                time_is_physical=time_is_physical,
            )
        )
    return step_offset + len(accepted)


def _last_accepted_energy(result: RCEResult) -> tuple[float | None, float | None]:
    for d in reversed(result.diagnostics):
        if d.accepted:
            return float(d.energy_gate_ratio), float(d.energy_committed_residual_rel)
    return None, None


def _snapshot_dict(result: RCEResult, spec: AnalyticOpacityRCESpec) -> dict[str, Any]:
    e_ratio, e_res = _last_accepted_energy(result)
    return {
        "flux_flatness": float(result.convergence.flux_flatness),
        "tendency_norm": float(result.convergence.tendency_norm),
        "temperature": np.asarray(result.final_state.temperature, dtype=np.float64),
        "flux_total": np.asarray(result.final_flux_total, dtype=np.float64),
        "flux_rad": np.asarray(result.final_flux_rad, dtype=np.float64),
        "flux_conv": np.asarray(result.final_flux_conv, dtype=np.float64),
        "mass_path": np.asarray(result.final_state.mass_path, dtype=np.float64),
        "f_int": float(spec.f_int),
        "convective_regions": result.convective_regions,
        "detached_convective_regions": result.detached_convective_regions,
        "energy_gate_ratio": e_ratio,
        "energy_committed_residual_rel": e_res,
    }


def _gates_from_result(
    result: RCEResult,
    spec: AnalyticOpacityRCESpec,
    *,
    gate: float,
    require_bottom_connected_cz: bool,
) -> GateEvaluation:
    return evaluate_physical_gates(
        _snapshot_dict(result, spec),
        gate=gate,
        require_bottom_connected_cz=require_bottom_connected_cz,
    )


def _live_solve(
    *,
    grid,
    t0: NDArray[np.float64],
    spec: AnalyticOpacityRCESpec,
    solver: SolverConfig,
    thermo: ConstantH2Thermo,
    max_steps: int,
    dt_accuracy: float,
    dt_hold_init: float | None,
    previous_rcb: float | None,
    gate: float,
    prescribed_dt: float | None,
    simulated_time_init: float = 0.0,
) -> tuple[RCEResult, RCEConfig]:
    cfg = production_rce_config(
        max_steps=max_steps,
        dt_accuracy=dt_accuracy,
        dt_hold_init=dt_hold_init,
        previous_rcb_init=previous_rcb,
        simulated_time_init=simulated_time_init,
        gate=gate,
        prescribed_dt=prescribed_dt,
    )
    res = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    return res, cfg


def _reduced(
    *,
    grid,
    t0: NDArray[np.float64],
    spec: AnalyticOpacityRCESpec,
    solver: SolverConfig,
    thermo: ConstantH2Thermo,
    log: Callable[[str], None] | None,
    label: str,
):
    if log:
        log(f"[{label}] reduced_RZ accelerator...")
    return solve_reduced_radiative_matching(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        config=ReducedRCEConfig(),
    )


def _passed(gates: GateEvaluation, require_topo: bool) -> bool:
    return gates.convergence_ok and (gates.topology_ok or not require_topo)


def run_production_rce(
    *,
    n_layers: int,
    alpha: float = DEFAULT_ALPHA,
    f_int: float = DEFAULT_F_INT,
    f_irr: float = DEFAULT_F_IRR,
    gravity: float = DEFAULT_GRAVITY,
    p_bottom: float = DEFAULT_P_BOTTOM,
    p_top: float = DEFAULT_P_TOP,
    seed: str = "radiative_convective",
    procedure: str = "production",
    controls: ProductionControls | None = None,
    temperature_initial: NDArray[np.float64] | None = None,
    log: Callable[[str], None] | None = None,
) -> ProductionRCERun:
    """Validated production procedure (discrete-RZ + five-check + recovery)."""
    ctrl = controls or ProductionControls()
    if ctrl.gate > PHYSICAL_GATE + 1.0e-15:
        raise ValueError(f"gate {ctrl.gate} exceeds frozen PHYSICAL_GATE={PHYSICAL_GATE}")
    spec = build_spec(
        n_layers=n_layers,
        alpha=alpha,
        f_int=f_int,
        f_irr=f_irr,
        gravity=gravity,
        p_bottom=p_bottom,
        p_top=p_top,
    )
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    t0 = (
        np.asarray(temperature_initial, dtype=np.float64).copy()
        if temperature_initial is not None
        else build_seed_temperature(spec, seed)
    )
    require_topo = abs(float(f_irr)) <= 1.0e-15
    time_physical = ctrl.prescribed_dt_s is not None
    rows: list[ConvergenceRow] = []
    phases: list[str] = []
    step_offset = 0

    if procedure == "adaptive_only":
        phases.append("adaptive_only")
        if log:
            log("[adaptive_only] starting adaptive solve...")
        res, cfg = _live_solve(
            grid=grid,
            t0=t0,
            spec=spec,
            solver=solver,
            thermo=thermo,
            max_steps=ctrl.max_steps_adaptive_only,
            dt_accuracy=ctrl.dt_accuracy_s if ctrl.prescribed_dt_s is None else 1.0e12,
            dt_hold_init=None,
            previous_rcb=None,
            gate=ctrl.gate,
            prescribed_dt=ctrl.prescribed_dt_s,
        )
        _append_history(
            rows, res, "adaptive_only", time_is_physical=time_physical, step_offset=0
        )
        nabla, nabla_ad, delta = _closure_gradients(res, thermo)
        return ProductionRCERun(
            result=res,
            temperature_initial=t0,
            spec=spec,
            solver=solver,
            rce_config_last=cfg,
            convergence_log=rows,
            phases=phases,
            gate=ctrl.gate,
            procedure=procedure,
            prescribed_dt=ctrl.prescribed_dt_s,
            nabla=nabla,
            nabla_ad=nabla_ad,
            delta_nabla=delta,
            pressure_centres=np.asarray(grid.pressure_centres, dtype=np.float64),
            pressure_edges=np.asarray(grid.pressure_edges, dtype=np.float64),
        )

    if procedure != "production":
        raise ValueError(f"unknown procedure {procedure!r}")

    phases.append("reduced_rz")
    reduced = _reduced(
        grid=grid, t0=t0, spec=spec, solver=solver, thermo=thermo, log=log, label="production"
    )
    t_work = (
        np.asarray(reduced.temperature, dtype=np.float64).copy()
        if reduced.improved
        else t0.copy()
    )
    rcb = None
    if reduced.trial is not None:
        rcb = _primary_rcb_log10p(grid, reduced.trial.closure, solver)

    phases.append("live_polish")
    if log:
        log("[live_polish] five-check polish...")
    res, cfg = _live_solve(
        grid=grid,
        t0=t_work,
        spec=spec,
        solver=solver,
        thermo=thermo,
        max_steps=ctrl.max_steps_live_polish,
        dt_accuracy=ctrl.dt_accuracy_s if ctrl.prescribed_dt_s is None else 1.0e12,
        dt_hold_init=None if ctrl.prescribed_dt_s is not None else ctrl.dt_hold_init_s,
        previous_rcb=rcb,
        gate=ctrl.gate,
        prescribed_dt=ctrl.prescribed_dt_s,
    )
    step_offset = _append_history(
        rows, res, "live_polish", time_is_physical=time_physical, step_offset=step_offset
    )
    gates = _gates_from_result(
        res, spec, gate=ctrl.gate, require_bottom_connected_cz=require_topo
    )
    if log:
        log(
            f"[live_polish] convergence_ok={gates.convergence_ok} "
            f"topology_ok={gates.topology_ok} flat={gates.flux_flatness:.6g}"
        )

    for cycle in range(int(ctrl.max_recovery_cycles)):
        if _passed(gates, require_topo):
            break
        phases.append(f"continuation[{cycle}]")
        if log:
            log(f"[continuation {cycle + 1}/{ctrl.max_recovery_cycles}] ...")
        t_cont = np.asarray(res.final_state.temperature, dtype=np.float64)
        res, cfg = _live_solve(
            grid=grid,
            t0=t_cont,
            spec=spec,
            solver=solver,
            thermo=thermo,
            max_steps=ctrl.max_steps_continuation,
            dt_accuracy=(
                ctrl.continuation_dt_accuracy_s
                if ctrl.prescribed_dt_s is None
                else 1.0e12
            ),
            dt_hold_init=(
                None
                if ctrl.prescribed_dt_s is not None
                else min(ctrl.dt_hold_init_s, ctrl.continuation_dt_accuracy_s)
            ),
            previous_rcb=res.primary_rcb_log10p,
            gate=ctrl.gate,
            prescribed_dt=ctrl.prescribed_dt_s,
            simulated_time_init=float(res.simulated_time),
        )
        step_offset = _append_history(
            rows,
            res,
            "continuation",
            time_is_physical=time_physical,
            step_offset=step_offset,
        )
        gates = _gates_from_result(
            res, spec, gate=ctrl.gate, require_bottom_connected_cz=require_topo
        )
        if _passed(gates, require_topo):
            break

        phases.append(f"repolish[{cycle}]")
        reduced = _reduced(
            grid=grid,
            t0=np.asarray(res.final_state.temperature, dtype=np.float64),
            spec=spec,
            solver=solver,
            thermo=thermo,
            log=log,
            label=f"repolish[{cycle}]",
        )
        t_work = (
            np.asarray(reduced.temperature, dtype=np.float64).copy()
            if reduced.improved
            else np.asarray(res.final_state.temperature, dtype=np.float64).copy()
        )
        rcb = None
        if reduced.trial is not None:
            rcb = _primary_rcb_log10p(grid, reduced.trial.closure, solver)
        if log:
            log(f"[repolish {cycle + 1}] live_polish...")
        res, cfg = _live_solve(
            grid=grid,
            t0=t_work,
            spec=spec,
            solver=solver,
            thermo=thermo,
            max_steps=ctrl.max_steps_live_polish,
            dt_accuracy=ctrl.dt_accuracy_s if ctrl.prescribed_dt_s is None else 1.0e12,
            dt_hold_init=None if ctrl.prescribed_dt_s is not None else ctrl.dt_hold_init_s,
            previous_rcb=rcb,
            gate=ctrl.gate,
            prescribed_dt=ctrl.prescribed_dt_s,
            simulated_time_init=float(res.simulated_time),
        )
        step_offset = _append_history(
            rows, res, "repolish", time_is_physical=time_physical, step_offset=step_offset
        )
        gates = _gates_from_result(
            res, spec, gate=ctrl.gate, require_bottom_connected_cz=require_topo
        )
        if log:
            log(
                f"[repolish {cycle + 1}] convergence_ok={gates.convergence_ok} "
                f"flat={gates.flux_flatness:.6g}"
            )

    nabla, nabla_ad, delta = _closure_gradients(res, thermo)
    return ProductionRCERun(
        result=res,
        temperature_initial=t0,
        spec=spec,
        solver=solver,
        rce_config_last=cfg,
        convergence_log=rows,
        phases=phases,
        gate=ctrl.gate,
        procedure=procedure,
        prescribed_dt=ctrl.prescribed_dt_s,
        nabla=nabla,
        nabla_ad=nabla_ad,
        delta_nabla=delta,
        pressure_centres=np.asarray(grid.pressure_centres, dtype=np.float64),
        pressure_edges=np.asarray(grid.pressure_edges, dtype=np.float64),
    )
