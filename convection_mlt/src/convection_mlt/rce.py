"""Stage 4 fixed-composition radiative-convective equilibrium solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .closure import ClosureResult, mixing_length_flux
from .config import PhysicsConfig, SolverConfig
from .energy import enthalpy_tendency
from .gravity import ConstantGravity, GravityLaw
from .grid import PressureGrid
from .hydrostatics import HydrostaticDomainError
from .opacity import PrescribedOpacity
from .radiation import (
    DEFAULT_DIFFUSIVITY,
    LowerFlux,
    LowerTemperature,
    RadiationResult,
    SolveRoute,
    TopIrradiation,
    solve_radiation,
)
from .solvers_enthalpy import _crossing_reason
from .state import ColumnState, build_column_state
from .thermodynamics import EnthalpyInversionError, ThermoDomainError, ThermoProvider


class RCERoute(str, Enum):
    UNSPLIT = "unsplit"
    SPLIT_RAD_THEN_CONV = "split_rad_then_conv"
    SPLIT_CONV_THEN_RAD = "split_conv_then_rad"


class RCETerminalStatus(str, Enum):
    CONVERGED = "converged"
    MAX_STEPS = "max_steps"
    DT_MIN_FAILURE = "dt_min_failure"
    STALLED = "stalled"


@dataclass(frozen=True)
class RCEConvergence:
    flux_flatness: float
    tendency_norm: float
    temp_change: float
    rcb_stable: bool
    finite_state: bool


@dataclass(frozen=True)
class RCEStepDiagnostics:
    dt: float
    accepted: bool
    route: RCERoute
    dt_mlt: float
    dt_rad: float
    dt_temp: float
    flux_boundary_work: float
    energy_lhs: float
    energy_residual: float
    energy_residual_rel: float
    flux_flatness: float
    boundary_mismatch: float
    temp_change: float
    tendency_norm: float
    primary_rcb_log10p: float | None
    n_bottom_connected_regions: int
    rejection_reason: str | None = None


@dataclass(frozen=True)
class RCEResult:
    status: RCETerminalStatus
    reason: str
    route: RCERoute
    steps_attempted: int
    steps_accepted: int
    rejections: int
    simulated_time: float
    final_state: ColumnState
    final_closure: ClosureResult
    final_radiation: RadiationResult
    final_flux_total: NDArray[np.float64]
    final_flux_conv: NDArray[np.float64]
    final_flux_rad: NDArray[np.float64]
    primary_rcb_log10p: float | None
    convective_regions: list[tuple[int, int]]
    convergence: RCEConvergence
    diagnostics: list[RCEStepDiagnostics]


@dataclass(frozen=True)
class RCEConfig:
    n_consec: int = 5
    flux_flatness_tolerance: float = 1e-10
    tendency_tolerance: float = 1e-10
    temp_change_tolerance: float = 1e-9
    rcb_stability_log10p_tolerance: float = 2e-3
    dt_min: float = 1e-12
    max_steps: int = 200000
    max_rejections: int = 50
    f_back: float = 0.5
    stall_window: int = 2000
    stall_rel_improvement: float = 1e-6
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY
    radiation_route: SolveRoute = SolveRoute.THOMAS
    prescribed_dt: float | None = None
    flux_scale_floor: float = 1e-30
    temp_scale_floor: float = 1e-12
    energy_scale_floor: float = 1e-30


@dataclass(frozen=True)
class ManufacturedRadiativeTarget:
    target_temperature: NDArray[np.float64]
    f0: float
    # Inverse-time Newtonian coefficient on enthalpy. Zero keeps the frozen
    # operator F_rad* = F0 - F_conv(T*). A positive value adds a conservative
    # interface flux whose divergence is -kappa (h - h*) and vanishes at T*.
    relaxation_coeff: float = 0.0


def _empty_radiation(n_layers: int) -> RadiationResult:
    n_iface = n_layers + 1
    return RadiationResult(
        flux_up=np.zeros((1, n_iface)),
        flux_down=np.zeros((1, n_iface)),
        flux_net_band=np.zeros((1, n_iface)),
        flux_net=np.zeros(n_iface),
        heating=np.zeros(n_layers),
        optical_depth=np.zeros((1, n_layers)),
        transmissivity=np.ones((1, n_layers)),
    )


def _dt_mlt_estimate(grid: PressureGrid, state: ColumnState, closure: ClosureResult, solver: SolverConfig) -> float:
    dz = np.diff(state.z_edges)
    adjacent_kh = np.maximum(closure.thermal_diffusivity[:-1], closure.thermal_diffusivity[1:])
    diff_bounds = np.full(grid.n_layers, np.inf)
    active = adjacent_kh > 0
    diff_bounds[active] = dz[active] ** 2 / adjacent_kh[active]
    return solver.c_diff * float(np.min(diff_bounds, initial=np.inf))


def _dt_rad_estimate(state: ColumnState, rad_heating: NDArray[np.float64], solver: SolverConfig, thermo: ThermoProvider) -> float:
    cp = thermo.specific_heat(state.temperature)
    tdot = rad_heating / cp
    active = np.abs(tdot) > 0.0
    if np.any(active):
        return solver.epsilon_temperature * float(np.min(state.temperature[active] / np.abs(tdot[active])))
    return np.inf


def _dt_temp_estimate(state: ColumnState, total_dhdt: NDArray[np.float64], solver: SolverConfig, thermo: ThermoProvider) -> float:
    cp = thermo.specific_heat(state.temperature)
    tdot = total_dhdt / cp
    active = np.abs(tdot) > 0.0
    if np.any(active):
        return solver.epsilon_temperature * float(np.min(state.temperature[active] / np.abs(tdot[active])))
    return np.inf


def _evaluate_closure(grid: PressureGrid, state: ColumnState, physics: PhysicsConfig, thermo: ThermoProvider) -> ClosureResult:
    return mixing_length_flux(
        grid,
        state.temperature,
        state.g_edges,
        physics.alpha,
        thermo,
        physics.closure_prefactor,
        use_entropy_instability=True,
    )


def _build_rad_from_target(
    grid: PressureGrid,
    state: ColumnState,
    closure_target: ClosureResult,
    target: ManufacturedRadiativeTarget,
    target_enthalpy: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    if target.target_temperature.shape != (grid.n_layers,):
        raise ValueError("target_temperature shape mismatch")
    f_rad = target.f0 - closure_target.flux
    if target.relaxation_coeff <= 0.0:
        return f_rad
    h_star = state.enthalpy if target_enthalpy is None else target_enthalpy
    q_layer = -target.relaxation_coeff * (state.enthalpy - h_star)
    f_corr = np.zeros_like(f_rad)
    for i in range(grid.n_layers):
        f_corr[i + 1] = f_corr[i] - q_layer[i] * state.mass_path[i]
    return f_rad + f_corr


def manufactured_operator_identity(
    grid: PressureGrid,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    target: ManufacturedRadiativeTarget,
    gravity: GravityLaw | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Evaluate F_rad* + F_conv(T*) and dh/dt at T* with no time integration."""
    grav = gravity or ConstantGravity(physics.gravity)
    state = build_column_state(grid, np.asarray(target.target_temperature, dtype=np.float64), thermo, grav)
    closure = _evaluate_closure(grid, state, physics, thermo)
    f_rad = _build_rad_from_target(grid, state, closure, target, target_enthalpy=state.enthalpy)
    f_total = f_rad + closure.flux
    dhdt = enthalpy_tendency(grid, f_total, state.mass_path)
    flux_err = float(np.max(np.abs(f_total - target.f0), initial=0.0))
    tend_err = float(np.max(np.abs(dhdt), initial=0.0))
    return f_total, dhdt, flux_err, tend_err


def _rcb_regions(closure: ClosureResult, solver: SolverConfig) -> list[tuple[int, int]]:
    delta_internal = closure.superadiabaticity[1:-1]
    active = delta_internal > solver.c_active * solver.epsilon_gradient
    regions: list[tuple[int, int]] = []
    i = 0
    n_layers = delta_internal.size + 1
    while i < n_layers:
        j = i
        while j < n_layers - 1 and active[j]:
            j += 1
        regions.append((i, j))
        i = j + 1
    return regions


def _primary_rcb_log10p(grid: PressureGrid, closure: ClosureResult, solver: SolverConfig) -> float | None:
    active = closure.superadiabaticity[1:-1] > solver.c_active * solver.epsilon_gradient
    if not np.any(active):
        return None
    if np.all(active):
        return float(np.log10(grid.pressure_edges[-1]))
    idx = 0
    while idx < active.size and active[idx]:
        idx += 1
    i_active = idx
    p_lo = grid.pressure_edges[i_active]
    p_hi = grid.pressure_edges[i_active + 1]
    d_lo = closure.superadiabaticity[i_active]
    d_hi = closure.superadiabaticity[i_active + 1]
    if np.isfinite(d_lo) and np.isfinite(d_hi) and (d_lo - d_hi) != 0:
        w = d_lo / (d_lo - d_hi)
        w = float(np.clip(w, 0.0, 1.0))
        logp = (1.0 - w) * np.log10(p_lo) + w * np.log10(p_hi)
        return float(logp)
    return float(np.log10(p_hi))


def _trial_atomic_state(
    grid: PressureGrid,
    state: ColumnState,
    dhdt: NDArray[np.float64],
    dt: float,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    solver: SolverConfig,
) -> tuple[ColumnState | None, str | None]:
    cp = thermo.specific_heat(state.temperature)
    t_tendency = dhdt / cp
    try:
        h_trial = state.enthalpy + dt * dhdt
        t_trial = thermo.invert_enthalpy(h_trial)
        if not np.all(np.isfinite(t_trial)) or np.any(t_trial <= 0.0):
            return None, "nonfinite/nonpositive trial temperature"
        _ = thermo.specific_heat(t_trial)
        if np.max(np.abs(dt * t_tendency) / state.temperature, initial=0.0) > (
            solver.epsilon_temperature * (1.0 + 1.0e-12)
        ):
            return None, "fractional-temperature bound exceeded"
        trial_state = build_column_state(grid, t_trial, thermo, gravity, enthalpy=h_trial)
        return trial_state, None
    except (ThermoDomainError, EnthalpyInversionError, HydrostaticDomainError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_unsplit(
    grid: PressureGrid,
    state: ColumnState,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    opacity: PrescribedOpacity,
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    lower_bc: LowerFlux | LowerTemperature,
    cfg: RCEConfig,
    manufactured: ManufacturedRadiativeTarget | None,
    gravity: GravityLaw,
) -> tuple[ClosureResult, RadiationResult | None, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    closure = _evaluate_closure(grid, state, physics, thermo)
    if manufactured is None:
        rad = solve_radiation(
            state.temperature,
            state.mass_path,
            opacity,
            pressure,
            top_bc,
            lower_bc,
            cfg.diffusivity_factor,
            cfg.radiation_route,
        )
        f_rad = rad.flux_net
    else:
        target_state = build_column_state(grid, manufactured.target_temperature, thermo, gravity)
        closure_target = _evaluate_closure(grid, target_state, physics, thermo)
        f_rad = _build_rad_from_target(
            grid, state, closure_target, manufactured, target_enthalpy=target_state.enthalpy
        )
        rad = None
    f_conv = closure.flux
    f_total = f_rad + f_conv
    return closure, rad, f_conv, f_rad, f_total


@dataclass(frozen=True)
class _SplitAttempt:
    ok: bool
    reason: str | None
    state: ColumnState
    f_conv: NDArray[np.float64]
    f_rad: NDArray[np.float64]
    boundary_work: float
    energy_lhs: float


def _run_split_macrostep(
    route: RCERoute,
    grid: PressureGrid,
    state: ColumnState,
    dt: float,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    opacity: PrescribedOpacity,
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    lower_bc: LowerFlux | LowerTemperature,
    cfg: RCEConfig,
    manufactured: ManufacturedRadiativeTarget | None,
    solver: SolverConfig,
) -> _SplitAttempt:
    nan_f = np.full(grid.n_layers + 1, np.nan)

    def _fail(reason: str) -> _SplitAttempt:
        return _SplitAttempt(False, reason, state, nan_f, nan_f, float("nan"), float("nan"))

    def rad_substep(s: ColumnState) -> tuple[ColumnState, NDArray[np.float64], float, float]:
        if manufactured is None:
            rr = solve_radiation(
                s.temperature, s.mass_path, opacity, pressure, top_bc, lower_bc,
                cfg.diffusivity_factor, cfg.radiation_route,
            )
            f_rad = rr.flux_net
        else:
            target_state = build_column_state(grid, manufactured.target_temperature, thermo, gravity)
            closure_target = _evaluate_closure(grid, target_state, physics, thermo)
            f_rad = _build_rad_from_target(
                grid, s, closure_target, manufactured, target_enthalpy=target_state.enthalpy
            )
        dhdt = enthalpy_tendency(grid, f_rad, s.mass_path)
        s_new, reason = _trial_atomic_state(grid, s, dhdt, dt, thermo, gravity, solver)
        if s_new is None:
            raise ThermoDomainError(reason or "radiation substep failed")
        work = dt * float(f_rad[0] - f_rad[-1])
        lhs = float(dt * np.sum(s.mass_path * dhdt))
        return s_new, f_rad, work, lhs

    def conv_substep(s: ColumnState) -> tuple[ColumnState, NDArray[np.float64], float, float]:
        c = _evaluate_closure(grid, s, physics, thermo)
        f_conv = c.flux
        dhdt = enthalpy_tendency(grid, f_conv, s.mass_path)
        s_new, reason = _trial_atomic_state(grid, s, dhdt, dt, thermo, gravity, solver)
        if s_new is None:
            raise ThermoDomainError(reason or "convection substep failed")
        work = dt * float(f_conv[0] - f_conv[-1])
        lhs = float(dt * np.sum(s.mass_path * dhdt))
        return s_new, f_conv, work, lhs

    try:
        if route == RCERoute.SPLIT_RAD_THEN_CONV:
            s1, f_rad, w1, lhs1 = rad_substep(state)
            s2, f_conv, w2, lhs2 = conv_substep(s1)
        elif route == RCERoute.SPLIT_CONV_THEN_RAD:
            s1, f_conv, w1, lhs1 = conv_substep(state)
            s2, f_rad, w2, lhs2 = rad_substep(s1)
        else:
            return _fail(f"Unsupported split route {route}")
        crossed = _crossing_reason(
            _evaluate_closure(grid, state, physics, thermo),
            _evaluate_closure(grid, s2, physics, thermo),
            solver,
        )
        if crossed is not None:
            return _fail(crossed)
        return _SplitAttempt(True, None, s2, f_conv, f_rad, w1 + w2, lhs1 + lhs2)
    except (ThermoDomainError, EnthalpyInversionError, HydrostaticDomainError) as exc:
        return _fail(str(exc))


def _energy_scale(work: float, f_scale: float, dt: float, cfg: RCEConfig) -> float:
    return max(abs(work), f_scale * dt, cfg.energy_scale_floor)


def _convergence_metrics(
    grid: PressureGrid,
    old_state: ColumnState,
    new_state: ColumnState,
    f_total: NDArray[np.float64],
    closure: ClosureResult,
    solver: SolverConfig,
    cfg: RCEConfig,
    previous_rcb: float | None,
    manufactured: ManufacturedRadiativeTarget | None,
) -> tuple[RCEConvergence, float, float | None, list[tuple[int, int]], float]:
    f_scale = max(cfg.flux_scale_floor, float(np.max(np.abs(f_total), initial=0.0)))
    if manufactured is None:
        f_ref = float(np.mean(f_total))
    else:
        f_ref = manufactured.f0
        f_scale = max(f_scale, abs(f_ref), cfg.flux_scale_floor)
    flux_flatness = float(np.max(np.abs(f_total - f_ref), initial=0.0)) / f_scale

    dh = new_state.enthalpy - old_state.enthalpy
    t_scale = np.maximum(np.abs(old_state.temperature), cfg.temp_scale_floor)
    temp_change = float(np.max(np.abs(new_state.temperature - old_state.temperature) / t_scale, initial=0.0))
    tendency_norm = float(np.max(np.abs(dh) / np.maximum(np.abs(old_state.enthalpy), cfg.temp_scale_floor), initial=0.0))

    rcb = _primary_rcb_log10p(grid, closure, solver)
    regions = _rcb_regions(closure, solver)
    if previous_rcb is None or rcb is None:
        rcb_stable = previous_rcb is None and rcb is None
    else:
        rcb_stable = abs(rcb - previous_rcb) <= cfg.rcb_stability_log10p_tolerance
    finite_state = bool(
        np.all(np.isfinite(new_state.temperature))
        and np.all(np.isfinite(new_state.enthalpy))
        and np.all(np.isfinite(new_state.mass_path))
        and np.all(new_state.temperature > 0.0)
        and np.all(new_state.mass_path > 0.0)
    )
    conv = RCEConvergence(
        flux_flatness=flux_flatness,
        tendency_norm=tendency_norm,
        temp_change=temp_change,
        rcb_stable=rcb_stable,
        finite_state=finite_state,
    )
    return conv, f_ref, rcb, regions, f_scale


def _rejected_diag(
    dt: float,
    route: RCERoute,
    dt_mlt: float,
    dt_rad: float,
    dt_temp: float,
    reason: str,
) -> RCEStepDiagnostics:
    return RCEStepDiagnostics(
        dt=dt,
        accepted=False,
        route=route,
        dt_mlt=dt_mlt,
        dt_rad=dt_rad,
        dt_temp=dt_temp,
        flux_boundary_work=float("nan"),
        energy_lhs=float("nan"),
        energy_residual=float("nan"),
        energy_residual_rel=float("nan"),
        flux_flatness=float("nan"),
        boundary_mismatch=float("nan"),
        temp_change=float("nan"),
        tendency_norm=float("nan"),
        primary_rcb_log10p=None,
        n_bottom_connected_regions=0,
        rejection_reason=reason,
    )


def solve_adaptive_rce(
    grid: PressureGrid,
    initial_temperature: NDArray[np.float64],
    physics: PhysicsConfig,
    solver: SolverConfig,
    thermo: ThermoProvider,
    opacity: PrescribedOpacity,
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    lower_bc: LowerFlux | LowerTemperature,
    *,
    gravity: GravityLaw | None = None,
    route: RCERoute = RCERoute.UNSPLIT,
    config: RCEConfig | None = None,
    manufactured: ManufacturedRadiativeTarget | None = None,
) -> RCEResult:
    cfg = config or RCEConfig()
    grav = gravity or ConstantGravity(physics.gravity)
    state = build_column_state(grid, np.asarray(initial_temperature, dtype=np.float64), thermo, grav)

    accepted_consec = 0
    prev_rcb: float | None = None
    diagnostics: list[RCEStepDiagnostics] = []
    simulated_time = 0.0
    rejections = 0
    best_resid = np.inf
    stall_counter = 0
    steps_accepted = 0

    final_closure = _evaluate_closure(grid, state, physics, thermo)
    final_rad: RadiationResult | None = None
    final_f_conv = np.zeros(grid.n_layers + 1)
    final_f_rad = np.zeros(grid.n_layers + 1)
    final_f_total = np.zeros(grid.n_layers + 1)
    final_regions: list[tuple[int, int]] = []
    final_conv = RCEConvergence(np.inf, np.inf, np.inf, False, False)
    final_rcb = None
    status = RCETerminalStatus.MAX_STEPS
    reason = "maximum step budget reached"

    for _step in range(cfg.max_steps):
        closure_for_dt, rad_for_dt, f_conv_for_dt, f_rad_for_dt, f_total_for_dt = _run_unsplit(
            grid, state, physics, thermo, opacity, pressure, top_bc, lower_bc, cfg, manufactured, grav
        )
        dt_mlt = _dt_mlt_estimate(grid, state, closure_for_dt, solver)
        if manufactured is None and rad_for_dt is not None:
            dt_rad = _dt_rad_estimate(state, rad_for_dt.heating, solver, thermo)
        else:
            dt_rad = _dt_rad_estimate(
                state, enthalpy_tendency(grid, f_rad_for_dt, state.mass_path), solver, thermo
            )
        dhdt_total = enthalpy_tendency(grid, f_total_for_dt, state.mass_path)
        dt_temp = _dt_temp_estimate(state, dhdt_total, solver, thermo)
        if cfg.prescribed_dt is not None:
            dt = float(cfg.prescribed_dt)
        else:
            dt = min(dt_mlt, dt_rad, dt_temp)
        if not np.isfinite(dt) or dt < cfg.dt_min:
            status = RCETerminalStatus.DT_MIN_FAILURE
            reason = f"timestep below dt_min: dt={dt}"
            break

        accepted = False
        rejection_reason = "unknown"

        for _attempt in range(cfg.max_rejections + 1):
            old_state = state
            if route == RCERoute.UNSPLIT:
                closure, rad, f_conv, f_rad, f_total = _run_unsplit(
                    grid, state, physics, thermo, opacity, pressure, top_bc, lower_bc, cfg, manufactured, grav
                )
                dhdt = enthalpy_tendency(grid, f_total, state.mass_path)
                trial_state, trial_reason = _trial_atomic_state(
                    grid, state, dhdt, dt, thermo, grav, solver
                )
                if trial_state is None:
                    rejection_reason = trial_reason or "unsplit trial failed"
                    rejections += 1
                    diagnostics.append(_rejected_diag(dt, route, dt_mlt, dt_rad, dt_temp, rejection_reason))
                    dt *= cfg.f_back
                    if dt < cfg.dt_min:
                        break
                    continue
                crossed = _crossing_reason(closure, _evaluate_closure(grid, trial_state, physics, thermo), solver)
                if crossed is not None:
                    rejection_reason = crossed
                    rejections += 1
                    diagnostics.append(_rejected_diag(dt, route, dt_mlt, dt_rad, dt_temp, rejection_reason))
                    dt *= cfg.f_back
                    if dt < cfg.dt_min:
                        break
                    continue
                boundary_work = dt * float(f_total[0] - f_total[-1])
                energy_lhs = float(dt * np.sum(state.mass_path * dhdt))
                energy_resid = energy_lhs - boundary_work
            else:
                attempt = _run_split_macrostep(
                    route, grid, state, dt, physics, thermo, grav, opacity, pressure,
                    top_bc, lower_bc, cfg, manufactured, solver,
                )
                if not attempt.ok:
                    rejection_reason = attempt.reason or "split macrostep failed"
                    rejections += 1
                    diagnostics.append(_rejected_diag(dt, route, dt_mlt, dt_rad, dt_temp, rejection_reason))
                    dt *= cfg.f_back
                    if dt < cfg.dt_min:
                        break
                    continue
                trial_state = attempt.state
                f_conv = attempt.f_conv
                f_rad = attempt.f_rad
                boundary_work = attempt.boundary_work
                energy_lhs = attempt.energy_lhs
                energy_resid = energy_lhs - boundary_work
                # Convergence / flatness from unsplit operators on the committed trial.
                closure, rad, f_conv, f_rad, f_total = _run_unsplit(
                    grid, trial_state, physics, thermo, opacity, pressure, top_bc, lower_bc,
                    cfg, manufactured, grav,
                )

            conv, _f_ref, rcb, regions, f_scale = _convergence_metrics(
                grid, old_state, trial_state, f_total, closure, solver, cfg, prev_rcb, manufactured
            )
            boundary_mismatch = abs(float(f_total[0] - f_total[-1])) / f_scale
            e_scale = _energy_scale(boundary_work, f_scale, dt, cfg)

            state = trial_state
            simulated_time += dt
            steps_accepted += 1
            accepted = True

            diagnostics.append(
                RCEStepDiagnostics(
                    dt=dt,
                    accepted=True,
                    route=route,
                    dt_mlt=dt_mlt,
                    dt_rad=dt_rad,
                    dt_temp=dt_temp,
                    flux_boundary_work=boundary_work,
                    energy_lhs=energy_lhs,
                    energy_residual=energy_resid,
                    energy_residual_rel=abs(energy_resid) / e_scale,
                    flux_flatness=conv.flux_flatness,
                    boundary_mismatch=boundary_mismatch,
                    temp_change=conv.temp_change,
                    tendency_norm=conv.tendency_norm,
                    primary_rcb_log10p=rcb,
                    n_bottom_connected_regions=len(regions),
                )
            )

            final_closure = closure
            final_rad = rad
            final_f_conv = f_conv
            final_f_rad = f_rad
            final_f_total = f_total
            final_regions = regions
            final_conv = conv
            final_rcb = rcb

            gate_ok = (
                conv.flux_flatness <= cfg.flux_flatness_tolerance
                and conv.tendency_norm <= cfg.tendency_tolerance
                and conv.temp_change <= cfg.temp_change_tolerance
                and conv.rcb_stable
                and conv.finite_state
            )
            if gate_ok:
                accepted_consec += 1
            else:
                accepted_consec = 0

            resid_scalar = max(conv.flux_flatness, conv.tendency_norm, conv.temp_change)
            if resid_scalar < best_resid * (1.0 - cfg.stall_rel_improvement):
                best_resid = resid_scalar
                stall_counter = 0
            else:
                stall_counter += 1

            prev_rcb = rcb
            if accepted_consec >= cfg.n_consec:
                status = RCETerminalStatus.CONVERGED
                reason = f"converged for {cfg.n_consec} consecutive accepted steps"
            elif stall_counter >= cfg.stall_window:
                status = RCETerminalStatus.STALLED
                reason = "residuals/mask stalled"
            break

        if accepted and status in (RCETerminalStatus.CONVERGED, RCETerminalStatus.STALLED):
            break
        if not accepted:
            if dt < cfg.dt_min:
                status = RCETerminalStatus.DT_MIN_FAILURE
                reason = f"dt fell below dt_min after rejections ({rejection_reason})"
                break
            continue
    else:
        status = RCETerminalStatus.MAX_STEPS
        reason = "maximum step budget reached"

    # Refresh unsplit operators on the committed state. Never replace a
    # manufactured radiative flux with a real RT solve.
    closure, rad, f_conv, f_rad, f_total = _run_unsplit(
        grid, state, physics, thermo, opacity, pressure, top_bc, lower_bc, cfg, manufactured, grav
    )
    final_closure = closure
    if manufactured is None:
        final_rad = rad
    final_f_conv = f_conv
    final_f_rad = f_rad
    final_f_total = f_total
    final_conv, _, final_rcb, final_regions, _ = _convergence_metrics(
        grid, state, state, final_f_total, final_closure, solver, cfg, prev_rcb, manufactured
    )

    return RCEResult(
        status=status,
        reason=reason,
        route=route,
        steps_attempted=len(diagnostics),
        steps_accepted=steps_accepted,
        rejections=rejections,
        simulated_time=simulated_time,
        final_state=state,
        final_closure=final_closure,
        final_radiation=final_rad if final_rad is not None else _empty_radiation(grid.n_layers),
        final_flux_total=final_f_total,
        final_flux_conv=final_f_conv,
        final_flux_rad=final_f_rad,
        primary_rcb_log10p=final_rcb,
        convective_regions=final_regions,
        convergence=final_conv,
        diagnostics=diagnostics,
    )
