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
from .grid import PressureGrid, build_grid
from .hydrostatics import HydrostaticDomainError
from .implicit_convection import (
    ImplicitConvectionConfig,
    ImplicitConvectionDiagnostics,
    require_constant_gravity,
    solve_implicit_convection,
)
from .opacity import AnalyticGreyOpacity, PrescribedOpacity
from .radiation import (
    DEFAULT_DIFFUSIVITY,
    LowerBoundary,
    LowerNetInternalFlux,
    RadiationResult,
    SolveRoute,
    TopIrradiation,
    solve_radiation,
)
from .solvers_enthalpy import _crossing_reason
from .state import ColumnState, build_column_state
from .thermodynamics import (
    ConstantH2Thermo,
    EnthalpyInversionError,
    ThermoDomainError,
    ThermoProvider,
    invert_psi_newton,
)


class RCERoute(str, Enum):
    UNSPLIT = "unsplit"
    SPLIT_RAD_THEN_CONV = "split_rad_then_conv"
    SPLIT_CONV_THEN_RAD = "split_conv_then_rad"
    SPLIT_RAD_THEN_IMPLICIT_CONV = "split_rad_then_implicit_conv"
    SPLIT_IMPLICIT_CONV_THEN_RAD = "split_implicit_conv_then_rad"
    SPLIT_STRANG_RAD_IMPLICIT_CONV = "split_strang_rad_implicit_conv"


class RCETerminalStatus(str, Enum):
    CONVERGED = "converged"
    MAX_STEPS = "max_steps"
    DT_MIN_FAILURE = "dt_min_failure"
    STALLED = "stalled"
    PRESCRIBED_DT_REJECTED = "prescribed_dt_rejected"


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
    energy_committed: float = float("nan")
    energy_committed_residual: float = float("nan")
    energy_committed_residual_rel: float = float("nan")
    energy_ulp_floor: float = float("nan")
    rejection_reason: str | None = None
    nonlinear_residual: float = float("nan")
    newton_iterations: int = 0
    line_search_backtracks: int = 0
    mask_outer_iterations: int = 0
    mlt_evals: int = 0


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
    detached_convective_regions: list[tuple[int, int]]
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
    # Implicit-route growth after an accepted step. Explicit routes still use 1/f_back.
    f_grow: float = 1.2
    n_hold_after_reject: int = 8
    stall_window: int = 2000
    stall_rel_improvement: float = 1e-6
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY
    radiation_route: SolveRoute = SolveRoute.THOMAS
    prescribed_dt: float | None = None
    t_final: float | None = None
    attraction_temperature_tolerance: float | None = None
    flux_scale_floor: float = 1e-30
    temp_scale_floor: float = 1e-12
    energy_scale_floor: float = 1e-30
    # Optional macrostep accuracy bound for implicit convection only.
    dt_accuracy: float | None = None
    implicit_convection: ImplicitConvectionConfig | None = None


def _uses_implicit_convection(route: RCERoute) -> bool:
    return route in (
        RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        RCERoute.SPLIT_IMPLICIT_CONV_THEN_RAD,
        RCERoute.SPLIT_STRANG_RAD_IMPLICIT_CONV,
    )


def dt_after_reject(dt: float, cfg: RCEConfig) -> tuple[float, float, int]:
    """Halve dt and install a ceiling held for n_hold_after_reject accepts."""
    reduced = float(dt) * cfg.f_back
    return reduced, reduced, int(cfg.n_hold_after_reject)


def dt_after_accept(
    dt: float,
    dt_est: float,
    cfg: RCEConfig,
    *,
    implicit: bool,
    dt_ceiling: float | None,
    hold_remaining: int,
) -> tuple[float, float | None, int]:
    """Next attempted dt. Implicit routes grow by f_grow, not 1/f_back."""
    if not implicit:
        grown = dt / cfg.f_back if cfg.f_back > 0.0 else dt
        nxt = min(grown, dt_est) if np.isfinite(dt_est) else grown
        return float(nxt), None, 0
    grown = dt * cfg.f_grow
    nxt = grown
    if np.isfinite(dt_est):
        nxt = min(nxt, dt_est)
    if dt_ceiling is not None and hold_remaining > 0:
        nxt = min(nxt, dt_ceiling)
        hold_remaining -= 1
        if hold_remaining <= 0:
            dt_ceiling = None
    return float(nxt), dt_ceiling, hold_remaining


@dataclass(frozen=True)
class AnalyticOpacityRCESpec:
    """Pressure-dependent grey column for a bottom-connected coupled RCE.

    κ(P) = κ₀ (P/P₀)^a with b = 0 keeps the discrete radiative seed exact.
    At depth ∇_rad → (a+1)/4; a > 1/7 is required for a deep convective
    region in diatomic H₂ (∇_ad = 2/7). A log-P grid puts almost all of
    τ into the bottom cell and never recovers that gradient, so the
    coupled column is built uniform in τ in the interior with a
    geometrically refined photosphere.
    """

    gravity: float = 15.0
    p_bottom: float = 1.0e6
    p_top: float = 1.0
    a: float = 0.5
    b: float = 0.0
    tau_total: float = 100.0
    f_int: float = 300.0
    f_irr: float = 120.0
    n_layers: int = 48
    n_photosphere: int = 16
    tau_photosphere: float = 1.0
    tau_photosphere_min: float = 1.0e-4
    alpha: float = 1.0
    closure_prefactor: float = 0.5
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY

    @property
    def kappa0(self) -> float:
        return self.tau_total * self.gravity * (self.a + 1.0) / self.p_bottom

    def opacity(self) -> AnalyticGreyOpacity:
        return AnalyticGreyOpacity(
            kappa0=self.kappa0,
            P0=self.p_bottom,
            T0=1.0,
            a=self.a,
            b=self.b,
        )

    def pressure_edges(self) -> NDArray[np.float64]:
        return analytic_opacity_pressure_edges(self)

    def grid(self) -> PressureGrid:
        return build_grid(self.pressure_edges(), self.gravity)

    def physics(self) -> PhysicsConfig:
        return PhysicsConfig(
            gravity=self.gravity,
            alpha=self.alpha,
            closure_prefactor=self.closure_prefactor,
        )


def analytic_opacity_pressure_edges(spec: AnalyticOpacityRCESpec) -> NDArray[np.float64]:
    """Interior uniform in τ, geometrically refined photosphere.

    For κ ∝ P^a, τ(P) ∝ P^{a+1} − P_top^{a+1}. Equal-τ spacing recovers
    ∇_rad → (a+1)/4 at depth. Extra photospheric layers keep DΔτ_top ≪ 1
    and prevent a single top cell from spanning decades of pressure.
    """
    if spec.n_photosphere < 2 or spec.n_photosphere >= spec.n_layers - 1:
        raise ValueError("n_photosphere must be in [2, n_layers-2]")
    if spec.tau_photosphere <= spec.tau_photosphere_min:
        raise ValueError("tau_photosphere must exceed tau_photosphere_min")
    if spec.tau_photosphere >= spec.tau_total:
        raise ValueError("tau_photosphere must be less than tau_total")
    a = spec.a
    exponent = a + 1.0

    def p_of_tau(tau: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            spec.p_top ** exponent + (tau / spec.tau_total) * spec.p_bottom ** exponent
        ) ** (1.0 / exponent)

    n_deep = spec.n_layers - spec.n_photosphere
    tau_top = np.concatenate(
        [
            np.asarray([0.0]),
            np.geomspace(spec.tau_photosphere_min, spec.tau_photosphere, spec.n_photosphere),
        ]
    )
    tau_deep = np.linspace(spec.tau_photosphere, spec.tau_total, n_deep + 1)[1:]
    tau = np.concatenate([tau_top, tau_deep])
    if tau.size != spec.n_layers + 1:
        raise ValueError("optical-depth edge count must be n_layers+1")
    pressure = p_of_tau(tau)[::-1]
    pressure[0] = spec.p_bottom
    pressure[-1] = spec.p_top
    if np.any(np.diff(pressure) >= 0.0):
        raise ValueError("analytic-opacity pressure edges must be strictly decreasing")
    return pressure


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


def grey_radiative_equilibrium_temperature(
    grid: PressureGrid,
    opacity: PrescribedOpacity,
    f_int: float,
    f_irr: float,
    *,
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY,
    pressure: NDArray[np.float64] | None = None,
    temperature_seed: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Exact discrete grey absorption-only RE for constant net flux F_int.

    With F↑ = F↓ + F_int and the two-stream sweep, the layer source is
    B_i = F↓[i+1] + F_int / (1 + 𝒯_i). For a single band this is σ T_i⁴.
    Mass paths are ΔP/g on ``grid``; opacity is evaluated at ``temperature_seed``
    if it depends on T (constant grey ignores the seed).
    """
    from .radiation import STEFAN_BOLTZMANN

    n = grid.n_layers
    p = grid.pressure_centres if pressure is None else np.asarray(pressure, dtype=np.float64)
    if temperature_seed is None:
        t_seed = np.full(n, max((abs(f_int) + abs(f_irr)) / STEFAN_BOLTZMANN, 1.0) ** 0.25)
    else:
        t_seed = np.asarray(temperature_seed, dtype=np.float64)
    kappa = opacity.evaluate(t_seed, p)
    weights = opacity.band_weights
    if kappa.shape[0] != 1 or weights.shape != (1,):
        raise ValueError("grey RE guess requires a single opacity band")
    dtau = diffusivity_factor * kappa[0] * grid.layer_mass
    trans = np.exp(-dtau)
    f_down = np.zeros(n + 1, dtype=np.float64)
    f_down[n] = float(f_irr)
    for i in range(n - 1, -1, -1):
        f_down[i] = f_down[i + 1] + float(f_int) * (1.0 - trans[i]) / (1.0 + trans[i])
    source = f_down[1:] + float(f_int) / (1.0 + trans)
    if np.any(source <= 0.0) or not np.all(np.isfinite(source)):
        raise ValueError("grey RE source must be finite and positive")
    return (source / STEFAN_BOLTZMANN) ** 0.25


def grey_layer_optical_thickness(
    grid: PressureGrid,
    opacity: PrescribedOpacity,
    temperature: NDArray[np.float64],
    *,
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY,
    pressure: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """D Δτ per layer for a grey (single-band) opacity."""
    p = grid.pressure_centres if pressure is None else np.asarray(pressure, dtype=np.float64)
    kappa = opacity.evaluate(np.asarray(temperature, dtype=np.float64), p)
    if kappa.shape[0] != 1:
        raise ValueError("grey_layer_optical_thickness requires a single opacity band")
    return diffusivity_factor * kappa[0] * grid.layer_mass


def _temperature_on_adiabat(
    thermo: ThermoProvider,
    t_join: float,
    p_join: float,
    pressure: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Deep convective seed: power-law for ConstantH2Thermo, isentrope otherwise."""
    if isinstance(thermo, ConstantH2Thermo):
        return t_join * (pressure / p_join) ** float(thermo.nabla_ad)
    s_join = float(thermo.entropy(np.asarray([t_join]), np.asarray([p_join]))[0])
    target_psi = s_join + thermo.gas_constant * np.log(pressure / thermo.p_ref)
    return invert_psi_newton(
        thermo,
        target_psi,
        t_min=float(thermo.t_min),
        t_max=float(thermo.t_max),
    )


def radiative_convective_initial_temperature(
    grid: PressureGrid,
    opacity: PrescribedOpacity,
    thermo: ThermoProvider,
    f_int: float,
    f_irr: float,
    *,
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY,
    pressure: NDArray[np.float64] | None = None,
    join_layers: int = 1,
) -> NDArray[np.float64]:
    """Grey RE estimate with an adiabat only in the bottom-connected unstable region.

    If the first internal interface is stable, detached upper unstable segments
    are left on the radiative-equilibrium seed. Smooths solely across the
    radiative–convective join. Does not cap F_conv, reduce α, or clip
    superadiabaticity after the join is set. Production thermodynamics use a
    constant-entropy inversion; ConstantH2Thermo keeps the exact power-law
    adiabat.
    """
    t_re = grey_radiative_equilibrium_temperature(
        grid, opacity, f_int, f_irr,
        diffusivity_factor=diffusivity_factor, pressure=pressure,
    )
    n = t_re.size
    if n < 2:
        return t_re
    log_t = np.log(t_re)
    log_p = np.log(grid.pressure_centres)
    nabla = (log_t[:-1] - log_t[1:]) / (log_p[:-1] - log_p[1:])
    nabla_ad = thermo.nabla_ad_at(t_re)
    nabla_ad_iface = 0.5 * (nabla_ad[:-1] + nabla_ad[1:])
    if nabla[0] <= nabla_ad_iface[0]:
        return t_re
    i = 0
    n_int = nabla.size
    while i < n_int and nabla[i] > nabla_ad_iface[i]:
        i += 1
    i_join = min(i, n - 1)
    p_join = float(grid.pressure_centres[i_join])
    t_join = float(t_re[i_join])
    t = t_re.copy()
    p_cz = grid.pressure_centres[:i_join]
    t_ad_cz = _temperature_on_adiabat(thermo, t_join, p_join, p_cz)
    t[:i_join] = t_ad_cz
    width = max(int(join_layers), 0)
    for k in range(max(0, i_join - width), i_join):
        t[k] = np.sqrt(t_ad_cz[k] * t_re[k])
    if np.any(t <= 0.0) or not np.all(np.isfinite(t)):
        raise ValueError("radiative–convective initial temperature must be finite and positive")
    return t


def _internal_flux_reference(
    lower_bc: LowerBoundary,
    manufactured: ManufacturedRadiativeTarget | None,
) -> float | None:
    if manufactured is not None:
        return float(manufactured.f0)
    if isinstance(lower_bc, LowerNetInternalFlux):
        return float(lower_bc.flux)
    return None


def _active_internal(closure: ClosureResult, solver: SolverConfig) -> tuple[NDArray[np.bool_], float]:
    threshold = solver.c_active * solver.epsilon_gradient
    return closure.superadiabaticity[1:-1] > threshold, threshold


def _rcb_regions(closure: ClosureResult, solver: SolverConfig) -> list[tuple[int, int]]:
    """Contiguous components of *active* internal interfaces only.

    Region (i_lo, i_hi) spans layers i_lo … i_hi connected by active
    interfaces i_lo+1 … i_hi. Stable singleton layers are not regions.
    """
    active, _ = _active_internal(closure, solver)
    regions: list[tuple[int, int]] = []
    j = 0
    n = active.size
    while j < n:
        if not active[j]:
            j += 1
            continue
        j0 = j
        while j < n and active[j]:
            j += 1
        regions.append((j0, j))
    return regions


def _partition_rcb_regions(
    regions: list[tuple[int, int]],
    active: NDArray[np.bool_],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bottom: list[tuple[int, int]] = []
    detached: list[tuple[int, int]] = []
    for region in regions:
        if region[0] == 0 and active.size > 0 and bool(active[0]):
            bottom.append(region)
        else:
            detached.append(region)
    return bottom, detached


def _primary_rcb_log10p(grid: PressureGrid, closure: ClosureResult, solver: SolverConfig) -> float | None:
    """Top of the deepest bottom-connected convective region, or None.

    Interpolation uses the activity threshold, not zero superadiabaticity.
    """
    active, threshold = _active_internal(closure, solver)
    if active.size == 0 or not bool(active[0]):
        return None
    idx = 0
    while idx < active.size and active[idx]:
        idx += 1
    if idx == active.size:
        return float(np.log10(grid.pressure_edges[-1]))
    i_act = idx
    i_inact = idx + 1
    d_lo = closure.superadiabaticity[i_act]
    d_hi = closure.superadiabaticity[i_inact]
    p_lo = grid.pressure_edges[i_act]
    p_hi = grid.pressure_edges[i_inact]
    if np.isfinite(d_lo) and np.isfinite(d_hi) and (d_lo - d_hi) != 0.0:
        w = (d_lo - threshold) / (d_lo - d_hi)
        w = float(np.clip(w, 0.0, 1.0))
        return float((1.0 - w) * np.log10(p_lo) + w * np.log10(p_hi))
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
    lower_bc: LowerBoundary,
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
            bottom_convective_flux=float(closure.flux[0]),
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
    implicit: ImplicitConvectionDiagnostics | None = None


def _external_flux_at_state(
    grid: PressureGrid,
    state: ColumnState,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    opacity: PrescribedOpacity,
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    lower_bc: LowerBoundary,
    cfg: RCEConfig,
    manufactured: ManufacturedRadiativeTarget | None,
    gravity: GravityLaw,
    *,
    prescribed_f_ext: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Stage 3 radiation, manufactured F_ext, or a prescribed interface flux."""
    if prescribed_f_ext is not None:
        f = np.asarray(prescribed_f_ext, dtype=np.float64)
        if f.shape != (grid.n_layers + 1,):
            raise ValueError("prescribed_f_ext must have length n_layers+1")
        return f
    if manufactured is None:
        f_conv0 = float(_evaluate_closure(grid, state, physics, thermo).flux[0])
        rr = solve_radiation(
            state.temperature, state.mass_path, opacity, pressure, top_bc, lower_bc,
            cfg.diffusivity_factor, cfg.radiation_route,
            bottom_convective_flux=f_conv0,
        )
        return rr.flux_net
    target_state = build_column_state(grid, manufactured.target_temperature, thermo, gravity)
    closure_target = _evaluate_closure(grid, target_state, physics, thermo)
    return _build_rad_from_target(
        grid, state, closure_target, manufactured, target_enthalpy=target_state.enthalpy
    )


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
    lower_bc: LowerBoundary,
    cfg: RCEConfig,
    manufactured: ManufacturedRadiativeTarget | None,
    solver: SolverConfig,
    *,
    prescribed_f_ext: NDArray[np.float64] | None = None,
) -> _SplitAttempt:
    nan_f = np.full(grid.n_layers + 1, np.nan)

    def _fail(reason: str, implicit: ImplicitConvectionDiagnostics | None = None) -> _SplitAttempt:
        return _SplitAttempt(False, reason, state, nan_f, nan_f, float("nan"), float("nan"), implicit)

    def rad_substep(s: ColumnState) -> tuple[ColumnState, NDArray[np.float64], float, float]:
        f_rad = _external_flux_at_state(
            grid, s, physics, thermo, opacity, pressure, top_bc, lower_bc, cfg,
            manufactured, gravity, prescribed_f_ext=prescribed_f_ext,
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

    def implicit_conv_substep(
        s_commit: ColumnState,
        h_star: NDArray[np.float64],
        dt_conv: float,
    ) -> tuple[ColumnState, NDArray[np.float64], float, float, ImplicitConvectionDiagnostics]:
        mass = s_commit.mass_path
        result = solve_implicit_convection(
            grid,
            s_commit,
            np.asarray(h_star, dtype=np.float64).copy(),
            physics,
            thermo,
            gravity,
            mass,
            dt_conv,
            solver,
            cfg=cfg.implicit_convection,
        )
        if not result.ok:
            raise ThermoDomainError(result.diagnostics.rejection_reason or "implicit convection failed")
        f_conv = result.f_conv
        work = dt_conv * float(f_conv[0] - f_conv[-1])  # ~0 by construction
        lhs = float(np.sum(mass * (result.state.enthalpy - h_star)))
        return result.state, f_conv, work, lhs, result.diagnostics

    def rad_substep_dt(
        s: ColumnState, dt_rad_step: float
    ) -> tuple[ColumnState, NDArray[np.float64], float, float]:
        f_rad = _external_flux_at_state(
            grid, s, physics, thermo, opacity, pressure, top_bc, lower_bc, cfg,
            manufactured, gravity, prescribed_f_ext=prescribed_f_ext,
        )
        dhdt = enthalpy_tendency(grid, f_rad, s.mass_path)
        s_new, reason = _trial_atomic_state(grid, s, dhdt, dt_rad_step, thermo, gravity, solver)
        if s_new is None:
            raise ThermoDomainError(reason or "radiation substep failed")
        work = dt_rad_step * float(f_rad[0] - f_rad[-1])
        lhs = float(dt_rad_step * np.sum(s.mass_path * dhdt))
        return s_new, f_rad, work, lhs

    try:
        if route == RCERoute.SPLIT_RAD_THEN_CONV:
            s1, f_rad, w1, lhs1 = rad_substep(state)
            s2, f_conv, w2, lhs2 = conv_substep(s1)
            crossed = _crossing_reason(
                _evaluate_closure(grid, state, physics, thermo),
                _evaluate_closure(grid, s2, physics, thermo),
                solver,
            )
            if crossed is not None:
                return _fail(crossed)
            return _SplitAttempt(True, None, s2, f_conv, f_rad, w1 + w2, lhs1 + lhs2)
        if route == RCERoute.SPLIT_CONV_THEN_RAD:
            s1, f_conv, w1, lhs1 = conv_substep(state)
            s2, f_rad, w2, lhs2 = rad_substep(s1)
            crossed = _crossing_reason(
                _evaluate_closure(grid, state, physics, thermo),
                _evaluate_closure(grid, s2, physics, thermo),
                solver,
            )
            if crossed is not None:
                return _fail(crossed)
            return _SplitAttempt(True, None, s2, f_conv, f_rad, w1 + w2, lhs1 + lhs2)
        if route == RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV:
            s1, f_rad, w1, lhs1 = rad_substep(state)
            s2, f_conv, w2, lhs2, idiag = implicit_conv_substep(s1, s1.enthalpy, dt)
            return _SplitAttempt(True, None, s2, f_conv, f_rad, w1 + w2, lhs1 + lhs2, idiag)
        if route == RCERoute.SPLIT_IMPLICIT_CONV_THEN_RAD:
            s1, f_conv, w1, lhs1, idiag = implicit_conv_substep(state, state.enthalpy, dt)
            s2, f_rad, w2, lhs2 = rad_substep(s1)
            return _SplitAttempt(True, None, s2, f_conv, f_rad, w1 + w2, lhs1 + lhs2, idiag)
        if route == RCERoute.SPLIT_STRANG_RAD_IMPLICIT_CONV:
            # Half radiation, full implicit convection, half radiation.
            s_a, f_rad_a, w_a, lhs_a = rad_substep_dt(state, 0.5 * dt)
            s_b, f_conv, w_b, lhs_b, idiag = implicit_conv_substep(s_a, s_a.enthalpy, dt)
            s_c, f_rad_c, w_c, lhs_c = rad_substep_dt(s_b, 0.5 * dt)
            return _SplitAttempt(
                True, None, s_c, f_conv, f_rad_c, w_a + w_b + w_c, lhs_a + lhs_b + lhs_c, idiag
            )
        return _fail(f"Unsupported split route {route}")
    except (ThermoDomainError, EnthalpyInversionError, HydrostaticDomainError) as exc:
        return _fail(str(exc))


def _energy_scale(work: float, f_scale: float, dt: float, cfg: RCEConfig) -> float:
    return max(abs(work), f_scale * dt, cfg.energy_scale_floor)


def _ulp_energy_floor(mass_path: NDArray[np.float64], enthalpy: NDArray[np.float64]) -> float:
    return float(np.sum(mass_path * np.abs(enthalpy) * np.finfo(np.float64).eps))


def _convergence_metrics(
    grid: PressureGrid,
    old_state: ColumnState,
    new_state: ColumnState,
    f_total: NDArray[np.float64],
    dhdt: NDArray[np.float64],
    closure: ClosureResult,
    solver: SolverConfig,
    cfg: RCEConfig,
    previous_rcb: float | None,
    f_int: float | None,
) -> tuple[RCEConvergence, float, float | None, list[tuple[int, int]], list[tuple[int, int]], float, float]:
    if f_int is not None:
        f_ref = float(f_int)
        f_scale = max(cfg.flux_scale_floor, abs(f_ref))
    else:
        f_ref = float(f_total[0])
        f_scale = max(cfg.flux_scale_floor, abs(f_ref))
    flux_flatness = float(np.max(np.abs(f_total - f_ref), initial=0.0)) / f_scale
    boundary_mismatch = max(abs(float(f_total[0]) - f_ref), abs(float(f_total[-1]) - f_ref)) / f_scale

    t_scale = np.maximum(np.abs(old_state.temperature), cfg.temp_scale_floor)
    temp_change = float(np.max(np.abs(new_state.temperature - old_state.temperature) / t_scale, initial=0.0))
    layer_div = np.abs(old_state.mass_path * dhdt)
    tendency_norm = float(np.max(layer_div, initial=0.0)) / f_scale

    rcb = _primary_rcb_log10p(grid, closure, solver)
    regions = _rcb_regions(closure, solver)
    active, _ = _active_internal(closure, solver)
    bottom_regions, detached_regions = _partition_rcb_regions(regions, active)
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
    return conv, f_ref, rcb, bottom_regions, detached_regions, f_scale, boundary_mismatch


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


def _attraction_temperature_residual(
    temperature: NDArray[np.float64],
    target: NDArray[np.float64],
    floor: float,
) -> float:
    scale = np.maximum(np.abs(target), floor)
    return float(np.max(np.abs(temperature - target) / scale, initial=0.0))


def _gate_ok(
    conv: RCEConvergence,
    cfg: RCEConfig,
    *,
    manufactured: ManufacturedRadiativeTarget | None = None,
    temperature: NDArray[np.float64] | None = None,
) -> bool:
    if (
        manufactured is not None
        and cfg.attraction_temperature_tolerance is not None
        and temperature is not None
    ):
        t_rel = _attraction_temperature_residual(
            temperature, manufactured.target_temperature, cfg.temp_scale_floor
        )
        return (
            t_rel <= cfg.attraction_temperature_tolerance
            and conv.temp_change <= cfg.temp_change_tolerance
            and conv.finite_state
            and conv.rcb_stable
        )
    return (
        conv.flux_flatness <= cfg.flux_flatness_tolerance
        and conv.tendency_norm <= cfg.tendency_tolerance
        and conv.temp_change <= cfg.temp_change_tolerance
        and conv.rcb_stable
        and conv.finite_state
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
    lower_bc: LowerBoundary,
    *,
    gravity: GravityLaw | None = None,
    route: RCERoute = RCERoute.UNSPLIT,
    config: RCEConfig | None = None,
    manufactured: ManufacturedRadiativeTarget | None = None,
) -> RCEResult:
    cfg = config or RCEConfig()
    grav = gravity or ConstantGravity(physics.gravity)
    if _uses_implicit_convection(route):
        require_constant_gravity(grav)
    state = build_column_state(grid, np.asarray(initial_temperature, dtype=np.float64), thermo, grav)
    f_int = _internal_flux_reference(lower_bc, manufactured)

    accepted_consec = 0
    prev_rcb: float | None = None
    diagnostics: list[RCEStepDiagnostics] = []
    simulated_time = 0.0
    rejections = 0
    best_resid = np.inf
    stall_counter = 0
    steps_accepted = 0
    dt_hold: float | None = None
    dt_ceiling: float | None = None
    hold_remaining = 0
    last_temp_change = float("inf")

    final_closure = _evaluate_closure(grid, state, physics, thermo)
    final_rad: RadiationResult | None = None
    final_f_conv = np.zeros(grid.n_layers + 1)
    final_f_rad = np.zeros(grid.n_layers + 1)
    final_f_total = np.zeros(grid.n_layers + 1)
    final_regions: list[tuple[int, int]] = []
    final_detached: list[tuple[int, int]] = []
    final_conv = RCEConvergence(np.inf, np.inf, np.inf, False, False)
    final_rcb = None
    status = RCETerminalStatus.MAX_STEPS
    reason = "maximum step budget reached"

    for _step in range(cfg.max_steps):
        if cfg.t_final is not None and simulated_time >= cfg.t_final:
            status = (
                RCETerminalStatus.CONVERGED
                if _gate_ok(final_conv, cfg, manufactured=manufactured, temperature=state.temperature)
                else RCETerminalStatus.MAX_STEPS
            )
            reason = "reached t_final"
            break

        closure_for_dt, rad_for_dt, _f_c0, f_rad_for_dt, f_total_for_dt = _run_unsplit(
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
        conv_now, _, rcb_now, bottom_now, detached_now, _, _ = _convergence_metrics(
            grid, state, state, f_total_for_dt, dhdt_total, closure_for_dt, solver, cfg, prev_rcb, f_int
        )
        radiation_only_control = physics.alpha == 0.0 and manufactured is None
        already_equilibrated = (
            radiation_only_control
            and conv_now.flux_flatness <= cfg.flux_flatness_tolerance
            and conv_now.tendency_norm <= cfg.tendency_tolerance
            and conv_now.finite_state
            and cfg.prescribed_dt is None
        )
        if already_equilibrated:
            final_closure = closure_for_dt
            final_rad = rad_for_dt
            final_f_conv = _f_c0
            final_f_rad = f_rad_for_dt
            final_f_total = f_total_for_dt
            final_regions = bottom_now + detached_now
            final_detached = detached_now
            final_rcb = rcb_now
            final_conv = RCEConvergence(
                flux_flatness=conv_now.flux_flatness,
                tendency_norm=conv_now.tendency_norm,
                temp_change=0.0 if steps_accepted == 0 else last_temp_change,
                rcb_stable=True if steps_accepted == 0 else conv_now.rcb_stable,
                finite_state=True,
            )
            status = RCETerminalStatus.CONVERGED
            reason = "instantaneous flux and tendency already within tolerance"
            break

        dt_temp = _dt_temp_estimate(state, dhdt_total, solver, thermo)
        if _uses_implicit_convection(route):
            # Radiation / external forcing only — never include explicit MLT CFL.
            dhdt_rad_only = enthalpy_tendency(grid, f_rad_for_dt, state.mass_path)
            dt_rad_T = _dt_temp_estimate(state, dhdt_rad_only, solver, thermo)
            candidates = [dt_rad, dt_rad_T]
            if cfg.dt_accuracy is not None and np.isfinite(cfg.dt_accuracy):
                candidates.append(float(cfg.dt_accuracy))
            dt_est = min(candidates)
            dt_mlt = float("inf")  # reported as unused for implicit convection
            dt_temp = dt_rad_T
        else:
            dt_est = min(dt_mlt, dt_rad, dt_temp)

        prescribed = cfg.prescribed_dt is not None
        if prescribed:
            dt = float(cfg.prescribed_dt)
        else:
            dt = dt_est
            if dt_hold is not None and np.isfinite(dt_hold):
                dt = min(dt, dt_hold) if np.isfinite(dt) else dt_hold
            if cfg.t_final is not None:
                remaining = cfg.t_final - simulated_time
                if remaining <= 0.0:
                    status = (
                        RCETerminalStatus.CONVERGED
                        if _gate_ok(final_conv, cfg, manufactured=manufactured, temperature=state.temperature)
                        else RCETerminalStatus.MAX_STEPS
                    )
                    reason = "reached t_final"
                    break
                if np.isfinite(dt):
                    dt = min(dt, remaining)

        if not np.isfinite(dt) or dt < cfg.dt_min:
            conv0, _, rcb0, bottom0, detached0, _, _ = _convergence_metrics(
                grid, state, state, f_total_for_dt, dhdt_total, closure_for_dt, solver, cfg, prev_rcb, f_int
            )
            conv0 = RCEConvergence(
                flux_flatness=conv0.flux_flatness,
                tendency_norm=conv0.tendency_norm,
                temp_change=0.0 if steps_accepted == 0 else last_temp_change,
                rcb_stable=True if steps_accepted == 0 else conv0.rcb_stable,
                finite_state=conv0.finite_state,
            )
            final_closure = closure_for_dt
            final_rad = rad_for_dt
            final_f_conv = _evaluate_closure(grid, state, physics, thermo).flux
            final_f_rad = f_rad_for_dt
            final_f_total = f_total_for_dt
            final_regions = bottom0 + detached0
            final_detached = detached0
            final_conv = conv0
            final_rcb = rcb0
            if (
                physics.alpha == 0.0
                and manufactured is None
                and conv0.flux_flatness <= cfg.flux_flatness_tolerance
                and conv0.tendency_norm <= cfg.tendency_tolerance
                and conv0.finite_state
            ):
                status = RCETerminalStatus.CONVERGED
                reason = "equilibrium: all timestep estimates infinite or below dt_min with residuals in tolerance"
            else:
                status = RCETerminalStatus.DT_MIN_FAILURE
                reason = f"timestep below dt_min: dt={dt}"
            break

        accepted = False
        rejection_reason = "unknown"
        n_attempts = 1 if prescribed else (cfg.max_rejections + 1)

        for _attempt in range(n_attempts):
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
                    if prescribed:
                        break
                    dt, dt_ceiling, hold_remaining = dt_after_reject(dt, cfg)
                    dt_hold = dt
                    if dt < cfg.dt_min:
                        break
                    continue
                crossed = _crossing_reason(closure, _evaluate_closure(grid, trial_state, physics, thermo), solver)
                if crossed is not None:
                    rejection_reason = crossed
                    rejections += 1
                    diagnostics.append(_rejected_diag(dt, route, dt_mlt, dt_rad, dt_temp, rejection_reason))
                    if prescribed:
                        break
                    dt, dt_ceiling, hold_remaining = dt_after_reject(dt, cfg)
                    dt_hold = dt
                    if dt < cfg.dt_min:
                        break
                    continue
                boundary_work = dt * float(f_total[0] - f_total[-1])
                energy_lhs = float(dt * np.sum(state.mass_path * dhdt))
                energy_resid = energy_lhs - boundary_work
                energy_committed = float(np.sum(old_state.mass_path * (trial_state.enthalpy - old_state.enthalpy)))
                implicit_diag = None
            else:
                attempt = _run_split_macrostep(
                    route, grid, state, dt, physics, thermo, grav, opacity, pressure,
                    top_bc, lower_bc, cfg, manufactured, solver,
                )
                if not attempt.ok:
                    rejection_reason = attempt.reason or "split macrostep failed"
                    rejections += 1
                    diagnostics.append(_rejected_diag(dt, route, dt_mlt, dt_rad, dt_temp, rejection_reason))
                    if prescribed:
                        break
                    dt, dt_ceiling, hold_remaining = dt_after_reject(dt, cfg)
                    dt_hold = dt
                    if dt < cfg.dt_min:
                        break
                    continue
                trial_state = attempt.state
                f_conv = attempt.f_conv
                f_rad = attempt.f_rad
                boundary_work = attempt.boundary_work
                energy_lhs = attempt.energy_lhs
                energy_resid = energy_lhs - boundary_work
                energy_committed = float(np.sum(old_state.mass_path * (trial_state.enthalpy - old_state.enthalpy)))
                # Recompute coupled metrics from F(T^{n+1}), not cached substep fluxes.
                closure, rad, f_conv, f_rad, f_total = _run_unsplit(
                    grid, trial_state, physics, thermo, opacity, pressure, top_bc, lower_bc,
                    cfg, manufactured, grav,
                )
                dhdt = enthalpy_tendency(grid, f_total, trial_state.mass_path)
                implicit_diag = attempt.implicit

            energy_committed_resid = energy_committed - boundary_work
            ulp_floor = _ulp_energy_floor(old_state.mass_path, trial_state.enthalpy)
            conv, _f_ref, rcb, bottom_regions, detached_regions, f_scale, boundary_mismatch = _convergence_metrics(
                grid, old_state, trial_state, f_total, dhdt, closure, solver, cfg, prev_rcb, f_int
            )
            e_scale = _energy_scale(boundary_work, f_scale, dt, cfg)
            e_committed_scale = max(e_scale, ulp_floor)

            state = trial_state
            simulated_time += dt
            steps_accepted += 1
            accepted = True
            last_temp_change = conv.temp_change
            if not prescribed:
                dt_hold, dt_ceiling, hold_remaining = dt_after_accept(
                    dt,
                    dt_est,
                    cfg,
                    implicit=_uses_implicit_convection(route),
                    dt_ceiling=dt_ceiling,
                    hold_remaining=hold_remaining,
                )

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
                    n_bottom_connected_regions=len(bottom_regions),
                    energy_committed=energy_committed,
                    energy_committed_residual=energy_committed_resid,
                    energy_committed_residual_rel=abs(energy_committed_resid) / e_committed_scale,
                    energy_ulp_floor=ulp_floor,
                    nonlinear_residual=(
                        float("nan") if implicit_diag is None else implicit_diag.residual_norm
                    ),
                    newton_iterations=0 if implicit_diag is None else implicit_diag.newton_iterations,
                    line_search_backtracks=(
                        0 if implicit_diag is None else implicit_diag.line_search_backtracks
                    ),
                    mask_outer_iterations=(
                        0 if implicit_diag is None else implicit_diag.mask_outer_iterations
                    ),
                    mlt_evals=0 if implicit_diag is None else implicit_diag.mlt_evals,
                )
            )

            final_closure = closure
            final_rad = rad
            final_f_conv = f_conv
            final_f_rad = f_rad
            final_f_total = f_total
            final_regions = bottom_regions + detached_regions
            final_detached = detached_regions
            final_conv = conv
            final_rcb = rcb

            if _gate_ok(conv, cfg, manufactured=manufactured, temperature=state.temperature):
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
            if prescribed:
                status = RCETerminalStatus.PRESCRIBED_DT_REJECTED
                reason = f"prescribed_dt rejected: {rejection_reason}"
                break
            if dt < cfg.dt_min:
                status = RCETerminalStatus.DT_MIN_FAILURE
                reason = f"dt fell below dt_min after rejections ({rejection_reason})"
                break
            continue
    else:
        status = RCETerminalStatus.MAX_STEPS
        reason = "maximum step budget reached"

    closure, rad, f_conv, f_rad, f_total = _run_unsplit(
        grid, state, physics, thermo, opacity, pressure, top_bc, lower_bc, cfg, manufactured, grav
    )
    final_closure = closure
    if manufactured is None:
        final_rad = rad
    final_f_conv = f_conv
    final_f_rad = f_rad
    final_f_total = f_total
    dhdt_final = enthalpy_tendency(grid, final_f_total, state.mass_path)
    conv_now, _, final_rcb, bottom_now, detached_now, _, _ = _convergence_metrics(
        grid, state, state, final_f_total, dhdt_final, final_closure, solver, cfg, prev_rcb, f_int
    )
    final_regions = bottom_now + detached_now
    final_detached = detached_now
    final_conv = RCEConvergence(
        flux_flatness=conv_now.flux_flatness,
        tendency_norm=conv_now.tendency_norm,
        temp_change=0.0 if steps_accepted == 0 else last_temp_change,
        rcb_stable=conv_now.rcb_stable if steps_accepted > 0 else True,
        finite_state=conv_now.finite_state,
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
        detached_convective_regions=final_detached,
        convergence=final_conv,
        diagnostics=diagnostics,
    )


def solve_adaptive_rce_with_prescribed_external_flux(
    grid: PressureGrid,
    initial_temperature: NDArray[np.float64],
    physics: PhysicsConfig,
    solver: SolverConfig,
    thermo: ThermoProvider,
    f_ext: NDArray[np.float64],
    *,
    gravity: GravityLaw | None = None,
    config: RCEConfig | None = None,
    f0: float | None = None,
) -> RCEResult:
    """Convection-on RCE with a prescribed interface external flux (MLT remains active).

    α=0 is a radiation-only control and must not be used as a convection-only
    helper. This path keeps α>0 and injects F_ext as the explicit operator.
    """
    if physics.alpha <= 0.0:
        raise ValueError("prescribed-external convection helper requires alpha > 0")
    grav = gravity or ConstantGravity(physics.gravity)
    require_constant_gravity(grav)
    # Opaque dummy: radiation is never evaluated when prescribed_f_ext is set,
    # but solve_adaptive_rce still needs an opacity object for the final audit.
    from .opacity import ConstantGreyOpacity

    opacity = ConstantGreyOpacity(0.0)
    pressure = grid.pressure_centres
    top = TopIrradiation(0.0)
    bot = LowerNetInternalFlux(float(f0) if f0 is not None else float(f_ext[0]))
    cfg = config or RCEConfig()
    # Wrap manufactured so flux reference uses f0; F_ext itself comes from prescribed.
    manufactured = None
    if f0 is not None:
        manufactured = ManufacturedRadiativeTarget(
            target_temperature=np.asarray(initial_temperature, dtype=np.float64),
            f0=float(f0),
            relaxation_coeff=0.0,
        )

    # Local copy of solve loop is heavy; reuse split path via a thin adapter.
    # Inject prescribed F_ext by monkey-patching through _run_split_macrostep kw.
    cfg_local = cfg
    route = RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV

    # Inline driver that passes prescribed_f_ext into the split macrostep.
    state = build_column_state(grid, np.asarray(initial_temperature, dtype=np.float64), thermo, grav)
    f_int = float(f0) if f0 is not None else float(f_ext[0])
    accepted_consec = 0
    prev_rcb: float | None = None
    diagnostics: list[RCEStepDiagnostics] = []
    simulated_time = 0.0
    rejections = 0
    best_resid = np.inf
    stall_counter = 0
    steps_accepted = 0
    dt_hold: float | None = None
    dt_ceiling: float | None = None
    hold_remaining = 0
    last_temp_change = float("inf")
    final_closure = _evaluate_closure(grid, state, physics, thermo)
    final_rad = None
    final_f_conv = np.zeros(grid.n_layers + 1)
    final_f_rad = np.asarray(f_ext, dtype=np.float64).copy()
    final_f_total = final_f_rad + final_f_conv
    final_regions: list[tuple[int, int]] = []
    final_detached: list[tuple[int, int]] = []
    final_conv = RCEConvergence(np.inf, np.inf, np.inf, False, False)
    final_rcb = None
    status = RCETerminalStatus.MAX_STEPS
    reason = "maximum step budget reached"

    for _step in range(cfg_local.max_steps):
        if cfg_local.t_final is not None and simulated_time >= cfg_local.t_final:
            status = (
                RCETerminalStatus.CONVERGED
                if _gate_ok(final_conv, cfg_local, manufactured=manufactured, temperature=state.temperature)
                else RCETerminalStatus.MAX_STEPS
            )
            reason = "reached t_final"
            break

        f_rad_for_dt = np.asarray(f_ext, dtype=np.float64)
        closure_for_dt = _evaluate_closure(grid, state, physics, thermo)
        f_total_for_dt = f_rad_for_dt + closure_for_dt.flux
        dhdt_rad = enthalpy_tendency(grid, f_rad_for_dt, state.mass_path)
        dt_rad = _dt_rad_estimate(state, dhdt_rad, solver, thermo)
        dt_rad_T = _dt_temp_estimate(state, dhdt_rad, solver, thermo)
        candidates = [dt_rad, dt_rad_T]
        if cfg_local.dt_accuracy is not None:
            candidates.append(float(cfg_local.dt_accuracy))
        dt_est = min(candidates)
        dt_mlt = float("inf")
        dt_temp = dt_rad_T

        prescribed = cfg_local.prescribed_dt is not None
        if prescribed:
            dt = float(cfg_local.prescribed_dt)
        else:
            dt = dt_est
            if dt_hold is not None and np.isfinite(dt_hold):
                dt = min(dt, dt_hold) if np.isfinite(dt) else dt_hold
            if cfg_local.t_final is not None:
                remaining = cfg_local.t_final - simulated_time
                if remaining <= 0.0:
                    break
                if np.isfinite(dt):
                    dt = min(dt, remaining)

        if not np.isfinite(dt) or dt < cfg_local.dt_min:
            status = RCETerminalStatus.DT_MIN_FAILURE
            reason = f"timestep below dt_min: dt={dt}"
            break

        accepted = False
        for _attempt in range(1 if prescribed else (cfg_local.max_rejections + 1)):
            old_state = state
            attempt = _run_split_macrostep(
                route, grid, state, dt, physics, thermo, grav, opacity, pressure,
                top, bot, cfg_local, manufactured, solver, prescribed_f_ext=f_rad_for_dt,
            )
            if not attempt.ok:
                rejections += 1
                diagnostics.append(
                    _rejected_diag(dt, route, dt_mlt, dt_rad, dt_temp, attempt.reason or "failed")
                )
                if prescribed:
                    status = RCETerminalStatus.PRESCRIBED_DT_REJECTED
                    reason = f"prescribed_dt rejected: {attempt.reason}"
                    break
                dt, dt_ceiling, hold_remaining = dt_after_reject(dt, cfg_local)
                dt_hold = dt
                if dt < cfg_local.dt_min:
                    break
                continue

            trial_state = attempt.state
            f_conv = attempt.f_conv
            f_rad = attempt.f_rad
            f_total = f_rad + f_conv
            # Re-evaluate ordinary MLT at committed T for metrics.
            closure = _evaluate_closure(grid, trial_state, physics, thermo)
            f_conv = closure.flux
            f_total = f_rad + f_conv
            dhdt = enthalpy_tendency(grid, f_total, trial_state.mass_path)
            boundary_work = attempt.boundary_work
            energy_lhs = attempt.energy_lhs
            energy_resid = energy_lhs - boundary_work
            energy_committed = float(
                np.sum(old_state.mass_path * (trial_state.enthalpy - old_state.enthalpy))
            )
            energy_committed_resid = energy_committed - boundary_work
            ulp_floor = _ulp_energy_floor(old_state.mass_path, trial_state.enthalpy)
            conv, _f_ref, rcb, bottom_regions, detached_regions, f_scale, boundary_mismatch = (
                _convergence_metrics(
                    grid, old_state, trial_state, f_total, dhdt, closure, solver,
                    cfg_local, prev_rcb, f_int,
                )
            )
            e_scale = _energy_scale(boundary_work, f_scale, dt, cfg_local)
            e_committed_scale = max(e_scale, ulp_floor)
            idiag = attempt.implicit

            state = trial_state
            simulated_time += dt
            steps_accepted += 1
            accepted = True
            last_temp_change = conv.temp_change
            if not prescribed:
                dt_hold, dt_ceiling, hold_remaining = dt_after_accept(
                    dt,
                    dt_est,
                    cfg_local,
                    implicit=True,
                    dt_ceiling=dt_ceiling,
                    hold_remaining=hold_remaining,
                )

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
                    n_bottom_connected_regions=len(bottom_regions),
                    energy_committed=energy_committed,
                    energy_committed_residual=energy_committed_resid,
                    energy_committed_residual_rel=abs(energy_committed_resid) / e_committed_scale,
                    energy_ulp_floor=ulp_floor,
                    nonlinear_residual=float("nan") if idiag is None else idiag.residual_norm,
                    newton_iterations=0 if idiag is None else idiag.newton_iterations,
                    line_search_backtracks=0 if idiag is None else idiag.line_search_backtracks,
                    mask_outer_iterations=0 if idiag is None else idiag.mask_outer_iterations,
                    mlt_evals=0 if idiag is None else idiag.mlt_evals,
                )
            )
            final_closure = closure
            final_f_conv = f_conv
            final_f_rad = f_rad
            final_f_total = f_total
            final_regions = bottom_regions + detached_regions
            final_detached = detached_regions
            final_conv = conv
            final_rcb = rcb
            if _gate_ok(conv, cfg_local, manufactured=manufactured, temperature=state.temperature):
                accepted_consec += 1
            else:
                accepted_consec = 0
            resid_scalar = max(conv.flux_flatness, conv.tendency_norm, conv.temp_change)
            if resid_scalar < best_resid * (1.0 - cfg_local.stall_rel_improvement):
                best_resid = resid_scalar
                stall_counter = 0
            else:
                stall_counter += 1
            prev_rcb = rcb
            if accepted_consec >= cfg_local.n_consec:
                status = RCETerminalStatus.CONVERGED
                reason = f"converged for {cfg_local.n_consec} consecutive accepted steps"
            elif stall_counter >= cfg_local.stall_window:
                status = RCETerminalStatus.STALLED
                reason = "residuals/mask stalled"
            break

        if accepted and status in (RCETerminalStatus.CONVERGED, RCETerminalStatus.STALLED):
            break
        if status == RCETerminalStatus.PRESCRIBED_DT_REJECTED:
            break
        if not accepted:
            if dt < cfg_local.dt_min:
                status = RCETerminalStatus.DT_MIN_FAILURE
                reason = "dt fell below dt_min after rejections"
                break
            continue
    else:
        status = RCETerminalStatus.MAX_STEPS
        reason = "maximum step budget reached"

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
        final_radiation=_empty_radiation(grid.n_layers),
        final_flux_total=final_f_total,
        final_flux_conv=final_f_conv,
        final_flux_rad=final_f_rad,
        primary_rcb_log10p=final_rcb,
        convective_regions=final_regions,
        detached_convective_regions=final_detached,
        convergence=final_conv,
        diagnostics=diagnostics,
    )
