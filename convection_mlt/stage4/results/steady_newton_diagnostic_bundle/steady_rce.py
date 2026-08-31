"""Steady flux-flatness Newton–Krylov solver for coupled analytic-opacity RCE.

The physical layer-divergence statement F_total,i − F_total,i+1 = 0 is
equivalent to a flat total-flux profile once the bottom boundary enforces
F_total,0 = F_int. This module residualises the second form:

    r_i(h) = (F_total,i+1(h) − F_int) / F_scale,   i = 0, …, N−1.

Newton therefore terminates on the declared flux-flatness gate
max_i |F_total,i − F_int| / F_scale rather than on an unpreconditioned
layer-divergence norm, which can be tiny on a shallow ramp that is still
several percent off F_int.

Unknown is layer enthalpy. No extra column-enthalpy constraint is imposed:
convection conserves enthalpy internally, while the column exchanges energy
through the fixed bottom flux and the TOA boundary.

The inner Newton freezes the current convective interface mask. After inner
convergence the mask is recomputed; a change restarts the inner solve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .closure import ClosureResult
from .config import PhysicsConfig, SolverConfig
from .gravity import ConstantGravity, GravityLaw
from .grid import PressureGrid
from .implicit_convection import _state_from_h
from .opacity import PrescribedOpacity
from .radiation import (
    DEFAULT_DIFFUSIVITY,
    LowerBoundary,
    RadiationResult,
    SolveRoute,
    TopIrradiation,
    solve_radiation,
)
from .rce import (
    _evaluate_closure,
    _internal_flux_reference,
    _partition_rcb_regions,
    _rcb_regions,
)
from .state import ColumnState, build_column_state
from .thermodynamics import ThermoProvider


class SteadyRCEStatus(str, Enum):
    CONVERGED = "converged"
    NEWTON_LIMIT = "newton_limit"
    INNER_STALLED = "inner_stalled"
    MASK_LIMIT = "mask_limit"
    LINE_SEARCH_FAILURE = "line_search_failure"
    DOMAIN_FAILURE = "domain_failure"
    INVALID_BOUNDARY = "invalid_boundary"


@dataclass(frozen=True)
class SteadyRCEConfig:
    flux_flatness_tolerance: float = 1.0e-3
    tendency_tolerance: float = 1.0e-3
    flux_scale_floor: float = 1.0e-30
    max_newton: int = 40
    max_line_search: int = 20
    min_line_search_factor: float = 1.0e-8
    armijo_c: float = 1.0e-4
    max_tendency_increase: float = 2.0
    max_step_rel: float = 0.25
    max_mask_outer: int = 12
    fd_rel: float = 1.0e-6
    subspace_fd_rel: float = 1.0e-8
    h_floor: float = 1.0e-30
    gmres_restart: int = 32
    gmres_maxiter: int = 48
    gmres_rtol_max: float = 0.1
    gmres_rtol_min: float = 1.0e-3
    use_subspace: bool = True
    use_gmres: bool = True
    reject_branch_crossing: bool = True
    max_consecutive_ls_fail: int = 3
    radiation_route: SolveRoute = SolveRoute.THOMAS
    diffusivity_factor: float | None = None


@dataclass(frozen=True)
class TrialFluxes:
    state: ColumnState
    closure: ClosureResult
    radiation: RadiationResult
    flux_conv: NDArray[np.float64]
    flux_rad: NDArray[np.float64]
    flux_total: NDArray[np.float64]
    residual: NDArray[np.float64]
    flux_flatness: float
    tendency_norm: float


@dataclass
class SteadyNewtonRecord:
    outer: int
    newton: int
    flux_flatness: float
    tendency_norm: float
    residual_two_norm: float
    flux_flatness_before: float
    flux_flatness_after: float
    tendency_norm_before: float
    tendency_norm_after: float
    residual_two_norm_before: float
    residual_two_norm_after: float
    residual_inf_before: float
    residual_inf_after: float
    residual_before: list[float]
    residual_after: list[float]
    step_rel: float
    step_rel_h_newton: float
    step_rel_h_accepted: float
    step_rel_T_accepted: float
    line_search_factor: float
    line_search_reason: str
    line_search_backtracks: int
    direction: str | None
    gmres_iters: int
    gmres_residual_norm: float
    gmres_rhs_norm: float
    gmres_rtol: float
    gmres_success: bool
    n_evals: int
    mask_before: list[bool]
    mask_after: list[bool]
    mask_changed: bool = False
    min_superadiabatic_excess_active: float = float("nan")
    max_superadiabatic_excess_inactive: float = float("nan")
    min_superadiabatic_excess_active_including_boundary: float = float("nan")
    activity_threshold: float = float("nan")
    n_inactive_above_threshold: int = 0
    rcb_active_excess: float = float("nan")
    rcb_inactive_excess: float = float("nan")
    rcb_active_distance_to_threshold: float = float("nan")
    rcb_inactive_distance_to_threshold: float = float("nan")
    n_branch_crossings: int = 0
    jv_n_branch_crossings: int = 0
    n_subspace_columns: int = 0
    n_subspace_columns_rejected: int = 0
    merit_before: float = float("nan")
    merit_after: float = float("nan")
    live_minus_lagged_conv_rel: float = float("nan")
    fd_rel: float = float("nan")
    jv_eps_used: float = float("nan")
    h_scale_rms: float = float("nan")
    gmres_linear_residual_ratio: float = float("nan")


@dataclass(frozen=True)
class SteadyRCEResult:
    status: SteadyRCEStatus
    reason: str
    state: ColumnState
    closure: ClosureResult
    radiation: RadiationResult | None
    flux_conv: NDArray[np.float64]
    flux_rad: NDArray[np.float64]
    flux_total: NDArray[np.float64]
    residual: NDArray[np.float64]
    flux_flatness: float
    tendency_norm: float
    convective_regions: list[tuple[int, int]]
    detached_convective_regions: list[tuple[int, int]]
    frozen_support: NDArray[np.bool_]
    newton_iterations: int
    line_search_backtracks: int
    mask_outer_iterations: int
    n_evals: int
    history: list[SteadyNewtonRecord] = field(default_factory=list)


ProgressFn = Callable[[dict[str, object]], None]


def interface_support_from_regions(
    n_layers: int, regions: list[tuple[int, int]]
) -> NDArray[np.bool_]:
    """Length-(N+1) interface mask from `_rcb_regions` layer spans.

    Region (i_lo, i_hi) covers layers i_lo…i_hi joined by internal interfaces
    i_lo+1…i_hi. A bottom-connected zone also keeps interface 0 active so
    F_conv,0 can remain nonzero. TOA flux is always zero.
    """
    support = np.zeros(n_layers + 1, dtype=bool)
    for i_lo, i_hi in regions:
        lo = int(i_lo)
        hi = int(i_hi)
        if lo == 0:
            support[0] = True
        if hi > lo:
            support[lo + 1 : hi + 1] = True
    support[-1] = False
    return support


def flux_flatness_residual(
    flux_total: NDArray[np.float64], f_int: float, f_scale: float
) -> NDArray[np.float64]:
    """N-vector (F_total[1:] − F_int) / F_scale. F_total[0] is the bottom BC."""
    return (np.asarray(flux_total, dtype=np.float64)[1:] - float(f_int)) / float(f_scale)


def live_convective_flux(
    grid: PressureGrid,
    state: ColumnState,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
) -> ClosureResult:
    """Unfrozen mixing-length closure: live F_conv and superadiabaticity."""
    return _evaluate_closure(grid, state, physics, thermo)


def active_interface_mask(
    n_layers: int,
    closure: ClosureResult,
    solver: SolverConfig,
) -> NDArray[np.bool_]:
    """Active convective interface mask from the live closure."""
    return interface_support_from_regions(n_layers, _rcb_regions(closure, solver))


def residual_merit(residual: NDArray[np.float64]) -> float:
    """Φ = ½ ∥r∥₂² used by the Armijo line search."""
    r = np.asarray(residual, dtype=np.float64)
    return 0.5 * float(np.dot(r, r))


def _internal_interface_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    internal = np.asarray(mask, dtype=bool).copy()
    if internal.size:
        internal[0] = False
        internal[-1] = False
    return internal


def count_internal_branch_crossings(
    n_layers: int,
    closure: ClosureResult,
    frozen_support: NDArray[np.bool_],
    solver: SolverConfig,
) -> int:
    """Internal interfaces whose live activity disagrees with the frozen mask."""
    live = _internal_interface_mask(active_interface_mask(n_layers, closure, solver))
    frozen = _internal_interface_mask(np.asarray(frozen_support, dtype=bool))
    return int(np.count_nonzero(live != frozen))


def mask_superadiabatic_excess(
    closure: ClosureResult,
    support: NDArray[np.bool_],
    solver: SolverConfig,
) -> dict[str, float]:
    """Superadiabatic excess on frozen-active vs inactive *internal* interfaces.

    Interface 0 is forced active for a bottom-connected CZ, but the closure
    stores identically zero superadiabaticity there. Including it makes
    min_active = 0 even when the RCB is well above the activity threshold.
    """
    sa = np.asarray(closure.superadiabaticity, dtype=np.float64)
    mask = np.asarray(support, dtype=bool)
    if sa.shape != mask.shape:
        raise ValueError("superadiabaticity and support length mismatch")
    threshold = float(solver.c_active * solver.epsilon_gradient)
    internal_active = _internal_interface_mask(mask)
    internal_inactive = _internal_interface_mask(~mask)
    active = sa[internal_active]
    inactive = sa[internal_inactive]
    active_all = sa[mask]
    min_active = float(np.min(active)) if active.size else float("nan")
    max_inactive = float(np.max(inactive)) if inactive.size else float("nan")
    min_active_all = float(np.min(active_all)) if active_all.size else float("nan")
    n_inactive_hot = int(np.count_nonzero(inactive > threshold)) if inactive.size else 0
    rcb_active = float("nan")
    rcb_inactive = float("nan")
    active_idx = np.flatnonzero(internal_active)
    if active_idx.size:
        last_act = int(active_idx.max())
        rcb_active = float(sa[last_act])
        nxt = last_act + 1
        if nxt < sa.size:
            rcb_inactive = float(sa[nxt])
    return {
        "activity_threshold": threshold,
        "min_superadiabatic_excess_active": min_active,
        "max_superadiabatic_excess_inactive": max_inactive,
        "min_superadiabatic_excess_active_including_boundary": min_active_all,
        "n_inactive_above_threshold": float(n_inactive_hot),
        "n_active_interfaces": float(int(np.count_nonzero(mask))),
        "n_active_internal_interfaces": float(int(np.count_nonzero(internal_active))),
        "rcb_active_excess": rcb_active,
        "rcb_inactive_excess": rcb_inactive,
        "rcb_active_distance_to_threshold": rcb_active - threshold
        if np.isfinite(rcb_active)
        else float("nan"),
        "rcb_inactive_distance_to_threshold": threshold - rcb_inactive
        if np.isfinite(rcb_inactive)
        else float("nan"),
    }


def flux_metrics(
    flux_total: NDArray[np.float64],
    mass_path: NDArray[np.float64],
    f_int: float,
    f_scale: float,
) -> tuple[NDArray[np.float64], float, float]:
    residual = flux_flatness_residual(flux_total, f_int, f_scale)
    flatness = float(np.max(np.abs(flux_total - f_int), initial=0.0)) / f_scale
    dhdt = enthalpy_tendency_from_flux(flux_total, mass_path)
    tendency = float(np.max(np.abs(mass_path * dhdt), initial=0.0)) / f_scale
    return residual, flatness, tendency


def enthalpy_tendency_from_flux(
    flux_total: NDArray[np.float64], mass_path: NDArray[np.float64]
) -> NDArray[np.float64]:
    return (flux_total[:-1] - flux_total[1:]) / mass_path


def restarted_gmres(
    apply_a: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    b: NDArray[np.float64],
    *,
    rtol: float,
    atol: float = 0.0,
    maxiter: int = 80,
    restart: int = 30,
) -> tuple[NDArray[np.float64], int, float, bool]:
    """Solve A x = b with restarted GMRES and modified Gram–Schmidt."""
    n = int(b.size)
    x = np.zeros(n, dtype=np.float64)
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return x, 0, 0.0, True
    tol = max(float(atol), float(rtol) * b_norm)
    iters = 0
    n_cycles = max(1, (int(maxiter) + int(restart) - 1) // int(restart))
    for _ in range(n_cycles):
        r = b - apply_a(x)
        r_norm = float(np.linalg.norm(r))
        if r_norm <= tol:
            return x, iters, r_norm, True
        q = np.zeros((n, restart + 1), dtype=np.float64)
        h = np.zeros((restart + 1, restart), dtype=np.float64)
        cs = np.zeros(restart, dtype=np.float64)
        sn = np.zeros(restart, dtype=np.float64)
        g = np.zeros(restart + 1, dtype=np.float64)
        q[:, 0] = r / r_norm
        g[0] = r_norm
        inner = 0
        lucky = False
        for j in range(restart):
            if iters >= maxiter:
                break
            v = np.asarray(apply_a(q[:, j]), dtype=np.float64)
            for i in range(j + 1):
                h[i, j] = float(np.dot(q[:, i], v))
                v = v - h[i, j] * q[:, i]
            for i in range(j + 1):
                hij = float(np.dot(q[:, i], v))
                h[i, j] += hij
                v = v - hij * q[:, i]
            h[j + 1, j] = float(np.linalg.norm(v))
            if h[j + 1, j] > 0.0:
                q[:, j + 1] = v / h[j + 1, j]
            else:
                lucky = True
            for i in range(j):
                t = cs[i] * h[i, j] + sn[i] * h[i + 1, j]
                h[i + 1, j] = -sn[i] * h[i, j] + cs[i] * h[i + 1, j]
                h[i, j] = t
            rho = float(np.hypot(h[j, j], h[j + 1, j]))
            if rho == 0.0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j] = h[j, j] / rho
                sn[j] = h[j + 1, j] / rho
            h[j, j] = rho
            h[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            iters += 1
            inner = j + 1
            if abs(g[j + 1]) <= tol or lucky:
                break
        y = np.zeros(inner, dtype=np.float64)
        for i in range(inner - 1, -1, -1):
            s = g[i]
            for k in range(i + 1, inner):
                s -= h[i, k] * y[k]
            y[i] = 0.0 if h[i, i] == 0.0 else s / h[i, i]
        x = x + q[:, :inner] @ y
        r_norm = float(np.linalg.norm(b - apply_a(x)))
        if r_norm <= tol or iters >= maxiter:
            return x, iters, r_norm, r_norm <= tol
    r_norm = float(np.linalg.norm(b - apply_a(x)))
    return x, iters, r_norm, r_norm <= tol


def _apply_frozen_convective_flux(
    flux: NDArray[np.float64], support: NDArray[np.bool_] | None
) -> NDArray[np.float64]:
    f = np.asarray(flux, dtype=np.float64).copy()
    if support is not None:
        f = np.where(np.asarray(support, dtype=bool), f, 0.0)
    f[-1] = 0.0
    return f


def evaluate_trial(
    grid: PressureGrid,
    enthalpy: NDArray[np.float64],
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    opacity: PrescribedOpacity,
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    lower_bc: LowerBoundary,
    gravity: GravityLaw,
    *,
    f_int: float,
    f_scale: float,
    frozen_support: NDArray[np.bool_] | None,
    diffusivity_factor: float,
    radiation_route: SolveRoute,
    frozen_flux_conv: NDArray[np.float64] | None = None,
) -> TrialFluxes | None:
    state = _state_from_h(grid, enthalpy, thermo, gravity)
    if state is None:
        return None
    if frozen_flux_conv is None:
        closure = _evaluate_closure(grid, state, physics, thermo)
        f_conv = _apply_frozen_convective_flux(closure.flux, frozen_support)
    else:
        f_conv = np.asarray(frozen_flux_conv, dtype=np.float64).copy()
        f_conv[-1] = 0.0
        closure = _evaluate_closure(grid, state, physics, thermo)
    rad = solve_radiation(
        state.temperature,
        state.mass_path,
        opacity,
        pressure,
        top_bc,
        lower_bc,
        diffusivity_factor,
        radiation_route,
        bottom_convective_flux=float(f_conv[0]),
    )
    f_rad = rad.flux_net
    f_total = f_rad + f_conv
    residual, flatness, tendency = flux_metrics(f_total, state.mass_path, f_int, f_scale)
    return TrialFluxes(
        state=state,
        closure=closure,
        radiation=rad,
        flux_conv=f_conv,
        flux_rad=f_rad,
        flux_total=f_total,
        residual=residual,
        flux_flatness=flatness,
        tendency_norm=tendency,
    )


def _unfrozen_regions(trial: TrialFluxes, solver: SolverConfig) -> tuple[
    list[tuple[int, int]], list[tuple[int, int]], NDArray[np.bool_]
]:
    regions = _rcb_regions(trial.closure, solver)
    active = trial.closure.superadiabaticity[1:-1] > (
        solver.c_active * solver.epsilon_gradient
    )
    bottom, detached = _partition_rcb_regions(regions, np.asarray(active, dtype=bool))
    support = interface_support_from_regions(trial.state.temperature.size, regions)
    return regions, detached, support


def _finite_difference_jv(
    residual_at: Callable[[NDArray[np.float64]], NDArray[np.float64] | None],
    h: NDArray[np.float64],
    r0: NDArray[np.float64],
    direction: NDArray[np.float64],
    fd_rel: float,
    h_floor: float,
) -> NDArray[np.float64] | None:
    dnorm = float(np.linalg.norm(direction))
    if dnorm == 0.0:
        return np.zeros_like(r0)
    h_norm = max(float(np.linalg.norm(h)), h_floor * np.sqrt(h.size))
    eps = fd_rel * h_norm / dnorm
    for _ in range(8):
        trial = residual_at(h + eps * direction)
        if trial is not None:
            return (trial - r0) / eps
        eps *= 0.1
        if eps == 0.0:
            break
    for _ in range(4):
        trial = residual_at(h - eps * direction)
        if trial is not None:
            return (r0 - trial) / eps
        eps *= 0.1
    return None


def _centred_live_jv(
    trial_at: Callable[[NDArray[np.float64]], TrialFluxes | None],
    h: NDArray[np.float64],
    r0: NDArray[np.float64],
    direction: NDArray[np.float64],
    fd_rel: float,
    h_floor: float,
    *,
    frozen_support: NDArray[np.bool_],
    solver: SolverConfig,
    n_layers: int,
    reject_branch_crossing: bool,
) -> tuple[NDArray[np.float64] | None, dict[str, object]]:
    """Centred Jv of the live-MLT residual; optionally reject branch crossings."""
    info: dict[str, object] = {
        "n_evals": 0,
        "n_branch_crossings": 0,
        "n_branch_crossings_plus": 0,
        "n_branch_crossings_minus": 0,
        "rejected": False,
        "eps": float("nan"),
    }
    dnorm = float(np.linalg.norm(direction))
    if dnorm == 0.0:
        return np.zeros_like(r0), info
    h_norm = max(float(np.linalg.norm(h)), h_floor * np.sqrt(h.size))
    eps = float(fd_rel) * h_norm / dnorm
    plus = minus = None
    for _ in range(8):
        plus = trial_at(h + eps * direction)
        minus = trial_at(h - eps * direction)
        info["n_evals"] = int(info["n_evals"]) + 2
        if plus is not None and minus is not None:
            break
        eps *= 0.1
        if eps == 0.0:
            break
    info["eps"] = eps
    if plus is None or minus is None:
        info["rejected"] = True
        return None, info
    n_plus = count_internal_branch_crossings(
        n_layers, plus.closure, frozen_support, solver
    )
    n_minus = count_internal_branch_crossings(
        n_layers, minus.closure, frozen_support, solver
    )
    info["n_branch_crossings_plus"] = n_plus
    info["n_branch_crossings_minus"] = n_minus
    info["n_branch_crossings"] = n_plus + n_minus
    if reject_branch_crossing and (n_plus or n_minus):
        info["rejected"] = True
        return None, info
    return np.asarray((plus.residual - minus.residual) / (2.0 * eps), dtype=np.float64), info


def jv_epsilon_ladder(
    residual_at: Callable[[NDArray[np.float64]], NDArray[np.float64] | None],
    h: NDArray[np.float64],
    r0: NDArray[np.float64],
    direction: NDArray[np.float64],
    *,
    fd_rels: tuple[float, ...] | None = None,
    h_floor: float = 1.0e-30,
    fine_fd_rel: float = 1.0e-6,
    trial_at: Callable[[NDArray[np.float64]], TrialFluxes | None] | None = None,
    frozen_support: NDArray[np.bool_] | None = None,
    solver: SolverConfig | None = None,
    n_layers: int | None = None,
) -> dict[str, object]:
    """Finite-difference Jv vs perturbation size.

    eps = fd_rel * ||h|| / ||direction||.
    Full-ladder maxima include fd_rel = 10⁻³, which is outside the local
    linear neighbourhood. `jv_stable` uses only rungs with
    from_fd_rel ≤ fine_fd_rel (default 10⁻⁶).
    """
    if fd_rels is None:
        fd_rels = (
            1.0e-3,
            1.0e-4,
            1.0e-5,
            1.0e-6,
            1.0e-7,
            1.0e-8,
            float(np.sqrt(np.finfo(np.float64).eps)),
        )
    dnorm = float(np.linalg.norm(direction))
    h_norm = max(float(np.linalg.norm(h)), h_floor * np.sqrt(h.size))
    rows: list[dict[str, object]] = []
    success: list[tuple[float, NDArray[np.float64]]] = []

    for fd_rel in fd_rels:
        if dnorm == 0.0:
            rows.append({"fd_rel": float(fd_rel), "ok": False, "reason": "zero_direction"})
            continue
        eps = float(fd_rel) * h_norm / dnorm
        plus_trial = trial_at(h + eps * direction) if trial_at is not None else None
        r_plus = None if plus_trial is None else plus_trial.residual
        if trial_at is None:
            r_plus = residual_at(h + eps * direction)
        if r_plus is None:
            rows.append({"fd_rel": float(fd_rel), "eps": eps, "ok": False, "reason": "domain_plus"})
            continue
        jv = np.asarray((r_plus - r0) / eps, dtype=np.float64)
        minus_trial = trial_at(h - eps * direction) if trial_at is not None else None
        r_minus = residual_at(h - eps * direction) if trial_at is None else (
            None if minus_trial is None else minus_trial.residual
        )
        centered = None if r_minus is None else (r_plus - r_minus) / (2.0 * eps)
        success.append((float(fd_rel), jv))
        row: dict[str, object] = {
            "fd_rel": float(fd_rel),
            "eps": eps,
            "ok": True,
            "jv_two_norm": float(np.linalg.norm(jv)),
            "jv_inf_norm": float(np.max(np.abs(jv), initial=0.0)),
            "delta_r_inf": float(np.max(np.abs(r_plus - r0), initial=0.0)),
        }
        if centered is not None:
            c = np.asarray(centered, dtype=np.float64)
            row["centered_jv_two_norm"] = float(np.linalg.norm(c))
            denom = max(float(np.linalg.norm(jv)), 1.0e-30)
            row["forward_vs_centered_rel"] = float(np.linalg.norm(jv - c) / denom)
        if (
            trial_at is not None
            and frozen_support is not None
            and solver is not None
            and n_layers is not None
        ):
            n_plus = 0 if plus_trial is None else count_internal_branch_crossings(
                n_layers, plus_trial.closure, frozen_support, solver
            )
            n_minus = 0 if minus_trial is None else count_internal_branch_crossings(
                n_layers, minus_trial.closure, frozen_support, solver
            )
            row["n_branch_crossings_plus"] = n_plus
            row["n_branch_crossings_minus"] = n_minus
            row["n_branch_crossings"] = n_plus + n_minus
        rows.append(row)
    pairwise: list[dict[str, object]] = []
    for i in range(len(success) - 1):
        fd_a, a = success[i]
        fd_b, b = success[i + 1]
        denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0e-30)
        pairwise.append(
            {
                "from_fd_rel": fd_a,
                "to_fd_rel": fd_b,
                "rel_two_change": float(np.linalg.norm(a - b) / denom),
            }
        )
    max_rel = max((float(p["rel_two_change"]) for p in pairwise), default=float("nan"))
    fine_pairs = [p for p in pairwise if float(p["from_fd_rel"]) <= float(fine_fd_rel)]
    max_rel_fine = max(
        (float(p["rel_two_change"]) for p in fine_pairs), default=float("nan")
    )
    return {
        "h_norm": h_norm,
        "direction_norm": dnorm,
        "ladder": rows,
        "pairwise_rel_two_change": pairwise,
        "max_pairwise_rel_two_change": max_rel,
        "max_pairwise_rel_two_change_fine": max_rel_fine,
        "fine_fd_rel": float(fine_fd_rel),
        "jv_stable": bool(np.isfinite(max_rel_fine) and max_rel_fine < 0.1),
        "jv_stable_full_ladder": bool(np.isfinite(max_rel) and max_rel < 0.1),
        "note": (
            "jv_stable uses only rungs with from_fd_rel ≤ fine_fd_rel "
            f"({fine_fd_rel:g}), the local linear neighbourhood. "
            "jv_stable_full_ladder includes fd_rel = 10⁻³, which is typically "
            "outside that neighbourhood. Larger fine-rung jumps indicate "
            "noise or MLT nonsmoothness; large full-ladder jumps at coarse "
            "fd_rel indicate nonlinearity."
        ),
    }


def _clip_step(
    dh: NDArray[np.float64], h_scale: NDArray[np.float64], max_step_rel: float
) -> tuple[NDArray[np.float64], float]:
    rel = np.abs(dh) / h_scale
    peak = float(np.max(rel, initial=0.0))
    if peak > max_step_rel and peak > 0.0:
        dh = dh * (max_step_rel / peak)
        peak = max_step_rel
    return dh, peak


def _convective_layer_mask(n_layers: int, support: NDArray[np.bool_] | None) -> NDArray[np.float64]:
    mask = np.zeros(n_layers, dtype=np.float64)
    if support is None or not np.any(support):
        return mask
    if bool(support[0]):
        mask[0] = 1.0
    for k in range(1, n_layers):
        if bool(support[k]):
            mask[k - 1] = 1.0
            mask[k] = 1.0
    return mask


def _subspace_basis(
    h_scale: NDArray[np.float64],
    layer_cz: NDArray[np.float64],
    logp: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    layer_rad = 1.0 - layer_cz
    logp_c = logp - float(np.mean(logp))
    raw = [
        h_scale,
        h_scale * layer_rad,
        h_scale * layer_cz,
        h_scale * logp_c,
        h_scale * layer_rad * logp_c,
    ]
    cols = []
    floor = 1.0e-30 * np.sqrt(h_scale.size)
    for col in raw:
        nrm = float(np.linalg.norm(col))
        if nrm > floor:
            cols.append(col / nrm)
    if not cols:
        return None
    stacked = np.stack(cols, axis=1)
    q, r = np.linalg.qr(stacked, mode="reduced")
    diag = np.abs(np.diag(r))
    keep = diag > (1.0e-10 * float(np.max(diag)))
    if not np.any(keep):
        return None
    return q[:, keep]


def _least_squares_step(
    basis: NDArray[np.float64],
    j_basis: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if basis.size == 0 or j_basis.size == 0:
        return None
    alpha, *_ = np.linalg.lstsq(j_basis, -residual, rcond=None)
    if not np.all(np.isfinite(alpha)):
        return None
    return basis @ alpha


def _line_search(
    h: NDArray[np.float64],
    dh: NDArray[np.float64],
    residual_at: Callable[
        [NDArray[np.float64], NDArray[np.bool_] | None], tuple[TrialFluxes | None, int]
    ],
    support: NDArray[np.bool_] | None,
    current: TrialFluxes,
    cfg: SteadyRCEConfig,
    tend_anchor: float | None = None,
) -> tuple[TrialFluxes | None, float, int, int, str]:
    """Armijo line search on Φ = ½∥r∥₂² with a tendency safeguard.

    Accept α if Φ(h+αΔh) ≤ (1 − c α) Φ(h) and tendency does not increase
    beyond max(anchor, gate) * max_tendency_increase. The anchor is the
    inner-loop starting tendency when provided, otherwise the current trial.
    Termination of the outer Newton loop remains on the declared flatness
    and tendency gates.
    """
    accepted = None
    alpha = 1.0
    extra = 0
    backs = 0
    used_alpha = 1.0
    reason = "no_attempt"
    phi0 = residual_merit(current.residual)
    tend0 = current.tendency_norm if tend_anchor is None else float(tend_anchor)
    tend_cap = max(tend0, cfg.tendency_tolerance) * cfg.max_tendency_increase
    for _ in range(cfg.max_line_search):
        cand, used = residual_at(h + alpha * dh, support)
        extra += used
        if cand is None:
            backs += 1
            reason = "domain"
            alpha *= 0.5
            used_alpha = alpha
            if alpha < cfg.min_line_search_factor:
                reason = "domain_min_alpha"
                break
            continue
        phi = residual_merit(cand.residual)
        armijo_ok = phi <= (1.0 - cfg.armijo_c * alpha) * phi0
        tendency_ok = cand.tendency_norm <= tend_cap
        if armijo_ok and tendency_ok:
            return cand, alpha, extra, backs, "accepted"
        backs += 1
        if not armijo_ok:
            reason = "armijo_not_met"
        else:
            reason = "tendency_increased"
        alpha *= 0.5
        used_alpha = alpha
        if alpha < cfg.min_line_search_factor:
            reason = (
                "armijo_not_met_min_alpha"
                if not armijo_ok
                else "tendency_increased_min_alpha"
            )
            break
    return accepted, used_alpha, extra, backs, reason


def _inner_converged(trial: TrialFluxes, cfg: SteadyRCEConfig) -> bool:
    return (
        trial.flux_flatness <= cfg.flux_flatness_tolerance
        and trial.tendency_norm <= cfg.tendency_tolerance
    )


def _make_newton_record(
    *,
    outer: int,
    newton: int,
    before: TrialFluxes,
    after: TrialFluxes | None,
    support: NDArray[np.bool_],
    solver: SolverConfig,
    dh: NDArray[np.float64] | None,
    h_scale: NDArray[np.float64],
    alpha: float,
    step_rel: float,
    ls_reason: str,
    ls_backs: int,
    direction: str | None,
    gmres_iters: int,
    gmres_rn: float,
    gmres_rhs: float,
    gmres_rtol: float,
    gmres_ok: bool,
    n_evals: int,
    lagged_flux: NDArray[np.float64] | None,
    fd_rel: float,
    jv_eps: float,
    jv_n_branch_crossings: int = 0,
    n_subspace_columns: int = 0,
    n_subspace_columns_rejected: int = 0,
) -> SteadyNewtonRecord:
    after_t = after if after is not None else before
    live_support = active_interface_mask(before.state.temperature.size, after_t.closure, solver)
    excess = mask_superadiabatic_excess(after_t.closure, support, solver)
    dh_acc = None if dh is None else alpha * dh
    if dh_acc is None:
        step_h_acc = 0.0
        step_t_acc = 0.0
    else:
        step_h_acc = float(np.max(np.abs(dh_acc) / h_scale, initial=0.0))
        t0 = before.state.temperature
        t1 = after_t.state.temperature
        t_scale = np.maximum(np.abs(t0), 1.0e-30)
        step_t_acc = float(np.max(np.abs(t1 - t0) / t_scale, initial=0.0))
    conv_rel = float("nan")
    if lagged_flux is not None:
        scale = max(float(np.max(np.abs(lagged_flux))), 1.0)
        conv_rel = float(np.max(np.abs(after_t.flux_conv - lagged_flux), initial=0.0)) / scale
    r_after = after_t.residual
    gmres_ratio = (
        float("nan")
        if not np.isfinite(gmres_rn) or gmres_rhs == 0.0
        else float(gmres_rn / gmres_rhs)
    )
    n_cross = count_internal_branch_crossings(
        before.state.temperature.size, after_t.closure, support, solver
    )
    return SteadyNewtonRecord(
        outer=outer,
        newton=newton,
        flux_flatness=after_t.flux_flatness,
        tendency_norm=after_t.tendency_norm,
        residual_two_norm=float(np.linalg.norm(r_after)),
        flux_flatness_before=before.flux_flatness,
        flux_flatness_after=after_t.flux_flatness,
        tendency_norm_before=before.tendency_norm,
        tendency_norm_after=after_t.tendency_norm,
        residual_two_norm_before=float(np.linalg.norm(before.residual)),
        residual_two_norm_after=float(np.linalg.norm(r_after)),
        residual_inf_before=float(np.max(np.abs(before.residual), initial=0.0)),
        residual_inf_after=float(np.max(np.abs(r_after), initial=0.0)),
        residual_before=np.asarray(before.residual, dtype=np.float64).tolist(),
        residual_after=np.asarray(r_after, dtype=np.float64).tolist(),
        step_rel=step_rel,
        step_rel_h_newton=step_rel,
        step_rel_h_accepted=step_h_acc,
        step_rel_T_accepted=step_t_acc,
        line_search_factor=alpha,
        line_search_reason=ls_reason,
        line_search_backtracks=ls_backs,
        direction=direction,
        gmres_iters=gmres_iters,
        gmres_residual_norm=gmres_rn,
        gmres_rhs_norm=gmres_rhs,
        gmres_rtol=gmres_rtol,
        gmres_success=gmres_ok,
        n_evals=n_evals,
        mask_before=np.asarray(support, dtype=bool).tolist(),
        mask_after=np.asarray(live_support, dtype=bool).tolist(),
        mask_changed=bool(not np.array_equal(live_support, support)),
        min_superadiabatic_excess_active=excess["min_superadiabatic_excess_active"],
        max_superadiabatic_excess_inactive=excess["max_superadiabatic_excess_inactive"],
        min_superadiabatic_excess_active_including_boundary=excess[
            "min_superadiabatic_excess_active_including_boundary"
        ],
        activity_threshold=excess["activity_threshold"],
        n_inactive_above_threshold=int(excess["n_inactive_above_threshold"]),
        rcb_active_excess=excess["rcb_active_excess"],
        rcb_inactive_excess=excess["rcb_inactive_excess"],
        rcb_active_distance_to_threshold=excess["rcb_active_distance_to_threshold"],
        rcb_inactive_distance_to_threshold=excess["rcb_inactive_distance_to_threshold"],
        n_branch_crossings=n_cross,
        jv_n_branch_crossings=jv_n_branch_crossings,
        n_subspace_columns=n_subspace_columns,
        n_subspace_columns_rejected=n_subspace_columns_rejected,
        merit_before=residual_merit(before.residual),
        merit_after=residual_merit(r_after),
        live_minus_lagged_conv_rel=conv_rel,
        fd_rel=fd_rel,
        jv_eps_used=jv_eps,
        h_scale_rms=float(np.sqrt(np.mean(np.square(h_scale)))),
        gmres_linear_residual_ratio=gmres_ratio,
    )


def solve_steady_rce(
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
    config: SteadyRCEConfig | None = None,
    initial_enthalpy: NDArray[np.float64] | None = None,
    progress: ProgressFn | None = None,
) -> SteadyRCEResult:
    """Damped Newton–Krylov on the flux-flatness residual with a frozen mask."""
    cfg = config or SteadyRCEConfig()
    grav = gravity or ConstantGravity(physics.gravity)
    f_int = _internal_flux_reference(lower_bc, None)
    if f_int is None:
        n = grid.n_layers
        empty = np.zeros(n + 1, dtype=np.float64)
        state0 = build_column_state(
            grid, np.asarray(initial_temperature, dtype=np.float64), thermo, grav
        )
        return SteadyRCEResult(
            SteadyRCEStatus.INVALID_BOUNDARY,
            "steady flux-defect solve requires LowerNetInternalFlux",
            state0,
            _evaluate_closure(grid, state0, physics, thermo),
            None,
            empty,
            empty,
            empty,
            np.zeros(n, dtype=np.float64),
            float("nan"),
            float("nan"),
            [],
            [],
            np.zeros(n + 1, dtype=bool),
            0,
            0,
            0,
            0,
            [],
        )
    f_int = float(f_int)
    f_scale = max(cfg.flux_scale_floor, abs(f_int))
    d_factor = (
        DEFAULT_DIFFUSIVITY if cfg.diffusivity_factor is None else float(cfg.diffusivity_factor)
    )

    t0 = np.asarray(initial_temperature, dtype=np.float64)
    if initial_enthalpy is None:
        h = np.asarray(thermo.enthalpy(t0), dtype=np.float64)
    else:
        h = np.asarray(initial_enthalpy, dtype=np.float64).copy()
        t0 = thermo.invert_enthalpy(h)

    history: list[SteadyNewtonRecord] = []
    n_evals = 0
    newton_total = 0
    backtracks = 0
    consecutive_ls_fail = 0
    support: NDArray[np.bool_] | None = None
    lagged_flux: NDArray[np.float64] | None = None

    def emit(event: dict[str, object]) -> None:
        if progress is not None:
            progress(event)

    def residual_at(
        enthalpy: NDArray[np.float64],
        frozen: NDArray[np.bool_] | None,
        *,
        lag_conv: bool = False,
    ) -> tuple[TrialFluxes | None, int]:
        trial = evaluate_trial(
            grid,
            enthalpy,
            physics,
            thermo,
            opacity,
            pressure,
            top_bc,
            lower_bc,
            grav,
            f_int=f_int,
            f_scale=f_scale,
            frozen_support=frozen,
            diffusivity_factor=d_factor,
            radiation_route=cfg.radiation_route,
            frozen_flux_conv=lagged_flux if lag_conv else None,
        )
        return trial, 1

    trial, used = residual_at(h, None)
    n_evals += used
    if trial is None:
        state0 = build_column_state(grid, t0, thermo, grav)
        return SteadyRCEResult(
            SteadyRCEStatus.DOMAIN_FAILURE,
            "initial enthalpy is outside the thermodynamic domain",
            state0,
            _evaluate_closure(grid, state0, physics, thermo),
            None,
            np.zeros(grid.n_layers + 1),
            np.zeros(grid.n_layers + 1),
            np.zeros(grid.n_layers + 1),
            np.zeros(grid.n_layers),
            float("nan"),
            float("nan"),
            [],
            [],
            np.zeros(grid.n_layers + 1, dtype=bool),
            0,
            0,
            0,
            n_evals,
            history,
        )

    regions, detached, support = _unfrozen_regions(trial, solver)
    frozen0, used = residual_at(h, support)
    n_evals += used
    if frozen0 is None:
        return _result(
            SteadyRCEStatus.DOMAIN_FAILURE,
            "frozen-mask evaluation of the initial state failed",
            trial,
            support,
            regions,
            detached,
            newton_total,
            backtracks,
            0,
            n_evals,
            history,
        )
    trial = frozen0
    lagged_flux = frozen0.flux_conv.copy()

    last_trial = trial
    inner_stop = "max_newton"
    last_mask_changed = False
    outer = 0
    for outer in range(1, cfg.max_mask_outer + 1):
        inner_ok = False
        consecutive_ls_fail = 0
        inner_stop = "max_newton"
        if _inner_converged(last_trial, cfg):
            inner_ok = True
            inner_stop = "converged"
        inner_tend0 = last_trial.tendency_norm
        for it in range(cfg.max_newton):
            if inner_ok:
                break
            h_scale = np.maximum(np.abs(h), cfg.h_floor)
            r0 = last_trial.residual
            r_two = float(np.linalg.norm(r0))
            logp = np.log(np.maximum(grid.pressure_centres, 1.0e-30))

            def apply_j_lagged(direction: NDArray[np.float64]) -> NDArray[np.float64] | None:
                nonlocal n_evals

                def r_of(enthalpy: NDArray[np.float64]) -> NDArray[np.float64] | None:
                    nonlocal n_evals
                    tr, used_local = residual_at(enthalpy, support, lag_conv=True)
                    n_evals += used_local
                    return None if tr is None else tr.residual

                return _finite_difference_jv(
                    r_of, h, r0, direction, cfg.fd_rel, cfg.h_floor
                )

            def trial_live(enthalpy: NDArray[np.float64]) -> TrialFluxes | None:
                nonlocal n_evals
                tr, used_local = residual_at(enthalpy, support, lag_conv=False)
                n_evals += used_local
                return tr

            gmres_iters = 0
            gmres_rn = float("nan")
            gmres_ok = False
            eta = min(
                cfg.gmres_rtol_max,
                max(cfg.gmres_rtol_min, 0.5 * last_trial.flux_flatness),
            )
            accepted = None
            ls_used = 1.0
            ls_reason = "no_attempt"
            last_backs = 0
            step_rel = 0.0
            chosen = None
            last_dh: NDArray[np.float64] | None = None
            n_subspace_cols = 0
            n_subspace_rejected = 0
            jv_crossings = 0
            dnorm = max(float(np.linalg.norm(h_scale)), cfg.h_floor)
            jv_eps = (
                cfg.subspace_fd_rel
                * max(float(np.linalg.norm(h)), cfg.h_floor * np.sqrt(h.size))
                / dnorm
            )

            def try_direction(name: str, raw_dh: NDArray[np.float64]) -> TrialFluxes | None:
                nonlocal ls_used, step_rel, chosen, n_evals, backtracks
                nonlocal ls_reason, last_backs, last_dh
                dh_loc, step_rel = _clip_step(raw_dh, h_scale, cfg.max_step_rel)
                last_dh = dh_loc
                cand, alpha, extra, backs, reason = _line_search(
                    h, dh_loc, residual_at, support, last_trial, cfg, inner_tend0
                )
                n_evals += extra
                backtracks += backs
                last_backs = backs
                ls_reason = reason
                ls_used = alpha
                if cand is not None:
                    chosen = name
                return cand

            if cfg.use_subspace:
                basis = _subspace_basis(
                    h_scale, _convective_layer_mask(h.size, support), logp
                )
                if basis is not None:
                    n_subspace_cols = int(basis.shape[1])
                    kept_basis = []
                    kept_jv = []
                    for k in range(basis.shape[1]):
                        jv, meta = _centred_live_jv(
                            trial_live,
                            h,
                            r0,
                            basis[:, k],
                            cfg.subspace_fd_rel,
                            cfg.h_floor,
                            frozen_support=support,
                            solver=solver,
                            n_layers=h.size,
                            reject_branch_crossing=cfg.reject_branch_crossing,
                        )
                        jv_crossings += int(meta.get("n_branch_crossings") or 0)
                        if jv is None or bool(meta.get("rejected")):
                            n_subspace_rejected += 1
                            continue
                        kept_basis.append(basis[:, k])
                        kept_jv.append(jv)
                    if kept_jv:
                        dh_sub = _least_squares_step(
                            np.stack(kept_basis, axis=1),
                            np.column_stack(kept_jv),
                            r0,
                        )
                        if dh_sub is not None and float(np.linalg.norm(dh_sub)) > 0.0:
                            accepted = try_direction("subspace", dh_sub)

            if accepted is None and cfg.use_gmres:
                def apply_j_scaled(v: NDArray[np.float64]) -> NDArray[np.float64]:
                    jv = apply_j_lagged(h_scale * np.asarray(v, dtype=np.float64))
                    return np.zeros_like(r0) if jv is None else jv

                step_scaled, gmres_iters, gmres_rn, gmres_ok = restarted_gmres(
                    apply_j_scaled,
                    -r0,
                    rtol=eta,
                    maxiter=cfg.gmres_maxiter,
                    restart=cfg.gmres_restart,
                )
                if gmres_ok or gmres_rn < r_two:
                    dh_g = h_scale * step_scaled
                    if float(np.linalg.norm(dh_g)) > 0.0:
                        gmres_name = "gmres" if gmres_ok else "gmres_fallback"
                        accepted = try_direction(gmres_name, dh_g)

            if accepted is None:
                accepted = try_direction("scaled_residual", -h_scale * r0)

            newton_total += 1
            if last_dh is not None:
                dnorm_dir = max(float(np.linalg.norm(last_dh)), cfg.h_floor)
                fd_for_eps = (
                    cfg.subspace_fd_rel if chosen == "subspace" else cfg.fd_rel
                )
                jv_eps = (
                    fd_for_eps
                    * max(float(np.linalg.norm(h)), cfg.h_floor * np.sqrt(h.size))
                    / dnorm_dir
                )
            rec = _make_newton_record(
                outer=outer,
                newton=newton_total,
                before=last_trial,
                after=accepted,
                support=support,
                solver=solver,
                dh=last_dh,
                h_scale=h_scale,
                alpha=ls_used,
                step_rel=step_rel,
                ls_reason=ls_reason,
                ls_backs=last_backs,
                direction=chosen,
                gmres_iters=gmres_iters,
                gmres_rn=gmres_rn,
                gmres_rhs=r_two,
                gmres_rtol=eta,
                gmres_ok=gmres_ok,
                n_evals=n_evals,
                lagged_flux=lagged_flux,
                fd_rel=cfg.subspace_fd_rel if chosen == "subspace" else cfg.fd_rel,
                jv_eps=jv_eps,
                jv_n_branch_crossings=jv_crossings,
                n_subspace_columns=n_subspace_cols,
                n_subspace_columns_rejected=n_subspace_rejected,
            )
            history.append(rec)
            if accepted is None:
                consecutive_ls_fail += 1
                emit(
                    {
                        "event": "line_search_failure",
                        "outer": outer,
                        "newton": newton_total,
                        "flux_flatness": last_trial.flux_flatness,
                        "line_search_reason": ls_reason,
                        "line_search_factor": ls_used,
                        "gmres_residual": gmres_rn,
                        "gmres_success": gmres_ok,
                        "consecutive": consecutive_ls_fail,
                    }
                )
                if consecutive_ls_fail >= cfg.max_consecutive_ls_fail:
                    inner_stop = "line_search_failure"
                    emit(
                        {
                            "event": "inner_stalled",
                            "outer": outer,
                            "newton": newton_total,
                            "flux_flatness": last_trial.flux_flatness,
                            "reason": "line_search_failure",
                        }
                    )
                    break
                continue
            consecutive_ls_fail = 0
            h = accepted.state.enthalpy.copy()
            last_trial = accepted
            lagged_flux = accepted.flux_conv.copy()
            emit(
                {
                    "event": "newton",
                    "outer": outer,
                    "newton": newton_total,
                    "flux_flatness": accepted.flux_flatness,
                    "flux_flatness_before": rec.flux_flatness_before,
                    "tendency_norm": accepted.tendency_norm,
                    "merit_before": rec.merit_before,
                    "merit_after": rec.merit_after,
                    "step_rel": step_rel,
                    "step_rel_h_accepted": rec.step_rel_h_accepted,
                    "step_rel_T_accepted": rec.step_rel_T_accepted,
                    "alpha": ls_used,
                    "line_search_reason": ls_reason,
                    "gmres_iters": gmres_iters,
                    "gmres_residual_norm": gmres_rn,
                    "gmres_success": gmres_ok,
                    "n_evals": n_evals,
                    "direction": chosen,
                    "min_superadiabatic_excess_active": rec.min_superadiabatic_excess_active,
                    "max_superadiabatic_excess_inactive": rec.max_superadiabatic_excess_inactive,
                    "rcb_active_distance_to_threshold": rec.rcb_active_distance_to_threshold,
                }
            )
            if _inner_converged(accepted, cfg):
                inner_ok = True
                inner_stop = "converged"
                break
            if len(history) >= 5:
                window = [item.residual_two_norm_after for item in history[-5:]]
                if window[0] - window[-1] < 1.0e-7 * max(window[0], 1.0e-12):
                    inner_stop = "inner_stalled"
                    emit(
                        {
                            "event": "inner_stalled",
                            "outer": outer,
                            "newton": newton_total,
                            "flux_flatness": accepted.flux_flatness,
                            "reason": "residual_two_norm_window",
                        }
                    )
                    break

        unfrozen, used = residual_at(h, None, lag_conv=False)
        n_evals += used
        if unfrozen is None:
            return _result(
                SteadyRCEStatus.DOMAIN_FAILURE,
                "unfrozen mask evaluation left the thermodynamic domain",
                last_trial,
                support,
                regions,
                detached,
                newton_total,
                backtracks,
                outer,
                n_evals,
                history,
            )
        if unfrozen.flux_flatness > max(1.0, 10.0 * last_trial.flux_flatness):
            emit(
                {
                    "event": "unfrozen_divergence",
                    "outer": outer,
                    "inner_flatness": last_trial.flux_flatness,
                    "unfrozen_flatness": unfrozen.flux_flatness,
                }
            )
            return _result(
                SteadyRCEStatus.NEWTON_LIMIT,
                "live MLT residual diverged after the inner Newton",
                last_trial,
                support,
                regions,
                detached,
                newton_total,
                backtracks,
                outer,
                n_evals,
                history,
            )
        new_regions, new_detached, new_support = _unfrozen_regions(unfrozen, solver)
        mask_changed = not np.array_equal(new_support, support)
        last_mask_changed = mask_changed
        emit(
            {
                "event": "mask",
                "outer": outer,
                "mask_changed": mask_changed,
                "inner_ok": inner_ok,
                "regions": new_regions,
                "detached": new_detached,
                "flux_flatness": unfrozen.flux_flatness,
            }
        )
        last_trial = unfrozen
        regions, detached = new_regions, new_detached
        if inner_ok and (not mask_changed) and _inner_converged(unfrozen, cfg):
            return _result(
                SteadyRCEStatus.CONVERGED,
                "mask unchanged and flux-flatness residual at gate",
                unfrozen,
                support,
                regions,
                detached,
                newton_total,
                backtracks,
                outer,
                n_evals,
                history,
            )
        if not mask_changed:
            break
        support = new_support
        lagged_flux = _apply_frozen_convective_flux(unfrozen.closure.flux, support)
        restarted, used = residual_at(h, support)
        n_evals += used
        if restarted is None:
            return _result(
                SteadyRCEStatus.DOMAIN_FAILURE,
                "restart with the updated mask left the thermodynamic domain",
                last_trial,
                support,
                regions,
                detached,
                newton_total,
                backtracks,
                outer,
                n_evals,
                history,
            )
        last_trial = restarted

    if last_mask_changed:
        status = SteadyRCEStatus.MASK_LIMIT
        reason = "convective mask kept changing through the outer iteration limit"
    elif inner_stop == "line_search_failure":
        status = SteadyRCEStatus.LINE_SEARCH_FAILURE
        reason = "line search failed to reduce the residual merit"
    elif inner_stop == "inner_stalled":
        status = SteadyRCEStatus.INNER_STALLED
        reason = "inner Newton stalled: residual two-norm window"
    else:
        status = SteadyRCEStatus.NEWTON_LIMIT
        reason = "inner Newton did not reach the flux-flatness gate; mask unchanged"
    return _result(
        status,
        reason,
        last_trial,
        support if support is not None else np.zeros(grid.n_layers + 1, dtype=bool),
        regions,
        detached,
        newton_total,
        backtracks,
        outer if cfg.max_mask_outer else 0,
        n_evals,
        history,
    )

def _result(
    status: SteadyRCEStatus,
    reason: str,
    trial: TrialFluxes,
    support: NDArray[np.bool_],
    regions: list[tuple[int, int]],
    detached: list[tuple[int, int]],
    newton: int,
    backtracks: int,
    outer: int,
    n_evals: int,
    history: list[SteadyNewtonRecord],
) -> SteadyRCEResult:
    return SteadyRCEResult(
        status=status,
        reason=reason,
        state=trial.state,
        closure=trial.closure,
        radiation=trial.radiation,
        flux_conv=trial.flux_conv,
        flux_rad=trial.flux_rad,
        flux_total=trial.flux_total,
        residual=trial.residual,
        flux_flatness=trial.flux_flatness,
        tendency_norm=trial.tendency_norm,
        convective_regions=list(regions),
        detached_convective_regions=list(detached),
        frozen_support=np.asarray(support, dtype=bool),
        newton_iterations=newton,
        line_search_backtracks=backtracks,
        mask_outer_iterations=outer,
        n_evals=n_evals,
        history=history,
    )
