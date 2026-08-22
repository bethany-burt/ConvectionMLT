"""Conservative backward-Euler convection substep with damped Newton.

This is a Stage 4 semi-implicit convection integrator. It is not the Stage 6
monolithic total-flux Newton solver. Radiation / external flux remains
explicit; only the MLT redistribution is treated implicitly.

Scope: constant-g, fixed-pressure Stage 4 columns. Each interface flux
depends only on adjacent layer temperatures, so ∂R/∂h is tridiagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .closure import ClosureResult, mixing_length_flux
from .config import PhysicsConfig, SolverConfig
from .gravity import ConstantGravity, GravityLaw
from .grid import PressureGrid
from .hydrostatics import HydrostaticDomainError
from .state import ColumnState, build_column_state
from .thermodynamics import EnthalpyInversionError, ThermoDomainError, ThermoProvider
from .tridiagonal import ThomasPivotError, thomas_solve_checked


class BorderedSchurError(ValueError):
    """wᵀz is too small to invert the conservation-constrained correction."""


@dataclass(frozen=True)
class ImplicitConvectionConfig:
    # Accepted-macrostep residual (ordinary unfrozen closure).
    residual_tolerance: float = 1.0e-10
    step_tolerance: float = 1.0e-10
    newton_residual_tolerance: float = 1.0e-12
    newton_step_tolerance: float = 1.0e-12
    conservation_tolerance: float = 1.0e-12
    schur_floor: float = 1.0e-12
    max_newton: int = 40
    max_line_search: int = 20
    min_line_search_factor: float = 1.0e-8
    armijo_c: float = 1.0e-4
    max_mask_outer: int = 8
    max_projection_continues: int = 8
    h_floor: float = 1.0e-30
    pivot_floor: float = 1.0e-30
    fd_rel: float = float(np.sqrt(np.finfo(np.float64).eps))
    # Production path is the bordered Newton. Projection is diagnostic only.
    use_bordered_newton: bool = True
    use_production_projection: bool = False


@dataclass(frozen=True)
class ImplicitConvectionDiagnostics:
    newton_iterations: int
    line_search_backtracks: int
    mask_outer_iterations: int
    residual_norm: float
    step_norm: float
    mlt_evals: int
    column_enthalpy_change: float
    rejection_reason: str | None = None
    conservation_residual: float = float("nan")
    multiplier: float = float("nan")
    schur_denominator: float = float("nan")
    projection_residual_diagnostic: float = float("nan")


@dataclass(frozen=True)
class ImplicitConvectionResult:
    ok: bool
    state: ColumnState
    closure: ClosureResult
    f_conv: NDArray[np.float64]
    diagnostics: ImplicitConvectionDiagnostics


def _empty_closure(n_iface: int) -> ClosureResult:
    z = np.full(n_iface, np.nan)
    return ClosureResult(
        z, z, z, z, z, z, z, z, z, z,
        np.zeros(n_iface, dtype=bool),
        np.zeros(n_iface, dtype=bool),
        None,
    )


def _fail(
    state: ColumnState,
    reason: str,
    *,
    newton: int = 0,
    backtracks: int = 0,
    mask_outer: int = 0,
    resid: float = float("nan"),
    step: float = float("nan"),
    mlt: int = 0,
    dH: float = float("nan"),
    conservation: float = float("nan"),
    multiplier: float = float("nan"),
    schur: float = float("nan"),
    proj_resid: float = float("nan"),
) -> ImplicitConvectionResult:
    n_iface = state.temperature.size + 1
    return ImplicitConvectionResult(
        False,
        state,
        _empty_closure(n_iface),
        np.full(n_iface, np.nan),
        ImplicitConvectionDiagnostics(
            newton, backtracks, mask_outer, resid, step, mlt, dH, reason,
            conservation, multiplier, schur, proj_resid,
        ),
    )


def require_constant_gravity(gravity: GravityLaw) -> None:
    """Stage 4 implicit Jacobian is tridiagonal only for constant g.

    Variable gravity makes hydrostatic reconstruction (and mass path) nonlocal
    in temperature; that coupling is a separately derived extension.
    """
    if not isinstance(gravity, ConstantGravity):
        raise ValueError(
            "implicit convection requires ConstantGravity; "
            "variable-g Jacobian is nonlocal and out of Stage 4 scope"
        )


def provisional_support(closure: ClosureResult) -> NDArray[np.bool_]:
    """Physical support for the Newton active-set: δ > 0 on all interfaces."""
    return np.asarray(closure.superadiabaticity > 0.0, dtype=bool)


def flux_with_provisional_support(
    closure: ClosureResult, support: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Zero interface flux outside the provisional support (numerical only)."""
    flux = np.asarray(closure.flux, dtype=np.float64).copy()
    inactive = ~np.asarray(support, dtype=bool)
    flux[inactive] = 0.0
    flux[0] = 0.0
    flux[-1] = 0.0
    return flux


def scaled_residual_norm(
    residual: NDArray[np.float64],
    h_star: NDArray[np.float64],
    temperature: NDArray[np.float64],
    cp: NDArray[np.float64],
    h_floor: float,
) -> float:
    scale = np.maximum(np.maximum(np.abs(h_star), cp * temperature), h_floor)
    return float(np.max(np.abs(residual) / scale, initial=0.0))


def _residual(
    h: NDArray[np.float64],
    h_star: NDArray[np.float64],
    f_conv: NDArray[np.float64],
    mass_path: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    return h - h_star - dt * (f_conv[:-1] - f_conv[1:]) / mass_path


def _state_from_h(
    grid: PressureGrid,
    h: NDArray[np.float64],
    thermo: ThermoProvider,
    gravity: GravityLaw,
) -> ColumnState | None:
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return None
    try:
        t = thermo.invert_enthalpy(h)
        if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
            return None
        _ = thermo.specific_heat(t)
        return build_column_state(grid, t, thermo, gravity, enthalpy=h)
    except (ThermoDomainError, EnthalpyInversionError, HydrostaticDomainError):
        return None


def evaluate_mlt(
    grid: PressureGrid,
    state: ColumnState,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
) -> ClosureResult:
    return mixing_length_flux(
        grid,
        state.temperature,
        state.g_edges,
        physics.alpha,
        thermo,
        physics.closure_prefactor,
        use_entropy_instability=True,
    )


def _residual_at(
    grid: PressureGrid,
    h: NDArray[np.float64],
    h_star: NDArray[np.float64],
    support: NDArray[np.bool_],
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    mass_path: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64] | None, int]:
    st = _state_from_h(grid, h, thermo, gravity)
    if st is None:
        return None, 0
    cl = evaluate_mlt(grid, st, physics, thermo)
    f = flux_with_provisional_support(cl, support)
    return _residual(h, h_star, f, mass_path, dt), 1


def conservation_weight(mass_path: NDArray[np.float64]) -> NDArray[np.float64]:
    mass = np.asarray(mass_path, dtype=np.float64)
    total = float(np.sum(mass))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("mass_path must have a positive finite sum")
    return mass / total


def conservation_residual(
    h: NDArray[np.float64],
    h_star: NDArray[np.float64],
    mass_path: NDArray[np.float64],
) -> float:
    w = conservation_weight(mass_path)
    return float(np.dot(w, np.asarray(h, dtype=np.float64) - np.asarray(h_star, dtype=np.float64)))


def _conservation_scale(h_star: NDArray[np.float64], h_floor: float) -> float:
    return max(float(np.max(np.abs(h_star))), h_floor)


def _merit(residual_norm: float, conservation: float, c_scale: float) -> float:
    return float(residual_norm) + abs(float(conservation)) / c_scale


def solve_bordered_correction(
    lower: NDArray[np.float64],
    diag: NDArray[np.float64],
    upper: NDArray[np.float64],
    residual: NDArray[np.float64],
    weight: NDArray[np.float64],
    conservation: float,
    multiplier: float,
    *,
    pivot_floor: float,
    schur_floor: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Two-Thomas solve of the conservation-bordered Newton system.

    [J  1] [δh] = [-(R + λ 1)]
    [wᵀ 0] [δλ]   [     -C    ]
    """
    ones = np.ones(diag.size, dtype=np.float64)
    rhs_y = -(np.asarray(residual, dtype=np.float64) + float(multiplier) * ones)
    y = thomas_solve_checked(lower, diag, upper, rhs_y, pivot_floor=pivot_floor)
    z = thomas_solve_checked(lower, diag, upper, ones, pivot_floor=pivot_floor)
    schur = float(np.dot(weight, z))
    scale = max(float(np.linalg.norm(weight) * np.linalg.norm(z)), 1.0e-30)
    if not np.isfinite(schur) or abs(schur) < schur_floor * scale:
        raise BorderedSchurError(
            f"bordered Schur denominator is poorly conditioned ({schur!r}, scale={scale!r})"
        )
    delta_lam = (float(conservation) + float(np.dot(weight, y))) / schur
    delta_h = y - z * delta_lam
    return delta_h, float(delta_lam), schur


def solve_bordered_dense(
    jacobian: NDArray[np.float64],
    residual: NDArray[np.float64],
    weight: NDArray[np.float64],
    conservation: float,
    multiplier: float,
) -> tuple[NDArray[np.float64], float]:
    """Dense (N+1)×(N+1) reference for the same bordered system."""
    n = jacobian.shape[0]
    a = np.zeros((n + 1, n + 1), dtype=np.float64)
    a[:n, :n] = jacobian
    a[:n, n] = 1.0
    a[n, :n] = weight
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[:n] = -(np.asarray(residual, dtype=np.float64) + float(multiplier) * np.ones(n))
    rhs[n] = -float(conservation)
    sol = np.linalg.solve(a, rhs)
    return sol[:n].copy(), float(sol[n])


def conservative_projection(
    grid: PressureGrid,
    h_star: NDArray[np.float64],
    state: ColumnState,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    mass_path: NDArray[np.float64],
    dt: float,
    cfg: ImplicitConvectionConfig,
) -> tuple[ColumnState | None, NDArray[np.float64], float, int]:
    """Diagnostic comparator only. Not used on the production accept path."""
    closure = evaluate_mlt(grid, state, physics, thermo)
    f = np.asarray(closure.flux, dtype=np.float64).copy()
    f[0] = 0.0
    f[-1] = 0.0
    h_proj = np.asarray(h_star, dtype=np.float64) + dt * (f[:-1] - f[1:]) / mass_path
    state_proj = _state_from_h(grid, h_proj, thermo, gravity)
    if state_proj is None:
        return None, f, float("nan"), 1
    closure_proj = evaluate_mlt(grid, state_proj, physics, thermo)
    f_proj = np.asarray(closure_proj.flux, dtype=np.float64).copy()
    f_proj[0] = 0.0
    f_proj[-1] = 0.0
    r_proj = _residual(h_proj, h_star, f_proj, mass_path, dt)
    cp = thermo.specific_heat(state_proj.temperature)
    resid = scaled_residual_norm(r_proj, h_star, state_proj.temperature, cp, cfg.h_floor)
    return state_proj, f_proj, resid, 2


def assemble_tridiagonal_jacobian(
    grid: PressureGrid,
    state: ColumnState,
    h_star: NDArray[np.float64],
    support: NDArray[np.bool_],
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    mass_path: NDArray[np.float64],
    dt: float,
    cfg: ImplicitConvectionConfig,
    residual0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Finite-difference tridiagonal ∂R/∂h with frozen mass_path and support.

    Uses a 3-colour centred-difference stencil: layers congruent mod 3 are
    perturbed together. Because ∂R/∂h is tridiagonal, each residual row sees
    only one perturbed neighbour per colour, so each Jacobian column is
    recovered from the colour residual (6 MLT calls total, not 2N).
    """
    n = grid.n_layers
    h0 = state.enthalpy.copy()
    lower = np.zeros(n - 1, dtype=np.float64)
    diag = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n - 1, dtype=np.float64)
    mlt_evals = 0
    h_scale = np.maximum(np.abs(h0), np.maximum(np.abs(h_star), cfg.h_floor))

    for colour in range(3):
        js = list(range(colour, n, 3))
        if not js:
            continue
        eps = np.zeros(n, dtype=np.float64)
        h_plus = h0.copy()
        h_minus = h0.copy()
        for j in js:
            e = cfg.fd_rel * h_scale[j]
            if e == 0.0:
                e = cfg.fd_rel * cfg.h_floor
            eps[j] = e
            h_plus[j] = h0[j] + e
            h_minus[j] = h0[j] - e
        r_plus, m1 = _residual_at(
            grid, h_plus, h_star, support, physics, thermo, gravity, mass_path, dt
        )
        r_minus, m2 = _residual_at(
            grid, h_minus, h_star, support, physics, thermo, gravity, mass_path, dt
        )
        mlt_evals += m1 + m2
        if r_plus is not None and r_minus is not None:
            delta_r = r_plus - r_minus
            denom_factor = 2.0
        elif r_plus is not None:
            delta_r = r_plus - residual0
            denom_factor = 1.0
        elif r_minus is not None:
            delta_r = residual0 - r_minus
            denom_factor = 1.0
        else:
            raise ThermoDomainError("jacobian finite-difference out of thermodynamic domain")
        for j in js:
            e = denom_factor * eps[j]
            diag[j] = delta_r[j] / e
            if j > 0:
                upper[j - 1] = delta_r[j - 1] / e  # A[j-1, j]
            if j < n - 1:
                lower[j] = delta_r[j + 1] / e  # A[j+1, j]

    return lower, diag, upper, mlt_evals


def assemble_dense_jacobian(
    grid: PressureGrid,
    state: ColumnState,
    h_star: NDArray[np.float64],
    support: NDArray[np.bool_],
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    mass_path: NDArray[np.float64],
    dt: float,
    cfg: ImplicitConvectionConfig,
    residual0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Full dense FD Jacobian for small-N verification against the tridiagonal."""
    n = grid.n_layers
    h0 = state.enthalpy.copy()
    jac = np.zeros((n, n), dtype=np.float64)
    h_scale = np.maximum(np.abs(h0), np.maximum(np.abs(h_star), cfg.h_floor))

    for j in range(n):
        eps = cfg.fd_rel * h_scale[j]
        if eps == 0.0:
            eps = cfg.fd_rel * cfg.h_floor
        h_plus = h0.copy()
        h_minus = h0.copy()
        h_plus[j] = h0[j] + eps
        h_minus[j] = h0[j] - eps
        r_plus, _ = _residual_at(
            grid, h_plus, h_star, support, physics, thermo, gravity, mass_path, dt
        )
        r_minus, _ = _residual_at(
            grid, h_minus, h_star, support, physics, thermo, gravity, mass_path, dt
        )
        if r_plus is not None and r_minus is not None:
            jac[:, j] = (r_plus - r_minus) / (2.0 * eps)
        elif r_plus is not None:
            jac[:, j] = (r_plus - residual0) / eps
        elif r_minus is not None:
            jac[:, j] = (residual0 - r_minus) / eps
        else:
            raise ThermoDomainError("dense jacobian out of thermodynamic domain")
    return jac


def _newton_start(
    grid: PressureGrid,
    state_init: ColumnState,
    h_star: NDArray[np.float64],
    thermo: ThermoProvider,
    gravity: GravityLaw,
) -> tuple[ColumnState, NDArray[np.float64]]:
    """Use h* only if it inverts. Otherwise start from a valid committed state."""
    st_star = _state_from_h(grid, h_star, thermo, gravity)
    if st_star is not None:
        return st_star, np.asarray(h_star, dtype=np.float64).copy()
    return state_init, np.asarray(state_init.enthalpy, dtype=np.float64).copy()


def solve_implicit_convection(
    grid: PressureGrid,
    state_init: ColumnState,
    h_star: NDArray[np.float64],
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    mass_path: NDArray[np.float64],
    dt: float,
    solver: SolverConfig,
    cfg: ImplicitConvectionConfig | None = None,
) -> ImplicitConvectionResult:
    """Backward-Euler convection with a conservation-bordered active-set Newton.

    ``state_init`` must be a valid invertible state. ``h_star`` is the radiative
    provisional enthalpy and need not invert.
    """
    del solver  # reserved for future fractional-T / accuracy hooks
    icfg = cfg or ImplicitConvectionConfig()
    require_constant_gravity(gravity)
    if dt <= 0.0 or not np.isfinite(dt):
        return _fail(state_init, "implicit convection requires positive finite dt")

    h_star = np.asarray(h_star, dtype=np.float64).copy()
    mass_path = np.asarray(mass_path, dtype=np.float64).copy()
    w = conservation_weight(mass_path)
    c_scale = _conservation_scale(h_star, icfg.h_floor)
    state, h = _newton_start(grid, state_init, h_star, thermo, gravity)
    total_newton = 0
    total_backtracks = 0
    total_mlt = 0
    mask_outer = 0
    prev_support: NDArray[np.bool_] | None = None
    prev_prev_support: NDArray[np.bool_] | None = None
    step_norm = float("inf")
    lam = 0.0
    schur = float("nan")
    conservation = conservation_residual(h, h_star, mass_path)
    residual_norm = float("inf")

    for mask_outer in range(1, icfg.max_mask_outer + 1):
        closure0 = evaluate_mlt(grid, state, physics, thermo)
        total_mlt += 1
        support = provisional_support(closure0)
        if prev_prev_support is not None and np.array_equal(support, prev_prev_support):
            return _fail(
                state_init,
                "mask_settling_failure: two-cycle oscillation",
                newton=total_newton,
                backtracks=total_backtracks,
                mask_outer=mask_outer,
                resid=residual_norm,
                mlt=total_mlt,
                conservation=conservation,
                multiplier=lam,
                schur=schur,
            )

        converged_inner = False
        f_conv = flux_with_provisional_support(closure0, support)
        for _newton in range(icfg.max_newton):
            total_newton += 1
            closure = evaluate_mlt(grid, state, physics, thermo)
            total_mlt += 1
            f_conv = flux_with_provisional_support(closure, support)
            residual = _residual(h, h_star, f_conv, mass_path, dt)
            conservation = conservation_residual(h, h_star, mass_path)
            cp = thermo.specific_heat(state.temperature)
            residual_norm = scaled_residual_norm(
                residual, h_star, state.temperature, cp, icfg.h_floor
            )
            c_ok = abs(conservation) / c_scale <= icfg.conservation_tolerance
            resid_ok = residual_norm <= icfg.newton_residual_tolerance
            if resid_ok and c_ok:
                if _newton == 0:
                    step_norm = 0.0
                converged_inner = True
                break

            try:
                lower, diag, upper, mlt_j = assemble_tridiagonal_jacobian(
                    grid, state, h_star, support, physics, thermo, gravity,
                    mass_path, dt, icfg, residual,
                )
                total_mlt += mlt_j
                if icfg.use_bordered_newton:
                    delta_h, delta_lam, schur = solve_bordered_correction(
                        lower, diag, upper, residual, w, conservation, lam,
                        pivot_floor=icfg.pivot_floor,
                        schur_floor=icfg.schur_floor,
                    )
                else:
                    delta_h = thomas_solve_checked(
                        lower, diag, upper, -residual, pivot_floor=icfg.pivot_floor
                    )
                    delta_lam = 0.0
                    schur = float("nan")
            except BorderedSchurError as exc:
                return _fail(
                    state_init,
                    f"bordered_schur_failure: {exc}",
                    newton=total_newton,
                    backtracks=total_backtracks,
                    mask_outer=mask_outer,
                    resid=residual_norm,
                    step=step_norm,
                    mlt=total_mlt,
                    conservation=conservation,
                    multiplier=lam,
                    schur=schur,
                )
            except ThomasPivotError as exc:
                return _fail(
                    state_init,
                    f"thomas_pivot_failure: {exc}",
                    newton=total_newton,
                    backtracks=total_backtracks,
                    mask_outer=mask_outer,
                    resid=residual_norm,
                    mlt=total_mlt,
                    conservation=conservation,
                    multiplier=lam,
                    schur=schur,
                )
            except ThermoDomainError as exc:
                return _fail(
                    state_init,
                    f"jacobian_domain_failure: {exc}",
                    newton=total_newton,
                    backtracks=total_backtracks,
                    mask_outer=mask_outer,
                    resid=residual_norm,
                    mlt=total_mlt,
                    conservation=conservation,
                    multiplier=lam,
                    schur=schur,
                )

            merit0 = _merit(residual_norm, conservation, c_scale)
            alpha = 1.0
            accepted_ls = False
            h_scale = np.maximum(np.abs(h), icfg.h_floor)
            for _ls in range(icfg.max_line_search):
                if alpha < icfg.min_line_search_factor:
                    break
                h_trial = h + alpha * delta_h
                st_trial = _state_from_h(grid, h_trial, thermo, gravity)
                if st_trial is None:
                    total_backtracks += 1
                    alpha *= 0.5
                    continue
                cl_trial = evaluate_mlt(grid, st_trial, physics, thermo)
                total_mlt += 1
                f_trial = flux_with_provisional_support(cl_trial, support)
                r_trial = _residual(h_trial, h_star, f_trial, mass_path, dt)
                c_trial = conservation_residual(h_trial, h_star, mass_path)
                cp_trial = thermo.specific_heat(st_trial.temperature)
                r_norm_trial = scaled_residual_norm(
                    r_trial, h_star, st_trial.temperature, cp_trial, icfg.h_floor
                )
                merit_trial = _merit(r_norm_trial, c_trial, c_scale)
                if merit_trial <= (1.0 - icfg.armijo_c * alpha) * max(merit0, 1.0e-30):
                    h = h_trial
                    state = st_trial
                    lam = lam + alpha * delta_lam
                    step_norm = float(np.max(np.abs(alpha * delta_h) / h_scale, initial=0.0))
                    residual_norm = r_norm_trial
                    conservation = c_trial
                    accepted_ls = True
                    break
                total_backtracks += 1
                alpha *= 0.5

            if not accepted_ls:
                if (
                    residual_norm <= icfg.residual_tolerance
                    and abs(conservation) / c_scale <= icfg.conservation_tolerance
                ):
                    converged_inner = True
                    break
                return _fail(
                    state_init,
                    "line_search_exhaustion",
                    newton=total_newton,
                    backtracks=total_backtracks,
                    mask_outer=mask_outer,
                    resid=residual_norm,
                    step=step_norm,
                    mlt=total_mlt,
                    conservation=conservation,
                    multiplier=lam,
                    schur=schur,
                )

        if not converged_inner:
            if (
                residual_norm <= icfg.residual_tolerance
                and abs(conservation) / c_scale <= icfg.conservation_tolerance
            ):
                pass
            else:
                return _fail(
                    state_init,
                    "newton_max_iterations",
                    newton=total_newton,
                    backtracks=total_backtracks,
                    mask_outer=mask_outer,
                    resid=residual_norm,
                    step=step_norm,
                    mlt=total_mlt,
                    conservation=conservation,
                    multiplier=lam,
                    schur=schur,
                )

        closure_unfrozen = evaluate_mlt(grid, state, physics, thermo)
        total_mlt += 1
        f_unfrozen = np.asarray(closure_unfrozen.flux, dtype=np.float64).copy()
        f_unfrozen[0] = 0.0
        f_unfrozen[-1] = 0.0
        support_new = provisional_support(closure_unfrozen)
        r_unfrozen = _residual(h, h_star, f_unfrozen, mass_path, dt)
        cp = thermo.specific_heat(state.temperature)
        resid_unfrozen = scaled_residual_norm(
            r_unfrozen, h_star, state.temperature, cp, icfg.h_floor
        )
        conservation = conservation_residual(h, h_star, mass_path)
        proj_state, _f_proj, proj_resid, m_proj = conservative_projection(
            grid, h_star, state, physics, thermo, gravity, mass_path, dt, icfg
        )
        total_mlt += m_proj
        del proj_state

        accept = (
            resid_unfrozen <= icfg.residual_tolerance
            and abs(conservation) / c_scale <= icfg.conservation_tolerance
            and np.array_equal(support_new, support)
        )
        if accept:
            dH = float(np.sum(mass_path * h) - np.sum(mass_path * h_star))
            return ImplicitConvectionResult(
                True,
                state,
                closure_unfrozen,
                f_unfrozen,
                ImplicitConvectionDiagnostics(
                    total_newton,
                    total_backtracks,
                    mask_outer,
                    resid_unfrozen,
                    step_norm,
                    total_mlt,
                    dH,
                    None,
                    conservation,
                    lam,
                    schur,
                    proj_resid,
                ),
            )

        if np.array_equal(support_new, support):
            return _fail(
                state_init,
                "active_set_residual_failure",
                newton=total_newton,
                backtracks=total_backtracks,
                mask_outer=mask_outer,
                resid=resid_unfrozen,
                step=step_norm,
                mlt=total_mlt,
                conservation=conservation,
                multiplier=lam,
                schur=schur,
                proj_resid=proj_resid,
            )

        prev_prev_support = prev_support
        prev_support = support
        # Continue from the Newton state with the new physical support.
        step_norm = float("inf")
        lam = 0.0

    return _fail(
        state_init,
        "mask_settling_failure: max outer iterations",
        newton=total_newton,
        backtracks=total_backtracks,
        mask_outer=mask_outer,
        resid=residual_norm,
        mlt=total_mlt,
        conservation=conservation,
        multiplier=lam,
        schur=schur,
    )
