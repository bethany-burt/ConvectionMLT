"""Reduced radiative-matching RCE that keeps a finite MLT superadiabatic excess.

The five-mode live-MLT subspace Newton is a documented negative result: it
minimised ½∥r∥₂² while the flux-flatness and tendency gates worsened. The
lagged Picard invert-MLT reconstruction is also documented: it moved the
global flux level but used a stale F_rad, so F_conv inferred from that field
was not the flux required by the reconstructed state.

The coupled path treats T_RCB as the primary global unknown. For each trial
value it reconstructs the CZ with the finite-MLT correction and solves the
radiative zone on the discrete Stage-3 two-stream operator so that
F_rad,j+1 − F_rad,j = 0 through every radiative layer. A scalar secant then
drives F_top − F_int → 0. An outer loop may update the RCB location. An
exactly adiabatic CZ is forbidden: F_conv ∝ (Δ∇)^{3/2} would then be
identically zero.

The reconstruction is an accelerator. The returned temperature must still be
polished with the unchanged live-MLT pseudo-time integrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .config import PhysicsConfig, SolverConfig
from .gravity import ConstantGravity, GravityLaw
from .grid import PressureGrid
from .opacity import PrescribedOpacity
from .radiation import (
    DEFAULT_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    LowerBoundary,
    SolveRoute,
    TopIrradiation,
    _sweep_down,
    _sweep_up,
)
from .rce import (
    _internal_flux_reference,
    _temperature_on_adiabat,
    grey_radiative_equilibrium_temperature,
)
from .state import build_column_state
from .steady_rce import (
    TrialFluxes,
    active_interface_mask,
    evaluate_trial,
    mask_superadiabatic_excess,
    residual_merit,
)
from .thermodynamics import ThermoProvider


class ReducedRCEStatus(str, Enum):
    CONVERGED = "converged"
    PICARD_STALL = "picard_stall"
    MATCHED = "matched"
    SECANT_STALL = "secant_stall"
    NO_IMPROVEMENT = "no_improvement"
    DOMAIN_FAILURE = "domain_failure"
    INVALID_BOUNDARY = "invalid_boundary"


@dataclass(frozen=True)
class ReducedRCEConfig:
    flux_flatness_tolerance: float = 1.0e-3
    tendency_tolerance: float = 1.0e-3
    flux_scale_floor: float = 1.0e-30
    max_picard: int = 24
    damping_init: float = 1.0
    damping_min: float = 1.0e-3
    damping_cut: float = 0.5
    max_consecutive_reject: int = 5
    logt_shift_max: float = 0.02
    n_logt_shifts: int = 9
    radiation_route: SolveRoute = SolveRoute.THOMAS
    diffusivity_factor: float | None = None
    match_rz_to_grey_re: bool = True
    rz_blend: float = 0.35
    coupling: str = "consistent"
    max_secant: int = 16
    t_rcb_rel_bracket: float = 0.03
    f_top_tolerance: float = 1.0e-3
    max_inner_picard: int = 8
    inner_damping: float = 0.5
    inner_delta_atol: float = 1.0e-12
    max_rcb_outer: int = 4
    rz_mode: str = "discrete"
    max_rz_kappa_picard: int = 4


@dataclass
class InnerPicardDiagnostics:
    n_inner: int = 0
    inner_converged: bool = False
    delta_abs_mismatch: float = float("nan")
    delta_rel_mismatch: float = float("nan")
    max_cz_mlt_flux_mismatch: float = float("nan")
    rz_max_flux_divergence: float = float("nan")
    rz_linear_ok: bool = True


@dataclass
class ReducedPicardRecord:
    picard: int
    flux_flatness: float
    tendency_norm: float
    residual_two_norm: float
    merit: float
    worst_gate: float
    damping: float
    logt_shift: float
    rcb_layer: int
    min_superadiabatic_excess_active: float
    accepted: bool
    reason: str
    stage: str = "picard"
    t_rcb: float = float("nan")
    f_top: float = float("nan")
    f_top_defect: float = float("nan")
    inner_picard: int = 0
    inner_converged: bool = False
    inner_delta_abs_mismatch: float = float("nan")
    inner_delta_rel_mismatch: float = float("nan")
    max_cz_mlt_flux_mismatch: float = float("nan")
    rz_max_flux_divergence: float = float("nan")


@dataclass(frozen=True)
class ReducedRCEResult:
    status: ReducedRCEStatus
    reason: str
    temperature: NDArray[np.float64]
    trial: TrialFluxes | None
    flux_flatness: float
    tendency_norm: float
    worst_gate: float
    convective_regions: list[tuple[int, int]]
    rcb_layer: int
    min_superadiabatic_excess_active: float
    n_picard: int
    n_evals: int
    improved: bool
    history: list[ReducedPicardRecord] = field(default_factory=list)
    f_top: float = float("nan")
    t_rcb: float = float("nan")
    f_top_defect: float = float("nan")
    n_inner_picard: int = 0
    n_rcb_outer: int = 0
    inner_converged: bool = False
    inner_delta_abs_mismatch: float = float("nan")
    inner_delta_rel_mismatch: float = float("nan")
    max_cz_mlt_flux_mismatch: float = float("nan")
    rz_max_flux_divergence: float = float("nan")


def worst_gate_violation(flatness: float, tendency: float, cfg: ReducedRCEConfig) -> float:
    return max(
        float(flatness) / cfg.flux_flatness_tolerance,
        float(tendency) / cfg.tendency_tolerance,
    )


def mlt_flux_coefficient(
    closure,
    physics: PhysicsConfig,
    g_edges: NDArray[np.float64],
    thermo: ThermoProvider,
) -> NDArray[np.float64]:
    """C on interfaces such that F_conv = C (Δ∇)^{3/2} for Δ∇ ≥ 0.

    Boundaries are identically zero: the closure does not apply MLT there.
    """
    n = int(np.asarray(closure.flux).size)
    coeff = np.zeros(n, dtype=np.float64)
    if n < 3:
        return coeff
    internal = slice(1, -1)
    hp = np.asarray(closure.scale_height, dtype=np.float64)[internal]
    ell = np.asarray(closure.mixing_length, dtype=np.float64)[internal]
    rho = np.asarray(closure.density_edges, dtype=np.float64)[internal]
    t_edge = np.asarray(closure.temperature_edges, dtype=np.float64)[internal]
    g = np.asarray(g_edges, dtype=np.float64)[internal]
    cp = np.asarray(thermo.specific_heat(t_edge), dtype=np.float64)
    safe = (hp > 0.0) & (g > 0.0) & np.isfinite(hp) & np.isfinite(g)
    c_int = np.zeros_like(hp)
    c_int[safe] = (
        float(physics.closure_prefactor)
        * rho[safe]
        * cp[safe]
        * t_edge[safe]
        * (ell[safe] / hp[safe])
        * ell[safe]
        * np.sqrt(g[safe] / hp[safe])
    )
    coeff[internal] = c_int
    return coeff


def invert_mlt_excess(
    flux_conv: NDArray[np.float64],
    coefficient: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Δ∇ = (F_conv / C)^{2/3} where C > 0 and F_conv > 0; else 0."""
    f = np.asarray(flux_conv, dtype=np.float64)
    c = np.asarray(coefficient, dtype=np.float64)
    delta = np.zeros_like(f)
    ok = (c > 0.0) & (f > 0.0) & np.isfinite(c) & np.isfinite(f)
    delta[ok] = (f[ok] / c[ok]) ** (2.0 / 3.0)
    return delta


def required_convective_flux(
    flux_rad: NDArray[np.float64],
    f_int: float,
    support: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """F_int − F_rad on the frozen CZ support only. Never ignites the RZ."""
    f_req = np.maximum(float(f_int) - np.asarray(flux_rad, dtype=np.float64), 0.0)
    mask = np.asarray(support, dtype=bool).copy()
    if mask.size:
        mask[0] = False
        mask[-1] = False
    return np.where(mask, f_req, 0.0)


def rcb_layer_from_support(support: NDArray[np.bool_]) -> int:
    internal = np.asarray(support, dtype=bool).copy()
    if internal.size:
        internal[0] = False
        internal[-1] = False
    active = np.flatnonzero(internal)
    if active.size == 0:
        return 0
    return int(active.max())


def reconstruct_cz_temperature(
    grid: PressureGrid,
    temperature: NDArray[np.float64],
    delta: NDArray[np.float64],
    i_hi: int,
    thermo: ThermoProvider,
) -> NDArray[np.float64]:
    """Anchor T[i_hi] and integrate downward through the CZ.

    For negligible MLT excess, use a constant-entropy (mixture-aware) adiabat.
    Otherwise integrate with temperature-dependent nabla_ad(T) + Delta_nabla.
    """
    t = np.asarray(temperature, dtype=np.float64).copy()
    p = np.asarray(grid.pressure_centres, dtype=np.float64)
    logp = np.log(np.maximum(p, 1.0e-30))
    delta_arr = np.asarray(delta, dtype=np.float64)
    i_hi = int(np.clip(i_hi, 0, t.size - 1))
    internal = delta_arr[1:-1] if delta_arr.size >= 3 else delta_arr
    if internal.size == 0 or float(np.max(np.abs(internal))) <= 1.0e-15:
        t[: i_hi + 1] = _temperature_on_adiabat(
            thermo, float(t[i_hi]), float(p[i_hi]), p[: i_hi + 1]
        )
    else:
        for k in range(i_hi, 0, -1):
            nabla_ad = float(
                np.asarray(thermo.nabla_ad_at(np.array([t[k]], dtype=np.float64))).reshape(-1)[0]
            )
            nabla = nabla_ad + float(delta_arr[k])
            t[k - 1] = t[k] * np.exp(nabla * (logp[k - 1] - logp[k]))
    if np.any(t <= 0.0) or not np.all(np.isfinite(t)):
        return np.asarray(temperature, dtype=np.float64).copy()
    return t


def reconstruct_column_from_rcb(
    grid: PressureGrid,
    t_ref: NDArray[np.float64],
    t_rcb: float,
    i_hi: int,
    delta: NDArray[np.float64],
    thermo: ThermoProvider,
    *,
    t_re: NDArray[np.float64] | None = None,
    rz_blend: float = 0.0,
) -> NDArray[np.float64]:
    """Rebuild the column from T_RCB with a finite-MLT CZ and continuous RZ.

    The CZ is integrated downward from ``t_rcb`` with ∇_ad+Δ∇, never as an
    exact adiabat. The RZ is the live shape scaled so T is continuous at the
    RCB; a grey-RE blend is optional.
    """
    t = np.asarray(t_ref, dtype=np.float64).copy()
    i_hi = int(np.clip(i_hi, 0, t.size - 1))
    t[i_hi] = float(t_rcb)
    t = reconstruct_cz_temperature(grid, t, delta, i_hi, thermo)
    t[i_hi] = float(t_rcb)
    if i_hi < t.size - 1:
        t_old = float(np.asarray(t_ref, dtype=np.float64)[i_hi])
        if t_old > 0.0 and np.isfinite(t_old):
            t[i_hi + 1 :] = float(t_rcb) * (
                np.asarray(t_ref, dtype=np.float64)[i_hi + 1 :] / t_old
            )
        if t_re is not None and float(rz_blend) > 0.0:
            re_anchor = float(np.asarray(t_re, dtype=np.float64)[i_hi])
            if re_anchor > 0.0 and np.isfinite(re_anchor):
                shaped = float(t_rcb) * np.asarray(t_re, dtype=np.float64) / re_anchor
                blend = float(np.clip(rz_blend, 0.0, 1.0))
                t[i_hi + 1 :] = (1.0 - blend) * t[i_hi + 1 :] + blend * shaped[i_hi + 1 :]
        t[i_hi] = float(t_rcb)
    if np.any(t <= 0.0) or not np.all(np.isfinite(t)):
        return np.asarray(t_ref, dtype=np.float64).copy()
    return t


def toa_total_flux(trial: TrialFluxes) -> float:
    return float(np.asarray(trial.flux_total, dtype=np.float64)[-1])


def f_top_defect(trial: TrialFluxes, f_int: float) -> float:
    return toa_total_flux(trial) - float(f_int)


def reduced_config_as_dict(cfg: ReducedRCEConfig) -> dict[str, object]:
    payload = {
        "flux_flatness_tolerance": cfg.flux_flatness_tolerance,
        "tendency_tolerance": cfg.tendency_tolerance,
        "flux_scale_floor": cfg.flux_scale_floor,
        "max_picard": cfg.max_picard,
        "damping_init": cfg.damping_init,
        "damping_min": cfg.damping_min,
        "damping_cut": cfg.damping_cut,
        "max_consecutive_reject": cfg.max_consecutive_reject,
        "logt_shift_max": cfg.logt_shift_max,
        "n_logt_shifts": cfg.n_logt_shifts,
        "radiation_route": cfg.radiation_route.name,
        "diffusivity_factor": cfg.diffusivity_factor,
        "match_rz_to_grey_re": cfg.match_rz_to_grey_re,
        "rz_blend": cfg.rz_blend,
        "coupling": cfg.coupling,
        "max_secant": cfg.max_secant,
        "t_rcb_rel_bracket": cfg.t_rcb_rel_bracket,
        "f_top_tolerance": cfg.f_top_tolerance,
        "max_inner_picard": cfg.max_inner_picard,
        "inner_damping": cfg.inner_damping,
        "inner_delta_atol": cfg.inner_delta_atol,
        "max_rcb_outer": cfg.max_rcb_outer,
        "rz_mode": cfg.rz_mode,
        "max_rz_kappa_picard": cfg.max_rz_kappa_picard,
    }
    return payload


def rz_layer_flux_divergence(flux: NDArray[np.float64], i_hi: int) -> NDArray[np.float64]:
    """F[j+1] − F[j] on radiative layers j = i_hi+1 … N−1."""
    f = np.asarray(flux, dtype=np.float64)
    i_lo = int(i_hi) + 1
    if i_lo >= f.size - 1:
        return np.zeros(0, dtype=np.float64)
    return f[i_lo + 1 :] - f[i_lo : -1]


def cz_mlt_flux_mismatch(
    flux_conv: NDArray[np.float64],
    flux_rad: NDArray[np.float64],
    f_int: float,
    support: NDArray[np.bool_],
) -> float:
    f_req = required_convective_flux(flux_rad, f_int, support)
    interior = _interior_support(support)
    if not np.any(interior):
        return 0.0
    f_c = np.asarray(flux_conv, dtype=np.float64)
    return float(np.max(np.abs(f_c[interior] - f_req[interior])))


def _net_flux_from_source(
    trans: NDArray[np.float64],
    emission_frac: NDArray[np.float64],
    source: NDArray[np.float64],
    f_down_top: float,
    f_int: float,
    f_conv_bottom: float,
) -> NDArray[np.float64]:
    n = int(trans.size)
    fd = _sweep_down(n, trans, emission_frac, source, float(f_down_top))
    f_up0 = float(fd[0]) + float(f_int) - float(f_conv_bottom)
    fu = _sweep_up(n, trans, emission_frac, source, f_up0)
    return fu - fd


def discrete_rz_equilibrium_temperature(
    temperature: NDArray[np.float64],
    i_hi: int,
    opacity: PrescribedOpacity,
    mass_path: NDArray[np.float64],
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    f_int: float,
    *,
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY,
    f_conv_bottom: float = 0.0,
    max_kappa_picard: int = 4,
    t_re_seed: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Solve B_RZ so F_rad,j+1 − F_rad,j = 0 on the Stage-3 two-stream operator.

    CZ temperatures through ``i_hi`` are held fixed. For T-independent opacity
    the map B → F_rad is linear and one solve is exact. A short kappa Picard
    covers T-dependent opacity. Grey-RE may seed kappa; it is not the answer.
    """
    t = np.asarray(temperature, dtype=np.float64).copy()
    n = t.size
    i_hi = int(np.clip(i_hi, 0, n - 1))
    rz = slice(i_hi + 1, n)
    n_rz = n - i_hi - 1
    info: dict[str, object] = {
        "n_rz": n_rz,
        "linear_ok": False,
        "n_kappa_picard": 0,
        "max_B_residual": float("nan"),
    }
    if n_rz <= 0:
        info["linear_ok"] = True
        return t, info
    if t_re_seed is not None:
        seed = np.asarray(t_re_seed, dtype=np.float64)
        if seed.size == n:
            t[rz] = seed[rz]
    f_irr = float(top_bc.flux)
    d_factor = float(diffusivity_factor)
    last_ok = t.copy()
    for k_it in range(1, max(int(max_kappa_picard), 1) + 1):
        info["n_kappa_picard"] = k_it
        kappa = np.asarray(opacity.evaluate(t, pressure), dtype=np.float64)
        if kappa.shape[0] != 1:
            raise ValueError("discrete RZ equilibrium requires a single opacity band")
        dtau = d_factor * kappa[0] * np.asarray(mass_path, dtype=np.float64)
        trans = np.exp(-dtau)
        emission_frac = -np.expm1(-dtau)
        source_cz = np.zeros(n, dtype=np.float64)
        source_cz[: i_hi + 1] = STEFAN_BOLTZMANN * np.maximum(t[: i_hi + 1], 0.0) ** 4
        f0 = _net_flux_from_source(
            trans, emission_frac, source_cz, f_irr, f_int, f_conv_bottom
        )
        r0 = rz_layer_flux_divergence(f0, i_hi)
        f_zero = _net_flux_from_source(
            trans,
            emission_frac,
            np.zeros(n, dtype=np.float64),
            f_irr,
            f_int,
            f_conv_bottom,
        )
        jacobian = np.zeros((n_rz, n_rz), dtype=np.float64)
        e = np.zeros(n, dtype=np.float64)
        for col, layer in enumerate(range(i_hi + 1, n)):
            e[layer] = 1.0
            f_unit = _net_flux_from_source(
                trans, emission_frac, e, f_irr, f_int, f_conv_bottom
            )
            e[layer] = 0.0
            jacobian[:, col] = rz_layer_flux_divergence(f_unit - f_zero, i_hi)
        try:
            b_rz = np.linalg.solve(jacobian, -r0)
        except np.linalg.LinAlgError:
            info["linear_ok"] = False
            return last_ok, info
        if np.any(b_rz <= 0.0) or not np.all(np.isfinite(b_rz)):
            info["linear_ok"] = False
            return last_ok, info
        t_new = t.copy()
        t_new[rz] = (b_rz / STEFAN_BOLTZMANN) ** 0.25
        if np.any(t_new[rz] <= 0.0) or not np.all(np.isfinite(t_new[rz])):
            info["linear_ok"] = False
            return last_ok, info
        rel = float(
            np.max(np.abs(t_new[rz] - t[rz]) / np.maximum(t[rz], 1.0e-30))
        )
        t = t_new
        last_ok = t.copy()
        source = source_cz.copy()
        source[rz] = b_rz
        f_sol = _net_flux_from_source(
            trans, emission_frac, source, f_irr, f_int, f_conv_bottom
        )
        info["max_B_residual"] = float(
            np.max(np.abs(rz_layer_flux_divergence(f_sol, i_hi)))
        )
        info["linear_ok"] = True
        if rel <= 1.0e-12:
            break
    return t, info


def _gates_ok(trial: TrialFluxes, cfg: ReducedRCEConfig) -> bool:
    return (
        trial.flux_flatness <= cfg.flux_flatness_tolerance
        and trial.tendency_norm <= cfg.tendency_tolerance
    )


def _excess(trial: TrialFluxes, grid: PressureGrid, solver: SolverConfig) -> float:
    support = active_interface_mask(grid.n_layers, trial.closure, solver)
    stats = mask_superadiabatic_excess(trial.closure, support, solver)
    val = float(stats["min_superadiabatic_excess_active"])
    if np.isfinite(val):
        return val
    sa = np.asarray(trial.closure.superadiabaticity, dtype=np.float64)[1:-1]
    positive = sa[sa > 0.0]
    return float(np.min(positive)) if positive.size else 0.0


def _regions(trial: TrialFluxes, solver: SolverConfig) -> list[tuple[int, int]]:
    from .rce import _rcb_regions

    return _rcb_regions(trial.closure, solver)


def _interior_support(support: NDArray[np.bool_]) -> NDArray[np.bool_]:
    interior = np.asarray(support, dtype=bool).copy()
    if interior.size:
        interior[0] = False
        interior[-1] = False
    return interior


def _blend_radiative_zone(
    temperature: NDArray[np.float64],
    i_hi: int,
    t_re: NDArray[np.float64],
    blend: float,
) -> NDArray[np.float64]:
    """Keep T continuous at the RCB; blend the RZ toward the grey-RE shape."""
    t = np.asarray(temperature, dtype=np.float64).copy()
    i_hi = int(np.clip(i_hi, 0, t.size - 2))
    t_anchor = float(t[i_hi])
    re_anchor = float(t_re[i_hi])
    if re_anchor <= 0.0 or not np.isfinite(re_anchor):
        return t
    shaped = t_anchor * np.asarray(t_re, dtype=np.float64) / re_anchor
    t[i_hi + 1 :] = (1.0 - blend) * t[i_hi + 1 :] + blend * shaped[i_hi + 1 :]
    t[i_hi] = t_anchor
    return t


def _logt_shift_grid(cfg: ReducedRCEConfig) -> NDArray[np.float64]:
    n = max(int(cfg.n_logt_shifts), 1)
    if n == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(-cfg.logt_shift_max, cfg.logt_shift_max, n)


def _result_from_trial(
    status: ReducedRCEStatus,
    reason: str,
    trial: TrialFluxes,
    grid: PressureGrid,
    solver: SolverConfig,
    cfg: ReducedRCEConfig,
    f_int: float,
    n_picard: int,
    n_evals: int,
    improved: bool,
    history: list[ReducedPicardRecord],
    *,
    t_rcb: float | None = None,
    n_inner_picard: int = 0,
    n_rcb_outer: int = 0,
    inner: InnerPicardDiagnostics | None = None,
) -> ReducedRCEResult:
    support = active_interface_mask(grid.n_layers, trial.closure, solver)
    i_hi = rcb_layer_from_support(support)
    f_top = toa_total_flux(trial)
    theta = float(trial.state.temperature[i_hi]) if t_rcb is None else float(t_rcb)
    diag = inner or InnerPicardDiagnostics()
    return ReducedRCEResult(
        status,
        reason,
        trial.state.temperature.copy(),
        trial,
        trial.flux_flatness,
        trial.tendency_norm,
        worst_gate_violation(trial.flux_flatness, trial.tendency_norm, cfg),
        _regions(trial, solver),
        i_hi,
        _excess(trial, grid, solver),
        n_picard,
        n_evals,
        improved,
        history,
        f_top,
        theta,
        f_top - float(f_int),
        n_inner_picard,
        n_rcb_outer,
        diag.inner_converged,
        diag.delta_abs_mismatch,
        diag.delta_rel_mismatch,
        diag.max_cz_mlt_flux_mismatch,
        diag.rz_max_flux_divergence,
    )


def solve_reduced_radiative_matching(
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
    config: ReducedRCEConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ReducedRCEResult:
    """Radiative-matching accelerator: coupled T_RCB secant, or lagged Picard."""
    cfg = config or ReducedRCEConfig()
    kwargs = dict(
        grid=grid,
        initial_temperature=initial_temperature,
        physics=physics,
        solver=solver,
        thermo=thermo,
        opacity=opacity,
        pressure=pressure,
        top_bc=top_bc,
        lower_bc=lower_bc,
        gravity=gravity,
        config=cfg,
        progress=progress,
    )
    if str(cfg.coupling).lower() == "lagged":
        return solve_lagged_radiative_matching(**kwargs)
    return solve_coupled_radiative_matching(**kwargs)


def solve_coupled_radiative_matching(
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
    config: ReducedRCEConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ReducedRCEResult:
    """Scalar T_RCB match with a discrete Stage-3 RZ and a finite-MLT CZ."""
    cfg = config or ReducedRCEConfig()
    grav = gravity or ConstantGravity(physics.gravity)
    f_int = _internal_flux_reference(lower_bc, None)
    t0 = np.asarray(initial_temperature, dtype=np.float64).copy()
    empty = ReducedRCEResult(
        ReducedRCEStatus.INVALID_BOUNDARY,
        "reduced radiative-matching solve requires LowerNetInternalFlux",
        t0,
        None,
        float("nan"),
        float("nan"),
        float("nan"),
        [],
        0,
        float("nan"),
        0,
        0,
        False,
        [],
    )
    if f_int is None:
        return empty
    f_int = float(f_int)
    f_scale = max(cfg.flux_scale_floor, abs(f_int))
    d_factor = (
        DEFAULT_DIFFUSIVITY if cfg.diffusivity_factor is None else float(cfg.diffusivity_factor)
    )
    history: list[ReducedPicardRecord] = []
    n_evals = 0
    n_secant = 0
    n_inner_total = 0

    def emit(event: dict[str, object]) -> None:
        if progress is not None:
            progress(event)

    def trial_of(
        temperature: NDArray[np.float64],
        frozen_support: NDArray[np.bool_] | None = None,
    ) -> TrialFluxes | None:
        nonlocal n_evals
        state = build_column_state(grid, temperature, thermo, grav)
        h = np.asarray(state.enthalpy, dtype=np.float64)
        n_evals += 1
        return evaluate_trial(
            grid,
            h,
            physics,
            thermo,
            opacity,
            pressure,
            top_bc,
            lower_bc,
            grav,
            f_int=f_int,
            f_scale=f_scale,
            frozen_support=frozen_support,
            diffusivity_factor=d_factor,
            radiation_route=cfg.radiation_route,
        )

    current = trial_of(t0)
    if current is None:
        return ReducedRCEResult(
            ReducedRCEStatus.DOMAIN_FAILURE,
            "initial temperature is outside the thermodynamic domain",
            t0,
            None,
            float("nan"),
            float("nan"),
            float("nan"),
            [],
            0,
            float("nan"),
            0,
            n_evals,
            False,
            history,
        )

    t_ref = current.state.temperature.copy()
    t_re = grey_radiative_equilibrium_temperature(
        grid,
        opacity,
        f_int,
        float(top_bc.flux),
        diffusivity_factor=d_factor,
        pressure=pressure,
        temperature_seed=t_ref,
    )
    start_worst = worst_gate_violation(current.flux_flatness, current.tendency_norm, cfg)
    start_defect = f_top_defect(current, f_int)
    if _gates_ok(current, cfg):
        return _result_from_trial(
            ReducedRCEStatus.CONVERGED,
            "initial state already at the physical gates",
            current,
            grid,
            solver,
            cfg,
            f_int,
            0,
            n_evals,
            False,
            history,
        )

    t_re_use = t_re if cfg.match_rz_to_grey_re else None
    rz_blend = cfg.rz_blend if cfg.match_rz_to_grey_re else 0.0
    rz_mode = str(cfg.rz_mode).lower()
    last_inner = InnerPicardDiagnostics()

    def assemble_column(
        t_rcb: float,
        i_hi: int,
        t_shape: NDArray[np.float64],
        delta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], bool]:
        t_col = np.asarray(t_shape, dtype=np.float64).copy()
        t_col[i_hi] = float(t_rcb)
        t_col = reconstruct_cz_temperature(grid, t_col, delta, i_hi, thermo)
        t_col[i_hi] = float(t_rcb)
        linear_ok = True
        if i_hi >= t_col.size - 1:
            return t_col, linear_ok
        if rz_mode == "discrete":
            t_col, rz_info = discrete_rz_equilibrium_temperature(
                t_col,
                i_hi,
                opacity,
                grid.layer_mass,
                pressure,
                top_bc,
                f_int,
                diffusivity_factor=d_factor,
                f_conv_bottom=0.0,
                max_kappa_picard=cfg.max_rz_kappa_picard,
                t_re_seed=t_re,
            )
            linear_ok = bool(rz_info.get("linear_ok", False))
        else:
            t_old = float(np.asarray(t_shape, dtype=np.float64)[i_hi])
            if t_old > 0.0 and np.isfinite(t_old):
                t_col[i_hi + 1 :] = float(t_rcb) * (
                    np.asarray(t_shape, dtype=np.float64)[i_hi + 1 :] / t_old
                )
            if t_re_use is not None and float(rz_blend) > 0.0:
                t_col = _blend_radiative_zone(t_col, i_hi, t_re_use, rz_blend)
            t_col[i_hi] = float(t_rcb)
        return t_col, linear_ok

    def diagnostics_of(
        trial: TrialFluxes,
        delta: NDArray[np.float64],
        i_hi: int,
        support: NDArray[np.bool_],
        n_inner: int,
        inner_converged: bool,
        linear_ok: bool,
    ) -> InnerPicardDiagnostics:
        interior = _interior_support(support)
        coeff = mlt_flux_coefficient(trial.closure, physics, trial.state.g_edges, thermo)
        f_req = required_convective_flux(trial.flux_rad, f_int, support)
        delta_new = invert_mlt_excess(f_req, coeff)
        if i_hi >= 1 and np.any(interior):
            abs_mis = float(np.max(np.abs(delta_new[interior] - delta[interior])))
            scale = max(float(np.max(np.abs(delta[interior]))), 1.0e-12)
            rel_mis = abs_mis / scale
        else:
            abs_mis = 0.0
            rel_mis = 0.0
        rz_div = rz_layer_flux_divergence(trial.flux_rad, i_hi)
        rz_max = float(np.max(np.abs(rz_div))) if rz_div.size else 0.0
        return InnerPicardDiagnostics(
            n_inner=n_inner,
            inner_converged=inner_converged,
            delta_abs_mismatch=abs_mis,
            delta_rel_mismatch=rel_mis,
            max_cz_mlt_flux_mismatch=cz_mlt_flux_mismatch(
                trial.flux_conv, trial.flux_rad, f_int, support
            ),
            rz_max_flux_divergence=rz_max,
            rz_linear_ok=linear_ok,
        )

    def consistent_column(
        t_rcb: float,
        i_hi: int,
        t_shape: NDArray[np.float64],
        delta0: NDArray[np.float64],
        support: NDArray[np.bool_],
    ) -> tuple[TrialFluxes | None, NDArray[np.float64], InnerPicardDiagnostics]:
        nonlocal n_inner_total, last_inner
        delta = np.asarray(delta0, dtype=np.float64).copy()
        interior = _interior_support(support)
        empty = InnerPicardDiagnostics()
        n_inner = 0
        linear_ok = True
        last: TrialFluxes | None = None
        inner_converged = False
        for j in range(1, cfg.max_inner_picard + 1):
            n_inner = j
            t_col, linear_ok = assemble_column(t_rcb, i_hi, t_shape, delta)
            trial = trial_of(t_col, frozen_support=support)
            if trial is None:
                n_inner_total += n_inner
                last_inner = empty
                return None, delta, empty
            last = trial
            coeff = mlt_flux_coefficient(
                trial.closure, physics, trial.state.g_edges, thermo
            )
            f_req = required_convective_flux(trial.flux_rad, f_int, support)
            delta_new = invert_mlt_excess(f_req, coeff)
            if i_hi < 1 or not np.any(interior):
                inner_converged = True
                break
            if not np.any(delta_new[interior] > 0.0):
                delta_new = np.where(
                    interior, np.maximum(np.asarray(delta0, dtype=np.float64), 0.0), 0.0
                )
            dmax = float(np.max(np.abs(delta_new[interior] - delta[interior])))
            scale = max(float(np.max(np.abs(delta[interior]))), 1.0e-12)
            if dmax <= max(cfg.inner_delta_atol, 1.0e-3 * scale):
                inner_converged = True
                break
            delta = (1.0 - cfg.inner_damping) * delta + cfg.inner_damping * delta_new
        else:
            t_col, linear_ok = assemble_column(t_rcb, i_hi, t_shape, delta)
            last = trial_of(t_col, frozen_support=support)
            n_inner += 1
        n_inner_total += n_inner
        if last is None:
            last_inner = empty
            return None, delta, empty
        diag = diagnostics_of(
            last, delta, i_hi, support, n_inner, inner_converged, linear_ok
        )
        last_inner = diag
        return last, delta, diag

    def record_eval(
        it: int,
        trial: TrialFluxes,
        *,
        t_rcb: float,
        i_hi: int,
        inner: InnerPicardDiagnostics,
        accepted: bool,
        reason: str,
        stage: str,
        theta0: float,
    ) -> ReducedPicardRecord:
        defect = f_top_defect(trial, f_int)
        rec = ReducedPicardRecord(
            picard=it,
            flux_flatness=trial.flux_flatness,
            tendency_norm=trial.tendency_norm,
            residual_two_norm=float(np.linalg.norm(trial.residual)),
            merit=residual_merit(trial.residual),
            worst_gate=worst_gate_violation(trial.flux_flatness, trial.tendency_norm, cfg),
            damping=cfg.inner_damping,
            logt_shift=float(np.log(max(t_rcb, 1.0e-30) / max(theta0, 1.0e-30))),
            rcb_layer=i_hi,
            min_superadiabatic_excess_active=_excess(trial, grid, solver),
            accepted=accepted,
            reason=reason,
            stage=stage,
            t_rcb=float(t_rcb),
            f_top=toa_total_flux(trial),
            f_top_defect=defect,
            inner_picard=inner.n_inner,
            inner_converged=inner.inner_converged,
            inner_delta_abs_mismatch=inner.delta_abs_mismatch,
            inner_delta_rel_mismatch=inner.delta_rel_mismatch,
            max_cz_mlt_flux_mismatch=inner.max_cz_mlt_flux_mismatch,
            rz_max_flux_divergence=inner.rz_max_flux_divergence,
        )
        history.append(rec)
        emit(
            {
                "event": "secant" if stage != "rcb_outer" else "rcb_outer",
                "picard": it,
                "accepted": accepted,
                "reason": reason,
                "stage": stage,
                "flatness": rec.flux_flatness,
                "tendency": rec.tendency_norm,
                "worst_gate": rec.worst_gate,
                "t_rcb": rec.t_rcb,
                "f_top": rec.f_top,
                "f_top_defect": rec.f_top_defect,
                "rcb_layer": i_hi,
                "inner_picard": inner.n_inner,
                "inner_converged": inner.inner_converged,
                "delta_abs": inner.delta_abs_mismatch,
                "cz_mlt_mismatch": inner.max_cz_mlt_flux_mismatch,
                "rz_div": inner.rz_max_flux_divergence,
                "min_excess": rec.min_superadiabatic_excess_active,
                "n_evals": n_evals,
            }
        )
        return rec

    def match_t_rcb(
        i_hi: int,
        t_shape: NDArray[np.float64],
        delta0: NDArray[np.float64],
        support: NDArray[np.bool_],
    ) -> tuple[TrialFluxes | None, float, NDArray[np.float64]]:
        nonlocal n_secant
        theta0 = float(t_shape[i_hi])
        rel = float(cfg.t_rcb_rel_bracket)
        t_min = theta0 * (1.0 - 2.0 * rel)
        t_max = theta0 * (1.0 + 2.0 * rel)
        samples: list[tuple[float, float, TrialFluxes, NDArray[np.float64]]] = []
        best: tuple[float, float, TrialFluxes, NDArray[np.float64]] | None = None

        def evaluate(theta: float, reason: str) -> tuple[float, float, TrialFluxes, NDArray[np.float64]] | None:
            nonlocal n_secant, best
            theta_c = float(np.clip(theta, t_min, t_max))
            trial, delta, inner = consistent_column(
                theta_c, i_hi, t_shape, delta0, support
            )
            n_secant += 1
            if trial is None:
                return None
            live = trial_of(trial.state.temperature) or trial
            defect = f_top_defect(live, f_int)
            sample = (theta_c, defect, live, delta)
            improved_sample = best is None or abs(defect) < abs(best[1])
            if improved_sample:
                best = sample
            record_eval(
                n_secant,
                live,
                t_rcb=theta_c,
                i_hi=i_hi,
                inner=inner,
                accepted=improved_sample,
                reason=reason,
                stage="secant",
                theta0=theta0,
            )
            samples.append(sample)
            return sample

        for fac in (1.0, 1.0 - rel, 1.0 + rel):
            evaluate(theta0 * fac, "bracket")
        if best is not None and abs(best[1]) <= cfg.f_top_tolerance * abs(f_int):
            return best[2], best[0], best[3]

        def opposite_pair():
            ordered = sorted(samples, key=lambda s: s[0])
            for i in range(len(ordered) - 1):
                if ordered[i][1] * ordered[i + 1][1] <= 0.0:
                    return ordered[i], ordered[i + 1]
            return None

        pair = opposite_pair()
        if pair is None:
            for fac in (1.0 - 2.0 * rel, 1.0 + 2.0 * rel):
                evaluate(theta0 * fac, "bracket_expand")
            pair = opposite_pair()
        if pair is None:
            if best is None:
                return None, theta0, delta0
            return best[2], best[0], best[3]

        left, right = pair
        a, fa, trial_a, delta_a = left
        b, fb, trial_b, delta_b = right
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            trial_a, trial_b = trial_b, trial_a
            delta_a, delta_b = delta_b, delta_a
        for _ in range(cfg.max_secant):
            if abs(fb) <= cfg.f_top_tolerance * abs(f_int):
                return trial_b, b, delta_b
            if abs(fb - fa) <= 1.0e-30 * max(abs(f_int), 1.0):
                break
            c = b - fb * (b - a) / (fb - fa)
            lo, hi = (a, b) if a < b else (b, a)
            if not (lo < c < hi):
                c = 0.5 * (a + b)
            if abs(c - b) < 1.0e-8 * max(abs(theta0), 1.0):
                c = 0.5 * (a + b)
            ev = evaluate(c, "secant")
            if ev is None:
                break
            c, fc, trial_c, delta_c = ev
            if fa * fc < 0.0:
                a, fa, trial_a, delta_a = b, fb, trial_b, delta_b
            else:
                fa *= 0.5
            b, fb, trial_b, delta_b = c, fc, trial_c, delta_c
        if best is None:
            return trial_b, b, delta_b
        return best[2], best[0], best[3]

    support = active_interface_mask(grid.n_layers, current.closure, solver)
    i_hi = rcb_layer_from_support(support)
    delta0 = np.asarray(current.closure.superadiabaticity, dtype=np.float64)
    if i_hi < 1 or not np.any(_interior_support(support)):
        return _result_from_trial(
            ReducedRCEStatus.NO_IMPROVEMENT,
            "no convective support for T_RCB matching",
            current,
            grid,
            solver,
            cfg,
            f_int,
            0,
            n_evals,
            False,
            history,
        )

    best_trial = current
    best_theta = float(t_ref[i_hi])
    n_rcb_outer = 0
    for outer in range(1, cfg.max_rcb_outer + 1):
        n_rcb_outer = outer
        matched, theta, delta_m = match_t_rcb(i_hi, t_ref, delta0, support)
        if matched is None:
            break
        live = trial_of(matched.state.temperature) or matched
        best_trial = live
        best_theta = theta
        record_eval(
            n_secant,
            live,
            t_rcb=theta,
            i_hi=i_hi,
            inner=last_inner,
            accepted=True,
            reason="rcb_outer",
            stage="rcb_outer",
            theta0=float(t0[i_hi]) if i_hi < t0.size else theta,
        )
        new_support = active_interface_mask(grid.n_layers, live.closure, solver)
        new_i_hi = rcb_layer_from_support(new_support)
        if new_i_hi == i_hi:
            break
        support = new_support
        i_hi = new_i_hi
        t_ref = live.state.temperature.copy()
        delta0 = np.asarray(live.closure.superadiabaticity, dtype=np.float64)
        if i_hi < 1 or not np.any(_interior_support(support)):
            break

    final_defect = f_top_defect(best_trial, f_int)
    final_worst = worst_gate_violation(best_trial.flux_flatness, best_trial.tendency_norm, cfg)
    improved = abs(final_defect) < abs(start_defect) or final_worst < start_worst
    if _gates_ok(best_trial, cfg):
        status = ReducedRCEStatus.CONVERGED
        reason = "coupled T_RCB matching reached the physical gates"
    elif abs(final_defect) <= cfg.f_top_tolerance * abs(f_int):
        status = ReducedRCEStatus.MATCHED
        reason = "F_top − F_int at tolerance; physical gates not yet met"
    elif improved:
        status = ReducedRCEStatus.SECANT_STALL
        reason = "coupled T_RCB secant stalled short of F_top matching; defect improved"
    else:
        status = ReducedRCEStatus.NO_IMPROVEMENT
        reason = "coupled T_RCB matching did not reduce F_top − F_int"
    return _result_from_trial(
        status,
        reason,
        best_trial,
        grid,
        solver,
        cfg,
        f_int,
        n_secant,
        n_evals,
        improved,
        history,
        t_rcb=best_theta,
        n_inner_picard=n_inner_total,
        n_rcb_outer=n_rcb_outer,
        inner=last_inner,
    )


def solve_lagged_radiative_matching(
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
    config: ReducedRCEConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ReducedRCEResult:
    """Lagged Picard invert-MLT CZ reconstruction plus a small entropy shift."""
    cfg = config or ReducedRCEConfig()
    grav = gravity or ConstantGravity(physics.gravity)
    f_int = _internal_flux_reference(lower_bc, None)
    t0 = np.asarray(initial_temperature, dtype=np.float64).copy()
    empty = ReducedRCEResult(
        ReducedRCEStatus.INVALID_BOUNDARY,
        "reduced radiative-matching solve requires LowerNetInternalFlux",
        t0,
        None,
        float("nan"),
        float("nan"),
        float("nan"),
        [],
        0,
        float("nan"),
        0,
        0,
        False,
        [],
    )
    if f_int is None:
        return empty
    f_int = float(f_int)
    f_scale = max(cfg.flux_scale_floor, abs(f_int))
    d_factor = (
        DEFAULT_DIFFUSIVITY if cfg.diffusivity_factor is None else float(cfg.diffusivity_factor)
    )
    history: list[ReducedPicardRecord] = []
    n_evals = 0

    def emit(event: dict[str, object]) -> None:
        if progress is not None:
            progress(event)

    def trial_of(temperature: NDArray[np.float64]) -> TrialFluxes | None:
        nonlocal n_evals
        state = build_column_state(grid, temperature, thermo, grav)
        h = np.asarray(state.enthalpy, dtype=np.float64)
        n_evals += 1
        return evaluate_trial(
            grid,
            h,
            physics,
            thermo,
            opacity,
            pressure,
            top_bc,
            lower_bc,
            grav,
            f_int=f_int,
            f_scale=f_scale,
            frozen_support=None,
            diffusivity_factor=d_factor,
            radiation_route=cfg.radiation_route,
        )

    current = trial_of(t0)
    if current is None:
        return ReducedRCEResult(
            ReducedRCEStatus.DOMAIN_FAILURE,
            "initial temperature is outside the thermodynamic domain",
            t0,
            None,
            float("nan"),
            float("nan"),
            float("nan"),
            [],
            0,
            float("nan"),
            0,
            n_evals,
            False,
            history,
        )

    t = current.state.temperature.copy()
    t_re = grey_radiative_equilibrium_temperature(
        grid,
        opacity,
        f_int,
        float(top_bc.flux),
        diffusivity_factor=d_factor,
        pressure=pressure,
        temperature_seed=t,
    )
    start_worst = worst_gate_violation(current.flux_flatness, current.tendency_norm, cfg)
    damping = cfg.damping_init
    consecutive_reject = 0

    def _inner_ok(trial: TrialFluxes) -> bool:
        return (
            trial.flux_flatness <= cfg.flux_flatness_tolerance
            and trial.tendency_norm <= cfg.tendency_tolerance
        )

    def _excess(trial: TrialFluxes) -> float:
        support = active_interface_mask(grid.n_layers, trial.closure, solver)
        stats = mask_superadiabatic_excess(trial.closure, support, solver)
        val = float(stats["min_superadiabatic_excess_active"])
        if np.isfinite(val):
            return val
        sa = np.asarray(trial.closure.superadiabaticity, dtype=np.float64)[1:-1]
        positive = sa[sa > 0.0]
        return float(np.min(positive)) if positive.size else 0.0

    def _regions(trial: TrialFluxes) -> list[tuple[int, int]]:
        from .rce import _rcb_regions

        return _rcb_regions(trial.closure, solver)

    if _inner_ok(current):
        support = active_interface_mask(grid.n_layers, current.closure, solver)
        return ReducedRCEResult(
            ReducedRCEStatus.CONVERGED,
            "initial state already at the physical gates",
            current.state.temperature.copy(),
            current,
            current.flux_flatness,
            current.tendency_norm,
            start_worst,
            _regions(current),
            rcb_layer_from_support(support),
            _excess(current),
            0,
            n_evals,
            False,
            history,
        )

    for it in range(1, cfg.max_picard + 1):
        support = active_interface_mask(grid.n_layers, current.closure, solver)
        i_hi = rcb_layer_from_support(support)
        coeff = mlt_flux_coefficient(
            current.closure, physics, current.state.g_edges, thermo
        )
        f_req = required_convective_flux(current.flux_rad, f_int, support)
        delta_req = invert_mlt_excess(f_req, coeff)
        live_delta = np.asarray(current.closure.superadiabaticity, dtype=np.float64)
        interior = np.asarray(support, dtype=bool).copy()
        if interior.size:
            interior[0] = False
            interior[-1] = False
        if i_hi < 1 or not np.any(interior):
            t_cz = t
        else:
            if not np.any(delta_req[interior] > 0.0):
                delta_req = np.where(interior, np.maximum(live_delta, 0.0), 0.0)
            t_cz = reconstruct_cz_temperature(grid, t, delta_req, i_hi, thermo)
        t_prop = (1.0 - damping) * t + damping * t_cz
        if cfg.match_rz_to_grey_re and i_hi < t_prop.size - 1:
            t_prop = _blend_radiative_zone(
                t_prop, i_hi, t_re, cfg.rz_blend * damping
            )

        best_trial = None
        best_score = float("inf")
        best_beta = 0.0
        best_t = t_prop
        for beta in _logt_shift_grid(cfg):
            t_shift = t_prop * np.exp(float(beta))
            trial = trial_of(t_shift)
            if trial is None:
                continue
            score = worst_gate_violation(trial.flux_flatness, trial.tendency_norm, cfg)
            if score < best_score:
                best_score = score
                best_trial = trial
                best_beta = float(beta)
                best_t = t_shift

        cur_score = worst_gate_violation(current.flux_flatness, current.tendency_norm, cfg)
        accepted = best_trial is not None and best_score < cur_score * (1.0 - 1.0e-12)
        reason = "worst_gate_reduced" if accepted else "filter_reject"
        if best_trial is None:
            reason = "domain"
        rec = ReducedPicardRecord(
            picard=it,
            flux_flatness=(best_trial.flux_flatness if best_trial is not None else current.flux_flatness),
            tendency_norm=(best_trial.tendency_norm if best_trial is not None else current.tendency_norm),
            residual_two_norm=(
                float(np.linalg.norm(best_trial.residual))
                if best_trial is not None
                else float(np.linalg.norm(current.residual))
            ),
            merit=(
                residual_merit(best_trial.residual)
                if best_trial is not None
                else residual_merit(current.residual)
            ),
            worst_gate=best_score if best_trial is not None else cur_score,
            damping=damping,
            logt_shift=best_beta,
            rcb_layer=i_hi,
            min_superadiabatic_excess_active=_excess(best_trial or current),
            accepted=accepted,
            reason=reason,
        )
        history.append(rec)
        emit(
            {
                "event": "reduced_picard",
                "picard": it,
                "accepted": accepted,
                "reason": reason,
                "flatness": rec.flux_flatness,
                "tendency": rec.tendency_norm,
                "worst_gate": rec.worst_gate,
                "damping": damping,
                "logt_shift": best_beta,
                "rcb_layer": i_hi,
                "min_excess": rec.min_superadiabatic_excess_active,
                "n_evals": n_evals,
            }
        )
        if accepted and best_trial is not None:
            current = best_trial
            t = best_t
            consecutive_reject = 0
            if _inner_ok(current):
                support = active_interface_mask(grid.n_layers, current.closure, solver)
                return ReducedRCEResult(
                    ReducedRCEStatus.CONVERGED,
                    "reduced radiative-matching iterate reached the physical gates",
                    current.state.temperature.copy(),
                    current,
                    current.flux_flatness,
                    current.tendency_norm,
                    worst_gate_violation(current.flux_flatness, current.tendency_norm, cfg),
                    _regions(current),
                    rcb_layer_from_support(support),
                    _excess(current),
                    it,
                    n_evals,
                    True,
                    history,
                )
            continue
        consecutive_reject += 1
        damping *= cfg.damping_cut
        if damping < cfg.damping_min or consecutive_reject >= cfg.max_consecutive_reject:
            break

    support = active_interface_mask(grid.n_layers, current.closure, solver)
    final_worst = worst_gate_violation(current.flux_flatness, current.tendency_norm, cfg)
    improved = final_worst < start_worst
    if improved:
        status = ReducedRCEStatus.PICARD_STALL
        reason = "reduced Picard stalled short of the physical gates; worst-gate improved"
    else:
        status = ReducedRCEStatus.NO_IMPROVEMENT
        reason = "reduced Picard did not reduce the worst gate violation"
    return ReducedRCEResult(
        status,
        reason,
        current.state.temperature.copy(),
        current,
        current.flux_flatness,
        current.tendency_norm,
        final_worst,
        _regions(current),
        rcb_layer_from_support(support),
        _excess(current),
        len(history),
        n_evals,
        improved,
        history,
    )
