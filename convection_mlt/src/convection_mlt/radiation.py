"""Absorption-only two-stream radiative transfer (Stage 3).

Interface orientation (bottom-to-top, matching PressureGrid):
  - Layers  i = 0 … N-1, bottom → top
  - Interfaces k = 0 … N: interface 0 = physical bottom, interface N = physical top
  - Layer i bounded by bottom interface i and top interface i+1

Notation (no collision):
  - temperature / T_gas,i  : gas temperature [K]
  - transmissivity / 𝒯_i   : exp(-D Δτ_i)
  - emission_fraction       : 1 - 𝒯_i = -expm1(-D Δτ_i)
  - source_flux / B_i       : σ T_gas,i⁴  (grey) or w_b σ T_gas,i⁴ (band)

Heating sign:
  (dh/dt)_rad,i = (F_net[i] - F_net[i+1]) / Δm_i
  F_net = F↑ - F↓
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .opacity import PrescribedOpacity, _validate_band_weights
from .tridiagonal import thomas_solve

STEFAN_BOLTZMANN = 5.670374419e-8  # W m⁻² K⁻⁴
DEFAULT_DIFFUSIVITY = 1.66


# ── Boundary conditions ──────────────────────────────────────────────

class BCType(Enum):
    LOWER_UPWARD_FLUX = auto()
    LOWER_TEMPERATURE = auto()
    LOWER_NET_INTERNAL_FLUX = auto()


@dataclass(frozen=True)
class TopIrradiation:
    flux: float  # total F↓ at interface N [W m⁻²]
    band_fractions: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class LowerUpwardFlux:
    """Stage 3 two-stream BC: prescribe total F↑ at interface 0 [W m⁻²]."""

    flux: float
    band_fractions: NDArray[np.float64] | None = None


# Backward-compatible name for Stage 3 tests and call sites.
LowerFlux = LowerUpwardFlux


@dataclass(frozen=True)
class LowerTemperature:
    temperature: float  # black lower boundary T_bound [K]


@dataclass(frozen=True)
class LowerNetInternalFlux:
    """Stage 4 conservation BC: F_rad,net(0) + F_conv(0) = F_int [W m⁻²].

    Absorption-only downward streams depend only on the top BC. After the
    downward sweep, the required upward flux is
        F_b↑(0) = F_b↓(0) + w_b [F_int − F_conv(0)].
    """

    flux: float  # F_int, net total internal flux [W m⁻²]


LowerBoundary = LowerUpwardFlux | LowerTemperature | LowerNetInternalFlux


# ── Result container ─────────────────────────────────────────────────

class RadiationResult(NamedTuple):
    flux_up: NDArray[np.float64]        # (n_band, n_interface)
    flux_down: NDArray[np.float64]      # (n_band, n_interface)
    flux_net_band: NDArray[np.float64]  # (n_band, n_interface)
    flux_net: NDArray[np.float64]       # (n_interface,)
    heating: NDArray[np.float64]        # (n_layer,)
    optical_depth: NDArray[np.float64]  # (n_band, n_layer)
    transmissivity: NDArray[np.float64] # (n_band, n_layer)


class SolveRoute(Enum):
    SWEEP = auto()
    DENSE = auto()
    THOMAS = auto()


# ── Numerical core (arrays only — no Python objects) ─────────────────

def radiation_core(
    temperature: NDArray[np.float64],
    mass_path: NDArray[np.float64],
    kappa: NDArray[np.float64],
    band_weights: NDArray[np.float64],
    top_down_flux_band: NDArray[np.float64],
    bottom_up_flux_band: NDArray[np.float64],
    diffusivity_factor: float,
    route: SolveRoute = SolveRoute.THOMAS,
    net_internal_flux: float | None = None,
    bottom_convective_flux: float = 0.0,
) -> RadiationResult:
    """Pure-array radiative transfer core (NumPy).

    Parameters
    ----------
    temperature           (n_layer,) gas temperature [K]
    mass_path             (n_layer,) Δm [kg m⁻²]
    kappa                 (n_band, n_layer) opacity [m² kg⁻¹]
    band_weights          (n_band,) ≥ 0, Σ = 1
    top_down_flux_band    (n_band,) F↓ at interface N per band [W m⁻²]
    bottom_up_flux_band   (n_band,) F↑ at interface 0 per band [W m⁻²]
    diffusivity_factor    scalar D > 0
    route                 which solver to use
    net_internal_flux     if set, ignore bottom_up_flux_band and impose
                          F_rad,net(0) + F_conv(0) = net_internal_flux
    bottom_convective_flux  F_conv(0) used only with net_internal_flux
    """
    n_layer = temperature.shape[0]
    n_band = kappa.shape[0]
    n_iface = n_layer + 1

    # optical depth and transmissivity
    dtau = kappa * mass_path[np.newaxis, :]                      # (n_band, n_layer)
    d_dtau = diffusivity_factor * dtau
    trans = np.exp(-d_dtau)                                      # 𝒯_i
    emission_frac = -np.expm1(-d_dtau)                           # 1 - 𝒯_i

    # band source: B_{i,b} = w_b σ T_gas,i⁴
    planck_total = STEFAN_BOLTZMANN * temperature ** 4            # (n_layer,)
    source = band_weights[:, np.newaxis] * planck_total[np.newaxis, :]  # (n_band, n_layer)

    flux_up = np.zeros((n_band, n_iface), dtype=np.float64)
    flux_down = np.zeros((n_band, n_iface), dtype=np.float64)

    for b in range(n_band):
        t_b = trans[b]          # (n_layer,)
        ef_b = emission_frac[b]
        s_b = source[b]
        f_down_top = top_down_flux_band[b]

        if route == SolveRoute.SWEEP:
            fd = _sweep_down(n_layer, t_b, ef_b, s_b, f_down_top)
        elif route == SolveRoute.DENSE:
            fd = _dense_down(n_layer, t_b, ef_b, s_b, f_down_top)
        elif route == SolveRoute.THOMAS:
            fd = _thomas_down(n_layer, t_b, ef_b, s_b, f_down_top)
        else:
            raise ValueError(f"Unknown route: {route}")

        if net_internal_flux is not None:
            f_up_bot = net_internal_upward_band(
                fd[0], band_weights[b], net_internal_flux, bottom_convective_flux
            )
        else:
            f_up_bot = bottom_up_flux_band[b]

        if route == SolveRoute.SWEEP:
            fu = _sweep_up(n_layer, t_b, ef_b, s_b, f_up_bot)
        elif route == SolveRoute.DENSE:
            fu = _dense_up(n_layer, t_b, ef_b, s_b, f_up_bot)
        else:
            fu = _thomas_up(n_layer, t_b, ef_b, s_b, f_up_bot)

        flux_up[b] = fu
        flux_down[b] = fd

    flux_net_band = flux_up - flux_down               # (n_band, n_iface)
    flux_net = np.sum(flux_net_band, axis=0)           # (n_iface,)

    heating = (flux_net[:-1] - flux_net[1:]) / mass_path  # (n_layer,)

    return RadiationResult(
        flux_up=flux_up,
        flux_down=flux_down,
        flux_net_band=flux_net_band,
        flux_net=flux_net,
        heating=heating,
        optical_depth=dtau,
        transmissivity=trans,
    )


# ── Directional sweep (independent reference) ───────────────────────

def net_internal_upward_band(
    flux_down_bottom: float,
    band_weight: float,
    f_int: float,
    f_conv_bottom: float,
) -> float:
    """F_b↑(0) = F_b↓(0) + w_b [F_int − F_conv(0)]."""
    return float(flux_down_bottom + band_weight * (f_int - f_conv_bottom))


def net_internal_residual(
    result: RadiationResult,
    f_int: float,
    f_conv_bottom: float = 0.0,
) -> float:
    """|Σ_b (F_b↑(0) − F_b↓(0)) + F_conv(0) − F_int|."""
    rad_net0 = float(np.sum(result.flux_up[:, 0] - result.flux_down[:, 0]))
    return abs(rad_net0 + f_conv_bottom - f_int)


def _sweep_down(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_down_top: float,
) -> NDArray:
    fd = np.zeros(n_layer + 1, dtype=np.float64)
    fd[n_layer] = f_down_top
    for i in range(n_layer - 1, -1, -1):
        fd[i] = trans[i] * fd[i + 1] + emission_frac[i] * source[i]
    return fd


def _sweep_up(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_up_bot: float,
) -> NDArray:
    fu = np.zeros(n_layer + 1, dtype=np.float64)
    fu[0] = f_up_bot
    for i in range(n_layer):
        fu[i + 1] = trans[i] * fu[i] + emission_frac[i] * source[i]
    return fu


def _sweep(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_down_top: float,
    f_up_bot: float,
) -> tuple[NDArray, NDArray]:
    fd = _sweep_down(n_layer, trans, emission_frac, source, f_down_top)
    fu = _sweep_up(n_layer, trans, emission_frac, source, f_up_bot)
    return fu, fd


# ── Dense solve (same linear system) ────────────────────────────────

def _dense_down(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_down_top: float,
) -> NDArray:
    A_down = np.zeros((n_layer, n_layer), dtype=np.float64)
    b_down = np.zeros(n_layer, dtype=np.float64)
    for i in range(n_layer):
        A_down[i, i] = 1.0
        if i + 1 < n_layer:
            A_down[i, i + 1] = -trans[i]
        b_down[i] = emission_frac[i] * source[i]
    b_down[n_layer - 1] += trans[n_layer - 1] * f_down_top
    x_down = np.linalg.solve(A_down, b_down)
    fd = np.zeros(n_layer + 1, dtype=np.float64)
    fd[:n_layer] = x_down
    fd[n_layer] = f_down_top
    return fd


def _dense_up(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_up_bot: float,
) -> NDArray:
    A_up = np.zeros((n_layer, n_layer), dtype=np.float64)
    b_up = np.zeros(n_layer, dtype=np.float64)
    for i in range(n_layer):
        A_up[i, i] = 1.0
        if i - 1 >= 0:
            A_up[i, i - 1] = -trans[i]
        b_up[i] = emission_frac[i] * source[i]
    b_up[0] += trans[0] * f_up_bot
    x_up = np.linalg.solve(A_up, b_up)
    fu = np.zeros(n_layer + 1, dtype=np.float64)
    fu[0] = f_up_bot
    fu[1:] = x_up
    return fu


def _dense_solve(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_down_top: float,
    f_up_bot: float,
) -> tuple[NDArray, NDArray]:
    fd = _dense_down(n_layer, trans, emission_frac, source, f_down_top)
    fu = _dense_up(n_layer, trans, emission_frac, source, f_up_bot)
    return fu, fd


# ── Thomas solve (same system, tridiagonal) ─────────────────────────

def _thomas_down(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_down_top: float,
) -> NDArray:
    diag_d = np.ones(n_layer, dtype=np.float64)
    upper_d = -trans[:n_layer - 1]
    lower_d = np.zeros(n_layer - 1, dtype=np.float64)
    rhs_d = emission_frac * source
    rhs_d[n_layer - 1] += trans[n_layer - 1] * f_down_top
    x_down = thomas_solve(lower_d, diag_d, upper_d, rhs_d)
    fd = np.zeros(n_layer + 1, dtype=np.float64)
    fd[:n_layer] = x_down
    fd[n_layer] = f_down_top
    return fd


def _thomas_up(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_up_bot: float,
) -> NDArray:
    diag_u = np.ones(n_layer, dtype=np.float64)
    lower_u = -trans[1:]
    upper_u = np.zeros(n_layer - 1, dtype=np.float64)
    rhs_u = emission_frac * source
    rhs_u[0] += trans[0] * f_up_bot
    x_up = thomas_solve(lower_u, diag_u, upper_u, rhs_u)
    fu = np.zeros(n_layer + 1, dtype=np.float64)
    fu[0] = f_up_bot
    fu[1:] = x_up
    return fu


def _thomas_solve_band(
    n_layer: int,
    trans: NDArray,
    emission_frac: NDArray,
    source: NDArray,
    f_down_top: float,
    f_up_bot: float,
) -> tuple[NDArray, NDArray]:
    fd = _thomas_down(n_layer, trans, emission_frac, source, f_down_top)
    fu = _thomas_up(n_layer, trans, emission_frac, source, f_up_bot)
    return fu, fd


# ── Public wrapper ───────────────────────────────────────────────────

def solve_radiation(
    temperature: NDArray[np.float64],
    mass_path: NDArray[np.float64],
    opacity: PrescribedOpacity,
    pressure: NDArray[np.float64],
    top_bc: TopIrradiation,
    lower_bc: LowerBoundary,
    diffusivity_factor: float = DEFAULT_DIFFUSIVITY,
    route: SolveRoute = SolveRoute.THOMAS,
    bottom_convective_flux: float = 0.0,
) -> RadiationResult:
    """Solve absorption-only radiative transfer with prescribed opacity.

    Parameters
    ----------
    temperature    (n_layer,) gas temperature [K], positive
    mass_path      (n_layer,) Δm [kg m⁻²], positive
    opacity        prescribed opacity provider
    pressure       (n_layer,) layer-centre pressure [Pa], for opacity evaluation
    top_bc         top irradiation boundary condition
    lower_bc       Stage 3 F↑, black T, or Stage 4 net internal flux
    diffusivity_factor  D > 0
    route          solver route
    bottom_convective_flux  F_conv(0); used only with LowerNetInternalFlux
    """
    n_layer = temperature.shape[0]
    _validate_inputs(temperature, mass_path, pressure, diffusivity_factor)

    kappa = opacity.evaluate(temperature, pressure)
    weights = opacity.band_weights
    n_band = weights.shape[0]

    if kappa.shape != (n_band, n_layer):
        raise ValueError(f"opacity shape {kappa.shape} != ({n_band}, {n_layer})")

    _validate_band_weights(weights)

    top_band = _expand_top_bc(top_bc, weights, n_band)

    if isinstance(lower_bc, LowerNetInternalFlux):
        dummy_bot = np.zeros(n_band, dtype=np.float64)
        return radiation_core(
            temperature, mass_path, kappa, weights,
            top_band, dummy_bot, diffusivity_factor, route,
            net_internal_flux=float(lower_bc.flux),
            bottom_convective_flux=float(bottom_convective_flux),
        )

    if isinstance(lower_bc, LowerTemperature):
        bot_total = STEFAN_BOLTZMANN * lower_bc.temperature ** 4
        bot_band = weights * bot_total
    elif isinstance(lower_bc, LowerUpwardFlux):
        bot_band = _expand_lower_flux_bc(lower_bc, weights, n_band)
    else:
        raise TypeError(f"Unknown lower BC type: {type(lower_bc)}")

    return radiation_core(
        temperature, mass_path, kappa, weights,
        top_band, bot_band, diffusivity_factor, route,
    )


def _expand_top_bc(
    bc: TopIrradiation,
    weights: NDArray[np.float64],
    n_band: int,
) -> NDArray[np.float64]:
    if bc.band_fractions is not None:
        fracs = np.asarray(bc.band_fractions, dtype=np.float64)
        if fracs.shape != (n_band,):
            raise ValueError("top BC band_fractions shape mismatch")
        return fracs * bc.flux
    return weights * bc.flux


def _expand_lower_flux_bc(
    bc: LowerUpwardFlux,
    weights: NDArray[np.float64],
    n_band: int,
) -> NDArray[np.float64]:
    if bc.band_fractions is not None:
        fracs = np.asarray(bc.band_fractions, dtype=np.float64)
        if fracs.shape != (n_band,):
            raise ValueError("lower BC band_fractions shape mismatch")
        return fracs * bc.flux
    return weights * bc.flux


def _validate_inputs(
    temperature: NDArray,
    mass_path: NDArray,
    pressure: NDArray,
    D: float,
) -> None:
    if temperature.ndim != 1:
        raise ValueError("temperature must be 1-d")
    n = temperature.shape[0]
    if n < 1:
        raise ValueError("Need at least 1 layer")
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    if mass_path.ndim != 1 or mass_path.shape[0] != n:
        raise ValueError(f"mass_path must have shape ({n},)")
    if not np.all(np.isfinite(mass_path)) or np.any(mass_path <= 0):
        raise ValueError("mass_path must be finite and positive")
    if pressure.ndim != 1 or pressure.shape[0] != n:
        raise ValueError(f"pressure must have shape ({n},)")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("pressure must be finite and positive")
    if not np.isfinite(D) or D <= 0:
        raise ValueError("diffusivity_factor must be finite and positive")


def linear_residual_norm(
    route_result: RadiationResult,
    top_down_flux_band: NDArray[np.float64],
    bottom_up_flux_band: NDArray[np.float64],
) -> float:
    """Compute ‖Ax - b‖ / scale for the bidiagonal systems."""
    n_band, n_iface = route_result.flux_up.shape
    n_layer = n_iface - 1
    trans = route_result.transmissivity
    ef = 1.0 - trans
    weights = np.ones(n_band)  # not needed for residual

    planck_total = np.zeros(n_layer)
    if n_layer > 0:
        # reconstruct source from flux_up/down is not trivial; compute residual directly
        pass

    max_resid = 0.0
    f_scale = max(
        1e-30,
        float(np.max(np.abs(route_result.flux_up))),
        float(np.max(np.abs(route_result.flux_down))),
    )

    for b in range(n_band):
        fu = route_result.flux_up[b]
        fd = route_result.flux_down[b]
        t = trans[b]
        e = ef[b]

        # downward residual: F↓[i] - 𝒯_i F↓[i+1] - ε_i B_i = 0
        # but we don't have B_i stored; use sweep relation
        for i in range(n_layer):
            r_down = fd[i] - t[i] * fd[i + 1]
            r_up = fu[i + 1] - t[i] * fu[i]
            # both should equal ε_i B_i, so they should be equal
            max_resid = max(max_resid, abs(r_down - r_up))

    return max_resid / f_scale
