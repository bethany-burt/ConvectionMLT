"""Hydrostatic reconstruction on a fixed pressure grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gravity import ConstantGravity, GravityLaw, InverseSquareGravity
from .grid import PressureGrid
from .thermodynamics import ThermoProvider
from .validate import positive, temperatures


class HydrostaticDomainError(ValueError):
    """Requested pressure drop is unreachable under the gravity law."""


@dataclass(frozen=True)
class HydrostaticState:
    z_edges: NDArray[np.float64]
    z_centres: NDArray[np.float64]
    g_edges: NDArray[np.float64]
    g_centres: NDArray[np.float64]
    mass_path: NDArray[np.float64]
    max_z_over_rp: float


def _integrate_layer_height(
    p_bottom: float,
    p_top: float,
    temperature: float,
    gas_constant: float,
    gravity: GravityLaw,
    z_bottom: float,
) -> float:
    """Integrate dlnP/dz = -g(z)/(R T) from p_bottom to p_top with fixed layer T."""
    positive("temperature", temperature)
    if p_top >= p_bottom:
        raise ValueError("layer pressures must decrease upward")
    r_mix_t = gas_constant * temperature
    if isinstance(gravity, ConstantGravity):
        return r_mix_t / gravity.g0 * np.log(p_bottom / p_top)

    # Inverse-square: analytic isothermal solution.
    # ln(P_b/P_t) = (GM/(R T)) * (1/r_b - 1/r_t) wait:
    # dlnP/dz = -GM/((Rp+z)^2 R T)
    # ∫_{zb}^{zt} dlnP = ln(Pt/Pb) = - (GM/(R T)) ∫ dz/(Rp+z)^2
    # = - (GM/(R T)) [ -1/(Rp+z) ]_{zb}^{zt} = (GM/(R T)) (1/r_t - 1/r_b)? No:
    # ∫ dz/(Rp+z)^2 = -1/(Rp+z)
    # ln(Pt/Pb) = -(GM/(RT)) * (-1/rt + 1/rb) = (GM/(RT))(1/rt - 1/rb)
    # So ln(Pb/Pt) = (GM/(RT))(1/rb - 1/rt)
    # 1/rt = 1/rb - (RT/GM) ln(Pb/Pt)
    if not isinstance(gravity, InverseSquareGravity):
        # Generic numeric fallback for other laws.
        n = 64
        log_p = np.linspace(np.log(p_bottom), np.log(p_top), n + 1)
        z = z_bottom
        for i in range(n):
            p_mid = np.exp(0.5 * (log_p[i] + log_p[i + 1]))
            dlogp = log_p[i + 1] - log_p[i]  # negative
            g = float(gravity.gravity(np.asarray([z]))[0])
            # dlnP/dz = -g/(RT) => dz = dlnP * (-RT/g)
            dz = dlogp * (-r_mix_t / g)
            if not np.isfinite(dz) or dz < 0.0:
                raise HydrostaticDomainError("nonfinite or negative hydrostatic step")
            z = z + dz
        return z - z_bottom

    rb = gravity.planet_radius + z_bottom
    target = 1.0 / rb - (r_mix_t / gravity.gm) * np.log(p_bottom / p_top)
    if target <= 0.0:
        raise HydrostaticDomainError(
            "inverse-square hydrostatic domain exceeded: requested pressure "
            "drop cannot be reached at any finite radius"
        )
    rt = 1.0 / target
    return rt - rb


def reconstruct_hydrostatic(
    grid: PressureGrid,
    temperature: ArrayLike,
    thermo: ThermoProvider,
    gravity: GravityLaw,
) -> HydrostaticState:
    """Build z and g at edges/centres; z=0 at the lower pressure interface."""
    t = temperatures(temperature, grid.n_layers)
    n = grid.n_layers
    z_edges = np.zeros(n + 1, dtype=float)
    for i in range(n):
        dz = _integrate_layer_height(
            float(grid.pressure_edges[i]),
            float(grid.pressure_edges[i + 1]),
            float(t[i]),
            float(thermo.gas_constant),
            gravity,
            float(z_edges[i]),
        )
        if not np.isfinite(dz) or dz < 0.0:
            raise HydrostaticDomainError("invalid layer height increment")
        z_edges[i + 1] = z_edges[i] + dz

    # Centre heights: evaluate z(P) at geometric-mean centres on the piecewise
    # layer path (same T within each layer).
    z_centres = np.empty(n, dtype=float)
    for i in range(n):
        dz = _integrate_layer_height(
            float(grid.pressure_edges[i]),
            float(grid.pressure_centres[i]),
            float(t[i]),
            float(thermo.gas_constant),
            gravity,
            float(z_edges[i]),
        )
        z_centres[i] = z_edges[i] + dz

    g_edges = gravity.gravity(z_edges)
    g_centres = gravity.gravity(z_centres)
    mass_path = mass_path_from_gravity(grid, z_edges, gravity)

    if isinstance(gravity, InverseSquareGravity):
        max_ratio = float(np.max(z_edges) / gravity.planet_radius)
    else:
        max_ratio = 0.0
    return HydrostaticState(
        z_edges=z_edges,
        z_centres=z_centres,
        g_edges=np.asarray(g_edges, dtype=float),
        g_centres=np.asarray(g_centres, dtype=float),
        mass_path=mass_path,
        max_z_over_rp=max_ratio,
    )


def mass_path_from_gravity(
    grid: PressureGrid,
    z_edges: ArrayLike,
    gravity: GravityLaw,
) -> NDArray[np.float64]:
    """Δm_i = ∫_{P_{i+1}}^{P_i} dP / g(P) using edge gravity trapezoid in P."""
    z = np.asarray(z_edges, dtype=float)
    if z.size != grid.n_layers + 1:
        raise ValueError("z_edges length mismatch")
    g_edges = np.asarray(gravity.gravity(z), dtype=float)
    mass = np.empty(grid.n_layers, dtype=float)
    for i in range(grid.n_layers):
        dp = float(grid.pressure_edges[i] - grid.pressure_edges[i + 1])
        # Trapezoidal in P using endpoint g values (exact for constant g).
        mass[i] = dp * 0.5 * (1.0 / g_edges[i] + 1.0 / g_edges[i + 1])
    return mass


def pressure_from_height(
    grid: PressureGrid,
    temperature: ArrayLike,
    thermo: ThermoProvider,
    gravity: GravityLaw,
    z_query: ArrayLike,
    hydro: HydrostaticState | None = None,
) -> NDArray[np.float64]:
    """Invert z→P with the same piecewise-constant-T layer convention."""
    state = hydro or reconstruct_hydrostatic(grid, temperature, thermo, gravity)
    zq = np.asarray(z_query, dtype=float)
    # Interpolate log P in z using edge samples from the reconstruction.
    log_p = np.log(grid.pressure_edges)
    return np.exp(np.interp(zq, state.z_edges, log_p))


def isothermal_constant_g_height(
    p_bottom: float, p_top: float, temperature: float, gas_constant: float, g0: float
) -> float:
    return gas_constant * temperature / g0 * np.log(p_bottom / p_top)


def isothermal_inverse_square_height(
    p_bottom: float,
    p_top: float,
    temperature: float,
    gas_constant: float,
    gravity: InverseSquareGravity,
    z_bottom: float = 0.0,
) -> float:
    return _integrate_layer_height(
        p_bottom, p_top, temperature, gas_constant, gravity, z_bottom
    )
