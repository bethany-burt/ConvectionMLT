"""Selected R0 MLT closure with Stage 2 entropy-based instability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import PressureGrid, interpolate_temperature_to_internal_edges
from .thermodynamics import ConstantH2Thermo, IdealH2, ThermoProvider
from .validate import nonnegative, positive, temperatures


@dataclass(frozen=True)
class ClosureResult:
    gradient: NDArray[np.float64]
    superadiabaticity: NDArray[np.float64]
    temperature_edges: NDArray[np.float64]
    density_edges: NDArray[np.float64]
    scale_height: NDArray[np.float64]
    mixing_length: NDArray[np.float64]
    velocity: NDArray[np.float64]
    flux: NDArray[np.float64]
    kzz: NDArray[np.float64]
    thermal_diffusivity: NDArray[np.float64]
    active: NDArray[np.bool_]
    mixing_length_applicable: NDArray[np.bool_]
    entropy_jump: NDArray[np.float64] | None = None


def _legacy_scalar_thermo(thermo: ThermoProvider | IdealH2 | None) -> bool:
    gas = thermo or IdealH2()
    return isinstance(gas, ConstantH2Thermo) and not hasattr(gas, "_force_entropy_path")


def mixing_length_flux(
    grid: PressureGrid,
    temperature: ArrayLike,
    gravity: float | ArrayLike,
    alpha: float,
    thermo: ThermoProvider | IdealH2 | None = None,
    prefactor: float = 0.5,
    *,
    use_entropy_instability: bool | None = None,
) -> ClosureResult:
    """Return upward-positive MLT quantities on N+1 interfaces.

    ``gravity`` may be a scalar (Stage 0/1) or an edge-length array (Stage 2).
    When ``use_entropy_instability`` is true (default for non-ConstantH2 providers),
    the finite-layer superadiabaticity is ``Δ∇_s = Δs / [cp ln(P_l/P_u)]``.
    """
    t = temperatures(temperature, grid.n_layers)
    nonnegative("alpha", alpha)
    nonnegative("prefactor", prefactor)
    gas = thermo or IdealH2()
    n_edges = grid.n_layers + 1

    g_edges = np.asarray(gravity, dtype=float)
    if g_edges.ndim == 0:
        positive("gravity", float(g_edges))
        g_edges = np.full(n_edges, float(g_edges), dtype=float)
    elif g_edges.shape != (n_edges,):
        raise ValueError(f"gravity must be scalar or length {n_edges}")
    if np.any(~np.isfinite(g_edges)) or np.any(g_edges <= 0.0):
        raise ValueError("gravity edges must be finite and positive")

    entropy_mode = (
        use_entropy_instability
        if use_entropy_instability is not None
        else not isinstance(gas, ConstantH2Thermo)
    )

    gradient = np.zeros(n_edges)
    delta = np.zeros(n_edges)
    entropy_jump = np.zeros(n_edges)
    t_edge = np.zeros(n_edges)
    rho = np.zeros(n_edges)
    hp = np.zeros(n_edges)
    ell = np.zeros(n_edges)
    velocity = np.zeros(n_edges)
    flux = np.zeros(n_edges)
    kzz = np.zeros(n_edges)
    applicable = np.zeros(n_edges, dtype=bool)

    if grid.n_layers > 1:
        internal = slice(1, -1)
        gradient[internal] = (
            np.log(t[:-1]) - np.log(t[1:])
        ) / (
            np.log(grid.pressure_centres[:-1])
            - np.log(grid.pressure_centres[1:])
        )
        t_edge[internal] = interpolate_temperature_to_internal_edges(grid, t)
        rho[internal] = gas.density(
            grid.pressure_edges[internal], t_edge[internal]
        ) if hasattr(gas, "density") else (
            grid.pressure_edges[internal]
            / (gas.gas_constant * t_edge[internal])
        )
        hp[internal] = gas.gas_constant * t_edge[internal] / g_edges[internal]
        ell[internal] = alpha * hp[internal]

        if entropy_mode:
            s = gas.entropy(t, grid.pressure_centres)
            entropy_jump[internal] = s[:-1] - s[1:]
            cp_edge = gas.specific_heat(t_edge[internal])
            log_pressure_ratio = np.log(
                grid.pressure_centres[:-1] / grid.pressure_centres[1:]
            )
            delta[internal] = np.maximum(
                entropy_jump[internal] / (cp_edge * log_pressure_ratio),
                0.0,
            )
            cp_for_flux = cp_edge
        else:
            # Stage 0/1: pointwise constant nabla_ad.
            nabla = float(gas.nabla_ad)
            delta[internal] = np.maximum(gradient[internal] - nabla, 0.0)
            cp_for_flux = np.full(grid.n_layers - 1, float(gas.cp))

        velocity[internal] = ell[internal] * np.sqrt(
            g_edges[internal] / hp[internal] * delta[internal]
        )
        flux[internal] = (
            prefactor
            * rho[internal]
            * cp_for_flux
            * velocity[internal]
            * t_edge[internal]
            * (ell[internal] / hp[internal])
            * delta[internal]
        )
        kzz[internal] = velocity[internal] * ell[internal]
        applicable[internal] = True

    kh = prefactor * kzz
    return ClosureResult(
        gradient,
        delta,
        t_edge,
        rho,
        hp,
        ell,
        velocity,
        flux,
        kzz,
        kh,
        delta > 0.0,
        applicable,
        entropy_jump if entropy_mode else None,
    )
