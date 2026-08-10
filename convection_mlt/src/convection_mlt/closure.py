"""Selected R0 MLT closure.

The pressure-coordinate closure follows the equations documented for AGNI
(Nicholls et al. 2025, AGNI development documentation accessed 2026-08-09)
and Lee, Tan & Tsai (2024), with the deliberate R0 choices ell=alpha*H_P,
prefactor 1/2, and no surface limiter. It is not the full AGNI implementation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import PressureGrid, interpolate_temperature_to_internal_edges
from .thermodynamics import IdealH2
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


def mixing_length_flux(
    grid: PressureGrid,
    temperature: ArrayLike,
    gravity: float,
    alpha: float,
    thermo: IdealH2 | None = None,
    prefactor: float = 0.5,
) -> ClosureResult:
    """Return upward-positive MLT quantities on N+1 interfaces."""
    t = temperatures(temperature, grid.n_layers)
    positive("gravity", gravity)
    nonnegative("alpha", alpha)
    nonnegative("prefactor", prefactor)
    gas = thermo or IdealH2()
    n_edges = grid.n_layers + 1

    gradient = np.zeros(n_edges)
    delta = np.zeros(n_edges)
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
        delta[internal] = np.maximum(gradient[internal] - gas.nabla_ad, 0.0)
        t_edge[internal] = interpolate_temperature_to_internal_edges(grid, t)
        rho[internal] = (
            grid.pressure_edges[internal] / (gas.gas_constant * t_edge[internal])
        )
        hp[internal] = gas.gas_constant * t_edge[internal] / gravity
        ell[internal] = alpha * hp[internal]
        velocity[internal] = ell[internal] * np.sqrt(
            gravity / hp[internal] * delta[internal]
        )
        flux[internal] = (
            prefactor
            * rho[internal]
            * gas.cp
            * velocity[internal]
            * t_edge[internal]
            * (ell[internal] / hp[internal])
            * delta[internal]
        )
        kzz[internal] = velocity[internal] * ell[internal]
        applicable[internal] = True

    # Boundaries remain explicit zeros and have mixing_length_applicable=False.
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
    )
