"""Bottom-to-top pressure grid and interface interpolation.

Edges satisfy P_e[0] > ... > P_e[N]. Layer i is bounded by edges i
(bottom) and i+1 (top). MLT gradients are always calculated in log pressure.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validate import positive, pressure_edges, temperatures


@dataclass(frozen=True)
class PressureGrid:
    pressure_edges: NDArray[np.float64]
    pressure_centres: NDArray[np.float64]
    layer_mass: NDArray[np.float64]

    @property
    def n_layers(self) -> int:
        return self.pressure_centres.size


def build_grid(edges: ArrayLike, gravity: float) -> PressureGrid:
    pressure = pressure_edges(edges)
    positive("gravity", gravity)
    centres = np.sqrt(pressure[:-1] * pressure[1:])
    mass = (pressure[:-1] - pressure[1:]) / gravity
    return PressureGrid(pressure.copy(), centres, mass)


def log_pressure_edges(
    p_bottom: float, p_top: float, n_layers: int
) -> NDArray[np.float64]:
    positive("p_bottom", p_bottom)
    positive("p_top", p_top)
    if p_bottom <= p_top:
        raise ValueError("p_bottom must exceed p_top")
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    return np.geomspace(p_bottom, p_top, n_layers + 1)


def interpolate_temperature_to_internal_edges(
    grid: PressureGrid, temperature: ArrayLike
) -> NDArray[np.float64]:
    """Interpolate ln(T) linearly in ln(P) to actual internal edges."""
    t = temperatures(temperature, grid.n_layers)
    if grid.n_layers < 2:
        return np.empty(0, dtype=float)
    log_p = np.log(grid.pressure_centres)
    log_t = np.log(t)
    targets = np.log(grid.pressure_edges[1:-1])
    fraction = (targets - log_p[:-1]) / (log_p[1:] - log_p[:-1])
    return np.exp(log_t[:-1] + fraction * (log_t[1:] - log_t[:-1]))


def hydrostatic_layer_thickness(
    grid: PressureGrid,
    temperature: ArrayLike,
    gas_constant: float,
    gravity: float,
) -> NDArray[np.float64]:
    """Diagnostic thickness; never used to evaluate the MLT gradient."""
    t = temperatures(temperature, grid.n_layers)
    positive("gas_constant", gas_constant)
    positive("gravity", gravity)
    return (
        gas_constant
        * t
        / gravity
        * np.log(grid.pressure_edges[:-1] / grid.pressure_edges[1:])
    )
