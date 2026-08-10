"""Conservative finite-volume energy operator."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import PressureGrid
from .validate import finite_1d, positive


def temperature_tendency(
    grid: PressureGrid, interface_flux: ArrayLike, cp: float
) -> NDArray[np.float64]:
    """Return dT/dt; flux is upward and edge i is layer i's bottom."""
    flux = finite_1d("interface_flux", interface_flux)
    if flux.size != grid.n_layers + 1:
        raise ValueError(f"interface_flux must have length {grid.n_layers + 1}")
    positive("cp", cp)
    return (flux[:-1] - flux[1:]) / (cp * grid.layer_mass)


def telescoping_residual(
    grid: PressureGrid,
    tendency: ArrayLike,
    cp: float,
    bottom_flux: float,
    top_flux: float,
) -> float:
    values = finite_1d("tendency", tendency)
    if values.size != grid.n_layers:
        raise ValueError(f"tendency must have length {grid.n_layers}")
    positive("cp", cp)
    return float(
        np.sum(cp * grid.layer_mass * values) - (bottom_flux - top_flux)
    )
