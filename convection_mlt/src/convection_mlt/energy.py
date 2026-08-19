"""Conservative finite-volume energy operator."""

from __future__ import annotations

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


def enthalpy_tendency(
    grid: PressureGrid,
    interface_flux: ArrayLike,
    mass_path: ArrayLike,
) -> NDArray[np.float64]:
    """Return dh/dt using accepted mass-path weights Δm^n."""
    flux = finite_1d("interface_flux", interface_flux)
    mass = finite_1d("mass_path", mass_path)
    if flux.size != grid.n_layers + 1:
        raise ValueError(f"interface_flux must have length {grid.n_layers + 1}")
    if mass.size != grid.n_layers:
        raise ValueError(f"mass_path must have length {grid.n_layers}")
    if np.any(mass <= 0.0):
        raise ValueError("mass_path must be positive")
    return (flux[:-1] - flux[1:]) / mass


def apply_enthalpy_step(
    enthalpy: ArrayLike,
    tendency: ArrayLike,
    dt: float,
) -> NDArray[np.float64]:
    h = finite_1d("enthalpy", enthalpy)
    dhdt = finite_1d("tendency", tendency)
    if h.size != dhdt.size:
        raise ValueError("enthalpy and tendency length mismatch")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return h + dt * dhdt


def flux_divergence_identity_residual(
    mass_path: ArrayLike,
    enthalpy_old: ArrayLike,
    enthalpy_new: ArrayLike,
    dt: float,
    bottom_flux: float,
    top_flux: float,
) -> float:
    """Σ Δm^n (h* - h^n) - dt (F_bot - F_top); zero for closed boundaries."""
    mass = finite_1d("mass_path", mass_path)
    h0 = finite_1d("enthalpy_old", enthalpy_old)
    h1 = finite_1d("enthalpy_new", enthalpy_new)
    return float(np.sum(mass * (h1 - h0)) - dt * (bottom_flux - top_flux))


def column_enthalpy_per_area(
    mass_path: ArrayLike, enthalpy: ArrayLike
) -> float:
    """Column enthalpy per unit area H = Σ Δm_i h_i [J m^-2]."""
    mass = finite_1d("mass_path", mass_path)
    h = finite_1d("enthalpy", enthalpy)
    if mass.size != h.size:
        raise ValueError("mass_path and enthalpy length mismatch")
    return float(np.sum(mass * h))
