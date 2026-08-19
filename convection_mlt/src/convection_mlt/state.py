"""Column state for atomic Stage 2 trial updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .gravity import GravityLaw
from .grid import PressureGrid
from .hydrostatics import HydrostaticState, reconstruct_hydrostatic
from .thermodynamics import ThermoProvider


@dataclass(frozen=True)
class ColumnState:
    temperature: NDArray[np.float64]
    enthalpy: NDArray[np.float64]
    density_centres: NDArray[np.float64]
    density_edges: NDArray[np.float64]
    z_centres: NDArray[np.float64]
    z_edges: NDArray[np.float64]
    g_centres: NDArray[np.float64]
    g_edges: NDArray[np.float64]
    mass_path: NDArray[np.float64]
    max_z_over_rp: float

    def copy(self) -> "ColumnState":
        return ColumnState(
            self.temperature.copy(),
            self.enthalpy.copy(),
            self.density_centres.copy(),
            self.density_edges.copy(),
            self.z_centres.copy(),
            self.z_edges.copy(),
            self.g_centres.copy(),
            self.g_edges.copy(),
            self.mass_path.copy(),
            self.max_z_over_rp,
        )


def build_column_state(
    grid: PressureGrid,
    temperature: NDArray[np.float64],
    thermo: ThermoProvider,
    gravity: GravityLaw,
    temperature_edges: NDArray[np.float64] | None = None,
    enthalpy: NDArray[np.float64] | None = None,
) -> ColumnState:
    from .grid import interpolate_temperature_to_internal_edges

    t = np.asarray(temperature, dtype=float).copy()
    # Prefer an explicit enthalpy field (e.g. conserved trial h) over h(T).
    # Recomputing h from inverted T each step accumulates NASA round-trip error.
    if enthalpy is None:
        h = np.asarray(thermo.enthalpy(t), dtype=float)
    else:
        h = np.asarray(enthalpy, dtype=float).copy()
        if h.shape != t.shape:
            raise ValueError("enthalpy must match temperature shape")
        if not np.all(np.isfinite(h)):
            raise ValueError("enthalpy must be finite")
    rho_c = np.asarray(thermo.density(grid.pressure_centres, t), dtype=float)
    t_edge = np.zeros(grid.n_layers + 1, dtype=float)
    if grid.n_layers > 1:
        if temperature_edges is None:
            t_edge[1:-1] = interpolate_temperature_to_internal_edges(grid, t)
        else:
            t_edge[:] = temperature_edges
        # Boundaries: use adjacent centres for density bookkeeping only.
        t_edge[0] = t[0]
        t_edge[-1] = t[-1]
    else:
        t_edge[:] = t[0]
    rho_e = np.asarray(thermo.density(grid.pressure_edges, t_edge), dtype=float)
    hydro = reconstruct_hydrostatic(grid, t, thermo, gravity)
    return ColumnState(
        temperature=t,
        enthalpy=h,
        density_centres=rho_c,
        density_edges=rho_e,
        z_centres=hydro.z_centres,
        z_edges=hydro.z_edges,
        g_centres=hydro.g_centres,
        g_edges=hydro.g_edges,
        mass_path=hydro.mass_path,
        max_z_over_rp=hydro.max_z_over_rp,
    )
