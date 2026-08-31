"""HELIOS geometric pressure grid (faithful port of host_functions.calculate_pressure_levels)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .helios_contracts import (
    GRAVITY_CGS,
    GRAVITY_SI,
    MICROBAR_TO_PA,
    PA_TO_MICROBAR,
    helios_gas_boa_microbar,
)


@dataclass(frozen=True)
class HeliosPressureGrid:
    """Bottom-first HELIOS column: interface 0 = BOA, index increases upward."""

    n_layers: int
    p_boa_microbar: float
    p_toa_microbar: float
    p_lay_microbar: NDArray[np.float64]
    p_int_microbar: NDArray[np.float64]
    p_lay_Pa: NDArray[np.float64]
    p_int_Pa: NDArray[np.float64]
    layer_mass_kg_m2: NDArray[np.float64]
    gravity_si: float

    @property
    def n_interfaces(self) -> int:
        return int(self.p_int_microbar.size)

    def tp_read_pressures_microbar(self) -> NDArray[np.float64]:
        """Targets for HELIOS read_temperature_file: [p_int[0]] + p_lay."""
        return np.concatenate([[self.p_int_microbar[0]], self.p_lay_microbar])


def calculate_pressure_levels(
    p_boa_microbar: float,
    p_toa_microbar: float,
    n_layers: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port of HELIOS host_functions.calculate_pressure_levels."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    p_boa = float(p_boa_microbar)
    p_toa = float(p_toa_microbar)
    if p_boa <= 0 or p_toa <= 0 or p_boa <= p_toa:
        raise ValueError("require p_boa > p_toa > 0")
    n = int(n_layers)
    denom = 2 * n - 1
    press_levels = [
        p_boa * (p_toa / p_boa) ** (i / denom) for i in range(2 * n)
    ]
    p_lay = np.asarray([press_levels[i] for i in range(1, 2 * n, 2)], dtype=np.float64)
    p_int = np.asarray([press_levels[i] for i in range(0, 2 * n, 2)], dtype=np.float64)
    p_int = np.append(
        p_int,
        p_toa * (p_toa / p_boa) ** (1.0 / denom),
    )
    return p_lay, p_int


def toa_center_from_top_interface(
    p_top_int_microbar: float,
    p_boa_microbar: float,
    n_layers: int,
) -> float:
    """HELIOS TOA param (upper layer centre) from desired top interface pressure."""
    n = int(n_layers)
    if n < 1:
        raise ValueError("n_layers must be >= 1")
    p_boa = float(p_boa_microbar)
    p_top = float(p_top_int_microbar)
    if p_boa <= 0 or p_top <= 0 or p_top >= p_boa:
        raise ValueError("require 0 < p_top_int < p_boa")
    log_p_toa = (np.log(p_boa) + (2 * n - 1) * np.log(p_top)) / (2 * n)
    return float(np.exp(log_p_toa))


def top_interface_from_toa_center(
    p_toa_microbar: float,
    p_boa_microbar: float,
    n_layers: int,
) -> float:
    """Top interface pressure for a given HELIOS TOA layer-centre parameter."""
    n = int(n_layers)
    p_toa = float(p_toa_microbar)
    p_boa = float(p_boa_microbar)
    return float(np.exp(np.log(p_toa) + np.log(p_toa / p_boa) / (2 * n - 1)))


def build_helios_pressure_grid(
    *,
    p_boa_microbar: float,
    p_toa_microbar: float,
    n_layers: int,
    gravity_si: float = GRAVITY_SI,
) -> HeliosPressureGrid:
    p_lay, p_int = calculate_pressure_levels(p_boa_microbar, p_toa_microbar, n_layers)
    if p_int.size != n_layers + 1:
        raise ValueError(f"expected {n_layers + 1} interfaces, got {p_int.size}")
    if p_lay.size != n_layers:
        raise ValueError(f"expected {n_layers} layer centres, got {p_lay.size}")
    if not np.isclose(p_int[0], p_boa_microbar):
        raise ValueError("p_int[0] must equal p_boa")
    if not np.isclose(p_lay[-1], p_toa_microbar):
        raise ValueError("p_lay[-1] must equal p_toa centre parameter")
    dp_pa = np.abs(np.diff(p_int)) * MICROBAR_TO_PA
    dm = dp_pa / float(gravity_si)
    return HeliosPressureGrid(
        n_layers=int(n_layers),
        p_boa_microbar=float(p_boa_microbar),
        p_toa_microbar=float(p_toa_microbar),
        p_lay_microbar=p_lay,
        p_int_microbar=p_int,
        p_lay_Pa=p_lay * MICROBAR_TO_PA,
        p_int_Pa=p_int * MICROBAR_TO_PA,
        layer_mass_kg_m2=dm,
        gravity_si=float(gravity_si),
    )


def build_helios_grid_from_nested_edges(
    pressure_edges_pa: NDArray[np.float64],
    n_layers: int,
    *,
    gravity_si: float = GRAVITY_SI,
    match_top_interface: bool = True,
) -> HeliosPressureGrid:
    """Build HELIOS grid anchored to nested bottom/top interface pressures."""
    edges = np.asarray(pressure_edges_pa, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("pressure_edges must have at least two values")
    p_boa = float(edges[0] * PA_TO_MICROBAR)
    if match_top_interface:
        p_top_int = float(edges[-1] * PA_TO_MICROBAR)
        p_toa = toa_center_from_top_interface(p_top_int, p_boa, n_layers)
    else:
        # Use nested top layer-centre if provided via midpoint of top cell
        p_toa = float(0.5 * (edges[-2] + edges[-1]) * PA_TO_MICROBAR)
    return build_helios_pressure_grid(
        p_boa_microbar=p_boa,
        p_toa_microbar=p_toa,
        n_layers=n_layers,
        gravity_si=gravity_si,
    )


def interpolate_log_pressure(
    source_p_pa: NDArray[np.float64],
    source_t_k: NDArray[np.float64],
    target_p_pa: NDArray[np.float64],
) -> NDArray[np.float64]:
    """HELIOS-style log10(P) interpolation (read.interpolate_to_own_press)."""
    sp = np.asarray(source_p_pa, dtype=np.float64)
    st = np.asarray(source_t_k, dtype=np.float64)
    tp = np.asarray(target_p_pa, dtype=np.float64)
    order = np.argsort(sp)
    sp = sp[order]
    st = st[order]
    logp = np.log10(np.maximum(sp, 1e-300))
    logt = np.interp(np.log10(np.maximum(tp, 1e-300)), logp, st)
    return np.asarray(logt, dtype=np.float64)


def sample_nested_tp_on_helios_grid(
    nested_record: dict,
    grid: HeliosPressureGrid,
) -> tuple[float, NDArray[np.float64]]:
    """Sample frozen nested T(P) onto HELIOS tp-read targets."""
    p_src = np.asarray(nested_record["pressure_centres"], dtype=np.float64)
    t_src = np.asarray(nested_record["temperature"], dtype=np.float64)
    if "pressure_edges" in nested_record:
        edges = np.asarray(nested_record["pressure_edges"], dtype=np.float64)
        p_all = np.concatenate([[edges[0]], p_src, [edges[-1]]])
        t_edge_bot = float(interpolate_log_pressure(p_src, t_src, np.array([edges[0]]))[0])
        t_edge_top = float(interpolate_log_pressure(p_src, t_src, np.array([edges[-1]]))[0])
        t_all = np.concatenate([[t_edge_bot], t_src, [t_edge_top]])
    else:
        p_all = p_src
        t_all = t_src
    targets_pa = grid.tp_read_pressures_microbar() * MICROBAR_TO_PA
    t_targets = interpolate_log_pressure(p_all, t_all, targets_pa)
    temperature_boa_k = float(t_targets[0])
    temperature_lay_k = np.asarray(t_targets[1:], dtype=np.float64)
    return temperature_boa_k, temperature_lay_k


def layer_optical_depth_cgs(
    kappa_cgs: float | NDArray[np.float64],
    delta_p_microbar: NDArray[np.float64],
    gravity_cgs: float = GRAVITY_CGS,
) -> NDArray[np.float64]:
    """Δτ = κ_cgs ΔP_cgs / g_cgs with ΔP in microbar (= dyn/cm²)."""
    k = np.asarray(kappa_cgs, dtype=np.float64)
    dp = np.asarray(delta_p_microbar, dtype=np.float64)
    return k * dp / float(gravity_cgs)


def layer_optical_depth_si(
    kappa_si: float | NDArray[np.float64],
    delta_p_pa: NDArray[np.float64],
    gravity_si: float = GRAVITY_SI,
) -> NDArray[np.float64]:
    """Δτ = κ_si ΔP_si / g_si."""
    k = np.asarray(kappa_si, dtype=np.float64)
    dp = np.asarray(delta_p_pa, dtype=np.float64)
    return k * dp / float(gravity_si)
