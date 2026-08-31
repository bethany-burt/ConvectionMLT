"""HELIOS pressure grid helper tests."""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.adapters.helios_contracts import GRAVITY_CGS, GRAVITY_SI, MICROBAR_TO_PA
from convection_mlt.adapters.helios_grid import (
    build_helios_grid_from_nested_edges,
    build_helios_pressure_grid,
    calculate_pressure_levels,
    layer_optical_depth_cgs,
    layer_optical_depth_si,
    sample_nested_tp_on_helios_grid,
    toa_center_from_top_interface,
    top_interface_from_toa_center,
)


@pytest.mark.parametrize("n", (8, 96))
def test_helios_grid_invariants(n: int):
    p_boa = 1.0e9
    p_toa = 10.0
    grid = build_helios_pressure_grid(p_boa_microbar=p_boa, p_toa_microbar=p_toa, n_layers=n)
    assert grid.p_lay_microbar.size == n
    assert grid.p_int_microbar.size == n + 1
    assert np.isclose(grid.p_int_microbar[0], p_boa)
    assert np.isclose(grid.p_lay_microbar[-1], p_toa)
    assert np.all(grid.p_int_microbar[:-1] > grid.p_int_microbar[1:])
    assert np.all(grid.p_lay_microbar[:-1] > grid.p_lay_microbar[1:])


def test_toa_center_roundtrip():
    n = 96
    p_boa = 1.0e9
    p_top_int = 5.0
    p_toa = toa_center_from_top_interface(p_top_int, p_boa, n)
    rebuilt = top_interface_from_toa_center(p_toa, p_boa, n)
    assert np.isclose(rebuilt, p_top_int, rtol=1e-12)


def test_delta_tau_si_cgs_agree():
    n = 8
    grid = build_helios_pressure_grid(p_boa_microbar=1e9, p_toa_microbar=10.0, n_layers=n)
    k_si = 0.01
    k_cgs = k_si * 10.0
    dp_pa = np.abs(np.diff(grid.p_int_microbar)) * MICROBAR_TO_PA
    dp_micro = np.abs(np.diff(grid.p_int_microbar))
    dt_si = layer_optical_depth_si(k_si, dp_pa)
    dt_cgs = layer_optical_depth_cgs(k_cgs, dp_micro)
    assert np.allclose(dt_si, dt_cgs, rtol=1e-12)


def test_sample_nested_tp_on_helios_grid():
    nested = {
        "pressure_centres": np.geomspace(1e5, 1e2, 96).tolist(),
        "pressure_edges": np.geomspace(1e6, 1e1, 97).tolist(),
        "temperature": np.linspace(700.0, 200.0, 96).tolist(),
    }
    edges = np.asarray(nested["pressure_edges"])
    grid = build_helios_grid_from_nested_edges(edges, 96)
    t_boa, t_lay = sample_nested_tp_on_helios_grid(nested, grid)
    assert t_boa > 500.0
    assert t_lay.size == 96
    assert t_lay[0] > t_lay[-1]
