"""Reduced radiative-matching RCE: MLT coefficient identity and small-column behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    ReducedRCEConfig,
    ReducedRCEStatus,
    TopIrradiation,
    discrete_rz_equilibrium_temperature,
    invert_mlt_excess,
    mlt_flux_coefficient,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    reconstruct_column_from_rcb,
    reduced_config_as_dict,
    required_convective_flux,
    rz_layer_flux_divergence,
    solve_lagged_radiative_matching,
    solve_reduced_radiative_matching,
)
from convection_mlt.config import SolverConfig
from convection_mlt.radiation import DEFAULT_DIFFUSIVITY, solve_radiation
from convection_mlt.rce import _evaluate_closure
from convection_mlt.state import build_column_state


def _nested_column(n_layers: int = 24):
    spec = nested_analytic_opacity_spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    t0 = radiative_convective_initial_temperature(
        grid, spec.opacity(), thermo, spec.f_int, spec.f_irr
    )
    return spec, grid, thermo, solver, t0


def test_mlt_coefficient_recovers_three_halves_flux():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    grav = ConstantGravity(spec.gravity)
    state = build_column_state(grid, t0, thermo, grav)
    closure = _evaluate_closure(grid, state, spec.physics(), thermo)
    coeff = mlt_flux_coefficient(closure, spec.physics(), state.g_edges, thermo)
    delta = np.asarray(closure.superadiabaticity, dtype=np.float64)
    reconstructed = coeff * np.maximum(delta, 0.0) ** 1.5
    flux = np.asarray(closure.flux, dtype=np.float64)
    internal = slice(1, -1)
    scale = np.maximum(np.abs(flux[internal]), 1.0e-12)
    rel = np.abs(reconstructed[internal] - flux[internal]) / scale
    assert float(np.max(rel)) < 1.0e-10
    assert float(flux[0]) == 0.0
    assert float(flux[-1]) == 0.0


def test_invert_mlt_excess_roundtrip():
    coeff = np.array([0.0, 10.0, 20.0, 5.0, 0.0])
    flux = np.array([0.0, 8.0, 0.0, 2.0, 0.0])
    delta = invert_mlt_excess(flux, coeff)
    assert delta[0] == 0.0
    assert delta[2] == 0.0
    assert delta[1] == pytest.approx((8.0 / 10.0) ** (2.0 / 3.0))
    recovered = coeff * delta ** 1.5
    assert recovered[1] == pytest.approx(8.0)
    assert recovered[3] == pytest.approx(2.0)


def test_required_flux_does_not_ignite_radiative_zone():
    f_rad = np.array([292.0, 292.0, 292.0, 292.0, 292.0])
    support = np.array([True, True, True, False, False])
    f_req = required_convective_flux(f_rad, 300.0, support)
    assert f_req[0] == 0.0
    assert f_req[-1] == 0.0
    assert f_req[3] == 0.0
    assert f_req[1] == pytest.approx(8.0)
    assert f_req[2] == pytest.approx(8.0)


def test_lagged_solve_keeps_finite_mlt_excess_on_nested_24():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    cfg = ReducedRCEConfig(
        coupling="lagged", max_picard=6, n_logt_shifts=5, logt_shift_max=0.01
    )
    res = solve_lagged_radiative_matching(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        config=cfg,
    )
    assert res.trial is not None
    assert np.all(np.isfinite(res.temperature))
    if res.convective_regions:
        assert res.min_superadiabatic_excess_active > 0.0
    assert res.status in {
        ReducedRCEStatus.CONVERGED,
        ReducedRCEStatus.PICARD_STALL,
        ReducedRCEStatus.NO_IMPROVEMENT,
    }


def test_reconstruct_column_anchors_rcb_and_scales_cz():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    grav = ConstantGravity(spec.gravity)
    state = build_column_state(grid, t0, thermo, grav)
    closure = _evaluate_closure(grid, state, spec.physics(), thermo)
    from convection_mlt.reduced_rce import rcb_layer_from_support
    from convection_mlt.steady_rce import active_interface_mask

    support = active_interface_mask(grid.n_layers, closure, solver)
    i_hi = rcb_layer_from_support(support)
    if i_hi < 1:
        pytest.skip("nested-24 seed has no convective support")
    delta = np.maximum(np.asarray(closure.superadiabaticity, dtype=np.float64), 0.0)
    lam = 1.01
    t_rcb = lam * float(t0[i_hi])
    t_rcb0 = float(t0[i_hi])
    t_base = reconstruct_column_from_rcb(grid, t0, t_rcb0, i_hi, delta, thermo)
    t_new = reconstruct_column_from_rcb(grid, t0, t_rcb, i_hi, delta, thermo)
    assert t_new[i_hi] == pytest.approx(t_rcb)
    assert np.all(t_new[: i_hi + 1] > t_base[: i_hi + 1])
    if i_hi < t0.size - 1:
        assert t_new[i_hi + 1] == pytest.approx(t_rcb * t0[i_hi + 1] / t0[i_hi])


def test_discrete_rz_zeros_layer_divergence_on_nested_24():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    grav = ConstantGravity(spec.gravity)
    state = build_column_state(grid, t0, thermo, grav)
    closure = _evaluate_closure(grid, state, spec.physics(), thermo)
    from convection_mlt.reduced_rce import rcb_layer_from_support
    from convection_mlt.steady_rce import active_interface_mask

    support = active_interface_mask(grid.n_layers, closure, solver)
    i_hi = rcb_layer_from_support(support)
    if i_hi < 1 or i_hi >= grid.n_layers - 2:
        pytest.skip("nested-24 seed has no radiative zone above the RCB")
    t_rz, info = discrete_rz_equilibrium_temperature(
        t0,
        i_hi,
        spec.opacity(),
        grid.layer_mass,
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        spec.f_int,
        diffusivity_factor=DEFAULT_DIFFUSIVITY,
        max_kappa_picard=1,
    )
    assert info["linear_ok"]
    assert np.allclose(t_rz[: i_hi + 1], t0[: i_hi + 1])
    rad = solve_radiation(
        t_rz,
        grid.layer_mass,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        DEFAULT_DIFFUSIVITY,
    )
    div = rz_layer_flux_divergence(rad.flux_net, i_hi)
    assert div.size == grid.n_layers - i_hi - 1
    assert float(np.max(np.abs(div))) < 1.0e-8 * spec.f_int


def test_reduced_config_records_rz_mode_and_blend():
    cfg = ReducedRCEConfig(
        coupling="consistent",
        rz_mode="discrete",
        match_rz_to_grey_re=False,
        rz_blend=0.0,
    )
    payload = reduced_config_as_dict(cfg)
    assert payload["rz_mode"] == "discrete"
    assert payload["match_rz_to_grey_re"] is False
    assert payload["rz_blend"] == 0.0
    assert payload["radiation_route"] == "THOMAS"


def test_coupled_solve_keeps_finite_mlt_excess_on_nested_24():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    cfg = ReducedRCEConfig(
        coupling="consistent",
        rz_mode="discrete",
        max_secant=8,
        max_inner_picard=4,
        max_rcb_outer=2,
        match_rz_to_grey_re=False,
        rz_blend=0.0,
    )
    res = solve_reduced_radiative_matching(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        config=cfg,
    )
    assert res.trial is not None
    assert np.all(np.isfinite(res.temperature))
    if res.convective_regions:
        assert res.min_superadiabatic_excess_active > 0.0
    assert res.status in {
        ReducedRCEStatus.CONVERGED,
        ReducedRCEStatus.MATCHED,
        ReducedRCEStatus.SECANT_STALL,
        ReducedRCEStatus.NO_IMPROVEMENT,
    }
    if res.status != ReducedRCEStatus.CONVERGED:
        assert any(rec.stage == "secant" for rec in res.history)
        assert np.isfinite(res.f_top_defect)
        assert np.isfinite(res.t_rcb)
        assert all(isinstance(rec.inner_converged, bool) for rec in res.history if rec.stage == "secant")
        if res.trial is not None and res.rcb_layer < res.temperature.size - 1:
            div = rz_layer_flux_divergence(res.trial.flux_rad, res.rcb_layer)
            if div.size:
                assert np.isfinite(res.rz_max_flux_divergence)


RESULTS = Path(__file__).resolve().parents[1] / "results"
N192 = RESULTS / "n192_implicit_rce.json"


@pytest.mark.skipif(not N192.exists(), reason="gated N=192 record not stored")
def test_reduced_solve_leaves_gated_n192_near_gate():
    rec = json.loads(N192.read_text())
    spec = nested_analytic_opacity_spec(192)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    t0 = np.asarray(rec["temperature"], dtype=np.float64)
    cfg = ReducedRCEConfig(
        coupling="consistent",
        max_secant=4,
        max_inner_picard=2,
        max_rcb_outer=1,
        rz_mode="discrete",
        match_rz_to_grey_re=False,
    )
    res = solve_reduced_radiative_matching(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        config=cfg,
    )
    assert res.flux_flatness <= 2.0e-3
    assert res.min_superadiabatic_excess_active > 1.0e-8
    scale = np.maximum(np.abs(t0), 1.0)
    assert float(np.max(np.abs(res.temperature - t0) / scale)) < 0.05
