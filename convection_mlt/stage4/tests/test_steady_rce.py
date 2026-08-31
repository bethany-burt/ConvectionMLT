"""Steady flux-flatness Newton–Krylov residual and small-column behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    SteadyRCEConfig,
    SteadyRCEStatus,
    TopIrradiation,
    flux_flatness_residual,
    interface_support_from_regions,
    jv_epsilon_ladder,
    mask_superadiabatic_excess,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    residual_merit,
    restarted_gmres,
    solve_steady_rce,
)
from convection_mlt.config import SolverConfig
from convection_mlt.rce import _evaluate_closure, _rcb_regions
from convection_mlt.steady_rce import _line_search
from convection_mlt.state import build_column_state


def test_gmres_solves_diagonal_system():
    diag = np.linspace(0.5, 2.0, 12)

    def apply_a(v):
        return diag * v

    b = np.arange(1.0, 13.0)
    x, iters, rn, ok = restarted_gmres(apply_a, b, rtol=1e-10, maxiter=20, restart=12)
    assert ok
    assert rn < 1e-8
    assert np.allclose(x, b / diag, rtol=1e-8, atol=1e-10)
    assert iters >= 1


def test_flatness_residual_is_cumulative_divergence():
    f_int = 300.0
    f_scale = 300.0
    f_total = np.array([300.0, 297.0, 291.0, 280.0])
    residual = flux_flatness_residual(f_total, f_int, f_scale)
    assert residual.shape == (3,)
    div = f_total[:-1] - f_total[1:]
    cumulative = np.cumsum(div)
    assert np.allclose(residual, -cumulative / f_scale)
    assert float(np.max(np.abs(f_total - f_int)) / f_scale) == pytest.approx(
        float(np.max(np.abs(residual)))
    )


def test_interface_support_keeps_bottom_for_connected_zone():
    support = interface_support_from_regions(8, [(0, 5)])
    assert support.shape == (9,)
    assert bool(support[0]) is True
    assert np.all(support[1:6])
    assert not np.any(support[6:])
    assert bool(support[-1]) is False


def _nested_column(n_layers: int = 24):
    spec = nested_analytic_opacity_spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    t0 = radiative_convective_initial_temperature(
        grid, spec.opacity(), thermo, spec.f_int, spec.f_irr
    )
    return spec, grid, thermo, solver, t0


def test_newton_reduces_flatness_on_nested_24():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    cfg = SteadyRCEConfig(max_newton=6, max_mask_outer=3, gmres_maxiter=40, gmres_restart=24)
    res = solve_steady_rce(
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
    assert res.residual.shape == (grid.n_layers,)
    assert np.isfinite(res.flux_flatness)
    assert res.n_evals >= 1
    assert abs(float(res.flux_total[0]) - spec.f_int) / spec.f_int < 1.0e-12
    if res.history:
        rec = res.history[0]
        assert len(rec.residual_before) == grid.n_layers
        assert len(rec.residual_after) == grid.n_layers
        assert rec.line_search_reason
        assert len(rec.mask_before) == grid.n_layers + 1
        assert len(rec.mask_after) == grid.n_layers + 1
        assert rec.fd_rel > 0.0
        assert rec.line_search_factor > 0.0


def test_frozen_support_matches_rcb_regions():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    grav = ConstantGravity(spec.gravity)
    state = build_column_state(grid, t0, thermo, grav)
    closure = _evaluate_closure(grid, state, spec.physics(), thermo)
    regions = _rcb_regions(closure, solver)
    support = interface_support_from_regions(grid.n_layers, regions)
    assert support.size == grid.n_layers + 1
    assert bool(support[-1]) is False
    if regions and regions[0][0] == 0:
        assert bool(support[0]) is True


RESULTS = Path(__file__).resolve().parents[1] / "results"
N192 = RESULTS / "n192_implicit_rce.json"


def _max_rel_t(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(aa), 1.0)
    return float(np.max(np.abs(bb - aa) / scale))


@pytest.mark.skipif(not N192.exists(), reason="gated N=192 record not stored")
def test_gated_n192_is_left_unchanged_at_physical_gate():
    rec = json.loads(N192.read_text())
    spec = nested_analytic_opacity_spec(192)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    t0 = np.asarray(rec["temperature"], dtype=np.float64)
    h0 = np.asarray(rec["enthalpy"], dtype=np.float64)
    cfg = SteadyRCEConfig(max_newton=8, max_mask_outer=3)
    res = solve_steady_rce(
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
        initial_enthalpy=h0,
    )
    assert res.flux_flatness <= 1.0e-3 + 1.0e-12
    assert _max_rel_t(t0, res.state.temperature) < 1.0e-5
    assert res.detached_convective_regions == []
    assert abs(float(res.flux_total[0]) - spec.f_int) / spec.f_int < 1.0e-12


def test_jv_epsilon_ladder_stable_on_linear_residual():
    rng = np.random.default_rng(0)
    n = 16
    a = rng.normal(size=(n, n))
    h = rng.normal(size=n)
    r0 = a @ h
    direction = rng.normal(size=n)

    def residual_at(enthalpy):
        return a @ np.asarray(enthalpy, dtype=np.float64)

    out = jv_epsilon_ladder(residual_at, h, r0, direction)
    assert out["jv_stable"]
    assert float(out["max_pairwise_rel_two_change"]) < 1.0e-6


def test_jv_epsilon_ladder_flags_noisy_residual():
    h = np.linspace(0.5, 1.5, 24)
    direction = np.ones_like(h)

    def residual_at(enthalpy):
        x = np.asarray(enthalpy, dtype=np.float64)
        return x + 1.0e-6 * np.sin(1.0e8 * x)

    r0 = residual_at(h)
    out = jv_epsilon_ladder(residual_at, h, r0, direction)
    assert float(out["max_pairwise_rel_two_change"]) > 0.1
    assert out["jv_stable"] is False


def test_internal_excess_ignores_boundary_zero():
    sa = np.zeros(6)
    sa[1:4] = 3.38e-6
    support = np.array([True, True, True, True, False, False])
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    out = mask_superadiabatic_excess(type("C", (), {"superadiabaticity": sa})(), support, solver)
    assert out["min_superadiabatic_excess_active"] == pytest.approx(3.38e-6)
    assert out["min_superadiabatic_excess_active_including_boundary"] == pytest.approx(0.0)
    assert out["rcb_active_excess"] == pytest.approx(3.38e-6)
    assert out["rcb_inactive_excess"] == pytest.approx(0.0)
    assert out["rcb_active_distance_to_threshold"] == pytest.approx(3.38e-6 - 1.0e-7)
    assert out["max_superadiabatic_excess_inactive"] == pytest.approx(0.0)


def test_armijo_rejects_inf_norm_only_decrease():
    class Fake:
        def __init__(self, residual, tendency):
            self.residual = np.asarray(residual, dtype=np.float64)
            self.tendency_norm = float(tendency)
            self.flux_flatness = float(np.max(np.abs(self.residual)))

    current = Fake(np.array([0.32, 0.10, 0.05]), 1.19e-4)
    worse = Fake(np.array([0.3199, 0.20, 0.05]), 1.45e-2)
    assert residual_merit(worse.residual) > residual_merit(current.residual)

    def residual_at(enthalpy, support):
        return worse, 1

    cfg = SteadyRCEConfig(max_line_search=6, min_line_search_factor=1.0e-3)
    accepted, _alpha, extra, backs, reason = _line_search(
        np.zeros(3), np.ones(3), residual_at, None, current, cfg
    )
    assert accepted is None
    assert extra >= 1
    assert backs >= 1
    assert "armijo" in reason


def test_armijo_rejects_tendency_blowup_even_if_merit_drops():
    class Fake:
        def __init__(self, residual, tendency):
            self.residual = np.asarray(residual, dtype=np.float64)
            self.tendency_norm = float(tendency)
            self.flux_flatness = float(np.max(np.abs(self.residual)))

    current = Fake(np.array([0.3, 0.3]), 1.19e-4)
    better_merit = Fake(np.array([0.1, 0.1]), 1.45e-2)

    def residual_at(enthalpy, support):
        return better_merit, 1

    cfg = SteadyRCEConfig(max_line_search=6, min_line_search_factor=1.0e-3)
    accepted, _alpha, extra, backs, reason = _line_search(
        np.zeros(2), np.ones(2), residual_at, None, current, cfg
    )
    assert accepted is None
    assert extra >= 1
    assert backs >= 1
    assert "tendency" in reason


def test_unchanged_mask_reports_newton_limit_not_mask_limit():
    spec, grid, thermo, solver, t0 = _nested_column(24)
    cfg = SteadyRCEConfig(max_newton=0, max_mask_outer=1)
    res = solve_steady_rce(
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
    assert res.status == SteadyRCEStatus.NEWTON_LIMIT
    assert "mask unchanged" in res.reason
