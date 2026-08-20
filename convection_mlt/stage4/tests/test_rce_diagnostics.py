"""Stage 4 RCE diagnostic and timestep-controller contracts."""

from __future__ import annotations

import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantGreyOpacity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    PhysicsConfig,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    build_grid,
    log_pressure_edges,
    solve_adaptive_rce,
)
from convection_mlt.closure import ClosureResult
from convection_mlt.rce import (
    ManufacturedRadiativeTarget,
    _partition_rcb_regions,
    _primary_rcb_log10p,
    _rcb_regions,
)


def _fake_closure(delta: np.ndarray) -> ClosureResult:
    n = delta.size
    z = np.zeros(n)
    return ClosureResult(
        z, delta, z, z, z, z, z, z, z, z, delta > 0.0, np.ones(n, dtype=bool), None
    )


def _solver() -> SolverConfig:
    return SolverConfig(c_active=10.0, epsilon_gradient=1.0e-8)


def test_rcb_regions_omit_stable_singletons():
    delta = np.zeros(7)
    regions = _rcb_regions(_fake_closure(delta), _solver())
    assert regions == []


def test_rcb_detached_zone_is_not_bottom_connected():
    # interfaces 0..6; internal 1..5. Detached activity at interfaces 2 and 3.
    delta = np.array([0.0, 0.0, 5.0e-7, 5.0e-7, 0.0, 0.0, 0.0])
    solver = _solver()
    closure = _fake_closure(delta)
    regions = _rcb_regions(closure, solver)
    active = delta[1:-1] > solver.c_active * solver.epsilon_gradient
    bottom, detached = _partition_rcb_regions(regions, active)
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e2, 6), 15.0)
    assert regions == [(1, 3)]
    assert bottom == []
    assert detached == [(1, 3)]
    assert _primary_rcb_log10p(grid, closure, solver) is None


def test_rcb_bottom_connected_interpolates_at_activity_threshold():
    solver = _solver()
    thr = solver.c_active * solver.epsilon_gradient
    # Last active internal interface is index 2 in the full array.
    delta = np.array([0.0, 10.0 * thr, 10.0 * thr, 0.0, 0.0, 0.0, 0.0])
    closure = _fake_closure(delta)
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e2, 6), 15.0)
    logp = _primary_rcb_log10p(grid, closure, solver)
    assert logp is not None
    p_lo = grid.pressure_edges[2]
    p_hi = grid.pressure_edges[3]
    w = (delta[2] - thr) / (delta[2] - delta[3])
    expected = (1.0 - w) * np.log10(p_lo) + w * np.log10(p_hi)
    assert abs(logp - expected) <= 1e-12
    regions = _rcb_regions(closure, solver)
    assert regions == [(0, 2)]


def _rce_case(n_layers: int = 12):
    g = 15.0
    grid = build_grid(log_pressure_edges(5.0e6, 1.0e2, n_layers), g)
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(gravity=g, alpha=1.0, closure_prefactor=0.5)
    solver = SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)
    p = grid.pressure_centres
    t = 900.0 * (p / p[0]) ** 0.58
    return grid, thermo, physics, solver, p, t


def test_tendency_norm_is_timestep_independent():
    grid, thermo, physics, solver, p, t = _rce_case()
    opacity = ConstantGreyOpacity(2.0e-4)
    manufactured = ManufacturedRadiativeTarget(target_temperature=t, f0=250.0, relaxation_coeff=1.0)
    initial = t.copy()
    initial[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 1.02
    top = TopIrradiation(120.0)
    bot = LowerNetInternalFlux(300.0)

    def _one(dt):
        return solve_adaptive_rce(
            grid, initial, physics, solver, thermo, opacity, p, top, bot,
            gravity=ConstantGravity(physics.gravity),
            route=RCERoute.UNSPLIT,
            config=RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=dt),
            manufactured=manufactured,
        )

    a = _one(1.0e-6)
    b = _one(2.0e-6)
    da = [d for d in a.diagnostics if d.accepted][0]
    db = [d for d in b.diagnostics if d.accepted][0]
    assert np.isclose(da.dt, 1.0e-6)
    assert np.isclose(db.dt, 2.0e-6)
    assert abs(da.tendency_norm - db.tendency_norm) <= 1e-14
    assert da.temp_change > 0.0
    assert db.temp_change > da.temp_change


def test_prescribed_dt_is_strict_and_does_not_backtrack():
    grid, thermo, physics, solver, p, t = _rce_case(n_layers=8)
    opacity = ConstantGreyOpacity(2.0e-4)
    manufactured = ManufacturedRadiativeTarget(target_temperature=t, f0=250.0)
    initial = t.copy()
    initial[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 1.02
    huge_dt = 1.0e12
    res = solve_adaptive_rce(
        grid, initial, physics, solver, thermo, opacity, p,
        TopIrradiation(120.0), LowerNetInternalFlux(300.0),
        gravity=ConstantGravity(physics.gravity),
        route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=4, n_consec=99, stall_window=10, prescribed_dt=huge_dt),
        manufactured=manufactured,
    )
    assert res.status == RCETerminalStatus.PRESCRIBED_DT_REJECTED
    assert res.steps_accepted == 0
    accepted_dts = [d.dt for d in res.diagnostics if d.accepted]
    assert accepted_dts == []
    rejected_dts = [d.dt for d in res.diagnostics if not d.accepted]
    assert rejected_dts
    assert all(np.isclose(dt, huge_dt) for dt in rejected_dts)
