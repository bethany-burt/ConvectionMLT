"""Phase 4 standalone grey RCE: discrete RE seed, α=0, then coupled."""

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
    grey_radiative_equilibrium_temperature,
    log_pressure_edges,
    manufactured_operator_identity,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.rce import ManufacturedRadiativeTarget, _run_unsplit
from convection_mlt.state import build_column_state


F_INT = 300.0
F_IRR = 120.0
KAPPA = 0.05
G = 15.0
P_BOTTOM = 1.0e6
P_TOP = 1.0e3


def _column(n_layers: int = 24):
    grid = build_grid(log_pressure_edges(P_BOTTOM, P_TOP, n_layers), G)
    thermo = ConstantH2Thermo()
    opacity = ConstantGreyOpacity(KAPPA)
    top = TopIrradiation(F_IRR)
    bot = LowerNetInternalFlux(F_INT)
    solver = SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)
    return grid, thermo, opacity, top, bot, solver


def test_grey_re_guess_matches_net_internal_to_roundoff():
    grid, thermo, opacity, top, bot, _solver = _column()
    t = grey_radiative_equilibrium_temperature(grid, opacity, F_INT, F_IRR)
    assert float(t.min()) > 200.0
    assert float(t.max()) < 5000.0
    state = build_column_state(grid, t, thermo, ConstantGravity(G))
    physics = PhysicsConfig(gravity=G, alpha=0.0, closure_prefactor=0.5)
    _c, rad, f_conv, f_rad, f_total = _run_unsplit(
        grid, state, physics, thermo, opacity, grid.pressure_centres,
        top, bot, RCEConfig(), None, ConstantGravity(G),
    )
    scale = max(abs(F_INT), float(np.max(np.abs(rad.flux_down))))
    assert np.max(np.abs(f_total - F_INT)) <= 64.0 * np.finfo(np.float64).eps * scale
    assert np.max(np.abs(f_conv)) == 0.0
    assert abs(float(f_rad[0] - F_INT)) <= 64.0 * np.finfo(np.float64).eps * scale


def test_rc_initial_does_not_create_a_2k_top_or_billion_watt_flux():
    grid, thermo, opacity, _top, _bot, _solver = _column()
    t = radiative_convective_initial_temperature(grid, opacity, thermo, F_INT, F_IRR)
    assert float(t.min()) > 200.0
    physics = PhysicsConfig(gravity=G, alpha=1.0, closure_prefactor=0.5)
    state = build_column_state(grid, t, thermo, ConstantGravity(G))
    from convection_mlt.rce import _evaluate_closure
    closure = _evaluate_closure(grid, state, physics, thermo)
    assert float(np.max(closure.flux)) < 1.0e8


def test_radiation_only_grey_re_converges_without_stepping():
    grid, thermo, opacity, top, bot, solver = _column()
    t = grey_radiative_equilibrium_temperature(grid, opacity, F_INT, F_IRR)
    physics = PhysicsConfig(gravity=G, alpha=0.0, closure_prefactor=0.5)
    res = solve_adaptive_rce(
        grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
        gravity=ConstantGravity(G), route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=8, n_consec=3, stall_window=20),
    )
    assert res.status == RCETerminalStatus.CONVERGED
    assert res.steps_accepted == 0
    assert res.convergence.flux_flatness < 1e-12
    assert res.convergence.tendency_norm < 1e-12
    assert np.max(np.abs(res.final_flux_conv)) == 0.0
    assert np.max(np.abs(res.final_flux_total - F_INT)) / F_INT < 1e-12


def test_alpha0_re_converges_at_two_resolutions():
    thermo = ConstantH2Thermo()
    opacity = ConstantGreyOpacity(KAPPA)
    physics = PhysicsConfig(gravity=G, alpha=0.0, closure_prefactor=0.5)
    solver = SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)
    flats = []
    for n in (16, 32):
        grid = build_grid(log_pressure_edges(P_BOTTOM, P_TOP, n), G)
        t = grey_radiative_equilibrium_temperature(grid, opacity, F_INT, F_IRR)
        res = solve_adaptive_rce(
            grid, t, physics, solver, thermo, opacity, grid.pressure_centres,
            TopIrradiation(F_IRR), LowerNetInternalFlux(F_INT),
            gravity=ConstantGravity(G), route=RCERoute.UNSPLIT,
            config=RCEConfig(max_steps=5, n_consec=2),
        )
        assert res.status == RCETerminalStatus.CONVERGED
        flats.append(res.convergence.flux_flatness)
    assert max(flats) < 1e-12


def test_coupled_from_rc_guess_stays_in_the_fint_basin():
    grid, thermo, opacity, top, bot, solver = _column()
    t = radiative_convective_initial_temperature(grid, opacity, thermo, F_INT, F_IRR)
    physics = PhysicsConfig(gravity=G, alpha=1.0, closure_prefactor=0.5)
    res = solve_adaptive_rce(
        grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
        gravity=ConstantGravity(G), route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=80, n_consec=8, stall_window=200,
                         flux_flatness_tolerance=1e-3, tendency_tolerance=1e-3,
                         temp_change_tolerance=1e-3),
    )
    assert res.steps_accepted >= 1
    assert abs(float(res.final_flux_total[0]) - F_INT) <= 1e-8 * F_INT
    residual = float(np.max(np.abs(res.final_flux_total - F_INT)) / F_INT)
    assert residual < 1.0
    assert float(np.max(np.abs(res.final_flux_conv))) < 1.0e6
    assert float(res.final_state.temperature.min()) > 200.0


def test_frozen_radiation_manufactured_attracts_hot_and_cold():
    grid, thermo, opacity, top, bot, solver = _column()
    t_star = grey_radiative_equilibrium_temperature(grid, opacity, F_INT, F_IRR)
    physics = PhysicsConfig(gravity=G, alpha=1.0, closure_prefactor=0.5)
    manufactured = ManufacturedRadiativeTarget(
        target_temperature=t_star, f0=F_INT, relaxation_coeff=1.0
    )
    _f, _d, flux_err, tend_err = manufactured_operator_identity(
        grid, physics, thermo, manufactured, gravity=ConstantGravity(G)
    )
    assert flux_err == 0.0
    assert tend_err == 0.0
    cfg = RCEConfig(
        max_steps=80, n_consec=4, stall_window=200,
        flux_flatness_tolerance=1.0,
        tendency_tolerance=1e-6,
        temp_change_tolerance=1e-6,
    )
    rels = []
    for factor in (1.02, 0.98):
        initial = t_star.copy()
        initial[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= factor
        res = solve_adaptive_rce(
            grid, initial, physics, solver, thermo, opacity, grid.pressure_centres,
            top, bot, gravity=ConstantGravity(G), route=RCERoute.UNSPLIT,
            config=cfg, manufactured=manufactured,
        )
        assert res.status == RCETerminalStatus.CONVERGED
        rel = float(np.max(np.abs(res.final_state.temperature - t_star) / t_star))
        rels.append(rel)
        assert rel < 1e-8
    assert abs(rels[0] - rels[1]) < 1e-8


def test_unsplit_and_splits_agree_from_rc_guess_at_common_dt():
    grid, thermo, opacity, top, bot, solver = _column(n_layers=16)
    t = radiative_convective_initial_temperature(grid, opacity, thermo, F_INT, F_IRR)
    physics = PhysicsConfig(gravity=G, alpha=1.0, closure_prefactor=0.5)
    probe = solve_adaptive_rce(
        grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
        gravity=ConstantGravity(G), route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=1, n_consec=99, stall_window=10),
    )
    accepted = [d for d in probe.diagnostics if d.accepted]
    assert accepted
    dt = 0.05 * min(accepted[0].dt_mlt, accepted[0].dt_rad, accepted[0].dt_temp, accepted[0].dt)
    cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=dt)

    def _run(route):
        return solve_adaptive_rce(
            grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
            gravity=ConstantGravity(G), route=route, config=cfg,
        )

    unsplit = _run(RCERoute.UNSPLIT)
    rc = _run(RCERoute.SPLIT_RAD_THEN_CONV)
    cr = _run(RCERoute.SPLIT_CONV_THEN_RAD)
    assert unsplit.steps_accepted == rc.steps_accepted == cr.steps_accepted == 1
    scale = np.maximum(np.abs(unsplit.final_state.temperature), 1.0)
    dT = np.max(np.abs(unsplit.final_state.temperature - t) / scale)
    err_rc = np.max(np.abs(rc.final_state.temperature - unsplit.final_state.temperature) / scale)
    err_cr = np.max(np.abs(cr.final_state.temperature - unsplit.final_state.temperature) / scale)
    assert dT > 0.0
    assert err_rc < 0.25 * dT + 1e-12
    assert err_cr < 0.25 * dT + 1e-12
