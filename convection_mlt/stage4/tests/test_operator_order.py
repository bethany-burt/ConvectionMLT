"""Operator-order routes and point-39 bracket smoke tests."""

from __future__ import annotations

import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    PhysicsConfig,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)


def _spec(n_layers: int = 16) -> AnalyticOpacityRCESpec:
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=6)


def _solver() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)


def _cfg(**kwargs) -> RCEConfig:
    base = dict(
        max_steps=40,
        n_consec=10**9,
        stall_window=10**9,
        flux_flatness_tolerance=1e-12,
        tendency_tolerance=1e-12,
        temp_change_tolerance=1e-12,
        dt_accuracy=200.0,
        prescribed_dt=50.0,
        t_final=500.0,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10,
            step_tolerance=1e-10,
            newton_residual_tolerance=1e-12,
            newton_step_tolerance=1e-12,
        ),
    )
    base.update(kwargs)
    return RCEConfig(**base)


def _run(route: RCERoute, *, alpha: float = 1.0, **cfg_kw):
    spec = _spec()
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    physics = PhysicsConfig(
        gravity=spec.gravity, alpha=alpha, closure_prefactor=spec.physics().closure_prefactor
    )
    return solve_adaptive_rce(
        grid,
        t0,
        physics,
        _solver(),
        thermo,
        opacity,
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=route,
        config=_cfg(**cfg_kw),
    )


def test_implicit_reverse_and_strang_routes_advance():
    for route in (
        RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        RCERoute.SPLIT_IMPLICIT_CONV_THEN_RAD,
        RCERoute.SPLIT_STRANG_RAD_IMPLICIT_CONV,
    ):
        res = _run(route)
        assert res.steps_accepted > 0, (route, res.status, res.reason)
        assert res.simulated_time >= 0.99 * 500.0
        assert np.all(np.isfinite(res.final_state.temperature))
        assert res.final_state.temperature.min() > 0.0


def test_operator_order_refinement_errors_shrink_with_dt():
    """Fixed-time first-order route: finer accuracy Δt should not blow up profile error."""
    spec = _spec(n_layers=16)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    t_final = 2.0e3
    dt0 = 200.0
    temps = {}
    for factor in (1.0, 0.5, 0.25):
        dt_acc = dt0 * factor
        res = solve_adaptive_rce(
            grid,
            t0,
            spec.physics(),
            _solver(),
            thermo,
            opacity,
            grid.pressure_centres,
            TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            gravity=ConstantGravity(spec.gravity),
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            config=_cfg(
                prescribed_dt=None,
                t_final=t_final,
                max_steps=int(t_final / dt_acc) + 100,
                dt_accuracy=dt_acc,
            ),
        )
        assert res.steps_accepted > 0, res.reason
        assert res.simulated_time >= 0.99 * t_final
        temps[factor] = res.final_state.temperature.copy()
    scale = np.maximum(np.abs(temps[0.25]), 1.0)
    e_coarse = float(np.max(np.abs(temps[1.0] - temps[0.5]) / scale))
    e_fine = float(np.max(np.abs(temps[0.5] - temps[0.25]) / scale))
    assert e_fine <= 1.5 * e_coarse + 1e-12


def test_timestep_refinement_at_fixed_n16_implicit():
    """Fixed-N profiles should not jump when the accuracy dt is halved."""
    spec = _spec(n_layers=16)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    t_final = 2.0e3
    temps = []
    for dt_acc in (200.0, 100.0):
        res = solve_adaptive_rce(
            grid,
            t0,
            spec.physics(),
            _solver(),
            thermo,
            opacity,
            grid.pressure_centres,
            TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            gravity=ConstantGravity(spec.gravity),
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            config=_cfg(
                prescribed_dt=None,
                t_final=t_final,
                max_steps=int(t_final / (0.5 * dt_acc)) + 80,
                dt_accuracy=dt_acc,
            ),
        )
        assert res.simulated_time >= 0.99 * t_final, res.reason
        temps.append(res.final_state.temperature.copy())
    scale = np.maximum(np.abs(temps[1]), 1.0)
    rel = float(np.max(np.abs(temps[0] - temps[1]) / scale))
    assert rel < 0.05


def test_point39_rad_only_and_coupled_semi_implicit_brackets():
    """Smoke: radiation-only and coupled semi-implicit both take stable steps."""
    rad = _run(RCERoute.UNSPLIT, alpha=0.0, prescribed_dt=100.0, t_final=500.0, max_steps=20)
    assert rad.steps_accepted > 0
    assert rad.status != RCETerminalStatus.PRESCRIBED_DT_REJECTED
    coup = _run(
        RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        prescribed_dt=100.0,
        t_final=500.0,
        max_steps=20,
    )
    assert coup.steps_accepted > 0
    assert coup.status != RCETerminalStatus.PRESCRIBED_DT_REJECTED
    assert np.all(np.isfinite(coup.final_state.temperature))
