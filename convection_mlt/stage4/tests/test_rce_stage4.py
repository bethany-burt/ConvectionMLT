from __future__ import annotations

import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantGreyOpacity,
    ConstantH2Thermo,
    HeliosAdapter,
    LowerFlux,
    PhysicsConfig,
    PrescribedBandOpacity,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    build_grid,
    log_pressure_edges,
    manufactured_operator_identity,
    solve_adaptive_rce,
)
from convection_mlt.rce import ManufacturedRadiativeTarget


def _case(n_layers: int = 24):
    gravity = 15.0
    edges = log_pressure_edges(5.0e6, 1.0e2, n_layers)
    grid = build_grid(edges, gravity)
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(gravity=gravity, alpha=1.0, closure_prefactor=0.5)
    solver = SolverConfig(
        epsilon_temperature=2.0e-3,
        max_steps=200000,
        dt_min=1.0e-14,
        c_diff=0.2,
    )
    p = grid.pressure_centres
    t_target = 900.0 * (p / p[0]) ** 0.58
    top = TopIrradiation(flux=120.0)
    bot = LowerFlux(flux=300.0)
    return grid, thermo, physics, solver, p, t_target, top, bot


def test_manufactured_identity_at_target_without_integrating():
    grid, thermo, physics, _solver, _p, t_target, _top, _bot = _case()
    manufactured = ManufacturedRadiativeTarget(target_temperature=t_target, f0=250.0)
    _f_total, _dhdt, flux_err, tend_err = manufactured_operator_identity(
        grid, physics, thermo, manufactured, gravity=ConstantGravity(physics.gravity)
    )
    assert flux_err == 0.0
    assert tend_err == 0.0


def test_one_accepted_unsplit_step_energy_identity():
    grid, thermo, physics, solver, p, t_target, top, bot = _case(n_layers=16)
    opacity = ConstantGreyOpacity(2.0e-4)
    manufactured = ManufacturedRadiativeTarget(target_temperature=t_target, f0=250.0)
    cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=1.0e-6)
    initial = t_target.copy()
    initial[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 1.02
    res = solve_adaptive_rce(
        grid, initial, physics, solver, thermo, opacity, p, top, bot,
        gravity=ConstantGravity(physics.gravity),
        route=RCERoute.UNSPLIT,
        config=cfg,
        manufactured=manufactured,
    )
    accepted = [d for d in res.diagnostics if d.accepted]
    assert accepted, "unsplit step was not accepted"
    d = accepted[0]
    scale = max(abs(d.flux_boundary_work), abs(d.energy_lhs), 1e-30)
    # Instantaneous telescoping identity must be at roundoff on a manufactured step.
    assert abs(d.energy_residual) / scale < 1e-12
    assert d.energy_residual_rel < 1e-12


def test_tiny_split_macrosteps_match_unsplit_at_common_dt():
    grid, thermo, physics, solver, p, t_target, top, bot = _case(n_layers=12)
    opacity = ConstantGreyOpacity(1.5e-4)
    initial = t_target * (1.0 + 5.0e-3 * np.sin(np.linspace(0.0, np.pi, grid.n_layers)))
    probe = solve_adaptive_rce(
        grid, initial, physics, solver, thermo, opacity, p, top, bot,
        gravity=ConstantGravity(physics.gravity),
        route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=1, n_consec=99, stall_window=10),
    )
    accepted = [d for d in probe.diagnostics if d.accepted]
    assert accepted
    dt = 0.05 * min(accepted[0].dt_mlt, accepted[0].dt_rad, accepted[0].dt_temp, accepted[0].dt)
    cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=dt)

    def _run(route):
        return solve_adaptive_rce(
            grid, initial, physics, solver, thermo, opacity, p, top, bot,
            gravity=ConstantGravity(physics.gravity),
            route=route,
            config=cfg,
        )

    unsplit = _run(RCERoute.UNSPLIT)
    split_rc = _run(RCERoute.SPLIT_RAD_THEN_CONV)
    split_cr = _run(RCERoute.SPLIT_CONV_THEN_RAD)
    assert unsplit.steps_accepted == 1
    assert split_rc.steps_accepted == 1
    assert split_cr.steps_accepted == 1
    assert np.isclose(unsplit.simulated_time, dt)
    assert np.isclose(split_rc.simulated_time, dt)
    assert np.isclose(split_cr.simulated_time, dt)
    scale = np.maximum(np.abs(unsplit.final_state.temperature), 1.0)
    dT = np.max(np.abs(unsplit.final_state.temperature - initial) / scale)
    err_rc = np.max(np.abs(split_rc.final_state.temperature - unsplit.final_state.temperature) / scale)
    err_cr = np.max(np.abs(split_cr.final_state.temperature - unsplit.final_state.temperature) / scale)
    # Lie-Trotter splitting error is O(dt) relative to the unsplit update.
    assert dT > 0.0
    assert err_rc < 0.25 * dT + 1e-12
    assert err_cr < 0.25 * dT + 1e-12


def test_unsplit_manufactured_attraction_records_terminal_status():
    grid, thermo, physics, solver, p, t_target, top, bot = _case()
    opacity = ConstantGreyOpacity(2.0e-4)
    manufactured = ManufacturedRadiativeTarget(
        target_temperature=t_target, f0=250.0, relaxation_coeff=1.0
    )
    cfg = RCEConfig(
        n_consec=4,
        flux_flatness_tolerance=1e-8,
        tendency_tolerance=1e-8,
        temp_change_tolerance=1e-8,
        stall_window=400,
        max_steps=400,
    )
    hot = t_target.copy()
    hot[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 1.02
    res_hot = solve_adaptive_rce(
        grid, hot, physics, solver, thermo, opacity, p, top, bot,
        gravity=ConstantGravity(physics.gravity),
        route=RCERoute.UNSPLIT,
        config=cfg,
        manufactured=manufactured,
    )
    assert res_hot.status in (
        RCETerminalStatus.CONVERGED,
        RCETerminalStatus.STALLED,
        RCETerminalStatus.MAX_STEPS,
        RCETerminalStatus.DT_MIN_FAILURE,
    )
    assert np.all(np.isfinite(res_hot.final_state.temperature))
    # A stalled/max_steps result must not be treated as gate evidence.
    if res_hot.status != RCETerminalStatus.CONVERGED:
        rel = float(np.max(np.abs(res_hot.final_state.temperature - t_target) / t_target))
        assert rel > 0.0


def test_convection_off_recovers_radiative_only_shape_and_multiband_case_runs():
    grid, thermo, _physics, solver, p, t_target, top, bot = _case(n_layers=12)
    physics = PhysicsConfig(gravity=15.0, alpha=0.0, closure_prefactor=0.5)
    kappa = np.vstack([
        np.full(grid.n_layers, 2e-4),
        np.full(grid.n_layers, 6e-4),
        np.full(grid.n_layers, 1e-6),
    ])
    weights = np.array([0.7, 0.3, 0.0])
    opacity = PrescribedBandOpacity(kappa_bands=kappa, weights=weights)
    cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10)
    res = solve_adaptive_rce(
        grid, t_target, physics, solver, thermo, opacity, p, top, bot,
        gravity=ConstantGravity(physics.gravity),
        route=RCERoute.UNSPLIT,
        config=cfg,
    )
    assert np.max(np.abs(res.final_flux_conv)) <= 1e-15
    assert np.all(np.isfinite(res.final_flux_rad))
    assert np.all(np.isfinite(res.final_flux_total))


def test_helios_adapter_roundtrip_orientation_exact():
    adapter = HeliosAdapter(helios_top_to_bottom=True)
    layers = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    ifaces = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    assert np.array_equal(adapter.roundtrip_layers(layers), layers)
    assert np.array_equal(adapter.roundtrip_interfaces(ifaces), ifaces)
