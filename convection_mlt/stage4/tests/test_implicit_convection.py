"""Implicit convection substep: conservation, attraction, Jacobian, parity."""

from __future__ import annotations

import pytest
import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantGreyOpacity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    ManufacturedRadiativeTarget,
    PhysicsConfig,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    assemble_dense_jacobian,
    assemble_tridiagonal_jacobian,
    build_grid,
    column_enthalpy_per_area,
    log_pressure_edges,
    provisional_support,
    solve_adaptive_rce,
    solve_implicit_convection,
)
from convection_mlt.energy import enthalpy_tendency
from convection_mlt.implicit_convection import (
    _residual,
    evaluate_mlt,
    flux_with_provisional_support,
)
from convection_mlt.rce import _evaluate_closure, _run_unsplit
from convection_mlt.state import build_column_state


G = 15.0
F0 = 250.0


def _column(n_layers: int = 12):
    grid = build_grid(log_pressure_edges(1.0e6, 1.0e2, n_layers), G)
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(gravity=G, alpha=1.0, closure_prefactor=0.5)
    solver = SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)
    p = grid.pressure_centres
    t = 900.0 * (p / p[0]) ** 0.5
    # Ensure a bottom-connected convective region for nontrivial MLT.
    t = t.copy()
    t[: n_layers // 3] *= 1.08
    return grid, thermo, physics, solver, t


def test_tridiagonal_jacobian_matches_dense_on_small_n():
    grid, thermo, physics, solver, t = _column(n_layers=6)
    grav = ConstantGravity(G)
    state = build_column_state(grid, t, thermo, grav)
    h_star = state.enthalpy.copy()
    mass = state.mass_path.copy()
    dt = 0.5
    closure = evaluate_mlt(grid, state, physics, thermo)
    support = provisional_support(closure)
    f = flux_with_provisional_support(closure, support)
    residual0 = _residual(state.enthalpy, h_star, f, mass, dt)
    cfg = ImplicitConvectionConfig()
    lower, diag, upper, _ = assemble_tridiagonal_jacobian(
        grid, state, h_star, support, physics, thermo, grav, mass, dt, cfg, residual0
    )
    dense = assemble_dense_jacobian(
        grid, state, h_star, support, physics, thermo, grav, mass, dt, cfg, residual0
    )
    recon = np.diag(diag)
    for i in range(grid.n_layers - 1):
        recon[i + 1, i] = lower[i]
        recon[i, i + 1] = upper[i]
    # Off-tridiagonal bands of the dense matrix must be ~0; bands must match.
    for i in range(grid.n_layers):
        for j in range(grid.n_layers):
            if abs(i - j) > 1:
                assert abs(dense[i, j]) <= 1e-8 * max(1.0, abs(dense[i, i]))
    assert np.allclose(recon, dense, rtol=1e-5, atol=1e-8)


def test_isoenthalpic_unforced_convection_conserves_column_enthalpy():
    grid, thermo, physics, solver, t = _column(n_layers=16)
    grav = ConstantGravity(G)
    state0 = build_column_state(grid, t, thermo, grav)
    H0 = column_enthalpy_per_area(state0.mass_path, state0.enthalpy)
    # Isoenthalpic redistribution: move enthalpy between layers, keep Σ Δm h.
    h = state0.enthalpy.copy()
    mass = state0.mass_path.copy()
    bump = 0.02 * h[0]
    h[0] += bump
    # Compensate in upper layers by mass-weighted debit.
    debit = bump * mass[0]
    for i in range(1, grid.n_layers):
        share = mass[i] / np.sum(mass[1:])
        h[i] -= debit * share / mass[i]
    assert abs(column_enthalpy_per_area(mass, h) - H0) <= 1e-9 * abs(H0)
    state = build_column_state(grid, thermo.invert_enthalpy(h), thermo, grav, enthalpy=h)
    assert abs(column_enthalpy_per_area(state.mass_path, state.enthalpy) - H0) <= 1e-9 * abs(H0)

    res = solve_implicit_convection(
        grid, state, state.enthalpy.copy(), physics, thermo, grav, state.mass_path,
        dt=2.0, solver=solver,
        cfg=ImplicitConvectionConfig(residual_tolerance=1e-10, step_tolerance=1e-10),
    )
    assert res.ok, res.diagnostics.rejection_reason
    H1 = column_enthalpy_per_area(res.state.mass_path, res.state.enthalpy)
    assert abs(H1 - H0) <= 64.0 * np.finfo(np.float64).eps * max(abs(H0), 1.0)
    assert abs(res.diagnostics.column_enthalpy_change) <= 64.0 * np.finfo(np.float64).eps * max(abs(H0), 1.0)
    assert float(res.state.temperature.min()) > 0.0


def test_manufactured_attraction_matches_boundary_work_energy_identity():
    grid, thermo, physics, solver, _t = _column(n_layers=16)
    grav = ConstantGravity(G)
    opacity = ConstantGreyOpacity(2.0e-4)
    # Mild radiative-equilibrium target so F_conv(T*) stays moderate.
    from convection_mlt import grey_radiative_equilibrium_temperature

    t_star = grey_radiative_equilibrium_temperature(grid, opacity, F0, 120.0)
    manufactured = ManufacturedRadiativeTarget(
        target_temperature=t_star, f0=F0, relaxation_coeff=1.0
    )
    hot = t_star.copy()
    hot[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 1.02
    cold = t_star.copy()
    cold[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 0.98
    cfg = RCEConfig(
        max_steps=300,
        n_consec=4,
        stall_window=400,
        flux_flatness_tolerance=1e-8,
        tendency_tolerance=1e-8,
        temp_change_tolerance=1e-8,
        attraction_temperature_tolerance=1e-8,
        dt_accuracy=1.0,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10, step_tolerance=1e-10
        ),
    )
    for initial in (hot, cold):
        state0 = build_column_state(grid, initial, thermo, grav)
        H0 = column_enthalpy_per_area(state0.mass_path, state0.enthalpy)
        res = solve_adaptive_rce(
            grid, initial, physics, solver, thermo, opacity, grid.pressure_centres,
            TopIrradiation(120.0), LowerNetInternalFlux(F0),
            gravity=grav, route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            config=cfg, manufactured=manufactured,
        )
        assert res.status == RCETerminalStatus.CONVERGED, res.reason
        rel = float(np.max(np.abs(res.final_state.temperature - t_star) / t_star))
        assert rel < 1e-8
        dH = column_enthalpy_per_area(res.final_state.mass_path, res.final_state.enthalpy) - H0
        work = sum(d.flux_boundary_work for d in res.diagnostics if d.accepted)
        scale = max(abs(H0), abs(work), 1.0)
        assert abs(dH - work) <= 1e-9 * scale


def test_single_step_explicit_implicit_differ_at_second_order():
    grid, thermo, physics, solver, t = _column(n_layers=12)
    grav = ConstantGravity(G)
    opacity = ConstantGreyOpacity(2.0e-4)
    top = TopIrradiation(120.0)
    bot = LowerNetInternalFlux(300.0)
    rce_cfg = RCEConfig(
        max_steps=1, n_consec=99, stall_window=10,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-11, step_tolerance=1e-11
        ),
    )
    # Choose a stable explicit dt from a probe.
    probe = solve_adaptive_rce(
        grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
        gravity=grav, route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=1, n_consec=99, stall_window=10),
    )
    accepted = [d for d in probe.diagnostics if d.accepted]
    assert accepted
    dt0 = 0.05 * min(accepted[0].dt_mlt, accepted[0].dt_rad, accepted[0].dt_temp, accepted[0].dt)

    errs = []
    for factor in (1.0, 0.5, 0.25):
        dt = factor * dt0
        cfg = RCEConfig(
            max_steps=1, n_consec=99, stall_window=10, prescribed_dt=dt,
            implicit_convection=rce_cfg.implicit_convection,
            coupled_picard=False,
        )
        unsplit = solve_adaptive_rce(
            grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
            gravity=grav, route=RCERoute.UNSPLIT, config=cfg,
        )
        implicit = solve_adaptive_rce(
            grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
            gravity=grav, route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV, config=cfg,
        )
        assert unsplit.steps_accepted == 1
        assert implicit.steps_accepted == 1
        scale = np.maximum(np.abs(unsplit.final_state.temperature), 1.0)
        err = float(np.max(np.abs(
            unsplit.final_state.temperature - implicit.final_state.temperature
        ) / scale))
        errs.append((dt, err))
    # Single FE vs BE step: difference O(dt^2).
    ratios = []
    for i in range(len(errs) - 1):
        dt_a, e_a = errs[i]
        dt_b, e_b = errs[i + 1]
        if e_b > 0.0 and e_a > 0.0:
            ratios.append(np.log(e_a / e_b) / np.log(dt_a / dt_b))
    assert ratios
    assert float(np.mean(ratios)) > 1.5


def test_fixed_time_explicit_implicit_differ_at_first_order():
    grid, thermo, physics, solver, t = _column(n_layers=10)
    grav = ConstantGravity(G)
    opacity = ConstantGreyOpacity(2.0e-4)
    top = TopIrradiation(120.0)
    bot = LowerNetInternalFlux(300.0)
    probe = solve_adaptive_rce(
        grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
        gravity=grav, route=RCERoute.UNSPLIT,
        config=RCEConfig(max_steps=1, n_consec=99, stall_window=10),
    )
    accepted = [d for d in probe.diagnostics if d.accepted]
    assert accepted
    dt0 = 0.02 * min(accepted[0].dt_mlt, accepted[0].dt_rad, accepted[0].dt_temp, accepted[0].dt)
    t_final = 8.0 * dt0
    errs = []
    for factor in (1.0, 0.5, 0.25):
        dt = factor * dt0
        n_steps = int(round(t_final / dt))
        cfg = RCEConfig(
            max_steps=n_steps, n_consec=10**9, stall_window=10**9,
            prescribed_dt=dt, t_final=t_final,
            coupled_picard=False,
            implicit_convection=ImplicitConvectionConfig(
                residual_tolerance=1e-11, step_tolerance=1e-11
            ),
        )
        unsplit = solve_adaptive_rce(
            grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
            gravity=grav, route=RCERoute.UNSPLIT, config=cfg,
        )
        implicit = solve_adaptive_rce(
            grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
            gravity=grav, route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV, config=cfg,
        )
        assert abs(unsplit.simulated_time - t_final) <= 1.5 * dt
        assert abs(implicit.simulated_time - t_final) <= 1.5 * dt
        scale = np.maximum(np.abs(unsplit.final_state.temperature), 1.0)
        err = float(np.max(np.abs(
            unsplit.final_state.temperature - implicit.final_state.temperature
        ) / scale))
        errs.append((dt, err))
    ratios = []
    for i in range(len(errs) - 1):
        dt_a, e_a = errs[i]
        dt_b, e_b = errs[i + 1]
        if e_b > 0.0 and e_a > 0.0:
            ratios.append(np.log(e_a / e_b) / np.log(dt_a / dt_b))
    assert ratios
    # Fixed-time FE vs BE: first-order difference O(dt).
    assert 0.6 < float(np.mean(ratios)) < 1.6


def test_implicit_convection_rejects_inverse_square_gravity():
    import pytest

    from convection_mlt import InverseSquareGravity

    grid, thermo, physics, solver, t = _column(n_layers=8)
    grav = InverseSquareGravity(g0=G, planet_radius=1.0e8)
    state = build_column_state(grid, t, thermo, grav)
    with pytest.raises(ValueError, match="ConstantGravity"):
        solve_implicit_convection(
            grid, state, state.enthalpy.copy(), physics, thermo, grav,
            state.mass_path, 1.0, solver,
        )
    with pytest.raises(ValueError, match="ConstantGravity"):
        solve_adaptive_rce(
            grid, t, physics, solver, thermo, ConstantGreyOpacity(2e-4),
            grid.pressure_centres, TopIrradiation(120.0), LowerNetInternalFlux(F0),
            gravity=grav, route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            config=RCEConfig(max_steps=1),
        )


def test_implicit_dt_hold_does_not_double_immediately_after_reject():
    import pytest

    from convection_mlt.rce import dt_after_accept, dt_after_reject

    cfg = RCEConfig(f_back=0.5, f_grow=1.2, n_hold_after_reject=4)
    dt, ceil, hold = dt_after_reject(1250.0, cfg)
    assert dt == 625.0
    nxt, ceil, hold = dt_after_accept(
        dt, 1e9, cfg, implicit=True, dt_ceiling=ceil, hold_remaining=hold
    )
    assert nxt <= 625.0 + 1e-12
    for _ in range(3):
        nxt, ceil, hold = dt_after_accept(
            nxt, 1e9, cfg, implicit=True, dt_ceiling=ceil, hold_remaining=hold
        )
    assert nxt <= 625.0 + 1e-12
    nxt, ceil, hold = dt_after_accept(
        nxt, 1e9, cfg, implicit=True, dt_ceiling=ceil, hold_remaining=hold
    )
    assert nxt == pytest.approx(625.0 * 1.2)
    explicit_nxt, _, _ = dt_after_accept(
        625.0, 1e9, cfg, implicit=False, dt_ceiling=None, hold_remaining=0
    )
    assert explicit_nxt == pytest.approx(1250.0)


def test_bordered_two_thomas_matches_dense_augmented():
    from convection_mlt import (
        conservation_residual,
        conservation_weight,
        solve_bordered_correction,
        solve_bordered_dense,
    )

    for n in (6, 8, 12):
        grid, thermo, physics, solver, t = _column(n_layers=n)
        grav = ConstantGravity(G)
        state = build_column_state(grid, t, thermo, grav)
        h_star = state.enthalpy * 0.999
        mass = state.mass_path.copy()
        dt = 1.0
        closure = evaluate_mlt(grid, state, physics, thermo)
        support = provisional_support(closure)
        f = flux_with_provisional_support(closure, support)
        residual = _residual(state.enthalpy, h_star, f, mass, dt)
        cfg = ImplicitConvectionConfig()
        lower, diag, upper, _ = assemble_tridiagonal_jacobian(
            grid, state, h_star, support, physics, thermo, grav, mass, dt, cfg, residual
        )
        dense = assemble_dense_jacobian(
            grid, state, h_star, support, physics, thermo, grav, mass, dt, cfg, residual
        )
        w = conservation_weight(mass)
        c = conservation_residual(state.enthalpy, h_star, mass)
        lam = 0.25
        dh_t, dlam_t, schur = solve_bordered_correction(
            lower, diag, upper, residual, w, c, lam,
            pivot_floor=cfg.pivot_floor, schur_floor=cfg.schur_floor,
        )
        dh_d, dlam_d = solve_bordered_dense(dense, residual, w, c, lam)
        assert np.isfinite(schur)
        assert np.allclose(dh_t, dh_d, rtol=1e-8, atol=1e-8)
        assert dlam_t == pytest.approx(dlam_d, rel=1e-8, abs=1e-8)


def test_bordered_newton_reaches_residual_conservation_and_lambda():
    grid, thermo, physics, solver, t = _column(n_layers=16)
    grav = ConstantGravity(G)
    state = build_column_state(grid, t, thermo, grav)
    h_star = state.enthalpy.copy()
    h_star[:4] *= 1.01
    res = solve_implicit_convection(
        grid, state, h_star, physics, thermo, grav, state.mass_path,
        dt=2.0, solver=solver,
        cfg=ImplicitConvectionConfig(),
    )
    assert res.ok, res.diagnostics.rejection_reason
    assert res.diagnostics.residual_norm <= 1e-10
    c_scale = max(float(np.max(np.abs(h_star))), 1e-30)
    assert abs(res.diagnostics.conservation_residual) / c_scale <= 1e-12
    assert abs(res.diagnostics.multiplier) <= 1e-8 or np.isfinite(res.diagnostics.multiplier)


def test_bordered_active_set_one_interface_rcb():
    grid, thermo, physics, solver, t = _column(n_layers=16)
    grav = ConstantGravity(G)
    # Deep convective, radiative lid: one RCB interface.
    t = t.copy()
    t[:8] *= 1.12
    t[8:] *= 0.96
    state = build_column_state(grid, t, thermo, grav)
    support0 = provisional_support(evaluate_mlt(grid, state, physics, thermo))
    crossings = np.where(support0[1:-1] != support0[2:])[0]
    assert crossings.size >= 1
    h_star = state.enthalpy.copy()
    # Heat just below the RCB so the active set may move one interface.
    rcb_layer = int(crossings[0])
    h_star[rcb_layer] *= 1.015
    res = solve_implicit_convection(
        grid, state, h_star, physics, thermo, grav, state.mass_path,
        dt=1.0, solver=solver,
        cfg=ImplicitConvectionConfig(),
    )
    assert res.ok, res.diagnostics.rejection_reason
    support1 = provisional_support(res.closure)
    crossings1 = np.where(support1[1:-1] != support1[2:])[0]
    assert crossings1.size >= 1
    assert res.diagnostics.mask_outer_iterations >= 1


def test_newton_starts_from_valid_state_if_h_star_does_not_invert():
    from convection_mlt.implicit_convection import _newton_start

    grid, thermo, physics, solver, t = _column(n_layers=8)
    grav = ConstantGravity(G)
    state = build_column_state(grid, t, thermo, grav)
    h_bad = np.full_like(state.enthalpy, np.nan)
    started, h0 = _newton_start(grid, state, h_bad, thermo, grav)
    assert np.allclose(started.enthalpy, state.enthalpy)
    assert np.allclose(h0, state.enthalpy)
    res = solve_implicit_convection(
        grid, state, h_bad, physics, thermo, grav, state.mass_path,
        dt=0.5, solver=solver,
    )
    assert res.diagnostics.newton_iterations >= 1
    assert res.diagnostics.rejection_reason != "coupled_picard_failure: h* invert"


def test_bordered_reproduces_n192_100s_and_probes_larger_dt():
    import json
    from pathlib import Path

    rec_path = Path(__file__).resolve().parents[1] / "results" / "n192_implicit_rce.json"
    if not rec_path.exists():
        import pytest
        pytest.skip("n192_implicit_rce.json not stored")
    record = json.loads(rec_path.read_text())
    spec = __import__("convection_mlt", fromlist=["nested_analytic_opacity_spec"]).nested_analytic_opacity_spec(192)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    t = np.asarray(record["temperature"], dtype=np.float64)
    probes = {}
    for dt in (100.0, 180.0, 300.0, 500.0, 1000.0):
        cfg = RCEConfig(
            max_steps=1,
            n_consec=99,
            stall_window=10**9,
            prescribed_dt=dt,
            coupled_picard=True,
            implicit_convection=ImplicitConvectionConfig(),
        )
        res = solve_adaptive_rce(
            grid, t, spec.physics(), solver, thermo, spec.opacity(),
            grid.pressure_centres, TopIrradiation(spec.f_irr),
            LowerNetInternalFlux(spec.f_int),
            gravity=ConstantGravity(spec.gravity),
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            config=cfg,
        )
        accepted = [d for d in res.diagnostics if d.accepted]
        probes[dt] = {
            "accepted": bool(accepted),
            "reason": res.reason if not accepted else None,
            "residual": None if not accepted else accepted[0].nonlinear_residual,
            "defect": None if not accepted else accepted[0].coupled_defect,
        }
    assert probes[100.0]["accepted"], probes[100.0]
    assert probes[100.0]["residual"] <= 1e-10
    # Record larger-dt outcomes without relaxing tolerances.
    record_path = Path(__file__).resolve().parents[1] / "results" / "n192_bordered_dt_probes.json"
    record_path.write_text(json.dumps({k: probes[k] for k in probes}, indent=2))
    for dt in (180.0, 300.0, 500.0, 1000.0):
        if probes[dt]["accepted"]:
            assert probes[dt]["residual"] <= 1e-10


def test_restart_seeds_dt_hold_and_simulated_time():
    import pytest

    grid, thermo, physics, solver, t = _column(n_layers=8)
    grav = ConstantGravity(G)
    opacity = ConstantGreyOpacity(2.0e-4)
    top = TopIrradiation(120.0)
    bot = LowerNetInternalFlux(F0)
    cfg1 = RCEConfig(
        max_steps=2,
        n_consec=10**9,
        stall_window=10**9,
        coupled_picard=True,
        dt_accuracy=5.0,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10, step_tolerance=1e-10
        ),
    )
    res1 = solve_adaptive_rce(
        grid, t, physics, solver, thermo, opacity, grid.pressure_centres, top, bot,
        gravity=grav, route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV, config=cfg1,
    )
    accepted = [d for d in res1.diagnostics if d.accepted]
    assert accepted
    last = accepted[-1]
    cfg2 = RCEConfig(
        max_steps=1,
        n_consec=10**9,
        stall_window=10**9,
        coupled_picard=True,
        dt_accuracy=1.0e9,
        dt_hold_init=last.dt,
        simulated_time_init=res1.simulated_time,
        previous_rcb_init=res1.primary_rcb_log10p,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10, step_tolerance=1e-10
        ),
    )
    res2 = solve_adaptive_rce(
        grid, res1.final_state.temperature, physics, solver, thermo, opacity,
        grid.pressure_centres, top, bot, gravity=grav,
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV, config=cfg2,
    )
    accepted2 = [d for d in res2.diagnostics if d.accepted]
    assert accepted2
    assert accepted2[0].dt <= last.dt + 1e-12
    assert res2.simulated_time == pytest.approx(res1.simulated_time + accepted2[0].dt)

