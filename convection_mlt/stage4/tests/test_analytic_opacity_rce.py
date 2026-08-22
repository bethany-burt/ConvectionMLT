"""Pressure-dependent grey RCE: bottom-connected seed and coupled control."""

from __future__ import annotations

import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    NASAThermo,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    grey_layer_optical_thickness,
    grey_radiative_equilibrium_temperature,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.rce import (
    DEFAULT_DIFFUSIVITY,
    analytic_opacity_tau_edges,
    _evaluate_closure,
)
from convection_mlt.state import build_column_state


NABLA_AD_H2 = 2.0 / 7.0
# Explicit MLT at c_diff=0.2 reaches a quasi-steady bottom-connected RCE
# with flux_flatness ≈ 8e-2. Structural / RCB regression only — NOT Stage 4 exit.
# Audit label: PILOT_GATE_REACHED / EXPLICIT_REFERENCE_LIMIT.
EXPLICIT_MLT_QUASISTEADY_GATE = 0.1
STAGE4_EXIT_FLUX_GATE = 1.0e-3


def _spec(n_layers: int = 48, **kwargs) -> AnalyticOpacityRCESpec:
    n_phot = kwargs.pop("n_photosphere", 16 if n_layers >= 48 else max(4, n_layers // 3))
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=n_phot, **kwargs)


def _solver() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)


def _implicit_cfg(max_steps: int, gate: float) -> RCEConfig:
    return RCEConfig(
        max_steps=max_steps,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=gate,
        tendency_tolerance=gate,
        temp_change_tolerance=gate,
        dt_accuracy=2500.0,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10,
            step_tolerance=1e-10,
            newton_residual_tolerance=1e-12,
            newton_step_tolerance=1e-12,
        ),
    )


def test_kappa0_matches_target_optical_depth():
    spec = _spec()
    expected = spec.tau_total * spec.gravity * (spec.a + 1.0) / spec.p_bottom
    assert abs(spec.kappa0 - expected) <= 1e-16 * expected
    assert abs(spec.kappa0 - 2.25e-3) <= 1e-12
    nabla_rad_deep = (spec.a + 1.0) / 4.0
    assert nabla_rad_deep > NABLA_AD_H2
    assert abs(nabla_rad_deep - 0.375) <= 1e-15


def test_analytic_opacity_top_is_optically_thin():
    spec = _spec()
    grid = spec.grid()
    opacity = spec.opacity()
    t = grey_radiative_equilibrium_temperature(grid, opacity, spec.f_int, spec.f_irr)
    dtau = grey_layer_optical_thickness(grid, opacity, t)
    assert float(dtau[-1]) < 0.2
    assert float(np.exp(-dtau[-1])) > 0.8
    assert float(np.sum(dtau) / DEFAULT_DIFFUSIVITY) > 50.0
    assert float(dtau[0]) < 20.0


def test_analytic_re_seed_is_bottom_unstable_and_rc_seed_is_bottom_connected():
    spec = _spec()
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t_re = grey_radiative_equilibrium_temperature(grid, opacity, spec.f_int, spec.f_irr)
    log_t = np.log(t_re)
    log_p = np.log(grid.pressure_centres)
    nabla = (log_t[:-1] - log_t[1:]) / (log_p[:-1] - log_p[1:])
    assert float(nabla[0]) > NABLA_AD_H2

    t_rc = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    physics = spec.physics()
    state = build_column_state(grid, t_rc, thermo, ConstantGravity(spec.gravity))
    closure = _evaluate_closure(grid, state, physics, thermo)
    assert float(np.max(np.abs(t_rc - t_re))) > 0.0
    assert np.array_equal(t_rc[-3:], t_re[-3:])
    assert float(np.max(closure.flux)) < 1.0e6


def test_rc_seed_ignores_detached_upper_unstable_segments():
    spec = AnalyticOpacityRCESpec(
        n_layers=24, n_photosphere=8, a=0.0, tau_total=5.0, p_top=1.0e3
    )
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t_re = grey_radiative_equilibrium_temperature(grid, opacity, spec.f_int, spec.f_irr)
    log_t = np.log(t_re)
    log_p = np.log(grid.pressure_centres)
    nabla = (log_t[:-1] - log_t[1:]) / (log_p[:-1] - log_p[1:])
    nabla_ad = float(thermo.nabla_ad)
    assert float(nabla[0]) <= nabla_ad
    t_rc = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    assert np.array_equal(t_rc, t_re)


def test_nasa_seed_is_isentropic_in_the_bottom_connected_region():
    spec = _spec(n_layers=24, n_photosphere=8)
    grid = spec.grid()
    thermo = NASAThermo.from_json()
    opacity = spec.opacity()
    t_re = grey_radiative_equilibrium_temperature(grid, opacity, spec.f_int, spec.f_irr)
    t_rc = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    log_t = np.log(t_re)
    log_p = np.log(grid.pressure_centres)
    nabla = (log_t[:-1] - log_t[1:]) / (log_p[:-1] - log_p[1:])
    nabla_ad = 0.5 * (thermo.nabla_ad_at(t_re)[:-1] + thermo.nabla_ad_at(t_re)[1:])
    assert float(nabla[0]) > float(nabla_ad[0])
    i = 0
    while i < nabla.size and nabla[i] > nabla_ad[i]:
        i += 1
    i_join = min(i, grid.n_layers - 1)
    s = thermo.entropy(t_rc, grid.pressure_centres)
    s_join = float(s[i_join])
    if i_join >= 2:
        rel = np.abs(s[: i_join - 1] - s_join) / max(abs(s_join), 1.0)
        assert float(np.max(rel)) < 1e-10
    assert np.array_equal(t_rc[i_join + 1 :], t_re[i_join + 1 :])


def _run_explicit_coupled(spec: AnalyticOpacityRCESpec, initial, max_steps: int, gate: float):
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    return solve_adaptive_rce(
        grid, initial, spec.physics(), _solver(), thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.UNSPLIT,
        config=RCEConfig(
            max_steps=max_steps,
            n_consec=5,
            stall_window=10**9,
            flux_flatness_tolerance=gate,
            tendency_tolerance=gate,
            temp_change_tolerance=gate,
        ),
    )


def test_explicit_mlt_quasisteady_is_structural_rcb_regression_not_exit_gate():
    """PILOT_GATE_REACHED / EXPLICIT_REFERENCE_LIMIT — not Stage 4 exit evidence.

    Converged to the declared exploratory tolerance of 0.1; Stage 4 exit
    tolerance of 1e-3 is not evaluated by this test.
    """
    spec = _spec()
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    res = _run_explicit_coupled(spec, t, max_steps=8000, gate=EXPLICIT_MLT_QUASISTEADY_GATE)
    assert res.status == RCETerminalStatus.CONVERGED
    assert res.convergence.flux_flatness <= EXPLICIT_MLT_QUASISTEADY_GATE
    assert res.primary_rcb_log10p is not None
    assert res.detached_convective_regions == []
    assert res.convective_regions and res.convective_regions[0][0] == 0
    # Must not be confused with the exit gate.
    assert STAGE4_EXIT_FLUX_GATE < EXPLICIT_MLT_QUASISTEADY_GATE


def test_two_resolutions_form_a_bottom_connected_rcb():
    """N=48 and N=96 must both develop a physical RCB under explicit MLT."""
    for n, n_phot, steps in ((48, 16, 1500), (96, 24, 1500)):
        spec = AnalyticOpacityRCESpec(n_layers=n, n_photosphere=n_phot)
        grid = spec.grid()
        thermo = ConstantH2Thermo()
        opacity = spec.opacity()
        t = radiative_convective_initial_temperature(
            grid, opacity, thermo, spec.f_int, spec.f_irr
        )
        res = _run_explicit_coupled(spec, t, max_steps=steps, gate=1e-12)
        assert res.primary_rcb_log10p is not None
        assert res.detached_convective_regions == []
        assert res.convective_regions and res.convective_regions[0][0] == 0
        assert abs(float(res.final_flux_total[0]) - spec.f_int) <= 1e-8 * spec.f_int
        assert float(res.final_flux_conv[1]) > 0.0


def test_implicit_analytic_opacity_reaches_stage4_exit_gate():
    """N=48 analytic-opacity benchmark: real 1e-3 RCE with implicit convection."""
    spec = _spec()
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    res = solve_adaptive_rce(
        grid, t, spec.physics(), _solver(), thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=_implicit_cfg(max_steps=400, gate=STAGE4_EXIT_FLUX_GATE),
    )
    assert res.status == RCETerminalStatus.CONVERGED, res.reason
    assert res.convergence.flux_flatness <= STAGE4_EXIT_FLUX_GATE
    assert res.convergence.tendency_norm <= STAGE4_EXIT_FLUX_GATE
    # Analytic-opacity benchmark requirement (not a universal RCE gate).
    assert res.primary_rcb_log10p is not None
    assert res.detached_convective_regions == []
    assert res.convective_regions and res.convective_regions[0][0] == 0
    assert abs(float(res.final_flux_total[0]) - spec.f_int) <= 1e-8 * spec.f_int
    assert float(res.final_flux_conv[1]) > 0.0
    assert float(res.final_state.temperature.min()) > 200.0


def _isoenthalpic_cz_redistribution(
    enthalpy: np.ndarray,
    mass_path: np.ndarray,
    thermo: ConstantH2Thermo,
    *,
    heat: slice,
    cool: slice,
    amplitude: float,
) -> np.ndarray:
    """Mass-conserving CZ reshape: heat one band, cool another (column ΔH ≈ 0).

    Cool-deep / heat-upper kicks leave this analytic-opacity basin (detached CZ).
    In-basin probes heat a deeper band and cool an upper band inside the CZ.
    """
    h = np.asarray(enthalpy, dtype=np.float64).copy()
    m = np.asarray(mass_path, dtype=np.float64)
    h0 = h.copy()
    h[heat] *= 1.0 + amplitude
    excess = float(np.sum(m * (h - h0)))
    h[cool] -= excess / float(np.sum(m[cool]))
    return thermo.invert_enthalpy(h)


def test_in_basin_isoenthalpic_path_independence():
    """Local path independence inside the existing convective basin.

    Two distinct in-basin isoenthalpic CZ redistributions must reconverge to the
    same bottom-connected RCE. This is not a global hot/cold attractor test:
    mid-column ± multiply and cool-deep kicks spawn detached zones on this
    benchmark and are excluded by construction.
    """
    spec = _spec()
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    settled = solve_adaptive_rce(
        grid, t0, spec.physics(), _solver(), thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=_implicit_cfg(max_steps=400, gate=STAGE4_EXIT_FLUX_GATE),
    )
    assert settled.status == RCETerminalStatus.CONVERGED, settled.reason
    t_star = settled.final_state.temperature.copy()
    h_star = settled.final_state.enthalpy
    m = settled.final_state.mass_path
    # Distinct spatial modes: both heat-deeper / cool-upper inside the CZ.
    initials = [
        _isoenthalpic_cz_redistribution(
            h_star, m, thermo, heat=slice(2, 12), cool=slice(12, 22), amplitude=0.01
        ),
        _isoenthalpic_cz_redistribution(
            h_star, m, thermo, heat=slice(8, 16), cool=slice(16, 26), amplitude=0.01
        ),
    ]
    finals = []
    for initial in initials:
        assert float(np.max(np.abs(initial - t_star) / np.maximum(t_star, 1.0))) > 1e-4
        res = solve_adaptive_rce(
            grid, initial, spec.physics(), _solver(), thermo, opacity, grid.pressure_centres,
            TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
            gravity=ConstantGravity(spec.gravity),
            route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
            config=_implicit_cfg(max_steps=200, gate=STAGE4_EXIT_FLUX_GATE),
        )
        assert res.status == RCETerminalStatus.CONVERGED, res.reason
        assert res.convergence.flux_flatness <= STAGE4_EXIT_FLUX_GATE
        assert res.primary_rcb_log10p is not None
        assert res.detached_convective_regions == []
        assert res.convective_regions and res.convective_regions[0][0] == 0
        finals.append(res)
    scale = np.maximum(np.abs(finals[0].final_state.temperature), 1.0)
    rel = float(np.max(np.abs(
        finals[0].final_state.temperature - finals[1].final_state.temperature
    ) / scale))
    assert rel < 5e-3
    assert abs(finals[0].primary_rcb_log10p - finals[1].primary_rcb_log10p) < 0.05
    for res in finals:
        rel_star = float(np.max(np.abs(res.final_state.temperature - t_star) / scale))
        assert rel_star < 5e-3


def test_nasa_h2_coupled_smoke_accepts_implicit_steps():
    """NASA H₂ is not the ConstantH2 parity EOS; this is a coupled-domain smoke.

    HELIOS comparison must use a matched ∇_ad / EOS. ConstantH2Thermo remains
    the analytic-opacity 1e-3 benchmark until that choice is locked.
    """
    spec = _spec(n_layers=16, n_photosphere=6)
    grid = spec.grid()
    thermo = NASAThermo.from_json()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    res = solve_adaptive_rce(
        grid, t0, spec.physics(), _solver(), thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=_implicit_cfg(max_steps=40, gate=STAGE4_EXIT_FLUX_GATE),
    )
    assert res.steps_accepted > 0, res.reason
    assert np.all(np.isfinite(res.final_state.temperature))
    assert float(res.final_state.temperature.min()) > 0.0
    assert res.status != RCETerminalStatus.DT_MIN_FAILURE


def test_nested_optical_depth_family_retains_tau1_and_endpoints():
    master = nested_analytic_opacity_spec(384)
    tau_master = analytic_opacity_tau_edges(
        AnalyticOpacityRCESpec(
            n_layers=master.nested_master_layers,
            n_photosphere=master.nested_master_photosphere,
        )
    )
    assert abs(float(tau_master[64]) - 1.0) <= 1e-15
    p_master = master.pressure_edges()
    for n, stride in ((192, 2), (96, 4), (48, 8)):
        spec = nested_analytic_opacity_spec(n)
        p = spec.pressure_edges()
        assert p.size == n + 1
        assert abs(float(p[0]) - spec.p_bottom) <= 1e-9 * spec.p_bottom
        assert abs(float(p[-1]) - spec.p_top) <= 1e-12
        np.testing.assert_allclose(p, p_master[::stride], rtol=0.0, atol=1e-9 * spec.p_bottom)
        tau = analytic_opacity_tau_edges(
            AnalyticOpacityRCESpec(
                n_layers=spec.nested_master_layers,
                n_photosphere=spec.nested_master_photosphere,
            )
        )[::stride]
        assert abs(float(tau[spec.n_photosphere]) - spec.tau_photosphere) <= 1e-12


def test_coupled_picard_reduces_defect_on_short_n16_run():
    spec = _spec(n_layers=16, n_photosphere=6)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    cfg = _implicit_cfg(max_steps=12, gate=1e-12)
    res = solve_adaptive_rce(
        grid, t0, spec.physics(), _solver(), thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    accepted = [d for d in res.diagnostics if d.accepted]
    assert accepted
    assert any(d.picard_iterations >= 1 for d in accepted)
    finite_def = [d.coupled_defect for d in accepted if np.isfinite(d.coupled_defect)]
    assert finite_def
    assert min(finite_def) <= 1e-8
    assert np.all(np.isfinite(res.final_state.temperature))
    for d in accepted:
        scale = max(abs(d.energy_committed), abs(d.flux_boundary_work), 1e-30)
        assert abs(d.energy_committed_residual) <= max(
            16.0 * d.energy_ulp_floor, 1e-12 * scale
        )
