"""Pressure-dependent grey RCE: bottom-connected seed and coupled control."""

from __future__ import annotations

import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    NASAThermo,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    grey_layer_optical_thickness,
    grey_radiative_equilibrium_temperature,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.rce import (
    DEFAULT_DIFFUSIVITY,
    _evaluate_closure,
)
from convection_mlt.state import build_column_state


NABLA_AD_H2 = 2.0 / 7.0
# Explicit MLT at c_diff=0.2 reaches a quasi-steady bottom-connected RCE
# with flux_flatness ≈ 8e-2. This is the declared gate for CONVERGED on
# this pilot; it is not a 1e-3 / 1e-8 real-RCE flux gate.
EXPLICIT_MLT_QUASISTEADY_GATE = 0.1


def _spec(n_layers: int = 48, **kwargs) -> AnalyticOpacityRCESpec:
    n_phot = kwargs.pop("n_photosphere", 16 if n_layers >= 48 else max(4, n_layers // 3))
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=n_phot, **kwargs)


def _solver() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)


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


def _run_coupled(spec: AnalyticOpacityRCESpec, initial, max_steps: int, gate: float):
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


def test_coupled_analytic_opacity_converges_with_bottom_connected_rcb():
    spec = _spec()
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    res = _run_coupled(spec, t, max_steps=8000, gate=EXPLICIT_MLT_QUASISTEADY_GATE)
    assert res.status == RCETerminalStatus.CONVERGED
    assert res.convergence.flux_flatness <= EXPLICIT_MLT_QUASISTEADY_GATE
    assert res.convergence.tendency_norm <= EXPLICIT_MLT_QUASISTEADY_GATE
    assert res.primary_rcb_log10p is not None
    assert res.detached_convective_regions == []
    assert res.convective_regions and res.convective_regions[0][0] == 0
    assert abs(float(res.final_flux_total[0]) - spec.f_int) <= 1e-8 * spec.f_int
    assert float(np.max(res.final_flux_conv)) < spec.f_int
    assert float(res.final_flux_conv[1]) > 0.0
    assert float(res.final_state.temperature.min()) > 200.0


def test_two_resolutions_form_a_bottom_connected_rcb():
    """N=48 and N=96 must both develop a physical RCB; flux gates differ by CFL."""
    for n, n_phot, steps in ((48, 16, 1500), (96, 24, 1500)):
        spec = AnalyticOpacityRCESpec(n_layers=n, n_photosphere=n_phot)
        grid = spec.grid()
        thermo = ConstantH2Thermo()
        opacity = spec.opacity()
        t = radiative_convective_initial_temperature(
            grid, opacity, thermo, spec.f_int, spec.f_irr
        )
        res = _run_coupled(spec, t, max_steps=steps, gate=1e-12)
        assert res.primary_rcb_log10p is not None
        assert res.detached_convective_regions == []
        assert res.convective_regions and res.convective_regions[0][0] == 0
        assert abs(float(res.final_flux_total[0]) - spec.f_int) <= 1e-8 * spec.f_int
        assert float(res.final_flux_conv[1]) > 0.0
