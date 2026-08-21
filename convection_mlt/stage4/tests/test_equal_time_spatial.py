"""Equal-time trajectories and spatial refinement at the 1e-3 gate."""

from __future__ import annotations

import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.energy import column_enthalpy_per_area


STAGE4_EXIT_FLUX_GATE = 1.0e-3


def _spec(n_layers: int) -> AnalyticOpacityRCESpec:
    n_phot = 16 if n_layers <= 48 else 24 if n_layers <= 96 else 32
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=n_phot)


def _solver() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)


def _cfg(max_steps: int, gate: float, t_final: float | None = None) -> RCEConfig:
    return RCEConfig(
        max_steps=max_steps,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=gate,
        tendency_tolerance=gate,
        temp_change_tolerance=gate,
        dt_accuracy=2500.0,
        t_final=t_final,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10,
            step_tolerance=1e-10,
            newton_residual_tolerance=1e-12,
            newton_step_tolerance=1e-12,
        ),
    )


def interpolate_temperature(
    log_p_src: np.ndarray, t_src: np.ndarray, log_p_dst: np.ndarray
) -> np.ndarray:
    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def _run(n_layers: int, *, max_steps: int, gate: float, t_final: float | None = None):
    spec = _spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
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
        config=_cfg(max_steps, gate, t_final),
    )
    return spec, grid, res


def test_equal_time_n48_n96_share_physical_time_axis():
    """N=48 and N=96 advance to the same t_final; series are vs time not steps."""
    t_final = 5.0e4
    _, _, r48 = _run(48, max_steps=400, gate=1e-12, t_final=t_final)
    _, _, r96 = _run(96, max_steps=800, gate=1e-12, t_final=t_final)
    assert r48.simulated_time >= 0.99 * t_final
    assert r96.simulated_time >= 0.99 * t_final
    assert abs(r48.simulated_time - r96.simulated_time) <= 0.02 * t_final
    for res in (r48, r96):
        accepted = [d for d in res.diagnostics if d.accepted]
        assert accepted
        times = np.cumsum([d.dt for d in accepted])
        assert abs(float(times[-1]) - res.simulated_time) <= 1e-9 * max(res.simulated_time, 1.0)
        assert res.detached_convective_regions == []
        assert res.convective_regions and res.convective_regions[0][0] == 0
    assert r48.steps_accepted > 0 and r96.steps_accepted > 0


def test_resolution_sensitivity_regression_n48_n96():
    """Regression guard for resolution sensitivity — not a grid-independence claim.

    N=48 and N=96 both reach the 1e-3 gate with a bottom-connected CZ. Profile
    differences remain O(10%) in T and O(0.3) dex in RCB; those bounds lock the
    present state. A genuine convergence test requires timestep-controlled,
    independently gate-converged solutions.
    """
    cases = {}
    for n, steps in ((48, 400), (96, 2500)):
        _, grid, res = _run(n, max_steps=steps, gate=STAGE4_EXIT_FLUX_GATE)
        assert res.status == RCETerminalStatus.CONVERGED, (n, res.reason, res.convergence.flux_flatness)
        assert res.convergence.flux_flatness <= STAGE4_EXIT_FLUX_GATE
        assert res.detached_convective_regions == []
        assert res.convective_regions and res.convective_regions[0][0] == 0
        cases[n] = {
            "grid": grid,
            "temperature": res.final_state.temperature,
            "flux_total": res.final_flux_total,
            "primary_rcb_log10p": res.primary_rcb_log10p,
            "column_enthalpy": column_enthalpy_per_area(
                res.final_state.mass_path, res.final_state.enthalpy
            ),
        }
    log_p48 = np.log10(cases[48]["grid"].pressure_centres)
    log_p96 = np.log10(cases[96]["grid"].pressure_centres)
    t96_on_48 = interpolate_temperature(log_p96, cases[96]["temperature"], log_p48)
    t48 = cases[48]["temperature"]
    rel_t = float(np.max(np.abs(t96_on_48 - t48) / np.maximum(t48, 1.0)))
    dlog_rcb = abs(
        float(cases[96]["primary_rcb_log10p"]) - float(cases[48]["primary_rcb_log10p"])
    )
    dlog_cell = float(np.mean(np.abs(np.diff(log_p48))))
    cells = dlog_rcb / max(dlog_cell, 1e-30)
    assert dlog_rcb < 0.35
    assert cells < 4.0
    assert rel_t < 0.15
    assert abs(float(cases[96]["flux_total"][0]) - float(cases[48]["flux_total"][0])) <= 1e-6 * abs(
        float(cases[48]["flux_total"][0])
    )


def test_interpolate_temperature_roundtrip():
    log_p = np.linspace(6.0, 2.0, 20)
    t = 300.0 + 50.0 * (log_p - 2.0)
    t2 = interpolate_temperature(log_p, t, log_p)
    assert float(np.max(np.abs(t2 - t))) < 1e-10
