"""Regression tests for H2/He mixture production RCE (post entropy-adiabat accelerator).

Fast tests run in CI by default. The N=96 acceptance matrix is marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.gravity import ConstantGravity
from convection_mlt.production_rce import (
    PHYSICAL_GATE,
    ProductionControls,
    _gates_from_result,
    production_solver_config,
    production_thermo,
    run_production_rce,
    validation_envelope,
)
from convection_mlt.rce import nested_analytic_opacity_spec, radiative_convective_initial_temperature
from convection_mlt.reduced_rce import (
    ReducedRCEConfig,
    reconstruct_cz_temperature,
    solve_reduced_radiative_matching,
)
from convection_mlt.radiation import LowerNetInternalFlux, TopIrradiation
from convection_mlt.thermodynamics import ConstantH2Thermo, h2_he_mixture

N96 = 96
F_INT = 300.0

ACCEPTANCE_MATRIX = (
    (0.0, 0.0),
    (0.0, 120.0),
    (0.0, 500.0),
    (0.1, 0.0),
    (0.1, 120.0),
    (0.1, 500.0),
    (0.2, 0.0),
    (0.2, 120.0),
    (0.2, 500.0),
)


def _run_production(*, x_he: float, f_irr: float) -> tuple[str, float]:
    run = run_production_rce(
        n_layers=N96,
        alpha=1.0,
        f_int=F_INT,
        f_irr=f_irr,
        x_he=x_he,
        seed="radiative_convective",
        controls=ProductionControls(max_recovery_cycles=2),
    )
    require_topo = abs(f_irr) <= 1.0e-15
    gates = _gates_from_result(
        run.result,
        run.spec,
        gate=PHYSICAL_GATE,
        require_bottom_connected_cz=require_topo,
    )
    passed = gates.convergence_ok and (gates.topology_ok or not require_topo)
    return ("CONVERGED" if passed else "NOT CONVERGED", float(gates.flux_flatness))


@pytest.mark.parametrize("x_he,f_irr", ACCEPTANCE_MATRIX)
@pytest.mark.slow
def test_n96_acceptance_matrix(x_he: float, f_irr: float) -> None:
    """Full production path at N=96 must pass the frozen 1e-3 gate."""
    verdict, flatness = _run_production(x_he=x_he, f_irr=f_irr)
    assert verdict == "CONVERGED", f"x_he={x_he} f_irr={f_irr} flat={flatness}"
    assert flatness <= PHYSICAL_GATE


@pytest.mark.parametrize("x_he", [0.1, 0.2])
def test_reconstruct_cz_zero_delta_preserves_entropy(x_he: float) -> None:
    """Guard: mixture CZ rebuild with Delta=0 must follow an isentrope."""
    spec = nested_analytic_opacity_spec(N96, alpha=1.0, f_int=F_INT, f_irr=0.0)
    grid = spec.grid()
    thermo = h2_he_mixture(x_he)
    t0 = radiative_convective_initial_temperature(
        grid, spec.opacity(), thermo, F_INT, 0.0
    )
    i_hi = 40
    delta = np.zeros(grid.n_layers + 1, dtype=np.float64)
    t_cz = reconstruct_cz_temperature(grid, t0.copy(), delta, i_hi, thermo)
    p = np.asarray(grid.pressure_centres[: i_hi + 1], dtype=np.float64)
    s = thermo.entropy(t_cz[: i_hi + 1], p)
    assert np.max(np.abs(s - float(s[-1]))) / max(abs(float(s[-1])), 1.0) <= 1.0e-10


def test_reconstruct_cz_pure_h2_matches_power_law() -> None:
    """Pure H2 limit must remain compatible with constant nabla_ad = 2/7."""
    spec = nested_analytic_opacity_spec(N96, alpha=1.0, f_int=F_INT, f_irr=0.0)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    p = np.asarray(grid.pressure_centres, dtype=np.float64)
    i_hi = 50
    t_anchor = 2200.0
    t0 = np.full(grid.n_layers, t_anchor, dtype=np.float64)
    delta = np.zeros(grid.n_layers + 1, dtype=np.float64)
    t_cz = reconstruct_cz_temperature(grid, t0.copy(), delta, i_hi, thermo)
    t_expected = t_anchor * (p[: i_hi + 1] / p[i_hi]) ** thermo.nabla_ad
    rel = np.max(np.abs(t_cz[: i_hi + 1] / t_expected - 1.0))
    assert rel <= 1.0e-12


@pytest.mark.parametrize("x_he,f_irr", [(0.2, 0.0), (0.2, 500.0)])
def test_reduced_rz_mixture_produces_finite_improvement(x_he: float, f_irr: float) -> None:
    """Reduced accelerator must return a usable trial for mixtures (no domain failure)."""
    spec = nested_analytic_opacity_spec(N96, alpha=1.0, f_int=F_INT, f_irr=f_irr)
    grid = spec.grid()
    thermo = production_thermo(x_he)
    solver = production_solver_config()
    t0 = radiative_convective_initial_temperature(
        grid, spec.opacity(), thermo, F_INT, f_irr
    )
    reduced = solve_reduced_radiative_matching(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(f_irr),
        LowerNetInternalFlux(F_INT),
        gravity=ConstantGravity(spec.gravity),
        config=ReducedRCEConfig(),
    )
    assert reduced.trial is not None
    assert np.all(np.isfinite(reduced.temperature))
    assert np.all(reduced.temperature > 0.0)
    assert reduced.improved or reduced.flux_flatness <= 500.0


@pytest.mark.parametrize(
    "x_he,f_irr,expected",
    [
        (0.0, 0.0, "INSIDE_VALIDATED_ENVELOPE"),
        (0.2, 0.0, "EXPERIMENTAL_OUTSIDE_VALIDATED_ENVELOPE"),
        (0.0, 500.0, "EXPERIMENTAL_OUTSIDE_VALIDATED_ENVELOPE"),
        (0.2, 500.0, "EXPERIMENTAL_OUTSIDE_VALIDATED_ENVELOPE"),
    ],
)
def test_validation_envelope_status(x_he: float, f_irr: float, expected: str) -> None:
    status, _warnings = validation_envelope(
        n_layers=N96,
        alpha=1.0,
        f_int=F_INT,
        f_irr=f_irr,
        gravity=15.0,
        p_bottom=1.0e6,
        p_top=1.0,
        composition="constant_h2",
        opacity_model="analytic_grey_powerlaw",
        x_he=x_he,
    )
    assert status == expected


@pytest.mark.slow
def test_cfg_demo_case_converges() -> None:
    """User cfg_demo defaults (x_he=0.2, f_irr=500) must pass under default budgets."""
    verdict, flatness = _run_production(x_he=0.2, f_irr=500.0)
    assert verdict == "CONVERGED"
    assert flatness <= PHYSICAL_GATE


# Realistic hot-Jupiter corners (see examples/rce/realistic_he_irr_sweep.py).
REALISTIC_CORNER_CASES = (
    (0.35, 2000.0),
    (0.4, 2000.0),
    (0.4, 1500.0),
)


@pytest.mark.parametrize("x_he,f_irr", REALISTIC_CORNER_CASES)
@pytest.mark.slow
def test_realistic_he_irr_corners(x_he: float, f_irr: float) -> None:
    """High He + strong irradiation must not require recovery cycles or blow step budget."""
    run = run_production_rce(
        n_layers=N96,
        alpha=1.0,
        f_int=F_INT,
        f_irr=f_irr,
        x_he=x_he,
        controls=ProductionControls(max_recovery_cycles=2),
    )
    require_topo = abs(f_irr) <= 1.0e-15
    gates = _gates_from_result(
        run.result,
        run.spec,
        gate=PHYSICAL_GATE,
        require_bottom_connected_cz=require_topo,
    )
    accepted = len([d for d in run.result.diagnostics if d.accepted])
    assert gates.convergence_ok and (gates.topology_ok or not require_topo)
    assert gates.flux_flatness <= PHYSICAL_GATE
    assert accepted <= 15, f"unexpected stiffness: {accepted} steps, phases={run.phases}"
    assert "continuation" not in "".join(run.phases)


def test_thermal_he_topology_boundary() -> None:
    """Document physical limit: x_He>=0.815 has no bottom-connected CZ at F_irr=0."""
    ok_run = run_production_rce(
        n_layers=N96, x_he=0.81, f_irr=0.0, controls=ProductionControls(max_recovery_cycles=0)
    )
    bad_run = run_production_rce(
        n_layers=N96, x_he=0.815, f_irr=0.0, controls=ProductionControls(max_recovery_cycles=0)
    )
    ok_g = _gates_from_result(ok_run.result, ok_run.spec, gate=PHYSICAL_GATE, require_bottom_connected_cz=True)
    bad_g = _gates_from_result(
        bad_run.result, bad_run.spec, gate=PHYSICAL_GATE, require_bottom_connected_cz=True
    )
    assert ok_g.convergence_ok and ok_g.topology_ok
    assert bad_g.flux_flatness_ok and bad_g.tendency_ok
    assert not bad_g.topology_ok
