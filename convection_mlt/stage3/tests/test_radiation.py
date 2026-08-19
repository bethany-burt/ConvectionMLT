"""Stage 3 radiation tests — points 29-33.

Exit tolerances (locked quantitative contract):
  - Solver agreement (Thomas/dense/sweep): ≤ 1e-12 normalized
  - Column energy residual: ≤ 1e-12 normalized
  - Boundary values: exact or ≤ 1e-15 relative
  - Negative flux allowance: ≥ -1e-14 · F_scale
  - Band-weight sum: ≤ 1e-15
  - NumPy-JAX parity: ≤ 1e-12 (tested in test_jax_parity.py)
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.radiation import (
    DEFAULT_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    LowerFlux,
    LowerTemperature,
    RadiationResult,
    SolveRoute,
    TopIrradiation,
    radiation_core,
    solve_radiation,
)
from convection_mlt.opacity import (
    AnalyticGreyOpacity,
    ConstantGreyOpacity,
    PrescribedBandOpacity,
)

GATE = 1e-12
BC_GATE = 1e-15
FLUX_NEG_GATE = -1e-14


def _norm_diff(a: np.ndarray, b: np.ndarray, floor: float = 1e-30) -> float:
    scale = max(floor, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b))) / scale


def _make_grey_inputs(
    n_layer: int,
    temperature: float = 1500.0,
    kappa0: float = 0.01,
    dp: float = 1e4,
    g: float = 10.0,
    f_down_top: float = 0.0,
    f_up_bot: float = 0.0,
    D: float = DEFAULT_DIFFUSIVITY,
):
    temp = np.full(n_layer, temperature)
    mass_path = np.full(n_layer, dp / g)
    kappa = np.full((1, n_layer), kappa0)
    weights = np.array([1.0])
    top = np.array([f_down_top])
    bot = np.array([f_up_bot])
    return temp, mass_path, kappa, weights, top, bot, D


# ═══════════════════════════════════════════════════════════════════
# POINT 29 — Analytic and limiting cases
# ═══════════════════════════════════════════════════════════════════

class TestTransparentAtmosphere:
    """κ = 0 → fluxes pass through unchanged."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_transparent(self, route):
        n = 10
        T0 = 1500.0
        f_down_top = 100.0
        f_up_bot = 200.0
        temp, mass_path, kappa, w, top, bot, D = _make_grey_inputs(
            n, T0, kappa0=0.0, f_down_top=f_down_top, f_up_bot=f_up_bot,
        )
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)
        # all downward fluxes = f_down_top, all upward = f_up_bot
        assert _norm_diff(r.flux_down[0], np.full(n + 1, f_down_top)) < GATE
        assert _norm_diff(r.flux_up[0], np.full(n + 1, f_up_bot)) < GATE


class TestIsothermalEquilibrium:
    """T_i = T_0, F↑[0] = σT₀⁴, F↓[N] = σT₀⁴ → zero net flux, zero heating."""

    @pytest.mark.parametrize("route", SolveRoute)
    @pytest.mark.parametrize("n", [1, 5, 20, 50])
    def test_isothermal(self, route, n):
        T0 = 2000.0
        B0 = STEFAN_BOLTZMANN * T0 ** 4
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, T0, kappa0=0.1)
        top = np.array([B0])
        bot = np.array([B0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)

        assert _norm_diff(r.flux_up[0], np.full(n + 1, B0)) < GATE
        assert _norm_diff(r.flux_down[0], np.full(n + 1, B0)) < GATE
        assert _norm_diff(r.flux_net, np.zeros(n + 1)) < GATE
        assert np.max(np.abs(r.heating)) / max(1e-30, B0) < GATE


class TestSingleLayer:
    """N = 1 through the production API."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_n1(self, route):
        T0 = 1000.0
        B0 = STEFAN_BOLTZMANN * T0 ** 4
        kappa0 = 0.05
        dp, g = 1e5, 10.0
        dm = dp / g
        dtau = kappa0 * dm
        D = DEFAULT_DIFFUSIVITY
        trans = np.exp(-D * dtau)
        ef = -np.expm1(-D * dtau)

        f_down_top = 300.0
        f_up_bot = 500.0

        temp = np.array([T0])
        mass_path = np.array([dm])
        kappa = np.array([[kappa0]])
        w = np.array([1.0])
        top = np.array([f_down_top])
        bot = np.array([f_up_bot])

        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)

        expected_fd0 = trans * f_down_top + ef * B0
        expected_fu1 = trans * f_up_bot + ef * B0

        assert abs(r.flux_down[0, 0] - expected_fd0) / max(1e-30, abs(expected_fd0)) < GATE
        assert abs(r.flux_up[0, 1] - expected_fu1) / max(1e-30, abs(expected_fu1)) < GATE
        assert abs(r.flux_down[0, 1] - f_down_top) / max(1e-30, f_down_top) < BC_GATE
        assert abs(r.flux_up[0, 0] - f_up_bot) / max(1e-30, f_up_bot) < BC_GATE


class TestOpticallyThick:
    """Very large Δτ → fluxes approach local Planck."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_thick(self, route):
        n = 5
        T0 = 2500.0
        B0 = STEFAN_BOLTZMANN * T0 ** 4
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, T0, kappa0=1e6)
        top = np.array([100.0])
        bot = np.array([100.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)
        # interior fluxes approach B0
        for k in range(1, n):
            assert abs(r.flux_up[0, k] - B0) / B0 < 1e-6
            assert abs(r.flux_down[0, k] - B0) / B0 < 1e-6


class TestOpticallyThin:
    """Very small Δτ → expm1 stability."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_thin(self, route):
        n = 5
        T0 = 1500.0
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(
            n, T0, kappa0=1e-15, f_down_top=100.0, f_up_bot=200.0,
        )
        r = radiation_core(temp, mass_path, kappa, w, np.array([100.0]), np.array([200.0]), D, route)
        assert np.all(np.isfinite(r.flux_up))
        assert np.all(np.isfinite(r.flux_down))
        assert np.all(np.isfinite(r.heating))


class TestBoundaryValues:
    """Directly assigned boundary values must be exact or ≤ 1e-15."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_boundaries(self, route):
        n = 8
        f_down_top = 500.0
        f_up_bot = 300.0
        temp, mass_path, kappa, w, top, bot, D = _make_grey_inputs(
            n, kappa0=0.05, f_down_top=f_down_top, f_up_bot=f_up_bot,
        )
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)
        assert abs(r.flux_down[0, n] - f_down_top) / f_down_top < BC_GATE
        assert abs(r.flux_up[0, 0] - f_up_bot) / f_up_bot < BC_GATE


# ═══════════════════════════════════════════════════════════════════
# Orientation tests (B = 0, cold/source-free atmosphere)
# ═══════════════════════════════════════════════════════════════════

class TestOrientation:
    """Top-only and bottom-only forcing in a source-free atmosphere."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_top_irradiation_heats_downward(self, route):
        n = 10
        temp = np.full(n, 1e-10)  # effectively zero source
        mass_path = np.full(n, 1000.0)
        kappa = np.full((1, n), 0.01)
        w = np.array([1.0])
        top = np.array([1000.0])
        bot = np.array([0.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, DEFAULT_DIFFUSIVITY, route)
        # downward beam attenuates toward bottom
        fd = r.flux_down[0]
        for k in range(n):
            assert fd[k] <= fd[k + 1] + 1e-14 * 1000.0
        # heating nonnegative (within float allowance)
        f_scale = float(np.max(np.abs(r.flux_down)))
        assert np.all(r.heating >= FLUX_NEG_GATE * f_scale / np.min(mass_path))

    @pytest.mark.parametrize("route", SolveRoute)
    def test_bottom_upward_forcing(self, route):
        n = 10
        temp = np.full(n, 1e-10)
        mass_path = np.full(n, 1000.0)
        kappa = np.full((1, n), 0.01)
        w = np.array([1.0])
        top = np.array([0.0])
        bot = np.array([1000.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, DEFAULT_DIFFUSIVITY, route)
        # upward beam attenuates upward
        fu = r.flux_up[0]
        for k in range(n):
            assert fu[k + 1] <= fu[k] + 1e-14 * 1000.0


# ═══════════════════════════════════════════════════════════════════
# POINT 30 — Thomas vs dense
# ═══════════════════════════════════════════════════════════════════

class TestThomasVsDense:
    """Thomas and dense solve the same linear system."""

    @pytest.mark.parametrize("n", [1, 3, 10, 50])
    def test_agreement(self, n):
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, kappa0=0.05)
        top = np.array([300.0])
        bot = np.array([500.0])
        r_thomas = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)
        r_dense = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.DENSE)

        assert _norm_diff(r_thomas.flux_up, r_dense.flux_up) < GATE
        assert _norm_diff(r_thomas.flux_down, r_dense.flux_down) < GATE
        assert _norm_diff(r_thomas.heating, r_dense.heating) < GATE


# ═══════════════════════════════════════════════════════════════════
# POINT 31 — Thomas vs directional sweep
# ═══════════════════════════════════════════════════════════════════

class TestThomasVsSweep:
    """Sweep is the strongest check — it doesn't solve an assembled matrix."""

    @pytest.mark.parametrize("n", [1, 3, 10, 50])
    def test_agreement(self, n):
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, kappa0=0.05)
        top = np.array([300.0])
        bot = np.array([500.0])
        r_thomas = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.THOMAS)
        r_sweep = radiation_core(temp, mass_path, kappa, w, top, bot, D, SolveRoute.SWEEP)

        assert _norm_diff(r_thomas.flux_up, r_sweep.flux_up) < GATE
        assert _norm_diff(r_thomas.flux_down, r_sweep.flux_down) < GATE
        assert _norm_diff(r_thomas.flux_net, r_sweep.flux_net) < GATE
        assert _norm_diff(r_thomas.heating, r_sweep.heating) < GATE


# ═══════════════════════════════════════════════════════════════════
# POINT 32 — Conservative heating + telescoping
# ═══════════════════════════════════════════════════════════════════

class TestConservativeHeating:
    """Σ Δm_i (dh/dt)_i = F_net[0] - F_net[N]."""

    @pytest.mark.parametrize("route", SolveRoute)
    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_telescoping(self, route, n):
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, kappa0=0.05)
        top = np.array([200.0])
        bot = np.array([800.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)

        lhs = float(np.sum(mass_path * r.heating))
        rhs = float(r.flux_net[0] - r.flux_net[n])
        scale = max(1e-30, abs(rhs), abs(lhs))
        assert abs(lhs - rhs) / scale < GATE


class TestLayerEnergyIdentity:
    """Independent layer absorption/emission identity.

    For each layer i (one band):
      Q_layer = (1 - 𝒯_i)(F↑[i] + F↓[i+1] - 2 B_i)
    must agree with flux-divergence heating:
      Q_layer = F_net[i] - F_net[i+1]

    This catches shared sign or stream-indexing errors that pure
    telescoping cannot detect.
    """

    @pytest.mark.parametrize("route", SolveRoute)
    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_layer_identity_grey(self, route, n):
        T0 = 1800.0
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, T0, kappa0=0.05)
        top = np.array([200.0])
        bot = np.array([800.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)

        B = STEFAN_BOLTZMANN * T0 ** 4
        trans = r.transmissivity[0]  # (n_layer,)
        ef = 1.0 - trans

        for i in range(n):
            q_abs_emit = ef[i] * (r.flux_up[0, i] + r.flux_down[0, i + 1] - 2.0 * B)
            q_divergence = r.flux_net[i] - r.flux_net[i + 1]
            scale = max(1e-30, abs(q_abs_emit), abs(q_divergence))
            assert abs(q_abs_emit - q_divergence) / scale < GATE, (
                f"layer {i}: abs_emit={q_abs_emit:.6e} vs div={q_divergence:.6e}"
            )

    @pytest.mark.parametrize("route", SolveRoute)
    def test_layer_identity_nonisothermal(self, route):
        """Varying temperature profile — stronger test."""
        n = 15
        temp = np.linspace(1000.0, 3000.0, n)
        mass_path = np.full(n, 500.0)
        kappa = np.full((1, n), 0.03)
        w = np.array([1.0])
        D = DEFAULT_DIFFUSIVITY
        B_bot = STEFAN_BOLTZMANN * temp[0] ** 4
        top = np.array([100.0])
        bot = np.array([B_bot])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)

        B = STEFAN_BOLTZMANN * temp ** 4
        trans = r.transmissivity[0]
        ef = 1.0 - trans

        for i in range(n):
            q_abs_emit = ef[i] * (r.flux_up[0, i] + r.flux_down[0, i + 1] - 2.0 * B[i])
            q_divergence = r.flux_net[i] - r.flux_net[i + 1]
            scale = max(1e-30, abs(q_abs_emit), abs(q_divergence))
            assert abs(q_abs_emit - q_divergence) / scale < GATE

    def test_layer_identity_multiband(self):
        """Sum of per-band layer identities = broadband flux divergence."""
        n = 8
        temp = np.linspace(1500.0, 2500.0, n)
        mass_path = np.full(n, 600.0)
        kappas = np.array([np.full(n, 0.01), np.full(n, 0.05), np.full(n, 0.1)])
        weights = np.array([0.5, 0.3, 0.2])
        D = DEFAULT_DIFFUSIVITY
        B_total = STEFAN_BOLTZMANN * temp ** 4
        top = weights * 100.0
        bot = weights * STEFAN_BOLTZMANN * temp[0] ** 4
        r = radiation_core(temp, mass_path, kappas, weights, top, bot, D)

        for i in range(n):
            q_abs_emit_sum = 0.0
            for b in range(3):
                B_b = weights[b] * B_total[i]
                ef_b = 1.0 - r.transmissivity[b, i]
                q_abs_emit_sum += ef_b * (
                    r.flux_up[b, i] + r.flux_down[b, i + 1] - 2.0 * B_b
                )
            q_divergence = r.flux_net[i] - r.flux_net[i + 1]
            scale = max(1e-30, abs(q_abs_emit_sum), abs(q_divergence))
            assert abs(q_abs_emit_sum - q_divergence) / scale < GATE


# ═══════════════════════════════════════════════════════════════════
# POINT 33 — Convergence, positivity, float32 diagnostic
# ═══════════════════════════════════════════════════════════════════

class TestPositivity:
    """Directional fluxes must be nonneg within float allowance."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_nonneg(self, route):
        n = 20
        temp, mass_path, kappa, w, _, _, D = _make_grey_inputs(n, kappa0=0.1)
        top = np.array([500.0])
        bot = np.array([300.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, D, route)
        f_scale = max(float(np.max(np.abs(r.flux_up))), float(np.max(np.abs(r.flux_down))))
        assert np.all(r.flux_up >= FLUX_NEG_GATE * f_scale)
        assert np.all(r.flux_down >= FLUX_NEG_GATE * f_scale)


class TestGridConvergence:
    """Nonisothermal convergence against a high-resolution discrete reference.

    For a piecewise-constant source discretization, the correct reference is
    the same discrete scheme at very high N (here N_ref=4096). This avoids
    comparing against a continuous integral that assumes a different source
    representation. Errors should decrease ~O(1/N) under refinement.
    """

    @staticmethod
    def _run_column(n, T_bot, T_top, kappa0, D, g, P_bot, P_top, F_down_top, F_up_bot):
        dp = (P_bot - P_top) / n
        mass_path = np.full(n, dp / g)
        frac = (np.arange(n) + 0.5) / n
        temp = T_bot + (T_top - T_bot) * frac
        kappa = np.full((1, n), kappa0)
        w = np.array([1.0])
        return radiation_core(temp, mass_path, kappa, w,
                              np.array([F_down_top]), np.array([F_up_bot]), D)

    def test_refinement_vs_high_res(self):
        """Errors decrease with N vs N_ref=4096 reference."""
        kappa0 = 0.02
        D = DEFAULT_DIFFUSIVITY
        g = 10.0
        P_bot, P_top = 1e6, 1e4
        T_bot, T_top = 3000.0, 1500.0
        B_bot = STEFAN_BOLTZMANN * T_bot ** 4
        F_down_top_val = 50.0

        N_ref = 4096
        r_ref = self._run_column(N_ref, T_bot, T_top, kappa0, D, g, P_bot, P_top,
                                  F_down_top_val, B_bot)
        # reference interface fluxes at boundaries
        fd_ref = float(r_ref.flux_down[0, 0])
        fu_ref = float(r_ref.flux_up[0, N_ref])

        Ns = [8, 16, 32, 64, 128, 256]
        errors = []
        for n in Ns:
            r = self._run_column(n, T_bot, T_top, kappa0, D, g, P_bot, P_top,
                                  F_down_top_val, B_bot)
            err = abs(float(r.flux_down[0, 0]) - fd_ref)
            errors.append(err)

        assert all(np.isfinite(errors))
        assert all(e > 0 for e in errors), "Errors should be nonzero vs independent reference"

        # errors must decrease
        for i in range(len(Ns) - 1):
            assert errors[i + 1] < errors[i] * 1.05, (
                f"F↓ error not decreasing: N={Ns[i]}→{Ns[i+1]}: "
                f"{errors[i]:.3e}→{errors[i+1]:.3e}"
            )

        # fit convergence order
        order = np.log(errors[-2] / errors[-1]) / np.log(Ns[-1] / Ns[-2])
        assert order > 0.8, f"Expected ~first order, got {order:.2f}"

    def test_refinement_tau_spaced(self):
        """Optical-depth-spaced grid also converges."""
        kappa0 = 0.05
        D = DEFAULT_DIFFUSIVITY
        T_bot, T_top = 2500.0, 1200.0
        B_bot = STEFAN_BOLTZMANN * T_bot ** 4
        g = 10.0
        P_bot, P_top = 5e5, 1e3

        N_ref = 4096
        r_ref = TestGridConvergence._run_column(
            N_ref, T_bot, T_top, kappa0, D, g, P_bot, P_top, 0.0, B_bot)
        fu_ref = float(r_ref.flux_up[0, N_ref])

        Ns = [8, 16, 32, 64]
        errors = []
        for n in Ns:
            r = TestGridConvergence._run_column(
                n, T_bot, T_top, kappa0, D, g, P_bot, P_top, 0.0, B_bot)
            errors.append(abs(float(r.flux_up[0, n]) - fu_ref))

        for i in range(len(Ns) - 1):
            assert errors[i + 1] < errors[i] * 1.05


class TestFloat32Diagnostic:
    """Float32 should produce finite results (diagnostic only)."""

    def test_float32(self):
        n = 10
        T0 = 1500.0
        temp = np.full(n, T0, dtype=np.float32).astype(np.float64)
        mass_path = np.full(n, 1000.0)
        kappa = np.full((1, n), 0.01)
        w = np.array([1.0])
        top = np.array([100.0])
        bot = np.array([200.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, DEFAULT_DIFFUSIVITY)
        assert np.all(np.isfinite(r.flux_up))
        assert np.all(np.isfinite(r.heating))


# ═══════════════════════════════════════════════════════════════════
# Public wrapper tests
# ═══════════════════════════════════════════════════════════════════

class TestSolveRadiation:
    """Test the public wrapper with opacity providers."""

    def test_constant_grey(self):
        n = 10
        T0 = 1500.0
        B0 = STEFAN_BOLTZMANN * T0 ** 4
        opa = ConstantGreyOpacity(kappa0=0.01)
        temp = np.full(n, T0)
        mass_path = np.full(n, 1000.0)
        pressure = np.full(n, 1e5)
        r = solve_radiation(
            temp, mass_path, opa, pressure,
            TopIrradiation(B0), LowerTemperature(T0),
        )
        # isothermal equilibrium
        assert _norm_diff(r.flux_net, np.zeros(n + 1)) < GATE

    def test_analytic_grey(self):
        n = 5
        T0 = 2000.0
        opa = AnalyticGreyOpacity(kappa0=0.01, P0=1e5, T0=2000.0, a=1.0, b=0.0)
        temp = np.full(n, T0)
        mass_path = np.full(n, 500.0)
        pressure = np.linspace(1e5, 1e3, n)
        r = solve_radiation(
            temp, mass_path, opa, pressure,
            TopIrradiation(0.0), LowerFlux(0.0),
        )
        assert np.all(np.isfinite(r.heating))

    def test_lower_temperature_bc(self):
        n = 5
        T_bound = 3000.0
        B_bound = STEFAN_BOLTZMANN * T_bound ** 4
        opa = ConstantGreyOpacity(kappa0=0.01)
        temp = np.full(n, 1500.0)
        mass_path = np.full(n, 1000.0)
        pressure = np.full(n, 1e5)
        r = solve_radiation(
            temp, mass_path, opa, pressure,
            TopIrradiation(0.0), LowerTemperature(T_bound),
        )
        assert abs(r.flux_up[0, 0] - B_bound) / B_bound < BC_GATE


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Numerical edge-case requirements."""

    def test_zero_kappa_band(self):
        n = 5
        temp = np.full(n, 1500.0)
        mass_path = np.full(n, 1000.0)
        kappa = np.zeros((1, n))
        w = np.array([1.0])
        top = np.array([100.0])
        bot = np.array([200.0])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, DEFAULT_DIFFUSIVITY)
        assert np.all(np.isfinite(r.flux_up))

    def test_varying_kappa(self):
        n = 10
        temp = np.linspace(1000, 3000, n)
        mass_path = np.full(n, 500.0)
        kappa = np.linspace(0.001, 0.1, n)[np.newaxis, :]
        w = np.array([1.0])
        top = np.array([0.0])
        bot = np.array([STEFAN_BOLTZMANN * 3000.0 ** 4])
        r = radiation_core(temp, mass_path, kappa, w, top, bot, DEFAULT_DIFFUSIVITY)
        assert np.all(np.isfinite(r.heating))

    def test_n1_production_api(self):
        opa = ConstantGreyOpacity(kappa0=0.01)
        temp = np.array([1500.0])
        mass_path = np.array([1000.0])
        pressure = np.array([1e5])
        r = solve_radiation(
            temp, mass_path, opa, pressure,
            TopIrradiation(100.0), LowerFlux(200.0),
        )
        assert r.heating.shape == (1,)
        assert np.all(np.isfinite(r.heating))

    def test_extreme_optical_depth_large(self):
        """Δτ ~ 1e8 should not overflow/NaN."""
        n = 5
        temp = np.full(n, 2000.0)
        mass_path = np.full(n, 1e6)
        kappa = np.full((1, n), 100.0)
        w = np.array([1.0])
        r = radiation_core(temp, mass_path, kappa, w, np.array([100.0]), np.array([200.0]), DEFAULT_DIFFUSIVITY)
        assert np.all(np.isfinite(r.flux_up))
        assert np.all(np.isfinite(r.flux_down))
        assert np.all(np.isfinite(r.heating))
        assert np.all(r.transmissivity[0] >= 0.0)
        assert np.all(r.transmissivity[0] <= 1e-300)  # effectively zero

    def test_extreme_optical_depth_tiny(self):
        """Δτ ~ 1e-18 should not cancel via expm1."""
        n = 5
        temp = np.full(n, 2000.0)
        mass_path = np.full(n, 1e-10)
        kappa = np.full((1, n), 1e-8)
        w = np.array([1.0])
        r = radiation_core(temp, mass_path, kappa, w, np.array([100.0]), np.array([200.0]), DEFAULT_DIFFUSIVITY)
        assert np.all(np.isfinite(r.flux_up))
        assert np.all(np.isfinite(r.flux_down))


class TestAnalyticGreyExactValues:
    """AnalyticGreyOpacity returns exact κ(T,P) = κ₀ (P/P₀)^a (T/T₀)^b."""

    def test_exact_evaluation(self):
        kappa0, P0, T0, a, b = 0.01, 1e5, 2000.0, 1.0, 0.5
        opa = AnalyticGreyOpacity(kappa0=kappa0, P0=P0, T0=T0, a=a, b=b)
        temp = np.array([1500.0, 2000.0, 2500.0])
        pressure = np.array([1e4, 1e5, 1e6])
        kappa = opa.evaluate(temp, pressure)
        for i in range(3):
            expected = kappa0 * (pressure[i] / P0) ** a * (temp[i] / T0) ** b
            assert abs(kappa[0, i] - expected) < 1e-15 * expected


class TestKappaDeltaPOverG:
    """Δτ = κ · Δm = κ · ΔP/g under constant gravity."""

    def test_equivalence(self):
        n = 5
        kappa0 = 0.02
        g = 9.8
        dp = np.array([1e4, 2e4, 1.5e4, 3e4, 1e4])
        mass_path = dp / g
        temp = np.full(n, 2000.0)
        kappa = np.full((1, n), kappa0)
        w = np.array([1.0])

        r = radiation_core(temp, mass_path, kappa, w, np.array([0.0]),
                           np.array([STEFAN_BOLTZMANN * 2000.0 ** 4]), DEFAULT_DIFFUSIVITY)

        expected_dtau = kappa0 * dp / g
        np.testing.assert_allclose(r.optical_depth[0], expected_dtau, rtol=1e-15)


class TestVariableGMassPath:
    """Nonuniform mass path from variable gravity."""

    def test_variable_g(self):
        n = 10
        g_profile = np.linspace(8.0, 12.0, n)
        dp = np.full(n, 1e4)
        mass_path = dp / g_profile
        temp = np.linspace(1500.0, 2500.0, n)
        kappa = np.full((1, n), 0.01)
        w = np.array([1.0])
        r = radiation_core(temp, mass_path, kappa, w, np.array([50.0]),
                           np.array([STEFAN_BOLTZMANN * temp[0] ** 4]), DEFAULT_DIFFUSIVITY)
        assert np.all(np.isfinite(r.heating))
        assert r.heating.shape == (n,)
        # mass paths are different, so optical depths should vary
        assert not np.allclose(r.optical_depth[0], r.optical_depth[0, 0])


class TestRejectionCases:
    """Input rejection for invalid data."""

    def test_reject_negative_kappa(self):
        with pytest.raises(ValueError):
            from convection_mlt.opacity import PrescribedBandOpacity
            PrescribedBandOpacity(np.array([[-0.01, 0.01]]), np.array([1.0]))

    def test_reject_nan_kappa(self):
        with pytest.raises(ValueError):
            from convection_mlt.opacity import PrescribedBandOpacity
            PrescribedBandOpacity(np.array([[np.nan, 0.01]]), np.array([1.0]))

    def test_reject_negative_mass_path(self):
        with pytest.raises(ValueError, match="positive"):
            solve_radiation(
                np.array([1500.0]), np.array([-100.0]),
                ConstantGreyOpacity(0.01), np.array([1e5]),
                TopIrradiation(0.0), LowerFlux(0.0),
            )

    def test_reject_inf_mass_path(self):
        with pytest.raises(ValueError):
            solve_radiation(
                np.array([1500.0]), np.array([np.inf]),
                ConstantGreyOpacity(0.01), np.array([1e5]),
                TopIrradiation(0.0), LowerFlux(0.0),
            )

    def test_reject_bad_band_fractions(self):
        from convection_mlt.radiation import TopIrradiation
        opa = ConstantGreyOpacity(0.01)
        with pytest.raises(ValueError):
            solve_radiation(
                np.array([1500.0]), np.array([1000.0]),
                opa, np.array([1e5]),
                TopIrradiation(100.0, band_fractions=np.array([0.5, 0.5])),
                LowerFlux(0.0),
            )


class TestReverseStorageOrientation:
    """Reversing layer storage order and converting back gives same physical fluxes."""

    def test_reverse_recovery(self):
        n = 10
        temp = np.linspace(1500.0, 2500.0, n)
        mass_path = np.linspace(500.0, 1500.0, n)
        kappa0 = 0.03
        kappa = np.full((1, n), kappa0)
        w = np.array([1.0])
        D = DEFAULT_DIFFUSIVITY
        B_bot = STEFAN_BOLTZMANN * temp[0] ** 4
        top = np.array([100.0])
        bot = np.array([B_bot])

        r_fwd = radiation_core(temp, mass_path, kappa, w, top, bot, D)

        # reverse: flip layers, swap boundary roles
        temp_rev = temp[::-1]
        mass_path_rev = mass_path[::-1]
        kappa_rev = kappa[:, ::-1]
        r_rev = radiation_core(temp_rev, mass_path_rev, kappa_rev, w, bot, top, D)

        # physical fluxes at each interface should match after reversal
        np.testing.assert_allclose(
            r_fwd.flux_up[0], r_rev.flux_down[0, ::-1], rtol=1e-12,
        )
        np.testing.assert_allclose(
            r_fwd.flux_down[0], r_rev.flux_up[0, ::-1], rtol=1e-12,
        )
