"""Stage 3 multiband tests — 3-band manufactured fixture.

Covers: summation, batching, one-band recovery, zero-weight bands, zero-κ bands.
"""

from __future__ import annotations

import numpy as np
import pytest

from convection_mlt.radiation import (
    DEFAULT_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    LowerFlux,
    LowerTemperature,
    SolveRoute,
    TopIrradiation,
    radiation_core,
    solve_radiation,
)
from convection_mlt.opacity import PrescribedBandOpacity

GATE = 1e-12
BC_GATE = 1e-15


def _norm_diff(a, b, floor=1e-30):
    scale = max(floor, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b))) / scale


def _three_band_fixture(n: int = 10):
    """Manufactured 3-band fixture with one zero-weight band."""
    T0 = 2000.0
    temp = np.full(n, T0)
    mass_path = np.full(n, 1000.0)
    kappas = np.array([
        np.full(n, 0.01),  # band 0: moderate opacity
        np.full(n, 0.1),   # band 1: higher opacity
        np.full(n, 0.05),  # band 2: zero weight
    ])
    weights = np.array([0.6, 0.4, 0.0])
    pressure = np.full(n, 1e5)
    return temp, mass_path, kappas, weights, pressure


class TestOneBandRecovery:
    """Multiband with n_band=1 must match grey."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_recovery(self, route):
        n = 10
        T0 = 1500.0
        kappa0 = 0.02
        temp = np.full(n, T0)
        mass_path = np.full(n, 500.0)
        kappa_grey = np.full((1, n), kappa0)
        w_grey = np.array([1.0])
        top = np.array([100.0])
        bot = np.array([200.0])
        D = DEFAULT_DIFFUSIVITY

        r_grey = radiation_core(temp, mass_path, kappa_grey, w_grey, top, bot, D, route)

        kappa_band = np.full((1, n), kappa0)
        w_band = np.array([1.0])
        r_band = radiation_core(temp, mass_path, kappa_band, w_band, top, bot, D, route)

        assert _norm_diff(r_grey.flux_up, r_band.flux_up) < GATE
        assert _norm_diff(r_grey.flux_down, r_band.flux_down) < GATE
        assert _norm_diff(r_grey.heating, r_band.heating) < GATE


class TestThreeBandSummation:
    """Broadband flux = Σ_b band flux (no second weighting)."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_summation(self, route):
        temp, mass_path, kappas, weights, _ = _three_band_fixture()
        n = temp.shape[0]
        D = DEFAULT_DIFFUSIVITY
        B0 = STEFAN_BOLTZMANN * temp[0] ** 4
        top = weights * 100.0
        bot = weights * B0
        r = radiation_core(temp, mass_path, kappas, weights, top, bot, D, route)
        computed_net = np.sum(r.flux_net_band, axis=0)
        assert _norm_diff(computed_net, r.flux_net) < GATE


class TestZeroWeightBand:
    """A zero-weight band contributes nothing."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_zero_weight(self, route):
        temp, mass_path, kappas, weights, _ = _three_band_fixture()
        n = temp.shape[0]
        D = DEFAULT_DIFFUSIVITY
        B0 = STEFAN_BOLTZMANN * temp[0] ** 4
        top = weights * 100.0
        bot = weights * B0
        r = radiation_core(temp, mass_path, kappas, weights, top, bot, D, route)

        # band 2 (zero weight): zero source, zero BC allocation → zero fluxes
        assert np.max(np.abs(r.flux_up[2])) < 1e-20
        assert np.max(np.abs(r.flux_down[2])) < 1e-20
        assert np.max(np.abs(r.flux_net_band[2])) < 1e-20


class TestZeroOpacityBand:
    """A band with κ=0 is transparent."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_zero_kappa(self, route):
        n = 5
        temp = np.full(n, 1500.0)
        mass_path = np.full(n, 1000.0)
        kappas = np.array([np.full(n, 0.0), np.full(n, 0.05)])
        weights = np.array([0.5, 0.5])
        top = np.array([50.0, 50.0])
        bot = np.array([100.0, 100.0])
        D = DEFAULT_DIFFUSIVITY
        r = radiation_core(temp, mass_path, kappas, weights, top, bot, D, route)
        # band 0 (zero opacity): fluxes pass through
        assert _norm_diff(r.flux_down[0], np.full(n + 1, 50.0), floor=50.0) < GATE
        assert _norm_diff(r.flux_up[0], np.full(n + 1, 100.0), floor=100.0) < GATE


class TestThreeBandSolverAgreement:
    """All three routes agree on multiband."""

    def test_three_routes(self):
        temp, mass_path, kappas, weights, _ = _three_band_fixture()
        n = temp.shape[0]
        D = DEFAULT_DIFFUSIVITY
        B0 = STEFAN_BOLTZMANN * temp[0] ** 4
        top = weights * 100.0
        bot = weights * B0
        results = {}
        for route in SolveRoute:
            results[route] = radiation_core(temp, mass_path, kappas, weights, top, bot, D, route)

        for r in [SolveRoute.DENSE, SolveRoute.SWEEP]:
            assert _norm_diff(results[SolveRoute.THOMAS].flux_up, results[r].flux_up) < GATE
            assert _norm_diff(results[SolveRoute.THOMAS].flux_down, results[r].flux_down) < GATE
            assert _norm_diff(results[SolveRoute.THOMAS].heating, results[r].heating) < GATE


class TestThreeBandConservation:
    """Telescoping identity holds for multiband."""

    @pytest.mark.parametrize("route", SolveRoute)
    def test_telescoping(self, route):
        temp, mass_path, kappas, weights, _ = _three_band_fixture()
        n = temp.shape[0]
        D = DEFAULT_DIFFUSIVITY
        B0 = STEFAN_BOLTZMANN * temp[0] ** 4
        top = weights * 100.0
        bot = weights * B0
        r = radiation_core(temp, mass_path, kappas, weights, top, bot, D, route)
        lhs = float(np.sum(mass_path * r.heating))
        rhs = float(r.flux_net[0] - r.flux_net[n])
        scale = max(1e-30, abs(rhs), abs(lhs))
        assert abs(lhs - rhs) / scale < GATE


class TestPrescribedBandOpacityWrapper:
    """Test the public wrapper with PrescribedBandOpacity."""

    def test_wrapper(self):
        n = 5
        temp = np.full(n, 2000.0)
        mass_path = np.full(n, 1000.0)
        pressure = np.full(n, 1e5)
        kappas = np.array([np.full(n, 0.01), np.full(n, 0.05), np.full(n, 0.1)])
        weights = np.array([0.5, 0.3, 0.2])
        opa = PrescribedBandOpacity(kappas, weights)
        r = solve_radiation(
            temp, mass_path, opa, pressure,
            TopIrradiation(100.0), LowerFlux(500.0),
        )
        assert r.heating.shape == (n,)
        assert np.all(np.isfinite(r.heating))


class TestBandWeightValidation:
    """Band weight constraints."""

    def test_sum_to_one(self):
        w = np.array([0.6, 0.4, 0.0])
        assert abs(np.sum(w) - 1.0) < 1e-15

    def test_reject_negative(self):
        with pytest.raises(ValueError, match=">="):
            PrescribedBandOpacity(
                np.ones((2, 5)), np.array([-0.5, 1.5]),
            )

    def test_reject_all_zero(self):
        with pytest.raises(ValueError, match="positive"):
            PrescribedBandOpacity(
                np.ones((2, 5)), np.array([0.0, 0.0]),
            )
