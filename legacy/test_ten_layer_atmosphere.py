"""
Simple 10-layer atmosphere test for MLT-based convective adjustment.

Workflow
--------
1. Define a 10-layer T/P profile that gradually cools with altitude.
2. Compute radiative fluxes using a gray-atmosphere diffusion approximation.
3. Assume net flux is zero (steady radiative equilibrium) so the convective flux
   must cancel the radiative flux: F_conv = F_tot - F_rad with F_tot=0.
4. For each layer, solve for the mixing-length parameter alpha that produces
   the required convective flux using `calculate_alpha_from_flux`.
5. Flag whether the layer is approximately adiabatic (∇ ≈ ∇_ad).

The script prints a summary table for all layers and can be used as a starting
point for integrating the alpha-from-flux logic into a full RT model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from mlt_flux_balance import calculate_alpha_from_flux

SIGMA_SB = 5.670374419e-8  # W m^-2 K^-4


@dataclass
class Layer:
    index: int
    P_top: float
    P_mid: float
    P_bot: float
    T_top: float
    T_mid: float
    T_bot: float
    rho_mid: float
    delta_z: float


@dataclass
class LayerResult:
    layer: Layer
    F_rad: float
    F_conv: float
    alpha: float | None
    nabla: float
    nabla_ad: float
    is_adiabatic: bool
    convergence_info: str


def setup_atmosphere_layers(
    num_layers: int = 10,
    P_bottom: float = 1e5,
    P_top: float = 1e3,
    T_bottom: float = 1700.0,
    nabla_target: float = 0.33,
    physical_params: Dict[str, float] | None = None,
) -> List[Layer]:
    """
    Build a simple monotonic T/P profile with logarithmically spaced pressures.

    Temperatures follow T ∝ P^nabla_target which guarantees a constant
    d(ln T)/d(ln P). Choosing nabla_target slightly above the adiabatic gradient
    produces a mildly superadiabatic profile for testing.
    """
    if physical_params is None:
        raise ValueError("physical_params must be provided")

    g = physical_params["g"]
    mu = physical_params["mu"]
    R_universal = physical_params["R_universal"]

    pressures = np.logspace(np.log10(P_bottom), np.log10(P_top), num_layers + 1)
    temperatures = T_bottom * (pressures / P_bottom) ** nabla_target

    layers: List[Layer] = []
    for i in range(num_layers):
        P_bot = pressures[i]
        P_top_i = pressures[i + 1]

        P_mid = np.sqrt(P_bot * P_top_i)
        T_bot = temperatures[i]
        T_top = temperatures[i + 1]
        T_mid = T_bottom * (P_mid / P_bottom) ** nabla_target

        rho_mid = (P_mid * mu) / (R_universal * T_mid)
        delta_z = abs((P_bot - P_top_i) / (rho_mid * g))

        layers.append(
            Layer(
                index=i,
                P_top=P_top_i,
                P_mid=P_mid,
                P_bot=P_bot,
                T_top=T_top,
                T_mid=T_mid,
                T_bot=T_bot,
                rho_mid=rho_mid,
                delta_z=delta_z,
            )
        )

    return layers


def compute_opacity(T: float, P: float, opacity_params: Dict[str, float]) -> float:
    """Power-law opacity parameterisation."""
    k0 = opacity_params.get("kappa0", 0.01)
    T_ref = opacity_params.get("T_ref", 1000.0)
    P_ref = opacity_params.get("P_ref", 1e5)
    a = opacity_params.get("a", 0.5)
    b = opacity_params.get("b", 0.5)
    return k0 * (T / T_ref) ** a * (P / P_ref) ** b


def calculate_radiative_flux(layer: Layer, opacity_params: Dict[str, float]) -> float:
    """
    Gray-atmosphere diffusion approximation for the net radiative flux.

    Positive flux corresponds to downward energy transport (because dT/dz < 0),
    so for a cooling-with-height profile F_rad is negative and convection must
    carry the opposite (upward) flux to keep the net zero.
    """
    kappa = compute_opacity(layer.T_mid, layer.P_mid, opacity_params)
    if kappa <= 0:
        raise ValueError("Opacity must be positive")

    dT4_dz = (layer.T_top ** 4 - layer.T_bot ** 4) / layer.delta_z
    F_rad = (4.0 * SIGMA_SB / (3.0 * kappa * layer.rho_mid)) * dT4_dz
    return F_rad


def solve_alpha_for_layers(
    layers: Sequence[Layer],
    physical_params: Dict[str, float],
    opacity_params: Dict[str, float],
    total_flux: float = 0.0,
    adiabat_tolerance: float = 5e-3,
) -> List[LayerResult]:
    results: List[LayerResult] = []

    for layer in layers:
        F_rad = calculate_radiative_flux(layer, opacity_params)
        F_conv = total_flux - F_rad

        layer_data = {
            "T_top": layer.T_top,
            "T_mid": layer.T_mid,
            "T_bot": layer.T_bot,
            "P_top": layer.P_top,
            "P_mid": layer.P_mid,
            "P_bot": layer.P_bot,
        }
        flux_data = {"F_tot": total_flux, "F_rad": F_rad}

        layer_physical_params = dict(physical_params)
        layer_physical_params["rho"] = layer.rho_mid

        mlt_result = calculate_alpha_from_flux(
            layer_data,
            flux_data,
            layer_physical_params,
            verbose=False,
        )

        nabla = mlt_result["nabla"]
        nabla_ad = mlt_result["nabla_ad"]
        is_adiabatic = abs(nabla - nabla_ad) < adiabat_tolerance

        results.append(
            LayerResult(
                layer=layer,
                F_rad=F_rad,
                F_conv=F_conv,
                alpha=mlt_result.get("alpha"),
                nabla=nabla,
                nabla_ad=nabla_ad,
                is_adiabatic=is_adiabatic,
                convergence_info=mlt_result.get("convergence_info", ""),
            )
        )

    return results


def verify_radiative_equilibrium(
    results: Sequence[LayerResult],
    total_flux: float,
    tolerance: float = 1e-6,
) -> bool:
    """
    Confirm that each layer satisfies F_tot = F_rad + F_conv within tolerance.
    """
    max_diff = 0.0
    for res in results:
        net = res.F_rad + res.F_conv - total_flux
        max_diff = max(max_diff, abs(net))
    within_tol = max_diff < tolerance
    status = "OK" if within_tol else "NOT MET"
    print(f"Radiative equilibrium check: {status} (max |F_tot - F_rad - F_conv| = {max_diff:.3e} W/m^2)")
    return within_tol


def print_summary(results: Sequence[LayerResult]) -> None:
    header = (
        f"{'Layer':>5} {'P_mid (bar)':>12} {'T_mid (K)':>10} "
        f"{'F_rad (W/m2)':>14} {'F_conv (W/m2)':>15} {'alpha':>8} "
        f"{'∇':>6} {'∇_ad':>6} {'Adiabat?':>10}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        layer = res.layer
        P_bar = layer.P_mid / 1e5
        alpha_display = f"{res.alpha:.3f}" if res.alpha is not None else "None"
        print(
            f"{layer.index:5d} {P_bar:12.4f} {layer.T_mid:10.1f} "
            f"{res.F_rad:14.3e} {res.F_conv:15.3e} {alpha_display:>8} "
            f"{res.nabla:6.3f} {res.nabla_ad:6.3f} {str(res.is_adiabatic):>10}"
        )


def main() -> None:
    physical_params = {
        "g": 10.0,
        "delta": 1.0,
        "R_universal": 8.314,
        "mu": 0.0022,
        "c_p": 14000.0,
    }
    opacity_params = {
        "kappa0": 0.01,
        "T_ref": 1000.0,
        "P_ref": 1e5,
        "a": 0.5,
        "b": 0.5,
    }

    layers = setup_atmosphere_layers(physical_params=physical_params)
    results = solve_alpha_for_layers(
        layers,
        physical_params=physical_params,
        opacity_params=opacity_params,
        total_flux=0.0,  # net flux zero assumption
        adiabat_tolerance=5e-3,
    )
    verify_radiative_equilibrium(results, total_flux=0.0)
    print_summary(results)


if __name__ == "__main__":
    main()

