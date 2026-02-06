"""
Mixing-Length Theory: Alpha from Flux Balance

This module implements MLT by solving for the mixing-length parameter α
such that the convective flux F_c(α) balances the required flux F_need.

The key equation:
    F_c(α) = F_need = F_tot - F_rad

where the MLT convective flux is:
    F_c(α) = ρ * c_p * T * S(α) * (∇ - ∇_ad)^(3/2)
    
with:
    S(α) = sqrt(g*δ/(8*H_p)) * (α*H_p)^2

Author: Generated for radiative transfer convection modeling
Date: November 2025
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from typing import Dict, Tuple, Optional
import warnings


def compute_gradient(T_top: float, T_mid: float, T_bot: float,
                    P_top: float, P_mid: float, P_bot: float) -> float:
    """
    Compute the temperature gradient ∇ = d(ln T)/d(ln P) from layer samples.
    
    Uses finite differences on the logarithmic quantities.
    
    Parameters:
        T_top, T_mid, T_bot: Temperatures at top, middle, bottom [K]
        P_top, P_mid, P_bot: Pressures at top, middle, bottom [Pa]
    
    Returns:
        float: Gradient ∇ = d(ln T)/d(ln P)
    """
    # Use central difference at middle point
    ln_T_top = np.log(T_top)
    ln_T_bot = np.log(T_bot)
    ln_P_top = np.log(P_top)
    ln_P_bot = np.log(P_bot)
    
    nabla = (ln_T_bot - ln_T_top) / (ln_P_bot - ln_P_top)
    
    return nabla


def compute_adiabatic_gradient(R: float, c_p: float) -> float:
    """
    Compute the adiabatic gradient ∇_ad = R/c_p.
    
    Parameters:
        R: Gas constant [J/(kg·K)] - note: use R/μ for specific gas constant
        c_p: Specific heat capacity at constant pressure [J/(kg·K)]
    
    Returns:
        float: Adiabatic gradient ∇_ad (dimensionless)
    """
    return R / c_p


def is_superadiabatic(nabla: float, nabla_ad: float, 
                     tolerance: float = 1e-6) -> bool:
    """
    Check if layer is superadiabatic (convectively unstable).
    
    Parameters:
        nabla: Actual temperature gradient ∇
        nabla_ad: Adiabatic gradient ∇_ad
        tolerance: Small tolerance for numerical comparison
    
    Returns:
        bool: True if superadiabatic (∇ > ∇_ad), False otherwise
    """
    return nabla > (nabla_ad + tolerance)


def compute_pressure_scale_height(T: float, P: float, g: float, 
                                  mu: float, R_universal: float) -> float:
    """
    Compute pressure scale height H_p.
    
    For ideal gas: H_p = (R*T) / (μ*g)
    
    Parameters:
        T: Temperature [K]
        P: Pressure [Pa] (not used for ideal gas, but included for generality)
        g: Gravitational acceleration [m/s²]
        mu: Mean molecular weight [kg/mol]
        R_universal: Universal gas constant [J/(mol·K)]
    
    Returns:
        float: Pressure scale height [m]
    """
    H_p = (R_universal * T) / (mu * g)
    return H_p


def compute_S_factor(alpha: float, H_p: float, g: float, delta: float) -> float:
    """
    Compute the S(α) factor in the MLT flux formula.
    
    S(α) = sqrt(g*δ/(8*H_p)) * l^2
         = sqrt(g*δ/(8*H_p)) * (α*H_p)^2
    
    Parameters:
        alpha: Mixing-length parameter (dimensionless)
        H_p: Pressure scale height [m]
        g: Gravitational acceleration [m/s²]
        delta: Thermodynamic parameter δ = -(∂ln ρ/∂ln T)_P (dimensionless)
               For ideal gas: δ = 1
    
    Returns:
        float: S factor [m^(3/2)/s]
    """
    l = alpha * H_p  # mixing length
    S = np.sqrt(g * delta / (8.0 * H_p)) * l**2
    return S


def compute_convective_flux(alpha: float, rho: float, c_p: float, T: float,
                           H_p: float, g: float, delta: float,
                           nabla: float, nabla_ad: float) -> float:
    """
    Compute the convective flux F_c using MLT.
    
    F_c(α) = ρ * c_p * T * S(α) * (∇ - ∇_ad)^(3/2)
    
    Parameters:
        alpha: Mixing-length parameter
        rho: Density [kg/m³]
        c_p: Specific heat capacity [J/(kg·K)]
        T: Temperature [K]
        H_p: Pressure scale height [m]
        g: Gravitational acceleration [m/s²]
        delta: Thermodynamic parameter (δ=1 for ideal gas)
        nabla: Actual temperature gradient ∇
        nabla_ad: Adiabatic gradient ∇_ad
    
    Returns:
        float: Convective flux [W/m²]
    """
    if nabla <= nabla_ad:
        return 0.0  # No convection if not superadiabatic
    
    S = compute_S_factor(alpha, H_p, g, delta)
    delta_nabla = nabla - nabla_ad
    
    F_c = rho * c_p * T * S * (delta_nabla)**(3.0/2.0)
    
    return F_c


def calculate_alpha_from_flux(layer_data: Dict, flux_data: Dict, 
                              physical_params: Dict,
                              alpha_bounds: Tuple[float, float] = (0.001, 10.0),
                              verbose: bool = False) -> Dict:
    """
    Calculate the mixing-length parameter α by balancing convective flux.
    
    Solves: F_c(α) = F_need = F_tot - F_rad
    
    Parameters:
        layer_data: Dictionary with keys:
            - T_top, T_mid, T_bot: Temperatures [K]
            - P_top, P_mid, P_bot: Pressures [Pa]
        
        flux_data: Dictionary with keys:
            - F_tot: Total flux through layer [W/m²]
            - F_rad: Radiative flux [W/m²]
        
        physical_params: Dictionary with keys:
            - g: Gravitational acceleration [m/s²]
            - delta: Thermodynamic parameter (1 for ideal gas)
            - R_universal: Universal gas constant [J/(mol·K)]
            - mu: Mean molecular weight [kg/mol]
            - c_p: Specific heat capacity [J/(kg·K)]
            - rho: Density [kg/m³]
        
        alpha_bounds: (min, max) bounds for α search
        verbose: If True, print detailed information
    
    Returns:
        Dictionary with:
            - alpha: Mixing-length parameter (or None if not superadiabatic)
            - F_c: Convective flux at solution [W/m²]
            - F_need: Required flux [W/m²]
            - nabla: Actual gradient
            - nabla_ad: Adiabatic gradient
            - H_p: Pressure scale height [m]
            - is_superadiabatic: Boolean
            - convergence_info: String describing solution status
    """
    # Extract data
    T_top = layer_data['T_top']
    T_mid = layer_data['T_mid']
    T_bot = layer_data['T_bot']
    P_top = layer_data['P_top']
    P_mid = layer_data['P_mid']
    P_bot = layer_data['P_bot']
    
    F_tot = flux_data['F_tot']
    F_rad = flux_data['F_rad']
    F_need = F_tot - F_rad
    
    g = physical_params['g']
    delta = physical_params['delta']
    R_universal = physical_params['R_universal']
    mu = physical_params['mu']
    c_p = physical_params['c_p']
    rho = physical_params['rho']
    
    # Calculate specific gas constant
    R_specific = R_universal / mu  # J/(kg·K)
    
    # Compute gradients
    nabla = compute_gradient(T_top, T_mid, T_bot, P_top, P_mid, P_bot)
    nabla_ad = compute_adiabatic_gradient(R_specific, c_p)
    
    # Compute pressure scale height (use middle values)
    H_p = compute_pressure_scale_height(T_mid, P_mid, g, mu, R_universal)
    
    # Check if superadiabatic
    superadiabatic = is_superadiabatic(nabla, nabla_ad)
    
    result = {
        'nabla': nabla,
        'nabla_ad': nabla_ad,
        'H_p': H_p,
        'is_superadiabatic': superadiabatic,
        'F_need': F_need,
    }
    
    if verbose:
        print(f"Layer Analysis:")
        print(f"  T: {T_top:.1f} K (top) -> {T_mid:.1f} K (mid) -> {T_bot:.1f} K (bot)")
        print(f"  P: {P_top:.2e} Pa (top) -> {P_mid:.2e} Pa (mid) -> {P_bot:.2e} Pa (bot)")
        print(f"  ∇ = {nabla:.6f}")
        print(f"  ∇_ad = {nabla_ad:.6f}")
        print(f"  H_p = {H_p:.2f} m")
        print(f"  Superadiabatic: {superadiabatic}")
    
    if not superadiabatic:
        result['alpha'] = None
        result['F_c'] = 0.0
        result['convergence_info'] = "Layer is stable (not superadiabatic), no convection needed"
        
        if verbose:
            print(f"  No convection needed (layer is stable)")
        
        return result
    
    if F_need <= 0:
        result['alpha'] = None
        result['F_c'] = 0.0
        result['convergence_info'] = f"F_need = {F_need:.2e} W/m² is non-positive, no convection needed"
        
        if verbose:
            print(f"  F_need = {F_need:.2e} W/m² (non-positive), no convection needed")
        
        return result
    
    # Define objective function for root finding
    def objective(alpha):
        F_c = compute_convective_flux(alpha, rho, c_p, T_mid, H_p, g, delta,
                                      nabla, nabla_ad)
        return F_c - F_need
    
    # Check if bounds are reasonable
    F_c_min = objective(alpha_bounds[0])
    F_c_max = objective(alpha_bounds[1])
    
    if verbose:
        print(f"\nFlux Balance:")
        print(f"  F_tot = {F_tot:.2e} W/m²")
        print(f"  F_rad = {F_rad:.2e} W/m²")
        print(f"  F_need = {F_need:.2e} W/m²")
        print(f"  F_c(α={alpha_bounds[0]}) = {F_c_min + F_need:.2e} W/m²")
        print(f"  F_c(α={alpha_bounds[1]}) = {F_c_max + F_need:.2e} W/m²")
    
    # Check if solution exists in bounds
    if F_c_min > 0:
        # Even minimum alpha gives too much flux
        result['alpha'] = alpha_bounds[0]
        result['F_c'] = F_c_min + F_need
        result['convergence_info'] = f"α at lower bound (even α_min={alpha_bounds[0]} gives F_c > F_need)"
        
        if verbose:
            print(f"\nWarning: Even minimum α gives too much flux")
            print(f"  Using α = {result['alpha']:.4f}")
        
        return result
    
    if F_c_max < 0:
        # Even maximum alpha can't provide enough flux
        result['alpha'] = alpha_bounds[1]
        result['F_c'] = F_c_max + F_need
        result['convergence_info'] = f"α at upper bound (even α_max={alpha_bounds[1]} gives F_c < F_need)"
        
        if verbose:
            print(f"\nWarning: Even maximum α cannot provide enough flux")
            print(f"  Using α = {result['alpha']:.4f}")
            print(f"  F_c achieved = {result['F_c']:.2e} W/m² < F_need = {F_need:.2e} W/m²")
        
        return result
    
    # Root exists in bounds, use Brent's method
    try:
        alpha_solution = brentq(objective, alpha_bounds[0], alpha_bounds[1], 
                               xtol=1e-6, rtol=1e-6)
        F_c_solution = compute_convective_flux(alpha_solution, rho, c_p, T_mid, 
                                              H_p, g, delta, nabla, nabla_ad)
        
        result['alpha'] = alpha_solution
        result['F_c'] = F_c_solution
        result['convergence_info'] = "Successfully converged"
        
        if verbose:
            print(f"\nSolution Found:")
            print(f"  α = {alpha_solution:.4f}")
            print(f"  l = α*H_p = {alpha_solution * H_p:.2f} m")
            print(f"  F_c = {F_c_solution:.2e} W/m²")
            print(f"  Relative error: {abs(F_c_solution - F_need)/F_need * 100:.2e}%")
        
    except Exception as e:
        # Fallback: shouldn't happen if bounds check passed
        result['alpha'] = None
        result['F_c'] = None
        result['convergence_info'] = f"Root finding failed: {str(e)}"
        
        if verbose:
            print(f"\nError: Root finding failed: {str(e)}")
    
    return result


def demo_calculation():
    """
    Demonstration with example values.
    """
    print("=" * 70)
    print("MLT Alpha from Flux Balance - Demo")
    print("=" * 70)
    
    # Example layer (hot Jupiter atmosphere, superadiabatic)
    # Make it clearly superadiabatic: ∇ > ∇_ad ≈ 0.27
    # Using very steep temperature gradient
    layer_data = {
        'T_top': 1000.0,     # K
        'T_mid': 1500.0,     # K
        'T_bot': 2250.0,     # K  (very steep gradient to ensure ∇ > ∇_ad)
        'P_top': 1e4,        # Pa (0.1 bar)
        'P_mid': 5e4,        # Pa (0.5 bar)
        'P_bot': 1e5,        # Pa (1.0 bar)
    }
    
    # Example fluxes
    flux_data = {
        'F_tot': 1e7,        # W/m² (total flux passing through)
        'F_rad': 8e6,        # W/m² (radiative component)
    }
    
    # Physical parameters (hot Jupiter, H2-dominated)
    physical_params = {
        'g': 10.0,           # m/s² (Earth-like)
        'delta': 1.0,        # Ideal gas
        'R_universal': 8.314, # J/(mol·K) - UNIVERSAL gas constant
        'mu': 0.0022,        # kg/mol (H2-rich)
        'c_p': 14000.0,      # J/(kg·K) (H2 specific heat)
        'rho': 0.005,        # kg/m³ (low density, high altitude)
    }
    
    # Calculate alpha
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    if result['alpha'] is not None:
        print(f"Mixing-length parameter: α = {result['alpha']:.4f}")
        print(f"Mixing length: l = {result['alpha'] * result['H_p']:.2f} m")
        print(f"Convective flux: F_c = {result['F_c']:.2e} W/m²")
    else:
        print(f"No solution: {result['convergence_info']}")
    print("=" * 70)


if __name__ == "__main__":
    demo_calculation()

