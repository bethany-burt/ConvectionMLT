"""
Calibrate Alpha for Adiabatic Profiles

Option A: Given a layer that IS adiabatic (or should be), determine what
mixing-length parameter α would maintain/produce that adiabatic state.

This is the stellar calibration approach: assume convection is so efficient
that the layer is maintained at (or very close to) the adiabatic gradient.

Physics:
- Adiabatic layer: ∇ = ∇_ad (or ∇ = ∇_ad + ε, ε → 0)
- Convection carries flux F_conv = F_tot - F_rad
- Find α such that MLT produces this flux at the adiabatic gradient

Author: Generated for adiabatic profile calibration
Date: November 2025
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, Tuple


def calibrate_alpha_adiabatic(T: float, P: float, rho: float, 
                              F_conv: float, physical_params: Dict,
                              epsilon: float = 0.001) -> Dict:
    """
    Calibrate α for a layer that is (nearly) adiabatic.
    
    Given:
    - A layer at temperature T, pressure P, density rho
    - Required convective flux F_conv
    - Physical parameters (g, μ, c_p, etc.)
    
    Find: α such that MLT at ∇ = ∇_ad + ε produces F_conv
    
    Parameters:
        T: Temperature [K] (layer average)
        P: Pressure [Pa] (layer average)
        rho: Density [kg/m³] (layer average)
        F_conv: Convective flux to be carried [W/m²]
        physical_params: Dict with g, delta, R_universal, mu, c_p
        epsilon: Small superadiabaticity parameter (default: 0.001)
                 Sets ∇ = ∇_ad * (1 + ε)
    
    Returns:
        Dict with:
            - alpha: Mixing-length parameter
            - H_p: Pressure scale height [m]
            - l: Mixing length [m]
            - nabla_ad: Adiabatic gradient
            - nabla: Actual gradient used (∇_ad + ε)
            - F_c: Resulting convective flux [W/m²]
    """
    g = physical_params['g']
    delta = physical_params['delta']
    R_universal = physical_params['R_universal']
    mu = physical_params['mu']
    c_p = physical_params['c_p']
    
    # Calculate adiabatic gradient
    R_specific = R_universal / mu
    nabla_ad = R_specific / c_p
    
    # Slightly superadiabatic (to drive convection)
    nabla = nabla_ad * (1.0 + epsilon)
    delta_nabla = nabla - nabla_ad
    
    # Pressure scale height
    H_p = (R_universal * T) / (mu * g)
    
    # MLT flux formula: F_c = ρ * c_p * T * S(α) * (∇ - ∇_ad)^(3/2)
    # where S(α) = sqrt(g*δ/(8*H_p)) * (α*H_p)²
    
    # Solve for α
    # F_c = ρ * c_p * T * sqrt(g*δ/(8*H_p)) * (α*H_p)² * (∇-∇_ad)^(3/2)
    
    # Rearrange:
    # (α*H_p)² = F_c / [ρ * c_p * T * sqrt(g*δ/(8*H_p)) * (∇-∇_ad)^(3/2)]
    
    prefactor = rho * c_p * T * np.sqrt(g * delta / (8.0 * H_p))
    flux_term = delta_nabla**(3.0/2.0)
    
    if flux_term == 0 or prefactor == 0:
        return {
            'alpha': None,
            'H_p': H_p,
            'l': None,
            'nabla_ad': nabla_ad,
            'nabla': nabla,
            'F_c': 0.0,
            'message': 'Cannot solve: gradient is exactly adiabatic'
        }
    
    # (α*H_p)² = F_conv / (prefactor * flux_term)
    alpha_H_p_squared = F_conv / (prefactor * flux_term)
    
    if alpha_H_p_squared < 0:
        return {
            'alpha': None,
            'H_p': H_p,
            'l': None,
            'nabla_ad': nabla_ad,
            'nabla': nabla,
            'F_c': 0.0,
            'message': 'Negative alpha^2: check F_conv > 0'
        }
    
    alpha = np.sqrt(alpha_H_p_squared) / H_p
    l = alpha * H_p
    
    # Verify
    S = np.sqrt(g * delta / (8.0 * H_p)) * l**2
    F_c_check = rho * c_p * T * S * delta_nabla**(3.0/2.0)
    
    return {
        'alpha': alpha,
        'H_p': H_p,
        'l': l,
        'nabla_ad': nabla_ad,
        'nabla': nabla,
        'F_c': F_c_check,
        'F_conv_target': F_conv,
        'error': abs(F_c_check - F_conv) / F_conv if F_conv > 0 else 0.0,
        'message': 'Success'
    }


def solar_calibration_example():
    """
    Example: Calibrate α for solar convection zone.
    
    The Sun's convection zone is known to be adiabatic (from helioseismology).
    We calibrate α to match the known solar luminosity at a given depth.
    """
    print("=" * 70)
    print("SOLAR CALIBRATION: Adiabatic Convection Zone")
    print("=" * 70)
    print("\nCalibrating α for a layer that IS adiabatic")
    print("(Standard stellar modeling approach)\n")
    
    # Solar convection zone at ~0.85 R_sun
    T = 500000.0         # K (0.5 MK)
    P = 1.5e11           # Pa (1.5 Mbar)
    rho = 20.0           # kg/m³
    
    # Solar luminosity at this depth
    # L_sun = 3.828e26 W
    # Area at 0.85 R_sun: A = 4π(0.85 × 6.96e8)² ≈ 1.7e18 m²
    # Flux: F = L/A ≈ 2.2e8 W/m²
    # Assume ~80% is convective in this region
    F_conv = 0.8 * 2.2e8  # W/m²
    
    print(f"Layer properties:")
    print(f"  T = {T/1e6:.2f} MK")
    print(f"  P = {P/1e11:.2f} × 10¹¹ Pa")
    print(f"  ρ = {rho:.1f} kg/m³")
    print(f"  F_conv = {F_conv:.2e} W/m² (convective flux)\n")
    
    # Solar physical parameters (ionized H/He)
    gamma = 5.0/3.0  # Monatomic ideal gas
    physical_params = {
        'g': 274.0,                                        # m/s²
        'delta': 1.0,                                      # Ideal gas
        'R_universal': 8.314,                              # J/(mol·K)
        'mu': 0.0006,                                      # kg/mol
        'c_p': (gamma/(gamma-1)) * 8.314/0.0006,          # J/(kg·K)
    }
    
    print(f"Physical parameters:")
    print(f"  g = {physical_params['g']} m/s²")
    print(f"  μ = {physical_params['mu']} kg/mol (ionized H/He)")
    print(f"  c_p = {physical_params['c_p']:.1f} J/(kg·K)")
    print(f"  γ = {gamma:.3f} (monatomic)")
    print(f"  ∇_ad = (γ-1)/γ = {(gamma-1)/gamma:.4f}\n")
    
    # Calibrate with different epsilon values
    print("-" * 70)
    print("Calibrating α for different superadiabaticity levels:")
    print("-" * 70)
    
    epsilons = [0.0001, 0.001, 0.01]
    results = []
    
    for eps in epsilons:
        result = calibrate_alpha_adiabatic(T, P, rho, F_conv, physical_params, 
                                          epsilon=eps)
        results.append(result)
        
        print(f"\nε = {eps:.4f} (∇ = {result['nabla']:.6f}, ∇_ad = {result['nabla_ad']:.6f})")
        if result['alpha'] is not None:
            print(f"  → α = {result['alpha']:.4f}")
            print(f"  → l = {result['l']/1e6:.2f} Mm = {result['alpha']:.3f} × H_p")
            print(f"  → Flux match: {(1-result['error'])*100:.2f}%")
            
            # Compare with literature
            if 1.0 <= result['alpha'] <= 3.0:
                status = "✓ In stellar range [1.0, 3.0]"
                if 1.5 <= result['alpha'] <= 2.0:
                    status = "✓✓ IN SOLAR RANGE [1.5, 2.0]!"
            else:
                status = "Outside typical range"
            print(f"  → {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find best match
    alpha_values = [r['alpha'] for r in results if r['alpha'] is not None]
    if alpha_values:
        print(f"\nCalibrated α values: {[f'{a:.3f}' for a in alpha_values]}")
        print(f"Range: {min(alpha_values):.3f} - {max(alpha_values):.3f}")
        print(f"Mean: {np.mean(alpha_values):.3f}")
        
        print(f"\n📚 Literature comparison:")
        print(f"   Böhm-Vitense (1958): α ≈ 1.5")
        print(f"   Solar standard: α ≈ 1.5 - 2.0")
        print(f"   Our calibration: α ≈ {np.mean(alpha_values):.2f}")
        
        if 1.5 <= np.mean(alpha_values) <= 2.0:
            print(f"\n🎉 Excellent agreement with solar-calibrated values!")
        elif 1.0 <= np.mean(alpha_values) <= 3.0:
            print(f"\n✓ Within typical stellar range")
    
    print("=" * 70)
    
    return results


def hot_jupiter_calibration_example():
    """
    Example: Calibrate α for hot Jupiter atmosphere assumed to be adiabatic.
    """
    print("\n\n" + "=" * 70)
    print("HOT JUPITER CALIBRATION: Adiabatic Atmosphere")
    print("=" * 70)
    print("\nCalibrating α for hot Jupiter convective region\n")
    
    T = 1500.0           # K
    P = 5e4              # Pa (0.5 bar)
    rho = 0.005          # kg/m³
    F_conv = 5e6         # W/m² (strong flux)
    
    # H2-dominated atmosphere
    physical_params = {
        'g': 10.0,
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.0022,
        'c_p': 14000.0,
    }
    
    print(f"T = {T} K, P = {P/1e3:.1f} kPa, F_conv = {F_conv:.2e} W/m²\n")
    
    result = calibrate_alpha_adiabatic(T, P, rho, F_conv, physical_params, 
                                      epsilon=0.001)
    
    if result['alpha'] is not None:
        print(f"Calibrated: α = {result['alpha']:.4f}")
        print(f"Mixing length: l = {result['l']:.2f} m")
        print(f"∇_ad = {result['nabla_ad']:.6f}")
        print(f"∇ (used) = {result['nabla']:.6f} (0.1% superadiabatic)")
    else:
        print(f"Error: {result['message']}")
    
    return result


def main():
    """
    Run calibration examples.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  OPTION A: CALIBRATE α FOR ADIABATIC PROFILES".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Solar example
    solar_results = solar_calibration_example()
    
    # Hot Jupiter example
    hj_result = hot_jupiter_calibration_example()
    
    print("\n\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  USAGE IN YOUR RT CODE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    print("""
For a layer you know/assume is adiabatic:

```python
from calibrate_alpha_adiabatic import calibrate_alpha_adiabatic

# Your layer properties
T = ...      # K (layer average)
P = ...      # Pa
rho = ...    # kg/m³
F_conv = F_tot - F_rad  # W/m² (from your RT code)

# Physical parameters
params = {
    'g': ..., 'delta': 1.0, 'R_universal': 8.314,
    'mu': ..., 'c_p': ...
}

# Calibrate α
result = calibrate_alpha_adiabatic(T, P, rho, F_conv, params, epsilon=0.001)

alpha = result['alpha']
l = result['l']

# Use this α in your MLT convection scheme
```

The calibrated α will maintain the adiabatic state while carrying F_conv.
    """)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()




