#!/usr/bin/env python3
"""
Analyze stability of temperature updates in convective flux solver.

This script investigates why some combinations of alpha and timestep lead to
temperature explosions, and derives a stability criterion.

Physical analysis:
- Temperature update: dT = -dt * (1/(ρ*c_p)) * dF_dz
- For stability: |dT| << T, so dt << T * (ρ*c_p) / |dF_dz|
- Flux depends on alpha: F_conv ∝ l² = (α * H_p)²
- Larger alpha → larger flux → larger |dF_dz| → need smaller dt
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import (
    run, temperature_gradient, adiabatic_gradient, calculate_c_p,
    convective_flux, G, RHO_TOA, RHO_BOA, T_TOA, T_BOA, N_DOF, MMW, R_SI
)

def analyze_stability_criterion(alpha, dt, n_layers=100, max_steps=10):
    """
    Run a short simulation and analyze stability at the first step.
    
    Returns:
        dict with stability analysis
    """
    print(f"\n{'='*70}")
    print(f"Stability Analysis: alpha={alpha}, dt={dt} s")
    print(f"{'='*70}")
    
    # Default Guillot parameters
    guillot_params = {
        'tint': 150.0,
        'tirr': 1200.0,
        'kappa_S': 0.01,
        'kappa0': 0.02,
        'kappa_cia': 0.0,
        'beta_S0': 1.0,
        'beta_L0': 1.0,
        'el1': 3.0/8.0,
        'el3': 1.0/3.0
    }
    
    try:
        # Calculate initial state manually (don't run full simulation)
        from convective_grid.convective_flux_v2 import setup_grid, initialize_profiles
        
        z, z_mid, dz = setup_grid(n_layers, 500_000)
        T, rho = initialize_profiles(z, z_mid, T_TOA, T_BOA, RHO_TOA, RHO_BOA,
                                      profile_type="guillot", guillot_params=guillot_params, g=G)
        
        # Calculate at layer centers
        T_mid = (T[:-1] + T[1:]) / 2.0
        rho_mid = (rho[:-1] + rho[1:]) / 2.0
        
        # Calculate gradients
        N = temperature_gradient(T, z)
        c_p = calculate_c_p(N_DOF, MMW)
        N_ad = adiabatic_gradient(G, c_p)
        
        # Calculate flux
        F_conv = convective_flux(rho_mid, c_p, alpha, G, T_mid, N, N_ad, MMW)
        
        # Calculate flux divergence at interfaces
        # Need dz at interfaces - use average of adjacent layer thicknesses for interior
        dz_cm = dz * 100.0  # cm
        dF_dz_erg_cm3_s = np.zeros(len(z))
        
        # Interior interfaces: use average dz
        dz_interior = (dz[:-1] + dz[1:]) / 2.0 * 100.0  # cm
        dF_dz_erg_cm3_s[1:-1] = (F_conv[1:] - F_conv[:-1]) / dz_interior
        
        # Boundaries
        dF_dz_erg_cm3_s[0] = (F_conv[0] - 0) / (dz_cm[0] / 2.0)
        dF_dz_erg_cm3_s[-1] = (0 - F_conv[-1]) / (dz_cm[-1] / 2.0)
        
        # Calculate dT
        DT_CONSTANT_interface = 1.0 / (rho * c_p)
        dT = -dt * DT_CONSTANT_interface * dF_dz_erg_cm3_s
        
        # Stability analysis
        dT_ratio = np.abs(dT) / np.maximum(T, 1.0)
        max_ratio_idx = np.argmax(dT_ratio)
        max_ratio = dT_ratio[max_ratio_idx]
        
        # Calculate stable dt
        stable_dt = 0.1 * T[max_ratio_idx] * (rho[max_ratio_idx] * c_p) / np.abs(dF_dz_erg_cm3_s[max_ratio_idx])
        
        print(f"\nResults:")
        print(f"  Max |dT|/T ratio: {max_ratio:.2e} at interface {max_ratio_idx} (z = {z[max_ratio_idx]/1000:.1f} km)")
        print(f"  Current dt: {dt:.2e} s")
        print(f"  Estimated stable dt (10% max change): {stable_dt:.2e} s")
        print(f"  dt ratio (current/stable): {dt/stable_dt:.2e}")
        
        if max_ratio > 0.5:
            print(f"  ⚠️  WARNING: Unstable! dT/T = {max_ratio:.2e} > 0.5")
            print(f"  → dt is {dt/stable_dt:.1f}x too large")
        elif max_ratio > 0.1:
            print(f"  ⚠️  CAUTION: Large changes (dT/T = {max_ratio:.2e} > 0.1)")
        else:
            print(f"  ✓ Stable: dT/T = {max_ratio:.2e} < 0.1")
        
        print(f"\n  Problematic interface details:")
        print(f"    z = {z[max_ratio_idx]/1000:.1f} km")
        print(f"    T = {T[max_ratio_idx]:.2f} K")
        print(f"    rho = {rho[max_ratio_idx]:.2e} g/cm³")
        print(f"    c_p = {c_p:.2e} erg/(g·K)")
        print(f"    1/(ρ*c_p) = {DT_CONSTANT_interface[max_ratio_idx]:.2e} cm³·K/erg")
        print(f"    dF_dz = {dF_dz_erg_cm3_s[max_ratio_idx]:.2e} erg cm⁻³ s⁻¹")
        print(f"    dT = {dT[max_ratio_idx]:.2e} K")
        print(f"    dT/T = {dT[max_ratio_idx]/T[max_ratio_idx]:.2e}")
        
        # Calculate H_p and l for context
        mmw_kg = MMW * 0.001  # kg/mol
        H_p = (R_SI * T[max_ratio_idx]) / (mmw_kg * G)  # m
        l = alpha * H_p  # m
        print(f"\n  Mixing length context:")
        print(f"    H_p = {H_p/1000:.1f} km")
        print(f"    l = α * H_p = {alpha} * {H_p/1000:.1f} km = {l/1000:.1f} km")
        print(f"    l² = {(l**2)/1e6:.2e} km² = {(l**2):.2e} m²")
        
        # Find flux values around this interface
        if max_ratio_idx > 0 and max_ratio_idx <= len(F_conv):
            print(f"\n  Flux context:")
            if max_ratio_idx == len(z) - 1:
                print(f"    F_conv[{len(F_conv)-1}] = {F_conv[-1]:.2e} erg cm⁻² s⁻¹")
            else:
                if max_ratio_idx > 0:
                    print(f"    F_conv[{max_ratio_idx-1}] (below) = {F_conv[max_ratio_idx-1]:.2e} erg cm⁻² s⁻¹")
                if max_ratio_idx < len(F_conv):
                    print(f"    F_conv[{max_ratio_idx}] (above) = {F_conv[max_ratio_idx]:.2e} erg cm⁻² s⁻¹")
        
        return {
            'stable': max_ratio < 0.5,
            'max_dT_ratio': max_ratio,
            'stable_dt': stable_dt,
            'dt_ratio': dt/stable_dt,
            'problematic_idx': max_ratio_idx,
            'z_problematic': z[max_ratio_idx],
            'T_problematic': T[max_ratio_idx],
            'dF_dz_problematic': dF_dz_erg_cm3_s[max_ratio_idx],
            'H_p': H_p,
            'l': l
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            'stable': False,
            'error': str(e)
        }


def main():
    """Run stability analysis for various alpha and dt combinations."""
    
    # Test combinations that might be problematic
    test_cases = [
        (0.1, 1),
        (0.1, 10),
        (0.1, 100),
        (0.1, 1000),
        (0.5, 1),
        (0.5, 10),
        (0.5, 100),
        (1.0, 1),
        (1.0, 10),
        (1.0, 100),
        (2.0, 1),
        (2.0, 10),
    ]
    
    results = []
    for alpha, dt in test_cases:
        result = analyze_stability_criterion(alpha, dt)
        result['alpha'] = alpha
        result['dt'] = dt
        results.append(result)
        print()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Alpha':<8} {'dt (s)':<10} {'Stable?':<10} {'dT/T max':<12} {'Stable dt':<15} {'dt ratio':<12}")
    print("-" * 70)
    for r in results:
        if 'error' not in r:
            stable_str = "✓ Yes" if r['stable'] else "✗ No"
            print(f"{r['alpha']:<8.1f} {r['dt']:<10.0f} {stable_str:<10} "
                  f"{r['max_dT_ratio']:<12.2e} {r['stable_dt']:<15.2e} {r['dt_ratio']:<12.2e}")
        else:
            print(f"{r['alpha']:<8.1f} {r['dt']:<10.0f} {'ERROR':<10} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    # Derive stability criterion
    print(f"\n{'='*70}")
    print("STABILITY CRITERION")
    print(f"{'='*70}")
    print("For stability: |dT| < f * T, where f is a safety factor (e.g., 0.1 = 10%)")
    print("This gives: dt < f * T * (ρ*c_p) / |dF_dz|")
    print("\nSince F_conv ∝ l² = (α * H_p)², larger alpha → larger flux → larger |dF_dz|")
    print("Therefore: dt_max ∝ 1/α² (approximately)")
    print("\nRecommended: dt < 0.1 * T_min * (ρ_min * c_p) / |dF_dz_max|")


if __name__ == "__main__":
    main()
