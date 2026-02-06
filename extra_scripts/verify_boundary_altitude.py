"""Verify the boundary altitude calculation and check for issues"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import (
    run, temperature_gradient, adiabatic_gradient, calculate_c_p,
    G, N_DOF, MMW
)

def find_radiative_boundary_detailed(z, T, N_ad):
    """Detailed version to verify boundary calculation"""
    N = temperature_gradient(T, z)
    convective_mask = N > N_ad
    
    print(f"\nDetailed boundary analysis:")
    print(f"  Grid: {len(z)} interfaces, {len(N)} layers")
    print(f"  Altitude range: {z[0]/1000:.1f} to {z[-1]/1000:.1f} km")
    print(f"  Adiabatic gradient: {N_ad:.6f} K/m")
    print(f"\nLayer-by-layer analysis:")
    print(f"{'Layer':<8} {'z_bottom (km)':<15} {'z_top (km)':<15} {'z_mid (km)':<15} {'N (K/m)':<15} {'N>N_ad?':<10}")
    print("-" * 80)
    
    for i in range(len(N)):
        z_bottom = z[i] / 1000.0
        z_top = z[i+1] / 1000.0
        z_mid = (z[i] + z[i+1]) / 2.0 / 1000.0
        conv_marker = "CONV" if convective_mask[i] else "RAD"
        print(f"{i:<8} {z_bottom:<15.1f} {z_top:<15.1f} {z_mid:<15.1f} {N[i]:<15.6f} {conv_marker:<10}")
    
    # Find transition
    n_convective = np.sum(convective_mask)
    n_radiative = len(N) - n_convective
    
    # Find first radiative layer from bottom
    radiative_indices = np.where(~convective_mask)[0]
    if len(radiative_indices) > 0:
        first_rad_idx = radiative_indices[0]
        boundary_altitude = z[first_rad_idx + 1] / 1000.0
        print(f"\nFirst radiative layer: Layer {first_rad_idx}")
        print(f"  Bottom altitude: {z[first_rad_idx]/1000:.1f} km")
        print(f"  Top altitude: {z[first_rad_idx+1]/1000:.1f} km")
        print(f"  Boundary altitude (top of first rad layer): {boundary_altitude:.1f} km")
    else:
        boundary_altitude = z[-1] / 1000.0
        print(f"\nAll layers convective - boundary at top: {boundary_altitude:.1f} km")
    
    return boundary_altitude, n_convective, n_radiative, convective_mask


# Test a few cases to verify
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

test_cases = [
    (10, 0.1, 1),
    (10, 0.1, 10),
    (10, 1.0, 100),
    (50, 0.1, 1),
    (50, 1.0, 100),
    (100, 0.1, 1),
    (100, 1.0, 100),
]

print("=" * 80)
print("Verifying Boundary Altitude Calculation")
print("=" * 80)

for n_layers, mixing_length, timestep in test_cases:
    print(f"\n{'='*80}")
    print(f"Test case: n_layers={n_layers}, l={mixing_length}, dt={timestep}")
    print(f"{'='*80}")
    
    z, T, rho, P, diagnostics = run(
        n_layers=n_layers,
        max_steps=50000,
        l=mixing_length,
        dt=timestep,
        debug=False,
        save_history=False,
        profile_type="guillot",
        guillot_params=guillot_params,
        convergence_tol=1e-10,
        check_adiabatic=True,
        adiabatic_tolerance=0.2
    )
    
    c_p = calculate_c_p(N_DOF, MMW)
    N_ad = adiabatic_gradient(G, c_p)
    
    boundary_alt, n_conv, n_rad, conv_mask = find_radiative_boundary_detailed(z, T, N_ad)
    
    print(f"\nSummary:")
    print(f"  Boundary altitude: {boundary_alt:.1f} km")
    print(f"  Convective layers: {n_conv} / {len(conv_mask)}")
    print(f"  Radiative layers: {n_rad} / {len(conv_mask)}")
