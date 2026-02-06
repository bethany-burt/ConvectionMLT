"""Analyze which layers should be convective vs radiative in Guillot profile"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import (
    run, temperature_gradient, adiabatic_gradient, calculate_c_p, 
    convective_flux, G, MMW, N_DOF
)

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

print("=" * 70)
print("Analyzing Guillot Profile: Convective vs Radiative Layers")
print("=" * 70)

# Run to get final state
z, T, rho, P, diagnostics = run(
    n_layers=10,
    max_steps=50000,
    alpha=0.1,
    dt=10.0,
    debug=False,
    save_history=False,
    profile_type="guillot",
    guillot_params=guillot_params,
    convergence_tol=1e-10,
    check_adiabatic=True,
    adiabatic_tolerance=0.2
)

# Calculate gradients and fluxes
N = temperature_gradient(T, z)
c_p = calculate_c_p(N_DOF, MMW)
N_ad = adiabatic_gradient(G, c_p)

# Interpolate to layer centers for flux calculation
T_mid = (T[:-1] + T[1:]) / 2.0
rho_mid = (rho[:-1] + rho[1:]) / 2.0
z_mid = (z[:-1] + z[1:]) / 2.0

# Calculate convective flux
# Note: convective_flux expects arrays, so we'll call it with arrays
F_conv = convective_flux(rho_mid, c_p, 0.1, G, T_mid, N, N_ad, MMW)

# Check which layers are convective (N > N_ad)
is_convective = N > N_ad
relative_diff = np.abs(N - N_ad) / N_ad

print(f"\nLayer Analysis:")
print(f"{'Layer':<8} {'z (km)':<12} {'T (K)':<10} {'N (K/m)':<15} {'N_ad':<10} {'N>N_ad?':<10} {'F_conv':<15} {'|N-N_ad|/N_ad':<15}")
print("-" * 100)

for i in range(len(N)):
    z_km = z_mid[i] / 1000.0
    conv_marker = "✓ CONV" if is_convective[i] else "  RAD"
    print(f"{i:<8} {z_km:<12.1f} {T_mid[i]:<10.1f} {N[i]:<15.6f} {N_ad:<10.6f} {conv_marker:<10} "
          f"{F_conv[i]:<15.2e} {relative_diff[i]:<15.6f}")

print(f"\nSummary:")
print(f"  Convective layers (N > N_ad): {np.sum(is_convective)} / {len(N)}")
print(f"  Radiative layers (N <= N_ad): {np.sum(~is_convective)} / {len(N)}")
print(f"  Layers with significant flux (F > 1 erg/cm²/s): {np.sum(F_conv > 1.0)} / {len(N)}")
print(f"\nConvergence issue:")
print(f"  The top layers (especially layer 9) have N << N_ad (radiative)")
print(f"  These layers may never become adiabatic if they're in radiative equilibrium")
print(f"  The convergence criterion requires ALL layers to be within 20% of adiabatic")
print(f"  This might be too strict for layers that should remain radiative")
