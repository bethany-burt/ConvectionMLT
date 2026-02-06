"""Diagnose which layers are preventing convergence in Guillot profile"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import (
    run, temperature_gradient, adiabatic_gradient, calculate_c_p, G, MMW, N_DOF
)

# Use same parameters as the sweep
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
print("Diagnosing Guillot Profile Gradient Issues")
print("=" * 70)

# Run with fewer steps to get intermediate state
z, T, rho, P, diagnostics = run(
    n_layers=10,
    max_steps=50000,  # Enough to see some evolution
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

# Calculate gradients
N = temperature_gradient(T, z)
c_p = calculate_c_p(N_DOF, MMW)
N_ad = adiabatic_gradient(G, c_p)

# Calculate relative differences
relative_diff = np.abs(N - N_ad) / N_ad
max_diff_idx = np.argmax(relative_diff)

print(f"\nGradient Analysis:")
print(f"  Adiabatic gradient (N_ad): {N_ad:.6f} K/m")
print(f"  Number of layers: {len(N)}")
print(f"\nLayer-by-layer breakdown:")
print(f"{'Layer':<8} {'z (km)':<12} {'N (K/m)':<15} {'N_ad (K/m)':<15} {'|N-N_ad|/N_ad':<15} {'Status':<20}")
print("-" * 85)

for i in range(len(N)):
    z_mid = (z[i] + z[i+1]) / 2.0 / 1000.0  # Convert to km
    rel_diff = relative_diff[i]
    status = "✓ Converged" if rel_diff < 0.2 else "✗ Not converged"
    marker = " <-- MAX" if i == max_diff_idx else ""
    print(f"{i:<8} {z_mid:<12.1f} {N[i]:<15.6f} {N_ad:<15.6f} {rel_diff:<15.6f} {status:<20}{marker}")

print(f"\nSummary:")
print(f"  Max relative difference: {np.max(relative_diff):.6f} ({np.max(relative_diff)*100:.2f}%)")
print(f"  Layers converged: {np.sum(relative_diff < 0.2)} / {len(N)}")
print(f"  Layers not converged: {np.sum(relative_diff >= 0.2)} / {len(N)}")
print(f"  Worst layer: Layer {max_diff_idx} at z = {(z[max_diff_idx] + z[max_diff_idx+1])/2/1000:.1f} km")

# Check if there's a pattern
print(f"\nPattern analysis:")
print(f"  Top layers (0-2): max diff = {np.max(relative_diff[:3]):.6f}")
print(f"  Middle layers (3-6): max diff = {np.max(relative_diff[3:7]):.6f}")
print(f"  Bottom layers (7-9): max diff = {np.max(relative_diff[7:]):.6f}")
