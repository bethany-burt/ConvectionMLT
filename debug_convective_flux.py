#!/usr/bin/env python3
"""
Debug and Verify Convective Flux Code

This script runs a test case with alpha=0.5, timestep=100s to verify:
1. Equations and unit consistency
2. Alpha parameter usage
3. Iteration tracking for a specific layer
4. Convergence criteria
5. Temperature profile visualization

Test parameters:
- alpha = 0.5
- dt = 100 s
- max_steps = 100000
- n_layers = 100
- Profile: Guillot (default)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import run

def main():
    """Run debug test case."""
    # Test parameters
    alpha = 1  # Larger mixing length for stronger convection
    dt = 1  # seconds
    n_layers = 100
    max_steps = 100000
    
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
    
    # Track middle layer
    track_layer = n_layers // 2
    track_steps = 20
    
    print("=" * 70)
    print("Debug Test Case: Convective Flux Verification")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  alpha = {alpha}")
    print(f"  dt = {dt} s")
    print(f"  n_layers = {n_layers}")
    print(f"  max_steps = {max_steps}")
    print(f"  track_layer = {track_layer}")
    print(f"  track_steps = {track_steps}")
    print()
    
    # Test 1: Constant = 1.0 (simplified)
    print("=" * 70)
    print("Energy Conservation Constant (1/(ρc_p))")
    print("=" * 70)
    print()
    
    # Run with energy conservation constant
    z, T, rho, P, diagnostics = run(
        n_layers=n_layers,
        alpha=alpha,
        dt=dt,
        max_steps=max_steps,
        debug=True,
        save_history=True,
        profile_type="guillot",
        guillot_params=guillot_params,
        check_adiabatic=True,
        adiabatic_tolerance=0.5,  # 50% tolerance: convective layers must have N/N_ad < 1.5
        use_energy_conservation=False,  # Use constant = 1.0
        track_layer=track_layer,
        track_steps=track_steps
    )
    
    print()
    print("=" * 70)
    print("Creating Convective Flux Summary Plot")
    print("=" * 70)
    
    # Create flux summary plot
    if 'history_T' in diagnostics and 'history_F' in diagnostics:
        plot_convective_flux_summary(
            diagnostics,
            output_file='plots/debug_convective_flux_summary.png'
        )
    else:
        print("  No history data available - skipping plot")
    
    print()
    print("=" * 70)
    print("Debug Test Complete")
    print("=" * 70)
    print()
    print("Results:")
    print(f"  Converged: {diagnostics['converged']}")
    print(f"  Steps: {diagnostics['steps']}")
    if 'converged_adiabatic' in diagnostics and diagnostics['converged_adiabatic'] is not None:
        print(f"  Adiabatic convergence: {diagnostics['converged_adiabatic']}")
    if 'max_dT_final' in diagnostics:
        print(f"  Final max|dT|: {diagnostics['max_dT_final']:.6e} K")
    if 'max_grad_diff_final' in diagnostics and diagnostics['max_grad_diff_final'] is not None:
        print(f"  Final max|N-N_ad|/N_ad: {diagnostics['max_grad_diff_final']:.4f}")
    print()


def plot_convective_flux_summary(diagnostics, output_file='plots/debug_convective_flux_summary.png'):
    """
    Plot flux, temperature, dF, and dT against number of steps.
    
    Similar to the plot_results function but formatted for debug output.
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Extract history data
    history_T = diagnostics['history_T']  # Shape: (n_steps, n_interfaces)
    history_dT = diagnostics['history_dT']  # Shape: (n_steps, n_interfaces)
    history_F = diagnostics['history_F']  # Shape: (n_steps, n_layers)
    history_dF = diagnostics['history_dF']  # Shape: (n_steps, n_interfaces)
    timesteps = diagnostics['timesteps']
    z = diagnostics['z']
    z_mid = diagnostics['z_mid']
    
    n_layers = len(z_mid)
    n_interfaces = len(z)
    n_steps = len(timesteps)
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Convective Flux Evolution (α={diagnostics.get("alpha", "N/A")}, '
                 f'n_layers={n_layers}, timesteps={n_steps})',
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature vs Timestep (at interfaces) - Top Left
    ax1 = axes[0, 0]
    # Plot a few representative layers (top, middle, bottom)
    if n_interfaces > 10:
        layer_indices = [0, n_interfaces//4, n_interfaces//2, 3*n_interfaces//4, n_interfaces-1]
    else:
        layer_indices = list(range(n_interfaces))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))
    for i, idx in enumerate(layer_indices):
        label = f'z={z[idx]/1000:.1f} km'
        ax1.plot(timesteps, history_T[:, idx], label=label, 
                color=colors[i], linewidth=1.5, marker='o', markersize=3)
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Temperature (K)', fontsize=11)
    ax1.set_title('Temperature vs Step (at Interfaces)', fontsize=12)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: dT vs Timestep (at interfaces) - Top Right
    ax2 = axes[0, 1]
    for i, idx in enumerate(layer_indices):
        label = f'z={z[idx]/1000:.1f} km'
        ax2.plot(timesteps, history_dT[:, idx], label=label, 
                color=colors[i], linewidth=1.5, marker='o', markersize=3)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Temperature Change dT (K)', fontsize=11)
    ax2.set_title('Temperature Change vs Step (at Interfaces)', fontsize=12)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('symlog', linthresh=1e-10)  # Symmetric log to handle negative values
    
    # Plot 3: Flux vs Timestep (at layer centers) - Bottom Left
    ax3 = axes[1, 0]
    # Plot a few representative layers
    if n_layers > 10:
        layer_indices_mid = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    else:
        layer_indices_mid = list(range(n_layers))
    
    colors_layers = plt.cm.plasma(np.linspace(0, 1, len(layer_indices_mid)))
    for i, idx in enumerate(layer_indices_mid):
        label = f'z={z_mid[idx]/1000:.1f} km'
        ax3.plot(timesteps, history_F[:, idx], label=label, 
                color=colors_layers[i], linewidth=1.5, marker='o', markersize=3)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Convective Flux F_conv (erg cm^-2 s^-1)', fontsize=11)
    ax3.set_title('Convective Flux vs Step (at Layer Centers)', fontsize=12)
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for flux
    
    # Plot 4: dFlux (dF/dz) vs Timestep (at interfaces) - Bottom Right
    ax4 = axes[1, 1]
    for i, idx in enumerate(layer_indices):
        label = f'z={z[idx]/1000:.1f} km'
        ax4.plot(timesteps, history_dF[:, idx], label=label, 
                color=colors[i], linewidth=1.5, marker='o', markersize=3)
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Flux Divergence dF/dz (erg cm^-3 s^-1)', fontsize=11)
    ax4.set_title('Flux Divergence vs Step (at Interfaces)', fontsize=12)
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('symlog', linthresh=1e-2)  # Symmetric log to handle negative values
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_temperature_verification_comparison(z1, T1_initial, T1_final, 
                                             z2, T2_initial, T2_final,
                                             diagnostics1, diagnostics2,
                                             dt=None, n_layers=None,
                                             output_file='plots/debug_temperature_verification.png'):
    """
    Create comparison plot showing temperature profiles before/after for both constant methods.
    """
    import matplotlib.patches as mpatches
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Convert to layer centers and km
    z1_mid = (z1[:-1] + z1[1:]) / 2.0 / 1000.0  # km
    z2_mid = (z2[:-1] + z2[1:]) / 2.0 / 1000.0  # km
    T1_initial_mid = (T1_initial[:-1] + T1_initial[1:]) / 2.0
    T1_final_mid = (T1_final[:-1] + T1_final[1:]) / 2.0
    T2_initial_mid = (T2_initial[:-1] + T2_initial[1:]) / 2.0
    T2_final_mid = (T2_final[:-1] + T2_final[1:]) / 2.0
    
    # Get N_final for layer type classification
    from convective_grid.convective_flux_v2 import temperature_gradient
    
    N1_final = diagnostics1['N_final']
    N2_final = diagnostics2['N_final']
    N_ad = diagnostics1['N_ad']
    
    # Classify layers
    layer_type1 = []
    layer_type2 = []
    for i in range(len(N1_final)):
        if N1_final[i] > N_ad:
            layer_type1.append('convective')
        else:
            layer_type1.append('radiative')
        if N2_final[i] > N_ad:
            layer_type2.append('convective')
        else:
            layer_type2.append('radiative')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Test 1: Constant = 1.0
    # Panel 1: Temperature profiles
    ax1 = axes[0, 0]
    ax1.plot(T1_initial_mid, z1_mid, 'b--', linewidth=2, label='Initial', alpha=0.7)
    ax1.plot(T1_final_mid, z1_mid, 'r-', linewidth=2, label='Final')
    ax1.set_xlabel('Temperature (K)', fontsize=11)
    ax1.set_ylabel('Altitude (km)', fontsize=11)
    ax1.set_title('Test 1: Constant = 1.0\nTemperature Profiles', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Temperature change
    ax2 = axes[0, 1]
    delta_T1 = T1_final_mid - T1_initial_mid
    ax2.plot(delta_T1, z1_mid, 'g-', linewidth=2)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('ΔT = T_final - T_initial (K)', fontsize=11)
    ax2.set_ylabel('Altitude (km)', fontsize=11)
    ax2.set_title('Test 1: Temperature Change', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Test 2: Energy Conservation
    # Panel 3: Temperature profiles
    ax3 = axes[1, 0]
    ax3.plot(T2_initial_mid, z2_mid, 'b--', linewidth=2, label='Initial', alpha=0.7)
    ax3.plot(T2_final_mid, z2_mid, 'r-', linewidth=2, label='Final')
    ax3.set_xlabel('Temperature (K)', fontsize=11)
    ax3.set_ylabel('Altitude (km)', fontsize=11)
    ax3.set_title('Test 2: Energy Conservation (1/(ρc_p))\nTemperature Profiles', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Temperature change
    ax4 = axes[1, 1]
    delta_T2 = T2_final_mid - T2_initial_mid
    ax4.plot(delta_T2, z2_mid, 'g-', linewidth=2)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('ΔT = T_final - T_initial (K)', fontsize=11)
    ax4.set_ylabel('Altitude (km)', fontsize=11)
    ax4.set_title('Test 2: Temperature Change', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Build title with available parameters
    title_parts = [f'α={diagnostics1["alpha"]}']
    if dt is not None:
        title_parts.append(f'dt={dt}s')
    if n_layers is not None:
        title_parts.append(f'n_layers={n_layers}')
    title = f'Convective Flux Verification: {", ".join(title_parts)}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    main()
