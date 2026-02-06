#!/usr/bin/env python3
"""
Plot before/after convergence for Guillot profile with normalization fix.

This script runs the convective flux simulation with history tracking
and creates a plot showing the convergence behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the convective_grid directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'convective_grid'))

from convective_flux_v2 import run

def main():
    # Parameters
    n_layers = 150
    max_steps = 500000
    alpha = 0.1
    dt = 100
    
    print("=" * 70)
    print("Running Convective Flux Simulation")
    print("=" * 70)
    print(f"Layers: {n_layers}")
    print(f"Max steps: {max_steps}")
    print(f"Alpha: {alpha}")
    print(f"Timestep: {dt} s")
    print()
    
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
    
    # Run simulation with history tracking
    z, T_final, rho_final, P_final, diagnostics = run(
        n_layers=n_layers,
        T_toa=800.0,
        T_boa=2000.0,
        rho_toa=0.1,
        rho_boa=1000.0,
        profile_type='guillot',
        guillot_params=guillot_params,
        g=15.0,
        alpha=alpha,
        dt=dt,
        max_steps=max_steps,
        convergence_tol=1e-6,
        debug=False,
        save_history=True,  # Enable history tracking
        n_dof=5,
        mmw=2.016
    )
    
    # Extract history
    history_T = diagnostics.get('history_T', [])
    history_dT = diagnostics.get('history_dT', [])
    timesteps = diagnostics.get('timesteps', [])
    
    if len(history_T) == 0:
        print("ERROR: No history data saved. Check save_history implementation.")
        return
    
    print(f"\nCollected {len(history_T)} history snapshots")
    
    # Convert to numpy arrays
    history_T = np.array(history_T)
    history_dT = np.array(history_dT)
    timesteps = np.array(timesteps)
    
    # Calculate convergence metrics
    max_dT_history = np.max(np.abs(history_dT), axis=1)  # Max |dT| at each step
    mean_dT_history = np.mean(np.abs(history_dT), axis=1)  # Mean |dT| at each step
    
    # Calculate temperature gradient at each step
    N_history = []
    N_ad = diagnostics.get('N_ad', 0.00052)
    for T_step in history_T:
        # Calculate temperature gradient
        N = -np.diff(T_step) / np.diff(z)
        N_history.append(N)
    N_history = np.array(N_history)
    
    # Calculate deviation from adiabat for convective layers
    # (only where N > N_ad)
    adiabatic_deviation = []
    for N_step in N_history:
        convective_mask = N_step > N_ad
        if np.any(convective_mask):
            deviation = np.abs(N_step[convective_mask] - N_ad) / N_ad
            adiabatic_deviation.append(np.max(deviation))
        else:
            adiabatic_deviation.append(np.nan)
    adiabatic_deviation = np.array(adiabatic_deviation)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Convergence Analysis: Guillot Profile (n_layers={n_layers}, α={alpha}, dt={dt}s)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Max |dT| vs timestep
    ax1 = axes[0, 0]
    ax1.semilogy(timesteps, max_dT_history, 'b-', linewidth=2, label='max|dT|')
    ax1.semilogy(timesteps, mean_dT_history, 'r--', linewidth=1.5, alpha=0.7, label='mean|dT|')
    ax1.axhline(y=1e-6, color='g', linestyle=':', linewidth=1, label='Tolerance (1e-6 K)')
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_ylabel('Temperature Change |dT| (K)', fontsize=11)
    ax1.set_title('Convergence: Temperature Change', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Temperature profiles at different stages
    ax2 = axes[0, 1]
    # Show initial, middle, and final profiles
    n_profiles = min(5, len(history_T))
    indices = np.linspace(0, len(history_T)-1, n_profiles, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    
    for i, idx in enumerate(indices):
        step = timesteps[idx]
        label = f'Step {step}'
        if idx == 0:
            label = 'Initial'
        elif idx == len(history_T) - 1:
            label = 'Final'
        ax2.plot(history_T[idx], z/1000, color=colors[i], linewidth=2, label=label)
    
    ax2.set_xlabel('Temperature (K)', fontsize=11)
    ax2.set_ylabel('Altitude (km)', fontsize=11)
    ax2.set_title('Temperature Profile Evolution', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Plot 3: Adiabatic deviation for convective layers
    ax3 = axes[1, 0]
    valid_mask = ~np.isnan(adiabatic_deviation)
    if np.any(valid_mask):
        ax3.semilogy(timesteps[valid_mask], adiabatic_deviation[valid_mask], 
                    'g-', linewidth=2, label='Max |N - N_ad|/N_ad')
        ax3.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, 
                   label='100% deviation (convergence threshold)')
        ax3.set_xlabel('Timestep', fontsize=11)
        ax3.set_ylabel('|N - N_ad| / N_ad (convective layers)', fontsize=11)
        ax3.set_title('Convergence to Adiabatic Gradient', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No convective layers detected', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Convergence to Adiabatic Gradient', fontsize=12, fontweight='bold')
    
    # Plot 4: Temperature gradient profiles
    ax4 = axes[1, 1]
    z_mid = (z[:-1] + z[1:]) / 2.0  # Layer centers
    
    # Show initial, middle, and final gradient profiles
    for i, idx in enumerate(indices):
        step = timesteps[idx]
        label = f'Step {step}'
        if idx == 0:
            label = 'Initial'
        elif idx == len(history_T) - 1:
            label = 'Final'
        ax4.plot(N_history[idx], z_mid/1000, color=colors[i], linewidth=2, label=label)
    
    # Add adiabatic gradient line
    ax4.axvline(x=N_ad, color='r', linestyle='--', linewidth=2, label=f'Adiabatic (N_ad={N_ad:.6f})')
    
    ax4.set_xlabel('Temperature Gradient N (K/m)', fontsize=11)
    ax4.set_ylabel('Altitude (km)', fontsize=11)
    ax4.set_title('Temperature Gradient Evolution', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    output_file = f'plots/convergence_guillot_{n_layers}layers_{max_steps}steps.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Convergence Summary")
    print("=" * 70)
    print(f"Initial max|dT|: {max_dT_history[0]:.6e} K")
    print(f"Final max|dT|: {max_dT_history[-1]:.6e} K")
    print(f"Convergence ratio: {max_dT_history[0]/max_dT_history[-1]:.2e}")
    if np.any(valid_mask):
        print(f"Initial adiabatic deviation: {adiabatic_deviation[valid_mask][0]:.4f}")
        print(f"Final adiabatic deviation: {adiabatic_deviation[valid_mask][-1]:.4f}")
    print(f"Total steps: {len(timesteps)}")
    print(f"Final step: {timesteps[-1]}")

if __name__ == '__main__':
    main()
