"""
Visualization: Mixing-Length Theory Concepts

Creates illustrative plots showing how MLT works and how α affects
convective properties.

Author: Generated for radiative transfer convection modeling
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mixing_length_theory import MixingLengthConvection


def plot_temperature_profiles():
    """
    Plot comparing temperature profiles with different treatments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define vertical levels (altitude)
    z = np.linspace(0, 10000, 100)  # 0-10 km
    
    # Define a superadiabatic initial profile
    T_initial = 1500 - 0.015 * z  # Superadiabatic: -15 K/km
    
    # Adiabatic profile (what old method forces to)
    T_adiabatic = 1500 - 0.010 * z  # Adiabatic: -10 K/km (g/cp with cp=1000, g=10)
    
    # MLT-adjusted profile (gradual adjustment)
    adjustment_factor = 0.3  # Partial adjustment
    T_mlt = T_initial + adjustment_factor * (T_adiabatic - T_initial)
    
    # Plot 1: All profiles together
    axes[0].plot(T_initial, z/1000, 'r-', linewidth=2, label='Initial (Superadiabatic)')
    axes[0].plot(T_adiabatic, z/1000, 'b--', linewidth=2, label='Adiabatic (Old Method)')
    axes[0].plot(T_mlt, z/1000, 'g-', linewidth=2, label='MLT Adjusted')
    axes[0].set_xlabel('Temperature [K]', fontsize=12)
    axes[0].set_ylabel('Altitude [km]', fontsize=12)
    axes[0].set_title('Temperature Profiles', fontsize=13, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    
    # Plot 2: Temperature gradients
    dT_dz_initial = np.gradient(T_initial, z)
    dT_dz_adiabatic = np.gradient(T_adiabatic, z)
    dT_dz_mlt = np.gradient(T_mlt, z)
    
    axes[1].plot(dT_dz_initial * 1000, z/1000, 'r-', linewidth=2, label='Initial')
    axes[1].axvline(-10, color='b', linestyle='--', linewidth=2, label='Adiabatic')
    axes[1].plot(dT_dz_mlt * 1000, z/1000, 'g-', linewidth=2, label='MLT')
    axes[1].set_xlabel('dT/dz [K/km]', fontsize=12)
    axes[1].set_ylabel('Altitude [km]', fontsize=12)
    axes[1].set_title('Temperature Gradient', fontsize=13, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()
    
    # Plot 3: Corrections applied
    correction_old = T_adiabatic - T_initial
    correction_mlt = T_mlt - T_initial
    
    axes[2].plot(correction_old, z/1000, 'b--', linewidth=2, label='Old Method')
    axes[2].plot(correction_mlt, z/1000, 'g-', linewidth=2, label='MLT Method')
    axes[2].axvline(0, color='k', linestyle=':', linewidth=1)
    axes[2].set_xlabel('Temperature Correction [K]', fontsize=12)
    axes[2].set_ylabel('Altitude [km]', fontsize=12)
    axes[2].set_title('Convective Corrections', fontsize=13, fontweight='bold')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('mlt_concept_temperature_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot 1: Temperature profiles comparison saved.")


def plot_alpha_effects():
    """
    Show how different α values affect mixing length and convective properties.
    """
    mlt = MixingLengthConvection(g=10.0, mu=0.0029, R=8.314)
    
    # Test conditions
    T = 1500.0
    P = 1e5
    cp = 1000.0
    
    # Calculate scale height
    H = mlt.pressure_scale_height(T, P)
    
    # Range of alpha values
    alpha_range = np.linspace(0.5, 3.0, 50)
    
    # Calculate mixing lengths
    mixing_lengths = [mlt.mixing_length(alpha, H) for alpha in alpha_range]
    
    # Calculate convective velocities for superadiabatic case
    grad_ad = mlt.adiabatic_gradient(T, P, cp)
    actual_grad = grad_ad * 1.2  # 20% superadiabatic
    delta_grad = actual_grad - grad_ad
    
    v_convs = [mlt.convective_velocity(l, delta_grad, T, H) 
               for l in mixing_lengths]
    
    # Calculate convective fluxes
    rho = (P * mlt.mu) / (mlt.R * T)
    F_convs = [mlt.convective_flux(l, delta_grad, T, rho, cp)
               for l in mixing_lengths]
    
    # Create plots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Mixing length vs alpha
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(alpha_range, np.array(mixing_lengths)/1000, 'b-', linewidth=2)
    ax1.axhline(H/1000, color='r', linestyle='--', linewidth=2, 
                label=f'Scale Height H = {H/1000:.1f} km')
    ax1.axvline(1.5, color='g', linestyle=':', linewidth=2, alpha=0.5, 
                label='Typical α = 1.5')
    ax1.set_xlabel(r'Mixing-length parameter $\alpha$', fontsize=12)
    ax1.set_ylabel('Mixing Length l [km]', fontsize=12)
    ax1.set_title('Mixing Length vs α', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convective velocity vs alpha
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(alpha_range, v_convs, 'orange', linewidth=2)
    ax2.axvline(1.5, color='g', linestyle=':', linewidth=2, alpha=0.5,
                label='Typical α = 1.5')
    ax2.set_xlabel(r'Mixing-length parameter $\alpha$', fontsize=12)
    ax2.set_ylabel('Convective Velocity [m/s]', fontsize=12)
    ax2.set_title('Convective Velocity vs α', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convective flux vs alpha
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(alpha_range, np.array(F_convs)/1e6, 'purple', linewidth=2)
    ax3.axvline(1.5, color='g', linestyle=':', linewidth=2, alpha=0.5,
                label='Typical α = 1.5')
    ax3.set_xlabel(r'Mixing-length parameter $\alpha$', fontsize=12)
    ax3.set_ylabel('Convective Flux [MW/m²]', fontsize=12)
    ax3.set_title('Convective Flux vs α', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mixing timescale vs alpha
    ax4 = fig.add_subplot(gs[1, 1])
    timescales = [l/v if v > 0 else np.inf for l, v in zip(mixing_lengths, v_convs)]
    ax4.plot(alpha_range, np.array(timescales)/60, 'brown', linewidth=2)
    ax4.axvline(1.5, color='g', linestyle=':', linewidth=2, alpha=0.5,
                label='Typical α = 1.5')
    ax4.set_xlabel(r'Mixing-length parameter $\alpha$', fontsize=12)
    ax4.set_ylabel('Mixing Timescale [minutes]', fontsize=12)
    ax4.set_title('Mixing Timescale vs α', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mlt_concept_alpha_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot 2: Alpha effects on convective properties saved.")
    print(f"\nAt α = 1.5:")
    idx = np.argmin(np.abs(alpha_range - 1.5))
    print(f"  Mixing length: {mixing_lengths[idx]/1000:.1f} km = {mixing_lengths[idx]/H:.2f} H")
    print(f"  Convective velocity: {v_convs[idx]:.2f} m/s")
    print(f"  Convective flux: {F_convs[idx]/1e6:.2f} MW/m²")
    print(f"  Mixing timescale: {timescales[idx]/60:.2f} minutes")


def plot_schwarzschild_criterion():
    """
    Illustrate the Schwarzschild criterion for convective instability.
    """
    # Temperature gradient range
    dT_dz = np.linspace(-0.020, -0.005, 100)  # K/m
    
    # Adiabatic gradient
    g = 10.0
    cp = 1000.0
    grad_ad = -g / cp  # -0.01 K/m
    
    # Stability indicator
    stability = np.where(dT_dz < grad_ad, 'Unstable\n(Convective)', 'Stable')
    colors = np.where(dT_dz < grad_ad, 'red', 'blue')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Temperature gradient with stability regions
    axes[0].scatter(dT_dz * 1000, np.ones_like(dT_dz), 
                   c=colors, s=50, alpha=0.6)
    axes[0].axvline(grad_ad * 1000, color='green', linestyle='--', 
                   linewidth=3, label='Adiabatic Gradient')
    axes[0].text(-18, 1.05, 'UNSTABLE\n(Convection)', 
                fontsize=11, ha='center', color='red', fontweight='bold')
    axes[0].text(-7, 1.05, 'STABLE\n(No Convection)', 
                fontsize=11, ha='center', color='blue', fontweight='bold')
    axes[0].set_xlabel('Temperature Gradient dT/dz [K/km]', fontsize=12)
    axes[0].set_ylabel('')
    axes[0].set_ylim(0.95, 1.15)
    axes[0].set_yticks([])
    axes[0].set_title('Schwarzschild Criterion', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Schematic of convective parcel
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Draw atmosphere layers
    for i in range(5):
        y = 2 + i * 1.5
        ax2.hlines(y, 0, 10, colors='gray', linestyles='--', alpha=0.5)
        if i < 4:
            ax2.text(0.2, y + 0.2, f'z{i}', fontsize=10)
    
    # Draw convective parcel rising
    parcel_x = [3, 3.5, 4.5, 5.5, 6.5, 7]
    parcel_y = [2.5, 3.5, 5.0, 6.5, 8.0, 9.0]
    
    ax2.plot(parcel_x, parcel_y, 'ro-', markersize=12, linewidth=2, 
            label='Rising Parcel')
    
    # Add arrows showing mixing
    ax2.annotate('', xy=(4, 5.5), xytext=(3.5, 4.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('', xy=(6, 7.0), xytext=(5.5, 6.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Add mixing length
    ax2.plot([7.5, 7.5], [5, 8], 'b-', linewidth=3, label='Mixing Length l')
    ax2.plot([7.4, 7.6], [5, 5], 'b-', linewidth=2)
    ax2.plot([7.4, 7.6], [8, 8], 'b-', linewidth=2)
    ax2.text(8.2, 6.5, r'$l = \alpha H$', fontsize=12, color='blue', fontweight='bold')
    
    # Labels
    ax2.text(5, 1, 'Superadiabatic Layer\n(Hotter than adiabat)', 
            fontsize=11, ha='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Horizontal Position', fontsize=12)
    ax2.set_ylabel('Altitude', fontsize=12)
    ax2.set_title('Convective Mixing Schematic', fontsize=13, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('mlt_concept_schwarzschild.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot 3: Schwarzschild criterion and convective mixing saved.")


def main():
    """Generate all conceptual visualization plots."""
    print("=" * 70)
    print("Generating Mixing-Length Theory Conceptual Visualizations")
    print("=" * 70)
    print()
    
    print("Creating Plot 1: Temperature Profiles...")
    plot_temperature_profiles()
    print()
    
    print("Creating Plot 2: Effects of α...")
    plot_alpha_effects()
    print()
    
    print("Creating Plot 3: Schwarzschild Criterion...")
    plot_schwarzschild_criterion()
    print()
    
    print("=" * 70)
    print("All visualization plots created successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. mlt_concept_temperature_profiles.png")
    print("  2. mlt_concept_alpha_effects.png")
    print("  3. mlt_concept_schwarzschild.png")
    print("\nThese plots illustrate key MLT concepts for presentations/papers.")


if __name__ == "__main__":
    main()


