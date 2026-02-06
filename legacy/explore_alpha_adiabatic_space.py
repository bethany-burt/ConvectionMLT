"""
Parameter Space Exploration: Alpha for Adiabatic Profiles

Explores how the mixing-length parameter α varies across T/P parameter space
when calibrating for adiabatic profiles (Option A).

For each T/P point, assumes layer is adiabatic and finds α that would
maintain that state while carrying the required convective flux.

Author: Generated for adiabatic calibration parameter space
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from calibrate_alpha_adiabatic import calibrate_alpha_adiabatic
import warnings
warnings.filterwarnings('ignore')


def explore_tp_space_adiabatic(T_range, P_range, rho_func, F_conv_func, 
                               physical_params, epsilon=0.001, verbose=False):
    """
    Explore parameter space for adiabatic calibration.
    
    Parameters:
        T_range: Array of temperatures [K]
        P_range: Array of pressures [Pa]
        rho_func: Function rho(T, P) returning density [kg/m³]
        F_conv_func: Function F_conv(T, P) returning convective flux [W/m²]
        physical_params: Dict with g, delta, R_universal, mu, c_p
        epsilon: Superadiabaticity parameter
        verbose: Print progress
    
    Returns:
        dict: Results with alpha_grid, l_grid, H_p_grid, etc.
    """
    n_T = len(T_range)
    n_P = len(P_range)
    
    # Initialize grids
    alpha_grid = np.full((n_T, n_P), np.nan)
    l_grid = np.full((n_T, n_P), np.nan)
    H_p_grid = np.full((n_T, n_P), np.nan)
    nabla_ad_grid = np.full((n_T, n_P), np.nan)
    F_conv_grid = np.full((n_T, n_P), np.nan)
    
    total = n_T * n_P
    count = 0
    
    for i, T in enumerate(T_range):
        for j, P in enumerate(P_range):
            count += 1
            
            if verbose and count % 50 == 0:
                print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")
            
            try:
                rho = rho_func(T, P)
                F_conv = F_conv_func(T, P)
                
                if F_conv <= 0:
                    continue
                
                result = calibrate_alpha_adiabatic(T, P, rho, F_conv, 
                                                  physical_params, epsilon)
                
                if result['alpha'] is not None:
                    alpha_grid[i, j] = result['alpha']
                    l_grid[i, j] = result['l']
                    H_p_grid[i, j] = result['H_p']
                    nabla_ad_grid[i, j] = result['nabla_ad']
                    F_conv_grid[i, j] = F_conv
                    
            except Exception as e:
                if verbose:
                    print(f"Error at T={T}, P={P}: {e}")
                continue
    
    return {
        'T_range': T_range,
        'P_range': P_range,
        'alpha_grid': alpha_grid,
        'l_grid': l_grid,
        'H_p_grid': H_p_grid,
        'nabla_ad_grid': nabla_ad_grid,
        'F_conv_grid': F_conv_grid,
        'physical_params': physical_params,
        'epsilon': epsilon,
    }


def plot_results(results, save_prefix='alpha_adiabatic_space'):
    """
    Create visualization of parameter space results.
    """
    T_range = results['T_range']
    P_range = results['P_range']
    alpha_grid = results['alpha_grid']
    
    # Create meshgrids
    T_mesh, P_mesh = np.meshgrid(T_range, P_range, indexing='ij')
    
    # Figure 1: Comprehensive 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Alpha heatmap (LOG SCALE)
    ax = axes[0, 0]
    alpha_masked = np.ma.masked_invalid(alpha_grid)
    
    # Use logarithmic color scale
    vmin = np.nanmin(alpha_masked[alpha_masked > 0]) if np.any(alpha_masked > 0) else 0.001
    vmax = np.nanmax(alpha_masked)
    
    im1 = ax.pcolormesh(T_mesh, P_mesh/1e5, alpha_masked, 
                        cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
                        shading='auto')
    cbar1 = plt.colorbar(im1, ax=ax, label=r'$\alpha$ (log scale)')
    
    # Add contour lines at log-spaced levels
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)
    contour_lines = ax.contour(T_mesh, P_mesh/1e5, alpha_masked,
                               levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('Pressure [bar]', fontsize=12)
    ax.set_yscale('log')
    ax.invert_yaxis()  # High pressure at bottom
    ax.set_title(r'Mixing-Length Parameter $\alpha$ (Adiabatic Calibration)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mixing length
    ax = axes[0, 1]
    l_masked = np.ma.masked_invalid(results['l_grid'])
    im2 = ax.contourf(T_mesh, P_mesh/1e5, l_masked/1000, levels=20, cmap='plasma')
    cbar2 = plt.colorbar(im2, ax=ax, label='Mixing Length [km]')
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('Pressure [bar]', fontsize=12)
    ax.set_yscale('log')
    ax.invert_yaxis()  # High pressure at bottom
    ax.set_title('Mixing Length l', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Alpha distribution
    ax = axes[1, 0]
    alpha_valid = alpha_grid[~np.isnan(alpha_grid)]
    if len(alpha_valid) > 0:
        ax.hist(alpha_valid, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.nanmean(alpha_valid), color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {np.nanmean(alpha_valid):.3f}')
        ax.axvline(np.nanmedian(alpha_valid), color='orange', linestyle='--', linewidth=2,
                  label=f'Median = {np.nanmedian(alpha_valid):.3f}')
        ax.set_xlabel(r'$\alpha$', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(r'Distribution of $\alpha$', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"N = {len(alpha_valid)}\n"
        stats_text += f"Std = {np.nanstd(alpha_valid):.3f}\n"
        stats_text += f"Min = {np.nanmin(alpha_valid):.3f}\n"
        stats_text += f"Max = {np.nanmax(alpha_valid):.3f}"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
               va='top', ha='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Alpha vs Temperature slices
    ax = axes[1, 1]
    n_curves = min(5, len(P_range))
    indices = np.linspace(0, len(P_range)-1, n_curves, dtype=int)
    for idx in indices:
        P_val = P_range[idx]
        ax.plot(T_range, alpha_grid[:, idx], 'o-', 
               label=f'P = {P_val/1e5:.2f} bar', markersize=4)
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=12)
    ax.set_title(r'$\alpha$ vs Temperature', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_prefix}_comprehensive.png")
    plt.close()
    
    # Figure 2: Detailed alpha heatmap (LOG SCALE)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use logarithmic color scale
    vmin = np.nanmin(alpha_masked[alpha_masked > 0]) if np.any(alpha_masked > 0) else 0.001
    vmax = np.nanmax(alpha_masked)
    
    im = ax.pcolormesh(T_mesh, P_mesh/1e5, alpha_masked, 
                       cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax),
                       shading='auto')
    cbar = plt.colorbar(im, ax=ax, label=r'Mixing-Length Parameter $\alpha$ (log scale)')
    
    # Add contour lines at log-spaced levels
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 12)
    contour_lines = ax.contour(T_mesh, P_mesh/1e5, alpha_masked,
                               levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('Pressure [bar]', fontsize=12)
    ax.set_yscale('log')
    ax.invert_yaxis()  # High pressure at bottom
    ax.set_title('Mixing-Length Parameter α: Adiabatic Calibration\n' +
                f'(ε = {results["epsilon"]}, ∇ = ∇_ad × (1 + ε))',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_prefix}_detailed.png")
    plt.close()


def main():
    """
    Run parameter space exploration with default settings.
    """
    print("=" * 70)
    print("  T/P Parameter Space: Adiabatic Calibration of α")
    print("=" * 70)
    print()
    
    # Define parameter ranges
    print("Setting up parameter ranges...")
    
    T_range = np.linspace(200, 2000, 20)    # K (coarser for speed)
    P_range = np.logspace(0, 8, 20)         # 1 Pa to 10^8 Pa (10^-5 to 10^3 bar)
    
    print(f"  Temperature: {T_range.min():.0f} - {T_range.max():.0f} K ({len(T_range)} points)")
    print(f"  Pressure: {P_range.min():.2e} - {P_range.max():.2e} Pa ({len(P_range)} points)")
    print(f"  Total calculations: {len(T_range) * len(P_range)}")
    print()
    
    # Physical parameters (H2-dominated atmosphere, e.g., hot Jupiter)
    physical_params = {
        'g': 10.0,            # m/s²
        'delta': 1.0,         # Ideal gas
        'R_universal': 8.314, # J/(mol·K)
        'mu': 0.0022,         # kg/mol (H2)
        'c_p': 14000.0,       # J/(kg·K) (H2)
    }
    
    print("Physical parameters (H2-dominated):")
    print(f"  g = {physical_params['g']} m/s²")
    print(f"  μ = {physical_params['mu']} kg/mol")
    print(f"  c_p = {physical_params['c_p']} J/(kg·K)")
    print()
    
    # Define density function (ideal gas)
    def rho_func(T, P):
        """Density from ideal gas law: ρ = P*μ/(R*T)"""
        return (P * physical_params['mu']) / (physical_params['R_universal'] * T)
    
    # Define convective flux function
    # Option 1: Constant flux
    F_conv_constant = 5e6  # W/m²
    def F_conv_func(T, P):
        return F_conv_constant
    
    # Option 2: Flux scales with T^4 (radiative scaling)
    # T_ref = 1500.0
    # F_ref = 5e6
    # def F_conv_func(T, P):
    #     return F_ref * (T / T_ref)**4
    
    print(f"Convective flux: F_conv = {F_conv_constant:.2e} W/m² (constant)")
    print()
    
    # Run exploration
    print("Exploring parameter space...")
    print("(This may take a minute)")
    print()
    
    results = explore_tp_space_adiabatic(T_range, P_range, rho_func, F_conv_func,
                                        physical_params, epsilon=0.001, verbose=True)
    
    # Statistics
    print()
    print("=" * 70)
    print("Results Summary:")
    print("=" * 70)
    
    alpha_valid = results['alpha_grid'][~np.isnan(results['alpha_grid'])]
    
    if len(alpha_valid) > 0:
        print(f"  Valid α solutions: {len(alpha_valid)} / {results['alpha_grid'].size}")
        print(f"  Mean α: {np.nanmean(alpha_valid):.4f}")
        print(f"  Median α: {np.nanmedian(alpha_valid):.4f}")
        print(f"  Std α: {np.nanstd(alpha_valid):.4f}")
        print(f"  Range: [{np.nanmin(alpha_valid):.4f}, {np.nanmax(alpha_valid):.4f}]")
        
        # Check ranges
        in_stellar = np.sum((alpha_valid >= 1.0) & (alpha_valid <= 3.0))
        in_solar = np.sum((alpha_valid >= 1.5) & (alpha_valid <= 2.0))
        
        print(f"\n  In stellar range [1.0, 3.0]: {in_stellar} ({100*in_stellar/len(alpha_valid):.1f}%)")
        print(f"  In solar range [1.5, 2.0]: {in_solar} ({100*in_solar/len(alpha_valid):.1f}%)")
    else:
        print("  No valid solutions found!")
    
    print()
    
    # Save results
    print("Saving results...")
    np.savez('alpha_adiabatic_space.npz', **results)
    print("  Saved: alpha_adiabatic_space.npz")
    
    # Create plots
    print()
    print("Creating visualizations...")
    plot_results(results)
    
    print()
    print("=" * 70)
    print("Complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print("  • alpha_adiabatic_space.npz (data)")
    print("  • alpha_adiabatic_space_comprehensive.png (4-panel plot)")
    print("  • alpha_adiabatic_space_detailed.png (detailed heatmap)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

