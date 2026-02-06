"""
Parameter Space Exploration: T/P Profiles and MLT Alpha

This script explores how the mixing-length parameter α varies across
different temperature and pressure profiles, solving the flux balance
equation for each combination.

Usage:
    python explore_tp_parameter_space.py

Author: Generated for radiative transfer convection modeling
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/batch processing
from mlt_flux_balance import calculate_alpha_from_flux
import warnings
warnings.filterwarnings('ignore')


def generate_layer_from_profile(T_top, T_bot, P_top, P_bot):
    """
    Generate a layer with top/mid/bot values from profile parameters.
    
    Parameters:
        T_top, T_bot: Top and bottom temperatures [K]
        P_top, P_bot: Top and bottom pressures [Pa]
    
    Returns:
        dict: Layer data with T_top, T_mid, T_bot, P_top, P_mid, P_bot
    """
    # Use logarithmic interpolation for pressure (more physical)
    P_mid = np.sqrt(P_top * P_bot)
    
    # Linear interpolation for temperature (could be improved)
    T_mid = (T_top + T_bot) / 2.0
    
    return {
        'T_top': T_top,
        'T_mid': T_mid,
        'T_bot': T_bot,
        'P_top': P_top,
        'P_mid': P_mid,
        'P_bot': P_bot,
    }


def explore_parameter_space(T_range, P_top_range, P_bot_fixed, 
                           flux_data, physical_params,
                           verbose=False):
    """
    Explore parameter space by varying temperature profile and pressure.
    
    Parameters:
        T_range: Array of (T_top, T_bot) tuples [K]
        P_top_range: Array of top pressures [Pa]
        P_bot_fixed: Fixed bottom pressure [Pa]
        flux_data: Dict with F_tot, F_rad
        physical_params: Dict with physical parameters
        verbose: Print progress
    
    Returns:
        dict: Results with alpha_grid, nabla_grid, etc.
    """
    n_T = len(T_range)
    n_P = len(P_top_range)
    
    # Initialize result grids
    alpha_grid = np.full((n_T, n_P), np.nan)
    F_c_grid = np.full((n_T, n_P), np.nan)
    nabla_grid = np.full((n_T, n_P), np.nan)
    nabla_ad_grid = np.full((n_T, n_P), np.nan)
    superadiabatic_grid = np.zeros((n_T, n_P), dtype=bool)
    
    total = n_T * n_P
    count = 0
    
    for i, (T_top, T_bot) in enumerate(T_range):
        for j, P_top in enumerate(P_top_range):
            count += 1
            
            if verbose and count % 50 == 0:
                print(f"Progress: {count}/{total} ({100*count/total:.1f}%)")
            
            # Generate layer
            layer_data = generate_layer_from_profile(T_top, T_bot, P_top, P_bot_fixed)
            
            # Calculate alpha
            try:
                result = calculate_alpha_from_flux(layer_data, flux_data, 
                                                  physical_params, verbose=False)
                
                alpha_grid[i, j] = result['alpha'] if result['alpha'] is not None else np.nan
                F_c_grid[i, j] = result['F_c'] if result['F_c'] is not None else 0.0
                nabla_grid[i, j] = result['nabla']
                nabla_ad_grid[i, j] = result['nabla_ad']
                superadiabatic_grid[i, j] = result['is_superadiabatic']
                
            except Exception as e:
                if verbose:
                    print(f"Error at i={i}, j={j}: {e}")
                continue
    
    results = {
        'T_range': T_range,
        'P_top_range': P_top_range,
        'P_bot_fixed': P_bot_fixed,
        'alpha_grid': alpha_grid,
        'F_c_grid': F_c_grid,
        'nabla_grid': nabla_grid,
        'nabla_ad_grid': nabla_ad_grid,
        'superadiabatic_grid': superadiabatic_grid,
        'flux_data': flux_data,
        'physical_params': physical_params,
    }
    
    return results


def plot_results(results, save_prefix='alpha_parameter_space'):
    """
    Create comprehensive visualization of parameter space results.
    
    Parameters:
        results: Dictionary from explore_parameter_space()
        save_prefix: Prefix for saved figure files
    """
    T_range = results['T_range']
    P_top_range = results['P_top_range']
    alpha_grid = results['alpha_grid']
    nabla_grid = results['nabla_grid']
    nabla_ad_grid = results['nabla_ad_grid']
    superadiabatic_grid = results['superadiabatic_grid']
    
    # Extract arrays for plotting
    T_top_vals = np.array([t[0] for t in T_range])
    T_bot_vals = np.array([t[1] for t in T_range])
    delta_T_vals = T_bot_vals - T_top_vals
    
    # Create meshgrids
    Delta_T_mesh, P_top_mesh = np.meshgrid(delta_T_vals, P_top_range, indexing='ij')
    
    # Figure 1: Alpha heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Alpha vs ΔT and P_top
    ax = axes[0, 0]
    # Mask non-superadiabatic regions
    alpha_masked = np.ma.masked_where(~superadiabatic_grid, alpha_grid)
    
    im1 = ax.contourf(Delta_T_mesh, P_top_mesh/1e3, alpha_masked, 
                     levels=20, cmap='viridis')
    cbar1 = plt.colorbar(im1, ax=ax, label=r'$\alpha$')
    ax.set_xlabel(r'$\Delta T = T_{bot} - T_{top}$ [K]', fontsize=11)
    ax.set_ylabel(r'$P_{top}$ [kPa]', fontsize=11)
    ax.set_title(r'Mixing-Length Parameter $\alpha$', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    
    # Overlay contours showing superadiabatic regions
    ax.contour(Delta_T_mesh, P_top_mesh/1e3, superadiabatic_grid.astype(int),
              levels=[0.5], colors='red', linewidths=2, linestyles='--')
    ax.text(0.02, 0.98, 'Red line: superadiabatic boundary', 
           transform=ax.transAxes, va='top', fontsize=9, color='red')
    
    # Plot 2: Gradient ∇ vs ΔT and P_top
    ax = axes[0, 1]
    im2 = ax.contourf(Delta_T_mesh, P_top_mesh/1e3, nabla_grid, 
                     levels=20, cmap='plasma')
    cbar2 = plt.colorbar(im2, ax=ax, label=r'$\nabla$')
    ax.set_xlabel(r'$\Delta T = T_{bot} - T_{top}$ [K]', fontsize=11)
    ax.set_ylabel(r'$P_{top}$ [kPa]', fontsize=11)
    ax.set_title(r'Temperature Gradient $\nabla$', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    
    # Overlay adiabatic gradient
    nabla_ad_mean = np.nanmean(nabla_ad_grid)
    CS = ax.contour(Delta_T_mesh, P_top_mesh/1e3, nabla_ad_grid,
                   levels=[nabla_ad_mean], colors='white', linewidths=2)
    ax.clabel(CS, inline=True, fontsize=9, fmt=r'$\nabla_{ad}$=%.3f')
    
    # Plot 3: Alpha vs ΔT (at different P_top)
    ax = axes[1, 0]
    
    # Select a few P_top values to plot
    n_curves = min(5, len(P_top_range))
    indices = np.linspace(0, len(P_top_range)-1, n_curves, dtype=int)
    
    for idx in indices:
        P_val = P_top_range[idx]
        ax.plot(delta_T_vals, alpha_grid[:, idx], 'o-', 
               label=f'P_top = {P_val/1e3:.1f} kPa', markersize=4)
    
    ax.set_xlabel(r'$\Delta T$ [K]', fontsize=11)
    ax.set_ylabel(r'$\alpha$', fontsize=11)
    ax.set_title(r'$\alpha$ vs Temperature Difference', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax = axes[1, 1]
    
    # Histogram of alpha values (only superadiabatic)
    alpha_valid = alpha_grid[superadiabatic_grid & ~np.isnan(alpha_grid)]
    
    if len(alpha_valid) > 0:
        ax.hist(alpha_valid, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.nanmean(alpha_valid), color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {np.nanmean(alpha_valid):.3f}')
        ax.axvline(np.nanmedian(alpha_valid), color='orange', linestyle='--', linewidth=2,
                  label=f'Median = {np.nanmedian(alpha_valid):.3f}')
        ax.set_xlabel(r'$\alpha$', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(r'Distribution of $\alpha$ Values', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text with statistics
        stats_text = f"N = {len(alpha_valid)}\n"
        stats_text += f"Std = {np.nanstd(alpha_valid):.3f}\n"
        stats_text += f"Min = {np.nanmin(alpha_valid):.3f}\n"
        stats_text += f"Max = {np.nanmax(alpha_valid):.3f}"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
               va='top', ha='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No superadiabatic cases found',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_prefix}_comprehensive.png")
    plt.close()
    
    # Figure 2: Detailed alpha map
    fig, ax = plt.subplots(figsize=(10, 8))
    
    alpha_masked = np.ma.masked_where(~superadiabatic_grid, alpha_grid)
    im = ax.contourf(Delta_T_mesh, P_top_mesh/1e3, alpha_masked, 
                    levels=25, cmap='viridis', extend='both')
    cbar = plt.colorbar(im, ax=ax, label=r'Mixing-Length Parameter $\alpha$')
    
    # Add contour lines
    contour_lines = ax.contour(Delta_T_mesh, P_top_mesh/1e3, alpha_masked,
                               levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    ax.set_xlabel(r'$\Delta T = T_{bot} - T_{top}$ [K]', fontsize=12)
    ax.set_ylabel(r'$P_{top}$ [kPa]', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(r'MLT Mixing-Length Parameter $\alpha$ from Flux Balance' + '\n' +
                f'(F_tot={results["flux_data"]["F_tot"]:.1e} W/m², ' +
                f'F_rad={results["flux_data"]["F_rad"]:.1e} W/m²)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_prefix}_detailed.png")
    plt.close()


def save_results(results, filename='alpha_parameter_space.npz'):
    """Save results to file for later use."""
    np.savez(filename, **results)
    print(f"Saved results to: {filename}")


def main():
    """
    Main execution: Run parameter space exploration with default settings.
    """
    print("="*70)
    print("  T/P Parameter Space Exploration for MLT Alpha")
    print("="*70)
    print()
    
    # Define parameter ranges
    print("Setting up parameter ranges...")
    
    # Temperature profiles: (T_top, T_bot) pairs
    T_top_range = np.linspace(1000, 1800, 25)  # K
    T_bot_range = np.linspace(1200, 2500, 25)  # K
    T_range = [(T_top, T_bot) for T_top in T_top_range 
               for T_bot in T_bot_range if T_bot > T_top]
    
    # Pressure range
    P_top_range = np.logspace(3, 5, 20)  # 1 kPa to 100 kPa
    P_bot_fixed = 1e5  # 100 kPa (1 bar)
    
    print(f"  Temperature profiles: {len(T_range)} combinations")
    print(f"  Pressure levels: {len(P_top_range)} values")
    print(f"  Total calculations: {len(T_range) * len(P_top_range)}")
    print()
    
    # Flux parameters
    flux_data = {
        'F_tot': 1e7,    # W/m²
        'F_rad': 5e6,    # W/m²
    }
    
    print(f"Flux parameters:")
    print(f"  F_tot = {flux_data['F_tot']:.2e} W/m²")
    print(f"  F_rad = {flux_data['F_rad']:.2e} W/m²")
    print(f"  F_need = {flux_data['F_tot'] - flux_data['F_rad']:.2e} W/m²")
    print()
    
    # Physical parameters (Hot Jupiter, H2-dominated)
    physical_params = {
        'g': 10.0,
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.0022,
        'c_p': 14000.0,
        'rho': 0.005,
    }
    
    print(f"Physical parameters:")
    print(f"  g = {physical_params['g']} m/s²")
    print(f"  μ = {physical_params['mu']} kg/mol")
    print(f"  c_p = {physical_params['c_p']} J/(kg·K)")
    print()
    
    # Explore parameter space
    print("Exploring parameter space...")
    print("(This may take a few minutes)")
    print()
    
    results = explore_parameter_space(T_range, P_top_range, P_bot_fixed,
                                     flux_data, physical_params, verbose=True)
    
    # Statistics
    print()
    print("="*70)
    print("Results Summary:")
    print("="*70)
    
    alpha_valid = results['alpha_grid'][results['superadiabatic_grid'] & 
                                       ~np.isnan(results['alpha_grid'])]
    
    if len(alpha_valid) > 0:
        print(f"  Superadiabatic cases: {np.sum(results['superadiabatic_grid'])} / {results['superadiabatic_grid'].size}")
        print(f"  Valid α solutions: {len(alpha_valid)}")
        print(f"  Mean α: {np.nanmean(alpha_valid):.4f}")
        print(f"  Median α: {np.nanmedian(alpha_valid):.4f}")
        print(f"  Std α: {np.nanstd(alpha_valid):.4f}")
        print(f"  Range: [{np.nanmin(alpha_valid):.4f}, {np.nanmax(alpha_valid):.4f}]")
    else:
        print("  No superadiabatic cases found - try different parameters!")
    
    print()
    
    # Save results
    print("Saving results...")
    save_results(results)
    
    # Create plots
    print()
    print("Creating visualizations...")
    plot_results(results)
    
    print()
    print("="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()




