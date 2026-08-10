"""
Analyze the radiative-convective boundary altitude for Guillot profile.
Find where radiative layers start and how mixing length/timestep affect this.
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import (
    run, temperature_gradient, adiabatic_gradient, calculate_c_p,
    G, N_DOF, MMW
)


def find_radiative_boundary(z, T, N_ad):
    """
    Find the altitude where radiative layers start (transition from convective to radiative).
    
    Returns:
        boundary_altitude: Altitude (km) where transition occurs (None if all convective or all radiative)
        n_convective: Number of convective layers
        n_radiative: Number of radiative layers
        convective_mask: Boolean array indicating which layers are convective
    """
    N = temperature_gradient(T, z)
    convective_mask = N > N_ad
    
    # Find transition point (going from bottom to top)
    # Look for first radiative layer after convective layers
    n_convective = np.sum(convective_mask)
    n_radiative = len(N) - n_convective
    
    # Find boundary altitude (altitude of first radiative layer, or top if all convective)
    boundary_altitude = None
    if n_radiative > 0:
        # Find first radiative layer (from bottom)
        radiative_indices = np.where(~convective_mask)[0]
        if len(radiative_indices) > 0:
            # Get altitude at the top of the first radiative layer
            first_rad_idx = radiative_indices[0]
            boundary_altitude = z[first_rad_idx + 1] / 1000.0  # Convert to km
        else:
            # All layers are radiative (unlikely but possible)
            boundary_altitude = z[0] / 1000.0
    else:
        # All layers are convective
        boundary_altitude = z[-1] / 1000.0  # Top of atmosphere
    
    return boundary_altitude, n_convective, n_radiative, convective_mask


def run_boundary_analysis(n_layers_list, alpha_list, timestep_list,
                         max_steps=50000, guillot_params=None):
    """
    Run parameter sweep and track radiative-convective boundary.
    """
    if guillot_params is None:
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
    print("Radiative-Convective Boundary Analysis")
    print("=" * 70)
    print(f"n_layers: {n_layers_list}")
    print(f"alpha: {alpha_list}")
    print(f"timestep: {timestep_list}")
    print(f"max_steps: {max_steps}")
    print()
    
    results = {
        'n_layers': [],
        'alpha': [],
        'timestep': [],
        'boundary_altitude': [],  # km
        'n_convective': [],
        'n_radiative': [],
        'max_grad_diff': [],
        'run_time': []
    }
    
    total_runs = len(n_layers_list) * len(alpha_list) * len(timestep_list)
    current_run = 0
    
    for n_layers in n_layers_list:
        for alpha in alpha_list:
            for timestep in timestep_list:
                current_run += 1
                run_start = time.time()
                
                print(f"[{current_run}/{total_runs}] n_layers={n_layers:3d}, "
                      f"alpha={alpha:5.2f}, dt={timestep:6.1f}s", end=" ... ")
                
                try:
                    z, T, rho, P, diagnostics = run(
                        n_layers=n_layers,
                        max_steps=max_steps,
                        alpha=alpha,
                        dt=timestep,
                        debug=False,
                        save_history=False,
                        profile_type="guillot",
                        guillot_params=guillot_params,
                        convergence_tol=1e-10,
                        check_adiabatic=True,
                        adiabatic_tolerance=0.2
                    )
                    
                    # Calculate adiabatic gradient
                    c_p = calculate_c_p(N_DOF, MMW)
                    N_ad = adiabatic_gradient(G, c_p)
                    
                    # Find radiative boundary
                    boundary_alt, n_conv, n_rad, conv_mask = find_radiative_boundary(z, T, N_ad)
                    
                    run_time = time.time() - run_start
                    
                    results['n_layers'].append(n_layers)
                    results['alpha'].append(alpha)
                    results['timestep'].append(timestep)
                    results['boundary_altitude'].append(boundary_alt)
                    results['n_convective'].append(n_conv)
                    results['n_radiative'].append(n_rad)
                    results['max_grad_diff'].append(diagnostics['max_grad_diff_final'])
                    results['run_time'].append(run_time)
                    
                    print(f"boundary={boundary_alt:.1f}km (conv={n_conv}, rad={n_rad}, "
                          f"max_diff={diagnostics['max_grad_diff_final']:.3f})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    # Record failed run
                    results['n_layers'].append(n_layers)
                    results['alpha'].append(alpha)
                    results['timestep'].append(timestep)
                    results['boundary_altitude'].append(np.nan)
                    results['n_convective'].append(np.nan)
                    results['n_radiative'].append(np.nan)
                    results['max_grad_diff'].append(np.nan)
                    results['run_time'].append(time.time() - run_start)
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def plot_boundary_analysis(results, output_prefix="radiative_boundary"):
    """
    Plot how boundary altitude changes with alpha and timestep.
    """
    n_layers_list = np.unique(results['n_layers'])
    # Handle both 'alpha' and 'mixing_length' for backward compatibility
    if 'alpha' in results:
        alpha_list = np.unique(results['alpha'])
        alpha_key = 'alpha'
    else:
        alpha_list = np.unique(results['mixing_length'])
        alpha_key = 'mixing_length'
    timestep_list = np.unique(results['timestep'])
    
    # Plot 1: Boundary altitude vs alpha (for different timesteps)
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Radiative-Convective Boundary Altitude vs Alpha (colored by timestep)', 
                  fontsize=16, fontweight='bold')
    axes1_flat = axes1.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(timestep_list)))
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes1_flat[idx]
        
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        for dt_idx, dt in enumerate(timestep_list):
            dt_mask = data_subset['timestep'] == dt
            if not np.any(dt_mask):
                continue
            
            dt_data = {key: data_subset[key][dt_mask] for key in data_subset.keys()}
            
            # Sort by alpha
            sort_idx = np.argsort(dt_data[alpha_key])
            alpha_sorted = dt_data[alpha_key][sort_idx]
            boundary_sorted = dt_data['boundary_altitude'][sort_idx]
            
            # Filter out NaN values
            valid = ~np.isnan(boundary_sorted)
            if np.any(valid):
                ax.plot(alpha_sorted[valid], boundary_sorted[valid], 'o-', 
                       color=colors[dt_idx], label=f'dt={dt:.0f}s', 
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax.set_ylabel('Boundary Altitude (km)', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes1_flat)):
        axes1_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_vs_alpha.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Plot 2: Boundary altitude vs timestep (for different alpha values)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Radiative-Convective Boundary Altitude vs Timestep (colored by alpha)', 
                  fontsize=16, fontweight='bold')
    axes2_flat = axes2.flatten()
    
    colors2 = plt.cm.tab10(np.linspace(0, 1, len(alpha_list)))
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes2_flat[idx]
        
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        for alpha_idx, alpha in enumerate(alpha_list):
            alpha_mask = data_subset[alpha_key] == alpha
            if not np.any(alpha_mask):
                continue
            
            alpha_data = {key: data_subset[key][alpha_mask] for key in data_subset.keys()}
            
            # Sort by timestep
            sort_idx = np.argsort(alpha_data['timestep'])
            dt_sorted = alpha_data['timestep'][sort_idx]
            boundary_sorted = alpha_data['boundary_altitude'][sort_idx]
            
            # Filter out NaN values
            valid = ~np.isnan(boundary_sorted)
            if np.any(valid):
                ax.plot(dt_sorted[valid], boundary_sorted[valid], 'o-', 
                       color=colors2[alpha_idx], label=f'α={alpha:.2f}', 
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Timestep (s)', fontsize=10)
        ax.set_ylabel('Boundary Altitude (km)', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes2_flat)):
        axes2_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_vs_timestep.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Plot 3: Heatmap of boundary altitude vs mixing length and timestep
    for n_layers in sorted(n_layers_list):
        fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
        fig3.suptitle(f'Radiative-Convective Boundary Altitude (n_layers = {n_layers})', 
                      fontsize=16, fontweight='bold')
        
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        # Create 2D grid for boundary altitude
        boundary_grid = np.full((len(timestep_list), len(alpha_list)), np.nan)
        
        for i, dt in enumerate(timestep_list):
            for j, alpha in enumerate(alpha_list):
                match = (data_subset['timestep'] == dt) & (data_subset[alpha_key] == alpha)
                if np.any(match):
                    boundary_grid[i, j] = data_subset['boundary_altitude'][match][0]
        
        # Plot 1: Linear scale
        ax1 = axes3[0]
        x_coords = np.array(alpha_list)
        y_coords = np.array(timestep_list)
        
        # Create edges for pcolormesh
        if len(x_coords) > 1:
            x_midpoints = (x_coords[:-1] + x_coords[1:]) / 2.0
            x_first_edge = x_coords[0] - (x_midpoints[0] - x_coords[0])
            x_last_edge = x_coords[-1] + (x_coords[-1] - x_midpoints[-1])
            x_edges = np.concatenate([[x_first_edge], x_midpoints, [x_last_edge]])
        else:
            x_edges = np.array([x_coords[0] - 0.1, x_coords[0] + 0.1])
            
        if len(y_coords) > 1:
            y_midpoints = (y_coords[:-1] + y_coords[1:]) / 2.0
            y_first_edge = y_coords[0] - (y_midpoints[0] - y_coords[0])
            y_last_edge = y_coords[-1] + (y_coords[-1] - y_midpoints[-1])
            y_edges = np.concatenate([[y_first_edge], y_midpoints, [y_last_edge]])
        else:
            y_edges = np.array([y_coords[0] - 0.1, y_coords[0] + 0.1])
        
        X, Y = np.meshgrid(x_edges, y_edges)
        im1 = ax1.pcolormesh(X, Y, boundary_grid, cmap='viridis', shading='flat')
        ax1.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax1.set_ylabel('Timestep (s)', fontsize=10)
        ax1.set_title('Boundary Altitude (km) - Linear Scale', fontsize=12)
        ax1.set_xticks(x_coords)
        ax1.set_xticklabels([f'{l:.2f}' for l in x_coords], rotation=45, ha='right')
        ax1.set_yticks(y_coords)
        ax1.set_yticklabels([f'{dt:.0f}' for dt in y_coords])
        plt.colorbar(im1, ax=ax1, label='Boundary Altitude (km)')
        
        # Plot 2: Log scale
        ax2 = axes3[1]
        x_log = np.log10(x_coords)
        y_log = np.log10(y_coords)
        
        if len(x_coords) > 1:
            x_midpoints_log = (x_log[:-1] + x_log[1:]) / 2.0
            x_first_edge_log = x_log[0] - (x_midpoints_log[0] - x_log[0])
            x_last_edge_log = x_log[-1] + (x_log[-1] - x_midpoints_log[-1])
            x_edges_log = np.concatenate([[x_first_edge_log], x_midpoints_log, [x_last_edge_log]])
        else:
            x_edges_log = np.array([x_log[0] - 0.1, x_log[0] + 0.1])
            
        if len(y_coords) > 1:
            y_midpoints_log = (y_log[:-1] + y_log[1:]) / 2.0
            y_first_edge_log = y_log[0] - (y_midpoints_log[0] - y_log[0])
            y_last_edge_log = y_log[-1] + (y_log[-1] - y_midpoints_log[-1])
            y_edges_log = np.concatenate([[y_first_edge_log], y_midpoints_log, [y_last_edge_log]])
        else:
            y_edges_log = np.array([y_log[0] - 0.1, y_log[0] + 0.1])
        
        x_edges2 = 10**x_edges_log
        y_edges2 = 10**y_edges_log
        X2, Y2 = np.meshgrid(x_edges2, y_edges2)
        im2 = ax2.pcolormesh(X2, Y2, boundary_grid, cmap='viridis', shading='flat')
        ax2.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax2.set_ylabel('Timestep (s)', fontsize=10)
        ax2.set_title('Boundary Altitude (km) - Log Scale', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xticks(x_coords)
        ax2.set_xticklabels([f'{l:.2f}' for l in x_coords], rotation=45, ha='right')
        ax2.set_yticks(y_coords)
        ax2.set_yticklabels([f'{dt:.0f}' for dt in y_coords])
        plt.colorbar(im2, ax=ax2, label='Boundary Altitude (km)')
        
        plt.tight_layout()
        output_file = f'{output_prefix}_heatmap_nlayers{n_layers}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


if __name__ == "__main__":
    # Reduced parameter ranges: 3×3×3 = 27 combinations
    n_layers_list = [10, 50, 100]
    alpha_list = [0.1, 0.5, 1.0]
    timestep_list = [1, 10, 100]
    
    print("Starting radiative-convective boundary analysis...")
    print(f"Total combinations: {len(n_layers_list)} × {len(alpha_list)} × {len(timestep_list)} = {len(n_layers_list) * len(alpha_list) * len(timestep_list)}")
    
    results = run_boundary_analysis(
        n_layers_list=n_layers_list,
        alpha_list=alpha_list,
        timestep_list=timestep_list,
        max_steps=50000  # Use fewer steps for faster analysis
    )
    
    # Save results
    output_file = "radiative_boundary_results.npz"
    np.savez(output_file, **results)
    print(f"\nResults saved to: {output_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_boundary_analysis(results)
    
    print("\nAnalysis complete!")
