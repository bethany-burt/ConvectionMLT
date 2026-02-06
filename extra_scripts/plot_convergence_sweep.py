"""
Visualization script for convergence parameter sweep results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(input_file="convergence_sweep_results.npz"):
    """Load sweep results from .npz file"""
    data = np.load(input_file)
    results = {key: data[key] for key in data.keys()}
    return results


def plot_heatmaps(results, output_prefix="convergence"):
    """
    Plot Type A: Heatmaps showing steps to converge vs alpha and timestep.
    One subplot per n_layers.
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
    
    # Adjust grid based on number of n_layers
    n_nlayers = len(n_layers_list)
    if n_nlayers <= 3:
        n_rows = 1
        n_cols = n_nlayers
        figsize = (6 * n_cols, 6)
    else:
        n_rows = 2
        n_cols = 3
        figsize = (18, 12)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Steps to Converge: Heatmaps (steps to converge vs alpha and timestep)', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes_flat[idx]
        
        # Filter data for this n_layers
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        # Create 2D grid
        steps_grid = np.full((len(timestep_list), len(alpha_list)), np.nan)
        no_conv_mask = np.zeros((len(timestep_list), len(alpha_list)), dtype=bool)
        
        for i, dt in enumerate(timestep_list):
            for j, alpha in enumerate(alpha_list):
                # Find matching entry
                match = (data_subset['timestep'] == dt) & (data_subset[alpha_key] == alpha)
                if np.any(match):
                    steps_grid[i, j] = data_subset['steps_to_converge'][match][0]
                    # Check if no convective layers
                    if 'n_convective' in data_subset:
                        n_conv = data_subset['n_convective'][match][0]
                        if np.isfinite(n_conv) and n_conv == 0:
                            no_conv_mask[i, j] = True
        
        # Create heatmap using pcolormesh with linear scale
        x_coords = np.array(alpha_list)
        y_coords = np.array(timestep_list)
        
        # Create edges for pcolormesh (arithmetic means for linear scale)
        # Calculate edges as midpoints between consecutive values
        if len(x_coords) > 1:
            x_midpoints = (x_coords[:-1] + x_coords[1:]) / 2.0
            # First and last edges: extend by same spacing as first/last interval
            x_first_edge = x_coords[0] - (x_midpoints[0] - x_coords[0])
            x_last_edge = x_coords[-1] + (x_coords[-1] - x_midpoints[-1])
            x_edges = np.concatenate([[x_first_edge], x_midpoints, [x_last_edge]])
        else:
            # Handle single value case
            x_edges = np.array([x_coords[0] - 0.1, x_coords[0] + 0.1])
            
        if len(y_coords) > 1:
            y_midpoints = (y_coords[:-1] + y_coords[1:]) / 2.0
            # First and last edges: extend by same spacing as first/last interval
            y_first_edge = y_coords[0] - (y_midpoints[0] - y_coords[0])
            y_last_edge = y_coords[-1] + (y_coords[-1] - y_midpoints[-1])
            y_edges = np.concatenate([[y_first_edge], y_midpoints, [y_last_edge]])
        else:
            # Handle single value case
            y_edges = np.array([y_coords[0] - 0.1, y_coords[0] + 0.1])
        
        X, Y = np.meshgrid(x_edges, y_edges)
        
        # Use log scale for color mapping since steps vary over orders of magnitude
        # Handle NaN values
        steps_grid_safe = np.where(np.isfinite(steps_grid), steps_grid, np.nan)
        steps_grid_safe = np.maximum(steps_grid_safe, 1.0)  # Floor at 1 for log
        steps_grid_log = np.log10(steps_grid_safe)
        
        # Set axis labels and linear scale
        ax.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax.set_ylabel('Timestep (s)', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        
        # Set axis limits to encompass all edges
        ax.set_xlim([x_edges[0], x_edges[-1]])
        ax.set_ylim([y_edges[0], y_edges[-1]])
        
        # Now plot with linear scale
        im = ax.pcolormesh(X, Y, steps_grid_log, cmap='viridis_r', shading='flat')
        
        # Grey out areas with no convective layers
        if np.any(no_conv_mask):
            # Create a mask grid for pcolormesh
            no_conv_grid = np.full_like(steps_grid_log, np.nan)
            no_conv_grid[no_conv_mask] = np.nanmin(steps_grid_log) - 0.5  # Below colorbar range
            # Overlay grey patches
            ax.pcolormesh(X, Y, no_conv_grid, cmap='gray', alpha=0.5, shading='flat', vmin=-10, vmax=-5)
        
        # Set ticks to actual parameter values (these should align with cell centers)
        ax.set_xticks(x_coords)
        ax.set_xticklabels([f'{l:.3f}' if l < 0.1 else f'{l:.2f}' for l in x_coords], rotation=45, ha='right')
        ax.set_yticks(y_coords)
        ax.set_yticklabels([f'{dt:.0f}' if dt >= 1 else f'{dt:.2f}' for dt in y_coords])
        
        # Add colorbar with log scale labels
        cbar = plt.colorbar(im, ax=ax)
        # Find reasonable tick positions in log space for colorbar
        min_val = np.nanmin(steps_grid_safe)
        max_val = np.nanmax(steps_grid_safe)
        if max_val > min_val and np.isfinite(min_val) and np.isfinite(max_val):
            log_min = np.floor(np.log10(min_val))
            log_max = np.ceil(np.log10(max_val))
            cbar_ticks_log = np.arange(log_min, log_max + 1)
            cbar_ticks_vals = 10**cbar_ticks_log
            # Only include ticks that are within the data range
            valid_mask = (cbar_ticks_vals >= min_val) & (cbar_ticks_vals <= max_val)
            cbar_ticks_log = cbar_ticks_log[valid_mask]
            cbar_ticks_vals = 10**cbar_ticks_log
            cbar.set_ticks(cbar_ticks_log)
            cbar.set_ticklabels([f'{int(v)}' if v >= 1 else f'{v:.1f}' for v in cbar_ticks_vals])
        cbar.set_label('Steps to Converge (log scale)', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_heatmaps.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_heatmaps_logscale(results, output_prefix="convergence"):
    """
    Plot Type A (Log Scale): Steps to converge heatmaps with log scale axes.
    One plot per n_layers, showing steps to converge vs alpha and timestep.
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
    
    # Adjust grid based on number of n_layers
    n_nlayers = len(n_layers_list)
    if n_nlayers <= 3:
        n_rows = 1
        n_cols = n_nlayers
        figsize = (6 * n_cols, 6)
    else:
        n_rows = 2
        n_cols = 3
        figsize = (18, 12)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Steps to Converge: Heatmaps with Log Scale Axes (steps vs alpha and timestep)', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes_flat[idx]
        
        # Filter data for this n_layers
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        # Create 2D grid
        steps_grid = np.full((len(timestep_list), len(alpha_list)), np.nan)
        
        for i, dt in enumerate(timestep_list):
            for j, alpha in enumerate(alpha_list):
                # Find matching entry
                match = (data_subset['timestep'] == dt) & (data_subset[alpha_key] == alpha)
                if np.any(match):
                    steps_grid[i, j] = data_subset['steps_to_converge'][match][0]
        
        # Create heatmap using pcolormesh with log scale
        x_coords = np.array(alpha_list)
        y_coords = np.array(timestep_list)
        
        # Create edges for pcolormesh (geometric spacing for log-scale data)
        # For log-scale data, edges between points should be geometric means
        x_log = np.log10(x_coords)
        y_log = np.log10(y_coords)
        
        # Calculate edges as geometric means between consecutive values
        if len(x_coords) > 1:
            # Edges between points: geometric mean in log space = arithmetic mean
            x_midpoints_log = (x_log[:-1] + x_log[1:]) / 2.0
            # First and last edges: extend by same spacing as first/last interval
            x_first_edge_log = x_log[0] - (x_midpoints_log[0] - x_log[0])
            x_last_edge_log = x_log[-1] + (x_log[-1] - x_midpoints_log[-1])
            x_edges_log = np.concatenate([[x_first_edge_log], x_midpoints_log, [x_last_edge_log]])
        else:
            # Handle single value case
            x_edges_log = np.array([x_log[0] - 0.1, x_log[0] + 0.1])
            
        if len(y_coords) > 1:
            # Edges between points: geometric mean in log space = arithmetic mean
            y_midpoints_log = (y_log[:-1] + y_log[1:]) / 2.0
            # First and last edges: extend by same spacing as first/last interval
            y_first_edge_log = y_log[0] - (y_midpoints_log[0] - y_log[0])
            y_last_edge_log = y_log[-1] + (y_log[-1] - y_midpoints_log[-1])
            y_edges_log = np.concatenate([[y_first_edge_log], y_midpoints_log, [y_last_edge_log]])
        else:
            # Handle single value case
            y_edges_log = np.array([y_log[0] - 0.1, y_log[0] + 0.1])
        
        # Convert back to linear space for pcolormesh
        x_edges = 10**x_edges_log
        y_edges = 10**y_edges_log
        X, Y = np.meshgrid(x_edges, y_edges)
        
        # Use log scale for color mapping since steps vary over orders of magnitude
        # Handle NaN values
        steps_grid_safe = np.where(np.isfinite(steps_grid), steps_grid, np.nan)
        steps_grid_safe = np.maximum(steps_grid_safe, 1.0)  # Floor at 1 for log
        steps_grid_log = np.log10(steps_grid_safe)
        
        # Set axis labels and log scale
        ax.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax.set_ylabel('Timestep (s)', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set axis limits to encompass all edges
        ax.set_xlim([x_edges[0], x_edges[-1]])
        ax.set_ylim([y_edges[0], y_edges[-1]])
        
        # Now plot
        im = ax.pcolormesh(X, Y, steps_grid_log, cmap='viridis_r', shading='flat')
        
        # Set ticks to actual parameter values (these should align with cell centers)
        ax.set_xticks(x_coords)
        ax.set_xticklabels([f'{l:.3f}' if l < 0.1 else f'{l:.2f}' for l in x_coords], rotation=45, ha='right')
        ax.set_yticks(y_coords)
        ax.set_yticklabels([f'{dt:.0f}' if dt >= 1 else f'{dt:.2f}' for dt in y_coords])
        
        # Add colorbar with log scale labels
        cbar = plt.colorbar(im, ax=ax)
        # Find reasonable tick positions in log space for colorbar
        min_val = np.nanmin(steps_grid_safe)
        max_val = np.nanmax(steps_grid_safe)
        if max_val > min_val and np.isfinite(min_val) and np.isfinite(max_val):
            log_min = np.floor(np.log10(min_val))
            log_max = np.ceil(np.log10(max_val))
            cbar_ticks_log = np.arange(log_min, log_max + 1)
            cbar_ticks_vals = 10**cbar_ticks_log
            # Only include ticks that are within the data range
            valid_mask = (cbar_ticks_vals >= min_val) & (cbar_ticks_vals <= max_val)
            cbar_ticks_log = cbar_ticks_log[valid_mask]
            cbar_ticks_vals = 10**cbar_ticks_log
            cbar.set_ticks(cbar_ticks_log)
            cbar.set_ticklabels([f'{int(v)}' if v >= 1 else f'{v:.1f}' for v in cbar_ticks_vals])
        cbar.set_label('Steps to Converge (log scale)', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_heatmaps_logscale.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_convergence_boundaries(results, output_prefix="convergence"):
    """
    Plot Type B: Convergence boundary plots showing converged vs non-converged.
    One plot per n_layers.
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
    
    # Adjust grid based on number of n_layers
    n_nlayers = len(n_layers_list)
    if n_nlayers <= 3:
        n_rows = 1
        n_cols = n_nlayers
        figsize = (6 * n_cols, 6)
    else:
        n_rows = 2
        n_cols = 3
        figsize = (18, 12)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Convergence Boundaries (green=converged, red=not converged)',
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes_flat[idx]
        
        # Filter data for this n_layers
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        # Create 2D grid
        converged_grid = np.full((len(timestep_list), len(alpha_list)), np.nan)
        
        for i, dt in enumerate(timestep_list):
            for j, alpha in enumerate(alpha_list):
                # Find matching entry
                match = (data_subset['timestep'] == dt) & (data_subset[alpha_key] == alpha)
                if np.any(match):
                    converged_grid[i, j] = 1.0 if data_subset['converged'][match][0] else 0.0
        
        # Create heatmap with binary colors
        im = ax.imshow(converged_grid, aspect='auto', origin='lower', 
                       cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        
        # Set ticks and labels
        # Set ticks and labels (log scale positioning)
        ax.set_xticks(range(len(alpha_list)))
        ax.set_xticklabels([f'{alpha:.2f}' for alpha in alpha_list])
        ax.set_yticks(range(len(timestep_list)))
        ax.set_yticklabels([f'{dt:.0f}' for dt in timestep_list])
        
        ax.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax.set_ylabel('Timestep (s)', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        
        # Count converged
        n_converged = np.sum(data_subset['converged'])
        n_total = len(data_subset['converged'])
        ax.text(0.02, 0.98, f'{n_converged}/{n_total} converged', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_boundaries.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_steps_vs_params(results, output_prefix="convergence"):
    """
    Plot Type C: Steps to converge vs alpha (with different timesteps as lines)
    and steps to converge vs timestep (with different alpha values as lines).
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
    
    # Plot 1: Steps vs alpha (different timesteps)
    n_nlayers = len(n_layers_list)
    if n_nlayers <= 3:
        n_rows1 = 1
        n_cols1 = n_nlayers
        figsize1 = (6 * n_cols1, 6)
    else:
        n_rows1 = 2
        n_cols1 = 3
        figsize1 = (18, 12)
    fig1, axes1 = plt.subplots(n_rows1, n_cols1, figsize=figsize1)
    fig1.suptitle('Steps to Converge vs Alpha (colored by timestep)', 
                  fontsize=16, fontweight='bold')
    # Handle both single row and multi-row cases
    if n_rows1 == 1:
        axes1_flat = axes1 if n_cols1 > 1 else [axes1]
    else:
        axes1_flat = axes1.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(timestep_list)))
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes1_flat[idx]
        
        # Filter data for this n_layers
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        # Plot one line per timestep
        for dt_idx, dt in enumerate(timestep_list):
            dt_mask = data_subset['timestep'] == dt
            if not np.any(dt_mask):
                continue
            
            dt_data = {key: data_subset[key][dt_mask] for key in data_subset.keys()}
            
            # Sort by alpha
            sort_idx = np.argsort(dt_data[alpha_key])
            alpha_sorted = dt_data[alpha_key][sort_idx]
            steps_sorted = dt_data['steps_to_converge'][sort_idx]
            
            ax.plot(alpha_sorted, steps_sorted, 'o-', color=colors[dt_idx], 
                   label=f'dt={dt:.0f}s', linewidth=2, markersize=4)
        
        ax.set_xlabel('Alpha (mixing length parameter)', fontsize=10)
        ax.set_ylabel('Steps to Converge', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes1_flat)):
        axes1_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_steps_vs_alpha.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Plot 2: Steps vs timestep (different alpha values)
    if n_nlayers <= 3:
        n_rows2 = 1
        n_cols2 = n_nlayers
        figsize2 = (6 * n_cols2, 6)
    else:
        n_rows2 = 2
        n_cols2 = 3
        figsize2 = (18, 12)
    fig2, axes2 = plt.subplots(n_rows2, n_cols2, figsize=figsize2)
    fig2.suptitle('Steps to Converge vs Timestep (colored by alpha)', 
                  fontsize=16, fontweight='bold')
    # Handle both single row and multi-row cases
    if n_rows2 == 1:
        axes2_flat = axes2 if n_cols2 > 1 else [axes2]
    else:
        axes2_flat = axes2.flatten()
    
    colors2 = plt.cm.tab10(np.linspace(0, 1, len(alpha_list)))
    
    for idx, n_layers in enumerate(sorted(n_layers_list)):
        ax = axes2_flat[idx]
        
        # Filter data for this n_layers
        mask = results['n_layers'] == n_layers
        data_subset = {key: results[key][mask] for key in results.keys()}
        
        # Plot one line per alpha
        for alpha_idx, alpha in enumerate(alpha_list):
            alpha_mask = data_subset[alpha_key] == alpha
            if not np.any(alpha_mask):
                continue
            
            alpha_data = {key: data_subset[key][alpha_mask] for key in data_subset.keys()}
            
            # Sort by timestep
            sort_idx = np.argsort(alpha_data['timestep'])
            dt_sorted = alpha_data['timestep'][sort_idx]
            steps_sorted = alpha_data['steps_to_converge'][sort_idx]
            
            ax.plot(dt_sorted, steps_sorted, 'o-', color=colors2[alpha_idx], 
                   label=f'α={alpha:.2f}', linewidth=2, markersize=4)
        
        ax.set_xlabel('Timestep (s)', fontsize=10)
        ax.set_ylabel('Steps to Converge', fontsize=10)
        ax.set_title(f'n_layers = {n_layers}', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(n_layers_list), len(axes2_flat)):
        axes2_flat[idx].axis('off')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_steps_vs_timestep.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_summary(results, output_prefix="convergence"):
    """
    Plot Type D: Summary plots showing relationships between n_layers and convergence.
    """
    n_layers_list = np.unique(results['n_layers'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Summary: Relationships with n_layers', fontsize=16, fontweight='bold')
    
    # Plot 1: Minimum steps to converge vs n_layers (for best parameters)
    ax1 = axes[0, 0]
    min_steps_per_layers = []
    for n_layers in sorted(n_layers_list):
        mask = (results['n_layers'] == n_layers) & results['converged']
        if np.any(mask):
            min_steps_per_layers.append(np.min(results['steps_to_converge'][mask]))
        else:
            min_steps_per_layers.append(np.nan)
    
    ax1.plot(sorted(n_layers_list), min_steps_per_layers, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Minimum Steps to Converge', fontsize=12)
    ax1.set_title('Minimum Steps (Best Parameters)', fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Minimum viable alpha vs n_layers
    ax2 = axes[0, 1]
    # Handle both 'alpha' and 'mixing_length' for backward compatibility
    alpha_key = 'alpha' if 'alpha' in results else 'mixing_length'
    min_alpha_per_layers = []
    for n_layers in sorted(n_layers_list):
        mask = (results['n_layers'] == n_layers) & results['converged']
        if np.any(mask):
            min_alpha_per_layers.append(np.min(results[alpha_key][mask]))
        else:
            min_alpha_per_layers.append(np.nan)
    
    ax2.plot(sorted(n_layers_list), min_alpha_per_layers, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Layers', fontsize=12)
    ax2.set_ylabel('Minimum Viable Alpha', fontsize=12)
    ax2.set_title('Minimum Alpha for Convergence', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Minimum viable timestep vs n_layers
    ax3 = axes[1, 0]
    min_dt_per_layers = []
    for n_layers in sorted(n_layers_list):
        mask = (results['n_layers'] == n_layers) & results['converged']
        if np.any(mask):
            min_dt_per_layers.append(np.min(results['timestep'][mask]))
        else:
            min_dt_per_layers.append(np.nan)
    
    ax3.plot(sorted(n_layers_list), min_dt_per_layers, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Number of Layers', fontsize=12)
    ax3.set_ylabel('Minimum Viable Timestep (s)', fontsize=12)
    ax3.set_title('Minimum Timestep for Convergence', fontsize=12)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence rate vs n_layers
    ax4 = axes[1, 1]
    convergence_rate = []
    for n_layers in sorted(n_layers_list):
        mask = results['n_layers'] == n_layers
        n_converged = np.sum(results['converged'][mask])
        n_total = np.sum(mask)
        convergence_rate.append(n_converged / n_total * 100.0)
    
    ax4.bar(sorted(n_layers_list), convergence_rate, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Number of Layers', fontsize=12)
    ax4.set_ylabel('Convergence Rate (%)', fontsize=12)
    ax4.set_title('Fraction of Parameter Combinations that Converged', fontsize=12)
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main(input_file="convergence_sweep_results.npz", output_prefix="convergence"):
    """Generate all plots"""
    print("Loading results...")
    results = load_results(input_file)
    
    print("Generating plots...")
    plot_heatmaps(results, output_prefix)
    plot_heatmaps_logscale(results, output_prefix)
    plot_convergence_boundaries(results, output_prefix)
    plot_steps_vs_params(results, output_prefix)
    plot_summary(results, output_prefix)
    
    print("\nAll plots generated!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot convergence sweep results")
    parser.add_argument("--input", type=str, default="convergence_sweep_results.npz",
                       help="Input .npz file with sweep results")
    parser.add_argument("--prefix", type=str, default="convergence",
                       help="Output plot filename prefix")
    
    args = parser.parse_args()
    main(args.input, args.prefix)
