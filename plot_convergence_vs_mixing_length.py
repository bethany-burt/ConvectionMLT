#!/usr/bin/env python3
"""
Plot Steps to Convergence vs Mixing Length (Alpha) for a given Timestep

Usage:
    python plot_convergence_vs_mixing_length.py --timestep 100
    python plot_convergence_vs_mixing_length.py --timestep 10 --output convergence_vs_alpha_dt10.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Default data file
DEFAULT_DATA_FILE = 'parameter_sweep_data.json'


def load_data(data_file):
    """Load parameter sweep data from JSON file."""
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        print(f"Please run 'collect_parameter_sweep_data.py' first to generate the data.")
        sys.exit(1)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_convergence_vs_mixing_length(data, timestep, output_file=None):
    """
    Plot steps to converge vs mixing length (alpha) for a given timestep.
    
    Args:
        data: Dictionary with 'metadata' and 'data' keys from JSON file
        timestep: Timestep value to filter by (seconds)
        output_file: Optional output filename
    """
    # Filter data for the specified timestep
    runs_at_timestep = [run for run in data['data'] if abs(run['timestep'] - timestep) < 1e-6]
    
    if len(runs_at_timestep) == 0:
        print(f"Error: No runs found with timestep = {timestep} s")
        print(f"Available timesteps: {sorted(set(r['timestep'] for r in data['data']))}")
        sys.exit(1)
    
    # Extract alpha and steps_to_converge
    alphas = [run['alpha'] for run in runs_at_timestep]
    steps = [run['steps_to_converge'] for run in runs_at_timestep]
    physical_time = [run['physical_time_to_converge'] / 3600.0 for run in runs_at_timestep]  # hours
    converged = [run['converged'] for run in runs_at_timestep]
    
    # Sort by alpha for clean plotting
    sort_idx = np.argsort(alphas)
    alphas = np.array(alphas)[sort_idx]
    steps = np.array(steps)[sort_idx]
    physical_time = np.array(physical_time)[sort_idx]
    converged = np.array(converged)[sort_idx]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Steps to converge vs alpha
    ax1.plot(alphas, steps, 'o-', linewidth=2, markersize=10, color='steelblue', label='Converged')
    
    # Mark non-converged runs differently
    not_converged_mask = ~np.array(converged)
    if np.any(not_converged_mask):
        ax1.scatter(alphas[not_converged_mask], steps[not_converged_mask], 
                   marker='x', s=200, color='red', linewidths=3, label='Not converged', zorder=10)
    
    ax1.set_xlabel('Mixing Length Parameter (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Steps to Converge', fontsize=12, fontweight='bold')
    ax1.set_title(f'Steps to Converge vs Mixing Length\nTimestep = {timestep} s', 
                 fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Add annotations for alpha values
    for i, (alpha, step) in enumerate(zip(alphas, steps)):
        if i % 2 == 0 or len(alphas) <= 5:  # Annotate all if few points, otherwise every other
            ax1.annotate(f'α={alpha:.2f}', (alpha, step), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9, alpha=0.7)
    
    # Plot 2: Physical time to converge vs alpha
    ax2.plot(alphas, physical_time, 's-', linewidth=2, markersize=10, color='darkgreen', label='Converged')
    
    if np.any(not_converged_mask):
        ax2.scatter(alphas[not_converged_mask], physical_time[not_converged_mask], 
                   marker='x', s=200, color='red', linewidths=3, label='Not converged', zorder=10)
    
    ax2.set_xlabel('Mixing Length Parameter (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Physical Time to Converge (hours)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Physical Time vs Mixing Length\nTimestep = {timestep} s', 
                 fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    # Add annotations
    for i, (alpha, time) in enumerate(zip(alphas, physical_time)):
        if i % 2 == 0 or len(alphas) <= 5:
            ax2.annotate(f'{time:.2f}h', (alpha, time), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary for timestep = {timestep} s:")
    print(f"  Total runs: {len(runs_at_timestep)}")
    print(f"  Converged: {np.sum(converged)}")
    print(f"  Not converged: {np.sum(~converged)}")
    print(f"  Alpha range: [{min(alphas):.3f}, {max(alphas):.3f}]")
    if np.any(converged):
        converged_steps = steps[converged]
        print(f"  Steps range (converged): [{min(converged_steps):.0f}, {max(converged_steps):.0f}]")
        converged_time = physical_time[converged]
        print(f"  Time range (converged): [{min(converged_time):.2f}, {max(converged_time):.2f}] hours")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Plot steps to convergence vs mixing length for a given timestep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_convergence_vs_mixing_length.py --timestep 100
  python plot_convergence_vs_mixing_length.py --timestep 10 --output plots/convergence_dt10.png
  python plot_convergence_vs_mixing_length.py --timestep 1000 --data my_data.json
        """
    )
    
    parser.add_argument('--timestep', '-t', type=float, required=True,
                       help='Timestep value (seconds) to filter runs')
    parser.add_argument('--data', '-d', type=str, default=DEFAULT_DATA_FILE,
                       help=f'Path to parameter sweep data JSON file (default: {DEFAULT_DATA_FILE})')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output filename for plot (default: show interactively)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = load_data(args.data)
    
    print(f"Total runs in dataset: {len(data['data'])}")
    print(f"Filtering for timestep = {args.timestep} s\n")
    
    # Create plot
    plot_convergence_vs_mixing_length(data, args.timestep, args.output)


if __name__ == '__main__':
    main()
