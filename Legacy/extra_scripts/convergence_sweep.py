"""
Parameter sweep to find relationships between alpha (mixing length parameter), timestep, 
and convergence rate for different numbers of layers.

Convergence is defined as: temperature gradients are similar to adiabatic gradient
(i.e., within ~20% of N_ad based on user's requirement).

This version uses Guillot TP profile instead of linear profile.
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convective_grid.convective_flux_v2 import (
    run, check_adiabatic_convergence, temperature_gradient, 
    G, RHO_TOA, RHO_BOA, T_TOA, T_BOA, N_DOF, MMW
)


def run_sweep(n_layers_list, alpha_list, timestep_list, 
              max_steps=100000, adiabatic_tolerance=0.2, 
              output_file="convergence_sweep_results_guillot.npz",
              guillot_params=None):
    """
    Run parameter sweep and track convergence.
    
    Args:
        n_layers_list: List of number of layers to test
        alpha_list: List of alpha (mixing length parameter) values to test
        timestep_list: List of timestep values to test
        max_steps: Maximum number of steps per run
        adiabatic_tolerance: Tolerance for adiabatic convergence (fractional, default 0.2 = 20%)
        output_file: Output file for results
    
    Returns:
        Dictionary with sweep results
    """
    # Default Guillot parameters (same as --no-prompt defaults)
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
    print("Convergence Parameter Sweep (Guillot TP Profile)")
    print("=" * 70)
    print(f"n_layers: {n_layers_list}")
    print(f"alpha: {alpha_list}")
    print(f"timestep: {timestep_list}")
    print(f"max_steps: {max_steps}")
    print(f"adiabatic_tolerance: {adiabatic_tolerance} (fractional)")
    print(f"TP Profile: Guillot")
    print(f"  tint = {guillot_params['tint']} K")
    print(f"  tirr = {guillot_params['tirr']} K")
    print(f"  kappa_S = {guillot_params['kappa_S']} cm^2/g")
    print(f"  kappa0 = {guillot_params['kappa0']} cm^2/g")
    print()
    
    # Initialize results arrays
    n_combinations = len(n_layers_list) * len(alpha_list) * len(timestep_list)
    
    results = {
        'n_layers': [],
        'alpha': [],
        'timestep': [],
        'steps_to_converge': [],
        'converged': [],
        'final_max_dt': [],
        'final_max_grad_diff': [],  # Maximum |N - N_ad| / N_ad across CONVECTIVE layers only
        'n_convective': [],  # Number of convective layers (N > N_ad)
        'run_time': []
    }
    
    total_runs = 0
    start_time = time.time()
    
    # Run sweep
    for n_layers in n_layers_list:
        for alpha in alpha_list:
            for timestep in timestep_list:
                total_runs += 1
                run_start = time.time()
                
                print(f"[{total_runs}/{n_combinations}] n_layers={n_layers:3d}, "
                      f"alpha={alpha:5.2f}, dt={timestep:6.1f}s", end=" ... ")
                
                try:
                    # Run solver with adiabaticity checking enabled
                    z, T, rho, P, diagnostics = run(
                        n_layers=n_layers,
                        max_steps=max_steps,
                        alpha=alpha,
                        dt=timestep,
                        debug=False,
                        save_history=False,
                        profile_type="guillot",
                        guillot_params=guillot_params,
                        convergence_tol=1e-10,  # Very loose, we use adiabaticity check instead
                        check_adiabatic=True,    # Enable adiabaticity convergence checking
                        adiabatic_tolerance=1.0   # 100% tolerance (only checks convective layers)
                    )
                    
                    # Get convergence information from diagnostics
                    converged = diagnostics['converged_adiabatic']
                    steps_to_converge = diagnostics['steps'] if converged else max_steps
                    max_grad_diff = diagnostics['max_grad_diff_final']
                    
                    # Count convective layers (N > N_ad)
                    N_final = diagnostics['N_final']
                    N_ad = diagnostics['N_ad']
                    n_convective = np.sum(N_final > N_ad)
                    
                    run_time = time.time() - run_start
                    
                    results['n_layers'].append(n_layers)
                    results['alpha'].append(alpha)
                    results['timestep'].append(timestep)
                    results['steps_to_converge'].append(steps_to_converge)
                    results['converged'].append(converged)
                    results['final_max_dt'].append(diagnostics['max_dT_final'])
                    results['final_max_grad_diff'].append(max_grad_diff)
                    results['n_convective'].append(n_convective)
                    results['run_time'].append(run_time)
                    
                    status = "CONVERGED" if converged else "NOT CONVERGED"
                    print(f"{status} ({steps_to_converge} steps, {run_time:.1f}s, "
                          f"max_diff={max_grad_diff:.4f})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    # Record failed run
                    results['n_layers'].append(n_layers)
                    results['alpha'].append(alpha)
                    results['timestep'].append(timestep)
                    results['steps_to_converge'].append(max_steps)
                    results['converged'].append(False)
                    results['final_max_dt'].append(np.nan)
                    results['final_max_grad_diff'].append(np.nan)
                    results['n_convective'].append(np.nan)
                    results['run_time'].append(time.time() - run_start)
    
    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print(f"Sweep complete! Total time: {total_time/60:.1f} minutes")
    print(f"Converged: {np.sum(results['converged'])} / {len(results['converged'])}")
    print("=" * 70)
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # Save results
    np.savez(output_file, **results)
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Parameter ranges for Guillot profile analysis
    n_layers_list = [10, 50, 100]
    alpha_list = [0.1, 0.5, 1.0]
    timestep_list = [1, 10, 100]
    
    # Guillot parameters (using defaults)
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
    
    # Run sweep
    results = run_sweep(
        n_layers_list=n_layers_list,
        alpha_list=alpha_list,
        timestep_list=timestep_list,
        max_steps=100000,
        adiabatic_tolerance=1.0,  # 100% tolerance (only checks convective layers, N can be 0 to 2*N_ad)
        output_file="convergence_sweep_results_guillot.npz",
        guillot_params=guillot_params
    )
