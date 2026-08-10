#!/usr/bin/env python3
"""
Parameter Sweep Data Collection

This script runs a parameter sweep over timestep and mixing length (alpha) values,
collects comprehensive data for each combination, and saves it to a file for later
plotting in a Jupyter notebook.

Data collected for each parameter combination:
- alpha, timestep
- steps to converge
- number of convective layers
- location of convective/adiabatic/radiative layers (before and after)
- degree of convergence to adiabaticity
- temperature profiles (before and after)
- pressure profiles (before and after)
"""

import numpy as np
import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convective_grid.convective_flux_v2 import (
    run, temperature_gradient, 
    G, RHO_TOA, RHO_BOA, T_TOA, T_BOA, N_DOF, MMW, R,
    convective_timescale, radiative_timescale
)

# ============================================================================
# PARAMETER SWEEP CONFIGURATION
# ============================================================================

# Parameter ranges to explore
ALPHA_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0]  # Mixing length parameter
TIMESTEP_VALUES = [0.01, 0.1, 1, 10, 100]  # Timestep in seconds

# Fixed parameters
N_LAYERS = 100
MAX_STEPS = 500000
ADIABATIC_TOLERANCE = 1.0  # 100% tolerance for convective layers

# Guillot profile parameters (default)
GUILLOT_PARAMS = {
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

OUTPUT_FILE = 'parameter_sweep_data_guillot.json'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_layer_type(N, N_ad, tolerance=0.01):
    """
    Classify each layer as convective, adiabatic, or radiative.
    
    Args:
        N: Temperature gradient at layer centers
        N_ad: Adiabatic temperature gradient
        tolerance: Tolerance for considering a layer adiabatic (fractional)
    
    Returns:
        Array of classifications: 'convective', 'adiabatic', or 'radiative'
    """
    classifications = np.empty(len(N), dtype=object)
    
    # Convective: N > N_ad (superadiabatic)
    convective_mask = N > N_ad
    classifications[convective_mask] = 'convective'
    
    # Adiabatic: N ≈ N_ad (within tolerance)
    adiabatic_mask = (N <= N_ad) & (np.abs(N - N_ad) / N_ad <= tolerance)
    classifications[adiabatic_mask] = 'adiabatic'
    
    # Radiative: N < N_ad (subadiabatic, stable)
    radiative_mask = (N < N_ad) & (np.abs(N - N_ad) / N_ad > tolerance)
    classifications[radiative_mask] = 'radiative'
    
    return classifications


def calculate_adiabaticity_metric(N, N_ad, convective_only=False):
    """
    Calculate how well layers have converged to adiabaticity.
    
    For convective layers: |N - N_ad| / N_ad (fractional deviation)
    For radiative layers: can be negative, but we take absolute value
    
    Args:
        N: Temperature gradient at layer centers
        N_ad: Adiabatic temperature gradient
        convective_only: If True, only calculate for convective layers
    
    Returns:
        Array of fractional deviations from adiabaticity
    """
    if convective_only:
        mask = N > N_ad
        metric = np.full(len(N), np.nan)
        metric[mask] = np.abs(N[mask] - N_ad) / N_ad
        return metric
    else:
        return np.abs(N - N_ad) / N_ad


# ============================================================================
# DATA COLLECTION FUNCTION
# ============================================================================

def collect_data_for_combination(alpha, dt, n_layers=N_LAYERS, max_steps=MAX_STEPS):
    """
    Run simulation for a single parameter combination and collect all data.
    
    Returns:
        Dictionary with all collected data, or None if run failed
    """
    try:
        # Run solver
        z, T_final, rho_final, P_final, diagnostics = run(
            n_layers=n_layers,
            max_steps=max_steps,
            alpha=alpha,
            dt=dt,
            debug=False,
            save_history=False,
            profile_type="guillot",
            guillot_params=GUILLOT_PARAMS,
            convergence_tol=1e-10,  # Very loose, we use adiabaticity check
            check_adiabatic=True,
            adiabatic_tolerance=ADIABATIC_TOLERANCE
        )
        
        # Extract data
        z_mid = (z[:-1] + z[1:]) / 2.0
        T_initial = diagnostics['T_initial']
        N_final = diagnostics['N_final']
        N_ad = diagnostics['N_ad']
        converged = diagnostics['converged_adiabatic']
        steps_to_converge = diagnostics['steps'] if converged else max_steps
        
        # Calculate initial gradient for comparison
        N_initial = temperature_gradient(T_initial, z)
        
        # Calculate initial pressure (same way as in the run function)
        R_specific = R / MMW  # erg g^-1 K^-1
        P_initial = rho_final * R_specific * T_initial  # dyne/cm^2
        
        # Calculate timescales at layer centers (final state)
        T_mid_final = (T_final[:-1] + T_final[1:]) / 2.0
        P_mid_final = (P_final[:-1] + P_final[1:]) / 2.0  # Pressure at layer centers (dyne/cm²)
        c_p = diagnostics['c_p']  # erg/(g·K)
        
        # Calculate final timescales
        t_conv_final = convective_timescale(G, T_mid_final, N_final, N_ad)
        tau_rad_final = radiative_timescale(P_mid_final, G, c_p, T_mid_final)
        
        # Classify layers
        layer_type_initial = classify_layer_type(N_initial, N_ad)
        layer_type_final = classify_layer_type(N_final, N_ad)
        
        # Count layers by type
        n_convective_initial = np.sum(layer_type_initial == 'convective')
        n_adiabatic_initial = np.sum(layer_type_initial == 'adiabatic')
        n_radiative_initial = np.sum(layer_type_initial == 'radiative')
        
        n_convective_final = np.sum(layer_type_final == 'convective')
        n_adiabatic_final = np.sum(layer_type_final == 'adiabatic')
        n_radiative_final = np.sum(layer_type_final == 'radiative')
        
        # Calculate adiabaticity metrics
        adiabaticity_initial = calculate_adiabaticity_metric(N_initial, N_ad, convective_only=False)
        adiabaticity_final = calculate_adiabaticity_metric(N_final, N_ad, convective_only=False)
        adiabaticity_final_convective = calculate_adiabaticity_metric(N_final, N_ad, convective_only=True)
        
        # Find locations (indices) of each layer type
        convective_locations_initial = np.where(layer_type_initial == 'convective')[0].tolist()
        adiabatic_locations_initial = np.where(layer_type_initial == 'adiabatic')[0].tolist()
        radiative_locations_initial = np.where(layer_type_initial == 'radiative')[0].tolist()
        
        convective_locations_final = np.where(layer_type_final == 'convective')[0].tolist()
        adiabatic_locations_final = np.where(layer_type_final == 'adiabatic')[0].tolist()
        radiative_locations_final = np.where(layer_type_final == 'radiative')[0].tolist()
        
        # Altitude ranges for each type (convert to km)
        convective_altitudes_initial = (z_mid[convective_locations_initial] / 1000.0).tolist() if len(convective_locations_initial) > 0 else []
        adiabatic_altitudes_initial = (z_mid[adiabatic_locations_initial] / 1000.0).tolist() if len(adiabatic_locations_initial) > 0 else []
        radiative_altitudes_initial = (z_mid[radiative_locations_initial] / 1000.0).tolist() if len(radiative_locations_initial) > 0 else []
        
        convective_altitudes_final = (z_mid[convective_locations_final] / 1000.0).tolist() if len(convective_locations_final) > 0 else []
        adiabatic_altitudes_final = (z_mid[adiabatic_locations_final] / 1000.0).tolist() if len(adiabatic_locations_final) > 0 else []
        radiative_altitudes_final = (z_mid[radiative_locations_final] / 1000.0).tolist() if len(radiative_locations_final) > 0 else []
        
        # Collect all data
        data = {
            # Parameters
            'alpha': float(alpha),
            'timestep': float(dt),
            'n_layers': int(n_layers),
            
            # Convergence metrics
            'converged': bool(converged),
            'steps_to_converge': int(steps_to_converge),
            'physical_time_to_converge': float(steps_to_converge * dt),  # seconds
            'max_dT_final': float(diagnostics['max_dT_final']),
            'max_grad_diff_final': float(diagnostics['max_grad_diff_final']) if diagnostics['max_grad_diff_final'] is not None else np.nan,
            
            # Layer counts
            'n_convective_initial': int(n_convective_initial),
            'n_adiabatic_initial': int(n_adiabatic_initial),
            'n_radiative_initial': int(n_radiative_initial),
            'n_convective_final': int(n_convective_final),
            'n_adiabatic_final': int(n_adiabatic_final),
            'n_radiative_final': int(n_radiative_final),
            
            # Layer locations (indices)
            'convective_locations_initial': convective_locations_initial,
            'adiabatic_locations_initial': adiabatic_locations_initial,
            'radiative_locations_initial': radiative_locations_initial,
            'convective_locations_final': convective_locations_final,
            'adiabatic_locations_final': adiabatic_locations_final,
            'radiative_locations_final': radiative_locations_final,
            
            # Layer altitudes (km)
            'convective_altitudes_initial': [float(x) for x in convective_altitudes_initial],
            'adiabatic_altitudes_initial': [float(x) for x in adiabatic_altitudes_initial],
            'radiative_altitudes_initial': [float(x) for x in radiative_altitudes_initial],
            'convective_altitudes_final': [float(x) for x in convective_altitudes_final],
            'adiabatic_altitudes_final': [float(x) for x in adiabatic_altitudes_final],
            'radiative_altitudes_final': [float(x) for x in radiative_altitudes_final],
            
            # Profiles (interfaces: n_layers + 1 points)
            'z_interfaces': [float(x) for x in z],  # meters
            'z_interfaces_km': [float(x/1000.0) for x in z],  # km
            'T_initial': [float(x) for x in T_initial],  # K
            'T_final': [float(x) for x in T_final],  # K
            'P_initial': [float(x/1e6) for x in P_initial],  # bar (convert from dyne/cm^2)
            'P_final': [float(x/1e6) for x in P_final],  # bar (convert from dyne/cm^2)
            
            # Profiles at layer centers (n_layers points)
            'z_centers': [float(x) for x in z_mid],  # meters
            'z_centers_km': [float(x/1000.0) for x in z_mid],  # km
            'N_initial': [float(x) for x in N_initial],  # K/m
            'N_final': [float(x) for x in N_final],  # K/m
            'N_ad': float(N_ad),  # K/m
            'adiabaticity_initial': [float(x) for x in adiabaticity_initial],
            'adiabaticity_final': [float(x) for x in adiabaticity_final],
            'adiabaticity_final_convective': [float(x) if not np.isnan(x) else None for x in adiabaticity_final_convective],
            
            # Layer type classifications
            'layer_type_initial': [str(x) for x in layer_type_initial],
            'layer_type_final': [str(x) for x in layer_type_final],
            
            # Additional diagnostics
            'F_conv_final': [float(x) for x in diagnostics['F_conv_final']],  # erg cm^-2 s^-1
            'rho_final': [float(x) for x in rho_final],  # g/cm^3
            
            # Timescales at layer centers (final state)
            't_conv_final': [float(x) if np.isfinite(x) else None for x in t_conv_final],  # s
            'tau_rad_final': [float(x) if np.isfinite(x) else None for x in tau_rad_final],  # s
        }
        
        return data
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Parameter Sweep Data Collection")
    print("=" * 70)
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"Timestep values: {TIMESTEP_VALUES}")
    print(f"Total combinations: {len(ALPHA_VALUES) * len(TIMESTEP_VALUES)}")
    print(f"Fixed: n_layers={N_LAYERS}, max_steps={MAX_STEPS}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    
    # Collect all data
    all_data = []
    total_runs = len(ALPHA_VALUES) * len(TIMESTEP_VALUES)
    current_run = 0
    start_time = time.time()
    
    for alpha in ALPHA_VALUES:
        for dt in TIMESTEP_VALUES:
            current_run += 1
            run_start = time.time()
            
            print(f"[{current_run}/{total_runs}] α={alpha:5.2f}, dt={dt:6.1f}s", end=" ... ")
            
            data = collect_data_for_combination(alpha, dt)
            run_time = time.time() - run_start
            
            if data is not None:
                data['run_time'] = float(run_time)
                all_data.append(data)
                
                status = "✓" if data['converged'] else "✗"
                print(f"{status} {data['steps_to_converge']:6d} steps, "
                      f"{data['n_convective_final']:3d} conv layers, "
                      f"{run_time:.1f}s")
            else:
                # Save failed run with minimal data for tracking
                failed_data = {
                    'alpha': float(alpha),
                    'timestep': float(dt),
                    'n_layers': int(N_LAYERS),
                    'converged': False,
                    'failed': True,
                    'error': 'Run failed with exception',
                    'run_time': float(run_time)
                }
                all_data.append(failed_data)
                print(f"✗ FAILED ({run_time:.1f}s)")
    
    total_time = time.time() - start_time
    
    # Save data to JSON file
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'alpha_values': ALPHA_VALUES,
            'timestep_values': TIMESTEP_VALUES,
            'n_layers': N_LAYERS,
            'max_steps': MAX_STEPS,
            'total_runs': total_runs,
            'successful_runs': len(all_data),
            'total_time_seconds': total_time,
            'guillot_params': GUILLOT_PARAMS,
        },
        'data': all_data
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("=" * 70)
    print("Data Collection Complete")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful runs: {len(all_data)}/{total_runs}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print()
    print("Data file contains:")
    print(f"  - {len(all_data)} parameter combinations")
    print("  - Each with: parameters, convergence metrics, layer classifications,")
    print("              temperature/pressure profiles, and adiabaticity metrics")
    print()
    print("You can now load this data in a Jupyter notebook using:")
    print(f"  import json")
    print(f"  with open('{OUTPUT_FILE}', 'r') as f:")
    print(f"      data = json.load(f)")


if __name__ == '__main__':
    main()
