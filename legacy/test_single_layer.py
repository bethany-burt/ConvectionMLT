"""
Test Script: Single Layer MLT Alpha Calculation

This script provides a simple interface to test the MLT flux balance
calculation with user-provided or preset values.

Usage:
    python test_single_layer.py

Author: Generated for radiative transfer convection modeling
Date: November 2025
"""

from mlt_flux_balance import calculate_alpha_from_flux
import numpy as np


def test_case_1_superadiabatic():
    """
    Test Case 1: Superadiabatic layer (should find α)
    Hot Jupiter atmosphere with strong temperature gradient
    """
    print("="*70)
    print("TEST CASE 1: Superadiabatic Layer")
    print("="*70)
    
    layer_data = {
        'T_top': 1000.0,     # K
        'T_mid': 1500.0,     # K
        'T_bot': 2250.0,     # K
        'P_top': 1e4,        # Pa (0.1 bar)
        'P_mid': 5e4,        # Pa (0.5 bar)
        'P_bot': 1e5,        # Pa (1.0 bar)
    }
    
    flux_data = {
        'F_tot': 1e7,        # W/m²
        'F_rad': 8e6,        # W/m²
    }
    
    physical_params = {
        'g': 10.0,           # m/s²
        'delta': 1.0,        # Ideal gas
        'R_universal': 8.314, # J/(mol·K)
        'mu': 0.0022,        # kg/mol (H2-rich)
        'c_p': 14000.0,      # J/(kg·K)
        'rho': 0.005,        # kg/m³
    }
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "="*70)
    print("RESULT:")
    if result['alpha'] is not None:
        print(f"  ✓ Found α = {result['alpha']:.4f}")
        print(f"  ✓ Mixing length l = {result['alpha'] * result['H_p']:.2f} m")
        print(f"  ✓ F_c = {result['F_c']:.2e} W/m²")
    else:
        print(f"  ✗ No solution: {result['convergence_info']}")
    print("="*70 + "\n")
    
    return result


def test_case_2_stable():
    """
    Test Case 2: Stable layer (should not require convection)
    Temperature gradient less than adiabatic
    """
    print("="*70)
    print("TEST CASE 2: Stable Layer (Not Superadiabatic)")
    print("="*70)
    
    layer_data = {
        'T_top': 1500.0,     # K
        'T_mid': 1520.0,     # K
        'T_bot': 1540.0,     # K (gentle gradient)
        'P_top': 1e4,        # Pa
        'P_mid': 5e4,        # Pa
        'P_bot': 1e5,        # Pa
    }
    
    flux_data = {
        'F_tot': 1e7,        # W/m²
        'F_rad': 9.9e6,      # W/m² (nearly all radiative)
    }
    
    physical_params = {
        'g': 10.0,
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.0022,
        'c_p': 14000.0,
        'rho': 0.005,
    }
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "="*70)
    print("RESULT:")
    if result['alpha'] is None:
        print(f"  ✓ Correctly identified as stable: {result['convergence_info']}")
    else:
        print(f"  ? Unexpected: Found α = {result['alpha']:.4f}")
    print("="*70 + "\n")
    
    return result


def test_case_3_high_flux():
    """
    Test Case 3: High required flux
    Tests boundary conditions when large α is needed
    """
    print("="*70)
    print("TEST CASE 3: High Required Convective Flux")
    print("="*70)
    
    layer_data = {
        'T_top': 1000.0,     # K
        'T_mid': 1500.0,     # K
        'T_bot': 2250.0,     # K
        'P_top': 1e4,        # Pa
        'P_mid': 5e4,        # Pa
        'P_bot': 1e5,        # Pa
    }
    
    flux_data = {
        'F_tot': 1e8,        # W/m² (very high total flux)
        'F_rad': 2e7,        # W/m² (large convective need)
    }
    
    physical_params = {
        'g': 10.0,
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.0022,
        'c_p': 14000.0,
        'rho': 0.005,
    }
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "="*70)
    print("RESULT:")
    if result['alpha'] is not None:
        print(f"  ✓ Found α = {result['alpha']:.4f}")
        print(f"  ✓ Mixing length l = {result['alpha'] * result['H_p']:.2f} m")
        if result['alpha'] > 3.0:
            print(f"  ⚠ Warning: Large α value (> 3.0), may be unphysical")
    else:
        print(f"  ✗ No solution: {result['convergence_info']}")
    print("="*70 + "\n")
    
    return result


def test_case_4_earth_like():
    """
    Test Case 4: Earth-like atmosphere
    More typical atmospheric parameters
    """
    print("="*70)
    print("TEST CASE 4: Earth-like Atmosphere")
    print("="*70)
    
    layer_data = {
        'T_top': 280.0,      # K
        'T_mid': 290.0,      # K
        'T_bot': 300.0,      # K
        'P_top': 8e4,        # Pa (0.8 bar)
        'P_mid': 9e4,        # Pa (0.9 bar)
        'P_bot': 1e5,        # Pa (1.0 bar)
    }
    
    flux_data = {
        'F_tot': 400.0,      # W/m² (typical solar heating)
        'F_rad': 300.0,      # W/m²
    }
    
    physical_params = {
        'g': 9.81,           # m/s² (Earth)
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.029,         # kg/mol (air)
        'c_p': 1005.0,       # J/(kg·K) (air)
        'rho': 1.2,          # kg/m³ (sea level)
    }
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "="*70)
    print("RESULT:")
    if result['alpha'] is not None:
        print(f"  ✓ Found α = {result['alpha']:.4f}")
        print(f"  ✓ Mixing length l = {result['alpha'] * result['H_p']:.2f} m")
        if 1.0 <= result['alpha'] <= 2.5:
            print(f"  ✓ α in typical range [1.0, 2.5]")
    else:
        print(f"  ○ {result['convergence_info']}")
    print("="*70 + "\n")
    
    return result


def test_custom_values():
    """
    Template for custom user-provided values.
    Edit the values below to test your specific case.
    """
    print("="*70)
    print("CUSTOM TEST: User-Provided Values")
    print("="*70)
    
    # ========== EDIT THESE VALUES ==========
    layer_data = {
        'T_top': 1200.0,     # K - YOUR VALUE
        'T_mid': 1400.0,     # K - YOUR VALUE
        'T_bot': 1600.0,     # K - YOUR VALUE
        'P_top': 1e4,        # Pa - YOUR VALUE
        'P_mid': 5e4,        # Pa - YOUR VALUE
        'P_bot': 1e5,        # Pa - YOUR VALUE
    }
    
    flux_data = {
        'F_tot': 1e7,        # W/m² - YOUR VALUE
        'F_rad': 8e6,        # W/m² - YOUR VALUE
    }
    
    physical_params = {
        'g': 10.0,           # m/s² - YOUR VALUE
        'delta': 1.0,        # dimensionless - YOUR VALUE
        'R_universal': 8.314, # J/(mol·K)
        'mu': 0.0022,        # kg/mol - YOUR VALUE
        'c_p': 14000.0,      # J/(kg·K) - YOUR VALUE
        'rho': 0.005,        # kg/m³ - YOUR VALUE
    }
    # =======================================
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "="*70)
    print("RESULT:")
    if result['alpha'] is not None:
        print(f"  α = {result['alpha']:.4f}")
        print(f"  l = {result['alpha'] * result['H_p']:.2f} m")
        print(f"  F_c = {result['F_c']:.2e} W/m²")
    else:
        print(f"  {result['convergence_info']}")
    print("="*70 + "\n")
    
    return result


def main():
    """
    Run all test cases.
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  MLT ALPHA CALCULATION - TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    results = {}
    
    # Run test cases
    results['test1'] = test_case_1_superadiabatic()
    results['test2'] = test_case_2_stable()
    results['test3'] = test_case_3_high_flux()
    results['test4'] = test_case_4_earth_like()
    
    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SUMMARY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    for name, result in results.items():
        status = "✓" if result['alpha'] is not None or not result['is_superadiabatic'] else "✗"
        alpha_str = f"α = {result['alpha']:.4f}" if result['alpha'] is not None else "No solution"
        print(f"{status} {name}: {alpha_str}")
    
    print("\n" + "█"*70 + "\n")
    
    print("To test custom values, edit the test_custom_values() function")
    print("or call calculate_alpha_from_flux() directly with your parameters.\n")


if __name__ == "__main__":
    main()




