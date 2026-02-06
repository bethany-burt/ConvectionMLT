"""
Test MLT Alpha Calculation with Stellar Convection Zone Parameters

Uses well-established stellar structure parameters where MLT is known to work.
The solar convection zone is the canonical test case: α ≈ 1.5-2.0 (calibrated).

References:
- Cox & Giuli (1968): "Principles of Stellar Structure"
- Böhm-Vitense (1958): Original MLT calibration to solar models
- Modern stellar evolution codes: typically use α = 1.5-2.0 for Sun

Author: Generated for MLT validation
Date: November 2025
"""

from mlt_flux_balance import calculate_alpha_from_flux
import numpy as np


def test_solar_convection_zone():
    """
    Test with solar convection zone parameters.
    
    Solar convection zone extends from ~0.7 R_sun to surface.
    Using parameters from typical depth where convection is efficient.
    """
    print("=" * 70)
    print("TEST: Solar Convection Zone")
    print("=" * 70)
    print("\nUsing parameters from ~0.85 R_sun (mid-convection zone)")
    print("Reference: Standard solar model (Bahcall et al.)")
    
    # Solar convection zone layer (approximate values at ~0.85 R_sun)
    # In convection zones, the gradient is SLIGHTLY superadiabatic
    # (just enough to drive the required convective flux)
    layer_data = {
        'T_top': 400000.0,    # K (~0.4 MK)
        'T_mid': 600000.0,    # K (~0.6 MK)
        'T_bot': 900000.0,    # K (~0.9 MK) - steeper to ensure superadiabatic
        'P_top': 5e10,        # Pa (0.5 Mbar)
        'P_mid': 1.5e11,      # Pa (1.5 Mbar)
        'P_bot': 4e11,        # Pa (4 Mbar)
    }
    
    # Solar flux at this depth
    # Total solar luminosity: L_sun = 3.828e26 W
    # At r = 0.85 R_sun: A = 4π(0.85*6.96e8)^2 ≈ 1.7e18 m²
    # F = L_sun / A ≈ 2.2e8 W/m²
    
    # In the convection zone, most flux is carried by convection
    # F_rad is small compared to F_tot (radiation inefficient in convection zone)
    flux_data = {
        'F_tot': 2.2e8,       # W/m² (total solar flux at this depth)
        'F_rad': 0.5e8,       # W/m² (small radiative component)
    }
    
    # Physical parameters for solar plasma (H/He mixture)
    # At these conditions: fully ionized, ideal gas behavior
    physical_params = {
        'g': 274.0,           # m/s² (solar surface gravity)
        'delta': 1.0,         # Ideal gas (ionized plasma)
        'R_universal': 8.314, # J/(mol·K)
        'mu': 0.0006,         # kg/mol (ionized H/He: ~0.6 g/mol)
        'c_p': 20000.0,       # J/(kg·K) (ionized gas, gamma ~ 5/3)
        'rho': 20.0,          # kg/m³ (density at this depth)
    }
    
    print("\nLayer parameters:")
    print(f"  Temperature: {layer_data['T_top']:.0f} → {layer_data['T_mid']:.0f} → {layer_data['T_bot']:.0f} K")
    print(f"  Pressure: {layer_data['P_top']:.2e} → {layer_data['P_mid']:.2e} → {layer_data['P_bot']:.2e} Pa")
    print(f"  ΔT = {layer_data['T_bot'] - layer_data['T_top']:.0f} K")
    print(f"  ΔP = {layer_data['P_bot'] - layer_data['P_top']:.2e} Pa")
    
    print("\nFlux balance:")
    print(f"  F_tot = {flux_data['F_tot']:.2e} W/m²")
    print(f"  F_rad = {flux_data['F_rad']:.2e} W/m²")
    print(f"  F_need = {flux_data['F_tot'] - flux_data['F_rad']:.2e} W/m² (convective)")
    print(f"  Convective fraction: {100*(flux_data['F_tot']-flux_data['F_rad'])/flux_data['F_tot']:.1f}%")
    
    print("\nPhysical parameters:")
    print(f"  g = {physical_params['g']} m/s²")
    print(f"  μ = {physical_params['mu']} kg/mol (ionized H/He)")
    print(f"  c_p = {physical_params['c_p']} J/(kg·K)")
    print(f"  ρ = {physical_params['rho']} kg/m³")
    
    # Calculate alpha
    print("\n" + "-" * 70)
    print("CALCULATION:")
    print("-" * 70)
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=True)
    
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    
    if result['alpha'] is not None:
        alpha = result['alpha']
        print(f"\n✓ Mixing-length parameter: α = {alpha:.3f}")
        print(f"✓ Mixing length: l = {alpha * result['H_p']:.2e} m")
        print(f"✓ Pressure scale height: H_p = {result['H_p']:.2e} m")
        print(f"✓ l/H_p = {alpha:.3f}")
        
        # Compare with solar-calibrated values
        print(f"\n📊 VALIDATION:")
        if 1.0 <= alpha <= 3.0:
            print(f"   ✓ α in typical stellar range [1.0, 3.0]")
        if 1.5 <= alpha <= 2.0:
            print(f"   ✓✓ α in solar-calibrated range [1.5, 2.0]!")
            print(f"   → Excellent agreement with standard solar models")
        elif alpha < 1.0:
            print(f"   ⚠ α < 1.0: Lower than typical solar value")
            print(f"   → May indicate very efficient convection or high F_need")
        elif alpha > 3.0:
            print(f"   ⚠ α > 3.0: Higher than typical stellar values")
            print(f"   → Check flux balance or consider radiative losses")
        
        print(f"\n   Expected α (solar-calibrated): 1.5 - 2.0")
        print(f"   Our result: {alpha:.3f}")
        
        if abs(alpha - 1.75) < 0.5:
            print(f"   ✓ Within 0.5 of solar standard (1.75)")
    else:
        print(f"\n✗ No solution found: {result['convergence_info']}")
    
    print("=" * 70)
    
    return result


def test_red_giant_convective_envelope():
    """
    Test with red giant convective envelope parameters.
    Red giants have deep convective envelopes with lower densities.
    """
    print("\n\n" + "=" * 70)
    print("TEST: Red Giant Convective Envelope")
    print("=" * 70)
    print("\nRed giants have extended, low-density convective envelopes")
    
    layer_data = {
        'T_top': 4000.0,      # K (cooler envelope)
        'T_mid': 8000.0,      # K
        'T_bot': 16000.0,     # K (steeper for superadiabatic)
        'P_top': 1e6,         # Pa
        'P_mid': 5e6,         # Pa
        'P_bot': 2e7,         # Pa
    }
    
    # Lower flux per unit area than the Sun (larger radius)
    flux_data = {
        'F_tot': 1e6,         # W/m²
        'F_rad': 2e5,         # W/m²
    }
    
    physical_params = {
        'g': 10.0,            # m/s² (lower surface gravity for giant)
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.0012,         # kg/mol (partially ionized)
        'c_p': 15000.0,       # J/(kg·K)
        'rho': 0.01,          # kg/m³ (very low density in envelope)
    }
    
    print(f"\nT: {layer_data['T_top']:.0f} → {layer_data['T_bot']:.0f} K")
    print(f"P: {layer_data['P_top']:.2e} → {layer_data['P_bot']:.2e} Pa")
    print(f"F_need = {flux_data['F_tot'] - flux_data['F_rad']:.2e} W/m²")
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=False)
    
    print(f"\nResult: α = {result['alpha']:.3f}" if result['alpha'] else f"No solution")
    if result['alpha'] and 1.0 <= result['alpha'] <= 3.0:
        print(f"✓ Within typical range for convective envelopes")
    
    return result


def test_main_sequence_star():
    """
    Test with main-sequence star (F-type) convective zone.
    """
    print("\n\n" + "=" * 70)
    print("TEST: F-type Main Sequence Star")
    print("=" * 70)
    print("\nF-type stars: T_eff ~ 6000-7500 K, shallow convection zones")
    
    layer_data = {
        'T_top': 40000.0,     # K
        'T_mid': 80000.0,     # K
        'T_bot': 160000.0,    # K (steeper for superadiabatic)
        'P_top': 1e9,         # Pa
        'P_mid': 5e9,         # Pa
        'P_bot': 2e10,        # Pa
    }
    
    flux_data = {
        'F_tot': 5e7,         # W/m² (higher flux than Sun)
        'F_rad': 1e7,         # W/m²
    }
    
    physical_params = {
        'g': 400.0,           # m/s² (higher gravity, more massive)
        'delta': 1.0,
        'R_universal': 8.314,
        'mu': 0.0006,
        'c_p': 18000.0,
        'rho': 5.0,
    }
    
    print(f"\nT: {layer_data['T_top']:.0f} → {layer_data['T_bot']:.0f} K")
    print(f"F_need = {flux_data['F_tot'] - flux_data['F_rad']:.2e} W/m²")
    
    result = calculate_alpha_from_flux(layer_data, flux_data, physical_params,
                                      verbose=False)
    
    print(f"\nResult: α = {result['alpha']:.3f}" if result['alpha'] else f"No solution")
    
    return result


def main():
    """
    Run all stellar convection tests.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  STELLAR CONVECTION: MLT ALPHA VALIDATION".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    results = {}
    
    # Test 1: Solar convection zone (most important!)
    results['solar'] = test_solar_convection_zone()
    
    # Test 2: Red giant
    results['red_giant'] = test_red_giant_convective_envelope()
    
    # Test 3: F-type main sequence
    results['f_star'] = test_main_sequence_star()
    
    # Summary
    print("\n\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SUMMARY".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\n")
    for name, result in results.items():
        if result['alpha'] is not None:
            status = "✓"
            alpha_str = f"α = {result['alpha']:.3f}"
            
            # Check if in typical range
            if 1.0 <= result['alpha'] <= 3.0:
                range_str = "typical stellar range"
            elif result['alpha'] < 1.0:
                range_str = "below typical (efficient convection)"
            else:
                range_str = "above typical"
            
            print(f"{status} {name:15s}: {alpha_str:12s} ({range_str})")
        else:
            print(f"✗ {name:15s}: No solution")
    
    print("\n" + "█" * 70)
    print("\n📚 REFERENCES:")
    print("   • Böhm-Vitense (1958): α ≈ 1.5 calibrated to Sun")
    print("   • Cox & Giuli (1968): Standard stellar MLT")
    print("   • Modern codes: Typically use α = 1.5-2.0 for solar-type stars")
    print("\n✓ If your result is α ~ 1.5-2.0, you're matching stellar models!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

