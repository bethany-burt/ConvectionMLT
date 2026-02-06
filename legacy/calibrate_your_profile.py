"""
Calibrate Alpha for YOUR Adiabatic Profile

Simple script where you input YOUR layer parameters and get α.

Usage:
1. Edit the parameters in the USER INPUT section below
2. Run: python calibrate_your_profile.py
3. Get your α value!

Author: Generated for user-specific calibration
Date: November 2025
"""

from calibrate_alpha_adiabatic import calibrate_alpha_adiabatic


def main():
    print("=" * 70)
    print("CALIBRATE α FOR YOUR ADIABATIC PROFILE")
    print("=" * 70)
    print()
    
    # ========================================================================
    # USER INPUT: EDIT THESE VALUES
    # ========================================================================
    
    # Layer properties (at the location you're interested in)
    T = 1500.0           # K - Temperature
    P = 5e4              # Pa - Pressure  
    rho = 0.005          # kg/m³ - Density
    
    # Convective flux requirement
    F_conv = 5e6         # W/m² - Flux to be carried by convection
                         # = F_tot - F_rad (from your RT code)
    
    # Physical parameters for your atmosphere
    g = 10.0             # m/s² - Gravitational acceleration
    mu = 0.0022          # kg/mol - Mean molecular weight
                         #   H2: 0.002, He: 0.004, air: 0.029
    c_p = 14000.0        # J/(kg·K) - Specific heat capacity
                         #   H2: ~14000, He: ~5000, air: ~1000
    
    # How superadiabatic? (0.001 = 0.1% above adiabat, typical for convection)
    epsilon = 0.001
    
    # ========================================================================
    # END USER INPUT
    # ========================================================================
    
    physical_params = {
        'g': g,
        'delta': 1.0,              # Ideal gas (usually fine)
        'R_universal': 8.314,       # J/(mol·K) - universal constant
        'mu': mu,
        'c_p': c_p,
    }
    
    # Display inputs
    print("Your Input Parameters:")
    print("-" * 70)
    print(f"  Temperature: T = {T} K")
    print(f"  Pressure: P = {P:.2e} Pa = {P/1e5:.4f} bar")
    print(f"  Density: ρ = {rho} kg/m³")
    print(f"  Convective flux: F_conv = {F_conv:.2e} W/m²")
    print()
    print(f"Physical Parameters:")
    print(f"  Gravity: g = {g} m/s²")
    print(f"  Mean molecular weight: μ = {mu} kg/mol")
    print(f"  Specific heat: c_p = {c_p} J/(kg·K)")
    print(f"  Superadiabaticity: ε = {epsilon} ({epsilon*100}%)")
    print()
    
    # Calculate
    print("=" * 70)
    print("CALCULATING...")
    print("=" * 70)
    
    result = calibrate_alpha_adiabatic(T, P, rho, F_conv, physical_params, 
                                      epsilon=epsilon)
    
    # Display result
    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print()
    
    if result['alpha'] is not None:
        alpha = result['alpha']
        l = result['l']
        H_p = result['H_p']
        
        print(f"✓ Mixing-length parameter: α = {alpha:.4f}")
        print(f"✓ Mixing length: l = {l:.2f} m = {l/1000:.2f} km")
        print(f"✓ Pressure scale height: H_p = {H_p:.2f} m = {H_p/1000:.2f} km")
        print(f"✓ Ratio: l/H_p = {alpha:.4f}")
        print()
        print(f"Adiabatic gradient: ∇_ad = {result['nabla_ad']:.6f}")
        print(f"Used gradient: ∇ = {result['nabla']:.6f}")
        print(f"Flux verification: {result['F_c']:.2e} W/m² (target: {F_conv:.2e})")
        print(f"Error: {result['error']*100:.3f}%")
        print()
        
        # Interpretation
        print("-" * 70)
        print("INTERPRETATION:")
        print("-" * 70)
        
        if alpha < 0.01:
            print(f"• α = {alpha:.4f} is VERY SMALL")
            print(f"  → Very efficient small-scale mixing")
            print(f"  → Typical for high flux / strongly convective regions")
        elif 0.01 <= alpha < 0.5:
            print(f"• α = {alpha:.4f} is SMALL")
            print(f"  → Efficient convection with small mixing scales")
        elif 0.5 <= alpha < 1.0:
            print(f"• α = {alpha:.4f} is MODERATE")
            print(f"  → Transitional regime")
        elif 1.0 <= alpha < 3.0:
            print(f"• α = {alpha:.4f} is in STELLAR RANGE")
            print(f"  → Typical for stellar convection zones")
            if 1.5 <= alpha <= 2.0:
                print(f"  → SOLAR-CALIBRATED RANGE! Matches Sun!")
        else:
            print(f"• α = {alpha:.4f} is LARGE")
            print(f"  → Large-scale mixing, check if physical")
        
        print()
        print("📝 To use in your RT code:")
        print(f"   alpha = {alpha:.4f}")
        print(f"   l = alpha * H_p = {alpha:.4f} * H_p")
        
    else:
        print(f"✗ Could not calibrate: {result['message']}")
        print()
        print("Possible issues:")
        print("  • F_conv might be too small or negative")
        print("  • Check that layer parameters are reasonable")
    
    print()
    print("=" * 70)
    
    # Save result
    print()
    print("To run with your own values:")
    print("  1. Edit the USER INPUT section in this file")
    print("  2. Run: python calibrate_your_profile.py")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()




