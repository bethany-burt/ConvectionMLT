"""
Example: Integrating Mixing-Length Theory into a Radiative Transfer Code

This script demonstrates how to use the mixing-length theory module
to replace simple adiabatic corrections in a radiative transfer code.

Author: Generated for radiative transfer convection modeling
Date: November 2025
"""

import numpy as np
from mixing_length_theory import MixingLengthConvection


class AtmosphericLayer:
    """Simple representation of an atmospheric layer."""
    
    def __init__(self, T_top, T_mid, T_bot, P_top, P_mid, P_bot, 
                 z_top, z_mid, z_bot, cp):
        self.T_top = T_top
        self.T_mid = T_mid
        self.T_bot = T_bot
        self.P_top = P_top
        self.P_mid = P_mid
        self.P_bot = P_bot
        self.z_top = z_top
        self.z_mid = z_mid
        self.z_bot = z_bot
        self.cp = cp
        
        # Calculate layer thickness
        self.dz = abs(z_top - z_bot)
        
        # Calculate actual temperature gradient
        self.dT_dz_actual = (T_top - T_bot) / (z_top - z_bot) if z_top != z_bot else 0.0


def old_convection_treatment(layer, mlt):
    """
    OLD METHOD: Simple adiabatic correction.
    
    If layer is superadiabatic, just force it to follow an adiabat.
    """
    # Calculate adiabatic gradient
    grad_ad = mlt.adiabatic_gradient(layer.T_mid, layer.P_mid, layer.cp)
    
    # Check if superadiabatic (Schwarzschild criterion)
    is_unstable = mlt.schwarzschild_criterion(layer.dT_dz_actual, grad_ad)
    
    if is_unstable:
        print(f"  Layer is superadiabatic!")
        print(f"    Actual dT/dz: {layer.dT_dz_actual:.6f} K/m")
        print(f"    Adiabatic dT/dz: {grad_ad:.6f} K/m")
        
        # OLD: Simply force to adiabat
        T_bot_new = layer.T_top + grad_ad * (layer.z_bot - layer.z_top)
        T_mid_new = layer.T_top + grad_ad * (layer.z_mid - layer.z_top)
        
        print(f"  Forcing to adiabat:")
        print(f"    Old T_bot: {layer.T_bot:.2f} K -> New T_bot: {T_bot_new:.2f} K")
        print(f"    Correction: {abs(T_bot_new - layer.T_bot):.2f} K")
        
        return T_mid_new, T_bot_new, True
    
    return layer.T_mid, layer.T_bot, False


def new_mlt_treatment(layer, mlt, alpha=None):
    """
    NEW METHOD: Mixing-length theory.
    
    Use MLT to calculate convective flux and adjust temperature profile
    based on convective energy transport.
    """
    # Calculate adiabatic gradient
    grad_ad = mlt.adiabatic_gradient(layer.T_mid, layer.P_mid, layer.cp)
    
    # Check if superadiabatic
    is_unstable = mlt.schwarzschild_criterion(layer.dT_dz_actual, grad_ad)
    
    if is_unstable:
        print(f"  Layer is superadiabatic!")
        print(f"    Actual dT/dz: {layer.dT_dz_actual:.6f} K/m")
        print(f"    Adiabatic dT/dz: {grad_ad:.6f} K/m")
        
        # Calculate or use provided alpha
        if alpha is None:
            alpha = mlt.find_alpha_for_adiabat(layer.T_mid, layer.P_mid, layer.cp,
                                              actual_dT_dz=layer.dT_dz_actual)
        
        # Calculate mixing length
        H = mlt.pressure_scale_height(layer.T_mid, layer.P_mid)
        l = mlt.mixing_length(alpha, H)
        
        print(f"  MLT parameters:")
        print(f"    α = {alpha:.3f}")
        print(f"    Scale height H = {H:.2f} m")
        print(f"    Mixing length l = {l:.2f} m = {l/H:.3f} H")
        
        # Calculate convective velocity
        delta_grad = layer.dT_dz_actual - grad_ad
        v_conv = mlt.convective_velocity(l, delta_grad, layer.T_mid, H)
        
        print(f"    Convective velocity: {v_conv:.2f} m/s")
        
        # Calculate convective flux
        rho = (layer.P_mid * mlt.mu) / (mlt.R * layer.T_mid)
        F_conv = mlt.convective_flux(l, delta_grad, layer.T_mid, rho, layer.cp)
        
        print(f"    Density: {rho:.4f} kg/m³")
        print(f"    Convective flux: {F_conv:.2e} W/m²")
        
        # Adjust temperature based on convective transport
        # The convective flux reduces the superadiabatic gradient
        # Simplified approach: adjust gradient toward adiabatic
        
        # Mixing timescale
        tau_mix = l / v_conv if v_conv > 0 else np.inf
        
        # Adjustment factor (how much to move toward adiabatic)
        # In a full RT code, this would be integrated with time-stepping
        adjustment_factor = min(1.0, layer.dz / l)  # Simplified
        
        # New gradient is between actual and adiabatic
        grad_new = layer.dT_dz_actual + adjustment_factor * (grad_ad - layer.dT_dz_actual)
        
        T_bot_new = layer.T_top + grad_new * (layer.z_bot - layer.z_top)
        T_mid_new = layer.T_top + grad_new * (layer.z_mid - layer.z_top)
        
        print(f"  MLT adjustment:")
        print(f"    Adjustment factor: {adjustment_factor:.3f}")
        print(f"    New dT/dz: {grad_new:.6f} K/m")
        print(f"    Old T_bot: {layer.T_bot:.2f} K -> New T_bot: {T_bot_new:.2f} K")
        print(f"    Correction: {abs(T_bot_new - layer.T_bot):.2f} K")
        print(f"    Mixing timescale: {tau_mix:.2f} s")
        
        return T_mid_new, T_bot_new, True
    
    return layer.T_mid, layer.T_bot, False


def main():
    """
    Demonstration comparing old and new convection treatments.
    """
    print("=" * 80)
    print("COMPARISON: Old Adiabatic Correction vs. Mixing-Length Theory")
    print("=" * 80)
    
    # Initialize MLT calculator
    g = 10.0        # m/s²
    mu = 0.0029     # kg/mol
    R = 8.314       # J/(mol·K)
    mlt = MixingLengthConvection(g=g, mu=mu, R=R)
    
    # Create an example superadiabatic layer
    # This might occur in a hot Jupiter atmosphere
    layer = AtmosphericLayer(
        T_top=1400.0,      # K
        T_mid=1450.0,      # K
        T_bot=1520.0,      # K (steeper than adiabatic)
        P_top=1e4,         # Pa
        P_mid=5e4,         # Pa
        P_bot=1e5,         # Pa
        z_top=50000.0,     # m
        z_mid=45000.0,     # m
        z_bot=40000.0,     # m
        cp=1000.0          # J/(kg·K)
    )
    
    print(f"\nInput Layer Properties:")
    print(f"  Top:    T = {layer.T_top:.2f} K, P = {layer.P_top/1e3:.2f} kPa, z = {layer.z_top/1e3:.2f} km")
    print(f"  Middle: T = {layer.T_mid:.2f} K, P = {layer.P_mid/1e3:.2f} kPa, z = {layer.z_mid/1e3:.2f} km")
    print(f"  Bottom: T = {layer.T_bot:.2f} K, P = {layer.P_bot/1e3:.2f} kPa, z = {layer.z_bot/1e3:.2f} km")
    print(f"  Actual temperature gradient: {layer.dT_dz_actual:.6f} K/m")
    print(f"  Layer thickness: {layer.dz/1e3:.2f} km")
    
    # Test OLD method
    print("\n" + "-" * 80)
    print("OLD METHOD: Simple Adiabatic Correction")
    print("-" * 80)
    T_mid_old, T_bot_old, corrected_old = old_convection_treatment(layer, mlt)
    
    # Test NEW method
    print("\n" + "-" * 80)
    print("NEW METHOD: Mixing-Length Theory")
    print("-" * 80)
    T_mid_new, T_bot_new, corrected_new = new_mlt_treatment(layer, mlt, alpha=1.5)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON OF RESULTS")
    print("=" * 80)
    print(f"\nBottom Temperature:")
    print(f"  Original:      {layer.T_bot:.2f} K")
    print(f"  Old method:    {T_bot_old:.2f} K (change: {T_bot_old - layer.T_bot:+.2f} K)")
    print(f"  New MLT:       {T_bot_new:.2f} K (change: {T_bot_new - layer.T_bot:+.2f} K)")
    print(f"  Difference (Old - MLT): {T_bot_old - T_bot_new:+.2f} K")
    
    print(f"\nKey Insight:")
    print(f"  The old method forces an exact adiabat immediately.")
    print(f"  MLT provides a more physical treatment based on:")
    print(f"    - Mixing length scale")
    print(f"    - Convective velocity")
    print(f"    - Energy flux")
    print(f"    - Adjustment timescale")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR YOUR RT CODE")
    print("=" * 80)
    print("""
1. Use pre-calculated α: Run parameter space exploration once, use mean α ≈ 1.5
   
2. Dynamic α: Calculate α for each layer (more accurate but slower)
   
3. Hybrid approach: Use lookup table from parameter space exploration
   
4. Full integration: Include MLT in time-dependent calculations with proper
   coupling between convective flux and radiative transfer
   
5. Validation: Compare with detailed models or observations to tune α
""")


if __name__ == "__main__":
    main()


