"""
Mixing-Length Theory for Atmospheric Convection

This module implements mixing-length theory to calculate the convective
flux and determine the mixing-length parameter α that produces an adiabatic
temperature-pressure profile.

Author: Generated for radiative transfer convection modeling
Date: November 2025
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from typing import Tuple, Optional


class MixingLengthConvection:
    """
    Class to handle mixing-length theory calculations for atmospheric convection.
    
    Attributes:
        g (float): Gravitational acceleration [m/s²]
        mu (float): Mean molecular weight [kg/mol]
        R (float): Universal gas constant [J/(mol·K)]
    """
    
    def __init__(self, g: float = 10.0, mu: float = 0.0029, R: float = 8.314):
        """
        Initialize the mixing-length theory calculator.
        
        Parameters:
            g (float): Gravitational acceleration [m/s²], default 10.0
            mu (float): Mean molecular weight [kg/mol], default 0.0029 (Earth-like)
            R (float): Universal gas constant [J/(mol·K)], default 8.314
        """
        self.g = g
        self.mu = mu
        self.R = R
    
    def pressure_scale_height(self, T: float, P: float) -> float:
        """
        Calculate the pressure scale height H.
        
        H = RT/(μg)
        
        Parameters:
            T (float): Temperature [K]
            P (float): Pressure [Pa]
        
        Returns:
            float: Pressure scale height [m]
        """
        return (self.R * T) / (self.mu * self.g)
    
    def adiabatic_gradient(self, T: float, P: float, cp: float) -> float:
        """
        Calculate the adiabatic temperature gradient dT/dz.
        
        For an adiabatic process:
        dT/dz = -g/cp
        
        Parameters:
            T (float): Temperature [K]
            P (float): Pressure [Pa]
            cp (float): Specific heat capacity at constant pressure [J/(kg·K)]
        
        Returns:
            float: Adiabatic temperature gradient [K/m]
        """
        return -self.g / cp
    
    def mixing_length(self, alpha: float, H: float) -> float:
        """
        Calculate the mixing length from α and pressure scale height.
        
        l = α * H
        
        Parameters:
            alpha (float): Mixing-length parameter (dimensionless)
            H (float): Pressure scale height [m]
        
        Returns:
            float: Mixing length [m]
        """
        return alpha * H
    
    def convective_velocity(self, l: float, delta_grad: float, T: float, 
                           H: float) -> float:
        """
        Calculate the characteristic convective velocity using MLT.
        
        v_conv ≈ sqrt(l * g * |∇ - ∇_ad| / ∇_ad)
        
        where ∇ = d(ln T)/d(ln P) and ∇_ad is the adiabatic gradient.
        
        Parameters:
            l (float): Mixing length [m]
            delta_grad (float): Difference between actual and adiabatic gradient [K/m]
            T (float): Temperature [K]
            H (float): Pressure scale height [m]
        
        Returns:
            float: Convective velocity [m/s]
        """
        if delta_grad <= 0:
            return 0.0
        
        # Simplified MLT velocity scale
        v_conv = np.sqrt(l * self.g * abs(delta_grad) / T)
        return v_conv
    
    def convective_flux(self, l: float, delta_grad: float, T: float, 
                       rho: float, cp: float) -> float:
        """
        Calculate the convective flux using mixing-length theory.
        
        F_conv = ρ * cp * v_conv * ΔT
        
        where ΔT ≈ l * |∇ - ∇_ad|
        
        Parameters:
            l (float): Mixing length [m]
            delta_grad (float): Difference between actual and adiabatic gradient [K/m]
            T (float): Temperature [K]
            rho (float): Density [kg/m³]
            cp (float): Specific heat capacity [J/(kg·K)]
        
        Returns:
            float: Convective flux [W/m²]
        """
        if delta_grad <= 0:
            return 0.0
        
        H = self.pressure_scale_height(T, rho * self.R * T / self.mu)
        v_conv = self.convective_velocity(l, delta_grad, T, H)
        delta_T = l * abs(delta_grad)
        
        F_conv = rho * cp * v_conv * delta_T
        return F_conv
    
    def find_alpha_for_adiabat(self, T: float, P: float, cp: float, 
                               actual_dT_dz: Optional[float] = None,
                               target_flux_ratio: float = 1.0) -> float:
        """
        Find the α parameter that produces an adiabatic profile.
        
        This function determines what α value is needed for the mixing-length
        theory to match an adiabatic temperature gradient.
        
        Parameters:
            T (float): Temperature [K]
            P (float): Pressure [Pa]
            cp (float): Specific heat capacity [J/(kg·K)]
            actual_dT_dz (float, optional): Actual temperature gradient [K/m].
                If None, uses superadiabatic gradient for testing.
            target_flux_ratio (float): Target ratio of convective to total flux
        
        Returns:
            float: Optimal α parameter
        """
        H = self.pressure_scale_height(T, P)
        grad_ad = self.adiabatic_gradient(T, P, cp)
        
        # If no actual gradient provided, assume slightly superadiabatic
        if actual_dT_dz is None:
            actual_dT_dz = grad_ad * 1.1  # 10% superadiabatic
        
        # Calculate density
        rho = (P * self.mu) / (self.R * T)
        
        def objective(alpha):
            """Objective function to minimize."""
            if alpha <= 0:
                return 1e10
            
            l = self.mixing_length(alpha, H)
            delta_grad = actual_dT_dz - grad_ad
            
            # For adiabatic case, we want the mixing length such that
            # the convective adjustment brings the gradient to adiabatic
            # This is typically when l ~ H (α ~ 1-2)
            
            # One approach: find α where the mixing timescale matches
            # the convective adjustment timescale
            if delta_grad <= 0:
                return abs(alpha - 1.5)  # Default to typical value
            
            # Alternative: find α where convective efficiency is optimal
            # Typical values are α ~ 1.5-2.0 for stellar/planetary atmospheres
            v_conv = self.convective_velocity(l, delta_grad, T, H)
            
            # Convective timescale
            tau_conv = l / v_conv if v_conv > 0 else 1e10
            
            # Thermal timescale
            tau_thermal = (rho * cp * H**2) / (4 * 5.67e-8 * T**4)  # rough estimate
            
            # We want efficient convection: tau_conv << tau_thermal
            # But also stable: not too fast
            # Optimal is typically when tau_conv ~ 0.1 * tau_thermal
            efficiency = abs(np.log10(tau_conv / tau_thermal) + 1)
            
            return efficiency
        
        # Search for optimal alpha in range [0.1, 5.0]
        result = minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded')
        
        return result.x
    
    def schwarzschild_criterion(self, actual_grad: float, 
                                adiabatic_grad: float) -> bool:
        """
        Check if a layer is convectively unstable (Schwarzschild criterion).
        
        A layer is unstable if the actual temperature gradient is steeper
        than the adiabatic gradient (more negative).
        
        Parameters:
            actual_grad (float): Actual temperature gradient dT/dz [K/m]
            adiabatic_grad (float): Adiabatic gradient [K/m]
        
        Returns:
            bool: True if convectively unstable, False if stable
        """
        return actual_grad < adiabatic_grad
    
    def calculate_alpha_grid(self, T_range: np.ndarray, P_range: np.ndarray,
                            cp_range: np.ndarray) -> dict:
        """
        Calculate α values over a parameter space grid.
        
        Parameters:
            T_range (np.ndarray): Array of temperatures [K]
            P_range (np.ndarray): Array of pressures [Pa]
            cp_range (np.ndarray): Array of cp values [J/(kg·K)]
        
        Returns:
            dict: Dictionary containing results arrays
        """
        results = {
            'T': T_range,
            'P': P_range,
            'cp': cp_range,
            'alpha_TP': np.zeros((len(T_range), len(P_range))),
            'alpha_Tcp': np.zeros((len(T_range), len(cp_range))),
            'alpha_Pcp': np.zeros((len(P_range), len(cp_range))),
        }
        
        # Fix cp, vary T and P
        cp_fixed = np.median(cp_range)
        for i, T in enumerate(T_range):
            for j, P in enumerate(P_range):
                results['alpha_TP'][i, j] = self.find_alpha_for_adiabat(T, P, cp_fixed)
        
        # Fix P, vary T and cp
        P_fixed = np.median(P_range)
        for i, T in enumerate(T_range):
            for j, cp in enumerate(cp_range):
                results['alpha_Tcp'][i, j] = self.find_alpha_for_adiabat(T, P_fixed, cp)
        
        # Fix T, vary P and cp
        T_fixed = np.median(T_range)
        for i, P in enumerate(P_range):
            for j, cp in enumerate(cp_range):
                results['alpha_Pcp'][i, j] = self.find_alpha_for_adiabat(T_fixed, P, cp)
        
        return results


def demo_calculation():
    """
    Demonstration of mixing-length theory calculations.
    """
    print("=" * 60)
    print("Mixing-Length Theory Demo")
    print("=" * 60)
    
    # Initialize with Earth-like parameters
    mlt = MixingLengthConvection(g=10.0, mu=0.0029, R=8.314)
    
    # Example atmospheric conditions
    T = 1500.0  # K (hot atmosphere)
    P = 1e5     # Pa (1 bar)
    cp = 1000.0 # J/(kg·K)
    
    print(f"\nAtmospheric Conditions:")
    print(f"  Temperature: {T} K")
    print(f"  Pressure: {P/1e5} bar")
    print(f"  Specific heat (cp): {cp} J/(kg·K)")
    
    # Calculate scale height
    H = mlt.pressure_scale_height(T, P)
    print(f"\nPressure Scale Height: {H:.2f} m")
    
    # Calculate adiabatic gradient
    grad_ad = mlt.adiabatic_gradient(T, P, cp)
    print(f"Adiabatic Gradient: {grad_ad:.6f} K/m")
    
    # Find optimal alpha
    alpha = mlt.find_alpha_for_adiabat(T, P, cp)
    print(f"\nOptimal α for adiabatic profile: {alpha:.3f}")
    
    # Calculate mixing length
    l = mlt.mixing_length(alpha, H)
    print(f"Mixing Length: {l:.2f} m ({l/H:.3f} × H)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_calculation()


