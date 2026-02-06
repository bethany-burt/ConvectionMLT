"""
Convective Flux Solver v2 - Iterative 1D Atmospheric Column Model

This script iteratively calculates convective flux for a model grid of atmospheric
layers, accounting for the effect of changing temperature on convective flux.

Based on mixing length theory:
    F_conv = ρ * c_p * l^2 * sqrt(g/T) * (N - N_ad)^(3/2)

Where:
    l = α × H_p (mixing length)
    H_p = RT/(μ·g) (pressure scale height)
    N = -dT/dz (temperature gradient)
    N_ad = g/c_p (adiabatic temperature gradient)

All calculations use SI units internally with explicit unit conversions.
See derivation_SI_units.txt for complete unit derivations.
"""

import numpy as np
import sys
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from scipy import special


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Boltzmann constant in erg K^-1 (for backward compatibility)
K_B = 1.380649e-16  # erg K^-1

# Ideal gas constant in erg mol^-1 K^-1 (for backward compatibility)
R = 8.314e7  # erg mol^-1 K^-1

# SI constants for internal calculations
R_SI = 8.314  # J/(mol·K) = kg·m²/(s²·mol·K)
K_B_SI = 1.380649e-23  # J/K = kg·m²/(s²·K)

# Stefan-Boltzmann constant
SIGMA_SB = 5.670374419e-8  # W/(m²·K⁴) = kg/(s³·K⁴)

# ============================================================================
# INPUT PARAMETERS
# ============================================================================

# Temperature boundaries (K)
T_TOA = 800.0   # Top of atmosphere
T_BOA = 2000.0  # Bottom of atmosphere

# Density boundaries (g/cm^3) - will convert to g/m^3 for consistency
RHO_TOA = 0.1    # g/cm^3
RHO_BOA = 1000.0 # g/cm^3

# Physical parameters
G = 15.0         # Gravity (m/s^2)
ALPHA = 0.1      # Mixing length parameter (dimensionless, α in l = α × H_p)
DT = 1       # Timestep (s)
MAX_Z = 500_000  # Maximum altitude (m)

# Grid parameters
N_LAYERS = 100    # Number of layers (will have n_layers+1 interfaces)

# Composition parameters (H2 dominated)
N_DOF = 5        # Degrees of freedom for H2 (3 translational + 2 rotational)
MMW = 2.016      # Mean molecular weight (g/mol) for H2

# Solver parameters
MAX_STEPS = 500000      # Maximum iteration steps
CONVERGENCE_TOL = 1e-5  # Convergence tolerance for max|dT| (K)
DEBUG_INTERVAL = 10     # Print debug info every N steps


# ============================================================================
# UNIT CONVERSION HELPER FUNCTIONS
# ============================================================================

def g_per_cm3_to_kg_per_m3(rho_g_cm3: float) -> float:
    """Convert density from g/cm³ to kg/m³."""
    return rho_g_cm3 * 1000.0

def erg_per_gK_to_J_per_kgK(c_p_erg_gK: float) -> float:
    """Convert specific heat capacity from erg/(g·K) to J/(kg·K)."""
    return c_p_erg_gK * 1e-4

def g_per_mol_to_kg_per_mol(mmw_g_mol: float) -> float:
    """Convert mean molecular weight from g/mol to kg/mol."""
    return mmw_g_mol * 0.001

def W_per_m2_to_erg_per_cm2_s(F_W_m2: float) -> float:
    """Convert energy flux from W/m² to erg cm⁻² s⁻¹."""
    return F_W_m2 * 1e3

def erg_per_cm2_s_to_W_per_m2(F_erg_cm2_s: float) -> float:
    """Convert energy flux from erg cm⁻² s⁻¹ to W/m²."""
    return F_erg_cm2_s * 1e-3

def dyne_per_cm2_to_Pa(P_dyne_cm2: float) -> float:
    """Convert pressure from dyne/cm² to Pa (kg/(m·s²)).
    
    1 dyne = 1 g·cm/s² = 10⁻⁵ N = 10⁻⁵ kg·m/s²
    1 cm² = 10⁻⁴ m²
    1 dyne/cm² = 10⁻¹ Pa = 0.1 Pa
    """
    return P_dyne_cm2 * 0.1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_c_p(n_dof: int, mmw: float) -> float:
    """
    Calculate specific heat capacity at constant pressure.
    
    c_p = (k_B * (2 + n_dof)) / m
    
    Where m is mass per molecule. Since R = N_A * k_B and m = mmw / N_A,
    we can rewrite as: c_p = (2 + n_dof) * R / mmw
    
    Args:
        n_dof: Number of degrees of freedom
        mmw: Mean molecular weight (g/mol)
    
    Returns:
        c_p in erg g^-1 K^-1 (for compatibility, but calculated in SI internally)
    """
    # Convert to SI, calculate in SI, then convert back for compatibility
    mmw_kg = g_per_mol_to_kg_per_mol(mmw)
    c_p_SI = (2 + n_dof) * R_SI / mmw_kg  # J/(kg·K)
    # Convert to erg/(g·K): 1 J/(kg·K) = 10^4 erg/(g·K)
    c_p = c_p_SI * 1e4
    return c_p


def setup_grid(n_layers: int, max_z: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Set up the vertical grid.
    
    Args:
        n_layers: Number of layers
        max_z: Maximum altitude (m)
    
    Returns:
        z: Altitude at interfaces (n_layers+1 points) in meters
        z_mid: Altitude at layer centers (n_layers points) in meters
        dz: Layer thickness (m)
    """
    # Interfaces from 0 to max_z
    z = np.linspace(0, max_z, n_layers + 1)
    
    # Layer centers (midpoints between interfaces)
    z_mid = (z[:-1] + z[1:]) / 2.0
    
    # Layer thickness (uniform for now)
    dz = max_z / n_layers
    
    return z, z_mid, dz


def guillot_tp_profile(m: float, m0: float, tint: float, tirr: float,
                       kappa_S: float, kappa0: float, kappa_cia: float,
                       beta_S0: float, beta_L0: float, el1: float, el3: float) -> float:
    """
    Compute Guillot temperature-pressure profile.
    
    Based on the Guillot (2010) analytical TP profile model.
    
    Args:
        m: Column mass (g/cm^2)
        m0: Bottom of atmosphere column mass (g/cm^2)
        tint: Internal temperature (K)
        tirr: Irradiation temperature (K)
        kappa_S: Shortwave opacity (cm^2/g)
        kappa0: Infrared opacity constant component (cm^2/g)
        kappa_cia: CIA opacity normalization (cm^2/g)
        beta_S0: Shortwave scattering parameter
        beta_L0: Longwave scattering parameter
        el1: First longwave Eddington coefficient
        el3: Second longwave Eddington coefficient
    
    Returns:
        Temperature (K)
    """
    albedo = (1.0 - beta_S0) / (1.0 + beta_S0)
    kappa_L = kappa0 + kappa_cia * m / m0
    beta_S = kappa_S * m / beta_S0
    coeff1 = 0.25 * (tint ** 4)
    coeff2 = 0.125 * (tirr ** 4) * (1.0 - albedo)
    term1 = 1.0 / el1 + m * (kappa0 + 0.5 * kappa_cia * m / m0) / el3 / (beta_L0 ** 2)
    term2 = 0.5 / el1 + special.expn(2, beta_S) * (kappa_S / kappa_L / beta_S0 - 
                                                    kappa_cia * m * beta_S0 / el3 / kappa_S / m0 / (beta_L0 ** 2))
    term3 = kappa0 * beta_S0 * (1.0 / 3.0 - special.expn(4, beta_S)) / el3 / kappa_S / (beta_L0 ** 2)
    term4 = kappa_cia * (beta_S0 ** 2) * (0.5 - special.expn(3, beta_S)) / el3 / m0 / (kappa_S ** 2) / (beta_L0 ** 2)
    result = (coeff1 * term1 + coeff2 * (term2 + term3 + term4)) ** 0.25
    return result


def initialize_profiles(z: np.ndarray, z_mid: np.ndarray, 
                        T_toa: float, T_boa: float,
                        rho_toa: float, rho_boa: float,
                        profile_type: str = "guillot",
                        guillot_params: Optional[dict] = None,
                        g: float = G) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize temperature and density profiles.
    
    Temperature: Either linear interpolation or Guillot TP profile
    Density: Linear interpolation from rho_boa (bottom) to rho_toa (top)
    
    Args:
        z: Altitude at interfaces (m)
        z_mid: Altitude at layer centers (m)
        T_toa: Temperature at top of atmosphere (K) - used for linear profile
        T_boa: Temperature at bottom of atmosphere (K) - used for linear profile
        rho_toa: Density at top (g/cm^3)
        rho_boa: Density at bottom (g/cm^3)
        profile_type: "linear" or "guillot"
        guillot_params: Dictionary with Guillot parameters (required if profile_type="guillot")
        g: Gravity (m/s^2) - needed for pressure calculation
    
    Returns:
        T: Temperature at interfaces (K)
        rho: Density at interfaces (g/cm^3)
    """
    # Linear density profile (interfaces)
    # Note: density decreases outward, so rho_boa at z=0, rho_toa at z=max
    rho = np.linspace(rho_boa, rho_toa, len(z))
    
    if profile_type == "linear":
        # Linear temperature profile (interfaces)
        T = np.linspace(T_boa, T_toa, len(z))
    
    elif profile_type == "guillot":
        if guillot_params is None:
            raise ValueError("guillot_params must be provided when profile_type='guillot'")
        
        # Use pressure directly (like tp.py) instead of converting from altitude
        # This ensures we get the same TP profile as tp.py
        
        # Convert gravity from m/s^2 to cm/s^2
        g_cgs = g * 100.0  # cm/s^2
        bar2cgs = 1e6  # Convert bar to dyne/cm^2
        
        # Generate pressure array (like tp.py does)
        # Use a pressure range that covers the altitude range
        # Default: log pressure from -3 to 2 (0.001 to 100 bar)
        # But we can adjust based on the number of layers
        logp_min = guillot_params.get('logp_min', -3.0)
        logp_max = guillot_params.get('logp_max', 2.0)
        
        # Create pressure array with same number of points as altitude grid
        logp = np.linspace(logp_max, logp_min, len(z))  # From high to low pressure (bottom to top)
        P_bar = 10.0 ** logp  # Pressure in bars
        
        # Calculate column mass from pressure: m = P / g (like tp.py)
        P_cgs = P_bar * bar2cgs  # Pressure in dyne/cm^2
        m = P_cgs / g_cgs  # Column mass in g/cm^2
        m0 = m[0]  # Bottom column mass (maximum pressure)
        
        # Extract Guillot parameters
        tint = guillot_params['tint']
        tirr = guillot_params['tirr']
        kappa_S = guillot_params['kappa_S']
        kappa0 = guillot_params['kappa0']
        kappa_cia = guillot_params.get('kappa_cia', 0.0)
        beta_S0 = guillot_params.get('beta_S0', 1.0)
        beta_L0 = guillot_params.get('beta_L0', 1.0)
        el1 = guillot_params.get('el1', 3.0/8.0)
        el3 = guillot_params.get('el3', 1.0/3.0)
        
        # Calculate temperature at each interface using Guillot profile
        T = np.zeros(len(z))
        for i in range(len(z)):
            T[i] = guillot_tp_profile(m[i], m0, tint, tirr, kappa_S, kappa0, kappa_cia,
                                     beta_S0, beta_L0, el1, el3)
        
        # Now convert pressure to altitude using hydrostatic equilibrium
        # This allows us to map the pressure-based TP profile to our altitude grid
        # dP/dz = -rho * g, so dz = -dP / (rho * g)
        # We'll integrate from bottom (high pressure) upward
        
        # For the conversion, we need density. We can use the ideal gas law:
        # P = rho * R_specific * T, so rho = P / (R_specific * T)
        # Or use the provided density profile and adjust
        #Use the density profile to convert P to z
        # Integrate: dz = -dP / (rho * g)
        z_from_pressure = np.zeros(len(z))
        z_from_pressure[0] = 0.0  # Start at bottom
        
        for i in range(1, len(z)):
            dP = P_cgs[i-1] - P_cgs[i]  # Pressure difference (positive going up)
            # Use average density in the layer
            rho_avg = (rho[i] + rho[i-1]) / 2.0
            # dz = dP / (rho * g) (positive because we're going up)
            dz_cm = dP / (rho_avg * g_cgs)
            z_from_pressure[i] = z_from_pressure[i-1] + dz_cm / 100.0  # Convert cm to m
        
        # Note: The altitude grid z is fixed, but we've calculated z_from_pressure
        # The temperature T is now based on pressure (like tp.py), which is correct
        # The z grid will be used for the solver, but the TP profile is pressure-based
    
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}. Must be 'linear' or 'guillot'")
    
    return T, rho


def adiabatic_gradient(g: float, c_p: float) -> float:
    """
    Calculate adiabatic temperature gradient.
    
    N_ad = g / c_p
    
    Args:
        g: Gravity (m/s^2)
        c_p: Specific heat capacity (erg g^-1 K^-1)
    
    Returns:
        N_ad: Adiabatic gradient (K/m)
    """
    # Convert c_p from erg/(g·K) to J/(kg·K) for SI calculation
    c_p_SI = erg_per_gK_to_J_per_kgK(c_p)  # J/(kg·K)
    # Calculate in SI: N_ad = g / c_p
    N_ad = g / c_p_SI  # (m/s²) / (J/(kg·K)) = (m/s²) / (m²/(s²·K)) = K/m
    return N_ad


def temperature_gradient(T: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculate temperature gradient using centered differences.
    
    N = -dT/dz
    
    Computes the gradient at layer centers from temperature values at interfaces.
    The gradient is NOT moved or shifted - it's calculated at the natural location
    (layer centers) from the interface values.
    
    Args:
        T: Temperature at interfaces (K)
        z: Altitude at interfaces (m)
    
    Returns:
        N: Temperature gradient at layer centers (K/m)
    """
    # Centered difference: dT/dz at layer centers
    # This computes the gradient between adjacent interfaces, placing it at the center
    dT = np.diff(T)
    dz = np.diff(z)
    dT_dz = dT / dz
    
    # N = -dT/dz (negative because T decreases with z)
    N = -dT_dz
    
    return N


def convective_flux(rho: np.ndarray, c_p: float, alpha: float, g: float,
                    T: np.ndarray, N: np.ndarray, N_ad: float, mmw: float) -> np.ndarray:
    """
    Calculate convective flux using mixing length theory.
    
    F_conv = ρ * c_p * l^2 * sqrt(g/T) * (N - N_ad)^(3/2)
    
    Where l = α × H_p and H_p = RT/(μ·g)
    
    Only applies when N > N_ad (convectively unstable).
    When N <= N_ad, F_conv = 0.
    
    Args:
        rho: Density at layer centers (g/cm^3) - will convert to kg/m³ internally
        c_p: Specific heat capacity (erg g^-1 K^-1) - will convert to J/(kg·K) internally
        alpha: Mixing length parameter (dimensionless, α in l = α × H_p)
        g: Gravity (m/s^2)
        T: Temperature at layer centers (K) - need to interpolate from interfaces
        N: Temperature gradient at layer centers (K/m)
        N_ad: Adiabatic gradient (K/m)
        mmw: Mean molecular weight (g/mol) - needed for H_p calculation
    
    Returns:
        F_conv: Convective flux at layer centers (erg cm^-2 s^-1) for compatibility
    """
    # Convert inputs to SI
    rho_SI = g_per_cm3_to_kg_per_m3(rho)  # kg/m³
    c_p_SI = erg_per_gK_to_J_per_kgK(c_p)  # J/(kg·K)
    mmw_kg = g_per_mol_to_kg_per_mol(mmw)  # kg/mol
    
    # Calculate pressure scale height: H_p = RT/(μ·g)
    H_p = (R_SI * T) / (mmw_kg * g)  # m
    
    # Calculate mixing length: l = α × H_p
    l = alpha * H_p  # m
    
    # Calculate (N - N_ad), but only where N > N_ad
    delta_N = N - N_ad
    delta_N = np.maximum(delta_N, 0.0)  # Floor at zero to avoid complex numbers
    
    # Calculate sqrt(g/T)
    sqrt_g_T = np.sqrt(g / T)
    
    # Calculate (N - N_ad)^(3/2)
    delta_N_power = delta_N ** 1.5
    
    # Calculate l²
    l_squared = l ** 2
    
    # Full formula in SI: F_conv = ρ × c_p × l² × sqrt(g/T) × (N - N_ad)^(3/2)
    # Check for potential overflow before multiplication
    max_factor = np.max([np.max(rho_SI), np.max(c_p_SI), np.max(l_squared), 
                         np.max(sqrt_g_T), np.max(delta_N_power)])
    if max_factor > 1e100:
        raise ValueError(f"Flux calculation would overflow: one of the factors is too large ({max_factor:.2e}). "
                        f"This usually means alpha is too large or the temperature gradient is extreme.")
    
    F_conv_SI = rho_SI * c_p_SI * l_squared * sqrt_g_T * delta_N_power  # W/m²
    
    # Check for overflow in result
    if np.any(np.isinf(F_conv_SI)) or np.any(F_conv_SI > 1e100):
        raise ValueError(f"Flux calculation overflow: F_conv contains Inf or extremely large values. "
                        f"This usually means alpha is too large for the current conditions.")
    
    # Convert to erg cm^-2 s^-1 for compatibility
    F_conv = W_per_m2_to_erg_per_cm2_s(F_conv_SI)
    
    return F_conv


def check_for_issues(T: np.ndarray, rho: np.ndarray, F_conv: np.ndarray,
                     N: np.ndarray, step: int, debug: bool = False) -> bool:
    """
    Check for numerical issues (NaNs, negative values, etc.).
    
    Args:
        T: Temperature array
        rho: Density array
        F_conv: Convective flux array
        N: Temperature gradient array
        step: Current iteration step
        debug: Whether to print debug messages
    
    Returns:
        has_issues: True if issues found
    """
    has_issues = False
    
    if np.any(np.isnan(T)) or np.any(np.isinf(T)):
        print(f"WARNING [Step {step}]: NaN or Inf in temperature!")
        has_issues = True
    
    if np.any(T < 0):
        print(f"WARNING [Step {step}]: Negative temperatures found! Min T = {np.min(T):.2f} K")
        has_issues = True
    
    if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
        print(f"WARNING [Step {step}]: NaN or Inf in density!")
        has_issues = True
    
    if np.any(rho < 0):
        print(f"WARNING [Step {step}]: Negative densities found!")
        has_issues = True
    
    if np.any(np.isnan(F_conv)) or np.any(np.isinf(F_conv)):
        print(f"WARNING [Step {step}]: NaN or Inf in convective flux!")
        has_issues = True
    
    if np.any(np.isnan(N)) or np.any(np.isinf(N)):
        print(f"WARNING [Step {step}]: NaN or Inf in temperature gradient!")
        has_issues = True
    
    if debug and has_issues:
        print(f"  T range: [{np.nanmin(T):.2f}, {np.nanmax(T):.2f}] K")
        print(f"  rho range: [{np.nanmin(rho):.2f}, {np.nanmax(rho):.2f}] g/cm^3")
        print(f"  F_conv range: [{np.nanmin(F_conv):.2e}, {np.nanmax(F_conv):.2e}] erg cm^-2 s^-1")
    
    return has_issues


def print_convective_layers(N: np.ndarray, N_ad: float, z: np.ndarray, 
                            z_mid: Optional[np.ndarray] = None,
                            T: Optional[np.ndarray] = None) -> None:
    """
    Print detailed information about which layers are convective after iterations.
    
    Args:
        N: Temperature gradient at layer centers (K/m)
        N_ad: Adiabatic gradient (K/m)
        z: Altitude at interfaces (m)
        z_mid: Optional altitude at layer centers (m). If None, calculated as midpoints.
        T: Optional temperature at interfaces (K) for display
    """
    # Calculate layer center altitudes if not provided
    if z_mid is None:
        z_mid = (z[:-1] + z[1:]) / 2.0
    
    # Identify convective and radiative layers
    convective_mask = N > N_ad
    radiative_mask = ~convective_mask
    
    n_convective = np.sum(convective_mask)
    n_radiative = np.sum(radiative_mask)
    n_total = len(N)
    
    print("=" * 70)
    print("Convective Layer Analysis")
    print("=" * 70)
    print(f"Adiabatic gradient: N_ad = {N_ad:.6f} K/m")
    print(f"Total layers: {n_total}")
    print(f"Convective layers (N > N_ad): {n_convective}")
    print(f"Radiative layers (N <= N_ad): {n_radiative}")
    print()
    
    if n_convective > 0:
        print("CONVECTIVE LAYERS:")
        print("-" * 70)
        print(f"{'Layer':<8} {'Altitude (km)':<15} {'N (K/m)':<15} {'N/N_ad':<12} {'|N-N_ad|/N_ad':<15}")
        print("-" * 70)
        
        convective_indices = np.where(convective_mask)[0]
        for idx in convective_indices:
            z_km = z_mid[idx] / 1000.0
            N_val = N[idx]
            N_ratio = N_val / N_ad
            relative_diff = np.abs(N_val - N_ad) / N_ad
            
            print(f"{idx:<8} {z_km:<15.2f} {N_val:<15.6f} {N_ratio:<12.4f} {relative_diff:<15.4f}")
        
        print()
        # Summary statistics for convective layers
        convective_N = N[convective_mask]
        print(f"Convective layer statistics:")
        print(f"  Min N: {np.min(convective_N):.6f} K/m ({np.min(convective_N)/N_ad:.4f} × N_ad)")
        print(f"  Max N: {np.max(convective_N):.6f} K/m ({np.max(convective_N)/N_ad:.4f} × N_ad)")
        print(f"  Mean N: {np.mean(convective_N):.6f} K/m ({np.mean(convective_N)/N_ad:.4f} × N_ad)")
        relative_diffs = np.abs(convective_N - N_ad) / N_ad
        print(f"  Mean |N-N_ad|/N_ad: {np.mean(relative_diffs):.4f}")
        print(f"  Max |N-N_ad|/N_ad: {np.max(relative_diffs):.4f}")
    else:
        print("No convective layers found - entire atmosphere is radiative.")
        print()
    
    if n_radiative > 0 and n_convective > 0:
        # Find radiative-convective boundary
        # Look for transition from convective (bottom) to radiative (top)
        radiative_indices = np.where(radiative_mask)[0]
        convective_indices = np.where(convective_mask)[0]
        
        if len(convective_indices) > 0 and len(radiative_indices) > 0:
            # Find highest convective layer center index
            highest_conv_idx = np.max(convective_indices)
            # Find lowest radiative layer center above convective layers
            radiative_above = radiative_indices[radiative_indices > highest_conv_idx]
            if len(radiative_above) > 0:
                # Boundary is at the interface between highest convective layer and first radiative layer
                # Interface index = highest_conv_idx + 1 (since layer centers are between interfaces)
                boundary_interface_idx = highest_conv_idx + 1
                if boundary_interface_idx < len(z):
                    boundary_altitude = z[boundary_interface_idx] / 1000.0
                    print(f"Radiative-Convective Boundary: Interface {boundary_interface_idx} at {boundary_altitude:.2f} km")
                    print(f"  (Between convective layer {highest_conv_idx} and radiative layer {radiative_above[0]})")
                    print()
    
    print("=" * 70)
    print()


def print_mixing_length_interpretation(alpha: float, z: np.ndarray, z_mid: np.ndarray, 
                                       T: np.ndarray, rho: np.ndarray,
                                       g: float, mmw: float, n_layers: int) -> None:
    """
    Print physical interpretation of mixing length parameter.
    
    Compares the dimensionless mixing length parameter α to:
    - Physical distance in meters (l = α * H_p)
    - Number of layers it spans
    - Fraction of pressure scale height
    - Fraction of layer thickness
    
    Args:
        alpha: Mixing length parameter (dimensionless, α in l = α × H_p)
        z: Altitude at interfaces (m)
        z_mid: Altitude at layer centers (m)
        T: Temperature at interfaces (K)
        rho: Density at interfaces (g/cm^3)
        g: Gravity (m/s^2)
        mmw: Mean molecular weight (g/mol)
        n_layers: Number of layers
    """
    # Universal gas constant in J mol^-1 K^-1 (for pressure scale height)
    R_J = 8.314  # J mol^-1 K^-1
    
    # Calculate layer thickness
    dz = (z[-1] - z[0]) / n_layers  # Average layer thickness in meters
    
    # Calculate pressure scale height at different layers
    # H_p = RT/(μg) where μ is in kg/mol, so convert mmw from g/mol to kg/mol
    mmw_kg = mmw / 1000.0  # kg/mol
    
    # Calculate H_p at layer centers (using T and rho to estimate pressure)
    # For ideal gas: P = ρRT/μ, so we can calculate H_p = RT/(μg) = P/(ρg)
    # But we can also use H_p = RT/(μg) directly
    T_mid = (T[:-1] + T[1:]) / 2.0  # Temperature at layer centers
    H_p = (R_J * T_mid) / (mmw_kg * g)  # Pressure scale height in meters
    
    # Calculate physical mixing length: l = α * H_p
    l_physical = alpha * H_p  # Physical mixing length in meters
    
    # Calculate statistics
    H_p_mean = np.mean(H_p)
    H_p_min = np.min(H_p)
    H_p_max = np.max(H_p)
    
    l_physical_mean = alpha * H_p_mean
    l_physical_min = alpha * H_p_min
    l_physical_max = alpha * H_p_max
    
    # How many layers does this span?
    layers_spanned_mean = l_physical_mean / dz
    layers_spanned_min = l_physical_min / dz
    layers_spanned_max = l_physical_max / dz
    
    print("=" * 70)
    print("Mixing Length Physical Interpretation")
    print("=" * 70)
    print(f"Mixing length parameter: α = {alpha:.3f}")
    print()
    print("Pressure Scale Height (H_p = RT/(μg)):")
    print(f"  Mean H_p: {H_p_mean/1000:.2f} km ({H_p_mean:.0f} m)")
    print(f"  Min H_p:  {H_p_min/1000:.2f} km ({H_p_min:.0f} m) [at top, T={T[-1]:.0f}K]")
    print(f"  Max H_p:  {H_p_max/1000:.2f} km ({H_p_max:.0f} m) [at bottom, T={T[0]:.0f}K]")
    print()
    print("Physical Mixing Length (l = α × H_p):")
    print(f"  Mean: {l_physical_mean/1000:.2f} km ({l_physical_mean:.0f} m)")
    print(f"  Min:  {l_physical_min/1000:.2f} km ({l_physical_min:.0f} m)")
    print(f"  Max:  {l_physical_max/1000:.2f} km ({l_physical_max:.0f} m)")
    print()
    print(f"Layer thickness: dz = {dz/1000:.2f} km ({dz:.0f} m)")
    print()
    print("Mixing Length in Terms of Layers:")
    print(f"  Mean: {layers_spanned_mean:.2f} layers")
    print(f"  Min:  {layers_spanned_min:.2f} layers")
    print(f"  Max:  {layers_spanned_max:.2f} layers")
    print()
    print("Mixing Length as Fraction of Scale Height:")
    print(f"  α = {alpha:.3f}")
    print(f"  This means: l = {alpha:.3f} × H_p")
    print()
    print("Interpretation:")
    if layers_spanned_mean < 0.5:
        print(f"  The mixing length ({l_physical_mean/1000:.2f} km) is less than half a layer.")
        print(f"  Convective parcels travel a very short distance (< {dz/1000:.2f} km).")
    elif layers_spanned_mean < 1.0:
        print(f"  The mixing length ({l_physical_mean/1000:.2f} km) spans less than one layer.")
        print(f"  Convective parcels travel within a single layer.")
    elif layers_spanned_mean < 5.0:
        print(f"  The mixing length ({l_physical_mean/1000:.2f} km) spans {layers_spanned_mean:.1f} layers.")
        print(f"  Convective parcels travel across {layers_spanned_mean:.1f} atmospheric layers.")
    else:
        print(f"  The mixing length ({l_physical_mean/1000:.2f} km) spans {layers_spanned_mean:.1f} layers.")
        print(f"  Convective parcels travel across many layers ({layers_spanned_mean:.1f} layers).")
    
    if alpha < 0.1:
        print(f"  α = {alpha:.3f} is very small - mixing occurs over a small fraction of scale height.")
    elif alpha < 1.0:
        print(f"  α = {alpha:.3f} - mixing occurs over a fraction of the pressure scale height.")
    elif alpha < 2.0:
        print(f"  α = {alpha:.3f} - mixing occurs over approximately one scale height.")
    else:
        print(f"  α = {alpha:.3f} - mixing occurs over multiple scale heights.")
    
    print("=" * 70)
    print()


def print_iteration_tracking(tracking: dict) -> None:
    """
    Print detailed iteration tracking for a specific layer.
    
    Args:
        tracking: Dictionary containing iteration tracking data
    """
    if not tracking or len(tracking['steps']) == 0:
        return
    
    print("=" * 70)
    print("Iteration Tracking for Layer", tracking['layer'])
    print("=" * 70)
    print(f"Layer altitude: z = {tracking['z_mid']/1000:.2f} km")
    print(f"Alpha parameter: α = {tracking['alpha']:.6f}")
    print(f"Adiabatic gradient: N_ad = {tracking['N_ad']:.6f} K/m")
    print(f"Tracked steps: {len(tracking['steps'])}")
    print()
    
    # Print table header
    print(f"{'Step':<6} {'T_mid':<10} {'T':<10} {'dT':<12} {'N':<12} {'N/N_ad':<10} "
          f"{'F_conv':<15} {'dF_dz':<15}")
    print("-" * 80)
    
    # Print each step
    for i, step in enumerate(tracking['steps']):
        T_mid = tracking['T_mid'][i] if i < len(tracking['T_mid']) else np.nan
        T = tracking['T'][i] if i < len(tracking['T']) else np.nan
        dT = tracking['dT'][i] if i < len(tracking['dT']) else np.nan
        N = tracking['N'][i] if i < len(tracking['N']) else np.nan
        N_ad = tracking['N_ad']
        N_ratio = N / N_ad if not np.isnan(N) else np.nan
        F_conv = tracking['F_conv'][i] if i < len(tracking['F_conv']) else np.nan
        dF_dz = tracking['dF_dz'][i] if i < len(tracking['dF_dz']) else np.nan
        
        print(f"{step:<6} {T_mid:<10.1f} {T:<10.1f} {dT:<12.2e} {N:<12.6f} {N_ratio:<10.4f} "
              f"{F_conv:<15.2e} {dF_dz:<15.2e}")
    
    print()
    
    # Show temperature evolution
    if len(tracking['T_mid']) > 0:
        T_init = tracking['T_mid'][0]
        T_final = tracking['T_mid'][-1]
        delta_T_total = T_final - T_init
        print(f"Temperature evolution:")
        print(f"  Initial T_mid: {T_init:.2f} K")
        print(f"  Final T_mid: {T_final:.2f} K")
        print(f"  Total change: {delta_T_total:.2f} K ({delta_T_total/T_init*100:.2f}%)")
        print()
    
    print("=" * 70)
    print()


def check_adiabatic_convergence(N: np.ndarray, N_ad: float, tolerance: float = 0.5, debug: bool = False) -> bool:
    """
    Check if temperature gradients are similar to adiabatic gradient (convergence criterion).
    
    Only checks CONVECTIVE layers (where N > N_ad). Radiative layers (N <= N_ad) are not required
    to be adiabatic, as they are in radiative equilibrium and stable.
    
    For convective layers: require |N - N_ad|/N_ad < tolerance
    This means N/N_ad < 1 + tolerance (e.g., with tolerance=0.5, require N/N_ad < 1.5)
    
    Args:
        N: Temperature gradient at layer centers (K/m)
        N_ad: Adiabatic gradient (K/m)
        tolerance: Fractional tolerance (default 0.5 = 50%)
                   For convective layers (N > N_ad), require |N - N_ad|/N_ad < tolerance
                   This means N/N_ad < 1 + tolerance (e.g., N/N_ad < 1.5 for tolerance=0.5)
                   Radiative layers (N <= N_ad) are acceptable as-is
    
    Returns:
        converged: True if all CONVECTIVE layers have |N - N_ad|/N_ad < tolerance
                  Returns True if there are no convective layers (all radiative)
    """
    # Identify convective layers (N > N_ad)
    convective_mask = N > N_ad
    
    # If no convective layers, consider it converged (all radiative is valid)
    if not np.any(convective_mask):
        return True
    
    # Only check convergence for convective layers
    # For convective layers: require |N - N_ad| / N_ad < tolerance
    # This means N/N_ad < 1 + tolerance (e.g., N/N_ad < 1.5 for tolerance=0.5)
    convective_N = N[convective_mask]
    relative_diff = np.abs(convective_N - N_ad) / N_ad
    converged = np.all(relative_diff < tolerance)
    
    if debug:
        n_conv = np.sum(convective_mask)
        print(f"  Adiabaticity convergence check:")
        print(f"    Convective layers: {n_conv} / {len(N)}")
        if n_conv > 0:
            print(f"    N_ad = {N_ad:.6f} K/m")
            print(f"    |N - N_ad|/N_ad range: [{np.min(relative_diff):.4f}, {np.max(relative_diff):.4f}]")
            print(f"    Max deviation: {np.max(relative_diff):.4f} ({np.max(relative_diff)*100:.1f}%)")
            print(f"    Tolerance: {tolerance:.2f} ({tolerance*100:.0f}%)")
            print(f"    Converged: {converged}")
    
    return converged


def convective_timescale(g: float, T: np.ndarray, N: np.ndarray, N_ad: float) -> np.ndarray:
    """
    Calculate convective timescale at each layer.
    
    t_conv = 1 / sqrt[(g/T) * (N - N_ad)]
    
    Where:
        g: Gravity (m/s²)
        T: Temperature at layer centers (K)
        N: Temperature gradient at layer centers (K/m)
        N_ad: Adiabatic gradient (K/m)
    
    Args:
        g: Gravity (m/s²)
        T: Temperature at layer centers (K)
        N: Temperature gradient at layer centers (K/m)
        N_ad: Adiabatic gradient (K/m)
    
    Returns:
        t_conv: Convective timescale at layer centers (s)
    """
    # Calculate (N - N_ad), but only where N > N_ad (convective layers)
    # For non-convective layers, set to a small positive value to avoid division by zero
    delta_N = N - N_ad
    delta_N = np.maximum(delta_N, 1e-10)  # Floor at small positive value
    
    # Calculate (g/T) * (N - N_ad)
    # Units: (m/s²) / K * (K/m) = 1/s²
    term = (g / T) * delta_N
    
    # Calculate t_conv = 1 / sqrt(term)
    # Units: 1 / (1/s) = s
    t_conv = 1.0 / np.sqrt(term)
    
    # For non-convective layers (where N <= N_ad), set timescale to infinity
    # or a very large value to indicate no convection
    non_convective = N <= N_ad
    t_conv[non_convective] = np.inf
    
    return t_conv


def radiative_timescale(P: np.ndarray, g: float, c_p: float, T: np.ndarray) -> np.ndarray:
    """
    Calculate radiative timescale at each layer.
    
    τ_rad ≈ (P/g) * (c_P / (4σT³))
    
    Where:
        P: Pressure at layer centers (dyne/cm²) - will convert to Pa
        g: Gravity (m/s²)
        c_P: Specific heat capacity (erg/(g·K)) - will convert to J/(kg·K)
        T: Temperature at layer centers (K)
        σ: Stefan-Boltzmann constant (W/(m²·K⁴))
    
    Args:
        P: Pressure at layer centers (dyne/cm²)
        g: Gravity (m/s²)
        c_p: Specific heat capacity (erg/(g·K))
        T: Temperature at layer centers (K)
    
    Returns:
        tau_rad: Radiative timescale at layer centers (s)
    """
    # Convert P from dyne/cm² to Pa
    P_Pa = dyne_per_cm2_to_Pa(P)  # Pa = kg/(m·s²)
    
    # Convert c_p from erg/(g·K) to J/(kg·K)
    c_p_SI = erg_per_gK_to_J_per_kgK(c_p)  # J/(kg·K) = m²/(s²·K)
    
    # Calculate term 1: P/g
    # Units: [kg/(m·s²)] / [m/s²] = kg/m
    term1 = P_Pa / g
    
    # Calculate term 2: c_P / (4σT³)
    # Units: [m²/(s²·K)] / ([kg/(s³·K⁴)] * [K³]) = [m²/(s²·K)] / [kg/(s³·K)] = m²·s/kg
    term2 = c_p_SI / (4.0 * SIGMA_SB * T**3)
    
    # Calculate τ_rad = term1 * term2
    # Units: [kg/m] * [m²·s/kg] = s
    tau_rad = term1 * term2
    
    return tau_rad


# ============================================================================
# MAIN SOLVER
# ============================================================================

def run(n_layers: int = N_LAYERS, max_z: float = MAX_Z,
        T_toa: float = T_TOA, T_boa: float = T_BOA,
        rho_toa: float = RHO_TOA, rho_boa: float = RHO_BOA,
        g: float = G, alpha: float = ALPHA, dt: float = DT,
        max_steps: int = MAX_STEPS, convergence_tol: float = CONVERGENCE_TOL,
        debug: bool = False, debug_interval: int = DEBUG_INTERVAL,
        n_dof: int = N_DOF, mmw: float = MMW,
        save_history: bool = False,
        history_interval: Optional[int] = None,
        profile_type: str = "guillot",
        guillot_params: Optional[dict] = None,
        check_adiabatic: bool = False,
        adiabatic_tolerance: float = 0.2,
        use_energy_conservation: bool = True,
        use_constant_dt_coefficient: bool = False,
        dt_constant_value: Optional[float] = 1,
        track_layer: Optional[int] = None,
        track_steps: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run the iterative convective flux solver.
    
    Args:
        n_layers: Number of layers
        max_z: Maximum altitude (m)
        T_toa: Temperature at top of atmosphere (K)
        T_boa: Temperature at bottom of atmosphere (K)
        rho_toa: Density at top (g/cm^3)
        rho_boa: Density at bottom (g/cm^3)
        g: Gravity (m/s^2)
        alpha: Mixing length parameter (dimensionless, α in l = α × H_p)
        dt: Timestep (s)
        max_steps: Maximum iteration steps
        convergence_tol: Convergence tolerance for max|dT| (K)
        debug: Enable verbose debug output
        debug_interval: Print debug info every N steps
        n_dof: Degrees of freedom for composition
        mmw: Mean molecular weight (g/mol)
    
    Returns:
        z: Altitude at interfaces (m)
        T: Final temperature at interfaces (K)
        rho: Final density at interfaces (g/cm^3)
        P: Final pressure at interfaces (dyne/cm^2)
        diagnostics: Dictionary with convergence info
    """
    print("=" * 70)
    print("Convective Flux Solver - Initialization")
    print("=" * 70)
    
    # Calculate c_p
    c_p = calculate_c_p(n_dof, mmw)
    print(f"Specific heat capacity: c_p = {c_p:.2e} erg g^-1 K^-1")
    
    # Calculate adiabatic gradient
    N_ad = adiabatic_gradient(g, c_p)
    print(f"Adiabatic gradient: N_ad = {N_ad:.4f} K/m")
    
    # Set up grid
    z, z_mid, dz = setup_grid(n_layers, max_z)
    print(f"Grid: {n_layers} layers, dz = {dz/1000:.2f} km")
    print(f"  Altitude range: 0 to {max_z/1000:.1f} km")
    
    # Initialize profiles
    T, rho = initialize_profiles(z, z_mid, T_toa, T_boa, rho_toa, rho_boa,
                                  profile_type=profile_type, guillot_params=guillot_params, g=g)
    T_initial = T.copy()  # Save initial temperature for plotting
    
    # Calculate initial pressure at interfaces from ideal gas law: P = ρ * R_specific * T
    # R_specific = R / mmw, where R is in erg mol^-1 K^-1, mmw is in g/mol
    # rho in g/cm^3, T in K, so P = (g/cm^3) * (erg mol^-1 K^-1 / g mol^-1) * K = erg/cm^3 = dyne/cm^2
    R_specific = R / mmw  # erg g^-1 K^-1 (specific gas constant)
    P = rho * R_specific * T  # Pressure in dyne/cm^2 at interfaces
    
    print(f"Profile type: {profile_type}")
    print(f"Initial T range: [{np.min(T):.1f}, {np.max(T):.1f}] K")
    print(f"Initial rho range: [{np.min(rho):.3f}, {np.max(rho):.1f}] g/cm^3")
    print(f"Initial P range: [{np.min(P)/1e6:.3e}, {np.max(P)/1e6:.3e}] bar")
    if use_constant_dt_coefficient:
        print(f"Temperature update method: Constant coefficient = {dt_constant_value:.2e} m·s²·K/kg")
    elif use_energy_conservation:
        print(f"Temperature update method: Energy conservation (1/(ρc_p))")
    else:
        print(f"Temperature update method: Energy conservation (1/(ρc_p)) [default for l^2 version]")
    print()
    
    # Initialize iteration tracking if requested
    iteration_tracking = None
    if track_layer is not None:
        if track_layer < 0 or track_layer >= n_layers:
            print(f"WARNING: track_layer={track_layer} out of range [0, {n_layers-1}], using middle layer")
            track_layer = n_layers // 2
        z_track = z_mid[track_layer] if track_layer < len(z_mid) else (z[track_layer] + z[track_layer+1])/2.0
        iteration_tracking = {
            'layer': track_layer,
            'steps': [],
            'T': [],  # Temperature at interface below layer
            'T_mid': [],  # Temperature at layer center
            'N': [],  # Temperature gradient
            'N_ad': N_ad,
            'F_conv': [],  # Convective flux
            'dF_dz': [],  # Flux divergence at interface
            'dT': [],  # Temperature change
            'alpha': alpha,
            'rho_mid': [],  # Density at layer center
            'z_mid': z_track
        }
        print(f"Iteration tracking enabled for layer {track_layer} (z = {z_track/1000:.2f} km)")
        print()
    
    # Convert density units: g/cm^3 to g/m^3 for consistency with gravity
    # Actually, let's work in cgs throughout and convert only where needed
    # For now, keep rho in g/cm^3 but be careful with units in flux calculation
    
    # Initialize history tracking if requested
    if save_history:
        history_T = []  # Temperature at interfaces
        history_dT = []  # Temperature change at interfaces
        history_F = []  # Flux at layer centers
        history_dF = []  # Flux change (dF/dz) at interfaces
        history_t_conv = []  # Convective timescale at layer centers
        history_tau_rad = []  # Radiative timescale at layer centers
        timesteps = []
    
    # Main iteration loop
    print("=" * 70)
    print("Starting iteration loop...")
    print("=" * 70)
    
    for step in range(max_steps):
        # Interpolate T and rho to layer centers for flux calculation
        T_mid = (T[:-1] + T[1:]) / 2.0
        rho_mid = (rho[:-1] + rho[1:]) / 2.0
        
        # Calculate temperature gradient at layer centers
        N = temperature_gradient(T, z)
        
        # Calculate convective flux at layer centers
        # This will also verify alpha usage if debug is enabled
        F_conv = convective_flux(rho_mid, c_p, alpha, g, T_mid, N, N_ad, mmw)
        
        # Calculate timescales at layer centers
        # Need pressure at layer centers for radiative timescale
        P_mid = (P[:-1] + P[1:]) / 2.0  # Pressure at layer centers (dyne/cm²)
        t_conv = convective_timescale(g, T_mid, N, N_ad)
        tau_rad = radiative_timescale(P_mid, g, c_p, T_mid)
        
        # Check for numerical issues
        check_for_issues(T, rho, F_conv, N, step, debug)
        
        # Debug output for first step
        if step == 0 and debug:
            print(f"\nDEBUG [Step 0]:")
            print(f"  Alpha parameter: α = {alpha:.6f} (dimensionless)")
            print(f"  Sample layer (middle): z_mid = {z_mid[len(z_mid)//2]/1000:.2f} km, T = {T_mid[len(z_mid)//2]:.1f} K")
            print()
            print(f"  N range: [{np.min(N):.6f}, {np.max(N):.6f}] K/m")
            print(f"  N_ad: {N_ad:.6f} K/m")
            print(f"  Convective layers (N > N_ad): {np.sum(N > N_ad)} / {len(N)}")
            print(f"  F_conv range: [{np.min(F_conv):.2e}, {np.max(F_conv):.2e}] erg cm^-2 s^-1")
            print(f"  T_mid range: [{np.min(T_mid):.1f}, {np.max(T_mid):.1f}] K")
            print(f"  rho_mid range: [{np.min(rho_mid):.3f}, {np.max(rho_mid):.1f}] g/cm^3")
        
        # Calculate dF_conv/dz at interfaces
        # F_conv is at layer centers, we need dF/dz at interfaces to update T at interfaces
        # This will be computed below in the update section with proper units
        
        # Check for NaN/Inf in flux before calculating divergence
        if np.any(np.isnan(F_conv)) or np.any(np.isinf(F_conv)):
            print(f"\n{'='*70}")
            print(f"FLUX EXPLOSION DETECTED at step {step+1}!")
            print(f"{'='*70}")
            print(f"  NaN in F_conv: {np.any(np.isnan(F_conv))}")
            print(f"  Inf in F_conv: {np.any(np.isinf(F_conv))}")
            print(f"  Parameters: alpha={alpha}, dt={dt} s")
            print(f"  F_conv range: [{np.nanmin(F_conv):.2e}, {np.nanmax(F_conv):.2e}] erg cm⁻² s⁻¹")
            print(f"  → Flux calculation exploded. This usually means dt is too large for alpha={alpha}.")
            print(f"  → Try reducing dt significantly (e.g., dt < {dt/10:.1f} s)")
            print(f"{'='*70}\n")
            raise ValueError(f"Flux explosion: NaN/Inf in F_conv at step {step+1}. "
                           f"dt={dt} s is too large for alpha={alpha}. "
                           f"Try reducing dt significantly.")
        
        # Update temperature: dT = dt * constant * dF_conv/dz
        # The constant is set to 1 for now (not related to density)
        # dT = dt * 1 * dF_dz
        
        # F_conv is in erg cm^-2 s^-1 at layer centers
        # dF_dz is computed at interfaces in erg cm^-2 s^-1 / m
        # Need to convert dz to cm for consistent units: 1 m = 100 cm
        dz_cm = dz * 100.0  # cm
        
        # dF_dz at interfaces (already computed above in erg cm^-2 s^-1 / m)
        # Convert to erg cm^-2 s^-1 / cm for consistency
        dF_dz_erg_cm3_s = np.zeros(len(z))  # erg cm^-3 s^-1 at interfaces
        
        # Interior interfaces
        dF_dz_erg_cm3_s[1:-1] = (F_conv[1:] - F_conv[:-1]) / dz_cm
        
        # Boundaries
        dF_dz_erg_cm3_s[0] = (F_conv[0] - 0) / (dz_cm / 2.0)
        dF_dz_erg_cm3_s[-1] = (0 - F_conv[-1]) / (dz_cm / 2.0)
        
        # Update: dT = -dt * constant * dF_dz
        # Negative sign because: if dF/dz > 0 (flux increasing upward), more energy leaves -> cooling (dT < 0)
        # 
        # Option 1: Constant = 1.0 (simplified, as per Overview.txt)
        # Option 2: Energy conservation form: dT = -dt / (ρ * c_p) * dF_dz
        #   Physical relationship: ∂T/∂t = -1/(ρc_p) * ∂F/∂z (Manabe & Strickler 1964, standard RCE models)
        #   Reference: Manabe & Strickler (1964), standard in radiative-convective equilibrium models
        #   Units: dT [K] = dt [s] * dF_dz [erg cm^-3 s^-1] / (ρ [g/cm^3] * c_p [erg g^-1 K^-1]) = [K] ✓
        # Option 3: User-specified constant coefficient (SI units: m·s²·K/kg)
        
        if use_constant_dt_coefficient:
            if dt_constant_value is None:
                raise ValueError("dt_constant_value must be provided when use_constant_dt_coefficient=True")
            # Convert dF_dz from erg cm^-3 s^-1 to W/m³ (SI)
            # 1 erg = 10^-7 J, 1 cm³ = 10^-6 m³
            # So: 1 erg cm^-3 s^-1 = 10^-7 J / (10^-6 m³·s) = 10^-1 J/(m³·s) = 0.1 W/m³
            dF_dz_SI = dF_dz_erg_cm3_s * 0.1  # erg cm^-3 s^-1 -> W/m³
            # dT/dt = -C × dF/dz in SI
            # dT = -dt × C × dF/dz
            dT = -dt * dt_constant_value * dF_dz_SI
        elif use_energy_conservation:
            # Energy conservation version: dT = -dt / (ρ * c_p) * dF_dz
            # Need rho at interfaces - interpolate from layer centers
            # rho is already at interfaces, so we can use it directly
            # Calculate constant at each interface: 1/(ρ * c_p)
            # Units: [g/cm^3] * [erg g^-1 K^-1] = [erg cm^-3 K^-1]
            # Result: [K] = [s] * [erg cm^-3 s^-1] / [erg cm^-3 K^-1] = [K] ✓
            DT_CONSTANT_interface = 1.0 / (rho * c_p)  # cm^3 K / erg (per interface)
            dT = -dt * DT_CONSTANT_interface * dF_dz_erg_cm3_s
        else:
            # Default: Use energy conservation form (required for l^2 version)
            # The simplified constant = 1.0 is not appropriate when using l^2 instead of alpha^2
            # because l^2 is much larger, causing flux values to explode
            DT_CONSTANT_interface = 1.0 / (rho * c_p)  # cm^3 K / erg (per interface)
            dT = -dt * DT_CONSTANT_interface * dF_dz_erg_cm3_s
        
        # Store history if requested (save at intervals to avoid excessive memory)
        if save_history:
            # Save every N steps, plus first and last steps
            if history_interval is None:
                history_interval_val = max(1, max_steps // 1000)  # Save ~1000 snapshots max
            else:
                history_interval_val = history_interval  # Use provided interval (1 = save every step)
            if step == 0 or step == max_steps - 1 or step % history_interval_val == 0:
                history_T.append(T.copy())
                history_dT.append(dT.copy())
                history_F.append(F_conv.copy())
                # dF_dz is already computed, store it
                history_dF.append(dF_dz_erg_cm3_s.copy())
                # Store timescales
                history_t_conv.append(t_conv.copy())
                history_tau_rad.append(tau_rad.copy())
                timesteps.append(step)
        
        # Store max change for convergence check
        max_dT = np.max(np.abs(dT))
        max_dT_idx = np.argmax(np.abs(dT))  # Index of interface with largest change
        
        # Track iteration for specific layer
        if track_layer is not None and step < track_steps:
            layer_idx = track_layer
            iteration_tracking['steps'].append(step)
            # Get values at layer center and adjacent interfaces
            if layer_idx < len(T_mid):
                iteration_tracking['T_mid'].append(T_mid[layer_idx])
                iteration_tracking['N'].append(N[layer_idx])
                iteration_tracking['F_conv'].append(F_conv[layer_idx])
                iteration_tracking['rho_mid'].append(rho_mid[layer_idx])
            else:
                # Fallback for edge cases
                iteration_tracking['T_mid'].append(T_mid[-1] if len(T_mid) > 0 else T[-1])
                iteration_tracking['N'].append(N[-1] if len(N) > 0 else N_ad)
                iteration_tracking['F_conv'].append(F_conv[-1] if len(F_conv) > 0 else 0.0)
                iteration_tracking['rho_mid'].append(rho_mid[-1] if len(rho_mid) > 0 else rho[-1])
            
            # Interface values (interface below layer, i.e., layer_idx+1)
            interface_idx = layer_idx + 1
            if interface_idx < len(T):
                iteration_tracking['T'].append(T[interface_idx])
                iteration_tracking['dF_dz'].append(dF_dz_erg_cm3_s[interface_idx] if interface_idx < len(dF_dz_erg_cm3_s) else 0.0)
                iteration_tracking['dT'].append(dT[interface_idx] if interface_idx < len(dT) else 0.0)
            else:
                iteration_tracking['T'].append(T[-1])
                iteration_tracking['dF_dz'].append(0.0)
                iteration_tracking['dT'].append(0.0)
        
        # Debug output for first step
        if step == 0 and debug:
            print(f"  dF_dz_erg_cm3_s range: [{np.min(dF_dz_erg_cm3_s):.2e}, {np.max(dF_dz_erg_cm3_s):.2e}] erg cm^-3 s^-1")
            print(f"  dt: {dt} s")
            if use_energy_conservation or (not use_constant_dt_coefficient):
                print(f"  constant: 1/(ρ*c_p) (varies by interface)")
                print(f"    constant range: [{np.min(1.0/(rho*c_p)):.2e}, {np.max(1.0/(rho*c_p)):.2e}] cm^3 K / erg")
                print(f"    Physical relationship: ∂T/∂t = -1/(ρc_p) * ∂F/∂z (Manabe & Strickler 1964)")
            else:
                print(f"  constant: {dt_constant_value:.2e} m·s²·K/kg (user-specified)")
            print(f"  dT range: [{np.min(dT):.2e}, {np.max(dT):.2e}] K")
            print(f"  max|dT|: {max_dT:.2e} K at interface {max_dT_idx} (z = {z[max_dT_idx]/1000:.1f} km)")
            print()
            
            # Detailed breakdown for problematic interface
            print(f"  DETAILED ANALYSIS for interface {max_dT_idx} (z = {z[max_dT_idx]/1000:.1f} km):")
            print(f"    T[{max_dT_idx}] = {T[max_dT_idx]:.2f} K")
            print(f"    dT[{max_dT_idx}] = {dT[max_dT_idx]:.2e} K")
            print(f"    dF_dz[{max_dT_idx}] = {dF_dz_erg_cm3_s[max_dT_idx]:.2e} erg cm^-3 s^-1")
            
            # Show flux values at adjacent layer centers
            if max_dT_idx == 0:
                print(f"    F_conv[0] (layer center below) = {F_conv[0]:.2e} erg cm^-2 s^-1")
                print(f"    Boundary: assuming F=0 below, so dF_dz = F[0] / (dz/2)")
            elif max_dT_idx == len(z) - 1:
                print(f"    F_conv[{len(F_conv)-1}] (layer center above) = {F_conv[-1]:.2e} erg cm^-2 s^-1")
                print(f"    Boundary: assuming F=0 above, so dF_dz = -F[-1] / (dz/2)")
            else:
                idx_lower = max_dT_idx - 1  # Layer center below interface
                idx_upper = max_dT_idx      # Layer center above interface
                print(f"    F_conv[{idx_lower}] (layer below) = {F_conv[idx_lower]:.2e} erg cm^-2 s^-1")
                print(f"    F_conv[{idx_upper}] (layer above) = {F_conv[idx_upper]:.2e} erg cm^-2 s^-1")
                print(f"    dF_dz = (F[{idx_upper}] - F[{idx_lower}]) / dz")
                print(f"          = ({F_conv[idx_upper]:.2e} - {F_conv[idx_lower]:.2e}) / {dz_cm:.2e} cm")
                print(f"          = {dF_dz_erg_cm3_s[max_dT_idx]:.2e} erg cm^-3 s^-1")
            
            # Show the constant used (depends on method)
            if use_constant_dt_coefficient:
                print(f"    Using constant coefficient: {dt_constant_value:.2e} m·s²·K/kg")
            elif use_energy_conservation or (not use_constant_dt_coefficient):
                const_val = DT_CONSTANT_interface[max_dT_idx] if max_dT_idx < len(DT_CONSTANT_interface) else DT_CONSTANT_interface[-1]
                print(f"    dT = -dt * constant * dF_dz = -{dt} * {const_val:.2e} * {dF_dz_erg_cm3_s[max_dT_idx]:.2e}")
            else:
                print(f"    dT = -dt * constant * dF_dz = -{dt} * {dt_constant_value:.2e} * {dF_dz_erg_cm3_s[max_dT_idx]:.2e}")
            print(f"       = {dT[max_dT_idx]:.2e} K")
            print()
            print(f"  SIGN CHECK:")
            print(f"    If dF/dz > 0: flux increasing upward -> more energy leaving -> should COOL (dT < 0)")
            print(f"    If dF/dz < 0: flux decreasing upward -> more energy entering -> should HEAT (dT > 0)")
            print(f"    Current: dF_dz[{max_dT_idx}] = {dF_dz_erg_cm3_s[max_dT_idx]:.2e}, dT[{max_dT_idx}] = {dT[max_dT_idx]:.2e}")
            if dF_dz_erg_cm3_s[max_dT_idx] > 0 and dT[max_dT_idx] > 0:
                print(f"    WARNING: dF/dz > 0 but dT > 0 - WRONG SIGN! Should be negative.")
            elif dF_dz_erg_cm3_s[max_dT_idx] < 0 and dT[max_dT_idx] < 0:
                print(f"    WARNING: dF/dz < 0 but dT < 0 - WRONG SIGN! Should be positive.")
            print()
            
            # Show all interfaces with significant changes
            print(f"  All interfaces with |dT| > 1 K:")
            significant = np.where(np.abs(dT) > 1.0)[0]
            if len(significant) > 0:
                for idx in significant[:10]:  # Show first 10
                    print(f"    Interface {idx:2d}: z={z[idx]/1000:6.1f} km, "
                          f"T={T[idx]:7.1f} K, dT={dT[idx]:8.2f} K, "
                          f"dF_dz={dF_dz_erg_cm3_s[idx]:.2e} erg cm^-3 s^-1")
                if len(significant) > 10:
                    print(f"    ... and {len(significant)-10} more")
            else:
                print(f"    (none)")
            print()
            
            # Show flux pattern across all layers
            print(f"  Flux pattern (F_conv at layer centers):")
            print(f"    {'Layer':<6} {'z (km)':<10} {'F_conv':<20} {'dF_dz (interface above)':<25} {'dT (interface above)':<20}")
            print(f"    {'-'*6} {'-'*10} {'-'*20} {'-'*25} {'-'*20}")
            for i in range(min(10, len(F_conv))):  # Show first 10 layers
                z_center = (z[i] + z[i+1]) / 2.0 / 1000.0
                if i < len(dF_dz_erg_cm3_s) - 1:
                    dF_dz_val = dF_dz_erg_cm3_s[i+1]  # dF/dz at interface above this layer
                    dT_val = dT[i+1]  # dT at interface above this layer
                else:
                    dF_dz_val = 0.0
                    dT_val = 0.0
                print(f"    {i:<6} {z_center:<10.1f} {F_conv[i]:<20.2e} {dF_dz_val:<25.2e} {dT_val:<20.2e}")
            if len(F_conv) > 10:
                print(f"    ... and {len(F_conv)-10} more layers")
            print()
        
        # Update temperature
        T_new = T + dT
        
        # Check for temperature explosions (instability detection)
        has_nan = np.any(np.isnan(T_new))
        has_inf = np.any(np.isinf(T_new))
        if has_nan or has_inf:
            print(f"\n{'='*70}")
            print(f"TEMPERATURE EXPLOSION DETECTED at step {step+1}!")
            print(f"{'='*70}")
            print(f"  NaN detected: {has_nan}")
            print(f"  Inf detected: {has_inf}")
            print(f"  Parameters: alpha={alpha}, dt={dt} s")
            print(f"  T range before update: [{np.nanmin(T):.2f}, {np.nanmax(T):.2f}] K")
            print(f"  dT range: [{np.nanmin(dT):.2e}, {np.nanmax(dT):.2e}] K")
            print(f"  max|dT|/T ratio: {np.nanmax(np.abs(dT) / np.maximum(T, 1.0)):.2e}")
            
            # Find problematic interface
            if has_nan:
                bad_idx = np.where(np.isnan(T_new))[0][0]
            else:
                bad_idx = np.where(np.isinf(T_new))[0][0]
            
            print(f"\n  Problematic interface {bad_idx} (z = {z[bad_idx]/1000:.1f} km):")
            print(f"    T[{bad_idx}] = {T[bad_idx]:.2f} K")
            print(f"    dT[{bad_idx}] = {dT[bad_idx]:.2e} K")
            print(f"    dT/T ratio = {dT[bad_idx]/T[bad_idx]:.2e}")
            print(f"    rho[{bad_idx}] = {rho[bad_idx]:.2e} g/cm³")
            print(f"    c_p = {c_p:.2e} erg/(g·K)")
            print(f"    1/(ρ*c_p) = {1.0/(rho[bad_idx]*c_p):.2e} cm³·K/erg")
            print(f"    dF_dz[{bad_idx}] = {dF_dz_erg_cm3_s[bad_idx]:.2e} erg cm⁻³ s⁻¹")
            
            # Calculate stability criterion
            # For stability: |dT| < f * T, where f is a safety factor (e.g., 0.1 = 10% change max)
            # This gives: dt * (1/(ρ*c_p)) * |dF_dz| < f * T
            # Rearranging: dt < f * T * (ρ*c_p) / |dF_dz|
            dF_dz_abs = np.abs(dF_dz_erg_cm3_s[bad_idx])
            if np.isnan(dF_dz_abs) or np.isinf(dF_dz_abs) or dF_dz_abs < 1e-30:
                print(f"\n  STABILITY ANALYSIS:")
                if np.isnan(dF_dz_abs) or np.isinf(dF_dz_abs):
                    print(f"    dF_dz is NaN/Inf - flux already exploded")
                else:
                    print(f"    dF_dz is very small ({dF_dz_abs:.2e}), cannot estimate stable dt")
                print(f"    → Try reducing dt by factor of 10-100")
                stability_dt = dt * 0.01  # Suggest reducing by factor of 100
            else:
                stability_dt = 0.1 * T[bad_idx] * (rho[bad_idx] * c_p) / dF_dz_abs
                print(f"\n  STABILITY ANALYSIS:")
                print(f"    Current dt = {dt:.2e} s")
                print(f"    Estimated stable dt < {stability_dt:.2e} s (for 10% max change)")
                if stability_dt > 0:
                    print(f"    dt ratio (current/stable) = {dt/stability_dt:.2e}")
                    print(f"    → dt is {dt/stability_dt:.1f}x too large for stability!")
                else:
                    print(f"    → dt must be much smaller (flux divergence too large)")
            
            # Also check flux values
            if bad_idx > 0 and bad_idx <= len(F_conv):
                print(f"\n  Flux context:")
                if bad_idx == len(z) - 1:
                    print(f"    F_conv[{len(F_conv)-1}] = {F_conv[-1]:.2e} erg cm⁻² s⁻¹")
                else:
                    print(f"    F_conv[{bad_idx-1}] (below) = {F_conv[bad_idx-1]:.2e} erg cm⁻² s⁻¹")
                    if bad_idx < len(F_conv):
                        print(f"    F_conv[{bad_idx}] (above) = {F_conv[bad_idx]:.2e} erg cm⁻² s⁻¹")
            
            print(f"{'='*70}\n")
            if not (np.isnan(dF_dz_abs) or np.isinf(dF_dz_abs)) and dF_dz_abs > 1e-30 and stability_dt > 0:
                raise ValueError(f"Temperature explosion: NaN/Inf detected at step {step+1}. "
                               f"dt={dt} s is too large for alpha={alpha}. "
                               f"Try dt < {stability_dt:.2e} s for stability.")
            else:
                raise ValueError(f"Temperature explosion: NaN/Inf detected at step {step+1}. "
                               f"dt={dt} s is too large for alpha={alpha}. "
                               f"Try reducing dt by factor of 10-100 (e.g., dt < {dt/10:.1f} s).")
        
        # Check for excessive temperature changes (warning before explosion)
        # Only warn if not already exploding (to avoid duplicate messages)
        if not (has_nan or has_inf):
            max_dT_ratio = np.max(np.abs(dT) / np.maximum(T, 1.0))
            if max_dT_ratio > 0.5:  # More than 50% change in one step
                max_ratio_idx = np.argmax(np.abs(dT) / np.maximum(T, 1.0))
                # Only print warning occasionally to avoid spam
                if step < 10 or step % 100 == 0:
                    print(f"\nWARNING at step {step+1}: Large temperature change detected!")
                    print(f"  max|dT|/T = {max_dT_ratio:.2e} (> 0.5 = 50% change)")
                    print(f"  This may lead to instability. Consider reducing dt.")
                    print(f"  Largest change at interface {max_ratio_idx}: dT/T = {np.abs(dT[max_ratio_idx])/T[max_ratio_idx]:.2e}")
        
        # Stability guard: ensure T > 0
        T_new = np.maximum(T_new, 1.0)  # Floor at 1 K
        
        # Update pressure at interfaces from ideal gas law (P = ρ * R_specific * T)
        # Pressure changes when temperature changes (assuming density stays constant)
        P_new = rho * R_specific * T_new  # Pressure in dyne/cm^2 at interfaces
        
        # Update pressure at interfaces from ideal gas law (P = ρ * R_specific * T)
        # Pressure changes when temperature changes (assuming density stays constant)
        P_new = rho * R_specific * T_new  # Pressure in dyne/cm^2 at interfaces
        
        # Check convergence on updated temperature profile
        converged_dt = False
        converged_adiabatic = False
        
        # Recalculate gradient from updated temperature profile for adiabatic check
        N_new = temperature_gradient(T_new, z)
        
        if check_adiabatic:
            # If adiabatic checking is enabled, require BOTH:
            # 1. Temperature changes are small (dT convergence)
            # 2. Convective layers are within tolerance of adiabatic (N/N_ad < 1.5 for N > N_ad)
            #    Radiative layers (N <= N_ad) are fine as-is
            
            # Check dT convergence
            if max_dT < convergence_tol:
                converged_dt = True
            
            # Check adiabaticity convergence for convective layers only
            # For convective layers (N > N_ad): require |N - N_ad|/N_ad < 0.5 (i.e., N/N_ad < 1.5)
            # For radiative layers (N <= N_ad): no requirement (already stable)
            converged_adiabatic = check_adiabatic_convergence(N_new, N_ad, adiabatic_tolerance)
            
            # Require both conditions for convergence
            if converged_dt and converged_adiabatic:
                print(f"\nConverged at step {step+1}!")
                print(f"  Max |dT| = {max_dT:.6e} K < tolerance {convergence_tol:.6e} K")
                print(f"  All convective layers within {adiabatic_tolerance*100:.0f}% of adiabatic (N_ad = {N_ad:.6f} K/m)")
                # Update T, P, and N to final values
                T = T_new
                P = P_new
                N = N_new
                break
            elif converged_dt and not converged_adiabatic:
                # dT converged but adiabaticity not reached - continue iterating
                convective_mask = N_new > N_ad
                if np.any(convective_mask):
                    convective_N = N_new[convective_mask]
                    relative_diff = np.abs(convective_N - N_ad) / N_ad
                    max_diff = np.max(relative_diff)
                    if debug and step % debug_interval == 0:
                        print(f"Step {step+1:4d}: dT converged but convective layers not yet adiabatic")
                        print(f"  Max |N-N_ad|/N_ad for convective layers: {max_diff:.4f} (need < {adiabatic_tolerance:.2f})")
                        print(f"  Continuing iteration...")
                # Don't break - continue iterating
            elif converged_adiabatic and not converged_dt:
                # Adiabatic but dT still changing - continue iterating
                if debug and step % debug_interval == 0:
                    print(f"Step {step+1:4d}: Adiabatic but max|dT| = {max_dT:.6e} K (need < {convergence_tol:.6e} K)")
                # Don't break - continue iterating
        else:
            # No adiabatic checking - just check dT convergence
            if max_dT < convergence_tol:
                converged_dt = True
                print(f"\nConverged at step {step+1}!")
                print(f"  Max |dT| = {max_dT:.6e} K < tolerance {convergence_tol:.6e} K")
                # Update T and P to final values
                T = T_new
                P = P_new
                break
        
        # Debug output
        if debug and (step % debug_interval == 0 or step == 0):
            max_dT_idx = np.argmax(np.abs(dT))
            print(f"Step {step+1:4d}: max|dT| = {max_dT:.6f} K at interface {max_dT_idx} "
                  f"(z={z[max_dT_idx]/1000:.1f} km, T={T[max_dT_idx]:.1f} K), "
                  f"max|F_conv| = {np.max(np.abs(F_conv)):.2e} erg cm^-2 s^-1, "
                  f"T range = [{np.min(T_new):.1f}, {np.max(T_new):.1f}] K")
            
            # Show sign distribution
            positive_dT = np.sum(dT > 0)
            negative_dT = np.sum(dT < 0)
            zero_dT = np.sum(dT == 0)
            print(f"         dT signs: {positive_dT} positive (heating), {negative_dT} negative (cooling), {zero_dT} zero")
        
        # Update for next iteration (unless we broke due to convergence)
        T = T_new
        P = P_new
    
    else:
        # Loop completed without convergence
        print(f"\nReached maximum steps ({max_steps}) without convergence")
        print(f"  Final max|dT| = {max_dT:.6f} K")
    
    # Final diagnostics
    print()
    print("=" * 70)
    print("Final Results")
    print("=" * 70)
    print(f"Final T range: [{np.min(T):.1f}, {np.max(T):.1f}] K")
    print(f"Final rho range: [{np.min(rho):.3f}, {np.max(rho):.1f}] g/cm^3")
    print(f"Final P range: [{np.min(P)/1e6:.3e}, {np.max(P)/1e6:.3e}] bar")
    
    # Calculate final flux for output
    T_mid_final = (T[:-1] + T[1:]) / 2.0
    rho_mid_final = (rho[:-1] + rho[1:]) / 2.0
    P_mid_final = (P[:-1] + P[1:]) / 2.0  # Pressure at layer centers (dyne/cm²)
    N_final = temperature_gradient(T, z)
    F_conv_final = convective_flux(rho_mid_final, c_p, alpha, g, T_mid_final, N_final, N_ad, mmw)
    
    # Calculate final timescales
    t_conv_final = convective_timescale(g, T_mid_final, N_final, N_ad)
    tau_rad_final = radiative_timescale(P_mid_final, g, c_p, T_mid_final)
    
    print(f"Final F_conv range: [{np.min(F_conv_final):.2e}, {np.max(F_conv_final):.2e}] erg cm^-2 s^-1")
    print(f"Convective layers: {np.sum(N_final > N_ad)} / {len(N_final)}")
    print(f"N_ad = {N_ad:.6f} K/m, N_final = {N_final}")
    print()
    
    # Print final timescales for each layer
    print("=" * 70)
    print("Final Timescales for Each Layer")
    print("=" * 70)
    print(f"{'Layer':<8} {'Altitude (km)':<15} {'T (K)':<10} {'t_conv (s)':<15} {'τ_rad (s)':<15} {'t_conv/τ_rad':<15}")
    print("-" * 70)
    for i in range(len(z_mid)):
        z_km = z_mid[i] / 1000.0
        T_val = T_mid_final[i]
        t_conv_val = t_conv_final[i]
        tau_rad_val = tau_rad_final[i]
        
        # Format timescales (handle infinity for non-convective layers)
        if np.isinf(t_conv_val):
            t_conv_str = "inf (radiative)"
            ratio_str = "N/A"
        else:
            t_conv_str = f"{t_conv_val:.2e}"
            if tau_rad_val > 0:
                ratio = t_conv_val / tau_rad_val
                ratio_str = f"{ratio:.2e}"
            else:
                ratio_str = "N/A"
        
        tau_rad_str = f"{tau_rad_val:.2e}" if not np.isinf(tau_rad_val) else "inf"
        
        print(f"{i:<8} {z_km:<15.2f} {T_val:<10.1f} {t_conv_str:<15} {tau_rad_str:<15} {ratio_str:<15}")
    
    # Summary statistics
    convective_mask = N_final > N_ad
    if np.any(convective_mask):
        t_conv_convective = t_conv_final[convective_mask]
        t_conv_convective = t_conv_convective[np.isfinite(t_conv_convective)]
        if len(t_conv_convective) > 0:
            print()
            print(f"Convective timescale statistics (convective layers only):")
            print(f"  Min: {np.min(t_conv_convective):.2e} s")
            print(f"  Max: {np.max(t_conv_convective):.2e} s")
            print(f"  Mean: {np.mean(t_conv_convective):.2e} s")
    
    tau_rad_finite = tau_rad_final[np.isfinite(tau_rad_final)]
    if len(tau_rad_finite) > 0:
        print()
        print(f"Radiative timescale statistics:")
        print(f"  Min: {np.min(tau_rad_finite):.2e} s")
        print(f"  Max: {np.max(tau_rad_finite):.2e} s")
        print(f"  Mean: {np.mean(tau_rad_finite):.2e} s")
    
    # Compare timescales where both are finite
    both_finite = np.isfinite(t_conv_final) & np.isfinite(tau_rad_final)
    if np.any(both_finite):
        ratio_final = t_conv_final[both_finite] / tau_rad_final[both_finite]
        print()
        print(f"Timescale ratio (t_conv/τ_rad) statistics (where both finite):")
        print(f"  Min: {np.min(ratio_final):.2e}")
        print(f"  Max: {np.max(ratio_final):.2e}")
        print(f"  Mean: {np.mean(ratio_final):.2e}")
        print(f"  Layers with t_conv < τ_rad (convection faster): {np.sum(ratio_final < 1)}")
        print(f"  Layers with t_conv > τ_rad (radiation faster): {np.sum(ratio_final > 1)}")
    
    print("=" * 70)
    print()
    
    # Print detailed convective layer information
    print_convective_layers(N_final, N_ad, z, z_mid, T)
    
    # Print mixing length physical interpretation
    print_mixing_length_interpretation(alpha, z, z_mid, T, rho, g, mmw, n_layers)
    
    # Print iteration tracking if enabled
    if iteration_tracking is not None:
        print_iteration_tracking(iteration_tracking)
    
    # Calculate final adiabaticity status
    final_adiabatic_converged = False
    final_max_grad_diff = np.nan
    if check_adiabatic:
        final_adiabatic_converged = check_adiabatic_convergence(N_final, N_ad, adiabatic_tolerance, debug=debug)
        # Calculate max gradient difference only for convective layers
        convective_mask = N_final > N_ad
        if np.any(convective_mask):
            convective_N = N_final[convective_mask]
            relative_diff = np.abs(convective_N - N_ad) / N_ad
            final_max_grad_diff = np.max(relative_diff)
            if debug:
                print(f"\nDEBUG: Final adiabaticity check")
                print(f"  Convective layers: {np.sum(convective_mask)} / {len(N_final)}")
                print(f"  N_ad = {N_ad:.6f} K/m")
                print(f"  Convective N range: [{np.min(convective_N):.6f}, {np.max(convective_N):.6f}] K/m")
                print(f"  |N - N_ad|/N_ad range: [{np.min(relative_diff):.4f}, {np.max(relative_diff):.4f}]")
                print(f"  Max deviation: {final_max_grad_diff:.4f} ({final_max_grad_diff*100:.1f}%)")
                print(f"  Tolerance: {adiabatic_tolerance:.2f} ({adiabatic_tolerance*100:.0f}%)")
                print(f"  Converged: {final_adiabatic_converged}")
        else:
            # All radiative - no convective layers to check
            final_max_grad_diff = np.nan
            if debug:
                print(f"\nDEBUG: Final adiabaticity check")
                print(f"  No convective layers found (all radiative)")
                print(f"  Converged: {final_adiabatic_converged}")
    
    diagnostics = {
        'steps': step + 1,
        'converged': max_dT < convergence_tol,
        'converged_adiabatic': final_adiabatic_converged if check_adiabatic else None,
        'max_dT_final': max_dT,
        'max_grad_diff_final': final_max_grad_diff if check_adiabatic else None,
        'c_p': c_p,
        'N_ad': N_ad,
        'F_conv_final': F_conv_final,
        'N_final': N_final,
        't_conv_final': t_conv_final,  # Final convective timescale at layer centers (s)
        'tau_rad_final': tau_rad_final,  # Final radiative timescale at layer centers (s)
        'T_initial': T_initial,  # Save initial temperature for plotting
        'P_final': P,  # Save final pressure for plotting
        'z': z,  # Save altitude grid for plotting
        'z_mid': z_mid,  # Save layer center altitudes
        'use_energy_conservation': use_energy_conservation,  # Track which constant method was used
        'use_constant_dt_coefficient': use_constant_dt_coefficient,
        'dt_constant_value': dt_constant_value,
        'alpha': alpha  # Save alpha for verification
    }
    
    # Add iteration tracking to diagnostics if enabled
    if iteration_tracking is not None:
        diagnostics['iteration_tracking'] = iteration_tracking
    
    # Add history to diagnostics if saved
    if save_history:
        diagnostics['history_T'] = np.array(history_T)
        diagnostics['history_dT'] = np.array(history_dT)
        diagnostics['history_F'] = np.array(history_F)
        diagnostics['history_dF'] = np.array(history_dF)
        diagnostics['history_t_conv'] = np.array(history_t_conv)  # Convective timescale history
        diagnostics['history_tau_rad'] = np.array(history_tau_rad)  # Radiative timescale history
        diagnostics['timesteps'] = np.array(timesteps)
        diagnostics['z_mid'] = z_mid
    
    return z, T, rho, P, diagnostics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_results(diagnostics: dict, output_prefix: str = "convective_flux"):
    """
    Create plots of temperature, dT, flux, and dFlux vs timestep in a 2x2 subplot layout.
    
    Args:
        diagnostics: Dictionary containing history data from run()
        output_prefix: Prefix for output plot filenames
    """
    history_T = diagnostics['history_T']  # Shape: (n_steps, n_interfaces)
    history_dT = diagnostics['history_dT']  # Shape: (n_steps, n_interfaces)
    history_F = diagnostics['history_F']  # Shape: (n_steps, n_layers)
    history_dF = diagnostics['history_dF']  # Shape: (n_steps, n_interfaces)
    timesteps = diagnostics['timesteps']
    z = diagnostics['z']
    z_mid = diagnostics['z_mid']
    
    n_layers = len(z_mid)
    n_interfaces = len(z)
    n_steps = len(timesteps)
    
    # Determine if we should show legend (hide if too many layers)
    show_legend = n_layers <= 10
    
    # Create color map for layers
    colors = plt.cm.viridis(np.linspace(0, 1, n_interfaces))
    colors_layers = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Convective Flux Evolution (n_layers={n_layers}, timesteps={n_steps})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature vs Timestep (at interfaces) - Top Left
    ax1 = axes[0, 0]
    for i in range(n_interfaces):
        label = f'Interface {i} (z={z[i]/1000:.1f} km)' if show_legend else None
        ax1.plot(timesteps, history_T[:, i], label=label, 
                color=colors[i], linewidth=1.5)
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_ylabel('Temperature (K)', fontsize=11)
    ax1.set_title('Temperature vs Timestep (at Interfaces)', fontsize=12)
    if show_legend:
        ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: dT vs Timestep (at interfaces) - Top Right
    ax2 = axes[0, 1]
    for i in range(n_interfaces):
        label = f'Interface {i} (z={z[i]/1000:.1f} km)' if show_legend else None
        ax2.plot(timesteps, history_dT[:, i], label=label, 
                color=colors[i], linewidth=1.5)
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Temperature Change dT (K)', fontsize=11)
    ax2.set_title('Temperature Change vs Timestep (at Interfaces)', fontsize=12)
    if show_legend:
        ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Flux vs Timestep (at layer centers) - Bottom Left
    ax3 = axes[1, 0]
    for i in range(n_layers):
        label = f'Layer {i} (z={z_mid[i]/1000:.1f} km)' if show_legend else None
        ax3.plot(timesteps, history_F[:, i], label=label, 
                color=colors_layers[i], linewidth=1.5)
    ax3.set_xlabel('Timestep', fontsize=11)
    ax3.set_ylabel('Convective Flux F_conv (erg cm^-2 s^-1)', fontsize=11)
    ax3.set_title('Convective Flux vs Timestep (at Layer Centers)', fontsize=12)
    if show_legend:
        ax3.legend(fontsize=7, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for flux
    
    # Plot 4: dFlux (dF/dz) vs Timestep (at interfaces) - Bottom Right
    ax4 = axes[1, 1]
    for i in range(n_interfaces):
        label = f'Interface {i} (z={z[i]/1000:.1f} km)' if show_legend else None
        ax4.plot(timesteps, history_dF[:, i], label=label, 
                color=colors[i], linewidth=1.5)
    ax4.set_xlabel('Timestep', fontsize=11)
    ax4.set_ylabel('Flux Divergence dF/dz (erg cm^-3 s^-1)', fontsize=11)
    ax4.set_title('Flux Divergence vs Timestep (at Interfaces)', fontsize=12)
    if show_legend:
        ax4.legend(fontsize=7, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_summary.png")
    plt.close()
    
    print(f"\nPlot saved: {output_prefix}_summary.png")
    
    # Plot timescales if available
    if 'history_t_conv' in diagnostics and 'history_tau_rad' in diagnostics:
        plot_timescales(diagnostics, output_prefix)


def plot_timescales(diagnostics: dict, output_prefix: str = "convective_flux"):
    """
    Create plots of convective and radiative timescales vs timestep for each layer.
    
    Args:
        diagnostics: Dictionary containing history data from run()
        output_prefix: Prefix for output plot filenames
    """
    history_t_conv = diagnostics['history_t_conv']  # Shape: (n_steps, n_layers)
    history_tau_rad = diagnostics['history_tau_rad']  # Shape: (n_steps, n_layers)
    timesteps = diagnostics['timesteps']
    z_mid = diagnostics['z_mid']
    
    n_layers = len(z_mid)
    n_steps = len(timesteps)
    
    # Determine if we should show legend (hide if too many layers)
    show_legend = n_layers <= 10
    
    # Create color map for layers
    colors_layers = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    # Create 2x1 subplot figure for timescales
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Timescales vs Timestep (n_layers={n_layers}, timesteps={n_steps})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Convective timescale vs Timestep (at layer centers) - Top
    ax1 = axes[0]
    for i in range(n_layers):
        # Filter out infinite values for plotting
        t_conv_data = history_t_conv[:, i]
        finite_mask = np.isfinite(t_conv_data)
        
        if np.any(finite_mask):
            label = f'Layer {i} (z={z_mid[i]/1000:.1f} km)' if show_legend else None
            ax1.plot(timesteps[finite_mask], t_conv_data[finite_mask], 
                    label=label, color=colors_layers[i], linewidth=1.5)
    
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_ylabel('Convective Timescale t_conv (s)', fontsize=11)
    ax1.set_title('Convective Timescale vs Timestep (at Layer Centers)', fontsize=12)
    ax1.set_yscale('log')
    if show_legend:
        ax1.legend(fontsize=7, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Radiative timescale vs Timestep (at layer centers) - Bottom
    ax2 = axes[1]
    for i in range(n_layers):
        # Filter out infinite values for plotting
        tau_rad_data = history_tau_rad[:, i]
        finite_mask = np.isfinite(tau_rad_data)
        
        if np.any(finite_mask):
            label = f'Layer {i} (z={z_mid[i]/1000:.1f} km)' if show_legend else None
            ax2.plot(timesteps[finite_mask], tau_rad_data[finite_mask], 
                    label=label, color=colors_layers[i], linewidth=1.5)
    
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Radiative Timescale τ_rad (s)', fontsize=11)
    ax2.set_title('Radiative Timescale vs Timestep (at Layer Centers)', fontsize=12)
    ax2.set_yscale('log')
    if show_legend:
        ax2.legend(fontsize=7, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_timescales.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_timescales.png")
    plt.close()
    
    print(f"\nTimescale plot saved: {output_prefix}_timescales.png")


def plot_temperature_heatmap(z: np.ndarray, T_initial: np.ndarray, T_final: np.ndarray,
                             output_prefix: str = "convective_flux", N_ad: float = None):
    """
    Plot temperature before and after convective flux evolution as a heatmap.
    
    Args:
        z: Altitude at interfaces (n_layers+1 points) in meters
        T_initial: Initial temperature at interfaces (K)
        T_final: Final temperature at interfaces (K)
        output_prefix: Prefix for output filename
        N_ad: Adiabatic gradient (K/m) - if provided, will highlight layers within ±50% of adiabatic
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    
    # Interpolate temperature from interfaces to layer centers
    z_mid = (z[:-1] + z[1:]) / 2.0  # Layer center altitudes
    T_initial_mid = (T_initial[:-1] + T_initial[1:]) / 2.0  # Average at layer centers
    T_final_mid = (T_final[:-1] + T_final[1:]) / 2.0
    
    n_layers = len(z_mid)
    
    # Calculate which layers are within ±50% of adiabatic (if N_ad provided)
    adiabatic_mask_initial = None
    adiabatic_mask_final = None
    if N_ad is not None:
        # Calculate temperature gradients
        N_initial = temperature_gradient(T_initial, z)
        N_final = temperature_gradient(T_final, z)
        
        # Check which layers are within ±50% of adiabatic
        relative_diff_initial = np.abs(N_initial - N_ad) / N_ad
        relative_diff_final = np.abs(N_final - N_ad) / N_ad
        
        adiabatic_mask_initial = relative_diff_initial < 0.5  # Within ±50%
        adiabatic_mask_final = relative_diff_final < 0.5
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(f'Temperature Profile: Before and After Convective Flux Evolution (n_layers = {n_layers})',
                 fontsize=14, fontweight='bold')
    
    # Prepare data for heatmap: layers (y-axis) vs single column (x-axis)
    # We'll create a 2D array with 2 columns: before and after
    T_data = np.column_stack([T_initial_mid, T_final_mid])
    
    # Create meshgrid for pcolormesh
    # X: [0, 1] for before/after
    # Y: layer indices (0 to n_layers-1)
    x_edges = np.array([-0.5, 0.5, 1.5])
    y_edges = np.arange(n_layers + 1) - 0.5
    X, Y = np.meshgrid(x_edges, y_edges)
    
    # Use same color scale for both plots
    vmin = min(np.min(T_initial_mid), np.min(T_final_mid))
    vmax = max(np.max(T_initial_mid), np.max(T_final_mid))
    
    # Plot 1: Before convective flux
    im1 = ax1.pcolormesh(X[:, :2], Y[:, :2], T_initial_mid.reshape(-1, 1), 
                         cmap='plasma', shading='flat', vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Initial State', fontsize=12)
    ax1.set_ylabel('Layer Number', fontsize=12)
    ax1.set_title('Temperature Before Convective Flux', fontsize=12)
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-0.5, n_layers - 0.5])
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Initial'])
    ax1.set_yticks(np.arange(0, n_layers, max(1, n_layers//10)))
    # Low altitude (layer 0) at bottom - no invert_yaxis()
    
    # Add horizontal lines at layer interfaces
    for i in range(n_layers + 1):
        ax1.axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # Add outline/texture for layers within ±50% of adiabatic (initial state)
    if adiabatic_mask_initial is not None:
        for i in range(n_layers):
            if adiabatic_mask_initial[i]:
                # Add rectangle outline
                rect = Rectangle((-0.5, i - 0.5), 1.0, 1.0, 
                                linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.8)
                ax1.add_patch(rect)
    
    # Plot 2: After convective flux
    im2 = ax2.pcolormesh(X[:, 1:], Y[:, 1:], T_final_mid.reshape(-1, 1),
                         cmap='plasma', shading='flat', vmin=vmin, vmax=vmax)
    ax2.set_xlabel('Final State', fontsize=12)
    ax2.set_ylabel('Layer Number', fontsize=12)
    ax2.set_title('Temperature After Convective Flux', fontsize=12)
    ax2.set_xlim([0.5, 1.5])
    ax2.set_ylim([-0.5, n_layers - 0.5])
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Final'])
    ax2.set_yticks(np.arange(0, n_layers, max(1, n_layers//10)))
    # Low altitude (layer 0) at bottom - no invert_yaxis()
    
    # Add horizontal lines at layer interfaces
    for i in range(n_layers + 1):
        ax2.axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # Add outline for layers within ±50% of adiabatic (final state)
    if adiabatic_mask_final is not None:
        from matplotlib.patches import Rectangle
        for i in range(n_layers):
            if adiabatic_mask_final[i]:
                # Add rectangle outline
                rect = Rectangle((0.5, i - 0.5), 1.0, 1.0, 
                                linewidth=0.5, edgecolor='cyan', facecolor='none', alpha=1)
                ax2.add_patch(rect)
    
    # Add altitude labels on right side of second plot
    ax2_alt = ax2.twinx()
    # Map layer numbers to altitudes (in km)
    alt_km = z_mid / 1000.0
    # Show altitude at selected layer indices
    selected_layers = np.arange(0, n_layers, max(1, n_layers//10))
    ax2_alt.set_ylim(ax2.get_ylim())
    ax2_alt.set_yticks(selected_layers)
    ax2_alt.set_yticklabels([f'{alt_km[i]:.0f}' for i in selected_layers])
    ax2_alt.set_ylabel('Altitude (km)', fontsize=10, rotation=270, labelpad=15)
    
    # Add single colorbar for both plots (positioned on the far right)
    # Use fig.colorbar and position it to the right of ax2 (the rightmost plot)
    cbar = fig.colorbar(im2, ax=ax2, pad=0.75, label='Temperature (K)')
    plt.tight_layout()
    output_file = f'{output_prefix}_temperature_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Also create a combined heatmap showing both side by side
    fig2, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    # Create 2-column array: before and after
    T_combined = np.column_stack([T_initial_mid, T_final_mid])
    
    # Create meshgrid
    x_edges_combined = np.array([-0.5, 0.5, 1.5])
    y_edges_combined = np.arange(n_layers + 1) - 0.5
    X_combined, Y_combined = np.meshgrid(x_edges_combined, y_edges_combined)
    
    im_combined = ax.pcolormesh(X_combined, Y_combined, T_combined,
                                cmap='plasma', shading='flat')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Layer Number', fontsize=12)
    ax.set_title(f'Temperature Profile: n_layers = {n_layers}',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, n_layers - 0.5])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_yticks(np.arange(0, n_layers, max(1, n_layers//10)))
    # Low altitude (layer 0) at bottom - no invert_yaxis()
    
    # Add horizontal lines at layer interfaces
    for i in range(n_layers + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # Add thick vertical line between Before and After regions
    ax.axvline(x=0.5, color='black', linewidth=3, alpha=0.8)
    
    # Add outline/texture for layers within ±50% of adiabatic
    if adiabatic_mask_initial is not None and adiabatic_mask_final is not None:
        for i in range(n_layers):
            # Before state
            if adiabatic_mask_initial[i]:
                rect1 = Rectangle((-0.5, i - 0.5), 1.0, 1.0, 
                                 linewidth=0.5, edgecolor='cyan', facecolor='none', alpha=1)
                ax.add_patch(rect1)
            # After state
            if adiabatic_mask_final[i]:
                rect2 = Rectangle((0.5, i - 0.5), 1.0, 1.0, 
                                 linewidth=0.5, edgecolor='cyan', facecolor='none', alpha=1)
                ax.add_patch(rect2)
    
    # Add altitude on right side
    ax_alt = ax.twinx()
    ax_alt.set_ylim(ax.get_ylim())
    ax_alt.set_yticks(selected_layers)
    ax_alt.set_yticklabels([f'{alt_km[i]:.0f}' for i in selected_layers])
    ax_alt.set_ylabel('Altitude (km)', fontsize=10, rotation=270, labelpad=15)
    
    # Position colorbar to the right of the plot
    # First do tight_layout to get final positions
    plt.tight_layout()
    # Get the position of the main axes after tight_layout
    pos = ax.get_position()
    # Create colorbar axes to the right of the plot (x1 is right edge, add 0.02 for spacing)
    cax = fig2.add_axes([pos.x1 + 0.1, pos.y0, 0.04, pos.height])
    cbar_combined = fig2.colorbar(im_combined, cax=cax, label='Temperature (K)')
    output_file2 = f'{output_prefix}_{n_layers}layers_{ALPHA}alpha_{DT}s.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")
    plt.close()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convective Flux Solver")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS,
                       help=f"Number of layers (default: {N_LAYERS})")
    parser.add_argument("--max-z", type=float, default=MAX_Z,
                       help=f"Maximum altitude in meters (default: {MAX_Z})")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug output")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                       help=f"Maximum iteration steps (default: {MAX_STEPS})")
    parser.add_argument("--tol", type=float, default=CONVERGENCE_TOL,
                       help=f"Convergence tolerance (default: {CONVERGENCE_TOL})")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path (optional)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots of T, dT, F, and dF vs timestep")
    parser.add_argument("--plot-prefix", type=str, default="convective_flux",
                       help="Prefix for plot filenames (default: convective_flux)")
    parser.add_argument("--profile-type", type=str, choices=["linear", "guillot"], default="linear",
                       help="TP profile type: 'linear' or 'guillot' (default: linear)")
    parser.add_argument("--no-prompt", action="store_true",
                       help="Skip interactive prompts, use defaults")
    
    args = parser.parse_args()
    
    # Determine profile type
    profile_type = args.profile_type
    guillot_params = None
    
    if not args.no_prompt:
        if profile_type is None:
            print("\n" + "=" * 70)
            print("TP Profile Selection")
            print("=" * 70)
            print("Choose TP profile type:")
            print("  1. Linear (simple linear interpolation)")
            print("  2. Guillot (realistic analytical TP profile)")
            choice = input("Enter choice (1 or 2, default=1): ").strip()
            if choice == "2" or choice.lower() == "guillot":
                profile_type = "guillot"
            else:
                profile_type = "linear"
        
        # If Guillot profile, prompt for parameters
        if profile_type == "guillot":
            print("\n" + "=" * 70)
            print("Guillot TP Profile Parameters")
            print("=" * 70)
            print("Enter values (press Enter for defaults):")
            
            def prompt_float(prompt, default):
                value = input(f"{prompt} (default={default}): ").strip()
                return float(value) if value else default
            
            tint = prompt_float("Internal temperature (K)", 150.0)
            tirr = prompt_float("Irradiation temperature (K)", 1200.0)
            kappa_S = prompt_float("Shortwave opacity (cm^2/g)", 0.01)
            kappa0 = prompt_float("Infrared opacity constant (cm^2/g)", 0.02)
            kappa_cia = prompt_float("CIA opacity normalization (cm^2/g)", 0.0)
            beta_S0 = prompt_float("Shortwave scattering parameter", 1.0)
            beta_L0 = prompt_float("Longwave scattering parameter", 1.0)
            el1 = prompt_float("First longwave Eddington coefficient", 3.0/8.0)
            el3 = prompt_float("Second longwave Eddington coefficient", 1.0/3.0)
            
            guillot_params = {
                'tint': tint,
                'tirr': tirr,
                'kappa_S': kappa_S,
                'kappa0': kappa0,
                'kappa_cia': kappa_cia,
                'beta_S0': beta_S0,
                'beta_L0': beta_L0,
                'el1': el1,
                'el3': el3
            }
            
            print(f"\nGuillot parameters set:")
            print(f"  tint = {tint} K")
            print(f"  tirr = {tirr} K")
            print(f"  kappa_S = {kappa_S} cm^2/g")
            print(f"  kappa0 = {kappa0} cm^2/g")
            print(f"  kappa_cia = {kappa_cia} cm^2/g")
            print(f"  beta_S0 = {beta_S0}")
            print(f"  beta_L0 = {beta_L0}")
            print(f"  el1 = {el1}")
            print(f"  el3 = {el3}")
    else:
        # Use defaults if no prompt
        if profile_type is None:
            profile_type = "linear"
        if profile_type == "guillot" and guillot_params is None:
            # Use default Guillot parameters
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
    
    # Run solver
    z, T, rho, P, diagnostics = run(
        n_layers=args.n_layers,
        max_z=args.max_z,
        debug=args.debug,
        max_steps=args.max_steps,
        convergence_tol=args.tol,
        save_history=args.plot,
        profile_type=profile_type,
        guillot_params=guillot_params
    )
    
    # Output CSV if requested
    if args.output:
        # Calculate final flux for output
        c_p = calculate_c_p(N_DOF, MMW)
        N_ad = adiabatic_gradient(G, c_p)
        z_mid = (z[:-1] + z[1:]) / 2.0
        T_mid = (T[:-1] + T[1:]) / 2.0
        rho_mid = (rho[:-1] + rho[1:]) / 2.0
        N = temperature_gradient(T, z)
        F_conv = convective_flux(rho_mid, c_p, ALPHA, G, T_mid, N, N_ad, MMW)
        
        # Write CSV
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['z_interface_m', 'T_interface_K', 'rho_interface_g_cm3',
                           'z_center_m', 'F_conv_center_erg_cm2_s'])
            
            # Write interface data
            for i in range(len(z)):
                if i < len(z_mid):
                    writer.writerow([z[i], T[i], rho[i], z_mid[i], F_conv[i]])
                else:
                    writer.writerow([z[i], T[i], rho[i], '', ''])
        
        print(f"Results written to {args.output}")
    
    # Generate plots if requested
    if args.plot:
        print("\n" + "=" * 70)
        print("Generating plots...")
        print("=" * 70)
        plot_results(diagnostics, args.plot_prefix)
        # Also plot temperature heatmap (pass N_ad for adiabatic highlighting)
        plot_temperature_heatmap(diagnostics['z'], diagnostics['T_initial'], T, 
                                args.plot_prefix, N_ad=diagnostics['N_ad'])
