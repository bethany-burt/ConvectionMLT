"""
Convective Flux Solver v3 - Iterative 1D Atmospheric Column Model with Dynamic Timestepping

This script iteratively calculates convective flux for a model grid of atmospheric
layers, accounting for the effect of changing temperature on convective flux.

Based on mixing length theory:
    F_conv = ρ * c_p * l^2 * sqrt(g/T) * (N - N_ad)^(3/2)

Where:
    l = α × H_p (mixing length)
    H_p = RT/(μ·g) (pressure scale height)
    N = -dT/dz (temperature gradient)
    N_ad = g/c_p (adiabatic temperature gradient)

KEY FEATURES IN V3:

1. PRESSURE-BASED GRID:
    - Grid is constructed from log-spaced pressure interfaces (P_top to P_bottom)
    - Altitude (z) and density (ρ) are derived from hydrostatic equilibrium
    - Ensures consistent P-z-ρ-T relationships throughout the atmosphere

2. SUPER-ADIABATIC INITIALIZATION:
    - Option to initialize with dry adiabat: T(P) = T0 * (P/P0)^((γ-1)/γ)
    - Perturbation added to make it super-adiabatic: T(P) = T_ad(P) * (1 + ε)
    - Ensures all layers start with N > N_ad (unstable, drives convection)
    - Default: T0 = 2000K, P0 = 1 bar, γ = 1.4 (H2), ε = 0.05 (5%)

3. DYNAMIC TIMESTEPPING (OPTIONAL):
    - To enable: use --dynamic-dt flag or pass dt=None to run()
    - Each layer calculates its own timestep based on convective timescale
    - Default method 'formal': dt = DT_CONST * [g/T * |N - N_ad|]^{-1/2} (continuous across RCB)
    
    Other methods for handling N <= N_ad layers (radiative/adiabatic):
    - 'gradient': dt ∝ 1/|N - N_ad| (allows fine convergence)
    - 'fixed': constant dt for radiative layers
    - 'absolute': use |N - N_ad| in formula
    - 'hybrid': convective timescale for N > N_ad, convergence timescale for N <= N_ad
    - 'minimum': use minimum dt floor for all layers
    - 'radiative': use radiative timescale (tau_rad) for radiative layers

4. ADIABATIC DAMPING:
    - Reduces timestep as layers approach adiabat to prevent overshooting
    - Two methods available:
      - 'current': Temperature + proximity-based damping (f_T * f_N)
      - 'restoring_force': Physics-based damping scaling with restoring force |N - N_ad|
    - Helps maintain stability near the radiative-convective boundary

5. ADIABATIC CONVERGENCE CRITERION:
    - Primary convergence: All convective layers within tolerance of adiabatic (N ≈ N_ad)
    - Secondary convergence: Maximum temperature change < tolerance
    - Default: 5% tolerance for adiabaticity check

By default, uses fixed timestepping (same as v2) for backward compatibility.

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
T_TOA = 2000.0   # Top of atmosphere
T_BOA = 800.0  # Bottom of atmosphere

# Density boundaries (g/cm^3) - will convert to g/m^3 for consistency
RHO_TOA = 0.1    # g/cm^3
RHO_BOA = 1000.0 # g/cm^3

# Physical parameters
G = 15.0         # Gravity (m/s^2)
ALPHA = 0.1      # Mixing length parameter (dimensionless, α in l = α × H_p)
# NOTE: Larger values (e.g., 0.5) cause stronger flux and overshooting of the adiabat.
# Keep α small (0.1) to allow gradual approach to adiabat without overshooting.
DT = 1       # Timestep (s)
MAX_Z = 5_000_000  # Maximum altitude (m)

# Grid parameters
N_LAYERS = 100    # Number of layers (will have n_layers+1 interfaces)

# Composition parameters (H2 dominated)
N_DOF = 5        # Degrees of freedom for H2 (3 translational + 2 rotational)
MMW = 2.016      # Mean molecular weight (g/mol) for H2

# Solver parameters
MAX_STEPS = 500000      # Maximum iteration steps
CONVERGENCE_TOL = 1e-5  # Convergence tolerance for max|dT| (K)
DEBUG_INTERVAL = 10     # Print debug info every N steps

# Dynamic timestepping parameters (v3) - only used when dt=None
DT_MIN = 1e-6           # Minimum timestep (s) - prevents numerical issues
DT_MAX = 1e6            # Maximum timestep (s) - allows large convective timescales near adiabat (was 1e4, too restrictive)
DT_MAX_RADIATIVE = 100.0  # Maximum dt for radiative layers (s) - prevents overshooting when close to adiabatic
DT_MAX_CHANGE_FRAC = 0.1  # Max fractional T change per step (0.1 = 10%) for stability-limited dt
DT_RADIATIVE_DEFAULT = 10.0      # Default fixed dt for radiative layers (s)
DT_CONVERGENCE_DEFAULT = 0.1     # Default convergence constant (s·K/m) for gradient method
DT_CONST = 1.0                   # Constant for timestep calculation (dimensionless scaling factor)


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


def setup_hydrostatic_grid(n_layers: int,
                           g: float = G,
                           mmw: float = MMW,
                           P_top_bar: float = 1e-6,
                           P_bottom_bar: float = 1e3,
                           T_profile_func=None,
                           T_profile_args=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a hydrostatic P-z grid with consistent P-z-rho-T relationships.
    
    Primary independent variable is pressure at interfaces (in bar), log-spaced from
    P_bottom_bar to P_top_bar. We then:
        - Compute column mass m = P / g (cgs),
        - Use T_profile_func to get T(P) (or isothermal if None),
        - Use ideal gas EOS to get rho(P, T),
        - Integrate hydrostatic equilibrium to get z(P),
        - Derive layer-centre altitudes, etc.
    
    Args:
        n_layers: Number of layers (will have n_layers+1 interfaces).
        g: Gravity (m/s^2).
        mmw: Mean molecular weight (g/mol).
        P_top_bar: Target top pressure (bar).
        P_bottom_bar: Target bottom pressure (bar).
        T_profile_func: Function that takes (m, m0, ...) and returns T(K). 
                        If None, uses isothermal profile with T from T_profile_args['T_iso'].
        T_profile_args: Dict of arguments for T_profile_func, or {'T_iso': T} for isothermal.
    
    Returns:
        z: Altitude at interfaces (m), shape (n_layers+1,)
        z_mid: Altitude at layer centres (m), shape (n_layers,)
        T: Temperature at interfaces (K), shape (n_layers+1,)
        rho: Density at interfaces (g/cm^3), shape (n_layers+1,)
        P_cgs: Pressure at interfaces (dyne/cm^2), shape (n_layers+1,)
    """
    n_interfaces = n_layers + 1
    
    # Log-pressure grid (interfaces), from high P (bottom) to low P (top)
    logP_bottom = np.log10(P_bottom_bar)
    logP_top = np.log10(P_top_bar)
    logP = np.linspace(logP_bottom, logP_top, n_interfaces)
    P_bar = 10.0 ** logP  # bar
    
    # Convert to cgs pressure (dyne/cm^2)
    bar_to_cgs = 1e6  # 1 bar = 1e6 dyne/cm^2
    P_cgs = P_bar * bar_to_cgs
    
    # Column mass m = P / g in cgs
    g_cgs = g * 100.0  # m/s^2 -> cm/s^2
    m = P_cgs / g_cgs  # g/cm^2
    m0 = m[0]          # bottom column mass
    
    # Temperature at interfaces
    T = np.zeros_like(P_bar)
    if T_profile_func is None:
        # Check if semi-isothermal (superadiabatic gradient) or fully isothermal
        if T_profile_args and T_profile_args.get('semi_iso', False):
            # Semi-isothermal: Start with dry adiabat, then add perturbation to make it super-adiabatic
            # Use pressure-based adiabatic profile: T(P) = T0 * (P/P0)^((γ-1)/γ)
            
            # Get parameters
            T0 = T_profile_args.get('T0', 2000.0)  # Reference temperature (K)
            P0 = T_profile_args.get('P0', 1.0)  # Reference pressure (bar)
            perturbation_factor = T_profile_args.get('epsilon', 0.05)  # Perturbation factor (0.05 = 5%)
            
            # Calculate γ = cp/cv for H2
            # For diatomic H2: N_DOF = 5 (3 translational + 2 rotational)
            # cv = (5/2)R, cp = cv + R = (7/2)R, so γ = cp/cv = 7/5 = 1.4
            # But we can calculate it from N_DOF:
            # cv = (N_DOF/2) * R, cp = cv + R = (N_DOF/2 + 1) * R
            # γ = cp/cv = (N_DOF/2 + 1) / (N_DOF/2) = (N_DOF + 2) / N_DOF
            # For N_DOF = 5: γ = 7/5 = 1.4
            n_dof = T_profile_args.get('n_dof', 5)  # Degrees of freedom for H2
            gamma = (n_dof + 2.0) / n_dof  # γ = cp/cv
            
            # Calculate adiabatic exponent: (γ-1)/γ
            adiabatic_exponent = (gamma - 1.0) / gamma
            
            # Calculate adiabatic temperature profile: T_ad(P) = T0 * (P/P0)^((γ-1)/γ)
            T_ad = T0 * (P_bar / P0) ** adiabatic_exponent
            
            # Add perturbation to make it super-adiabatic: T(P) = T_ad(P) * (1 + ε)
            T = T_ad * (1.0 + perturbation_factor)
            
            # Note: rho and z will be calculated below from T using ideal gas EOS and hydrostatic equilibrium
        else:
            # Fully isothermal profile
            T_iso = T_profile_args.get('T_iso', 1000.0) if T_profile_args else 1000.0
            T[:] = T_iso
    else:
        # Use provided temperature profile function
        # Extract parameters for guillot_tp_profile
        tint = T_profile_args['tint']
        tirr = T_profile_args['tirr']
        kappa_S = T_profile_args['kappa_S']
        kappa0 = T_profile_args['kappa0']
        kappa_cia = T_profile_args.get('kappa_cia', 0.0)
        beta_S0 = T_profile_args.get('beta_S0', 1.0)
        beta_L0 = T_profile_args.get('beta_L0', 1.0)
        el1 = T_profile_args.get('el1', 3.0/8.0)
        el3 = T_profile_args.get('el3', 1.0/3.0)
        
        for i in range(n_interfaces):
            T[i] = T_profile_func(m[i], m0, tint, tirr, kappa_S, kappa0, kappa_cia,
                                 beta_S0, beta_L0, el1, el3)
    
    # Ideal gas EOS to get density at interfaces: P = rho * R_specific * T
    # R_specific = R / mmw, R in erg mol^-1 K^-1, mmw in g/mol
    R_specific = R / mmw  # erg g^-1 K^-1
    # P_cgs in dyne/cm^2 = erg/cm^3, so rho in g/cm^3:
    rho = P_cgs / (R_specific * T)
    
    # Integrate hydrostatic equilibrium to get altitude z at interfaces
    z = np.zeros(n_interfaces)
    z[0] = 0.0  # bottom at z=0
    
    for i in range(1, n_interfaces):
        dP = P_cgs[i-1] - P_cgs[i]  # positive going upward
        rho_avg = 0.5 * (rho[i-1] + rho[i])
        # dz (cm) = dP / (rho * g_cgs)
        dz_cm = dP / (rho_avg * g_cgs)
        z[i] = z[i-1] + dz_cm / 100.0  # convert to meters
    
    # Layer centres
    z_mid = 0.5 * (z[:-1] + z[1:])
    
    return z, z_mid, T, rho, P_cgs


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
    
    # Print all layers with their type (convective or radiative)
    print("ALL LAYERS:")
    print("-" * 85)
    print(f"{'Layer':<8} {'Type':<12} {'Altitude (km)':<15} {'N (K/m)':<15} {'N/N_ad':<12} {'|N-N_ad|/N_ad':<15}")
    print("-" * 85)
    
    for idx in range(n_total):
        z_km = z_mid[idx] / 1000.0
        N_val = N[idx]
        N_ratio = N_val / N_ad
        relative_diff = np.abs(N_val - N_ad) / N_ad
        
        # Determine layer type
        if convective_mask[idx]:
            layer_type = "Convective"
        else:
            layer_type = "Radiative"
        
        print(f"{idx:<8} {layer_type:<12} {z_km:<15.2f} {N_val:<15.6f} {N_ratio:<12.4f} {relative_diff:<15.4f}")
    
    print()
    
    # Summary statistics for convective layers
    if n_convective > 0:
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
    
    # Summary statistics for radiative layers
    if n_radiative > 0:
        radiative_N = N[radiative_mask]
        print(f"\nRadiative layer statistics:")
        print(f"  Min N: {np.min(radiative_N):.6f} K/m ({np.min(radiative_N)/N_ad:.4f} × N_ad)")
        print(f"  Max N: {np.max(radiative_N):.6f} K/m ({np.max(radiative_N)/N_ad:.4f} × N_ad)")
        print(f"  Mean N: {np.mean(radiative_N):.6f} K/m ({np.mean(radiative_N)/N_ad:.4f} × N_ad)")
        relative_diffs = np.abs(radiative_N - N_ad) / N_ad
        print(f"  Mean |N-N_ad|/N_ad: {np.mean(relative_diffs):.4f}")
        print(f"  Max |N-N_ad|/N_ad: {np.max(relative_diffs):.4f}")
    
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
    t_conv_raw = 1.0 / np.sqrt(term)  # Raw timescale before DT_CONST multiplication
    t_conv = DT_CONST * t_conv_raw
    
    # #region agent log: investigate dt calculation (H1-H4)
    import json
    convective_mask = N > N_ad
    if np.any(convective_mask):
        conv_term = term[convective_mask]
        conv_t_conv_raw = t_conv_raw[convective_mask]
        conv_t_conv = t_conv[convective_mask]
        conv_T = T[convective_mask]
        conv_delta_N = delta_N[convective_mask]
        log_data = {
            "location": "convective_flux_v3.py:1016",
            "message": "convective_timescale calculation details",
            "data": {
                "DT_CONST": float(DT_CONST),
                "g": float(g),
                "N_ad": float(N_ad),
                "convective_layers_count": int(np.sum(convective_mask)),
                "term_range": [float(np.min(conv_term)), float(np.max(conv_term))],
                "term_mean": float(np.mean(conv_term)),
                "t_conv_raw_range": [float(np.min(conv_t_conv_raw)), float(np.max(conv_t_conv_raw))],
                "t_conv_raw_mean": float(np.mean(conv_t_conv_raw)),
                "t_conv_after_DT_CONST_range": [float(np.min(conv_t_conv)), float(np.max(conv_t_conv))],
                "t_conv_after_DT_CONST_mean": float(np.mean(conv_t_conv)),
                "T_range": [float(np.min(conv_T)), float(np.max(conv_T))],
                "delta_N_range": [float(np.min(conv_delta_N)), float(np.max(conv_delta_N))],
                "sample_layer_0": {
                    "T": float(conv_T[0]),
                    "delta_N": float(conv_delta_N[0]),
                    "term": float(conv_term[0]),
                    "t_conv_raw": float(conv_t_conv_raw[0]),
                    "t_conv_final": float(conv_t_conv[0])
                }
            },
            "timestamp": int(__import__('time').time() * 1000),
            "runId": "investigate_dt_size",
            "hypothesisId": "H1,H2"
        }
        with open('/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/.cursor/debug.log', 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    # #endregion
    
    # For non-convective layers (where N <= N_ad), set timescale to infinity
    # or a very large value to indicate no convection
    non_convective = N <= N_ad
    t_conv[non_convective] = np.inf
    
    return t_conv


def apply_dt_method(dt_layer: np.ndarray, N: np.ndarray, N_ad: float, g: float,
                    T: np.ndarray, dt_method: str = 'convective',
                    dt_radiative: Optional[float] = None,
                    dt_convergence: Optional[float] = None,
                    tau_rad: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply method-specific handling for layers where N <= N_ad (radiative/adiabatic).
    
    The convective_timescale() function sets these layers to np.inf. This function
    replaces those values with finite timesteps based on the selected method.
    
    Args:
        dt_layer: Timestep at layer centers (contains np.inf for N <= N_ad)
        N: Temperature gradient at layer centers (K/m)
        N_ad: Adiabatic temperature gradient (K/m)
        g: Gravity (m/s²)
        T: Temperature at layer centers (K)
        dt_method: Method to use ('gradient', 'fixed', 'absolute', 'hybrid', 'minimum', 'convective', 'formal', 'radiative')
        dt_radiative: Fixed dt for radiative layers (used by 'fixed' method, default: DT_RADIATIVE_DEFAULT)
        dt_convergence: Convergence constant for gradient method (default: DT_CONVERGENCE_DEFAULT)
        tau_rad: Radiative timescale at layer centers (s), required for 'radiative' method
    
    Returns:
        dt_layer: Updated timestep array with finite values for all layers
    """
    # Set defaults
    if dt_radiative is None:
        dt_radiative = DT_RADIATIVE_DEFAULT
    if dt_convergence is None:
        dt_convergence = DT_CONVERGENCE_DEFAULT
    
    # Identify non-convective layers (N <= N_ad)
    non_conv = N <= N_ad
    conv = ~non_conv
    
    # Small minimum value to prevent division by zero
    EPSILON_MIN = 1e-10
    
    if dt_method == 'gradient':
        # dt = DT_CONVERGENCE / |N - N_ad|
        # Smaller gap from adiabatic → smaller dt → allows fine convergence
        # BUT: cap at DT_MAX_RADIATIVE to prevent overshooting when very close to adiabatic
        delta_N_abs = np.abs(N[non_conv] - N_ad)
        delta_N_abs = np.maximum(delta_N_abs, EPSILON_MIN)  # Prevent division by zero
        dt_layer[non_conv] = dt_convergence / delta_N_abs
        # Cap radiative layer dt to prevent overshooting when very close to adiabatic
        dt_layer[non_conv] = np.minimum(dt_layer[non_conv], DT_MAX_RADIATIVE)
        
    elif dt_method == 'fixed':
        # Use constant moderate value for all N <= N_ad layers
        dt_layer[non_conv] = dt_radiative
        
    elif dt_method == 'absolute':
        # Use dt = [g/T * |N - N_ad|]^{-1/2} even when N <= N_ad
        delta_N_abs = np.abs(N[non_conv] - N_ad)
        delta_N_abs = np.maximum(delta_N_abs, EPSILON_MIN)
        term = (g / T[non_conv]) * delta_N_abs
        dt_layer[non_conv] = 1.0 / np.sqrt(term)
        
    elif dt_method == 'hybrid':
        # Convective timescale for N > N_ad (already calculated)
        # Convergence timescale for N <= N_ad
        delta_N_abs = np.abs(N[non_conv] - N_ad)
        delta_N_abs = np.maximum(delta_N_abs, EPSILON_MIN)
        dt_layer[non_conv] = dt_convergence / delta_N_abs
        # Cap radiative layer dt to prevent overshooting when very close to adiabatic
        dt_layer[non_conv] = np.minimum(dt_layer[non_conv], DT_MAX_RADIATIVE)
        
    elif dt_method == 'minimum':
        # Use minimum dt floor for all layers (convective and radiative)
        dt_layer = np.maximum(dt_layer, DT_MIN)
        
    elif dt_method == 'convective':
        # Use convective timescale formula for radiative layers:
        # dt = DT_CONST * [g/T * |N - N_ad|]^{-1/2}
        # This uses the same formula as convective_timescale() but with |N - N_ad|
        # to handle radiative layers (N <= N_ad)
        delta_N_abs = np.abs(N[non_conv] - N_ad)
        delta_N_abs = np.maximum(delta_N_abs, EPSILON_MIN)
        term = (g / T[non_conv]) * delta_N_abs
        dt_layer[non_conv] = DT_CONST * (1.0 / np.sqrt(term))
        # Convective layers already have dt from convective_timescale(), keep them
        
    elif dt_method == 'formal':
        # Use the "formal" timescale everywhere:
        # dt_formal = DT_CONST * [g/T * |N - N_ad|]^{-1/2}
        # This is finite and continuous across the RCB and applies to both
        # convective (N > N_ad) and radiative (N <= N_ad) layers.
        delta_N_abs_all = np.abs(N - N_ad)
        delta_N_abs_all = np.maximum(delta_N_abs_all, EPSILON_MIN)
        term_all = (g / T) * delta_N_abs_all
        dt_layer = DT_CONST * (1.0 / np.sqrt(term_all))
        
    elif dt_method == 'radiative':
        # Use radiative timescale for radiative layers (N <= N_ad)
        # This is physically appropriate since radiative layers evolve on radiative timescales
        if tau_rad is None:
            raise ValueError(
                "tau_rad must be provided when dt_method='radiative'. "
                "Calculate tau_rad using radiative_timescale() before calling apply_dt_method()."
            )
        # For radiative layers, use the radiative timescale
        dt_layer[non_conv] = tau_rad[non_conv]
        # Convective layers already have dt from convective_timescale(), keep them
        
    else:
        raise ValueError(
            f"Unknown dt_method: {dt_method}. Must be one of: "
            f"'gradient', 'fixed', 'absolute', 'hybrid', 'minimum', 'convective', 'formal', 'radiative'"
        )
    
    # Apply global safety checks
    # Replace any remaining inf/nan with fallback
    dt_layer = np.where(np.isfinite(dt_layer), dt_layer, dt_radiative)
    
    # Clip to reasonable range
    dt_layer = np.clip(dt_layer, DT_MIN, DT_MAX)
    
    return dt_layer


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
        g: float = G, alpha: float = ALPHA, dt: Optional[float] = DT,
        max_steps: int = MAX_STEPS, convergence_tol: float = CONVERGENCE_TOL,
        debug: bool = False, debug_interval: int = DEBUG_INTERVAL,
        n_dof: int = N_DOF, mmw: float = MMW,
        save_history: bool = False,
        history_interval: Optional[int] = None,
        profile_type: str = "guillot",
        guillot_params: Optional[dict] = None,
        check_adiabatic: bool = True,  # Default to True - adiabaticity is the PRIMARY convergence criterion
        adiabatic_tolerance: float = 0.05,  # Default to 5% tolerance: require |N - N_ad|/N_ad < 0.05 for convective layers
        use_energy_conservation: bool = True,
        use_constant_dt_coefficient: bool = False,
        dt_constant_value: Optional[float] = 1,
        track_layer: Optional[int] = None,
        track_steps: int = 20,
        # Dynamic timestepping parameters (v3) - only used when dt=None
        dt_method: str = 'formal',
        dt_radiative: Optional[float] = None,
        dt_convergence: Optional[float] = None,
        # Debug parameters
        stop_on_first_radiative: bool = False,
        # Damping parameters
        damping_method: str = 'current') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
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
        convergence_tol: Convergence tolerance for max|dT| (K) - SECONDARY criterion
        check_adiabatic: If True (default), use adiabaticity (N ≈ N_ad) as PRIMARY convergence criterion
        adiabatic_tolerance: Fractional tolerance for adiabaticity check (default: 0.05 = 5%)
                           For convective layers (N > N_ad), require |N - N_ad|/N_ad < tolerance
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
    
    # Calculate R_specific (needed for pressure calculation)
    # R_specific = R / mmw, where R is in erg mol^-1 K^-1, mmw is in g/mol
    R_specific = R / mmw  # erg g^-1 K^-1 (specific gas constant)
    
    # Initialize profiles - use hydrostatic grid for Guillot, isothermal, and semi-isothermal profiles
    if profile_type in ["guillot", "isothermal", "semi-isothermal"]:
        # Use hydrostatic grid setup (v4-style) for consistent P-z-rho-T
        P_top_bar = 1e-6
        P_bottom_bar = 1e3
        
        if profile_type == "guillot":
            if guillot_params is None:
                raise ValueError("guillot_params must be provided when profile_type='guillot'")
            # Use Guillot TP profile function
            z, z_mid, T, rho, P_cgs = setup_hydrostatic_grid(
                n_layers=n_layers,
                g=g,
                mmw=mmw,
                P_top_bar=P_top_bar,
                P_bottom_bar=P_bottom_bar,
                T_profile_func=guillot_tp_profile,
                T_profile_args=guillot_params
            )
        elif profile_type == "isothermal":
            # Use fully isothermal temperature profile (N = 0, stable, no convection)
            T_iso = T_boa  # Use bottom temperature as isothermal value
            z, z_mid, T, rho, P_cgs = setup_hydrostatic_grid(
                n_layers=n_layers,
                g=g,
                mmw=mmw,
                P_top_bar=P_top_bar,
                P_bottom_bar=P_bottom_bar,
                T_profile_func=None,
                T_profile_args={'T_iso': T_iso, 'semi_iso': False}
            )
        else:  # semi-isothermal
            # Use pressure-based dry adiabat with perturbation to make it super-adiabatic
            # T(P) = T0 * (P/P0)^((γ-1)/γ) * (1 + ε)
            # where T0 = 2000K, P0 = 1 bar, γ = 1.4 (for H2), ε = 0.05
            
            T0 = 2000.0  # Reference temperature (K)
            P0 = 1.0     # Reference pressure (bar)
            perturbation_factor = 0.05  # Perturbation factor (5% to make it super-adiabatic)
            
            z, z_mid, T, rho, P_cgs = setup_hydrostatic_grid(
                n_layers=n_layers,
                g=g,
                mmw=mmw,
                P_top_bar=P_top_bar,
                P_bottom_bar=P_bottom_bar,
                T_profile_func=None,
                T_profile_args={
                    'semi_iso': True,
                    'T0': T0,
                    'P0': P0,
                    'epsilon': perturbation_factor,  # Keep 'epsilon' key for backward compatibility
                    'n_dof': n_dof  # Pass N_DOF to calculate γ
                }
            )
            
            # Calculate and report γ
            gamma = (n_dof + 2.0) / n_dof
            adiabatic_exponent = (gamma - 1.0) / gamma
            print(f"Dry adiabat initialization:")
            print(f"  T0 = {T0:.1f} K, P0 = {P0:.1f} bar")
            print(f"  γ = cp/cv = {gamma:.3f} (for H2 with {n_dof} DOF)")
            print(f"  Adiabatic exponent: (γ-1)/γ = {adiabatic_exponent:.4f}")
            print(f"  Perturbation: ε = {perturbation_factor:.2%} (T = T_ad × (1 + ε))")
            print(f"  T range: [{np.min(T):.1f}, {np.max(T):.1f}] K")
        
        P = P_cgs  # Pressure already in dyne/cm^2
        avg_dz = (z[-1] - z[0]) / n_layers
        print(f"Grid (hydrostatic): {n_layers} layers, ⟨dz⟩ = {avg_dz/1000:.2f} km")
        print(f"  Altitude range: 0 to {z[-1]/1000:.1f} km")
        print(f"  Pressure range: {P_top_bar:.2e} to {P_bottom_bar:.2e} bar")
    else:
        # Use old grid setup for linear profile (backward compatibility)
        z, z_mid, dz = setup_grid(n_layers, max_z)
        print(f"Grid: {n_layers} layers, dz = {dz/1000:.2f} km")
        print(f"  Altitude range: 0 to {max_z/1000:.1f} km")
        
        # Initialize profiles
        T, rho = initialize_profiles(z, z_mid, T_toa, T_boa, rho_toa, rho_boa,
                                      profile_type=profile_type, guillot_params=guillot_params, g=g)
        
        # Calculate initial pressure at interfaces from ideal gas law: P = ρ * R_specific * T
        # rho in g/cm^3, T in K, R_specific in erg g^-1 K^-1
        # so P = (g/cm^3) * (erg g^-1 K^-1) * K = erg/cm^3 = dyne/cm^2
        P = rho * R_specific * T  # Pressure in dyne/cm^2 at interfaces
    
    T_initial = T.copy()  # Save initial temperature for plotting
    
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
    
    # Check initial superadiabaticity before starting iteration
    print("=" * 70)
    print("Checking initial superadiabaticity...")
    print("=" * 70)
    
    # Calculate initial temperature gradient
    N_initial = temperature_gradient(T, z)
    
    # Find sub-adiabatic layers (N <= N_ad)
    subadiabatic_mask = N_initial <= N_ad
    n_subadiabatic = np.sum(subadiabatic_mask)
    subadiabatic_indices = np.where(subadiabatic_mask)[0]
    
    print(f"Adiabatic gradient: N_ad = {N_ad:.6f} K/m")
    print(f"Total layers: {n_layers}")
    print(f"Super-adiabatic layers (N > N_ad): {n_layers - n_subadiabatic}")
    print(f"Sub-adiabatic layers (N <= N_ad): {n_subadiabatic}")
    
    if n_subadiabatic > 0:
        print(f"\nWARNING: {n_subadiabatic} layer(s) are sub-adiabatic:")
        print("Layer    Altitude (km)   N (K/m)         N/N_ad       Status")
        print("-" * 70)
        for idx in subadiabatic_indices[:20]:  # Show first 20
            alt_km = z_mid[idx] / 1000.0 if idx < len(z_mid) else z[idx] / 1000.0
            N_val = N_initial[idx]
            ratio = N_val / N_ad if N_ad > 0 else np.inf
            print(f"{idx:4d}     {alt_km:10.2f}   {N_val:12.6e}   {ratio:10.6f}   SUB-ADIABATIC")
        if len(subadiabatic_indices) > 20:
            print(f"... and {len(subadiabatic_indices) - 20} more sub-adiabatic layers")
        
        # Show statistics
        N_subadiabatic = N_initial[subadiabatic_mask]
        print(f"\nSub-adiabatic layer statistics:")
        print(f"  Min N: {np.min(N_subadiabatic):.6e} K/m ({np.min(N_subadiabatic)/N_ad:.6f} × N_ad)")
        print(f"  Max N: {np.max(N_subadiabatic):.6e} K/m ({np.max(N_subadiabatic)/N_ad:.6f} × N_ad)")
        print(f"  Mean N: {np.mean(N_subadiabatic):.6e} K/m ({np.mean(N_subadiabatic)/N_ad:.6f} × N_ad)")
        
        print(f"\nAll sub-adiabatic layer indices: {list(subadiabatic_indices)}")
    else:
        print("✓ All layers are super-adiabatic (N > N_ad)")
    
    print()
    
    # Main iteration loop
    print("=" * 70)
    print("Starting iteration loop...")
    print("=" * 70)
    
    # History tracking for debug (only if stop_on_first_radiative is enabled)
    # Structure: history[step_idx][layer_idx] = {T, P, F, t_conv, dT, dF_dz, N, N_ad}
    debug_history = []  # List of dicts, one per step, containing arrays for each metric
    MAX_HISTORY_STEPS = 5 if stop_on_first_radiative else 0
    
    for step in range(max_steps):
        # Interpolate T and rho to layer centers for flux calculation
        T_mid = (T[:-1] + T[1:]) / 2.0
        rho_mid = (rho[:-1] + rho[1:]) / 2.0
        
        # Calculate temperature gradient at layer centers
        N = temperature_gradient(T, z)
        
        # Store previous N for transition detection (before temperature update)
        N_prev = N.copy()
        
        # Calculate convective flux at layer centers
        # This will also verify alpha usage if debug is enabled
        F_conv = convective_flux(rho_mid, c_p, alpha, g, T_mid, N, N_ad, mmw)
        
        # #region agent log: investigate dF/dz (H1-H5)
        if step == 0:
            import json
            log_data = {
                "location": f"convective_flux_v3.py:{1328}",
                "message": "F_conv and N values at step 0",
                "data": {
                    "step": step,
                    "F_conv_sample": {
                        "layer_0": float(F_conv[0]),
                        "layer_10": float(F_conv[10]),
                        "layer_25": float(F_conv[25]),
                        "layer_50": float(F_conv[50]),
                        "layer_75": float(F_conv[75]),
                        "layer_90": float(F_conv[90]),
                        "F_conv_range": [float(np.min(F_conv)), float(np.max(F_conv))],
                        "F_conv_mean": float(np.mean(F_conv)),
                        "F_conv_std": float(np.std(F_conv)),
                        "F_conv_relative_variation": float(np.std(F_conv) / np.mean(F_conv)) if np.mean(F_conv) > 0 else 0.0
                    },
                    "N_sample": {
                        "layer_0": float(N[0]),
                        "layer_10": float(N[10]),
                        "layer_25": float(N[25]),
                        "layer_50": float(N[50]),
                        "layer_75": float(N[75]),
                        "layer_90": float(N[90]),
                        "N_range": [float(np.min(N)), float(np.max(N))],
                        "N_mean": float(np.mean(N)),
                        "N_std": float(np.std(N)),
                        "N_ad": float(N_ad),
                        "N_relative_variation": float(np.std(N) / np.mean(N)) if np.mean(N) > 0 else 0.0
                    },
                    "delta_N_sample": {
                        "layer_0": float(N[0] - N_ad),
                        "layer_10": float(N[10] - N_ad),
                        "layer_25": float(N[25] - N_ad),
                        "layer_50": float(N[50] - N_ad),
                        "layer_75": float(N[75] - N_ad),
                        "layer_90": float(N[90] - N_ad),
                        "delta_N_range": [float(np.min(N - N_ad)), float(np.max(N - N_ad))],
                        "delta_N_mean": float(np.mean(N - N_ad)),
                        "delta_N_std": float(np.std(N - N_ad))
                    }
                },
                "timestamp": int(__import__('time').time() * 1000),
                "runId": "investigate_dFdz",
                "hypothesisId": "H1"
            }
            with open('/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        # #endregion
        
        # Calculate timescales at layer centers
        # Need pressure at layer centers for radiative timescale
        P_mid = (P[:-1] + P[1:]) / 2.0  # Pressure at layer centers (dyne/cm²)
        t_conv = convective_timescale(g, T_mid, N, N_ad)
        tau_rad = radiative_timescale(P_mid, g, c_p, T_mid)
        
        # Determine if using dynamic timestepping (v3)
        use_dynamic_dt = (dt is None)
        
        # Calculate dynamic timestep if enabled
        if use_dynamic_dt:
            # Start with convective timescale (contains np.inf for N <= N_ad layers)
            dt_layer = t_conv.copy()
            
            # Debug: Track initial state
            if debug and (step == 0 or step % debug_interval == 0):
                n_conv = np.sum(N > N_ad)
                n_rad = np.sum(N <= N_ad)
                n_inf = np.sum(np.isinf(dt_layer))
                print(f"\nDEBUG [Step {step}] DYNAMIC DT CALCULATION:")
                print(f"  Convective layers (N > N_ad): {n_conv} / {len(N)}")
                print(f"  Radiative layers (N <= N_ad): {n_rad} / {len(N)}")
                print(f"  np.inf in t_conv: {n_inf} layers")
                if n_rad > 0:
                    rad_mask = N <= N_ad
                    rad_indices = np.where(rad_mask)[0]
                    print(f"  Radiative layer indices: {rad_indices[:10]}{'...' if len(rad_indices) > 10 else ''}")
            
            # Apply method-specific handling for N <= N_ad layers (replaces np.inf with finite values)
            dt_layer_before_method = dt_layer.copy()
            dt_layer = apply_dt_method(dt_layer, N, N_ad, g, T_mid, dt_method,
                                     dt_radiative, dt_convergence, tau_rad=tau_rad)
            
            # Apply damping to dt as layers approach adiabat
            # Two methods available:
            #   1. 'current': Temperature + proximity-based damping (more stringent version)
            #   2. 'restoring_force': Physics-based damping scaling with restoring force
            if damping_method == 'restoring_force':
                # Physics-based approach: damping scales inversely with restoring force |N - N_ad|
                # As N → N_ad, restoring force → 0, so dt should be reduced
                delta_N_frac = np.abs(N - N_ad) / np.maximum(N_ad, 1e-10)  # |N-N_ad|/N_ad
                # Add offset to control damping strength (smaller damping_offset = stronger damping effect)
                damping_offset = 0.0001  # Small value to increase damping effect (reduce dt more when close to adiabat)
                # Damping factor: dt_damped = dt_base * (delta_N_frac + damping_offset) / (delta_N_frac + damping_offset + scale_factor)
                # When damping_offset << scale_factor and delta_N_frac is small, damping ≈ damping_offset/(damping_offset + scale_factor) ≈ damping_offset/scale_factor (strong reduction)
                # When |N-N_ad|/N_ad is small → damping is small (strong damping effect) due to small damping_offset
                scale_factor = 0.05  # Controls strength of damping near adiabat
                damping = (delta_N_frac + damping_offset) / (delta_N_frac + damping_offset + scale_factor)
                # Ensure damping is between 0.01 and 1.0
                damping = np.maximum(damping, 0.01)  # Minimum damping = 0.01 (reduce dt by at most 100x)
                damping = np.minimum(damping, 1.0)    # Maximum damping = 1.0 (no reduction)
                
            elif damping_method == 'current' or damping_method == 'stringent':
                # More stringent version of current method
                # Damping reduces dt when:
                #   1. Temperature is low (dT/T is larger, more sensitive)
                #   2. Close to adiabat (easier to overshoot)
                # Damping factor: damping = f_T * f_N
                #   f_T = (T / T_mean)^(1/4) - stronger reduction for low T (was sqrt)
                #   f_N = sqrt(max(0.01, |N-N_ad|/N_ad) / 0.05) - stronger reduction near adiabat (was 0.1 threshold)
                T_mean = np.mean(T_mid[T_mid > 0])  # Mean temperature (avoid zero)
                T_ref = np.maximum(T_mid, T_mean * 0.1)  # Reference T, floor at 10% of mean
                f_T = (T_ref / T_mean) ** 0.25  # Temperature damping: lower T → smaller factor (stronger than sqrt)
                
                # Proximity to adiabat damping (more stringent)
                delta_N_frac = np.abs(N - N_ad) / np.maximum(N_ad, 1e-10)  # |N-N_ad|/N_ad
                delta_N_frac_clamped = np.maximum(delta_N_frac, 0.005)  # Floor at 0.5% (was 1%)
                threshold = 0.05  # 5% from adiabat is "close" (was 10%)
                f_N = np.sqrt(delta_N_frac_clamped / threshold)  # Proximity damping: closer → smaller factor
                f_N = np.minimum(f_N, 1.0)  # Cap at 1.0 (no damping when far from adiabat)
                
                # Combined damping factor
                damping = f_T * f_N
                damping = np.maximum(damping, 0.05)  # Minimum damping = 0.05 (reduce dt by at most 20x, was 0.1)
                
            else:
                # Default: no damping (for testing)
                damping = np.ones_like(dt_layer)
            
            # Apply damping to dt_layer (only for finite values)
            finite_mask = np.isfinite(dt_layer)
            dt_layer[finite_mask] = dt_layer[finite_mask] * damping[finite_mask]
                        
            # #region agent log: investigate dt after apply_dt_method (H3)
            import json
            convective_mask = N > N_ad
            if np.any(convective_mask):
                conv_dt_before = dt_layer_before_method[convective_mask]
                conv_dt_after = dt_layer[convective_mask]
                log_data = {
                    "location": "convective_flux_v3.py:1473",
                    "message": "dt_layer after apply_dt_method for convective layers",
                    "data": {
                        "dt_method": dt_method,
                        "dt_before_method_range": [float(np.min(conv_dt_before)), float(np.max(conv_dt_before))],
                        "dt_after_method_range": [float(np.min(conv_dt_after)), float(np.max(conv_dt_after))],
                        "dt_mean_before": float(np.mean(conv_dt_before)),
                        "dt_mean_after": float(np.mean(conv_dt_after)),
                        "DT_MIN": float(DT_MIN),
                        "DT_MAX": float(DT_MAX),
                        "hitting_DT_MIN": int(np.sum(conv_dt_after == DT_MIN)),
                        "hitting_DT_MAX": int(np.sum(conv_dt_after == DT_MAX))
                    },
                    "timestamp": int(__import__('time').time() * 1000),
                    "runId": "investigate_dt_size",
                    "hypothesisId": "H3,H4"
                }
                with open('/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
            # #endregion
            
            # Debug: Track after apply_dt_method
            if debug and (step == 0 or step % debug_interval == 0):
                rad_mask = N <= N_ad
                conv_mask = N > N_ad
                if np.any(rad_mask):
                    rad_dt = dt_layer[rad_mask]
                    print(f"  After apply_dt_method (method={dt_method}):")
                    print(f"    Radiative layers dt: min={np.min(rad_dt):.2e}, max={np.max(rad_dt):.2e}, "
                          f"mean={np.mean(rad_dt):.2e} s")
                if np.any(conv_mask):
                    conv_dt = dt_layer[conv_mask]
                    print(f"    Convective layers dt: min={np.min(conv_dt):.2e}, max={np.max(conv_dt):.2e}, "
                          f"mean={np.mean(conv_dt):.2e} s")
            # Safety check: ensure no np.inf or np.nan values remain
            if np.any(~np.isfinite(dt_layer)):
                if dt_radiative is None:
                    dt_radiative = DT_RADIATIVE_DEFAULT
                print(f"WARNING: Non-finite values in dt_layer after apply_dt_method! Replacing with dt_radiative={dt_radiative}")
                dt_layer = np.where(np.isfinite(dt_layer), dt_layer, dt_radiative)
                dt_layer = np.clip(dt_layer, DT_MIN, DT_MAX)
            # Use dt_layer directly at interfaces (simpler approach, like v2)
            # Interface i corresponds to the top of layer i-1, so we use dt_layer[i-1]
            # Interface 0 uses dt_layer[0], interface n_layers uses dt_layer[n_layers-1]
            n_interfaces = len(T)
            dt_interface = np.zeros(n_interfaces)
            dt_interface[0] = dt_layer[0]  # Bottom interface uses first layer's dt
            dt_interface[1:] = dt_layer  # Interface i uses dt_layer[i-1] (i.e., dt_layer for layer below interface)
            
            # Debug: Track dt values
            if debug and (step == 0 or step % debug_interval == 0):
                print(f"  dt_layer range: [{np.min(dt_layer):.2e}, {np.max(dt_layer):.2e}] s")
                print(f"  dt_interface range: [{np.min(dt_interface):.2e}, {np.max(dt_interface):.2e}] s")
                # Check for large dt_interface values that might cause issues
                large_dt_threshold = 100.0  # Flag interfaces with dt > 100 s
                large_dt_mask = dt_interface > large_dt_threshold
                if np.any(large_dt_mask):
                    large_dt_indices = np.where(large_dt_mask)[0]
                    print(f"    WARNING: {np.sum(large_dt_mask)} interfaces have dt > {large_dt_threshold} s")
                    print(f"      Large dt interfaces: {large_dt_indices[:10]}{'...' if len(large_dt_indices) > 10 else ''}")
            
            # Safety check: ensure no np.inf or np.nan values in dt_interface
            if np.any(~np.isfinite(dt_interface)):
                if dt_radiative is None:
                    dt_radiative = DT_RADIATIVE_DEFAULT
                print(f"WARNING: Non-finite values in dt_interface! Replacing with dt_radiative={dt_radiative}")
                dt_interface = np.where(np.isfinite(dt_interface), dt_interface, dt_radiative)
                dt_interface = np.clip(dt_interface, DT_MIN, DT_MAX)
        else:
            # Use fixed dt (backward compatibility with v2)
            dt_interface = None  # Will use scalar dt in temperature update
            dt_layer = None
        
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
        # dF_dz is computed at interfaces in erg cm^-3 s^-1
        # Use variable dz for hydrostatic grid, or constant dz for linear grid
        dF_dz_erg_cm3_s = np.zeros(len(z))  # erg cm^-3 s^-1 at interfaces
        
        if profile_type in ["guillot", "isothermal", "semi-isothermal"]:
            # Use variable dz for hydrostatic grid (like v4)
            # Distances between layer centres
            dz_mid_cm = (z_mid[1:] - z_mid[:-1]) * 100.0  # m -> cm
            
            # Interior interfaces: between layer j-1 and j
            # VERIFICATION: dF/dz calculation
            # Interface j is between layer j-1 (below) and layer j (above)
            # dF/dz[j] = (F_conv[j] - F_conv[j-1]) / dz_mid_cm[j-1]
            # If F_conv[j] > F_conv[j-1]: flux increases upward → dF/dz > 0 → more energy leaving → cooling → dT < 0 ✓
            # Sign check: dT = -dt * constant * dF_dz, so dF/dz > 0 → dT < 0 (correct)
            dF_dz_erg_cm3_s[1:-1] = (F_conv[1:] - F_conv[:-1]) / dz_mid_cm
            
            # Bottom boundary: between bottom boundary (F=0) and first layer centre
            dz_bottom_cm = (z_mid[0] - z[0]) * 100.0
            dF_dz_erg_cm3_s[0] = (F_conv[0] - 0.0) / dz_bottom_cm
            
            # Top boundary: between last layer centre and space (F=0 above)
            dz_top_cm = (z[-1] - z_mid[-1]) * 100.0
            dF_dz_erg_cm3_s[-1] = (0.0 - F_conv[-1]) / dz_top_cm
            
            # #region agent log: investigate dF/dz calculation (H2-H5)
            if step == 0:
                import json
                # Sample several interface calculations
                sample_indices = [0, 10, 25, 50, 75, 90, 99]
                dF_dz_details = {}
                for idx in sample_indices:
                    if idx == 0:
                        # Bottom boundary
                        dF_dz_details[f"interface_{idx}"] = {
                            "F_conv_below": 0.0,
                            "F_conv_above": float(F_conv[0]),
                            "dF": float(F_conv[0] - 0.0),
                            "dz_cm": float(dz_bottom_cm),
                            "dF_dz": float(dF_dz_erg_cm3_s[0]),
                            "z_interface": float(z[0]),
                            "z_mid_below": float(z[0]),
                            "z_mid_above": float(z_mid[0])
                        }
                    elif idx == len(z) - 1:
                        # Top boundary
                        dF_dz_details[f"interface_{idx}"] = {
                            "F_conv_below": float(F_conv[-1]),
                            "F_conv_above": 0.0,
                            "dF": float(0.0 - F_conv[-1]),
                            "dz_cm": float(dz_top_cm),
                            "dF_dz": float(dF_dz_erg_cm3_s[-1]),
                            "z_interface": float(z[-1]),
                            "z_mid_below": float(z_mid[-1]),
                            "z_mid_above": float(z[-1])
                        }
                    else:
                        # Interior interface
                        dF_dz_details[f"interface_{idx}"] = {
                            "F_conv_below": float(F_conv[idx-1]),
                            "F_conv_above": float(F_conv[idx]),
                            "dF": float(F_conv[idx] - F_conv[idx-1]),
                            "dz_cm": float(dz_mid_cm[idx-1]),
                            "dF_dz": float(dF_dz_erg_cm3_s[idx]),
                            "z_interface": float(z[idx]),
                            "z_mid_below": float(z_mid[idx-1]),
                            "z_mid_above": float(z_mid[idx])
                        }
                
                log_data = {
                    "location": f"convective_flux_v3.py:{1465}",
                    "message": "dF/dz calculation details at step 0",
                    "data": {
                        "step": step,
                        "dz_mid_cm_sample": {
                            "min": float(np.min(dz_mid_cm)),
                            "max": float(np.max(dz_mid_cm)),
                            "mean": float(np.mean(dz_mid_cm)),
                            "std": float(np.std(dz_mid_cm)),
                            "dz_bottom_cm": float(dz_bottom_cm),
                            "dz_top_cm": float(dz_top_cm)
                        },
                        "dF_dz_details": dF_dz_details,
                        "dF_dz_summary": {
                            "min": float(np.min(dF_dz_erg_cm3_s)),
                            "max": float(np.max(dF_dz_erg_cm3_s)),
                            "mean": float(np.mean(dF_dz_erg_cm3_s)),
                            "std": float(np.std(dF_dz_erg_cm3_s)),
                            "abs_mean": float(np.mean(np.abs(dF_dz_erg_cm3_s)))
                        },
                        "F_conv_differences": {
                            "adjacent_layers_0_1": float(F_conv[1] - F_conv[0]),
                            "adjacent_layers_25_26": float(F_conv[26] - F_conv[25]),
                            "adjacent_layers_50_51": float(F_conv[51] - F_conv[50]),
                            "adjacent_layers_75_76": float(F_conv[76] - F_conv[75]),
                            "max_adjacent_diff": float(np.max(np.abs(np.diff(F_conv)))),
                            "mean_adjacent_diff": float(np.mean(np.abs(np.diff(F_conv))))
                        }
                    },
                    "timestamp": int(__import__('time').time() * 1000),
                    "runId": "investigate_dFdz",
                    "hypothesisId": "H2,H3,H4,H5"
                }
                with open('/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps(log_data) + '\n')
            # #endregion
        else:
            # Use constant dz for linear grid (backward compatibility)
            # dz is defined in the linear profile path above
            dz_cm = dz * 100.0  # cm
            
            # Interior interfaces
            dF_dz_erg_cm3_s[1:-1] = (F_conv[1:] - F_conv[:-1]) / dz_cm
            
            # Boundaries
            dF_dz_erg_cm3_s[0] = (F_conv[0] - 0) / (dz_cm / 2.0)
            dF_dz_erg_cm3_s[-1] = (0 - F_conv[-1]) / (dz_cm / 2.0)
        
        # Stability limit on dt: ensure |dT| <= DT_MAX_CHANGE_FRAC * T at each interface (v3 dynamic timestepping)
        # |dT| = dt * (1/(rho*c_p)) * |dF_dz|  =>  dt <= DT_MAX_CHANGE_FRAC * T * (rho*c_p) / |dF_dz|
        if use_dynamic_dt:
            dt_interface_before_stability = dt_interface.copy()
            dF_dz_abs = np.maximum(np.abs(dF_dz_erg_cm3_s), 1e-30)  # avoid divide by zero
            dt_stable = DT_MAX_CHANGE_FRAC * T * (rho * c_p) / dF_dz_abs
            dt_interface = np.minimum(dt_interface, dt_stable)
            dt_interface_before_clip = dt_interface.copy()
            dt_interface = np.clip(dt_interface, DT_MIN, DT_MAX)
            
            # #region agent log: investigate stability limit and clipping (H3,H4)
            import json
            n_limited_by_stability = np.sum(dt_interface_before_clip < dt_interface_before_stability)
            n_hitting_DT_MIN = np.sum(dt_interface == DT_MIN)
            n_hitting_DT_MAX = np.sum(dt_interface == DT_MAX)
            log_data = {
                "location": "convective_flux_v3.py:1687",
                "message": "stability limit and clipping analysis",
                "data": {
                    "DT_MAX_CHANGE_FRAC": float(DT_MAX_CHANGE_FRAC),
                    "DT_MIN": float(DT_MIN),
                    "DT_MAX": float(DT_MAX),
                    "dt_interface_before_stability_range": [float(np.min(dt_interface_before_stability)), float(np.max(dt_interface_before_stability))],
                    "dt_stable_range": [float(np.min(dt_stable)), float(np.max(dt_stable))],
                    "dt_interface_after_stability_range": [float(np.min(dt_interface_before_clip)), float(np.max(dt_interface_before_clip))],
                    "dt_interface_final_range": [float(np.min(dt_interface)), float(np.max(dt_interface))],
                    "n_limited_by_stability": int(n_limited_by_stability),
                    "n_hitting_DT_MIN": int(n_hitting_DT_MIN),
                    "n_hitting_DT_MAX": int(n_hitting_DT_MAX),
                    "dF_dz_abs_range": [float(np.min(dF_dz_abs)), float(np.max(dF_dz_abs))],
                    "T_range": [float(np.min(T)), float(np.max(T))],
                    "rho_cp_range": [float(np.min(rho * c_p)), float(np.max(rho * c_p))],
                    "sample_interface_88": {
                        "dt_before_stability": float(dt_interface_before_stability[88]),
                        "dt_stable": float(dt_stable[88]),
                        "dt_after_stability": float(dt_interface_before_clip[88]),
                        "dt_final": float(dt_interface[88]),
                        "dF_dz_abs": float(dF_dz_abs[88]),
                        "T": float(T[88]),
                        "rho_cp": float(rho[88] * c_p)
                    }
                },
                "timestamp": int(__import__('time').time() * 1000),
                "runId": "investigate_dt_size",
                "hypothesisId": "H3,H4"
            }
            with open('/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
            # #endregion
            
            # Debug: Track stability limit application
            if debug and (step == 0 or step % debug_interval == 0):
                n_limited = np.sum(dt_interface < dt_interface_before_stability)
                if n_limited > 0:
                    limited_indices = np.where(dt_interface < dt_interface_before_stability)[0]
                    print(f"  After stability limit:")
                    print(f"    Stability limit applied to {n_limited} interfaces")
                    for idx in limited_indices[:5]:  # Show first 5
                        print(f"      Interface {idx}: dt reduced from {dt_interface_before_stability[idx]:.2e} to "
                              f"{dt_interface[idx]:.2e} s (stable limit: {dt_stable[idx]:.2e} s)")
                print(f"    Final dt_interface range: [{np.min(dt_interface):.2e}, {np.max(dt_interface):.2e}] s")
        
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
            # Choose dt to use: scalar dt for fixed timestepping (backward compatibility with v2),
            # dt_interface array for dynamic timestepping
            dt_to_use = dt_interface if use_dynamic_dt else dt
            dT = -dt_to_use * dt_constant_value * dF_dz_SI
        elif use_energy_conservation:
            # Energy conservation version: dT = -dt / (ρ * c_p) * dF_dz
            # Need rho at interfaces - interpolate from layer centers
            # rho is already at interfaces, so we can use it directly
            # Calculate constant at each interface: 1/(ρ * c_p)
            # Units: [g/cm^3] * [erg g^-1 K^-1] = [erg cm^-3 K^-1]
            # Result: [K] = [s] * [erg cm^-3 s^-1] / [erg cm^-3 K^-1] = [K] ✓
            DT_CONSTANT_interface = 1.0 / (rho * c_p)  # cm^3 K / erg (per interface)
            # Choose dt to use: scalar dt for fixed timestepping (backward compatibility with v2),
            # dt_interface array for dynamic timestepping
            dt_to_use = dt_interface if use_dynamic_dt else dt
            # Scalar dt for fixed timestepping (v2 compatibility), element-wise for dynamic
            dT = -dt_to_use * DT_CONSTANT_interface * dF_dz_erg_cm3_s
        else:
            # Default: Use energy conservation form (required for l^2 version)
            # The simplified constant = 1.0 is not appropriate when using l^2 instead of alpha^2
            # because l^2 is much larger, causing flux values to explode
            DT_CONSTANT_interface = 1.0 / (rho * c_p)  # cm^3 K / erg (per interface)
            # Choose dt to use: scalar dt for fixed timestepping (backward compatibility with v2),
            # dt_interface array for dynamic timestepping
            dt_to_use = dt_interface if use_dynamic_dt else dt
            # Scalar dt for fixed timestepping (v2 compatibility), element-wise for dynamic
            dT = -dt_to_use * DT_CONSTANT_interface * dF_dz_erg_cm3_s
        
        # Store debug history (last 5 steps) for transition analysis
        # Store metrics at layer centers (where F_conv, N are defined)
        # Map dT and dF_dz from interfaces to layer centers (use average of adjacent interfaces)
        dT_mid = np.zeros(len(T_mid))
        if len(dT) == len(T_mid) + 1:  # Interfaces = layers + 1
            dT_mid = (dT[:-1] + dT[1:]) / 2.0
        elif len(dT) == len(T_mid):
            dT_mid = dT.copy()
        elif len(dT) > 0:
            # Fallback: use nearest interface value
            for i in range(len(T_mid)):
                if i < len(dT):
                    dT_mid[i] = dT[i]
                elif i > 0:
                    dT_mid[i] = dT[-1]
        
        dF_dz_mid = np.zeros(len(T_mid))
        if len(dF_dz_erg_cm3_s) == len(T_mid) + 1:  # Interfaces = layers + 1
            dF_dz_mid = (dF_dz_erg_cm3_s[:-1] + dF_dz_erg_cm3_s[1:]) / 2.0
        elif len(dF_dz_erg_cm3_s) == len(T_mid):
            dF_dz_mid = dF_dz_erg_cm3_s.copy()
        elif len(dF_dz_erg_cm3_s) > 0:
            # Fallback: use nearest interface value
            for i in range(len(T_mid)):
                if i < len(dF_dz_erg_cm3_s):
                    dF_dz_mid[i] = dF_dz_erg_cm3_s[i]
                elif i > 0:
                    dF_dz_mid[i] = dF_dz_erg_cm3_s[-1]
        
        # Store debug history only if enabled
        if stop_on_first_radiative:
            step_data = {
                'T_mid': T_mid.copy(),
                'P_mid': P_mid.copy(),
                'F_conv': F_conv.copy(),
                't_conv': t_conv.copy(),
                'tau_rad': tau_rad.copy(),
                'N': N.copy(),
                'N_ad': np.full_like(N, N_ad),
                'dT': dT_mid.copy(),
                'dF_dz': dF_dz_mid.copy()
            }
            
            debug_history.append(step_data)
            # Keep only last MAX_HISTORY_STEPS
            if len(debug_history) > MAX_HISTORY_STEPS:
                debug_history.pop(0)
        
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
            if use_dynamic_dt:
                print(f"  dt: DYNAMIC (per-interface)")
                print(f"    dt_interface range: [{np.min(dt_interface):.2e}, {np.max(dt_interface):.2e}] s")
                print(f"    dt_layer range: [{np.min(dt_layer):.2e}, {np.max(dt_layer):.2e}] s")
            else:
                print(f"  dt: {dt} s (FIXED)")
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
                # Calculate dz_cm based on profile type
                if profile_type in ["guillot", "isothermal", "semi-isothermal"]:
                    # For hydrostatic grid, use dz_mid_cm (distance between layer centers)
                    dz_cm_debug = (z_mid[idx_upper] - z_mid[idx_lower]) * 100.0  # m -> cm
                else:
                    # For linear grid, use constant dz
                    dz_cm_debug = dz * 100.0  # cm
                print(f"    F_conv[{idx_lower}] (layer below) = {F_conv[idx_lower]:.2e} erg cm^-2 s^-1")
                print(f"    F_conv[{idx_upper}] (layer above) = {F_conv[idx_upper]:.2e} erg cm^-2 s^-1")
                print(f"    dF_dz = (F[{idx_upper}] - F[{idx_lower}]) / dz")
                print(f"          = ({F_conv[idx_upper]:.2e} - {F_conv[idx_lower]:.2e}) / {dz_cm_debug:.2e} cm")
                print(f"          = {dF_dz_erg_cm3_s[max_dT_idx]:.2e} erg cm^-3 s^-1")
            
            # Show the constant used (depends on method)
            # Get the actual dt value used at this interface
            if use_dynamic_dt:
                dt_val_used = dt_interface[max_dT_idx] if max_dT_idx < len(dt_interface) else dt_interface[-1]
            else:
                dt_val_used = dt
            
            if use_constant_dt_coefficient:
                print(f"    Using constant coefficient: {dt_constant_value:.2e} m·s²·K/kg")
                print(f"    dT = -dt * constant * dF_dz = -{dt_val_used:.2e} * {dt_constant_value:.2e} * {dF_dz_erg_cm3_s[max_dT_idx]:.2e}")
            elif use_energy_conservation or (not use_constant_dt_coefficient):
                const_val = DT_CONSTANT_interface[max_dT_idx] if max_dT_idx < len(DT_CONSTANT_interface) else DT_CONSTANT_interface[-1]
                print(f"    dT = -dt * constant * dF_dz = -{dt_val_used:.2e} * {const_val:.2e} * {dF_dz_erg_cm3_s[max_dT_idx]:.2e}")
            else:
                print(f"    dT = -dt * constant * dF_dz = -{dt_val_used:.2e} * {dt_constant_value:.2e} * {dF_dz_erg_cm3_s[max_dT_idx]:.2e}")
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
        # Units: rho (g/cm^3) * R_specific (erg g^-1 K^-1) * T (K) = erg/cm^3 = dyne/cm^2
        P_new = rho * R_specific * T_new  # Pressure in dyne/cm^2 at interfaces
        
        # Check convergence on updated temperature profile
        # PRIMARY criterion: Adiabaticity (N ≈ N_ad for convective layers)
        # SECONDARY criterion: dT small (to ensure we're not still rapidly evolving)
        
        # Recalculate gradient from updated temperature profile for adiabatic check
        N_new = temperature_gradient(T_new, z)
        
        # Debug: Track layers transitioning from convective to radiative (overshooting)
        # Stop and print detailed info when FIRST radiative layer is detected (only if enabled)
        was_convective = N_prev > N_ad
        now_radiative = N_new <= N_ad
        transitioned_to_radiative = was_convective & now_radiative
        
        if stop_on_first_radiative and np.any(transitioned_to_radiative):
            # Find first transitioning layer (lowest index)
            first_transition_idx = np.where(transitioned_to_radiative)[0][0]
            
            print(f"\n{'='*70}")
            print(f"FIRST RADIATIVE LAYER DETECTED at step {step+1}")
            print(f"{'='*70}")
            print(f"Layer {first_transition_idx} transitioned from CONVECTIVE → RADIATIVE")
            print(f"  N_prev = {N_prev[first_transition_idx]:.6e} K/m (was convective: N > N_ad = {N_ad:.6e} K/m)")
            print(f"  N_new  = {N_new[first_transition_idx]:.6e} K/m (now radiative: N ≤ N_ad)")
            print(f"  Overshoot: ΔN = {N_new[first_transition_idx] - N_prev[first_transition_idx]:.6e} K/m")
            print()
            
            # Print detailed metrics for transitioning layer and 2 above/below
            layer_range = range(max(0, first_transition_idx - 2), 
                               min(len(T_mid), first_transition_idx + 3))
            
            print(f"Detailed metrics for layers {layer_range.start}-{layer_range.stop-1} "
                  f"(transitioning layer {first_transition_idx} ± 2):")
            print(f"{'Layer':<8} {'z (km)':<12} {'T (K)':<12} {'P (bar)':<15} {'F_conv':<20} {'N':<15} {'N_ad':<15} {'dT':<15} {'dF/dz':<20} {'t_conv':<15} {'τ_rad':<15}")
            print("-" * 150)
            
            for layer_idx in layer_range:
                z_km = z_mid[layer_idx] / 1000.0 if layer_idx < len(z_mid) else z[layer_idx] / 1000.0
                T_val = T_mid[layer_idx] if layer_idx < len(T_mid) else T[layer_idx]
                P_val = P_mid[layer_idx] / 1e6 if layer_idx < len(P_mid) else P[layer_idx] / 1e6  # Convert to bar
                F_val = F_conv[layer_idx] if layer_idx < len(F_conv) else 0.0
                N_val = N_new[layer_idx] if layer_idx < len(N_new) else N_ad
                N_ad_val = N_ad
                dT_val = step_data['dT'][layer_idx] if layer_idx < len(step_data['dT']) else 0.0
                dF_dz_val = step_data['dF_dz'][layer_idx] if layer_idx < len(step_data['dF_dz']) else 0.0
                t_conv_val = t_conv[layer_idx] if layer_idx < len(t_conv) else 0.0
                tau_rad_val = tau_rad[layer_idx] if layer_idx < len(tau_rad) else 0.0
                
                marker = " ← TRANSITION" if layer_idx == first_transition_idx else ""
                print(f"{layer_idx:<8} {z_km:<12.2f} {T_val:<12.1f} {P_val:<15.6e} {F_val:<20.6e} "
                      f"{N_val:<15.6e} {N_ad_val:<15.6e} {dT_val:<15.6e} {dF_dz_val:<20.6e} "
                      f"{t_conv_val:<15.6e} {tau_rad_val:<15.6e}{marker}")
            
            print()
            print(f"History for last {len(debug_history)} steps (before transition):")
            print(f"{'Step':<8} {'Layer':<8} {'T (K)':<12} {'P (bar)':<15} {'F_conv':<20} {'N':<15} {'N_ad':<15} {'dT':<15} {'dF/dz':<20} {'t_conv':<15} {'τ_rad':<15}")
            print("-" * 150)
            
            for hist_step_idx, hist_data in enumerate(debug_history):
                hist_step_num = step + 1 - (len(debug_history) - hist_step_idx)
                for layer_idx in layer_range:
                    if layer_idx < len(hist_data['T_mid']):
                        z_km = z_mid[layer_idx] / 1000.0 if layer_idx < len(z_mid) else z[layer_idx] / 1000.0
                        T_val = hist_data['T_mid'][layer_idx]
                        P_val = hist_data['P_mid'][layer_idx] / 1e6  # Convert to bar
                        F_val = hist_data['F_conv'][layer_idx]
                        N_val = hist_data['N'][layer_idx]
                        N_ad_val = hist_data['N_ad'][layer_idx]
                        dT_val = hist_data['dT'][layer_idx]
                        dF_dz_val = hist_data['dF_dz'][layer_idx]
                        t_conv_val = hist_data['t_conv'][layer_idx]
                        tau_rad_val = hist_data['tau_rad'][layer_idx]
                        
                        marker = " ← TRANSITION" if (layer_idx == first_transition_idx and hist_step_idx == len(debug_history) - 1) else ""
                        print(f"{hist_step_num:<8} {layer_idx:<8} {T_val:<12.1f} {P_val:<15.6e} {F_val:<20.6e} "
                              f"{N_val:<15.6e} {N_ad_val:<15.6e} {dT_val:<15.6e} {dF_dz_val:<20.6e} "
                              f"{t_conv_val:<15.6e} {tau_rad_val:<15.6e}{marker}")
                if hist_step_idx < len(debug_history) - 1:
                    print()  # Blank line between steps
            
            print(f"\n{'='*70}")
            print(f"STOPPING ITERATION: First radiative layer detected")
            print(f"{'='*70}\n")
            
            # Update T and P to current values before returning
            T = T_new.copy()
            P = P_new.copy()
            
            # Return early with current state
            return T, z, rho, P, {
                'converged': False,
                'final_step': step + 1,
                'reason': 'First radiative layer detected (overshooting)',
                'transition_layer': int(first_transition_idx),
                'N_prev': float(N_prev[first_transition_idx]),
                'N_new': float(N_new[first_transition_idx]),
                **({k: v for k, v in locals().items() if k in ['history_T', 'history_dT', 'history_F', 'history_dF', 'history_t_conv', 'history_tau_rad', 'timesteps']})
            }
        
        # Original transition summary (keep for other transitions if needed)
        if np.any(transitioned_to_radiative):
            transition_indices = np.where(transitioned_to_radiative)[0]
            # Calculate statistics for transitioning layers
            N_old_trans = N_prev[transition_indices]
            N_new_trans = N_new[transition_indices]
            F_conv_trans = F_conv[transition_indices] if len(F_conv) > len(transition_indices) else F_conv[:len(transition_indices)]
            
            # Calculate timescales BEFORE transition (when still convective)
            # Use T_mid and P_mid from current iteration (before update)
            T_mid_trans = T_mid[transition_indices] if len(T_mid) > len(transition_indices) else T_mid[:len(transition_indices)]
            P_mid_trans = P_mid[transition_indices] if len(P_mid) > len(transition_indices) else P_mid[:len(transition_indices)]
            t_conv_trans = convective_timescale(g, T_mid_trans, N_old_trans, N_ad)
            tau_rad_trans = radiative_timescale(P_mid_trans, g, c_p, T_mid_trans)
            
            # Get dF_dz values (approximate - use interface above layer)
            dF_dz_trans = []
            dT_trans = []
            for idx in transition_indices:
                if idx < len(dF_dz_erg_cm3_s) - 1:
                    dF_dz_trans.append(dF_dz_erg_cm3_s[idx+1])
                elif idx > 0:
                    dF_dz_trans.append(dF_dz_erg_cm3_s[idx])
                else:
                    dF_dz_trans.append(dF_dz_erg_cm3_s[0] if len(dF_dz_erg_cm3_s) > 0 else 0.0)
                
                if idx < len(dT) - 1:
                    dT_trans.append(dT[idx+1])
                elif idx < len(dT):
                    dT_trans.append(dT[idx])
                else:
                    dT_trans.append(dT[-1] if len(dT) > 0 else 0.0)
            
            dF_dz_trans = np.array(dF_dz_trans)
            dT_trans = np.array(dT_trans)
            
            # Check if radiative timescale < convective timescale (radiation faster)
            rad_faster = tau_rad_trans < t_conv_trans
            n_rad_faster = np.sum(rad_faster)
            
            # Print concise summary
            print(f"\n[Step {step+1}] {len(transition_indices)} layer(s) CONVECTIVE→RADIATIVE: "
                  f"layers {transition_indices[0]}-{transition_indices[-1]} "
                  f"(N: {np.mean(N_old_trans):.6e}→{np.mean(N_new_trans):.6e} K/m, "
                  f"mean |dT|={np.mean(np.abs(dT_trans)):.2e} K, "
                  f"mean |dF/dz|={np.mean(np.abs(dF_dz_trans)):.2e} erg cm^-3 s^-1)")
            print(f"  Timescales BEFORE transition: mean t_conv={np.mean(t_conv_trans):.2e} s, "
                  f"mean τ_rad={np.mean(tau_rad_trans):.2e} s, "
                  f"mean ratio τ_rad/t_conv={np.mean(tau_rad_trans/t_conv_trans):.3f}")
            if n_rad_faster > 0:
                print(f"  ⚠️  {n_rad_faster}/{len(transition_indices)} layers have τ_rad < t_conv (radiation faster than convection!)")
            
            # Show worst-case example (largest overshoot)
            worst_idx = transition_indices[np.argmax(N_old_trans - N_new_trans)]
            worst_N_old = N_prev[worst_idx]
            worst_N_new = N_new[worst_idx]
            worst_dT = dT_trans[np.argmax(N_old_trans - N_new_trans)]
            worst_dF_dz = dF_dz_trans[np.argmax(N_old_trans - N_new_trans)]
            worst_t_conv = t_conv_trans[np.argmax(N_old_trans - N_new_trans)]
            worst_tau_rad = tau_rad_trans[np.argmax(N_old_trans - N_new_trans)]
            print(f"  Worst: layer {worst_idx} (N: {worst_N_old:.6e}→{worst_N_new:.6e} K/m, "
                  f"dT={worst_dT:.2e} K, dF/dz={worst_dF_dz:.2e} erg cm^-3 s^-1, "
                  f"t_conv={worst_t_conv:.2e} s, τ_rad={worst_tau_rad:.2e} s, "
                  f"τ_rad/t_conv={worst_tau_rad/worst_t_conv:.3f})")
        if step % 1000 == 0 or step < 10:  # Every 1000 steps or first 10
            n_convective = np.sum(N_new > N_ad)
            n_radiative = np.sum(N_new <= N_ad)
            convective_N = N_new[N_new > N_ad]
            if len(convective_N) > 0:
                mean_N_conv = np.mean(convective_N)
                max_diff_conv = np.max(np.abs(convective_N - N_ad)) / N_ad
                print(f"[Step {step+1}] Convective: {n_convective}, Radiative: {n_radiative} | "
                      f"Conv layers: mean N={mean_N_conv:.6e} K/m, max |N-N_ad|/N_ad={max_diff_conv:.4f}")
        
        # PRIMARY: Check adiabaticity convergence for convective layers
        # For convective layers (N > N_ad): require |N - N_ad|/N_ad < tolerance
        # For radiative layers (N <= N_ad): no requirement (already stable)
        converged_adiabatic = check_adiabatic_convergence(N_new, N_ad, adiabatic_tolerance, debug=(debug and step % debug_interval == 0))
        
        # SECONDARY: Check dT convergence (ensure we're not still rapidly evolving)
        # Use a relaxed tolerance for dT - we care more about adiabaticity
        # But if dT is very large, we're still evolving rapidly, so continue
        converged_dt = (max_dT < convergence_tol)
        
        # Convergence requires:
        # 1. Adiabaticity achieved (PRIMARY)
        # 2. dT is small enough that we're not rapidly evolving (SECONDARY)
        if converged_adiabatic and converged_dt:
            print(f"\nConverged at step {step+1}!")
            print(f"  PRIMARY: All convective layers within {adiabatic_tolerance*100:.1f}% of adiabatic (N_ad = {N_ad:.6f} K/m)")
            print(f"  SECONDARY: Max |dT| = {max_dT:.6e} K < tolerance {convergence_tol:.6e} K (temperature changes are small)")
            # Update T, P, and N to final values
            T = T_new
            P = P_new
            N = N_new
            break
        elif converged_adiabatic and not converged_dt:
            # Adiabatic but dT still changing - this is OK if dT is reasonably small
            # Only warn if dT is very large (rapid evolution)
            if max_dT > 10 * convergence_tol:  # dT is 10x larger than tolerance
                if debug and step % debug_interval == 0:
                    print(f"Step {step+1:4d}: Adiabatic but max|dT| = {max_dT:.6e} K is large (continuing...)")
            # Continue iterating - adiabaticity is achieved, but let it settle
        elif not converged_adiabatic:
            # Not yet adiabatic - continue iterating (this is the primary goal)
            convective_mask = N_new > N_ad
            if np.any(convective_mask):
                convective_N = N_new[convective_mask]
                relative_diff = np.abs(convective_N - N_ad) / N_ad
                max_diff = np.max(relative_diff)
                if debug and step % debug_interval == 0:
                    print(f"Step {step+1:4d}: Not yet adiabatic (PRIMARY criterion)")
                    print(f"  Max |N-N_ad|/N_ad for convective layers: {max_diff:.4f} (need < {adiabatic_tolerance:.2f})")
                    print(f"  Max |dT| = {max_dT:.6e} K (secondary)")
            # Don't break - continue iterating to reach adiabaticity
        
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
        # Check final adiabaticity status
        N_final_check = temperature_gradient(T_new, z)
        converged_adiabatic_final = check_adiabatic_convergence(N_final_check, N_ad, adiabatic_tolerance, debug=True)
        if not converged_adiabatic_final:
            convective_mask = N_final_check > N_ad
            if np.any(convective_mask):
                convective_N = N_final_check[convective_mask]
                relative_diff = np.abs(convective_N - N_ad) / N_ad
                max_diff = np.max(relative_diff)
                print(f"  Final max |N-N_ad|/N_ad for convective layers: {max_diff:.4f} (need < {adiabatic_tolerance:.2f})")
    
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
        
        # Format timescales (print actual values, even if infinite)
        # For radiative layers, show what dynamic timestepping would use
        if np.isinf(t_conv_val):
            # Layer is radiative (N <= N_ad)
            if dt_method == 'radiative':
                # Use radiative timescale for radiative layers
                t_conv_str = f"{tau_rad_val:.2e}"
                if tau_rad_val > 0 and not np.isinf(tau_rad_val):
                    ratio = tau_rad_val / tau_rad_val  # Should be 1.0
                    ratio_str = f"{ratio:.2e}"
                else:
                    ratio_str = "N/A"
            else:
                # For other methods, calculate what apply_dt_method would use
                # Using the 'convective' method formula: dt = DT_CONST * [g/T * |N - N_ad|]^{-1/2}
                EPSILON_MIN = 1e-10
                delta_N_abs = np.abs(N_final[i] - N_ad)
                delta_N_abs = max(delta_N_abs, EPSILON_MIN)
                term = (g / T_val) * delta_N_abs
                dt_dynamic = DT_CONST * (1.0 / np.sqrt(term))
                dt_dynamic = np.clip(dt_dynamic, DT_MIN, DT_MAX)
                t_conv_str = f"{dt_dynamic:.2e}"
                if tau_rad_val > 0 and not np.isinf(tau_rad_val):
                    ratio = dt_dynamic / tau_rad_val
                    ratio_str = f"{ratio:.2e}"
                else:
                    ratio_str = "N/A"
        else:
            t_conv_str = f"{t_conv_val:.2e}"
            if tau_rad_val > 0 and not np.isinf(tau_rad_val):
                ratio = t_conv_val / tau_rad_val
                ratio_str = f"{ratio:.2e}"
            else:
                ratio_str = "N/A"
        
        if np.isinf(tau_rad_val):
            tau_rad_str = "inf"
        else:
            tau_rad_str = f"{tau_rad_val:.2e}"
        
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
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Fixed timestep in seconds. "
            "If omitted and --dynamic-dt is not set, uses default DT from code "
            f"(DT = {DT} s). If --dynamic-dt is set and --dt is omitted, "
            "dynamic per-layer timestepping is used."
        ),
    )
    parser.add_argument(
        "--dynamic-dt",
        action="store_true",
        help=(
            "Enable dynamic timestepping (dt computed per layer from convective "
            "timescale). When this flag is set and --dt is NOT provided, "
            "the solver runs in dynamic mode (dt=None internally). "
            "By default (without this flag), timestepping is FIXED."
        ),
    )
    parser.add_argument(
        "--dt-method",
        type=str,
        choices=["gradient", "fixed", "absolute", "hybrid", "minimum", "convective", "formal", "radiative"],
        default="formal",
        help=(
            "Method for handling layers with N <= N_ad when dynamic dt is enabled. "
            "'formal' (default) uses DT_CONST * [g/T * |N - N_ad|]^{-1/2} for all layers, continuous across RCB. "
            "'convective' uses convective timescale formula DT_CONST * [g/T * |N - N_ad|]^{-1/2} for radiative layers. "
            "'radiative' uses the radiative timescale (tau_rad) for radiative layers, which is physically appropriate. "
            "Ignored when using fixed dt."
        ),
    )
    parser.add_argument(
        "--dt-radiative",
        type=float,
        default=None,
        help=(
            "Fixed dt (s) for radiative layers when --dt-method=fixed. "
            f"Defaults to DT_RADIATIVE_DEFAULT = {DT_RADIATIVE_DEFAULT} s if omitted."
        ),
    )
    parser.add_argument(
        "--dt-convergence",
        type=float,
        default=None,
        help=(
            "Convergence constant (s·K/m) for --dt-method=gradient or hybrid. "
            f"Defaults to DT_CONVERGENCE_DEFAULT = {DT_CONVERGENCE_DEFAULT} if omitted."
        ),
    )
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
    parser.add_argument("--profile-type", type=str, choices=["linear", "guillot", "isothermal", "semi-isothermal"], default="linear",
                       help="TP profile type: 'linear', 'guillot', 'isothermal', or 'semi-isothermal' (default: linear). "
                            "'isothermal' uses hydrostatic grid with constant T=T_boa (N=0, stable). "
                            "'semi-isothermal' uses hydrostatic grid with superadiabatic gradient (N>N_ad, unstable, drives convection).")
    parser.add_argument("--no-prompt", action="store_true",
                       help="Skip interactive prompts, use defaults")
    parser.add_argument("--stop-on-first-radiative", action="store_true",
                       help="Stop iteration when first radiative layer is detected (debug mode)")
    parser.add_argument("--damping-method", type=str, 
                       choices=['current', 'stringent', 'restoring_force', 'none'],
                       default='current',
                       help=("Damping method for dynamic timestepping near adiabat: "
                             "'current' = original damping (default), "
                             "'stringent' = stronger damping (same as 'current'), "
                             "'restoring_force' = physics-based damping scaling with |N-N_ad|, "
                             "'none' = no damping"))
    
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
    
    # Decide on timestep configuration
    # Default behaviour (no --dynamic-dt): fixed dt
    # - If --dt is provided, use that value
    # - If --dt is omitted, use code default DT
    # Dynamic behaviour (with --dynamic-dt): per-layer dt
    # - If --dynamic-dt is set, dt_internal = None (dynamic mode), regardless of --dt
    if args.dynamic_dt:
        dt_internal = None  # enables dynamic timestepping inside run()
    else:
        # Fixed dt mode
        if args.dt is not None:
            dt_internal = args.dt
        else:
            dt_internal = DT
    
    # Run solver
    z, T, rho, P, diagnostics = run(
        n_layers=args.n_layers,
        max_z=args.max_z,
        dt=dt_internal,
        debug=args.debug,
        max_steps=args.max_steps,
        convergence_tol=args.tol,
        save_history=args.plot,
        profile_type=profile_type,
        guillot_params=guillot_params,
        dt_method=args.dt_method,
        dt_radiative=args.dt_radiative,
        dt_convergence=args.dt_convergence,
        stop_on_first_radiative=args.stop_on_first_radiative,
        damping_method=args.damping_method,
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
