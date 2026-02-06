"""
Compare Alpha Across Planet Types

Generates parameter space plots for three planet types:
1. Hot Jupiter (H2-dominated)
2. Sub-Neptune (H2/He mix)
3. Terrestrial (Earth-like, N2/O2)

Author: Generated for multi-planet comparison
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from calibrate_alpha_adiabatic import calibrate_alpha_adiabatic
from explore_alpha_adiabatic_space import explore_tp_space_adiabatic, plot_results
import warnings
warnings.filterwarnings('ignore')


def run_planet_type(name, params_dict, T_range, P_range):
    """
    Run parameter space exploration for one planet type.
    
    Parameters:
        name: Planet type name (for filenames)
        params_dict: Dict with physical_params, F_conv, description
        T_range, P_range: Temperature and pressure ranges
    """
    print("\n" + "=" * 70)
    print(f"  {params_dict['description'].upper()}")
    print("=" * 70)
    print()
    
    physical_params = params_dict['physical_params']
    F_conv = params_dict['F_conv']
    
    print(f"Physical parameters:")
    print(f"  g = {physical_params['g']} m/s²")
    print(f"  μ = {physical_params['mu']} kg/mol")
    print(f"  c_p = {physical_params['c_p']} J/(kg·K)")
    
    # Calculate ∇_ad
    R_spec = physical_params['R_universal'] / physical_params['mu']
    nabla_ad = R_spec / physical_params['c_p']
    print(f"  ∇_ad = {nabla_ad:.4f}")
    
    print(f"\nConvective flux: F_conv = {F_conv:.2e} W/m²")
    print()
    
    # Density function
    def rho_func(T, P):
        return (P * physical_params['mu']) / (physical_params['R_universal'] * T)
    
    # Flux function
    def F_conv_func(T, P):
        return F_conv
    
    print(f"Running {len(T_range)} × {len(P_range)} = {len(T_range)*len(P_range)} calculations...")
    
    results = explore_tp_space_adiabatic(T_range, P_range, rho_func, F_conv_func,
                                        physical_params, epsilon=0.001, verbose=False)
    
    # Statistics
    alpha_valid = results['alpha_grid'][~np.isnan(results['alpha_grid'])]
    
    print(f"\nResults:")
    print(f"  Valid solutions: {len(alpha_valid)} / {results['alpha_grid'].size}")
    print(f"  Mean α: {np.nanmean(alpha_valid):.4f}")
    print(f"  Median α: {np.nanmedian(alpha_valid):.4f}")
    print(f"  Range: [{np.nanmin(alpha_valid):.4f}, {np.nanmax(alpha_valid):.4f}]")
    
    in_stellar = np.sum((alpha_valid >= 1.0) & (alpha_valid <= 3.0))
    in_solar = np.sum((alpha_valid >= 1.5) & (alpha_valid <= 2.0))
    print(f"  In stellar range [1.0-3.0]: {in_stellar} ({100*in_stellar/len(alpha_valid):.1f}%)")
    print(f"  In solar range [1.5-2.0]: {in_solar} ({100*in_solar/len(alpha_valid):.1f}%)")
    
    # Save and plot
    filename = f'alpha_{name}'
    np.savez(f'{filename}.npz', **results)
    plot_results(results, save_prefix=filename)
    
    print(f"\nSaved:")
    print(f"  • {filename}.npz")
    print(f"  • {filename}_comprehensive.png")
    print(f"  • {filename}_detailed.png")
    
    return results, alpha_valid


def main():
    """
    Run all three planet types.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  PLANET TYPE COMPARISON: α PARAMETER SPACE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Common ranges
    T_range = np.linspace(200, 2000, 20)    # K
    P_range = np.logspace(0, 8, 20)         # 1 Pa to 10^8 Pa
    
    print(f"\nParameter space (all planet types):")
    print(f"  Temperature: {T_range.min():.0f} - {T_range.max():.0f} K ({len(T_range)} points)")
    print(f"  Pressure: {P_range.min():.2e} - {P_range.max():.2e} Pa ({len(P_range)} points)")
    print(f"            = {P_range.min()/1e5:.1e} - {P_range.max()/1e5:.1e} bar")
    
    # ===== 1. HOT JUPITER =====
    hot_jupiter = {
        'description': 'Hot Jupiter (H2-dominated)',
        'physical_params': {
            'g': 10.0,
            'delta': 1.0,
            'R_universal': 8.314,
            'mu': 0.0022,
            'c_p': 14000.0,
        },
        'F_conv': 5e6,
    }
    
    results_hj, alpha_hj = run_planet_type('hot_jupiter', hot_jupiter, T_range, P_range)
    
    # ===== 2. SUB-NEPTUNE =====
    sub_neptune = {
        'description': 'Sub-Neptune (H2/He mix)',
        'physical_params': {
            'g': 15.0,
            'delta': 1.0,
            'R_universal': 8.314,
            'mu': 0.0035,
            'c_p': 10000.0,
        },
        'F_conv': 1e6,
    }
    
    results_sn, alpha_sn = run_planet_type('sub_neptune', sub_neptune, T_range, P_range)
    
    # ===== 3. TERRESTRIAL (EARTH-LIKE) =====
    terrestrial = {
        'description': 'Terrestrial (Earth-like, N2/O2)',
        'physical_params': {
            'g': 9.81,
            'delta': 1.0,
            'R_universal': 8.314,
            'mu': 0.029,
            'c_p': 1005.0,
        },
        'F_conv': 200,
    }
    
    results_terr, alpha_terr = run_planet_type('terrestrial_earth', terrestrial, T_range, P_range)
    
    # ===== COMPARISON SUMMARY =====
    print("\n\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  COMPARISON SUMMARY".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()
    
    print(f"{'Planet Type':<25} {'Mean α':<12} {'Median α':<12} {'Range':<20}")
    print("-" * 70)
    print(f"{'Hot Jupiter':<25} {np.mean(alpha_hj):>10.3f}  {np.median(alpha_hj):>10.3f}  "
          f"{np.min(alpha_hj):>7.3f} - {np.max(alpha_hj):<7.3f}")
    print(f"{'Sub-Neptune':<25} {np.mean(alpha_sn):>10.3f}  {np.median(alpha_sn):>10.3f}  "
          f"{np.min(alpha_sn):>7.3f} - {np.max(alpha_sn):<7.3f}")
    print(f"{'Terrestrial (Earth)':<25} {np.mean(alpha_terr):>10.3f}  {np.median(alpha_terr):>10.3f}  "
          f"{np.min(alpha_terr):>7.3f} - {np.max(alpha_terr):<7.3f}")
    
    print()
    print("=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print()
    print("1. Hot Jupiter (light atmosphere, high flux):")
    print(f"   • Smaller α values (mean: {np.mean(alpha_hj):.2f})")
    print(f"   • Efficient small-scale convection")
    print()
    print("2. Sub-Neptune (intermediate):")
    print(f"   • Moderate α values (mean: {np.mean(alpha_sn):.2f})")
    print(f"   • Transitional convection regime")
    print()
    print("3. Terrestrial (heavy atmosphere, low flux):")
    print(f"   • Larger α values (mean: {np.mean(alpha_terr):.2f})")
    print(f"   • Larger-scale mixing expected")
    print()
    print("=" * 70)
    print()
    print("All plots saved! Check these files:")
    print("  • alpha_hot_jupiter_comprehensive.png")
    print("  • alpha_hot_jupiter_detailed.png")
    print("  • alpha_sub_neptune_comprehensive.png")
    print("  • alpha_sub_neptune_detailed.png")
    print("  • alpha_terrestrial_earth_comprehensive.png")
    print("  • alpha_terrestrial_earth_detailed.png")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()




