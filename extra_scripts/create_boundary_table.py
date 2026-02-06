"""Create a table showing mixing length, timestep, and boundary altitude from results"""

import numpy as np
import pandas as pd

# Load results
results_file = "/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/plots/radiative_boundary/radiative_boundary_results.npz"
data = np.load(results_file)

# Extract data
n_layers = data['n_layers']
mixing_length = data['mixing_length']
timestep = data['timestep']
boundary_altitude = data['boundary_altitude']
n_convective = data['n_convective']
n_radiative = data['n_radiative']

# Create DataFrame
df = pd.DataFrame({
    'n_layers': n_layers,
    'mixing_length': mixing_length,
    'timestep': timestep,
    'boundary_altitude_km': boundary_altitude,
    'n_convective': n_convective,
    'n_radiative': n_radiative
})

# Sort by n_layers, then mixing_length, then timestep
df = df.sort_values(['n_layers', 'mixing_length', 'timestep']).reset_index(drop=True)

# Print table for each n_layers
n_layers_list = sorted(df['n_layers'].unique())

print("=" * 100)
print("Radiative-Convective Boundary Altitude Table")
print("=" * 100)
print()

for n_layers in n_layers_list:
    df_subset = df[df['n_layers'] == n_layers].copy()
    
    print(f"\n{'='*100}")
    print(f"n_layers = {n_layers}")
    print(f"{'='*100}")
    print(f"{'Mixing Length':<15} {'Timestep (s)':<15} {'Boundary Altitude (km)':<25} {'N Convective':<15} {'N Radiative':<15}")
    print("-" * 100)
    
    for _, row in df_subset.iterrows():
        ml = row['mixing_length']
        dt = row['timestep']
        alt = row['boundary_altitude_km']
        n_conv = int(row['n_convective']) if not np.isnan(row['n_convective']) else 'NaN'
        n_rad = int(row['n_radiative']) if not np.isnan(row['n_radiative']) else 'NaN'
        
        if np.isnan(alt):
            alt_str = "NaN"
        else:
            alt_str = f"{alt:.2f}"
        
        print(f"{ml:<15.2f} {dt:<15.0f} {alt_str:<25} {n_conv:<15} {n_rad:<15}")

# Also create a pivot table for easier viewing
print("\n\n" + "=" * 100)
print("Pivot Tables: Boundary Altitude (km) by Mixing Length and Timestep")
print("=" * 100)

for n_layers in n_layers_list:
    df_subset = df[df['n_layers'] == n_layers].copy()
    
    # Create pivot table
    pivot = df_subset.pivot_table(
        values='boundary_altitude_km',
        index='mixing_length',
        columns='timestep',
        aggfunc='first'
    )
    
    print(f"\n{'='*100}")
    print(f"n_layers = {n_layers}")
    print(f"{'='*100}")
    print("\nBoundary Altitude (km):")
    print(pivot.to_string(float_format='%.2f'))
    
    # Also show number of convective layers
    pivot_conv = df_subset.pivot_table(
        values='n_convective',
        index='mixing_length',
        columns='timestep',
        aggfunc='first'
    )
    
    print(f"\nNumber of Convective Layers:")
    print(pivot_conv.to_string(float_format='%.0f'))

# Save to CSV
output_csv = "/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/plots/radiative_boundary/radiative_boundary_table.csv"
df.to_csv(output_csv, index=False)
print(f"\n\nTable saved to: {output_csv}")
