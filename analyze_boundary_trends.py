"""Deep analysis of trends in radiative boundary data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results_file = "/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/plots/radiative_boundary/radiative_boundary_results.npz"
data = np.load(results_file)

# Create DataFrame
df = pd.DataFrame({
    'n_layers': data['n_layers'],
    'mixing_length': data['mixing_length'],
    'timestep': data['timestep'],
    'boundary_altitude': data['boundary_altitude'],
    'n_convective': data['n_convective'],
    'n_radiative': data['n_radiative'],
    'max_grad_diff': data['max_grad_diff']
})

print("=" * 100)
print("DEEP TREND ANALYSIS")
print("=" * 100)

# 1. Check altitude measurement consistency
print("\n1. ALTITUDE MEASUREMENT CONSISTENCY CHECK")
print("-" * 100)
print("\nGrouping by boundary altitude to check for clustering:")
altitude_groups = df.groupby('boundary_altitude').agg({
    'n_layers': ['count', 'unique'],
    'mixing_length': 'unique',
    'timestep': 'unique',
    'n_convective': ['mean', 'min', 'max'],
    'n_radiative': ['mean', 'min', 'max']
}).round(2)

print(altitude_groups)

print("\nUnique boundary altitudes:")
unique_altitudes = sorted(df['boundary_altitude'].unique())
print(f"  {unique_altitudes}")
print(f"\n  Total unique values: {len(unique_altitudes)}")
print(f"  Most common altitude: {df['boundary_altitude'].mode().values[0]:.1f} km ({df['boundary_altitude'].value_counts().iloc[0]} occurrences)")

# 2. Analyze trends by parameter
print("\n\n2. TREND ANALYSIS BY PARAMETER")
print("-" * 100)

# 2a. Effect of mixing length
print("\n2a. Effect of Mixing Length (averaged across timesteps and n_layers):")
ml_effect = df.groupby('mixing_length').agg({
    'boundary_altitude': ['mean', 'std', 'min', 'max'],
    'n_convective': ['mean', 'std'],
    'n_radiative': ['mean', 'std']
}).round(2)
print(ml_effect)

# 2b. Effect of timestep
print("\n2b. Effect of Timestep (averaged across mixing lengths and n_layers):")
dt_effect = df.groupby('timestep').agg({
    'boundary_altitude': ['mean', 'std', 'min', 'max'],
    'n_convective': ['mean', 'std'],
    'n_radiative': ['mean', 'std']
}).round(2)
print(dt_effect)

# 2c. Effect of n_layers
print("\n2c. Effect of Number of Layers (averaged across mixing lengths and timesteps):")
nl_effect = df.groupby('n_layers').agg({
    'boundary_altitude': ['mean', 'std', 'min', 'max'],
    'n_convective': ['mean', 'std'],
    'n_radiative': ['mean', 'std']
}).round(2)
print(nl_effect)

# 3. Interaction effects
print("\n\n3. INTERACTION EFFECTS")
print("-" * 100)

# 3a. Mixing length × Timestep interaction
print("\n3a. Mixing Length × Timestep Interaction (averaged across n_layers):")
ml_dt_interaction = df.groupby(['mixing_length', 'timestep']).agg({
    'boundary_altitude': 'mean',
    'n_convective': 'mean',
    'n_radiative': 'mean'
}).round(2)
print(ml_dt_interaction)

# 3b. Mixing length × n_layers interaction
print("\n3b. Mixing Length × n_layers Interaction (averaged across timesteps):")
ml_nl_interaction = df.groupby(['mixing_length', 'n_layers']).agg({
    'boundary_altitude': 'mean',
    'n_convective': 'mean',
    'n_radiative': 'mean'
}).round(2)
print(ml_nl_interaction)

# 3c. Timestep × n_layers interaction
print("\n3c. Timestep × n_layers Interaction (averaged across mixing lengths):")
dt_nl_interaction = df.groupby(['timestep', 'n_layers']).agg({
    'boundary_altitude': 'mean',
    'n_convective': 'mean',
    'n_radiative': 'mean'
}).round(2)
print(dt_nl_interaction)

# 4. Conditional trends (user's observations)
print("\n\n4. VERIFICATION OF USER OBSERVATIONS")
print("-" * 100)

# 4a. At short mixing length, effect of timestep
print("\n4a. At SHORT mixing length (0.1), effect of increasing timestep:")
short_ml = df[df['mixing_length'] == 0.1].groupby('timestep').agg({
    'n_convective': 'mean',
    'n_radiative': 'mean',
    'boundary_altitude': 'mean'
}).round(2)
print(short_ml)
print("  Trend: Increasing timestep → ", end="")
if short_ml['n_convective'].is_monotonic_increasing:
    print("INCREASES convective layers")
elif short_ml['n_convective'].is_monotonic_decreasing:
    print("DECREASES convective layers")
else:
    print("MIXED effect on convective layers")

# 4b. At long mixing length, effect of timestep
print("\n4b. At LONG mixing length (1.0), effect of increasing timestep:")
long_ml = df[df['mixing_length'] == 1.0].groupby('timestep').agg({
    'n_convective': 'mean',
    'n_radiative': 'mean',
    'boundary_altitude': 'mean'
}).round(2)
print(long_ml)
print("  Trend: Increasing timestep → ", end="")
if long_ml['n_convective'].is_monotonic_increasing:
    print("INCREASES convective layers")
elif long_ml['n_convective'].is_monotonic_decreasing:
    print("DECREASES convective layers")
else:
    print("DECREASES convective layers (strongly)")

# 4c. Effect of increasing mixing length
print("\n4c. Effect of increasing mixing length (averaged across all timesteps and n_layers):")
ml_increase = df.groupby('mixing_length')['n_convective'].mean()
print(ml_increase)
print("  Trend: Increasing mixing length → ", end="")
if ml_increase.is_monotonic_increasing:
    print("INCREASES convective layers")
elif ml_increase.is_monotonic_decreasing:
    print("DECREASES convective layers")
else:
    print("MIXED effect")

# 5. Find additional trends
print("\n\n5. ADDITIONAL TRENDS DISCOVERED")
print("-" * 100)

# 5a. Boundary altitude vs layer resolution
print("\n5a. Boundary altitude sensitivity to layer resolution:")
for nl in sorted(df['n_layers'].unique()):
    nl_data = df[df['n_layers'] == nl]
    print(f"  n_layers={nl:3d}: boundary altitude range = [{nl_data['boundary_altitude'].min():.1f}, {nl_data['boundary_altitude'].max():.1f}] km, "
          f"std = {nl_data['boundary_altitude'].std():.1f} km")

# 5b. Cases where entire atmosphere becomes radiative
print("\n5b. Cases where entire atmosphere becomes radiative (n_convective = 0):")
all_rad = df[df['n_convective'] == 0]
if len(all_rad) > 0:
    print(f"  Found {len(all_rad)} cases:")
    print(all_rad[['n_layers', 'mixing_length', 'timestep', 'boundary_altitude']].to_string(index=False))
else:
    print("  None found")

# 5c. Cases where entire atmosphere becomes convective
print("\n5c. Cases where entire atmosphere becomes convective (n_radiative = 0):")
all_conv = df[df['n_radiative'] == 0]
if len(all_conv) > 0:
    print(f"  Found {len(all_conv)} cases:")
    print(all_conv[['n_layers', 'mixing_length', 'timestep', 'boundary_altitude']].to_string(index=False))
else:
    print("  None found")

# 5d. Correlation analysis
print("\n5d. Correlation coefficients:")
correlations = df[['mixing_length', 'timestep', 'n_layers', 'boundary_altitude', 'n_convective', 'n_radiative']].corr()
print(correlations[['boundary_altitude', 'n_convective', 'n_radiative']].round(3))

# 5e. Ratio analysis
print("\n5e. Convective fraction analysis:")
df['convective_fraction'] = df['n_convective'] / df['n_layers']
conv_frac_by_ml = df.groupby('mixing_length')['convective_fraction'].agg(['mean', 'std', 'min', 'max'])
print("\nConvective fraction by mixing length:")
print(conv_frac_by_ml.round(3))

conv_frac_by_dt = df.groupby('timestep')['convective_fraction'].agg(['mean', 'std', 'min', 'max'])
print("\nConvective fraction by timestep:")
print(conv_frac_by_dt.round(3))

# 5f. Parameter combinations that maximize/minimize convective layers
print("\n5f. Parameter combinations:")
max_conv = df.loc[df['n_convective'].idxmax()]
print(f"  Maximum convective layers ({max_conv['n_convective']:.0f}): "
      f"n_layers={max_conv['n_layers']:.0f}, l={max_conv['mixing_length']:.2f}, dt={max_conv['timestep']:.0f}")

min_conv = df.loc[df['n_convective'].idxmin()]
print(f"  Minimum convective layers ({min_conv['n_convective']:.0f}): "
      f"n_layers={min_conv['n_layers']:.0f}, l={min_conv['mixing_length']:.2f}, dt={min_conv['timestep']:.0f}")

print("\n" + "=" * 100)
print("Analysis complete!")
print("=" * 100)
