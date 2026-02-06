# Mixing-Length Theory for Atmospheric Convection

This directory contains tools for implementing mixing-length theory (MLT) as an alternative to simple adiabatic corrections in radiative transfer codes.

## Purpose

Many radiative transfer codes use a simplified convection treatment where layers are simply forced to follow an adiabat when they become superadiabatic. This implementation provides a more physically-motivated approach using mixing-length theory to calculate convective energy transport.

## Key Concept

In mixing-length theory, the mixing length is defined as:

$$l = \alpha H$$

where:
- `l` is the mixing length (characteristic size of convective eddies)
- `α` is the dimensionless mixing-length parameter
- `H` is the pressure scale height

The goal is to determine what value of `α` produces an adiabatic temperature profile through convective energy transport.

## Files

### `mixing_length_theory.py`
Core Python module implementing MLT calculations:
- `MixingLengthConvection` class with methods for:
  - Calculating pressure scale height
  - Computing adiabatic gradients
  - Finding optimal α for adiabatic profiles
  - Calculating convective velocities and fluxes
  - Parameter space exploration

### `explore_alpha_parameter.ipynb`
Interactive Jupyter notebook for:
- Parameter space exploration (T, P, cp)
- Visualization of α variation with heatmaps
- Statistical analysis of optimal α values
- Sensitivity analysis
- Export of results for use in RT codes

## Quick Start

1. **Install dependencies:**
```bash
pip install numpy scipy matplotlib seaborn jupyter
```

2. **Run a quick test:**
```bash
python mixing_length_theory.py
```

3. **Explore parameter space:**
```bash
jupyter notebook explore_alpha_parameter.ipynb
```

## Physical Background

### Adiabatic Temperature Gradient

For an adiabatic atmosphere:

$$\frac{dT}{dz} = -\frac{g}{c_p}$$

where:
- `g` is gravitational acceleration [m/s²]
- `cp` is specific heat capacity at constant pressure [J/(kg·K)]

### Schwarzschild Criterion

A layer is convectively unstable if:

$$\frac{dT}{dz} < \frac{dT}{dz}\bigg|_{ad}$$

i.e., the actual gradient is steeper (more negative) than the adiabatic gradient.

### Mixing-Length Theory

MLT provides a framework to calculate convective energy transport:

1. **Pressure scale height:**
   $$H = \frac{RT}{\mu g}$$

2. **Mixing length:**
   $$l = \alpha H$$

3. **Convective velocity:**
   $$v_{conv} \sim \sqrt{l \cdot g \cdot |\nabla - \nabla_{ad}| / \nabla_{ad}}$$

4. **Convective flux:**
   $$F_{conv} = \rho c_p v_{conv} \Delta T$$

## Typical α Values

Based on stellar and planetary atmosphere models:
- **Solar convection zone:** α ≈ 1.5 - 2.0
- **Hot Jupiter atmospheres:** α ≈ 1.0 - 2.5
- **Brown dwarfs:** α ≈ 1.0 - 1.5

The parameter space exploration helps determine appropriate values for your specific application.

## Usage in Radiative Transfer Codes

### Option 1: Constant α
Use the mean value from parameter space exploration:
```python
alpha = 1.5  # Typical value
l = alpha * H
# Use l in convective flux calculations
```

### Option 2: Parameter-Dependent α
Implement interpolation from the parameter space results:
```python
# Load pre-computed results
data = np.load('alpha_parameter_space.npz')
# Interpolate for your specific T, P, cp
alpha = interpolate_alpha(T, P, cp, data)
```

### Option 3: Dynamic α
Calculate α on-the-fly for each layer:
```python
mlt = MixingLengthConvection(g, mu, R)
for layer in atmosphere:
    alpha = mlt.find_alpha_for_adiabat(layer.T, layer.P, layer.cp)
    # Apply convective correction
```

## Customization

You can adjust the following parameters in the code:

1. **Physical constants:**
   - `g`: Gravitational acceleration (planet-specific)
   - `mu`: Mean molecular weight (composition-dependent)
   - `R`: Gas constant (8.314 J/(mol·K))

2. **Parameter ranges:**
   - Temperature: Adjust based on your atmosphere
   - Pressure: Set appropriate vertical extent
   - Specific heat: Depends on atmospheric composition

3. **MLT implementation:**
   - Modify `find_alpha_for_adiabat()` for different optimization criteria
   - Adjust convective velocity formulation
   - Add additional physics (radiation, diffusion, etc.)

## References

- Böhm-Vitense, E. (1958). "Über die Wasserstoffkonvektionszone in Sternen verschiedener Effektivtemperaturen und Leuchtkräfte." *Zeitschrift für Astrophysik*, 46, 108.
- Henyey, L., et al. (1965). "Studies in Stellar Evolution. V." *ApJ*, 142, 841.
- Hansen, C. J., & Kawaler, S. D. (1994). *Stellar Interiors*. Springer.
- Guillot, T. (2010). "On the radiative equilibrium of irradiated planetary atmospheres." *A&A*, 520, A27.

## Next Steps

1. Run parameter space exploration with your atmospheric conditions
2. Determine optimal α value(s) for your application
3. Implement MLT in your radiative transfer code
4. Compare results with simple adiabatic adjustment
5. Validate against observations or detailed models

## Contact

For questions or improvements to this implementation, please refer to the main project documentation.


