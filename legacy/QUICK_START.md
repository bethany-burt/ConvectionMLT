# Quick Start Guide: Mixing-Length Theory for Convection

This guide will help you get started with using mixing-length theory (MLT) as an alternative to simple adiabatic corrections in your radiative transfer code.

## Installation

```bash
cd /Users/burt/Desktop/USM/Dynamics/ConvectionMLT
pip install -r requirements.txt
```

## Files Overview

| File | Purpose |
|------|---------|
| `mixing_length_theory.py` | Core MLT implementation (main module) |
| `explore_alpha_parameter.ipynb` | Interactive notebook for parameter space exploration |
| `example_rt_integration.py` | Example showing how to integrate MLT into RT code |
| `README.md` | Comprehensive documentation |
| `requirements.txt` | Python dependencies |

## Quick Test

Run the demo to verify installation:

```bash
python mixing_length_theory.py
```

Expected output: Calculation of optimal α ≈ 1.5 for test conditions.

## Basic Usage

### 1. Simple: Use Default α

```python
from mixing_length_theory import MixingLengthConvection

# Initialize
mlt = MixingLengthConvection(g=10.0, mu=0.0029, R=8.314)

# For each atmospheric layer:
T, P, cp = 1500, 1e5, 1000  # Your layer properties

# Calculate scale height and mixing length
H = mlt.pressure_scale_height(T, P)
l = mlt.mixing_length(alpha=1.5, H=H)  # α = 1.5 is typical

# Check if layer is convectively unstable
grad_ad = mlt.adiabatic_gradient(T, P, cp)
actual_grad = -0.012  # From your RT code (dT/dz)
is_unstable = mlt.schwarzschild_criterion(actual_grad, grad_ad)

if is_unstable:
    # Apply convective adjustment using MLT
    # (see example_rt_integration.py for details)
    pass
```

### 2. Advanced: Calculate Optimal α

```python
# For specific conditions, find best α:
alpha_optimal = mlt.find_alpha_for_adiabat(T, P, cp)
print(f"Optimal α = {alpha_optimal:.3f}")
```

### 3. Complete: Parameter Space Exploration

Run the Jupyter notebook to explore how α varies across your parameter space:

```bash
jupyter notebook explore_alpha_parameter.ipynb
```

This will generate:
- 3 heatmaps showing α(T,P), α(T,cp), α(P,cp)
- Statistical analysis
- Recommended α value for your application
- Saved results file (`alpha_parameter_space.npz`)

## Integration into Your RT Code

See `example_rt_integration.py` for a complete working example.

### Key Steps:

1. **Check for convective instability** (Schwarzschild criterion)
2. **Calculate mixing length** from α and pressure scale height
3. **Compute convective velocity** from superadiabaticity
4. **Calculate convective flux**
5. **Adjust temperature profile** based on convective transport

### Comparison with Old Method:

| Aspect | Old Method | MLT Method |
|--------|-----------|------------|
| Treatment | Force to exact adiabat | Physics-based adjustment |
| Parameters | None | α (mixing-length parameter) |
| Timescale | Instantaneous | Considers mixing timescale |
| Flux | Not calculated | Explicit convective flux |
| Flexibility | Fixed | Adjustable via α |

## Running the Example

```bash
python example_rt_integration.py
```

This demonstrates:
- Old adiabatic correction: ΔT = -20 K (forces exact adiabat)
- MLT treatment: ΔT = -0.32 K (gradual adjustment)

## Parameter Space Exploration Results

After running the notebook, you'll get recommendations like:

```
Recommended α value: 1.500 ± 0.150
Safe range: [1.350, 1.650]
```

### Using Results in Your Code:

**Option A:** Use constant α
```python
alpha = 1.5  # From exploration
```

**Option B:** Load parameter-dependent α
```python
import numpy as np
data = np.load('alpha_parameter_space.npz')
# Interpolate for your specific T, P, cp
```

## Adjusting Parameters

### For Your Specific Planet/Star:

Edit the initialization values:

```python
g = 24.8       # Gravitational acceleration [m/s²] for your object
mu = 0.0028    # Mean molecular weight [kg/mol] for your atmosphere
R = 8.314      # Universal gas constant (usually fixed)

mlt = MixingLengthConvection(g=g, mu=mu, R=R)
```

### For Your Parameter Ranges:

In the notebook, adjust:

```python
# Temperature range
T_min, T_max = 500, 3000  # K

# Pressure range  
P_min, P_max = 1e2, 1e7   # Pa

# Specific heat range
cp_min, cp_max = 500, 2000  # J/(kg·K)
```

## Typical α Values

From literature and this implementation:

- **Solar convection zone:** α ≈ 1.5 - 2.0
- **Hot Jupiters:** α ≈ 1.0 - 2.5
- **Brown dwarfs:** α ≈ 1.0 - 1.5
- **Terrestrial planets:** α ≈ 1.0 - 2.0

Your parameter space exploration will find the optimal value for your specific application.

## Physics Summary

### Key Equations:

1. **Mixing length:** l = α H
2. **Pressure scale height:** H = RT/(μg)
3. **Adiabatic gradient:** dT/dz = -g/cp
4. **Schwarzschild criterion:** Layer unstable if dT/dz < (dT/dz)_ad

### What α Represents:

α is the ratio of the mixing length (characteristic size of convective eddies) to the pressure scale height. It's a tuning parameter that encapsulates complex 3D hydrodynamics into a 1D model.

## Next Steps

1. ✅ Install dependencies
2. ✅ Run `python mixing_length_theory.py` to test
3. ✅ Adjust `g`, `mu` for your atmosphere
4. ✅ Run notebook to explore parameter space
5. ✅ Review `example_rt_integration.py`
6. ⬜ Integrate MLT into your RT code
7. ⬜ Validate against observations/detailed models
8. ⬜ Tune α if needed

## Troubleshooting

**Issue:** Import error for `mixing_length_theory`

**Solution:** Make sure you're in the correct directory or add it to your Python path:
```python
import sys
sys.path.append('/Users/burt/Desktop/USM/Dynamics/ConvectionMLT')
```

**Issue:** Notebook can't find module

**Solution:** Run Jupyter from the same directory:
```bash
cd /Users/burt/Desktop/USM/Dynamics/ConvectionMLT
jupyter notebook
```

**Issue:** α values seem unreasonable

**Solution:** Check your parameter ranges and physical constants (g, μ, R). Typical α should be 0.5-3.0.

## Questions?

- See `README.md` for detailed documentation
- Check `example_rt_integration.py` for implementation patterns
- Review literature cited in README for theoretical background

## Citation

If you use this in publications, consider citing key MLT references:
- Böhm-Vitense (1958) - Original MLT formulation
- Guillot (2010) - MLT for planetary atmospheres


