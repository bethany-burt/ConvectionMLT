# MLT Alpha from Flux Balance

This implementation determines the mixing-length parameter α by solving the flux balance equation:

**F_c(α) = F_need = F_tot - F_rad**

This approach directly links MLT to the energy transport requirements in your radiative transfer code.

## Quick Start

### 1. Single Layer Calculation

```python
from mlt_flux_balance import calculate_alpha_from_flux

# Define your layer
layer_data = {
    'T_top': 1000.0, 'T_mid': 1500.0, 'T_bot': 2250.0,  # K
    'P_top': 1e4, 'P_mid': 5e4, 'P_bot': 1e5,  # Pa
}

# Flux requirements
flux_data = {
    'F_tot': 1e7,  # W/m² - total flux through layer
    'F_rad': 5e6,  # W/m² - radiative component
}

# Physical parameters
physical_params = {
    'g': 10.0,           # m/s²
    'delta': 1.0,        # dimensionless (1 for ideal gas)
    'R_universal': 8.314, # J/(mol·K)
    'mu': 0.0022,        # kg/mol
    'c_p': 14000.0,      # J/(kg·K)
    'rho': 0.005,        # kg/m³
}

# Calculate alpha
result = calculate_alpha_from_flux(layer_data, flux_data, 
                                   physical_params, verbose=True)

print(f"α = {result['alpha']:.4f}")
print(f"l = {result['alpha'] * result['H_p']:.2f} m")
```

### 2. Test Multiple Cases

```bash
python test_single_layer.py
```

Runs 4 preset test cases:
- Superadiabatic layer (hot Jupiter)
- Stable layer (no convection)
- High flux requirement
- Earth-like atmosphere

### 3. Explore Parameter Space

```bash
python explore_tp_parameter_space.py
```

Generates heatmaps showing how α varies across T/P profiles.

### 4. Interactive Exploration

```bash
jupyter notebook explore_mlt_alpha.ipynb
```

Full interactive notebook with:
- Adjustable parameters
- Flux sensitivity analysis
- Parameter space scans
- Visualization
- Export for RT code

## Physics Background

### The Flux Balance Equation

In a convectively unstable layer, energy is transported by both radiation and convection:

**F_total = F_rad + F_conv**

If your RT code computes F_rad and you know F_total, then:

**F_need = F_total - F_rad**

This is the flux that must be carried by convection.

### MLT Convective Flux

The mixing-length theory expression for convective flux is:

**F_c(α) = ρ c_p T S(α) (∇ - ∇_ad)^(3/2)**

where:
- **S(α) = √(gδ/(8H_p)) · l²** with **l = α H_p**
- **∇ = d(ln T)/d(ln P)** - actual temperature gradient
- **∇_ad = R/c_p** - adiabatic gradient
- **H_p = RT/(μg)** - pressure scale height

### Solving for α

Setting F_c(α) = F_need gives us an equation to solve for α:

**ρ c_p T √(gδ/(8H_p)) (α H_p)² (∇ - ∇_ad)^(3/2) = F_need**

This is solved numerically using root-finding (Brent's method) with bounds α ∈ [0.001, 10].

## Key Equations

```
# Check if superadiabatic (Schwarzschild criterion)
∇ = d(ln T) / d(ln P)    # from finite differences on layer
∇_ad = R / c_p
Superadiabatic if: ∇ > ∇_ad

# Pressure scale height
H_p = (R * T) / (μ * g)

# MLT flux formula
S(α) = sqrt(g * δ / (8 * H_p)) * (α * H_p)²
F_c(α) = ρ * c_p * T * S(α) * (∇ - ∇_ad)^(3/2)

# Solve for α
F_c(α) = F_need
```

## Files

| File | Purpose |
|------|---------|
| `mlt_flux_balance.py` | Core module with flux balance calculations |
| `test_single_layer.py` | Test script with 4 preset cases |
| `explore_tp_parameter_space.py` | Parameter space exploration with heatmaps |
| `explore_mlt_alpha.ipynb` | Interactive Jupyter notebook |
| `README_flux_balance.md` | This file |

## Integration into RT Code

### Workflow

For each atmospheric layer in your RT calculation:

1. **Check stability**: Calculate ∇ and ∇_ad, check if ∇ > ∇_ad
2. **Calculate F_need**: F_need = F_tot - F_rad (from your RT solver)
3. **Solve for α**: Use `calculate_alpha_from_flux()`
4. **Get mixing length**: l = α × H_p
5. **Apply adjustment**: Use F_c(α) to adjust T/P profile

### Example Integration

```python
# In your RT code's convection module:

from mlt_flux_balance import calculate_alpha_from_flux

def apply_convection_mlt(layer, F_tot, F_rad, physical_params):
    """
    Apply MLT convective adjustment to a layer.
    """
    # Prepare layer data
    layer_data = {
        'T_top': layer.T_top,
        'T_mid': layer.T_mid,
        'T_bot': layer.T_bot,
        'P_top': layer.P_top,
        'P_mid': layer.P_mid,
        'P_bot': layer.P_bot,
    }
    
    flux_data = {
        'F_tot': F_tot,
        'F_rad': F_rad,
    }
    
    # Calculate alpha
    result = calculate_alpha_from_flux(layer_data, flux_data, 
                                      physical_params, verbose=False)
    
    if result['alpha'] is not None:
        # Apply convective adjustment
        alpha = result['alpha']
        l = alpha * result['H_p']
        F_c = result['F_c']
        
        # Update temperature profile based on convective transport
        # (implementation depends on your RT code structure)
        layer.apply_convective_adjustment(alpha, l, F_c)
        
        return alpha, l, F_c
    else:
        # No convection needed
        return None, None, 0.0
```

## Parameter Ranges

### Typical α Values

Based on parameter space explorations:

| Atmosphere Type | α Range | Reference |
|----------------|---------|-----------|
| Solar convection zone | 1.5 - 2.0 | Calibrated |
| Hot Jupiters | 0.001 - 0.01 | This work (high F_need) |
| Brown dwarfs | 1.0 - 1.5 | Literature |
| Earth troposphere | 1.0 - 2.0 | Typical |

**Note**: The α values from this flux balance approach can be quite small (< 1) when F_need is large compared to the maximum convective efficiency. This is physically meaningful - it indicates that even small-scale mixing can transport the required flux when the layer is sufficiently superadiabatic.

### Physical Parameters

Adjust these for your specific atmosphere:

```python
physical_params = {
    'g': 10.0,           # Your planet's surface gravity [m/s²]
    'delta': 1.0,        # 1 for ideal gas; varies for real gases
    'R_universal': 8.314, # Universal gas constant [J/(mol·K)]
    'mu': 0.0022,        # Mean molecular weight [kg/mol] - composition dependent
    'c_p': 14000.0,      # Specific heat [J/(kg·K)] - composition dependent
    'rho': 0.005,        # Density [kg/m³] - from your RT grid
}
```

## Validation

### Test Cases Provided

1. **Superadiabatic layer**: Should find α solution
2. **Stable layer**: Should return None (no convection)
3. **High flux**: Should find α, possibly at upper bound
4. **Earth-like**: Realistic atmospheric conditions

Run `python test_single_layer.py` to verify all cases work correctly.

### Expected Behavior

- **∇ > ∇_ad**: Finds α such that F_c(α) = F_need
- **∇ ≤ ∇_ad**: Returns α = None (stable, no convection)
- **F_need ≤ 0**: Returns α = None (radiative transport sufficient)
- **F_need very large**: α may hit upper bound (10.0), warns user

## Differences from Simple Adiabatic Forcing

| Aspect | Old (Force Adiabat) | New (MLT Flux Balance) |
|--------|---------------------|------------------------|
| Method | Set ∇ = ∇_ad instantly | Solve F_c(α) = F_need |
| Energy | Not conserved | Explicitly balanced |
| Parameter | None | α (physical meaning) |
| Flexibility | Fixed correction | Adjusts to flux needs |
| Physics | Phenomenological | Theory-based (MLT) |

## Troubleshooting

### No superadiabatic cases found

- **Check temperature gradients**: Ensure ∇ > ∇_ad
- **Verify c_p value**: Too large c_p → large ∇_ad → stable
- **Check layer structure**: Sufficient ΔT between top/bottom?

### α values too small (< 0.01)

- **High F_need**: Large required flux → efficient small-scale mixing
- **Very superadiabatic**: Large (∇ - ∇_ad) → less mixing length needed
- **This is physical**: Not necessarily a problem

### α at upper bound (10.0)

- **Insufficient convection**: Even large α can't provide F_need
- **Check physical parameters**: Verify ρ, c_p, T are reasonable
- **Consider**: May need larger F_rad (more radiative transport)

## References

- **Böhm-Vitense (1958)**: Original MLT formulation
- **Henyey et al. (1965)**: Stellar structure applications
- **Cox & Giuli (1968)**: "Principles of Stellar Structure" - MLT chapter
- **Guillot (2010)**: MLT for planetary atmospheres
- **Kippenhahn & Weigert (1990)**: "Stellar Structure and Evolution"

## Citation

If you use this in publications, please cite the relevant MLT papers above and acknowledge this implementation.

## Support

For questions or issues:
1. Check test cases work: `python test_single_layer.py`
2. Run parameter space exploration: `python explore_tp_parameter_space.py`
3. Try interactive notebook: `jupyter notebook explore_mlt_alpha.ipynb`
4. Verify physical parameters match your atmosphere

---

*Implementation Date: November 2025*
*Based on standard MLT formulations for atmospheric convection*




