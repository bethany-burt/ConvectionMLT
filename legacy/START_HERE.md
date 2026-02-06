# Start Here: MLT Alpha from Flux Balance

## What You Have

A complete implementation to calculate the mixing-length parameter **α** by solving:

**F_c(α) = F_need = F_tot - F_rad**

This determines α based on the energy flux that must be carried by convection in your radiative transfer code.

## The 30-Second Test

```bash
cd /Users/burt/Desktop/USM/Dynamics/ConvectionMLT
python mlt_flux_balance.py
```

You should see:
```
α = 0.0013
l = 737.37 m
F_c = 2.00e+06 W/m²
```

If this works, you're ready to go!

## Quick Usage

```python
from mlt_flux_balance import calculate_alpha_from_flux

# Your layer data (top, middle, bottom of layer)
layer_data = {
    'T_top': 1000.0, 'T_mid': 1500.0, 'T_bot': 2250.0,  # K
    'P_top': 1e4, 'P_mid': 5e4, 'P_bot': 1e5,  # Pa
}

# Flux from your RT code
flux_data = {
    'F_tot': 1e7,  # W/m² - total flux
    'F_rad': 5e6,  # W/m² - radiative part
}

# Your atmosphere's properties
physical_params = {
    'g': 10.0,           # m/s²
    'delta': 1.0,        # ideal gas
    'R_universal': 8.314, # J/(mol·K)
    'mu': 0.0022,        # kg/mol
    'c_p': 14000.0,      # J/(kg·K)
    'rho': 0.005,        # kg/m³
}

# Solve!
result = calculate_alpha_from_flux(layer_data, flux_data, 
                                   physical_params, verbose=True)

if result['alpha'] is not None:
    print(f"α = {result['alpha']:.4f}")
    print(f"Mixing length = {result['alpha'] * result['H_p']:.2f} m")
else:
    print(f"No convection: {result['convergence_info']}")
```

## File Guide

| File | What It Does | When to Use |
|------|--------------|-------------|
| **mlt_flux_balance.py** | Core calculation | Import this in your RT code |
| **test_single_layer.py** | 4 test cases | Verify everything works |
| **explore_tp_parameter_space.py** | Scan T/P space | Understand α behavior |
| **explore_mlt_alpha.ipynb** | Interactive notebook | Explore & visualize |
| **README_flux_balance.md** | Full documentation | Learn the physics |
| **IMPLEMENTATION_COMPLETE.md** | What was built | See complete overview |

## Three Usage Modes

### Mode 1: Direct Calculation (Simplest)

```python
from mlt_flux_balance import calculate_alpha_from_flux

# Your values here
result = calculate_alpha_from_flux(layer, flux, params, verbose=True)
alpha = result['alpha']
```

**Use when**: You have one layer to test

### Mode 2: Test Suite (Validation)

```bash
python test_single_layer.py
```

**Use when**: You want to verify the code works correctly

### Mode 3: Parameter Space (Understanding)

```bash
python explore_tp_parameter_space.py
# or
jupyter notebook explore_mlt_alpha.ipynb
```

**Use when**: You want to explore how α varies across conditions

## The Key Physics

### What This Does

1. Checks if layer is **superadiabatic**: ∇ > ∇_ad (Schwarzschild criterion)
2. If yes, calculates how much flux needs convection: **F_need = F_tot - F_rad**
3. Solves for α that makes MLT give exactly that flux: **F_c(α) = F_need**
4. Returns α, mixing length l = α·H_p, and convective flux

### Why This Matters

- **Energy conserving**: Explicitly balances radiation + convection
- **Physical basis**: Uses standard MLT theory
- **Adjustable**: α responds to actual flux requirements
- **Better than forcing adiabat**: Doesn't arbitrarily set ∇ = ∇_ad

## Typical Results

For hot Jupiter atmospheres with high flux needs:
- **α ≈ 0.001 - 0.01** (small-scale efficient mixing)

For Earth-like or solar conditions:
- **α ≈ 1.0 - 2.0** (typical convection)

For stable layers:
- **α = None** (no convection needed)

## Integration Checklist

- [ ] Test runs successfully: `python mlt_flux_balance.py`
- [ ] All tests pass: `python test_single_layer.py`  
- [ ] Adjust parameters for your atmosphere
- [ ] Test with one of your RT layers
- [ ] Verify F_c matches F_need
- [ ] Integrate into your RT loop
- [ ] Validate against observations/models

## Need Help?

1. **Code doesn't run?** Check Python packages: `pip install numpy scipy matplotlib`
2. **No superadiabatic cases?** Check ∇ > ∇_ad. Adjust T gradient or c_p
3. **α values seem odd?** Read `README_flux_balance.md` section on typical values
4. **Want to understand more?** Open `explore_mlt_alpha.ipynb`

## What's Next?

1. **Test**: Run `python mlt_flux_balance.py` ✓
2. **Customize**: Edit parameters in `test_single_layer.py` for your atmosphere
3. **Explore**: Run parameter space scan or notebook
4. **Integrate**: Add to your RT code (see `README_flux_balance.md`)
5. **Validate**: Compare with your current method or observations

## Questions to Answer

Before integrating, consider:

1. **What are my typical T/P profiles?** → Run parameter space scan
2. **What F_need do I expect?** → Test sensitivity in notebook
3. **What physical parameters?** → Adjust g, μ, c_p for your atmosphere
4. **Single α or vary per layer?** → Explore to see variation

## Summary

You now have:
- ✓ Working flux balance calculation
- ✓ Tested on 4 different cases
- ✓ Parameter space exploration tools
- ✓ Interactive notebook
- ✓ Complete documentation
- ✓ Integration guide

**Ready to use in your radiative transfer code!**

---

**Quick links:**
- Full docs: `README_flux_balance.md`
- Physics background: Read "Flux Balance Equation" section in README
- Integration example: See "Integration into RT Code" section
- Troubleshooting: See "Troubleshooting" section in README

*Last updated: November 10, 2025*




