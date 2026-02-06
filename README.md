# ConvectionMLT - Mixing Length Theory Convection Model

A 1D atmospheric column model implementing mixing length theory (MLT) for convective flux calculations in planetary atmospheres.

## Overview

This repository contains code for:
- Iterative calculation of convective flux in atmospheric layers
- Temperature-pressure profile evolution using mixing length theory
- Parameter sweeps and convergence analysis
- Stability analysis and boundary detection

## Key Components

### Main Scripts

- **`tp.py`** - Temperature-pressure profile computation (by Kevin Heng, 2014)
- **`convective_grid/convective_flux_v2.py`** - Convective flux solver with fixed timestepping
- **`convective_grid/convective_flux_v3.py`** - Convective flux solver with dynamic (per-layer) timestepping
- **`extra_scripts/collect_parameter_sweep_data.py`** - Parameter sweep data collection
- **`plot_convergence_vs_mixing_length.py`** - Convergence analysis plotting
- **`analyze_stability.py`** - Stability analysis tools
- **`debug_convective_flux.py`** - Debugging utilities

### Analysis Scripts

- **`analyze_guillot_layers.py`** - Analysis of Guillot profile convergence
- **`analyze_radiative_boundary.py`** - Radiative boundary analysis
- **`diagnose_guillot_gradient.py`** - Gradient diagnostics for Guillot profiles
- **`plot_convergence_before_after.py`** - Before/after convergence comparisons

### Notebooks

- **`parameter_analysis.ipynb`** - Main parameter analysis notebook
- **`load_sweep_data_example.ipynb`** - Example notebook for loading sweep data

### Documentation

- **`guillot_convergence_analysis.md`** - Analysis of Guillot profile convergence issues
- **`convective_grid/temperature_update_derivation.md`** - Detailed derivation of temperature update equation
- **`convective_grid/derivation_SI_units.txt`** - Unit derivations in SI units
- **`convective_grid/Overview.txt`** - Overview of convective flux grid implementation

## Physical Model

The model implements mixing length theory for convective flux:

```
F_conv = ρ × c_p × l² × sqrt(g/T) × (N - N_ad)^(3/2)
```

Where:
- `l = α × H_p` (mixing length)
- `H_p = RT/(μ·g)` (pressure scale height)
- `N = -dT/dz` (temperature gradient)
- `N_ad = g/c_p` (adiabatic temperature gradient)

All calculations use SI units internally with explicit unit conversions.

## Usage

### Example commands: convective_flux_v2.py and convective_flux_v3.py

Run from the repository root. Use `--no-prompt` to skip interactive prompts and use default parameters.

**Linear T-P profile (default):**
```bash
python convective_grid/convective_flux_v2.py --no-prompt
```

**Guillot T-P profile (default Guillot parameters):**
```bash
python convective_grid/convective_flux_v2.py --profile-type guillot --no-prompt
```

**Guillot profile with plots:**
```bash
python convective_grid/convective_flux_v2.py --profile-type guillot --no-prompt --plot
```

**v3 with dynamic timestepping (Guillot, gradient method):**
```bash
python convective_grid/convective_flux_v3.py --profile-type guillot --no-prompt --dt-method gradient
```

**v3 with fixed timestepping (same as v2 behaviour):**
```bash
python convective_grid/convective_flux_v3.py --profile-type guillot --no-prompt --dt 1.0
```

**Other options:** `--n-layers 50`, `--max-steps 10000`, `--tol 1e-4`, `--debug`, `--output results.csv`, `--plot-prefix my_run`

See individual script docstrings and notebooks for full usage.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Notes

- Large data files (`.npz`, `.json`) and plot outputs are excluded from git
- See `.gitignore` for excluded files
- Legacy code is preserved in the `legacy/` directory
