# Mixing-Length Theory Calibration

## Methodology

We calibrate the mixing-length parameter $\alpha$ for adiabatic atmospheric layers using standard mixing-length theory (MLT). For each temperature-pressure combination $(T, P)$, representing the center of a convective layer, we determine the $\alpha$ value that allows MLT to transport the required convective flux while maintaining a nearly adiabatic temperature gradient.

## Key Equations

### Pressure Scale Height
$$H_p = \frac{RT}{\mu g}$$

where:
- $R = 8.314$ J/(mol·K) is the universal gas constant
- $\mu$ is the mean molecular weight [kg/mol]
- $g$ is surface gravity [m/s²]

### Mixing Length
$$\ell = \alpha H_p$$

where $\alpha$ is the dimensionless mixing-length parameter.

### Adiabatic Gradient
$$\nabla_{\text{ad}} = \frac{R}{\mu c_p}$$

where $c_p$ is the specific heat capacity at constant pressure [J/(kg·K)].

### Assumed Temperature Gradient
$$\nabla = \nabla_{\text{ad}} \times (1 + \epsilon)$$

with $\epsilon = 0.001$ (0.1% superadiabatic), required to drive convection.

### MLT Convective Flux
$$F_c(\alpha) = \rho \, c_p \, T \, S(\alpha) \, (\nabla - \nabla_{\text{ad}})^{3/2}$$

where $\rho$ is density [kg/m³] and

$$S(\alpha) = \sqrt{\frac{g\delta}{8H_p}} \, \ell^2 = \sqrt{\frac{g\delta}{8H_p}} \, (\alpha H_p)^2$$

with $\delta = 1$ for an ideal gas.

### Calibration Condition
$$F_c(\alpha) = F_{\text{conv}}$$

We solve this equation for $\alpha$, where $F_{\text{conv}}$ is the convective flux requirement for each atmospheric regime.

## Parameter Space

**Temperature range:** 200 - 2000 K (20 points, linearly spaced)
**Pressure range:** 10⁻⁵ - 10³ bar (20 points, logarithmically spaced)
**Grid size:** 20 × 20 = 400 calculations per planet type

## Three Planetary Regimes

### Hot Jupiter (H₂-dominated)
- **Mean molecular weight:** $\mu = 0.0022$ kg/mol
- **Specific heat:** $c_p = 14000$ J/(kg·K)
- **Surface gravity:** $g = 10$ m/s²
- **Convective flux:** $F_{\text{conv}} = 5 \times 10^6$ W/m²
- **Adiabatic gradient:** $\nabla_{\text{ad}} = 0.27$

### Sub-Neptune (H₂/He mix)
- **Mean molecular weight:** $\mu = 0.0035$ kg/mol
- **Specific heat:** $c_p = 10000$ J/(kg·K)
- **Surface gravity:** $g = 15$ m/s²
- **Convective flux:** $F_{\text{conv}} = 1 \times 10^6$ W/m²
- **Adiabatic gradient:** $\nabla_{\text{ad}} = 0.24$

### Terrestrial (N₂/O₂, Earth-like)
- **Mean molecular weight:** $\mu = 0.029$ kg/mol
- **Specific heat:** $c_p = 1005$ J/(kg·K)
- **Surface gravity:** $g = 9.81$ m/s²
- **Convective flux:** $F_{\text{conv}} = 200$ W/m²
- **Adiabatic gradient:** $\nabla_{\text{ad}} = 0.29$

## Results Summary

| Planet Type | Mean α | Median α | Range |
|-------------|--------|----------|-------|
| Hot Jupiter | 5.35 | 0.36 | 0.002 - 114 |
| Sub-Neptune | 4.29 | 0.29 | 0.002 - 92 |
| Terrestrial | 0.23 | 0.015 | 0.0001 - 4.9 |

### Key Findings

- $\alpha$ varies over ~5 orders of magnitude across the T/P parameter space
- **Terrestrial atmospheres** have systematically **smaller** $\alpha$ values (median = 0.015) compared to gas giants (median ~ 0.3)
- This reflects:
	- Heavier molecular weight → larger $\nabla_{\text{ad}}$
	- Lower convective flux requirement
	- More efficient small-scale mixing in terrestrial atmospheres

### Physical Interpretation

The mixing length $\ell = \alpha H_p$ represents the characteristic distance over which convective eddies transport heat. Small $\alpha$ indicates efficient convection through small-scale mixing, while large $\alpha$ suggests larger convective cells relative to the atmospheric scale height.

---

**Date:** November 2025
**Location:** `/Users/burt/Desktop/USM/Dynamics/ConvectionMLT/`
**Related files:** 
- `alpha_hot_jupiter_comprehensive.png`
- `alpha_sub_neptune_comprehensive.png`
- `alpha_terrestrial_earth_comprehensive.png`


