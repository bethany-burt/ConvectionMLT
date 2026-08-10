# Temperature Update Equation - Explicit Derivation

## Physical Principle: Energy Conservation

Energy conservation in a volume element states that the rate of change of internal energy equals the net energy flux into/out of the volume.

## Step-by-Step Derivation

### 1. Energy Density (Energy per Unit Volume)

The internal energy per unit volume is:
```
E = ρ × c_p × T
```

**Units:**
- ρ: kg/m³ (mass per unit volume)
- c_p: J/(kg·K) (energy per unit mass per unit temperature)
- T: K (temperature)

**Unit check:**
```
E = (kg/m³) × (J/(kg·K)) × (K)
  = (kg/m³) × (kg·m²/s²)/(kg·K) × (K)
  = (kg/m³) × (m²/s²)
  = kg·m²/(m³·s²)
  = kg/(m·s²)
```

Wait, this doesn't look right. Let me reconsider...

Actually, the internal energy per unit volume for an ideal gas is:
```
E = ρ × u
```

where u is specific internal energy. For a calorically perfect gas:
```
u = c_v × T
```

But we're working with c_p. The relationship is more complex, but for our purposes, the energy change per unit volume when temperature changes is:
```
dE = ρ × c_p × dT
```

This gives us the energy density change, not the absolute energy density. Let's work with the rate of change:

### 2. Rate of Change of Energy Density

The energy change per unit volume when temperature changes is:
```
dE = ρ × c_p × dT
```

Taking the time derivative:
```
dE/dt = ρ × c_p × dT/dt
```

**Units:**
- ρ: kg/m³
- c_p: J/(kg·K) = (kg·m²/s²)/(kg·K) = m²/(s²·K)
- dT/dt: K/s

**Explicit unit check:**
```
dE/dt = (kg/m³) × (m²/(s²·K)) × (K/s)
      = (kg/m³) × (m²/s²) × (K/s) × (1/K)
      = (kg/m³) × (m²/s²) × (1/s)
      = kg·m²/(m³·s³)
      = kg/(m·s³)
```

**Physical meaning:** Energy density rate of change has units of energy per unit volume per unit time.
- Energy: J = kg·m²/s²
- Volume: m³
- Time: s
- Energy density rate: (kg·m²/s²)/(m³·s) = kg/(m·s³) ✓

### 3. Energy Flux Divergence

In 1D (vertical direction z), the energy flux divergence is:
```
∇·F = dF/dz
```

**Units:**
- F: Energy flux = W/m² = J/(m²·s) = (kg·m²/s²)/(m²·s) = kg/(m·s³)
- z: m
- dF/dz: (kg/(m·s³))/m = kg/(m²·s³)

Wait, let me recalculate F_conv units more carefully:

**F_conv units (from our derivation):**
```
F_conv = ρ × c_p × l² × sqrt(g/T) × (N - N_ad)^(3/2)
```

We derived this gives: kg/s³

But energy flux should be energy per area per time:
- Energy: J = kg·m²/s²
- Area: m²
- Time: s
- Energy flux: J/(m²·s) = (kg·m²/s²)/(m²·s) = kg/s³ ✓

So F_conv = kg/s³ = J/(m²·s) = W/m² ✓

**Now dF/dz:**
- F: W/m² = J/(m²·s) = (kg·m²/s²)/(m²·s) = kg/s³
- z: m
- dF/dz: (J/(m²·s))/m = J/(m³·s) = W/m³ = kg/(m·s³)

### 4. Energy Conservation Equation

Energy conservation states:
```
dE/dt = -∇·F
```

**Sign convention explanation:**
- Positive flux divergence (dF/dz > 0) means more energy flows out than in → energy decreases
- Negative flux divergence (dF/dz < 0) means more energy flows in than out → energy increases
- The negative sign ensures: when dF/dz > 0, dE/dt < 0 (energy decreases)

Substituting our expressions:
```
ρ × c_p × dT/dt = -dF/dz
```

**Unit check:**
- Left side: kg/(m·s³) (from step 2)
- Right side: kg/(m·s³) (from step 3, with negative sign)
- ✓ Units match!

### 5. Solve for dT/dt

```
dT/dt = -(1/(ρ × c_p)) × dF/dz
```

**Explicit unit check:**
- 1/(ρ × c_p): 1/((kg/m³) × (m²/(s²·K))) = 1/(kg·m²/(m³·s²·K)) = m³·s²·K/(kg·m²) = m·s²·K/kg
- dF/dz: kg/(m·s³)

**Result:**
```
dT/dt = (m·s²·K/kg) × (kg/(m·s³))
      = K/s ✓
```

## Final Equation

```
dT/dt = -(1/(ρ × c_p)) × dF/dz
```

**In discrete form (for numerical integration):**
```
ΔT = -Δt × (1/(ρ × c_p)) × (dF/dz)
```

**Where:**
- ΔT: Temperature change (K)
- Δt: Time step (s)
- ρ: Density (kg/m³)
- c_p: Specific heat capacity (J/(kg·K))
- dF/dz: Flux divergence (W/m³ = J/(m³·s))

## Unit Verification Summary

| Term | Units | Expanded | Notes |
|------|-------|----------|-------|
| ρ | kg/m³ | kg/m³ | Mass density |
| c_p | J/(kg·K) | m²/(s²·K) | Specific heat capacity |
| dT/dt | K/s | K/s | Temperature rate of change |
| F | W/m² | J/(m²·s) = kg/s³ | Energy flux |
| dF/dz | W/m³ | J/(m³·s) = kg/(m·s³) | Flux divergence |
| ρ·c_p | J/(m³·K) | kg·m²/(m³·s²·K) = kg/(m·s²·K) | Energy density per K |
| 1/(ρ·c_p) | m·s²·K/kg | m·s²·K/kg | Inverse energy density per K |
| (1/(ρ·c_p)) × dF/dz | K/s | (m·s²·K/kg) × (kg/(m·s³)) = K/s ✓ | Final result |

## Explicit Final Verification

Starting from:
```
dT/dt = -(1/(ρ × c_p)) × dF/dz
```

**Step 1: Calculate 1/(ρ × c_p)**
```
ρ × c_p = (kg/m³) × (m²/(s²·K))
        = kg·m²/(m³·s²·K)
        = kg/(m·s²·K)

1/(ρ × c_p) = 1/(kg/(m·s²·K))
            = m·s²·K/kg
```

**Step 2: Calculate dF/dz**
```
F = W/m² = J/(m²·s) = (kg·m²/s²)/(m²·s) = kg/s³

dF/dz = (kg/s³)/m = kg/(m·s³)
```

**Step 3: Multiply**
```
dT/dt = -(m·s²·K/kg) × (kg/(m·s³))
      = -K/s ✓
```

**Result:** Temperature changes at a rate of K/s, which is correct!

## Conversion to CGS Units (for reference)

If working in CGS:
- ρ: g/cm³
- c_p: erg/(g·K)
- dF/dz: erg/(cm³·s)
- 1/(ρ·c_p): cm³·K/erg
- Result: (cm³·K/erg) × (erg/(cm³·s)) = K/s ✓

Same result, different unit system.
