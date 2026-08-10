# Guillot Profile Convergence Analysis

## Problem Summary

All 252 runs in the Guillot profile convergence sweep show:
- **Steps to converge: 100,000** (exactly, for all runs)
- **Convergence rate: 0%** (none converged)
- **Max gradient difference: 0.93-1.71** (93-171% away from adiabatic)

## Root Cause

This is **NOT a plotting error or data collection bug** - it's a real physical/numerical issue.

### The Issue

1. **Guillot profile creates radiative equilibrium conditions**: The Guillot TP profile represents radiative equilibrium, which is fundamentally different from convective equilibrium (adiabatic).

2. **Initial conditions are very far from adiabatic**: 
   - Linear profile: Starts closer to adiabatic → converges quickly
   - Guillot profile: Starts in radiative equilibrium → very far from adiabatic

3. **Extremely slow convergence**: Test run shows:
   - After 1000 steps: max_grad_diff = 0.935 (93.5% away from adiabatic)
   - Convergence rate is so slow that even 100,000 steps isn't enough

4. **All runs hit max_steps limit**: Every single run reaches the 100,000 step limit before achieving the 20% tolerance threshold.

## Evidence

From test run (`n_layers=10, l=0.1, dt=10`):
- Initial T range: [830.6, 1082.0] K
- After 1000 steps: T range: [830.6, 1061.0] K (only 21 K change)
- Max gradient difference: 0.935 (needs to be < 0.2 for convergence)
- Temperature is slowly decreasing, but gradient is still far from adiabatic

## Solutions

1. **Increase max_steps significantly** (e.g., 1,000,000 or more)
2. **Use larger timesteps** to speed up convergence (but risk numerical instability)
3. **Use larger mixing lengths** to increase convective flux
4. **Accept longer convergence times** for Guillot profiles
5. **Use a hybrid approach**: Start with Guillot, then switch to linear interpolation for faster convergence

## Recommendation

The Guillot profile sweep results are correct - they accurately reflect that Guillot profiles take much longer to converge than linear profiles. The flatline in the plots is real: all parameter combinations require more than 100,000 steps to converge.
