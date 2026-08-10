# Frozen legacy comparison

The files under `Legacy/` preserve the previous experiment for provenance only.
They are not a physics or implementation source for R0.

Frozen baseline command:

```bash
python Legacy/convective_grid/convective_flux_v3.py --no-prompt
```

The R0 reference kernel intentionally differs as required by
`HELIOS_VULCAN_MLT_Project_Plan.pdf`:

- temperature and thermodynamics are at N layer centres; flux is at N+1 edges;
- pressure edges are indexed bottom-to-top and the conservative update uses
  `dT_i/dt = (F_e,i - F_e,i+1)/(cp Delta_m_i)`;
- gradients and interface interpolation use log pressure, not an altitude
  finite difference;
- calorically perfect H2 uses `cp = (7/2) R_specific`, not the legacy
  degree-of-freedom expression that omitted the one-half factor;
- the selected closure uses the documented 1/2 prefactor and records it
  independently of alpha;
- every step is global and conservative; no per-layer timestep, clipping, or
  Guillot-profile validation is retained.

Consequently, numerical agreement with v2/v3 is not an acceptance criterion.
The comparison requirement is satisfied by preserving this invocation and
explaining each expected difference from the corrected discretisation.
