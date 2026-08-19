# Stage 2 Validation Plotting Plan

## Purpose

This document defines the plots required to demonstrate that Stage 2 satisfies its scientific, numerical, and implementation exit gates. The evidence set must distinguish between:

- thermodynamic and hydrostatic identities established by deterministic tests;
- physical relaxation demonstrated by representative profiles and histories;
- robustness demonstrated by the completed production campaign;
- diagnostics that are informative but are not formal pass/fail gates.

The completed production campaign contains 27 converged cases and no failures. Of these, 24 are unique constant-gravity parameter combinations, two are inverse-square-gravity cases, and one repeats the regular-grid, pure-H2, N = 50 constant-gravity baseline. The repeat should be used as a reproducibility check but counted only once in the parameter matrix.

The production JSON contains terminal case summaries only. It can support the campaign-level figures described below, but detailed profiles, time histories, thermodynamic identities, and hydrostatic refinement results must be taken from separate test or diagnostic outputs.

## Stage 2 exit claims

The final evidence set should support the following statement:

> The solver reaches a grid-independent variable-thermodynamic isentrope and conserves column enthalpy under constant gravity. Its thermodynamic providers satisfy their defining identities, while hydrostatic reconstruction and inverse-square gravity satisfy analytic, round-trip, refinement, and constant-gravity-limit checks.

Variable-gravity enthalpy change is diagnostic only. Because the pressure-layer mass paths change as the hydrostatic structure and gravity are recomputed, Stage 2 does not claim strict conservation for those cases.

---

## Figure 01 — Thermodynamic provider audit

### Purpose and gate

This figure proves that the Stage 2 thermodynamic providers are internally consistent. Production convergence alone cannot prove that the implemented heat capacity, enthalpy, entropy, mixture rules, or inverse functions are correct.

### Required panels

1. Molar or mass-specific cp(T) for NASA H2, exact monatomic He, and the production H2/He mixtures.
2. Adiabatic gradient, nabla_ad(T) = R_mix / cp_mix(T).
3. Normalized derivative residual, |dh/dT - cp| / cp.
4. T -> h -> T inversion error and, if used by the solver, T -> Psi -> T entropy-function inversion error.

### Cases and annotations

- Show x_He = 0, 0.1, and 0.25.
- Mark every NASA validity boundary and the 1000 K polynomial breakpoint.
- Mark the breakpoint-continuity tolerance and the hard temperature-domain limits.
- State units and the common enthalpy reference temperature.

### Required interpretation

Each provider must satisfy its own identities. Agreement between the analytic and NASA providers is not itself an exit requirement. The analytic provider is a manufactured-test oracle; the NASA provider is the production model.

### Data source

This cannot be generated from production_campaign.json. Export the deterministic thermodynamic audit values or evaluate the checked-in providers directly using the same public API exercised by the tests.

---

## Figure 02 — Representative numerical isentrope

### Purpose and gate

This is the primary physical demonstration of Stage 2. It must show that an initially superadiabatic column relaxes to the variable-cp equilibrium defined by constant entropy, rather than merely reaching a terminal status.

### Recommended case

Use a well-resolved constant-gravity production case, preferably N = 100 with NASA H2 thermodynamics. A mixture case such as x_He = 0.1 may be added, but one uncluttered reference case is sufficient for the main figure.

### Required panels

1. Initial, final, and enthalpy-normalized isentropic-reference T(P).
2. Final residual T_final - T_isentrope versus pressure.
3. Final entropy or generalized potential temperature versus pressure.
4. Evolution of entropy span, maximum positive finite-layer superadiabaticity, and maximum convective flux.

### Required interpretation

The figure should demonstrate

- entropy becoming vertically uniform;
- final temperature approaching the enthalpy-normalized numerical isentrope;
- positive superadiabaticity falling below its stopping tolerance;
- convective flux tending to zero as equilibrium is approached.

The final state need not retain active convection. The equilibrium isentrope is the outcome of the preceding convective redistribution.

### Data source

The summary JSON supplies only the final RMS temperature error and final maximum superadiabaticity. Save or rerun one representative case with pressure profiles and time histories enabled.

---

## Figure 03 — Production robustness matrix

### Purpose and gate

This is the principal campaign-level exit-gate summary. It proves that all tested resolutions, compositions, and grid placements reach accepted equilibria while satisfying the constant-gravity conservation requirement.

### Rows

Use the 24 unique constant-gravity combinations:

- N = 25, 50, 100, and 200;
- x_He = 0, 0.1, and 0.25;
- regular and irregular pressure grids.

Treat the repeated N = 50, x_He = 0, regular-grid baseline separately as a reproducibility annotation.

### Columns

1. Terminal status.
2. Normalized enthalpy-drift score.
3. Normalized isentrope-RMS score.
4. Normalized maximum-superadiabaticity score.
5. Accepted step count as an unscored diagnostic.

For every gated metric, plot S = observed value / configured tolerance.

### Presentation rules

- Use green for S < 1 and red only for S >= 1.
- Print at least four significant digits.
- Do not round near-threshold superadiabaticity scores to an ambiguous value of 1.
- Use N/A rather than a failure colour where a metric is not applicable.
- Keep step count visually separate from pass/fail colours.

### Campaign result to expose

Across the constant-gravity cases, the terminal values are:

- relative enthalpy drift: 1.68e-16 to 5.01e-15;
- isentrope temperature RMS: approximately 4.91e-9 to 5.17e-9;
- maximum positive superadiabaticity: 9.959e-9 to 9.99995e-9.

The superadiabaticity values intentionally finish close to the 1e-8 stopping threshold. This reflects termination at the requested tolerance, not marginal physical instability.

---

## Figure 04 — Resolution, composition, and grid robustness

### Purpose and gate

This figure demonstrates that the accepted equilibrium is insensitive to numerical resolution, H2/He composition, and pressure-grid placement over the production campaign.

### Required panels

1. Final isentrope RMS error versus N.
2. Final maximum positive superadiabaticity versus N.
3. Relative enthalpy drift versus N for constant gravity.

### Visual encoding

- Colour: x_He = 0, 0.1, and 0.25.
- Filled markers and solid lines: regular grid.
- Open markers and dashed lines: irregular grid.
- Horizontal lines: configured tolerances.

### Required interpretation

The regular- and irregular-grid equilibrium results differ by at most about 0.11 percent. This is strong evidence that grid placement does not bias the accepted equilibrium.

The equilibrium metrics should not be described as spatial-convergence errors because they do not decrease with N. They remain approximately constant because they are controlled by fixed stopping tolerances. The correct claim is tolerance-limited grid independence.

---

## Figure 05 — Constant-gravity enthalpy conservation

### Purpose and gate

This figure proves the strict closed-column conservation claim under constant gravity.

### Required panels

1. Terminal relative enthalpy drift versus N for every composition and grid placement.
2. The same results normalized by the configured conservation tolerance.
3. If time-history data are available, representative |H(t) - H0| / H_scale curves versus accepted step or physical time.
4. If available, a normalized one-step flux-telescoping residual.

### Required interpretation

All unique constant-gravity production cases finish between approximately 1.68e-16 and 5.01e-15 relative drift. If the final configured gate is 1e-12, the full campaign passes by a large margin.

The terminal JSON demonstrates end-to-end budget closure but not explicitly that the budget remained bounded at every intermediate time. A representative H(t) history should therefore be included when the documented gate refers to conservation throughout relaxation.

Do not mix inverse-square apparent enthalpy drift into the formal conservation panel.

---

## Figure 06 — Hydrostatic verification

### Purpose and gate

This figure independently validates the pressure-height mapping. Solver convergence and a self-consistent P -> z -> P round trip are not sufficient by themselves, because matching forward and inverse errors could cancel.

### Required comparisons

1. Constant-gravity isothermal solution against its analytic height profile.
2. Inverse-square-gravity isothermal solution against its analytic height profile.
3. Nonisothermal manufactured profile against an independent high-accuracy integration.
4. Pressure round-trip error using the documented forward and inverse conventions.
5. Error versus N for the analytic and independent-reference comparisons.

### Required interpretation

Show both the absolute or relative error profiles and refinement behaviour. Retain any small-amplitude case used by CI, but also include a stronger nonisothermal stress profile. For the stronger case, convergence with refinement is more meaningful than forcing it below a small-amplitude absolute threshold.

### Data source

These quantities are absent from production_campaign.json. Generate them from the hydrostatic validation driver or export the detailed results produced by the hydrostatic tests.

---

## Figure 07 — Inverse-square gravity sweep and scope diagnostic

### Purpose and gate

This figure shows that the inverse-square hydrostatic plumbing converges thermodynamically and clarifies the physical regime of the two completed gravity cases. It must also distinguish valid Stage 2 diagnostics from claims that have deliberately not been made.

### Required panels from the current JSON

1. Maximum z/Rp versus planet radius.
2. Final isentrope RMS and maximum positive superadiabaticity versus planet radius.
3. Apparent enthalpy drift versus maximum z/Rp.

### Current results

- Rp = 1e7 m: max(z/Rp) = 1.419, isentrope RMS = 4.39e-9, apparent drift = 3.55e-3.
- Rp = 1e8 m: max(z/Rp) = 0.0632, isentrope RMS = 4.87e-9, apparent drift = 2.22e-4.

### Required interpretation

Both cases reach the entropy-based equilibrium tolerance. However, max(z/Rp) = 1.419 is far outside the thin, plane-parallel regime. The Rp = 1e7 m case should therefore be labelled an extreme hydrostatic stress test, not a physically self-consistent plane-parallel planetary atmosphere.

The apparent enthalpy drift must be labelled:

> Diagnostic only: pressure-layer mass paths change under evolving inverse-square gravity; strict conservation is not claimed.

### Additional evidence needed for the original gravity gate

The present campaign has only two finite-radius cases and no direct profile differences relative to constant gravity. To demonstrate the constant-gravity limit rigorously, add larger radii or use dedicated hydrostatic outputs to plot:

- profile differences in z(P), g(P), pressure scale height, and mass path;
- Delta g/g against z/Rp, with the small-height prediction Delta g/g approximately -2z/Rp;
- monotonic convergence toward the constant-gravity solution as z/Rp approaches zero.

Do not claim that the constant-gravity-limit sweep has been visually established using only the two summary points currently available.

---

## Figure 08 — Accepted-step scaling

### Purpose

This is a performance and numerical-consistency diagnostic rather than a physical exit gate. It demonstrates that the explicit relaxation cost behaves as expected as resolution increases.

### Required plot

Plot accepted step count versus N on log-log axes, separated by composition and grid type. Include a fitted power law and a reference N^2 line.

For the regular, pure-H2 sequence, the production campaign reports:

- N = 25: 2,847 steps;
- N = 50: 10,624 steps;
- N = 100: 41,356 steps;
- N = 200: 163,174 steps.

The fitted relation is approximately n_steps proportional to N^1.95, consistent with the expected N^2 explicit diffusion restriction.

### Required interpretation

Describe this as accepted-step scaling only. Do not infer wall-time scaling because the JSON contains no timing data, and do not treat the scaling exponent as a physical acceptance gate.

---

## Figure 09 — Deterministic audit table

### Purpose and gate

This table provides the point-by-point numerical contract behind the plotted campaign evidence. It should record the observed value, tolerance, status, test or case identifier, and a concise interpretation for every Stage 2 requirement.

### Required rows

- dh/dT = cp identity;
- equation-of-state identity;
- pure-species and mixture limits;
- enthalpy inversion;
- entropy-function inversion;
- NASA breakpoint continuity;
- cp positivity and enthalpy monotonicity;
- manufactured-isentrope entropy flatness;
- finite-layer entropy instability;
- rejected-state purity for T, h, density, z, g, and mass paths;
- constant-gravity enthalpy conservation;
- constant-gravity isothermal hydrostatic error;
- inverse-square isothermal hydrostatic error;
- nonisothermal independent-reference error;
- pressure round-trip error;
- approach to the constant-gravity limit;
- hard failure outside thermodynamic and hydrostatic domains;
- production completion: 27/27 completed, 27/27 converged, zero failures.

### Presentation rules

- Use exact observed values and tolerances rather than only PASS labels.
- Use N/A where a contract is deliberately not applicable.
- Add notes explaining why variable-gravity apparent energy change is not a conservation failure.
- Retain the machine-readable audit output alongside the rendered table.

---

## Recommended production order

1. Generate Figures 03, 04, 05, 07, and 08 directly from production_campaign.json.
2. Export one representative constant-gravity profile and time history for Figure 02.
3. Run or expose the deterministic thermodynamic audit for Figure 01.
4. Export the analytic, independent-reference, round-trip, and refinement results for Figure 06.
5. Assemble all exact thresholds and observed values into Figure 09.
6. Review every figure first for scientific correctness and then separately for presentation readiness.

## Final acceptance checklist

Stage 2 can be frozen when the evidence set shows all of the following:

- all thermodynamic identities and inversions meet their defined tolerances;
- the representative column relaxes to a constant-entropy numerical isentrope;
- every constant-gravity production combination converges;
- equilibrium results are insensitive to N, composition, and grid placement;
- strict constant-gravity enthalpy conservation passes;
- hydrostatic reconstruction passes analytic, independent-reference, round-trip, and refinement checks;
- inverse-square gravity reaches the entropy-based equilibrium and approaches constant gravity in the appropriate limit;
- variable-gravity mass-path effects are labelled as diagnostics rather than conservation failures;
- all deterministic failure modes and rejected-state purity checks pass;
- the figures, tables, source JSON, and plotting scripts are retained together for reproducibility.
