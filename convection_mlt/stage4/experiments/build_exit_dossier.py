from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from convection_mlt import (
    ConstantGravity,
    ConstantGreyOpacity,
    ConstantH2Thermo,
    HeliosAdapter,
    LowerNetInternalFlux,
    PhysicsConfig,
    PrescribedBandOpacity,
    RCEConfig,
    RCERoute,
    RCETerminalStatus,
    SolverConfig,
    TopIrradiation,
    build_grid,
    load_integrated_flux,
    log_pressure_edges,
    manufactured_operator_identity,
    solve_adaptive_rce,
)
from convection_mlt.rce import ManufacturedRadiativeTarget, _evaluate_closure, _run_unsplit
from convection_mlt.state import build_column_state

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "stage4" / "results"
PLOTS = ROOT / "stage4" / "plots" / "generated"
LIVE_HELIOS = RESULTS / "live_helios_comparison.json"
FIXTURE_DIR = ROOT / "stage4" / "fixtures" / "helios"

GATES = {
    "manufactured_hot": 1e-8,
    "manufactured_cold": 1e-8,
    "unsplit_split_t": 1e-8,
    "flux_flatness": 1e-8,
    "boundary_mismatch": 1e-8,
    "energy_residual": 1e-12,
    "hot_cold": 1e-8,
    "identity_flux": 1e-15,
    "identity_tendency": 1e-15,
    "helios_t": 1e-8,
    "helios_flux": 1e-8,
    "orientation": 0.0,
}


def _setup(n_layers: int = 24):
    g = 15.0
    grid = build_grid(log_pressure_edges(5e6, 1e2, n_layers), g)
    p = grid.pressure_centres
    t = 900.0 * (p / p[0]) ** 0.58
    return grid, p, t


def _run(route: RCERoute, opacity, initial, manufactured=None, cfg=None, solver=None):
    grid, p, _t = _setup(initial.size)
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(gravity=15.0, alpha=1.0, closure_prefactor=0.5)
    solver = solver or SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    cfg = cfg or RCEConfig(
        n_consec=4,
        flux_flatness_tolerance=1e-8,
        tendency_tolerance=1e-8,
        temp_change_tolerance=1e-8,
        stall_window=200,
        max_steps=200,
    )
    return solve_adaptive_rce(
        grid, initial, physics, solver, thermo, opacity, p,
        TopIrradiation(flux=120.0), LowerNetInternalFlux(flux=300.0),
        gravity=ConstantGravity(15.0),
        route=route,
        config=cfg,
        manufactured=manufactured,
    )


def _row(*, name, observed, tolerance, criterion, scale, category, source, status=None, extra=None):
    obs = None if observed is None else float(observed) if np.isscalar(observed) else observed
    finite = isinstance(obs, (int, float)) and np.isfinite(obs)
    if status is None:
        if not finite:
            status = "FAIL"
        elif criterion == "<=":
            status = "PASS" if obs <= tolerance else "FAIL"
        elif criterion == "==":
            status = "PASS" if obs == tolerance else "FAIL"
        elif criterion == "true":
            status = "PASS" if bool(obs) else "FAIL"
        else:
            status = "FAIL"
    row = {
        "name": name,
        "observed": obs if finite or isinstance(obs, str) else observed,
        "tolerance": tolerance,
        "criterion": criterion,
        "status": status,
        "scale": scale,
        "category": category,
        "source": source,
    }
    if extra:
        row.update(extra)
    return row


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _bottleneck(grid, p, t_target, opacity):
    thermo = ConstantH2Thermo()
    physics = PhysicsConfig(gravity=15.0, alpha=1.0, closure_prefactor=0.5)
    solver = SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)
    cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10)
    state = build_column_state(grid, t_target * 1.01, thermo, ConstantGravity(15.0))
    closure, rad, f_conv, f_rad, f_total = _run_unsplit(
        grid, state, physics, thermo, opacity, p,
        TopIrradiation(120.0), LowerNetInternalFlux(300.0), cfg, None, ConstantGravity(15.0),
    )
    from convection_mlt.energy import enthalpy_tendency
    from convection_mlt.rce import _dt_mlt_estimate, _dt_rad_estimate

    dt_mlt = _dt_mlt_estimate(grid, state, closure, solver)
    dt_rad = _dt_rad_estimate(
        state,
        rad.heating if rad is not None else enthalpy_tendency(grid, f_rad, state.mass_path),
        solver,
        thermo,
    )

    def _stable(dt):
        res = solve_adaptive_rce(
            grid, state.temperature, physics, solver, thermo, opacity, p,
            TopIrradiation(120.0), LowerNetInternalFlux(300.0),
            gravity=ConstantGravity(15.0),
            route=RCERoute.UNSPLIT,
            config=RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=dt),
        )
        return res.steps_accepted == 1

    lo = min(dt_mlt, dt_rad) if np.isfinite(min(dt_mlt, dt_rad)) else 1e-8
    hi = lo
    for _ in range(12):
        if _stable(hi):
            lo = hi
            hi *= 2.0
        else:
            break
    bracket_low = lo
    bracket_high = hi
    t0 = time.perf_counter()
    for _ in range(20):
        _evaluate_closure(grid, state, physics, thermo)
    t_conv = (time.perf_counter() - t0) / 20.0
    t0 = time.perf_counter()
    for _ in range(20):
        _run_unsplit(
            grid, state, physics, thermo, opacity, p,
            TopIrradiation(120.0), LowerNetInternalFlux(300.0), cfg, None, ConstantGravity(15.0),
        )
    t_rad = (time.perf_counter() - t0) / 20.0
    return {
        "dt_mlt": float(dt_mlt),
        "dt_rad": float(dt_rad),
        "bracketed_stable_dt_low": float(bracket_low),
        "bracketed_stable_dt_high": float(bracket_high),
        "convection_only_call_time_s": float(t_conv),
        "radiation_plus_conv_call_time_s": float(t_rad),
        "call_count_each": 20,
        "smallest_stable_operator": "radiation" if dt_rad <= dt_mlt else "convection",
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    grid, p, t_target = _setup(24)
    opacity_grey = ConstantGreyOpacity(2e-4)
    manufactured = ManufacturedRadiativeTarget(target_temperature=t_target, f0=250.0, relaxation_coeff=1.0)
    manufactured_frozen = ManufacturedRadiativeTarget(target_temperature=t_target, f0=250.0, relaxation_coeff=0.0)

    _f_tot, _dhdt, ident_flux, ident_tend = manufactured_operator_identity(
        grid, PhysicsConfig(gravity=15.0, alpha=1.0, closure_prefactor=0.5),
        ConstantH2Thermo(), manufactured_frozen, gravity=ConstantGravity(15.0),
    )

    hot = t_target.copy()
    hot[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 1.02
    cold = t_target.copy()
    cold[grid.n_layers // 3 : 2 * grid.n_layers // 3] *= 0.98
    attr_cfg = RCEConfig(
        n_consec=4,
        flux_flatness_tolerance=1e-8,
        tendency_tolerance=1e-8,
        temp_change_tolerance=1e-8,
        stall_window=250,
        max_steps=250,
    )
    res_hot = _run(RCERoute.UNSPLIT, opacity_grey, hot, manufactured=manufactured, cfg=attr_cfg)
    res_cold = _run(RCERoute.UNSPLIT, opacity_grey, cold, manufactured=manufactured, cfg=attr_cfg)

    one_cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=1e-6)
    res_one = _run(RCERoute.UNSPLIT, opacity_grey, hot, manufactured=manufactured_frozen, cfg=one_cfg)
    accepted_one = [d for d in res_one.diagnostics if d.accepted]
    one_energy = accepted_one[0].energy_residual_rel if accepted_one else float("nan")

    probe = _run(RCERoute.UNSPLIT, opacity_grey, t_target * (1 + 0.005 * np.sin(np.linspace(0, np.pi, grid.n_layers))), cfg=RCEConfig(max_steps=1, n_consec=99, stall_window=10))
    dt_common = 0.05 * probe.diagnostics[0].dt if probe.diagnostics else 1e-6
    split_cfg = RCEConfig(max_steps=1, n_consec=99, stall_window=10, prescribed_dt=dt_common)
    ic = t_target * (1 + 0.005 * np.sin(np.linspace(0, np.pi, grid.n_layers)))
    res_unsplit = _run(RCERoute.UNSPLIT, opacity_grey, ic, cfg=split_cfg)
    res_split_rc = _run(RCERoute.SPLIT_RAD_THEN_CONV, opacity_grey, ic, cfg=split_cfg)
    res_split_cr = _run(RCERoute.SPLIT_CONV_THEN_RAD, opacity_grey, ic, cfg=split_cfg)

    def _max_temp_diff(a, b):
        scale = np.maximum(np.abs(a.final_state.temperature), 1e-12)
        return float(np.max(np.abs(a.final_state.temperature - b.final_state.temperature) / scale))

    long_cfg = RCEConfig(
        n_consec=4,
        flux_flatness_tolerance=1e-8,
        tendency_tolerance=1e-8,
        temp_change_tolerance=1e-8,
        stall_window=200,
        max_steps=200,
    )
    res_real = _run(RCERoute.UNSPLIT, opacity_grey, ic, cfg=long_cfg)
    accepted_unsplit = [d for d in res_real.diagnostics if d.accepted]
    max_energy_res = (
        float(max(d.energy_residual_rel for d in accepted_unsplit))
        if accepted_unsplit else float("nan")
    )

    kappa = np.vstack([np.full(grid.n_layers, 2e-4), np.full(grid.n_layers, 6e-4), np.full(grid.n_layers, 1e-6)])
    _ = _run(RCERoute.UNSPLIT, PrescribedBandOpacity(kappa, np.array([0.7, 0.3, 0.0])), t_target, cfg=RCEConfig(max_steps=1, n_consec=99, stall_window=10))

    t0 = time.perf_counter()
    _ = _run(RCERoute.UNSPLIT, opacity_grey, t_target * 1.01, cfg=RCEConfig(max_steps=30, n_consec=99, stall_window=30))
    wall_unsplit = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = _run(RCERoute.SPLIT_RAD_THEN_CONV, opacity_grey, t_target * 1.01, cfg=RCEConfig(max_steps=30, n_consec=99, stall_window=30))
    wall_split_rc = time.perf_counter() - t0
    bottleneck = _bottleneck(grid, p, t_target, opacity_grey)
    bottleneck["wall_time_unsplit_s"] = wall_unsplit
    bottleneck["wall_time_split_rc_s"] = wall_split_rc

    adapter = HeliosAdapter(helios_top_to_bottom=True)
    layers = np.array([10.0, 20.0, 30.0, 40.0])
    orientation_exact = bool(np.array_equal(adapter.roundtrip_layers(layers), layers))
    fixture_flux = FIXTURE_DIR / "sample_integrated_flux.dat"
    fixture_checksum = _sha256_file(fixture_flux)

    live = {}
    if LIVE_HELIOS.exists():
        live = json.loads(LIVE_HELIOS.read_text(encoding="utf-8"))

    live_t = live.get("metrics", {}).get("equilibrium_temperature_max_rel")
    live_f = live.get("metrics", {}).get("equilibrium_flux_total_max_rel")
    p38 = live.get("point_38", {})
    conv_off_rel = p38.get("convection_off_flux_max_rel")
    conv_off_portable = p38.get("convection_off_flux_file_portable")
    conv_off_checksum = p38.get("convection_off_checksum_sha256")

    hot_rel = float(np.max(np.abs(res_hot.final_state.temperature - t_target) / t_target))
    cold_rel = float(np.max(np.abs(res_cold.final_state.temperature - t_target) / t_target))
    t_init = 0.02
    if hot_rel > 0.0 and res_hot.simulated_time > 0.0 and hot_rel < t_init:
        bottleneck["physical_relaxation_tau_s"] = float(
            res_hot.simulated_time / np.log(t_init / hot_rel)
        )
    else:
        bottleneck["physical_relaxation_tau_s"] = float("nan")
    bottleneck["physical_relaxation_residual"] = "max |T-T*| / T*"

    rows = [
        _row(name="35_manufactured_identity_flux", observed=ident_flux, tolerance=GATES["identity_flux"], criterion="<=", scale="max |F_rad*+F_conv-F0|", category="35", source="manufactured_operator_identity"),
        _row(name="35_manufactured_identity_tendency", observed=ident_tend, tolerance=GATES["identity_tendency"], criterion="<=", scale="max |dh/dt| at T*", category="35", source="manufactured_operator_identity"),
        _row(
            name="35_manufactured_attraction_hot",
            observed=hot_rel,
            tolerance=GATES["manufactured_hot"],
            criterion="<=",
            scale="max relative T",
            category="35",
            source="solve_adaptive_rce unsplit manufactured",
            extra={
                "terminal_status": res_hot.status.value,
                "steps_accepted": res_hot.steps_accepted,
                "simulated_time": res_hot.simulated_time,
                "auditable": res_hot.status == RCETerminalStatus.CONVERGED,
            },
        ),
        _row(
            name="35_manufactured_attraction_cold",
            observed=cold_rel,
            tolerance=GATES["manufactured_cold"],
            criterion="<=",
            scale="max relative T",
            category="35",
            source="solve_adaptive_rce unsplit manufactured",
            extra={
                "terminal_status": res_cold.status.value,
                "steps_accepted": res_cold.steps_accepted,
                "simulated_time": res_cold.simulated_time,
                "auditable": res_cold.status == RCETerminalStatus.CONVERGED,
            },
        ),
        _row(name="36_one_step_energy_residual_rel", observed=one_energy, tolerance=GATES["energy_residual"], criterion="<=", scale="max(|RHS|, F_scale dt, E_floor)", category="36", source="one accepted unsplit manufactured step"),
        _row(name="36_unsplit_split_rc_one_step", observed=_max_temp_diff(res_unsplit, res_split_rc), tolerance=GATES["unsplit_split_t"], criterion="<=", scale="max relative T, common dt", category="36", source="prescribed_dt one-step"),
        _row(name="36_unsplit_split_cr_one_step", observed=_max_temp_diff(res_unsplit, res_split_cr), tolerance=GATES["unsplit_split_t"], criterion="<=", scale="max relative T, common dt", category="36", source="prescribed_dt one-step"),
        _row(
            name="36_flux_flatness_unsplit",
            observed=float(res_real.convergence.flux_flatness),
            tolerance=GATES["flux_flatness"],
            criterion="<=",
            scale="max |F-F_ref| / F_scale",
            category="36",
            source="real unsplit RCE",
            extra={"terminal_status": res_real.status.value, "auditable": res_real.status == RCETerminalStatus.CONVERGED},
        ),
        _row(
            name="36_boundary_mismatch_unsplit",
            observed=float(abs(res_real.final_flux_total[0] - res_real.final_flux_total[-1]) / max(np.max(np.abs(res_real.final_flux_total)), 1e-30)),
            tolerance=GATES["boundary_mismatch"],
            criterion="<=",
            scale="|F(0)-F(N)| / F_scale",
            category="36",
            source="real unsplit RCE",
            extra={"terminal_status": res_real.status.value, "auditable": res_real.status == RCETerminalStatus.CONVERGED},
        ),
        _row(name="36_time_integrated_energy_residual_max", observed=max_energy_res, tolerance=GATES["energy_residual"], criterion="<=", scale="max(|RHS|, F_scale dt, E_floor)", category="36", source="accepted unsplit steps"),
        _row(name="37_helios_temperature_max_rel", observed=live_t, tolerance=GATES["helios_t"], criterion="<=", scale="max relative T, matched setup required", category="37", source="live_helios_comparison.json"),
        _row(name="37_helios_flux_max_rel", observed=live_f, tolerance=GATES["helios_flux"], criterion="<=", scale="max relative F_total, matched setup required", category="37", source="live_helios_comparison.json"),
        _row(name="38_orientation_roundtrip_exact", observed=orientation_exact, tolerance=True, criterion="true", scale="exact array equality", category="38", source="HeliosAdapter.roundtrip_layers"),
        _row(
            name="38_convection_off_flux_max_rel",
            observed=conv_off_rel,
            tolerance=GATES["helios_flux"],
            criterion="<=",
            scale="max relative F, convection-off",
            category="38",
            source="live_helios_comparison.json",
            extra={
                "portable_name": conv_off_portable,
                "checksum_sha256": conv_off_checksum,
                "fixture_portable_name": fixture_flux.name,
                "fixture_checksum_sha256": fixture_checksum,
            },
        ),
        _row(name="40_hot_cold_path_independence", observed=_max_temp_diff(res_hot, res_cold), tolerance=GATES["hot_cold"], criterion="<=", scale="max relative T", category="40", source="manufactured hot vs cold"),
    ]

    # Non-converged attraction/RCE rows cannot pass even if a number is small.
    for row in rows:
        if row.get("auditable") is False:
            row["status"] = "FAIL"
            row["note"] = "terminal status is not converged; result is not auditable"

    required = {r["name"] for r in rows}
    # Point 39 completeness is a separate boolean row.
    p39_complete = all(
        k in bottleneck and np.isfinite(bottleneck[k])
        for k in ("dt_mlt", "dt_rad", "bracketed_stable_dt_low", "bracketed_stable_dt_high", "physical_relaxation_tau_s")
    )
    rows.append(_row(
        name="39_bottleneck_fields_complete",
        observed=p39_complete,
        tolerance=True,
        criterion="true",
        scale="dt_rad, dt_MLT, bracketed stable dt present",
        category="39",
        source="frozen-operator bottleneck",
        extra={"metrics": bottleneck},
    ))
    required.add("39_bottleneck_fields_complete")

    full_claim = all(r["status"] == "PASS" for r in rows if r["name"] in required)
    audit = {
        "stage": "4",
        "full_stage4_claim": full_claim,
        "claim_text": (
            "Stage 4 points 35-40 satisfied including live pinned HELIOS comparison."
            if full_claim
            else "Live HELIOS execution completed; Stage 4 core and HELIOS parity remain pending."
        ),
        "gates": GATES,
        "rows": rows,
        "points": {
            "35": {
                "identity_flux_err": ident_flux,
                "identity_tendency_err": ident_tend,
                "manufactured_attraction_hot_max_rel": hot_rel,
                "manufactured_attraction_cold_max_rel": cold_rel,
                "hot_status": res_hot.status.value,
                "cold_status": res_cold.status.value,
            },
            "36": {
                "unsplit_status": res_real.status.value,
                "unsplit_split_rc_temp_max_rel": _max_temp_diff(res_unsplit, res_split_rc),
                "unsplit_split_cr_temp_max_rel": _max_temp_diff(res_unsplit, res_split_cr),
                "flux_flatness_unsplit": float(res_real.convergence.flux_flatness),
                "boundary_mismatch_unsplit": float(abs(res_real.final_flux_total[0] - res_real.final_flux_total[-1]) / max(np.max(np.abs(res_real.final_flux_total)), 1e-30)),
                "time_integrated_energy_residual_max": max_energy_res,
                "one_step_energy_residual_rel": one_energy,
            },
            "37": {
                "live_helios_status": live.get("status", "pending"),
                "helios_commit": live.get("helios_commit"),
                "equilibrium_temperature_max_rel": live_t,
                "equilibrium_flux_total_max_rel": live_f,
                "note": "unmatched live HELIOS run is a pilot, not parity",
            },
            "38": {
                "adapter_orientation_fixture_status": "implemented",
                "orientation_roundtrip_exact": orientation_exact,
                "fixture_portable_name": fixture_flux.name,
                "fixture_checksum_sha256": fixture_checksum,
                "live_helios_boundary_status": live.get("status", "pending"),
                "convection_off_flux_max_rel": conv_off_rel,
                "convection_off_flux_file_portable": conv_off_portable,
                "convection_off_checksum_sha256": conv_off_checksum,
            },
            "39": bottleneck,
            "40": {
                "hot_cold_path_independence_max_rel": _max_temp_diff(res_hot, res_cold),
                "hot_status": res_hot.status.value,
                "cold_status": res_cold.status.value,
            },
        },
    }

    with (RESULTS / "exit_gate_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)


if __name__ == "__main__":
    main()
