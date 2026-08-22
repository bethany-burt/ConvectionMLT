"""Serialize a coupled RCE result with profiles, history, configs, and checksums."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ImplicitConvectionConfig,
    RCEConfig,
    RCEResult,
    RCERoute,
    SolverConfig,
)
from convection_mlt.energy import column_enthalpy_per_area
from convection_mlt.metadata import git_commit, git_dirty, json_safe

REPO = Path(__file__).resolve().parents[3]
REQUESTED_ROUTE = RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV
ACTUAL_INTEGRATOR_PICARD = "coupled_picard_backward_euler"
PHYSICAL_GATE = 1.0e-3


def production_solver_config() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2e-3, c_diff=0.2, dt_min=1e-14)


def production_implicit_config() -> ImplicitConvectionConfig:
    return ImplicitConvectionConfig(
        residual_tolerance=1e-10,
        step_tolerance=1e-10,
        newton_residual_tolerance=1e-12,
        newton_step_tolerance=1e-12,
    )


def production_rce_config(
    *,
    max_steps: int,
    dt_accuracy: float = 2500.0,
    dt_hold_init: float | None = None,
    previous_rcb_init: float | None = None,
    simulated_time_init: float = 0.0,
    gate: float = PHYSICAL_GATE,
) -> RCEConfig:
    return RCEConfig(
        max_steps=max_steps,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=gate,
        tendency_tolerance=gate,
        temp_change_tolerance=gate,
        dt_accuracy=dt_accuracy,
        coupled_picard=True,
        use_coupled_tendency_dt=True,
        dt_hold_init=dt_hold_init,
        previous_rcb_init=previous_rcb_init,
        simulated_time_init=simulated_time_init,
        implicit_convection=production_implicit_config(),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return json_safe(value)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _record_checksum(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {
            "record_checksum_sha256",
            "profile_checksum_sha256",
            "checksum_sha256",
        }
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def algebraic_identities(payload: dict[str, Any]) -> dict[str, float]:
    """Construction identities supporting the 1e-12 algebraic gate."""
    f_tot = np.asarray(payload["flux_total"], dtype=np.float64)
    f_rad = np.asarray(payload["flux_rad"], dtype=np.float64)
    f_conv = np.asarray(payload["flux_conv"], dtype=np.float64)
    mass = np.asarray(payload["mass_path"], dtype=np.float64)
    f_int = float(payload["f_int"])
    scale = max(abs(f_int), 1.0)
    heating = (f_tot[:-1] - f_tot[1:]) / mass
    return {
        "flux_split_identity_rel": float(np.max(np.abs(f_rad + f_conv - f_tot)) / scale),
        "telescoping_column_energy_rel": float(
            abs(float(np.sum(mass * heating)) - float(f_tot[0] - f_tot[-1])) / scale
        ),
        "bottom_boundary_exactness_rel": float(abs(float(f_tot[0]) - f_int) / scale),
    }


def serialize_rce_result(
    res: RCEResult,
    spec: AnalyticOpacityRCESpec,
    *,
    pressure_centres: np.ndarray,
    pressure_edges: np.ndarray | None = None,
    solver: SolverConfig | None = None,
    rce_config: RCEConfig | None = None,
    requested_route: RCERoute | str = REQUESTED_ROUTE,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = [d for d in res.diagnostics if d.accepted]
    rejected = [d for d in res.diagnostics if not d.accepted]
    p = np.asarray(pressure_centres, dtype=np.float64)
    edges = (
        np.asarray(pressure_edges, dtype=np.float64)
        if pressure_edges is not None
        else None
    )
    t = np.asarray(res.final_state.temperature, dtype=np.float64)
    h = np.asarray(res.final_state.enthalpy, dtype=np.float64)
    mass = np.asarray(res.final_state.mass_path, dtype=np.float64)
    ftot = np.asarray(res.final_flux_total, dtype=np.float64)
    frad = np.asarray(res.final_flux_rad, dtype=np.float64)
    fconv = np.asarray(res.final_flux_conv, dtype=np.float64)
    dts = [float(d.dt) for d in accepted]
    last = accepted[-1] if accepted else None
    coupled = True if rce_config is None else bool(rce_config.coupled_picard)
    actual = ACTUAL_INTEGRATOR_PICARD if coupled else str(
        requested_route.value if isinstance(requested_route, RCERoute) else requested_route
    )
    arrays_for_profile = [t, p, ftot, frad, fconv, h, mass]
    if edges is not None:
        arrays_for_profile.append(edges)
    profile_checksum = _sha256_arrays(*arrays_for_profile)
    payload: dict[str, Any] = {
        "n_layers": spec.n_layers,
        "n_photosphere": spec.n_photosphere,
        "nested_master_layers": spec.nested_master_layers,
        "nested_master_photosphere": spec.nested_master_photosphere,
        "p_bottom": spec.p_bottom,
        "p_top": spec.p_top,
        "tau_total": spec.tau_total,
        "f_int": spec.f_int,
        "f_irr": spec.f_irr,
        "gravity": spec.gravity,
        "eos": "ConstantH2Thermo",
        "requested_route": (
            requested_route.value if isinstance(requested_route, RCERoute) else requested_route
        ),
        "actual_integrator": actual,
        "route": res.route.value,
        "status": res.status.value,
        "reason": res.reason,
        "steps_accepted": res.steps_accepted,
        "rejections": res.rejections,
        "simulated_time": res.simulated_time,
        "flux_flatness": res.convergence.flux_flatness,
        "tendency_norm": res.convergence.tendency_norm,
        "temp_change": res.convergence.temp_change,
        "rcb_stable": res.convergence.rcb_stable,
        "primary_rcb_log10p": res.primary_rcb_log10p,
        "convective_regions": res.convective_regions,
        "detached_convective_regions": res.detached_convective_regions,
        "pressure_centres": p.tolist(),
        "pressure_edges": None if edges is None else edges.tolist(),
        "temperature": t.tolist(),
        "enthalpy": h.tolist(),
        "mass_path": mass.tolist(),
        "column_enthalpy": column_enthalpy_per_area(mass, h),
        "flux_total": ftot.tolist(),
        "flux_rad": frad.tolist(),
        "flux_conv": fconv.tolist(),
        "history": {
            "dt": dts,
            "flux_flatness": [float(d.flux_flatness) for d in accepted],
            "tendency_norm": [float(d.tendency_norm) for d in accepted],
            "picard_iterations": [int(d.picard_iterations) for d in accepted],
            "coupled_defect": [float(d.coupled_defect) for d in accepted],
            "newton_iterations": [int(d.newton_iterations) for d in accepted],
            "energy_residual_rel": [float(d.energy_residual_rel) for d in accepted],
            "boundary_mismatch": [float(d.boundary_mismatch) for d in accepted],
        },
        "rejection_reasons": [d.rejection_reason for d in rejected],
        "median_accepted_dt": float(np.median(dts)) if dts else float("nan"),
        "last_accepted_dt": dts[-1] if dts else None,
        "energy_residual_rel": None if last is None else float(last.energy_residual_rel),
        "energy_committed_residual_rel": (
            None if last is None else float(last.energy_committed_residual_rel)
        ),
        "boundary_mismatch": None if last is None else float(last.boundary_mismatch),
        "coupled_defect": None if last is None else float(last.coupled_defect),
        "rce_config": None if rce_config is None else _jsonable(rce_config),
        "solver_config": None if solver is None else _jsonable(solver),
        "implicit_convection_config": _jsonable(
            rce_config.implicit_convection if rce_config is not None else production_implicit_config()
        ),
        "physics_config": _jsonable(spec.physics()),
        "environment": code_snapshot(),
        "profile_checksum_sha256": profile_checksum,
        "checksum_sha256": profile_checksum,
    }
    payload.update(algebraic_identities(payload))
    if extra:
        payload.update(extra)
    payload["record_checksum_sha256"] = _record_checksum(payload)
    return payload


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, allow_nan=True)


def code_snapshot() -> dict[str, Any]:
    return {
        "git_commit": git_commit(REPO),
        "git_dirty": git_dirty(REPO),
        "python": __import__("platform").python_version(),
        "numpy": np.__version__,
    }


def enrich_stored_record(record: dict[str, Any]) -> dict[str, Any]:
    """Fill provenance on a previously stored record without re-running."""
    from convection_mlt import nested_analytic_opacity_spec

    n = int(record.get("n_layers") or 0)
    if n and record.get("nested_master_layers") and record.get("pressure_edges") is None:
        spec = nested_analytic_opacity_spec(n)
        grid = spec.grid()
        record["pressure_edges"] = grid.pressure_edges.tolist()
    record.setdefault("requested_route", "split_rad_then_implicit_conv")
    record.setdefault("actual_integrator", ACTUAL_INTEGRATOR_PICARD)
    if record.get("profile_checksum_sha256") is None and record.get("checksum_sha256"):
        record["profile_checksum_sha256"] = record["checksum_sha256"]
    if record.get("checksum_sha256") is None and record.get("profile_checksum_sha256"):
        record["checksum_sha256"] = record["profile_checksum_sha256"]
    if record.get("rce_config") is None:
        record["rce_config"] = _jsonable(production_rce_config(
            max_steps=int(record.get("steps_accepted") or 0),
            dt_accuracy=2500.0,
        ))
        record["rce_config"]["note"] = (
            "Reconstructed production defaults for the original stored run; "
            "continuation chunks replace this with the live config."
        )
    if record.get("solver_config") is None:
        record["solver_config"] = _jsonable(production_solver_config())
    if record.get("implicit_convection_config") is None:
        record["implicit_convection_config"] = _jsonable(production_implicit_config())
    if record.get("environment") is None:
        record["environment"] = code_snapshot()
    cont = record.setdefault("continuation", {})
    versions = list(cont.get("code_versions") or [])
    if not versions and int(record.get("steps_accepted") or 0) >= 62000:
        versions.append({
            "extra_accepted_from": 0,
            "extra_accepted_to": int((cont.get("extra_accepted") or 50000)),
            "git_commit": "0bd78d10792cfe8f8be284930bd2c3299d8319a2",
            "git_dirty": True,
            "note": (
                "50k Δt=100 s continuation ran on a dirty 0bd78d1 tree; "
                "the archived snapshot of that record is 709113f."
            ),
        })
        cont["code_versions"] = versions
        record["continuation"] = cont
    if all(k in record for k in ("flux_total", "flux_rad", "flux_conv", "mass_path", "f_int")):
        record.update(algebraic_identities(record))
    record["record_checksum_sha256"] = _record_checksum(record)
    return record


def merge_continuation(base: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    """Append a continuation chunk onto a stored complete record."""
    merged = dict(base)
    hist = dict(base.get("history") or {})
    chunk_hist = chunk.get("history") or {}
    n_old = len(hist.get("dt") or [])
    for key in (
        "dt",
        "flux_flatness",
        "tendency_norm",
        "picard_iterations",
        "coupled_defect",
        "newton_iterations",
        "energy_residual_rel",
        "boundary_mismatch",
    ):
        old = list(hist.get(key) or [])
        if len(old) < n_old:
            old = old + [None] * (n_old - len(old))
        hist[key] = old + list(chunk_hist.get(key) or [])
    merged["history"] = hist
    merged["rejection_reasons"] = list(base.get("rejection_reasons") or []) + list(
        chunk.get("rejection_reasons") or []
    )
    merged["steps_accepted"] = int(base.get("steps_accepted") or 0) + int(chunk.get("steps_accepted") or 0)
    merged["rejections"] = int(base.get("rejections") or 0) + int(chunk.get("rejections") or 0)
    merged["simulated_time"] = float(base.get("simulated_time") or 0.0) + float(
        chunk.get("simulated_time") or 0.0
    )
    merged["wall_time_s"] = float(base.get("wall_time_s") or 0.0) + float(chunk.get("wall_time_s") or 0.0)
    for key in (
        "status",
        "reason",
        "flux_flatness",
        "tendency_norm",
        "temp_change",
        "rcb_stable",
        "primary_rcb_log10p",
        "convective_regions",
        "detached_convective_regions",
        "temperature",
        "enthalpy",
        "mass_path",
        "column_enthalpy",
        "flux_total",
        "flux_rad",
        "flux_conv",
        "median_accepted_dt",
        "last_accepted_dt",
        "energy_residual_rel",
        "energy_committed_residual_rel",
        "boundary_mismatch",
        "coupled_defect",
        "profile_checksum_sha256",
        "checksum_sha256",
        "rce_config",
        "solver_config",
        "implicit_convection_config",
        "environment",
        "requested_route",
        "actual_integrator",
        "pressure_edges",
        "pressure_centres",
    ):
        if key in chunk:
            merged[key] = chunk[key]
    merged.update({k: chunk[k] for k in algebraic_identities(chunk)})
    cont = dict(merged.get("continuation") or {})
    versions = list(cont.get("code_versions") or [])
    env = chunk.get("environment") or {}
    extra_to = int(merged.get("continuation", {}).get("extra_accepted") or merged.get("steps_accepted") or 0)
    extra_from = extra_to - int(chunk.get("steps_accepted") or 0)
    versions.append({
        "extra_accepted_from": extra_from,
        "extra_accepted_to": extra_to,
        "git_commit": env.get("git_commit"),
        "git_dirty": env.get("git_dirty"),
        "python": env.get("python"),
        "numpy": env.get("numpy"),
    })
    cont["code_versions"] = versions
    merged["continuation"] = cont
    merged["record_checksum_sha256"] = _record_checksum(merged)
    return merged
