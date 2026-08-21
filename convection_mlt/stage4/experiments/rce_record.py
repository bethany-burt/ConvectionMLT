"""Serialize a coupled RCE result with profiles, history, and checksum."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from convection_mlt import AnalyticOpacityRCESpec, RCEResult


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def serialize_rce_result(
    res: RCEResult,
    spec: AnalyticOpacityRCESpec,
    *,
    pressure_centres: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = [d for d in res.diagnostics if d.accepted]
    rejected = [d for d in res.diagnostics if not d.accepted]
    p = np.asarray(pressure_centres, dtype=np.float64)
    t = np.asarray(res.final_state.temperature, dtype=np.float64)
    ftot = np.asarray(res.final_flux_total, dtype=np.float64)
    dts = [float(d.dt) for d in accepted]
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
        "eos": "ConstantH2Thermo",
        "route": res.route.value,
        "status": res.status.value,
        "reason": res.reason,
        "steps_accepted": res.steps_accepted,
        "rejections": res.rejections,
        "simulated_time": res.simulated_time,
        "flux_flatness": res.convergence.flux_flatness,
        "tendency_norm": res.convergence.tendency_norm,
        "temp_change": res.convergence.temp_change,
        "primary_rcb_log10p": res.primary_rcb_log10p,
        "convective_regions": res.convective_regions,
        "detached_convective_regions": res.detached_convective_regions,
        "pressure_centres": p.tolist(),
        "temperature": t.tolist(),
        "enthalpy": np.asarray(res.final_state.enthalpy, dtype=np.float64).tolist(),
        "mass_path": np.asarray(res.final_state.mass_path, dtype=np.float64).tolist(),
        "flux_total": ftot.tolist(),
        "flux_rad": np.asarray(res.final_flux_rad, dtype=np.float64).tolist(),
        "flux_conv": np.asarray(res.final_flux_conv, dtype=np.float64).tolist(),
        "history": {
            "dt": dts,
            "flux_flatness": [float(d.flux_flatness) for d in accepted],
            "tendency_norm": [float(d.tendency_norm) for d in accepted],
            "picard_iterations": [int(d.picard_iterations) for d in accepted],
            "coupled_defect": [float(d.coupled_defect) for d in accepted],
            "newton_iterations": [int(d.newton_iterations) for d in accepted],
        },
        "rejection_reasons": [d.rejection_reason for d in rejected],
        "median_accepted_dt": float(np.median(dts)) if dts else float("nan"),
        "checksum_sha256": _sha256_arrays(t, p, ftot),
    }
    if extra:
        payload.update(extra)
    return payload


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, allow_nan=True)
