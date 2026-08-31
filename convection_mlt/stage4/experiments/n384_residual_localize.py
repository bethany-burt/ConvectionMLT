"""Localize the N=384 flux-flatness residual versus pressure.

Does not continue the production job or loosen gates. Writes a compact
profile JSON for plotting and a one-line classification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from run_nested_family import interpolate_temperature

RESULTS = ROOT / "results"
N384 = RESULTS / "n384_implicit_rce.json"
N192 = RESULTS / "n192_implicit_rce.json"
STEP500 = RESULTS / "n384_probe_step0500.json"
OUT = RESULTS / "n384_residual_localize.json"
NABLA_AD = 2.0 / 7.0
NSAMP = 48


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _mask(n_layers: int, regions) -> np.ndarray:
    mask = np.zeros(n_layers, dtype=bool)
    for lo, hi in regions or []:
        mask[int(lo) : int(hi) + 1] = True
    return mask


def main() -> dict:
    rec = _load(N384)
    n192 = _load(N192) if N192.exists() else None
    step500 = _load(STEP500) if STEP500.exists() else None
    p = np.asarray(rec["pressure_centres"], dtype=np.float64)
    t = np.asarray(rec["temperature"], dtype=np.float64)
    mass = np.asarray(rec["mass_path"], dtype=np.float64)
    f_tot = np.asarray(rec["flux_total"], dtype=np.float64)
    f_rad = np.asarray(rec["flux_rad"], dtype=np.float64)
    f_conv = np.asarray(rec["flux_conv"], dtype=np.float64)
    f_int = float(rec["f_int"])
    n = t.size
    mask = _mask(n, rec.get("convective_regions"))
    dF = f_tot[:-1] - f_tot[1:]
    heating = dF / np.maximum(mass, 1e-30)
    dlnT = np.diff(np.log(t))
    dlnP = np.diff(np.log(p))
    nabla = np.concatenate([dlnT / np.maximum(dlnP, 1e-30), [dlnT[-1] / max(dlnP[-1], 1e-30)]])
    superad = np.clip(nabla - NABLA_AD, -2.0, 2.0)
    excess = f_tot - f_int
    i_ex = int(np.argmax(np.abs(excess)))
    i_heat = int(np.argmax(np.abs(heating)))
    i_dT = None
    dT_vs_n192 = None
    dT_recent = None
    if n192 is not None:
        t192_on_384 = interpolate_temperature(
            np.log(np.asarray(n192["pressure_centres"], dtype=np.float64)),
            n192["temperature"],
            np.log(p),
        )
        dT_vs_n192 = t - t192_on_384
        i_dT = int(np.argmax(np.abs(dT_vs_n192) / np.maximum(np.abs(t192_on_384), 1.0)))
    if step500 is not None:
        t500 = np.asarray(step500["temperature"], dtype=np.float64)
        dT_recent = t - t500

    p_lay_log = np.log10(p)
    p_int = np.asarray(rec.get("pressure_edges") or p, dtype=np.float64)
    if p_int.size == f_tot.size:
        p_int_log = np.log10(p_int)
    else:
        p_int_log = np.concatenate([p_lay_log, [p_lay_log[-1]]])

    conv_frac = float(np.mean(mask))
    rad_excess_rms = float(np.sqrt(np.mean(excess[1:][~mask] ** 2))) if np.any(~mask) else 0.0
    conv_excess_rms = float(np.sqrt(np.mean(excess[1:][mask] ** 2))) if np.any(mask) else 0.0
    rcb_log10p = rec.get("primary_rcb_log10p")
    near_rcb = np.abs(p_lay_log - float(rcb_log10p)) < 0.15 if rcb_log10p is not None else np.zeros(n, dtype=bool)
    heat_near_rcb = float(np.max(np.abs(heating[near_rcb]))) if np.any(near_rcb) else 0.0
    heat_global = float(np.max(np.abs(heating)))
    i_ex_layer = min(i_ex, n - 1)
    classification = (
        "column_wide_radiative_mode"
        if (not mask[i_ex_layer]) or rad_excess_rms >= conv_excess_rms
        else "rcb_active_set"
    )

    hist = rec.get("history") or {}
    last_n = 50
    def tail(key):
        arr = hist.get(key) or []
        return [float(x) for x in arr[-last_n:]]

    samp = np.unique(np.round(np.linspace(0, n - 1, NSAMP)).astype(int))
    profile = {
        "log10p": p_lay_log[samp].tolist(),
        "temperature_K": t[samp].tolist(),
        "F_total_minus_F_int": excess[np.minimum(samp, excess.size - 1)].tolist(),
        "F_rad": f_rad[np.minimum(samp, f_rad.size - 1)].tolist(),
        "F_conv": f_conv[np.minimum(samp, f_conv.size - 1)].tolist(),
        "heating_W_kg": heating[np.minimum(samp, heating.size - 1)].tolist(),
        "superadiabaticity": superad[samp].tolist(),
        "convective_mask": mask[samp].astype(int).tolist(),
        "delta_T_vs_n192_K": None if dT_vs_n192 is None else dT_vs_n192[samp].tolist(),
        "delta_T_step500_to_9500_K": None if dT_recent is None else dT_recent[samp].tolist(),
        "layer_flux_divergence": dF[np.minimum(samp, dF.size - 1)].tolist(),
    }

    payload = {
        "source": str(N384),
        "n_layers": n,
        "steps_accepted": rec.get("steps_accepted"),
        "status": rec.get("status"),
        "flux_flatness": rec.get("flux_flatness"),
        "tendency_norm": rec.get("tendency_norm"),
        "energy_gate_ratio": rec.get("energy_gate_ratio"),
        "coupled_defect": rec.get("coupled_defect"),
        "last_accepted_dt": rec.get("last_accepted_dt"),
        "primary_rcb_log10p": rcb_log10p,
        "convective_regions": rec.get("convective_regions"),
        "detached_convective_regions": rec.get("detached_convective_regions"),
        "convective_fraction_of_layers": conv_frac,
        "classification": classification,
        "max_abs_F_minus_Fint": {
            "value": float(excess[i_ex]),
            "abs": float(abs(excess[i_ex])),
            "interface": i_ex,
            "pressure_Pa": float(p_int[min(i_ex, p_int.size - 1)]),
            "log10p": float(p_int_log[min(i_ex, p_int_log.size - 1)]),
        },
        "max_abs_heating": {
            "value": float(heating[i_heat]),
            "layer": i_heat,
            "pressure_Pa": float(p[i_heat]),
            "log10p": float(p_lay_log[i_heat]),
        },
        "radiative_zone_excess_rms": rad_excess_rms,
        "convective_zone_excess_rms": conv_excess_rms,
        "heat_max_near_rcb": heat_near_rcb,
        "heat_max_global": heat_global,
        "max_rel_T_vs_n192": None if dT_vs_n192 is None else {
            "layer": i_dT,
            "pressure_Pa": float(p[i_dT]),
            "log10p": float(p_lay_log[i_dT]),
            "dT_K": float(dT_vs_n192[i_dT]),
            "rel": float(abs(dT_vs_n192[i_dT]) / max(abs(t[i_dT]), 1.0)),
        },
        "history_tail": {
            "n": last_n,
            "flux_flatness": tail("flux_flatness"),
            "tendency_norm": tail("tendency_norm"),
            "dt": tail("dt"),
            "picard_iterations": tail("picard_iterations"),
            "coupled_defect": tail("coupled_defect"),
            "energy_gate_ratio": tail("energy_gate_ratio"),
        },
        "linear_tail_flatness_per_step": (
            None if len(tail("flux_flatness")) < 2
            else float(tail("flux_flatness")[0] - tail("flux_flatness")[-1]) / max(len(tail("flux_flatness")) - 1, 1)
        ),
        "profile": profile,
        "note": (
            "Diagnostic only. Formal spatial PASS still requires physically gated N=384. "
            "Do not restart from N=192 or loosen the 1e-3 physical gate."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "classification": classification,
        "flatness": rec.get("flux_flatness"),
        "max_abs_F_minus_Fint": payload["max_abs_F_minus_Fint"],
        "max_abs_heating": payload["max_abs_heating"],
        "rad_excess_rms": rad_excess_rms,
        "conv_excess_rms": conv_excess_rms,
        "linear_tail_flatness_per_step": payload["linear_tail_flatness_per_step"],
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()
