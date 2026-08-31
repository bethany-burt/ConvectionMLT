"""Score coupled HELIOS RCE against the F_irr=0 nested τ-grid MLT reference.

Agreement between independently discretized RCE solutions with matched forcing,
opacity law, gravity, EOS/adiabatic gradient, and lower net flux. HELIOS
evolves on its geometric grid; the MLT reference is nested-τ. This is a
benchmark of convection closures, not grid-level parity. Irradiated nested MLT
is a structural diagnostic only.

N=96 is the pilot. Headline and full_stage4_claim require N=192; this scorer
never sets full_stage4_claim. Infrastructure failures return
INFRASTRUCTURE_FAIL and do not evaluate physical gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import load_integrated_flux
from convection_mlt.adapters.helios import flux_cgs_to_si, load_tp_profile, to_canonical_interfaces
from convection_mlt.adapters.helios_contracts import F_INT, MICROBAR_TO_PA, helios_track_status
from export_helios_grid_reference import _load_record

FIXTURES = ROOT / "fixtures" / "helios"
TOLERANCES = FIXTURES / "coupled_rce_benchmark_tolerances.json"
RESULTS = ROOT / "results"
MLT_REF = {
    96: FIXTURES / "mlt_nested_tau_n96_firr0.json",
    192: FIXTURES / "mlt_nested_tau_n192_firr0.json",
}
BENCHMARK_NOTE = (
    "Agreement between independently discretized RCE solutions with matched "
    "forcing, opacity law, gravity, EOS/adiabatic gradient, and lower net flux. "
    "HELIOS geometric grid vs nested-τ MLT; not grid-level parity."
)


def interpolate_temperature(log_p_src, t_src, log_p_dst):
    """Linear interp in log-P; endpoints only — prefer common_domain helper for scoring."""
    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def interpolate_temperature_common_domain(log_p_src, t_src, log_p_dst):
    """Interpolate only on the overlapping log-P interval; no endpoint extrapolation."""
    order = np.argsort(np.asarray(log_p_src, dtype=np.float64))
    src = np.asarray(log_p_src, dtype=np.float64)[order]
    t = np.asarray(t_src, dtype=np.float64)[order]
    dst = np.asarray(log_p_dst, dtype=np.float64)
    lo, hi = float(src[0]), float(src[-1])
    mask = (dst >= lo) & (dst <= hi) & np.isfinite(dst)
    out = np.full(dst.shape, np.nan, dtype=np.float64)
    if not np.any(mask):
        raise ValueError(
            f"no overlapping log-P domain: src=[{lo}, {hi}], "
            f"dst=[{float(np.nanmin(dst))}, {float(np.nanmax(dst))}]"
        )
    out[mask] = np.interp(dst[mask], src, t)
    return out, mask


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mlt_reference(n_layers: int) -> dict:
    path = MLT_REF[n_layers]
    if not path.exists():
        raise FileNotFoundError(
            f"missing F_irr=0 nested τ-grid MLT reference {path}. "
            "Continue the nested F_irr=0 solve through the 1e-3 gate and freeze it."
        )
    rec = json.loads(path.read_text())
    if "f_irr" not in rec or float(rec["f_irr"]) != 0.0:
        raise ValueError(f"{path} is not an F_irr=0 reference")
    return rec


def helios_total_flux_si(flux) -> dict:
    """F_total = F_rad,net + F_conv at every interface, SI W m^-2."""
    n = int(flux.interface_index.size)
    rad = flux_cgs_to_si(np.asarray(flux.flux_net_cgs, dtype=np.float64))
    conv = flux_cgs_to_si(np.asarray(flux.flux_conv_net_cgs, dtype=np.float64))
    intern = flux_cgs_to_si(np.asarray(flux.flux_intern_cgs, dtype=np.float64))
    p = np.asarray(flux.pressure_microbar, dtype=np.float64)
    rad = to_canonical_interfaces(rad, p, n_layers=n - 1)
    conv = to_canonical_interfaces(conv, p, n_layers=n - 1)
    intern = to_canonical_interfaces(intern, p, n_layers=n - 1)
    if not np.all(np.isfinite(rad)):
        raise ValueError("HELIOS F_net has non-finite values")
    if not np.all(np.isfinite(conv)):
        raise ValueError(
            "HELIOS F_net_conv is required for coupled total-flux scoring; "
            "got non-finite values"
        )
    total = rad + conv
    intern_boa = float(intern[0]) if np.isfinite(intern[0]) else float("nan")
    return {
        "rad": rad,
        "conv": conv,
        "total": total,
        "intern_boa": intern_boa,
        "pressure_microbar": to_canonical_interfaces(p, p, n_layers=n - 1),
        "n_interfaces": n,
    }


def flux_column_metrics(total: np.ndarray, f_int: float, intern_boa: float) -> dict:
    scale = max(abs(f_int), 1.0)
    toa = float(total[-1])
    boa = float(total[0])
    flat = np.abs(total - f_int) / scale
    imax = int(np.argmax(flat))
    dF = np.diff(np.asarray(total, dtype=np.float64))
    i_div = int(np.argmax(np.abs(dF))) if dF.size else 0
    return {
        "toa_total_flux_rel": abs(toa - f_int) / scale,
        "boa_total_flux_rel": abs(boa - f_int) / scale,
        "max_column_flatness": float(flat[imax]),
        "max_column_flatness_interface": imax,
        "column_closure_rel": abs(boa - toa) / scale,
        "max_layer_total_flux_divergence_rel": (
            float(np.max(np.abs(dF)) / scale) if dF.size else 0.0
        ),
        "max_layer_total_flux_divergence_interface": i_div,
        "f_intern_rel": (
            abs(intern_boa - f_int) / scale if np.isfinite(intern_boa) else float("nan")
        ),
        "f_intern_W_m2": intern_boa,
        "f_total_toa_W_m2": toa,
        "f_total_boa_W_m2": boa,
        "energy_closure_rel": abs(boa - toa) / scale,
    }


def radiative_zone_flux_residual(
    rad: np.ndarray,
    conv: np.ndarray,
    f_int: float,
    *,
    conv_threshold_frac: float = 1.0e-4,
    radiative_mask: np.ndarray | None = None,
) -> dict:
    """Independent radiative-zone layer residual on low-F_conv interfaces."""
    rad = np.asarray(rad, dtype=np.float64)
    conv = np.asarray(conv, dtype=np.float64)
    scale = max(abs(f_int), 1.0)
    if radiative_mask is None:
        rad_mask = np.abs(conv) <= conv_threshold_frac * scale
    else:
        rad_mask = np.asarray(radiative_mask, dtype=bool)
        if rad_mask.shape != rad.shape:
            n = min(rad_mask.size, rad.size)
            rad_mask = rad_mask[:n]
            rad = rad[:n]
            conv = conv[:n]
    if int(np.sum(rad_mask)) < 2:
        return {
            "n_radiative_interfaces": int(np.sum(rad_mask)),
            "max_abs_dF_rad_W_m2": None,
            "max_abs_dF_rad_rel": None,
            "conv_threshold_frac": conv_threshold_frac,
            "note": "fewer than 2 radiative interfaces under conv threshold",
        }
    idx = np.flatnonzero(rad_mask)
    deltas = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            deltas.append(abs(float(rad[b] - rad[a])))
    if not deltas:
        return {
            "n_radiative_interfaces": int(np.sum(rad_mask)),
            "max_abs_dF_rad_W_m2": None,
            "max_abs_dF_rad_rel": None,
            "conv_threshold_frac": conv_threshold_frac,
            "note": "no contiguous radiative interface pairs",
        }
    max_abs = float(np.max(deltas))
    return {
        "n_radiative_interfaces": int(np.sum(rad_mask)),
        "n_contiguous_pairs": len(deltas),
        "max_abs_dF_rad_W_m2": max_abs,
        "max_abs_dF_rad_rel": max_abs / scale,
        "conv_threshold_frac": conv_threshold_frac,
    }


def _mlt_topology(rec: dict) -> dict:
    regions = rec.get("convective_regions") or []
    detached = rec.get("detached_convective_regions") or []
    bottom = [r for r in regions if r and r[0] == 0]
    return {
        "convective_regions": regions,
        "detached_convective_regions": detached,
        "single_bottom_cz": len(bottom) == 1 and len(detached) == 0,
        "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
    }


def _helios_rcb_and_topology(tp, conv_si: np.ndarray, f_int: float) -> dict:
    p_pa = np.asarray(tp.pressure_microbar, dtype=np.float64) * MICROBAR_TO_PA
    flag_u = np.asarray(tp.conv_unstable_flag, dtype=np.float64)
    flag_l = np.asarray(tp.conv_lapse_flag, dtype=np.float64)
    lay = tp.layer_index != -1
    p_lay = p_pa[lay]
    # HELIOS coupled runs often leave conv.unstable?=0 while conv.lapse-rate?=1
    # marks the adjusted convective zone (see job 16015698 layers 0–20).
    flag_lay = flag_u[lay]
    lapse_lay = flag_l[lay]
    if np.any(np.isfinite(flag_lay) & (flag_lay > 0.5)):
        unstable = flag_lay > 0.5
        marker = "conv_unstable_flag"
    elif np.any(np.isfinite(lapse_lay) & (lapse_lay > 0.5)):
        unstable = lapse_lay > 0.5
        marker = "conv_lapse_flag"
    else:
        n = min(conv_si.size, p_lay.size)
        # Interface F_conv → layer activity via adjacent interfaces.
        unstable = np.abs(conv_si[:n]) > 0.01 * abs(f_int)
        # Prefer layer centres: if conv is on interfaces (n_layers+1), use max of bounding ifaces.
        if conv_si.size == p_lay.size + 1:
            unstable = np.maximum(np.abs(conv_si[:-1]), np.abs(conv_si[1:])) > 0.01 * abs(f_int)
            p_lay = p_lay
        else:
            p_lay = p_lay[:n]
        marker = "f_conv_threshold"
    if not np.any(unstable):
        return {
            "primary_rcb_log10p": None,
            "single_bottom_cz": False,
            "n_detached": None,
            "n_unstable": 0,
            "marker": marker,
        }
    i0 = int(np.argmax(unstable))
    i_hi = i0
    while i_hi + 1 < unstable.size and unstable[i_hi + 1]:
        i_hi += 1
    detached = bool(np.any(unstable[i_hi + 1 :])) if i_hi + 1 < unstable.size else False
    return {
        "primary_rcb_log10p": float(np.log10(float(p_lay[i_hi]))),
        "single_bottom_cz": i0 == 0 and not detached,
        "n_detached": int(detached),
        "n_unstable": int(np.sum(unstable)),
        "marker": marker,
        "convective_layer_range": [i0, i_hi],
    }


def _coupled_status(n_layers: int, this_status: str) -> dict:
    n96_path = RESULTS / "helios_coupled_rce_n96.json"
    n192_path = RESULTS / "helios_coupled_rce_n192.json"
    scored = this_status if this_status in ("PASS", "FAIL") else "NOT_RUN"

    def _load(path: Path) -> str:
        if not path.exists():
            return "NOT_RUN"
        st = json.loads(path.read_text()).get("status", "NOT_RUN")
        return st if st in ("PASS", "FAIL") else "NOT_RUN"

    n96 = scored if n_layers == 96 else _load(n96_path)
    n192 = scored if n_layers == 192 else _load(n192_path)
    if this_status not in ("PASS", "FAIL"):
        if n_layers == 96:
            n96 = "NOT_RUN"
        else:
            n192 = "NOT_RUN"
    return helios_track_status(adapter_contract="PASS", coupled_n96=n96, coupled_n192=n192)


def detect_helios_convergence(
    *,
    case_dir: Path | None,
    case_name: str | None,
    helios_log: Path | None,
    declared_criterion: float = 1.0e-8,
) -> dict:
    """HELIOS-native coupled completion evidence.

    Full per-layer radiative counters are *not* required for the rad–conv path.
    Accept: Done + global energy imbalance below the declared criterion, or
    legacy full-layer / coupling_convergence markers. Preserve a 0/N radiative
    counter as ``PASS_WITH_DIAGNOSTIC_WARNING``.
    """
    reasons: list[str] = []
    coupling = None
    warnings: list[str] = []
    if case_dir is not None and case_name:
        candidates = [
            case_dir / case_name / f"{case_name}_coupling_convergence.dat",
            case_dir / f"{case_name}_coupling_convergence.dat",
        ]
        for path in candidates:
            if path.exists():
                text = path.read_text().strip()
                coupling = {"path": str(path), "value": text}
                if text.split()[0] == "1":
                    return {
                        "converged": True,
                        "helios_native_convergence": "PASS",
                        "source": "coupling_convergence.dat",
                        "termination_mode": "coupling_convergence_dat",
                        "warnings": warnings,
                        **coupling,
                    }
                reasons.append(f"coupling_convergence.dat={text!r}")
                break

    if helios_log is None or not helios_log.exists():
        if helios_log is not None:
            reasons.append(f"missing helios log {helios_log}")
        if not reasons:
            reasons.append(
                "no coupling_convergence.dat and no helios log; "
                "pass --helios-log or keep the HELIOS case directory"
            )
        return {"converged": False, "reasons": reasons, "coupling": coupling}

    text = helios_log.read_text(errors="replace")
    done = bool(
        re.search(r"Done!\s*Everything appears to have worked fine", text)
        or re.search(r"Done!\s*Everything appears to have worked", text)
    )
    abort = "Aborting" in text or "Traceback (most recent call last):" in text

    # Last radiative-layer counter in the rad–conv section (bookkeeping only).
    rad_counts = re.findall(
        r"Number of radiative layers converged:\s*(\d+)\s*out of\s*(\d+)", text
    )
    radiative_layer_counter_final = None
    if rad_counts:
        a, b = rad_counts[-1]
        radiative_layer_counter_final = f"{int(a)}/{int(b)}"
        if int(a) != int(b):
            warnings.append(
                f"radiative_layer_counter_final = {radiative_layer_counter_final} "
                "(HELIOS bookkeeping; not sole convergence decision)"
            )

    # Last pure-radiative layer counter (before convection), for provenance.
    layer_counts = re.findall(
        r"Layers\s*\(&\s*surface/BOA\)\s*converged:\s*(\d+)\s*out of\s*(\d+)", text
    )
    pure_rad_counter_final = None
    if layer_counts:
        a, b = layer_counts[-1]
        pure_rad_counter_final = f"{int(a)}/{int(b)}"

    imb_vals = [
        float(x)
        for x in re.findall(
            r"Global energy imbalance is\s*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)",
            text,
        )
    ]
    final_ppm = None
    m_ppm = re.search(
        r"Global energy imbalance:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))\s*ppm", text
    )
    if m_ppm:
        final_ppm = float(m_ppm.group(1))
    last_imbalance = imb_vals[-1] if imb_vals else None
    if last_imbalance is None and final_ppm is not None:
        last_imbalance = abs(final_ppm) * 1.0e-6

    # Legacy full-layer success.
    if layer_counts and int(layer_counts[-1][0]) == int(layer_counts[-1][1]):
        return {
            "converged": True,
            "helios_native_convergence": "PASS",
            "source": "helios_log_layers",
            "termination_mode": "pure_radiative_full_layers",
            "converged_layers": int(layer_counts[-1][0]),
            "n_expected": int(layer_counts[-1][1]),
            "radiative_layer_counter_final": radiative_layer_counter_final,
            "warnings": warnings,
            "log": str(helios_log),
        }
    if rad_counts and int(rad_counts[-1][0]) == int(rad_counts[-1][1]):
        return {
            "converged": True,
            "helios_native_convergence": "PASS",
            "source": "helios_log_radiative",
            "termination_mode": "radconv_full_radiative_layers",
            "radiative_layer_counter_final": radiative_layer_counter_final,
            "warnings": warnings,
            "log": str(helios_log),
        }

    # Coupled rad–conv: Done + global imbalance ≤ declared criterion.
    # Prefer the final ppm summary (HELIOS may exit while the last loop print
    # is still slightly above criterion).
    imbalance_ok = False
    if final_ppm is not None and abs(final_ppm) * 1.0e-6 <= declared_criterion:
        imbalance_ok = True
        last_imbalance = abs(final_ppm) * 1.0e-6
    elif last_imbalance is not None and last_imbalance <= declared_criterion:
        imbalance_ok = True

    if done and not abort and imbalance_ok:
        status = "PASS_WITH_DIAGNOSTIC_WARNING" if warnings else "PASS"
        return {
            "converged": True,
            "helios_native_convergence": status,
            "source": "helios_log_radconv_global_imbalance",
            "termination_mode": "radconv_global_energy_imbalance",
            "global_energy_imbalance": last_imbalance,
            "global_energy_imbalance_ppm": final_ppm,
            "declared_criterion": declared_criterion,
            "radiative_layer_counter_final": radiative_layer_counter_final,
            "pure_radiative_counter_final": pure_rad_counter_final,
            "warnings": warnings,
            "log": str(helios_log),
        }

    if done and not abort:
        reasons.append(
            "HELIOS printed Done but global energy imbalance "
            f"{last_imbalance} (ppm={final_ppm}) exceeds declared criterion "
            f"{declared_criterion}"
        )
    elif abort:
        reasons.append("HELIOS aborted or raised a traceback")
    else:
        reasons.append(f"no HELIOS Done / convergence evidence in {helios_log}")
    return {
        "converged": False,
        "reasons": reasons,
        "coupling": coupling,
        "radiative_layer_counter_final": radiative_layer_counter_final,
        "global_energy_imbalance": last_imbalance,
        "warnings": warnings,
    }


def score_structural_irradiated(n_layers: int, rec_firr0: dict) -> dict:
    """Optional irradiated nested-MLT diagnostic. Missing inputs are not a failure."""
    try:
        nested = _load_record(n_layers)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "STRUCTURAL_NOT_SCORED",
            "note": (
                "Irradiated nested MLT diagnostic skipped: nested_rce_family "
                f"(or N={n_layers} member) unavailable ({exc}). "
                "Not required for the coupled F_irr=0 HELIOS benchmark."
            ),
            "nested_f_irr": None,
            "reference_f_irr": rec_firr0.get("f_irr"),
            "max_rel_T": None,
            "n_common_pressure": 0,
            "rcb_dex": None,
            "nested_checksum": None,
            "reference_checksum": rec_firr0.get("profile_checksum_sha256"),
        }
    p0 = np.asarray(rec_firr0["pressure_centres"], dtype=np.float64)
    t0 = np.asarray(rec_firr0["temperature"], dtype=np.float64)
    log0 = np.log(p0)
    t_n, mask = interpolate_temperature_common_domain(
        np.log(np.asarray(nested["pressure_centres"], dtype=np.float64)),
        nested["temperature"],
        log0,
    )
    rel = np.abs(t_n[mask] - t0[mask]) / np.maximum(np.abs(t0[mask]), 1.0)
    rcb0 = rec_firr0.get("primary_rcb_log10p")
    rcbn = nested.get("primary_rcb_log10p")
    return {
        "status": "STRUCTURAL_NOT_SCORED",
        "note": (
            "Irradiated nested MLT (F_irr=120) vs F_irr=0 nested-τ MLT. "
            "Different boundary forcing; not the coupled HELIOS benchmark."
        ),
        "nested_f_irr": nested.get("f_irr"),
        "reference_f_irr": rec_firr0.get("f_irr"),
        "max_rel_T": float(np.max(rel)) if rel.size else None,
        "n_common_pressure": int(np.sum(mask)),
        "rcb_dex": (
            None if rcb0 is None or rcbn is None else abs(float(rcbn) - float(rcb0))
        ),
        "nested_checksum": nested.get("profile_checksum_sha256"),
        "reference_checksum": rec_firr0.get("profile_checksum_sha256"),
    }


def find_helios_abort(case_dir: Path | None, case_name: str | None = None) -> Path | None:
    """Return the HELIOS *_ABORT.dat path if present under the case directory."""
    if case_dir is None:
        return None
    case_dir = Path(case_dir)
    candidates: list[Path] = []
    if case_name:
        candidates.extend(
            [
                case_dir / case_name / f"{case_name}_ABORT.dat",
                case_dir / f"{case_name}_ABORT.dat",
            ]
        )
    candidates.extend(sorted(case_dir.glob("*_ABORT.dat")))
    candidates.extend(sorted(case_dir.glob("*/*_ABORT.dat")))
    for path in candidates:
        if path.is_file():
            return path
    return None


def helios_abort_payload(
    *,
    n_layers: int,
    abort_path: Path | None,
    helios_tp: Path | None,
    helios_flux: Path | None,
    helios_log: Path | None = None,
    runtime: dict | None = None,
    structural: dict | None = None,
) -> dict:
    """Abort/crash without final outputs: physical gates were not evaluated."""
    abort_text = None
    if abort_path is not None and abort_path.exists():
        abort_text = abort_path.read_text(errors="replace").strip()
    log_text = ""
    if helios_log is not None and Path(helios_log).exists():
        log_text = Path(helios_log).read_text(errors="replace")
    traceback_hit = "Traceback (most recent call last):" in log_text
    iso_conv_crash = (
        "sum(quant.conv_unstable)" in log_text
        or "conv_unstable" in log_text and "TypeError" in log_text
    )
    if traceback_hit:
        status = "HELIOS_CRASH"
        failure_stage = "helios_runtime"
        note = (
            "HELIOS exited with a Python traceback before writing final TP/"
            "integrated-flux outputs. Physical coupled gates were not scored. "
            + BENCHMARK_NOTE
        )
        if iso_conv_crash:
            note = (
                "HELIOS convection_loop crashed with conv_unstable=None "
                "(isothermal layers=yes / iso==1 skips conv_check). "
                "Physical coupled gates were not scored. " + BENCHMARK_NOTE
            )
            failure_stage = "helios_iso_convection_incompatible"
    else:
        status = "HELIOS_ABORT"
        failure_stage = "helios_convergence"
        note = (
            "HELIOS aborted or did not write final TP/integrated-flux outputs. "
            "Physical coupled gates were not scored. " + BENCHMARK_NOTE
        )
    payload = {
        "comparison_type": "independently_discretized_rce_matched_forcing",
        "forcing": "F_int=300, F_irr=0",
        "n_layers": n_layers,
        "status": status,
        "execution_status": status,
        "failure_stage": failure_stage,
        "helios_abort_path": None if abort_path is None else str(abort_path),
        "helios_abort_text": abort_text,
        "helios_traceback": traceback_hit,
        "helios_tp_present": bool(helios_tp is not None and helios_tp.exists()),
        "helios_flux_present": bool(helios_flux is not None and helios_flux.exists()),
        "helios_log": None if helios_log is None else str(helios_log),
        "helios_runtime_config": runtime or {},
        "structural_irradiated": structural
        or {"status": "STRUCTURAL_NOT_SCORED", "note": "not evaluated on abort path"},
        "note": note,
        **_coupled_status(n_layers, "NOT_RUN"),
        "full_stage4_claim": False,
    }
    return payload


def _infrastructure_payload(base: dict, reason: str, details: dict | None = None) -> dict:
    out = dict(base)
    out.update({
        "status": "INFRASTRUCTURE_FAIL",
        "infrastructure_reason": reason,
        "infrastructure_details": details or {},
        "note": (
            "Infrastructure or I/O failure: physical gates were not evaluated. "
            + BENCHMARK_NOTE
        ),
        **_coupled_status(int(base["n_layers"]), "INFRASTRUCTURE_FAIL"),
        "full_stage4_claim": False,
    })
    return out


def score(
    *,
    n_layers: int,
    helios_tp: Path | None,
    helios_flux: Path | None,
    tolerances: dict,
    runtime: dict | None = None,
    helios_log: Path | None = None,
    case_dir: Path | None = None,
    case_name: str | None = None,
) -> dict:
    rec = load_mlt_reference(n_layers)
    gates = tolerances.get("gates") or {}
    mlt_topo = _mlt_topology(rec)
    structural = score_structural_irradiated(n_layers, rec)
    payload = {
        "comparison_type": "independently_discretized_rce_matched_forcing",
        "forcing": "F_int=300, F_irr=0",
        "mlt_grid": rec.get("mlt_grid") or "nested_tau",
        "helios_grid": "helios_geometric",
        "benchmark_interpretation": BENCHMARK_NOTE,
        "frozen_before_live": tolerances.get("frozen_before_live"),
        "n_layers": n_layers,
        "expected_n_interfaces": n_layers + 1,
        "mlt_reference": str(MLT_REF[n_layers]),
        "mlt_reference_file_sha256": file_sha256(MLT_REF[n_layers]),
        "mlt_profile_checksum_sha256": rec.get("profile_checksum_sha256") or rec.get("checksum_sha256"),
        "mlt_rcb_log10p": rec.get("primary_rcb_log10p"),
        "mlt_topology": mlt_topo,
        "mlt_f_irr": rec.get("f_irr"),
        "gates": gates,
        "helios_runtime_config": runtime or {},
        "status": "NOT_RUN",
        "structural_irradiated": structural,
        **_coupled_status(n_layers, "NOT_RUN"),
    }
    if case_dir is None and helios_tp is not None:
        case_dir = helios_tp.parent.parent if helios_tp.parent.name != "." else helios_tp.parent
    if case_name is None and helios_tp is not None:
        case_name = helios_tp.stem.replace("_tp", "")

    abort_path = find_helios_abort(case_dir, case_name)
    tp_ok = helios_tp is not None and helios_tp.exists()
    flux_ok = helios_flux is not None and helios_flux.exists()
    if abort_path is not None:
        abort = helios_abort_payload(
            n_layers=n_layers,
            abort_path=abort_path,
            helios_tp=helios_tp,
            helios_flux=helios_flux,
            helios_log=helios_log,
            runtime=runtime,
            structural=structural,
        )
        abort.update({
            "mlt_reference": payload["mlt_reference"],
            "mlt_profile_checksum_sha256": payload["mlt_profile_checksum_sha256"],
            "mlt_f_irr": payload["mlt_f_irr"],
            "gates": gates,
            "frozen_before_live": tolerances.get("frozen_before_live"),
            "benchmark_interpretation": BENCHMARK_NOTE,
        })
        return abort

    if not tp_ok or not flux_ok:
        payload["note"] = (
            "HELIOS iterative output not present. " + BENCHMARK_NOTE
            + " Irradiated nested MLT is structural only."
        )
        return payload

    conv = detect_helios_convergence(
        case_dir=case_dir,
        case_name=case_name,
        helios_log=helios_log,
        declared_criterion=float(
            (runtime or {}).get("radiative_equilibrium_criterion")
            or gates.get("helios_global_imbalance")
            or 1.0e-8
        ),
    )
    if not conv.get("converged"):
        return _infrastructure_payload(
            payload, "helios_not_converged", {"convergence": conv}
        )

    try:
        tp = load_tp_profile(helios_tp)
        flux = load_integrated_flux(helios_flux)
        fluxes = helios_total_flux_si(flux)
    except ValueError as exc:
        return _infrastructure_payload(payload, "helios_io_invalid", {"error": str(exc)})

    expected_ifaces = n_layers + 1
    if int(fluxes["n_interfaces"]) != expected_ifaces:
        return _infrastructure_payload(
            payload,
            "unexpected_interface_count",
            {
                "expected_n_interfaces": expected_ifaces,
                "observed_n_interfaces": int(fluxes["n_interfaces"]),
            },
        )

    f_int = float(rec.get("f_int") or F_INT)
    flux_metrics = flux_column_metrics(fluxes["total"], f_int, fluxes["intern_boa"])
    # Prefer HELIOS conv-unstable flags for radiative-zone residual when present.
    rad_mask = None
    try:
        tp_probe = load_tp_profile(helios_tp)
        flag = np.asarray(tp_probe.conv_unstable_flag, dtype=np.float64)
        if flag.size == fluxes["conv"].size and np.any(np.isfinite(flag)):
            # Interfaces: map layer flags conservatively — radiative where not unstable.
            # Use |F_conv| small as primary; flags as soft prior via AND.
            rad_mask = (np.abs(fluxes["conv"]) <= 1.0e-4 * max(abs(f_int), 1.0)) & (
                ~(np.nan_to_num(flag, nan=0.0) > 0.5)
            )
    except Exception:  # noqa: BLE001
        rad_mask = None
    rad_zone = radiative_zone_flux_residual(
        fluxes["rad"], fluxes["conv"], f_int, radiative_mask=rad_mask
    )
    # Independent confirmation of HELIOS-native Done: total-flux column balance.
    # Radiative-zone residual is recorded as a diagnostic, not a hard gate here.
    declared = float(conv.get("declared_criterion") or 1.0e-8)
    balance_tol = max(declared * 10.0, 1.0e-6)
    independent_ok = (
        flux_metrics["max_column_flatness"] <= balance_tol
        and flux_metrics["column_closure_rel"] <= balance_tol
        and flux_metrics["boa_total_flux_rel"] <= balance_tol
    )
    if not independent_ok:
        return _infrastructure_payload(
            payload,
            "helios_done_but_independent_flux_unbalanced",
            {
                "convergence": conv,
                "flux_metrics": flux_metrics,
                "radiative_zone_residual": rad_zone,
                "balance_tol": balance_tol,
            },
        )

    p_compare = np.asarray(
        rec.get("helios_p_lay_Pa") or rec["pressure_centres"], dtype=np.float64
    )
    t_mlt_on_compare, mask_mlt = interpolate_temperature_common_domain(
        np.log(np.asarray(rec["pressure_centres"], dtype=np.float64)),
        rec["temperature"],
        np.log(p_compare),
    )
    lay = tp.layer_index != -1
    p_h = np.asarray(tp.pressure_microbar[lay], dtype=np.float64) * MICROBAR_TO_PA
    t_h = np.asarray(tp.temperature_k[lay], dtype=np.float64)
    t_h_on_compare, mask_h = interpolate_temperature_common_domain(
        np.log(p_h), t_h, np.log(p_compare)
    )
    mask = mask_mlt & mask_h & np.isfinite(t_mlt_on_compare) & np.isfinite(t_h_on_compare)
    if int(np.sum(mask)) < max(8, n_layers // 4):
        return _infrastructure_payload(
            payload,
            "insufficient_common_pressure_domain",
            {"n_common": int(np.sum(mask)), "n_compare": int(p_compare.size)},
        )
    rel = np.abs(t_h_on_compare[mask] - t_mlt_on_compare[mask]) / np.maximum(
        np.abs(t_mlt_on_compare[mask]), 1.0
    )
    imax_local = int(np.argmax(rel))
    compare_idx = np.flatnonzero(mask)[imax_local]
    helios_topo = _helios_rcb_and_topology(tp, fluxes["conv"], f_int)
    rcb_mlt = rec.get("primary_rcb_log10p")
    rcb_h = helios_topo.get("primary_rcb_log10p")
    rcb_dex = None if rcb_mlt is None or rcb_h is None else abs(float(rcb_h) - float(rcb_mlt))
    metrics = {
        **flux_metrics,
        "toa_flux_rel": flux_metrics["toa_total_flux_rel"],
        "max_rel_T": float(rel[imax_local]),
        "max_rel_T_index": int(compare_idx),
        "max_rel_T_pressure": float(p_compare[compare_idx]),
        "n_common_pressure": int(np.sum(mask)),
        "temperature_domain": "common_logP_no_extrapolation",
        "rcb_dex": rcb_dex,
        "helios_rcb_log10p": rcb_h,
        "topology_single_bottom_cz": bool(
            mlt_topo["single_bottom_cz"] and helios_topo.get("single_bottom_cz")
        ),
        "no_detached_convective_regions": bool(
            not (rec.get("detached_convective_regions") or [])
            and not helios_topo.get("n_detached")
        ),
        "n_interfaces": int(fluxes["n_interfaces"]),
        "helios_converged": True,
        "helios_convergence": conv,
        "radiative_zone_residual": rad_zone,
    }
    if not helios_topo.get("single_bottom_cz"):
        metrics["no_bottom_cz_note"] = (
            "No bottom-connected HELIOS convection: treat first as a "
            "geometric-grid / optical-depth-resolution result, given the failed "
            "MLT log-P experiment — not automatically as a convection-closure disagreement."
        )
    intern_ok = np.isfinite(metrics["f_intern_rel"]) and metrics["f_intern_rel"] <= float(
        gates["energy_closure_rel"]
    )
    checks = {
        "toa_flux_rel": metrics["toa_total_flux_rel"] <= float(gates["toa_flux_rel"]),
        "boa_total_flux_rel": metrics["boa_total_flux_rel"] <= float(gates["toa_flux_rel"]),
        "max_column_flatness": metrics["max_column_flatness"] <= float(gates["toa_flux_rel"]),
        "max_rel_T": metrics["max_rel_T"] <= float(gates["max_rel_T"]),
        "rcb_dex": rcb_dex is not None and rcb_dex <= float(gates["rcb_dex"]),
        "topology_single_bottom_cz": metrics["topology_single_bottom_cz"]
        == bool(gates.get("topology_single_bottom_cz", True)),
        "no_detached_convective_regions": metrics["no_detached_convective_regions"]
        == bool(gates.get("no_detached_convective_regions", True)),
        "energy_closure_rel": metrics["column_closure_rel"] <= float(gates["energy_closure_rel"]),
        "f_intern_rel": intern_ok,
    }
    passed = all(checks.values())
    this_status = "PASS" if passed else "FAIL"
    payload.update({
        "status": this_status,
        "execution_status": "HELIOS_OK",
        "helios_executable_infrastructure": "PASS",
        "helios_native_coupled_convergence": conv.get(
            "helios_native_convergence", "PASS"
        ),
        "helios_convergence_warnings": conv.get("warnings") or [],
        "termination_mode": conv.get("termination_mode"),
        "metrics": metrics,
        "checks": checks,
        "helios_topology": helios_topo,
        "helios_tp": str(helios_tp),
        "helios_flux": str(helios_flux),
        **_coupled_status(n_layers, this_status),
        "full_stage4_claim": False,
        "note": (
            BENCHMARK_NOTE
            + " Physical gates evaluated. full_stage4_claim is never set here."
            + (
                " HELIOS-native convergence carries diagnostic warning(s): "
                + "; ".join(conv.get("warnings") or [])
                if conv.get("warnings")
                else ""
            )
        ),
    })
    return payload


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96, choices=(96, 192))
    parser.add_argument("--helios-tp", type=Path, default=None)
    parser.add_argument("--helios-flux", type=Path, default=None)
    parser.add_argument("--helios-log", type=Path, default=None)
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--case-name", type=str, default=None)
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--tolerances", type=Path, default=TOLERANCES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tols = json.loads(args.tolerances.read_text())
    runtime = json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    payload = score(
        n_layers=args.layers,
        helios_tp=args.helios_tp,
        helios_flux=args.helios_flux,
        tolerances=tols,
        runtime=runtime,
        helios_log=args.helios_log,
        case_dir=args.case_dir,
        case_name=args.case_name,
    )
    out = args.output or RESULTS / f"helios_coupled_rce_n{args.layers}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "helios_coupled_rce_n96_status": payload.get("helios_coupled_rce_n96_status"),
        "helios_coupled_rce_n192_status": payload.get("helios_coupled_rce_n192_status"),
        "helios_coupled_rce_status": payload["helios_coupled_rce_status"],
        "helios_parity_headline": payload["helios_parity_headline"],
        "full_stage4_claim": payload["full_stage4_claim"],
        "infrastructure_reason": payload.get("infrastructure_reason"),
        "out": str(out),
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()
