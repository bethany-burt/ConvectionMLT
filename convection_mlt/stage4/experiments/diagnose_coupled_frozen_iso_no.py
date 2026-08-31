"""Score frozen-T HELIOS radiation diagnostic (iso=no vs iso=yes on same MLT T).

Not radiation-only parity and not the coupled RCE score. Confirms finite,
sensibly oriented fluxes under isothermal_layers=no before the iterative run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import load_integrated_flux
from convection_mlt.adapters.helios import flux_cgs_to_si, to_canonical_interfaces
from convection_mlt.adapters.helios_contracts import F_INT
from export_coupled_helios_case import PLANCK_T_CEILING_K

RESULTS = ROOT / "results"


def _load_flux(path: Path, n_layers: int) -> dict:
    flux = load_integrated_flux(path)
    net = to_canonical_interfaces(
        flux_cgs_to_si(np.asarray(flux.flux_net_cgs, dtype=np.float64)),
        flux.pressure_microbar,
        n_layers=n_layers,
    )
    up = to_canonical_interfaces(
        flux_cgs_to_si(np.asarray(flux.flux_up_cgs, dtype=np.float64)),
        flux.pressure_microbar,
        n_layers=n_layers,
    )
    down = to_canonical_interfaces(
        flux_cgs_to_si(np.asarray(flux.flux_down_cgs, dtype=np.float64)),
        flux.pressure_microbar,
        n_layers=n_layers,
    )
    return {"net": net, "up": up, "down": down, "p": np.asarray(flux.pressure_microbar, dtype=np.float64)}


def analyze_frozen_iso_diag(
    *,
    n_layers: int,
    flux_iso_no: Path,
    flux_iso_yes: Path | None = None,
    f_int: float = F_INT,
) -> dict:
    iso_no = _load_flux(flux_iso_no, n_layers)
    checks = {
        "finite_net": bool(np.all(np.isfinite(iso_no["net"]))),
        "finite_up": bool(np.all(np.isfinite(iso_no["up"]))),
        "finite_down": bool(np.all(np.isfinite(iso_no["down"]))),
        "up_nonnegative": bool(np.all(iso_no["up"] >= -1.0e-6 * max(abs(f_int), 1.0))),
        "down_nonnegative": bool(np.all(iso_no["down"] >= -1.0e-6 * max(abs(f_int), 1.0))),
    }
    # HELIOS net orientation: deep/BOA net should be order F_int for F_irr=0.
    boa_net = float(iso_no["net"][0])
    toa_net = float(iso_no["net"][-1])
    scale = max(abs(f_int), 1.0)
    checks["boa_net_order_f_int"] = abs(boa_net - f_int) / scale < 2.0
    checks["toa_net_finite_scale"] = abs(toa_net) / scale < 50.0
    checks["no_planck_runaway_in_flux"] = float(np.nanmax(np.abs(iso_no["net"]))) < 1.0e6

    metrics = {
        "f_int": f_int,
        "boa_net_W_m2": boa_net,
        "toa_net_W_m2": toa_net,
        "max_abs_net_W_m2": float(np.nanmax(np.abs(iso_no["net"]))),
        "max_abs_up_W_m2": float(np.nanmax(np.abs(iso_no["up"]))),
        "max_abs_down_W_m2": float(np.nanmax(np.abs(iso_no["down"]))),
        "planck_t_ceiling_k": PLANCK_T_CEILING_K,
    }
    delta = None
    if flux_iso_yes is not None and flux_iso_yes.exists():
        iso_yes = _load_flux(flux_iso_yes, n_layers)
        n = min(iso_no["net"].size, iso_yes["net"].size)
        diff = iso_no["net"][:n] - iso_yes["net"][:n]
        denom = np.maximum.reduce(
            [np.abs(iso_no["net"][:n]), np.abs(iso_yes["net"][:n]), np.full(n, scale)]
        )
        rel = np.abs(diff) / denom
        i = int(np.argmax(rel))
        delta = {
            "max_abs_net_delta_W_m2": float(np.max(np.abs(diff))),
            "max_rel_net": float(rel[i]),
            "max_rel_interface": i,
            "iso_no_at_max": float(iso_no["net"][i]),
            "iso_yes_at_max": float(iso_yes["net"][i]),
            "note": (
                "Same frozen MLT F_irr=0 T(P); only isothermal_layers changed. "
                "Not required to match prior radiation-only parity."
            ),
        }
        metrics["iso_yes_boa_net_W_m2"] = float(iso_yes["net"][0])
        metrics["iso_yes_toa_net_W_m2"] = float(iso_yes["net"][-1])

    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "purpose": "frozen_T_iso_no_radiation_preflight",
        "n_layers": n_layers,
        "checks": checks,
        "metrics": metrics,
        "iso_yes_vs_iso_no": delta,
        "helios_coupled_rce_n96_status": "NOT_RUN",
        "full_stage4_claim": False,
        "note": (
            "Preflight only. Quantifies iso layer-source change and confirms "
            "finite oriented fluxes before iterative coupled HELIOS."
        ),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument("--flux-iso-no", type=Path, required=True)
    parser.add_argument("--flux-iso-yes", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = analyze_frozen_iso_diag(
        n_layers=args.layers,
        flux_iso_no=args.flux_iso_no,
        flux_iso_yes=args.flux_iso_yes,
    )
    out = args.output or RESULTS / f"helios_coupled_frozen_iso_diag_n{args.layers}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "out": str(out), **{k: payload["checks"][k] for k in payload["checks"]}}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(4)
    return payload


if __name__ == "__main__":
    main()
