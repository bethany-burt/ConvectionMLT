"""Compare HELIOS _opacities.dat to the intended layer κ and Δτ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np

from convection_mlt import nested_analytic_opacity_spec
from convection_mlt.adapters.helios_contracts import (
    HELIOS_DEFAULT_DIFFUSIVITY,
    MICROBAR_TO_PA,
    OPACITY_SI_TO_CGS,
)


def load_helios_opacities_dat(path: Path) -> dict:
    """Parse HELIOS write_opacities output (bin rows, layer columns)."""
    rows = []
    n_layers = None
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] in ("This", "Opacity", "bin"):
                if parts and parts[0] == "bin":
                    n_layers = sum(1 for p in parts if p.startswith("opac_lay"))
                continue
            try:
                bin_idx = int(float(parts[0]))
            except ValueError:
                continue
            values = [float(x) for x in parts[4:]]
            if n_layers is None:
                n_layers = len(values)
            rows.append((bin_idx, values))
    if not rows:
        raise ValueError(f"no opacity rows in {path}")
    n_bin = len(rows)
    n_lay = len(rows[0][1])
    opac = np.zeros((n_bin, n_lay), dtype=np.float64)
    for bin_idx, values in rows:
        opac[bin_idx, : len(values)] = values
    return {"n_bin": n_bin, "n_layer": n_lay, "opac_band_lay_cgs": opac}


def _power_law_exponent(pressure: np.ndarray, values: np.ndarray) -> float:
    p = np.asarray(pressure, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    mask = (p > 0) & (v > 0) & np.isfinite(p) & np.isfinite(v)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    return float(np.polyfit(np.log(p[mask]), np.log(v[mask]), 1)[0])


def compare_layer_opacity(
    *,
    opacities_path: Path,
    reference: dict,
    mode: str,
    constant_kappa_si: float | None = None,
    tagged_scale_cgs: float = 1.0,
    diffusivity: float = HELIOS_DEFAULT_DIFFUSIVITY,
) -> dict:
    dump = load_helios_opacities_dat(opacities_path)
    helios_k = dump["opac_band_lay_cgs"][0]
    n = int(reference["n_layers"])
    helios_k = helios_k[:n]
    p_lay_pa = np.asarray(reference["grid"]["p_lay_Pa"], dtype=np.float64)
    mass = np.asarray(reference["grid"]["layer_mass_kg_m2"], dtype=np.float64)
    if mode == "constant":
        spec = nested_analytic_opacity_spec(n)
        kappa_si = float(constant_kappa_si if constant_kappa_si is not None else spec.opacity().kappa0)
        expected_si = np.full(n, kappa_si, dtype=np.float64)
    elif mode == "analytic":
        opacity = nested_analytic_opacity_spec(n).opacity()
        expected_si = opacity.evaluate(
            np.asarray(reference["frozen"]["temperature_lay_k"], dtype=np.float64),
            p_lay_pa,
        )[0]
    elif mode == "tagged":
        expected_si = (tagged_scale_cgs * (p_lay_pa / 1.0e5)) / OPACITY_SI_TO_CGS
    else:
        raise ValueError(f"unknown mode: {mode}")
    expected_cgs = expected_si * OPACITY_SI_TO_CGS
    dtau_expected = diffusivity * expected_si * mass
    dtau_helios = diffusivity * (helios_k / OPACITY_SI_TO_CGS) * mass
    scale = np.maximum(np.abs(expected_cgs), 1e-30)
    kappa_rel = np.abs(helios_k - expected_cgs) / scale
    i = int(np.argmax(kappa_rel))
    k_exp = _power_law_exponent(p_lay_pa, helios_k)
    k_exp_expected = _power_law_exponent(p_lay_pa, expected_cgs)
    dtau_exp = _power_law_exponent(p_lay_pa, dtau_helios)
    dtau_exp_expected = _power_law_exponent(p_lay_pa, dtau_expected)
    kappa_max_rel = float(kappa_rel[i])
    failures = []
    if kappa_max_rel > 1.0e-3:
        failures.append("kappa_max_rel")
    if mode == "tagged":
        if abs(k_exp - 1.0) > 0.05:
            failures.append("kappa_exponent")
        if abs(dtau_exp - 2.0) > 0.05:
            failures.append("dtau_exponent")
    elif mode == "analytic":
        if abs(k_exp - 0.5) > 0.05:
            failures.append("kappa_exponent")
        if abs(dtau_exp - 1.5) > 0.05:
            failures.append("dtau_exponent")
    return {
        "mode": mode,
        "helios_opacities_file": str(opacities_path),
        "n_layer": n,
        "n_bin": dump["n_bin"],
        "diffusivity_factor": diffusivity,
        "kappa_max_rel": kappa_max_rel,
        "kappa_max_layer": i,
        "pressure_max_Pa": float(p_lay_pa[i]),
        "helios_kappa_cgs": helios_k.tolist(),
        "expected_kappa_cgs": expected_cgs.tolist(),
        "helios_dtau": dtau_helios.tolist(),
        "expected_dtau": dtau_expected.tolist(),
        "helios_kappa_vs_P_exponent": k_exp,
        "expected_kappa_vs_P_exponent": k_exp_expected,
        "helios_dtau_vs_P_exponent": dtau_exp,
        "expected_dtau_vs_P_exponent": dtau_exp_expected,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "note": (
            "Compares HELIOS interpolated layer opacity to the intended law. "
            "On a geometric grid ΔP∝P, so κ∝P^a implies Δτ∝P^(a+1). "
            "Tagged requires κ exponent≈1 and Δτ exponent≈2; analytic κ∝P^{1/2} "
            "requires ≈0.5 and ≈1.5. Layerwise 1e-3 is vs the true law, so the "
            "HDF5 log-P grid must be fine enough for HELIOS's linear-in-index interpolant."
        ),
    }


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opacities", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mode", choices=("analytic", "constant", "tagged"), required=True)
    parser.add_argument("--constant-kappa-si", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    result = compare_layer_opacity(
        opacities_path=args.opacities,
        reference=reference,
        mode=args.mode,
        constant_kappa_si=args.constant_kappa_si,
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.output) if args.output else None,
        "status": result["status"],
        "failures": result["failures"],
        "kappa_max_rel": result["kappa_max_rel"],
        "helios_kappa_vs_P_exponent": result["helios_kappa_vs_P_exponent"],
        "expected_kappa_vs_P_exponent": result["expected_kappa_vs_P_exponent"],
        "helios_dtau_vs_P_exponent": result["helios_dtau_vs_P_exponent"],
        "expected_dtau_vs_P_exponent": result["expected_dtau_vs_P_exponent"],
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
