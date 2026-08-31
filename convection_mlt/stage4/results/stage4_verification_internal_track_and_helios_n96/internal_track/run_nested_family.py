"""Gate-converged nested N=48 and N=96 from the same 384-layer master as N=192.

Independent-grid N=48/96 remain the cheap regression pair. Richardson analysis
uses this nested family only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from convection_mlt import (
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    RCERoute,
    TopIrradiation,
    nested_analytic_opacity_spec,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.energy import column_enthalpy_per_area

from rce_record import (
    PHYSICAL_GATE,
    dumps,
    production_rce_config,
    production_solver_config,
    serialize_rce_result,
)


OUT = ROOT / "results" / "nested_rce_family.json"
N384_OUT = ROOT / "results" / "n384_implicit_rce.json"
MAX_STEPS = {48: 5000, 96: 20000, 384: 20000}
TIGHTEN_STEPS = 8
TIGHTEN_DT = 500.0


def _run_one(
    n_layers: int,
    *,
    t0=None,
    dt_accuracy: float = 2500.0,
    dt_hold: float | None = None,
    previous_rcb: float | None = None,
    simulated_time: float = 0.0,
    max_steps: int | None = None,
) -> dict:
    spec = nested_analytic_opacity_spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    solver = production_solver_config()
    seeded = t0 is not None
    if t0 is None:
        t0 = radiative_convective_initial_temperature(
            grid, spec.opacity(), thermo, spec.f_int, spec.f_irr
        )
    cfg = production_rce_config(
        max_steps=max_steps if max_steps is not None else MAX_STEPS[n_layers],
        dt_accuracy=dt_accuracy,
        dt_hold_init=dt_hold,
        previous_rcb_init=previous_rcb,
        simulated_time_init=simulated_time,
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        solver,
        thermo,
        spec.opacity(),
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=cfg,
    )
    wall = time.perf_counter() - wall0
    payload = serialize_rce_result(
        res,
        spec,
        pressure_centres=grid.pressure_centres,
        pressure_edges=grid.pressure_edges,
        solver=solver,
        rce_config=cfg,
        extra={
            "wall_time_s": wall,
            "max_steps_budget": max_steps if max_steps is not None else MAX_STEPS[n_layers],
            "physical_gate": PHYSICAL_GATE,
            "initial_from": "interpolated_n192" if n_layers == 384 and seeded else "radiative_convective_seed",
        },
    )
    print(
        f"N={n_layers}",
        res.status.value,
        "flat", res.convergence.flux_flatness,
        "tend", res.convergence.tendency_norm,
        "acc", res.steps_accepted,
        "rej", res.rejections,
        "rcb", res.primary_rcb_log10p,
        "wall", round(wall, 1),
        flush=True,
    )
    return payload


def interpolate_temperature(log_p_src, t_src, log_p_dst):
    import numpy as np

    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def _physically_gated(rec: dict) -> bool:
    return (
        rec.get("status") == "converged"
        and float(rec.get("flux_flatness") or 1.0) <= PHYSICAL_GATE
        and float(rec.get("tendency_norm") or 1.0) <= PHYSICAL_GATE
        and not (rec.get("detached_convective_regions") or [])
    )


def _topology(rec: dict) -> dict:
    regions = rec.get("convective_regions") or []
    detached = rec.get("detached_convective_regions") or []
    bottom = [r for r in regions if r and r[0] == 0]
    return {
        "convective_regions": regions,
        "detached_convective_regions": detached,
        "n_bottom_connected": len(bottom),
        "n_detached": len(detached),
        "single_bottom_cz": len(bottom) == 1 and len(detached) == 0,
        "physically_gated": _physically_gated(rec),
        "profile_checksum_sha256": rec.get("profile_checksum_sha256") or rec.get("checksum_sha256"),
        "record_checksum_sha256": rec.get("record_checksum_sha256"),
        "status": rec.get("status"),
        "flux_flatness": rec.get("flux_flatness"),
        "tendency_norm": rec.get("tendency_norm"),
        "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
    }


def _compare(coarse: dict, fine: dict) -> dict:
    import numpy as np

    p_c = np.asarray(coarse["pressure_centres"], dtype=np.float64)
    p_f = np.asarray(fine["pressure_centres"], dtype=np.float64)
    t_c = np.asarray(coarse["temperature"], dtype=np.float64)
    t_f_on_c = interpolate_temperature(np.log(p_f), fine["temperature"], np.log(p_c))
    scale = np.maximum(np.abs(t_c), 1.0)
    rel = np.abs(t_f_on_c - t_c) / scale
    imax = int(np.argmax(rel))
    rcb_c = coarse.get("primary_rcb_log10p")
    rcb_f = fine.get("primary_rcb_log10p")
    h_c = float(coarse.get("column_enthalpy") or column_enthalpy_per_area(
        coarse["mass_path"], coarse["enthalpy"]
    ))
    h_f = float(fine.get("column_enthalpy") or column_enthalpy_per_area(
        fine["mass_path"], fine["enthalpy"]
    ))
    topo_c = _topology(coarse)
    topo_f = _topology(fine)
    return {
        "coarse_n": coarse["n_layers"],
        "fine_n": fine["n_layers"],
        "both_converged": coarse.get("status") == "converged" and fine.get("status") == "converged",
        "both_physically_gated": bool(topo_c["physically_gated"] and topo_f["physically_gated"]),
        "max_rel_T_on_coarse_P": float(rel[imax]),
        "max_rel_T_index": imax,
        "max_rel_T_pressure": float(p_c[imax]),
        "max_rel_T_log10p": float(np.log10(p_c[imax])),
        "delta_log10_P_rcb": (
            None if rcb_c is None or rcb_f is None else abs(float(rcb_f) - float(rcb_c))
        ),
        "column_enthalpy_rel": abs(h_f - h_c) / max(abs(h_c), abs(h_f), 1.0),
        "coarse_rcb": rcb_c,
        "fine_rcb": rcb_f,
        "topology_agree": bool(
            topo_c["single_bottom_cz"] and topo_f["single_bottom_cz"]
        ),
        "coarse": topo_c,
        "fine": topo_f,
    }


def _error_on_grid(ref: dict, other: dict, p_grid) -> float:
    import numpy as np

    t_ref = interpolate_temperature(
        np.log(np.asarray(ref["pressure_centres"], dtype=np.float64)),
        ref["temperature"],
        np.log(p_grid),
    )
    t_other = interpolate_temperature(
        np.log(np.asarray(other["pressure_centres"], dtype=np.float64)),
        other["temperature"],
        np.log(p_grid),
    )
    scale = np.maximum(np.abs(t_ref), 1.0)
    return float(np.max(np.abs(t_other - t_ref) / scale))


def richardson_order(e_coarse_vs_mid: float, e_mid_vs_fine: float) -> float | None:
    import numpy as np

    if e_coarse_vs_mid <= 0.0 or e_mid_vs_fine <= 0.0:
        return None
    return float(np.log2(e_coarse_vs_mid / e_mid_vs_fine))


def _plot_nested_family(cases: dict, n192: dict | None, n384: dict | None):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    series = []
    if "96" in cases:
        series.append((96, cases["96"], "C0", "-"))
    if n192 is not None:
        series.append((192, n192, "C1", "--"))
    if n384 is not None:
        series.append((384, n384, "C2", "-"))
    if len(series) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5))
    for n, rec, color, ls in series:
        p = np.asarray(rec["pressure_centres"], dtype=np.float64)
        e = np.asarray(rec["pressure_edges"], dtype=np.float64)
        axes[0].plot(rec["temperature"], p, color=color, ls=ls, label=f"N={n}")
        axes[1].plot(rec["flux_total"], e, color=color, ls=ls, label=rf"$F$ N={n}")
        axes[2].plot(rec["flux_conv"], e, color=color, ls=ls, label=rf"$F_c$ N={n}")
        rcb = rec.get("primary_rcb_log10p")
        if rcb is not None:
            axes[0].axhline(10 ** float(rcb), color=color, ls=":", lw=0.9)
    for ax in axes:
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    axes[0].set_xlabel("T (K)")
    axes[0].set_ylabel("P (Pa)")
    axes[0].set_title("Nested T(P) at 1e-3 gate")
    axes[1].set_xlabel("F_total (W m$^{-2}$)")
    axes[1].set_title("total flux")
    axes[2].set_xlabel("F_conv (W m$^{-2}$)")
    axes[2].set_title("convective flux")
    fig.tight_layout()
    path = ROOT / "plots" / "generated" / "nested_spatial_1e-3.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(layers=(48, 96), force: bool = False) -> dict:
    import numpy as np

    cases = {}
    if OUT.exists():
        cases = json.loads(OUT.read_text())
        cases = dict(cases.get("members") or cases)
    for n in layers:
        key = str(n)
        existing = cases.get(key)
        if n == 384:
            print(
                "Refusing to re-solve N=384; using gated live record "
                f"{N384_OUT.name}",
                flush=True,
            )
            continue
        if existing and existing.get("status") == "converged" and not force:
            print(f"N={n} already converged; skipping", flush=True)
            continue
        cases[key] = _run_one(n)
        OUT.write_text(dumps({"members": cases}))

    n192_path = ROOT / "results" / "n192_implicit_rce.json"
    n192 = json.loads(n192_path.read_text()) if n192_path.exists() else None
    comparisons = {}
    if "48" in cases and "96" in cases:
        comparisons["96_vs_48"] = _compare(cases["48"], cases["96"])
    if n192 is not None and "96" in cases:
        comparisons["192_vs_96"] = _compare(cases["96"], n192)
    if n192 is not None and "48" in cases:
        comparisons["192_vs_48"] = _compare(cases["48"], n192)
    n384 = None
    if N384_OUT.exists():
        n384 = json.loads(N384_OUT.read_text())
        cases["384"] = n384
    if n384 is not None and n192 is not None:
        comparisons["384_vs_192"] = _compare(n192, n384)
    if n384 is not None and "96" in cases:
        comparisons["384_vs_96"] = _compare(cases["96"], n384)

    order_48_96_192 = None
    pair_c = comparisons.get("96_vs_48") or {}
    pair_f = comparisons.get("192_vs_96") or {}
    if pair_c.get("both_converged") and pair_f.get("both_converged"):
        order_48_96_192 = richardson_order(
            float(pair_c["max_rel_T_on_coarse_P"]),
            float(pair_f["max_rel_T_on_coarse_P"]),
        )
    order_96_192_384_pairwise = None
    pair_m = comparisons.get("384_vs_192") or {}
    if pair_f.get("both_converged") and pair_m.get("both_converged"):
        order_96_192_384_pairwise = richardson_order(
            float(pair_f["max_rel_T_on_coarse_P"]),
            float(pair_m["max_rel_T_on_coarse_P"]),
        )

    order_96_192_384 = None
    richardson_norm = None
    if n192 is not None and n384 is not None and "96" in cases:
        import numpy as np

        p_ref = np.asarray(cases["96"]["pressure_centres"], dtype=np.float64)
        e_192_96 = _error_on_grid(cases["96"], n192, p_ref)
        e_384_192 = _error_on_grid(n192, n384, p_ref)
        richardson_norm = {
            "grid": "nested_n96_pressure_centres",
            "norm": "max_rel_T",
            "e_192_vs_96": e_192_96,
            "e_384_vs_192": e_384_192,
        }
        if (
            pair_f.get("both_converged")
            and pair_m.get("both_converged")
            and e_192_96 > 0.0
            and e_384_192 > 0.0
        ):
            order_96_192_384 = richardson_order(e_192_96, e_384_192)

    member_checksums = {
        key: {
            "n_layers": rec.get("n_layers"),
            "status": rec.get("status"),
            "physically_gated": _physically_gated(rec),
            "flux_flatness": rec.get("flux_flatness"),
            "tendency_norm": rec.get("tendency_norm"),
            "primary_rcb_log10p": rec.get("primary_rcb_log10p"),
            "convective_regions": rec.get("convective_regions"),
            "detached_convective_regions": rec.get("detached_convective_regions") or [],
            "profile_checksum_sha256": rec.get("profile_checksum_sha256") or rec.get("checksum_sha256"),
            "record_checksum_sha256": rec.get("record_checksum_sha256"),
        }
        for key, rec in cases.items()
        if isinstance(rec, dict)
    }
    if n192 is not None:
        member_checksums["192"] = {
            "n_layers": n192.get("n_layers"),
            "status": n192.get("status"),
            "physically_gated": _physically_gated(n192),
            "flux_flatness": n192.get("flux_flatness"),
            "tendency_norm": n192.get("tendency_norm"),
            "primary_rcb_log10p": n192.get("primary_rcb_log10p"),
            "convective_regions": n192.get("convective_regions"),
            "detached_convective_regions": n192.get("detached_convective_regions") or [],
            "profile_checksum_sha256": n192.get("profile_checksum_sha256") or n192.get("checksum_sha256"),
            "record_checksum_sha256": n192.get("record_checksum_sha256"),
            "source": "n192_implicit_rce.json",
        }

    payload = {
        "members": cases,
        "comparisons": comparisons,
        "member_checksums": member_checksums,
        "richardson_norm": richardson_norm,
        "richardson_order_from_max_rel_T": order_48_96_192,
        "richardson_order_48_96_192": order_48_96_192,
        "richardson_order_96_192_384_pairwise_coarse": order_96_192_384_pairwise,
        "richardson_order_96_192_384": order_96_192_384,
        "n384_source": str(N384_OUT) if n384 is not None else None,
        "n384_profile_checksum_sha256": None if n384 is None else (
            n384.get("profile_checksum_sha256") or n384.get("checksum_sha256")
        ),
        "note": (
            "Nested τ-family from master N=384, n_phot=64. N=384 is the "
            "physically gated live record (five-check polish), not the "
            "accelerator snapshot. Richardson 96/192/384 uses max relative T "
            "on the nested N=96 pressure grid at every resolution. "
            "The spatial exit pair is 192→384. 48→96 and 96→192 are diagnostics. "
            "Independent-grid N=48/96 are not used here."
        ),
    }
    OUT.write_text(dumps(payload))
    plot_path = _plot_nested_family(cases, n192, n384)
    print(json.dumps({
        "out": str(OUT),
        "plot": str(plot_path) if plot_path else None,
        "comparisons": {
            key: {
                k: v for k, v in pair.items()
                if k not in {"coarse", "fine"}
            }
            for key, pair in comparisons.items()
        },
        "richardson_norm": richardson_norm,
        "richardson_order_48_96_192": order_48_96_192,
        "richardson_order_96_192_384": order_96_192_384,
        "richardson_order_96_192_384_pairwise_coarse": order_96_192_384_pairwise,
        "member_checksums": member_checksums,
    }, indent=2, default=str), flush=True)
    return payload


def run_n384_from_n192() -> dict:
    """Prolong the gated N=192 column onto the 384-edge master and relax.

    Disabled: the live N=384 record is the physically gated five-check polish.
    Re-solving here would overwrite that source of truth.
    """
    raise RuntimeError(
        "Refusing to overwrite n384_implicit_rce.json. The gated live record "
        "is the five-check polish (profile 5e0bd…). Use continue_n384.py only "
        "if you intend to replace that file."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nested analytic-opacity RCE family")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[48, 96],
        help=(
            "Family members to (re)solve. N=384 is always loaded from the "
            "gated live record; this flag will not overwrite it."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run members even if a converged record is already stored.",
    )
    args = parser.parse_args()
    main(layers=tuple(args.layers), force=args.force)
