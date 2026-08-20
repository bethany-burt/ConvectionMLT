"""Equal-time N=48 vs N=96 trajectories and post-1e-3 spatial comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    ImplicitConvectionConfig,
    LowerNetInternalFlux,
    RCEConfig,
    RCERoute,
    SolverConfig,
    TopIrradiation,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)
from convection_mlt.energy import column_enthalpy_per_area


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "plots" / "generated"
DATA_DIR = ROOT / "plots" / "data"
STAGE4_EXIT_FLUX_GATE = 1.0e-3


def _spec(n_layers: int) -> AnalyticOpacityRCESpec:
    if n_layers <= 48:
        n_phot = 16
    elif n_layers <= 96:
        n_phot = 24
    else:
        n_phot = 32
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=n_phot)


def _solver() -> SolverConfig:
    return SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)


def _implicit_cfg(
    *,
    max_steps: int,
    gate: float,
    t_final: float | None = None,
    dt_accuracy: float = 2500.0,
) -> RCEConfig:
    return RCEConfig(
        max_steps=max_steps,
        n_consec=5,
        stall_window=10**9,
        flux_flatness_tolerance=gate,
        tendency_tolerance=gate,
        temp_change_tolerance=gate,
        dt_accuracy=dt_accuracy,
        t_final=t_final,
        implicit_convection=ImplicitConvectionConfig(
            residual_tolerance=1e-10,
            step_tolerance=1e-10,
        ),
    )


def _time_series(res) -> dict:
    accepted = [d for d in res.diagnostics if d.accepted]
    times = np.cumsum([d.dt for d in accepted]) if accepted else np.asarray([])
    return {
        "time": times.tolist(),
        "flux_flatness": [d.flux_flatness for d in accepted],
        "tendency_norm": [d.tendency_norm for d in accepted],
        "temp_change": [d.temp_change for d in accepted],
        "primary_rcb_log10p": [d.primary_rcb_log10p for d in accepted],
        "max_f_conv": [
            float(np.nanmax(np.abs(res.final_flux_conv)))  # placeholder; filled below per-step if needed
        ],
    }


def _run_equal_time(n_layers: int, t_final: float, max_steps: int = 2000) -> dict:
    spec = _spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        _solver(),
        thermo,
        opacity,
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=_implicit_cfg(
            max_steps=max_steps,
            gate=1e-12,
            t_final=t_final,
        ),
    )
    wall = time.perf_counter() - wall0
    accepted = [d for d in res.diagnostics if d.accepted]
    times = np.cumsum([d.dt for d in accepted]) if accepted else np.asarray([0.0])
    # Reconstruct max |F_conv| along the trajectory from final only is weak;
    # store final and per-step flatness/tendency/RCB which the driver records.
    return {
        "n_layers": n_layers,
        "status": res.status.value,
        "reason": res.reason,
        "steps_accepted": res.steps_accepted,
        "rejections": res.rejections,
        "simulated_time": res.simulated_time,
        "wall_time_s": wall,
        "simulated_per_wall": (
            float(res.simulated_time / wall) if wall > 0.0 else float("nan")
        ),
        "flux_flatness": res.convergence.flux_flatness,
        "tendency_norm": res.convergence.tendency_norm,
        "primary_rcb_log10p": res.primary_rcb_log10p,
        "convective_regions": res.convective_regions,
        "detached_convective_regions": res.detached_convective_regions,
        "pressure_centres": grid.pressure_centres.tolist(),
        "temperature": res.final_state.temperature.tolist(),
        "flux_total": res.final_flux_total.tolist(),
        "flux_rad": res.final_flux_rad.tolist(),
        "flux_conv": res.final_flux_conv.tolist(),
        "max_f_conv_final": float(np.max(np.abs(res.final_flux_conv))),
        "column_enthalpy": float(
            column_enthalpy_per_area(res.final_state.mass_path, res.final_state.enthalpy)
        ),
        "time": times.tolist(),
        "flux_flatness_series": [float(d.flux_flatness) for d in accepted],
        "tendency_series": [float(d.tendency_norm) for d in accepted],
        "temp_change_series": [float(d.temp_change) for d in accepted],
        "rcb_series": [
            None if d.primary_rcb_log10p is None else float(d.primary_rcb_log10p)
            for d in accepted
        ],
        "f_int": spec.f_int,
    }


def _run_to_gate(n_layers: int, gate: float = STAGE4_EXIT_FLUX_GATE, max_steps: int = 800) -> dict:
    spec = _spec(n_layers)
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t0 = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    wall0 = time.perf_counter()
    res = solve_adaptive_rce(
        grid,
        t0,
        spec.physics(),
        _solver(),
        thermo,
        opacity,
        grid.pressure_centres,
        TopIrradiation(spec.f_irr),
        LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.SPLIT_RAD_THEN_IMPLICIT_CONV,
        config=_implicit_cfg(max_steps=max_steps, gate=gate),
    )
    wall = time.perf_counter() - wall0
    return {
        "n_layers": n_layers,
        "status": res.status.value,
        "reason": res.reason,
        "steps_accepted": res.steps_accepted,
        "rejections": res.rejections,
        "simulated_time": res.simulated_time,
        "wall_time_s": wall,
        "flux_flatness": res.convergence.flux_flatness,
        "tendency_norm": res.convergence.tendency_norm,
        "primary_rcb_log10p": res.primary_rcb_log10p,
        "convective_regions": res.convective_regions,
        "detached_convective_regions": res.detached_convective_regions,
        "pressure_centres": grid.pressure_centres.tolist(),
        "pressure_edges": grid.pressure_edges.tolist(),
        "temperature": res.final_state.temperature.tolist(),
        "flux_total": res.final_flux_total.tolist(),
        "flux_rad": res.final_flux_rad.tolist(),
        "flux_conv": res.final_flux_conv.tolist(),
        "column_enthalpy": float(
            column_enthalpy_per_area(res.final_state.mass_path, res.final_state.enthalpy)
        ),
        "f_int": spec.f_int,
        "cz_extent_log10p": _cz_extent_log10p(grid.pressure_centres, res.convective_regions),
    }


def _cz_extent_log10p(pressure_centres, regions) -> float | None:
    if not regions:
        return None
    i0, i1 = regions[0]
    p = np.asarray(pressure_centres, dtype=np.float64)
    return float(np.log10(p[i0]) - np.log10(p[min(i1, len(p) - 1)]))


def interpolate_temperature(log_p_src: np.ndarray, t_src: np.ndarray, log_p_dst: np.ndarray) -> np.ndarray:
    order = np.argsort(log_p_src)
    return np.interp(log_p_dst, log_p_src[order], np.asarray(t_src, dtype=np.float64)[order])


def spatial_comparison(cases: dict[int, dict], reference_n: int = 48) -> dict:
    ref = cases[reference_n]
    log_p_ref = np.log10(np.asarray(ref["pressure_centres"], dtype=np.float64))
    t_ref = np.asarray(ref["temperature"], dtype=np.float64)
    out: dict = {"reference_n": reference_n, "pairs": {}}
    for n, case in cases.items():
        if n == reference_n:
            continue
        log_p = np.log10(np.asarray(case["pressure_centres"], dtype=np.float64))
        t_on_ref = interpolate_temperature(log_p, case["temperature"], log_p_ref)
        rel_t = float(np.max(np.abs(t_on_ref - t_ref) / np.maximum(t_ref, 1.0)))
        dlog_rcb = None
        if ref["primary_rcb_log10p"] is not None and case["primary_rcb_log10p"] is not None:
            dlog_rcb = abs(float(case["primary_rcb_log10p"]) - float(ref["primary_rcb_log10p"]))
        # Grid-cell measure: ΔlogP / mean ΔlogP of reference.
        dlog_cell = float(np.mean(np.abs(np.diff(log_p_ref))))
        cells = None if dlog_rcb is None else dlog_rcb / max(dlog_cell, 1e-30)
        f_ref = np.asarray(ref["flux_total"], dtype=np.float64)
        # Compare fluxes on edges via logP edges when available.
        out["pairs"][f"{n}_vs_{reference_n}"] = {
            "max_rel_T_on_ref_P": rel_t,
            "delta_log10_P_rcb": dlog_rcb,
            "delta_rcb_in_ref_cells": cells,
            "delta_column_enthalpy_rel": abs(
                float(case["column_enthalpy"]) - float(ref["column_enthalpy"])
            ) / max(abs(float(ref["column_enthalpy"])), 1.0),
            "boundary_flux_bottom": float(case["flux_total"][0]),
            "boundary_flux_top": float(case["flux_total"][-1]),
            "ref_boundary_flux_bottom": float(f_ref[0]),
            "ref_boundary_flux_top": float(f_ref[-1]),
            "cz_extent_log10p": case.get("cz_extent_log10p"),
            "ref_cz_extent_log10p": ref.get("cz_extent_log10p"),
            "detached": case.get("detached_convective_regions"),
        }
    return out


def _plot_equal_time(cases: dict[int, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    for n, color in ((48, "C0"), (96, "C1")):
        c = cases[n]
        t = np.asarray(c["time"], dtype=np.float64)
        axes[0, 0].semilogy(t, np.maximum(c["flux_flatness_series"], 1e-16), color=color, label=f"N={n}")
        axes[0, 1].semilogy(t, np.maximum(c["tendency_series"], 1e-16), color=color, label=f"N={n}")
        axes[0, 2].plot(t, c["temp_change_series"], color=color, label=f"N={n}")
        rcb = [np.nan if v is None else v for v in c["rcb_series"]]
        axes[1, 0].plot(t, rcb, color=color, label=f"N={n}")
        axes[1, 1].axhline(c["max_f_conv_final"], color=color, label=f"N={n} final max|F_conv|")
        axes[1, 2].bar(
            [str(n)],
            [c["simulated_per_wall"]],
            color=color,
            alpha=0.8,
        )
    axes[0, 0].set_title("flux flatness vs t")
    axes[0, 1].set_title("tendency vs t")
    axes[0, 2].set_title(r"$\|\Delta T\|/T$ vs t")
    axes[1, 0].set_title(r"$\log_{10} P_{\mathrm{RCB}}$ vs t")
    axes[1, 1].set_title(r"final $\max|F_{\mathrm{conv}}|$")
    axes[1, 2].set_title("simulated time / wall-s")
    for ax in axes[0, :]:
        ax.set_xlabel("t")
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("t")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].set_ylabel("s / wall-s")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_spatial(cases: dict[int, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5))
    for n, color, ls in ((48, "C0", "-"), (96, "C1", "--"), (192, "C2", ":")):
        if n not in cases:
            continue
        c = cases[n]
        p = np.asarray(c["pressure_centres"])
        axes[0].plot(c["temperature"], p, color=color, ls=ls, label=f"N={n}")
        e = np.asarray(c["pressure_edges"])
        axes[1].plot(c["flux_total"], e, color=color, ls=ls, label=rf"$F$ N={n}")
        axes[2].plot(c["flux_conv"], e, color=color, ls=ls, label=rf"$F_c$ N={n}")
        if c["primary_rcb_log10p"] is not None:
            axes[0].axhline(10 ** c["primary_rcb_log10p"], color=color, ls=":", lw=0.9)
    for ax in axes:
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    axes[0].set_xlabel("T (K)")
    axes[0].set_ylabel("P (Pa)")
    axes[0].set_title("T(P) at 1e-3 gate")
    axes[1].set_xlabel("F_total")
    axes[1].set_title("total flux")
    axes[2].set_xlabel("F_conv")
    axes[2].set_title("convective flux")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(include_n192: bool = True, t_final: float = 2.0e5) -> dict:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    equal = {
        48: _run_equal_time(48, t_final=t_final),
        96: _run_equal_time(96, t_final=t_final, max_steps=4000),
    }
    _plot_equal_time(equal, PLOT_DIR / "equal_time_n48_n96.png")

    steady: dict[int, dict] = {
        48: _run_to_gate(48, max_steps=400),
        96: _run_to_gate(96, max_steps=1200),
    }
    if include_n192:
        try:
            steady[192] = _run_to_gate(192, max_steps=4000)
        except Exception as exc:  # noqa: BLE001 — affordability gate
            steady["n192_error"] = str(exc)  # type: ignore[assignment]

    comparison = spatial_comparison({k: v for k, v in steady.items() if isinstance(k, int)})
    _plot_spatial({k: v for k, v in steady.items() if isinstance(k, int)}, PLOT_DIR / "spatial_1e-3.png")

    payload = {
        "t_final_equal_time": t_final,
        "equal_time": equal,
        "steady_1e-3": {str(k): v for k, v in steady.items()},
        "spatial_comparison": comparison,
    }
    out = DATA_DIR / "equal_time_spatial.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(json.dumps({
        "equal_time_flatness": {str(k): equal[k]["flux_flatness"] for k in equal},
        "steady_status": {str(k): (v["status"] if isinstance(v, dict) else v) for k, v in steady.items()},
        "spatial": comparison,
        "plot_equal": str(PLOT_DIR / "equal_time_n48_n96.png"),
        "plot_spatial": str(PLOT_DIR / "spatial_1e-3.png"),
        "data": str(out),
    }, indent=2, allow_nan=True))
    return payload


if __name__ == "__main__":
    main()
