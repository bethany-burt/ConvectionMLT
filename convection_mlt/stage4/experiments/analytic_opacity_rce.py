"""Local analytic-opacity coupled RCE figure: T(P), fluxes, residual, resolution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convection_mlt import (
    AnalyticOpacityRCESpec,
    ConstantGravity,
    ConstantH2Thermo,
    LowerNetInternalFlux,
    RCEConfig,
    RCERoute,
    SolverConfig,
    TopIrradiation,
    grey_layer_optical_thickness,
    radiative_convective_initial_temperature,
    solve_adaptive_rce,
)


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "plots" / "generated"
DATA_DIR = ROOT / "plots" / "data"


def _spec(n_layers: int) -> AnalyticOpacityRCESpec:
    n_phot = 16 if n_layers <= 48 else 24
    return AnalyticOpacityRCESpec(n_layers=n_layers, n_photosphere=n_phot)


def _run(spec: AnalyticOpacityRCESpec, max_steps: int):
    grid = spec.grid()
    thermo = ConstantH2Thermo()
    opacity = spec.opacity()
    t = radiative_convective_initial_temperature(
        grid, opacity, thermo, spec.f_int, spec.f_irr
    )
    solver = SolverConfig(epsilon_temperature=2.0e-3, c_diff=0.2, dt_min=1.0e-14)
    cfg = RCEConfig(
        max_steps=max_steps,
        n_consec=10**9,
        stall_window=10**9,
        flux_flatness_tolerance=1e-12,
        tendency_tolerance=1e-12,
        temp_change_tolerance=1e-12,
    )
    res = solve_adaptive_rce(
        grid, t, spec.physics(), solver, thermo, opacity, grid.pressure_centres,
        TopIrradiation(spec.f_irr), LowerNetInternalFlux(spec.f_int),
        gravity=ConstantGravity(spec.gravity),
        route=RCERoute.UNSPLIT,
        config=cfg,
    )
    dtau = grey_layer_optical_thickness(grid, opacity, res.final_state.temperature)
    accepted = [d for d in res.diagnostics if d.accepted]
    return {
        "n_layers": spec.n_layers,
        "status": res.status.value,
        "reason": res.reason,
        "steps_accepted": res.steps_accepted,
        "simulated_time": res.simulated_time,
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
        "superadiabaticity": res.final_closure.superadiabaticity.tolist(),
        "active": res.final_closure.active.tolist(),
        "dtau_top": float(dtau[-1]),
        "dtau_bottom": float(dtau[0]),
        "f_int": spec.f_int,
        "residual_steps": [d.flux_flatness for d in accepted],
        "tendency_steps": [d.tendency_norm for d in accepted],
        "time_steps": np.cumsum([d.dt for d in accepted]).tolist(),
        "energy_residual_rel": accepted[-1].energy_residual_rel if accepted else None,
        "energy_committed_residual": accepted[-1].energy_committed_residual if accepted else None,
        "energy_ulp_floor": accepted[-1].energy_ulp_floor if accepted else None,
    }


def _plot(cases: dict[int, dict]) -> None:
    c48 = cases[48]
    c96 = cases[96]
    f_int = c48["f_int"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))

    def _p(ax, y, x, **kw):
        ax.plot(x, y, **kw)
        ax.set_yscale("log")
        ax.invert_yaxis()

    p48 = np.asarray(c48["pressure_centres"])
    e48 = np.asarray(c48["pressure_edges"])
    _p(axes[0, 0], p48, c48["temperature"], color="C0", lw=1.6, label="N=48")
    _p(axes[0, 0], np.asarray(c96["pressure_centres"]), c96["temperature"],
       color="C1", lw=1.2, ls="--", label="N=96")
    if c48["primary_rcb_log10p"] is not None:
        axes[0, 0].axhline(10 ** c48["primary_rcb_log10p"], color="C0", ls=":", lw=1.0)
    if c96["primary_rcb_log10p"] is not None:
        axes[0, 0].axhline(10 ** c96["primary_rcb_log10p"], color="C1", ls=":", lw=1.0)
    axes[0, 0].set_xlabel("T (K)")
    axes[0, 0].set_ylabel("P (Pa)")
    axes[0, 0].set_title("T(P) and RCB")
    axes[0, 0].legend(fontsize=8)

    _p(axes[0, 1], e48, c48["flux_rad"], color="C2", lw=1.3, label=r"$F_{\mathrm{rad}}$")
    _p(axes[0, 1], e48, c48["flux_conv"], color="C3", lw=1.3, label=r"$F_{\mathrm{conv}}$")
    _p(axes[0, 1], e48, c48["flux_total"], color="k", lw=1.4, label=r"$F_{\mathrm{total}}$")
    axes[0, 1].axvline(f_int, color="0.5", ls="--", lw=0.8)
    axes[0, 1].set_xlabel(r"$F$ (W m$^{-2}$)")
    axes[0, 1].set_title("Fluxes, N=48")
    axes[0, 1].legend(fontsize=8)

    resid = np.asarray(c48["flux_total"]) - f_int
    _p(axes[0, 2], e48, resid, color="C4", lw=1.4)
    axes[0, 2].axvline(0.0, color="0.5", ls="--", lw=0.8)
    axes[0, 2].set_xlabel(r"$F_{\mathrm{total}}-F_{\mathrm{int}}$")
    axes[0, 2].set_title("Flux residual")

    delta = np.asarray(c48["superadiabaticity"])
    active = np.asarray(c48["active"], dtype=bool)
    _p(axes[1, 0], e48, delta, color="C5", lw=1.3, label=r"$\nabla-\nabla_{\mathrm{ad}}$")
    if np.any(active):
        axes[1, 0].scatter(
            delta[active], e48[active], s=12, color="C3", zorder=3, label="active"
        )
    axes[1, 0].axvline(0.0, color="0.5", ls="--", lw=0.8)
    axes[1, 0].set_xlabel("superadiabaticity")
    axes[1, 0].set_ylabel("P (Pa)")
    axes[1, 0].set_title("Activity mask")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].semilogy(c48["residual_steps"], color="C0", lw=1.3, label="N=48 flatness")
    axes[1, 1].semilogy(c48["tendency_steps"], color="C0", lw=1.0, ls="--", label="N=48 tendency")
    axes[1, 1].semilogy(c96["residual_steps"], color="C1", lw=1.1, label="N=96 flatness")
    axes[1, 1].axhline(0.1, color="0.4", ls=":", lw=1.0, label="declared gate 0.1")
    axes[1, 1].set_xlabel("accepted step")
    axes[1, 1].set_ylabel("residual")
    axes[1, 1].set_title("Residual vs step")
    axes[1, 1].legend(fontsize=7)

    _p(axes[1, 2], p48, c48["temperature"], color="C0", lw=1.6, label="N=48")
    _p(axes[1, 2], np.asarray(c96["pressure_centres"]), c96["temperature"],
       color="C1", lw=1.2, ls="--", label="N=96")
    axes[1, 2].set_xlabel("T (K)")
    axes[1, 2].set_title(r"$T(P)$ resolution")
    axes[1, 2].legend(fontsize=8)
    txt = []
    for n, case in ((48, c48), (96, c96)):
        rcb = case["primary_rcb_log10p"]
        p_rcb = None if rcb is None else 10 ** rcb
        txt.append(f"N={n}  P_RCB={p_rcb:.3e} Pa" if p_rcb else f"N={n}  P_RCB=None")
    axes[1, 2].text(
        0.05, 0.12, "\n".join(txt), transform=axes[1, 2].transAxes, fontsize=8,
        va="bottom",
    )

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "analytic_opacity_rce.png", dpi=140)
    plt.close(fig)


def main() -> None:
    cases = {
        48: _run(_spec(48), max_steps=8000),
        96: _run(_spec(96), max_steps=8000),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        n: {
            k: cases[n][k]
            for k in (
                "n_layers", "status", "reason", "steps_accepted", "simulated_time",
                "flux_flatness", "tendency_norm", "primary_rcb_log10p",
                "convective_regions", "detached_convective_regions",
                "dtau_top", "dtau_bottom", "energy_residual_rel",
                "energy_committed_residual", "energy_ulp_floor",
            )
        }
        for n in cases
    }
    (DATA_DIR / "analytic_opacity_rce.json").write_text(json.dumps(summary, indent=2))
    _plot(cases)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
