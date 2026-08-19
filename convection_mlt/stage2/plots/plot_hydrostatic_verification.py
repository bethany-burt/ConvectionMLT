"""Figure 06 — hydrostatic verification."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PLOTS_ROOT = Path(__file__).resolve().parent
if str(PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTS_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATA_DIR,
    apply_style,
    fit_log_log_slope,
    load_enriched_campaign,
    pressure_axis,
    read_json,
    require_source,
    save_figure,
)


def main() -> None:
    path = require_source(DATA_DIR / "hydro_references.json", description="hydro reference JSON")
    data = read_json(path)
    campaign = load_enriched_campaign()
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))

    iso = data["isothermal_constant_g"]
    p = np.asarray(iso["pressure_edges_pa"])
    axes[0, 0].plot(iso["z_analytic_m"], p, color="C1", ls="--", lw=1.6, label="analytic")
    axes[0, 0].plot(iso["z_model_m"], p, color="C0", lw=1.2, label="model")
    pressure_axis(axes[0, 0])
    axes[0, 0].set_xlabel("z [m]")
    axes[0, 0].set_title("Constant-g isothermal")
    axes[0, 0].legend(frameon=False)

    isq = data["isothermal_inverse_square"]
    p2 = np.asarray(isq["pressure_edges_pa"])
    axes[0, 1].plot(isq["z_analytic_m"], p2, color="C1", ls="--", lw=1.6, label="analytic")
    axes[0, 1].plot(isq["z_model_m"], p2, color="C0", lw=1.2, label="model")
    pressure_axis(axes[0, 1])
    axes[0, 1].set_xlabel("z [m]")
    axes[0, 1].set_title("Inverse-square isothermal")

    ns = data["nonisothermal"]
    nvals = np.array(sorted(int(k) for k in ns["mild_relative_error_vs_ode"]))
    mild = np.array([ns["mild_relative_error_vs_ode"][str(n)] for n in nvals], dtype=float)
    strong = np.array([ns["strong_relative_error_vs_ode"][str(n)] for n in nvals], dtype=float)
    axes[1, 0].loglog(nvals, mild, marker="o", label="mild T(P)")
    axes[1, 0].loglog(nvals, strong, marker="s", label="strong T(P)")
    mild_fit = fit_log_log_slope(nvals.astype(float), mild)
    strong_fit = fit_log_log_slope(nvals.astype(float), strong)
    axes[1, 0].set_xlabel("N")
    axes[1, 0].set_ylabel(r"$\max|z-z_{\mathrm{ODE}}|/\max|z_{\mathrm{ODE}}|$")
    axes[1, 0].set_title("Refinement vs independent ODE")
    axes[1, 0].legend(frameon=False, loc="upper right")
    axes[1, 0].text(
        0.05,
        0.08,
        rf"mild slope $={mild_fit['slope']:.2f}$"
        "\n"
        rf"strong slope $={strong_fit['slope']:.2f}$"
        "\n"
        r"(expected $\propto N^{-2}$)",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        va="bottom",
    )

    prof = ns["profiles"]["100"]
    p3 = np.asarray(prof["pressure_edges_pa"])
    axes[1, 1].plot(
        np.asarray(prof["z_model_strong_m"]) - np.asarray(prof["z_ref_strong_m"]),
        p3,
        color="C3",
        lw=1.4,
    )
    pressure_axis(axes[1, 1])
    axes[1, 1].set_xlabel(r"$z_{\mathrm{model}}-z_{\mathrm{ODE}}$ [m]")
    axes[1, 1].set_title(
        "Strong nonisothermal residual (N=100)\n"
        "stress test; not held to the mild absolute gate"
    )

    fig.suptitle("Figure 06 — Hydrostatic verification (independent adaptive ODE reference)")
    save_figure(
        fig,
        "fig06_hydrostatic_verification",
        source_files=[path],
        tolerances=campaign["campaign_config"],
        cases_included=[],
        extra={
            "reference_method": data["ode_reference"]["reference_method"],
            "relative_tolerance": data["ode_reference"]["relative_tolerance"],
            "absolute_tolerance": data["ode_reference"]["absolute_tolerance"],
            "maximum_step": data["ode_reference"]["maximum_step"],
            "reference_refinement_check": data["reference_refinement_check"]["max_relative_change"],
            "round_trip": data["round_trip_relative_pressure_error"],
            "mild_refinement_slope": mild_fit["slope"],
            "strong_refinement_slope": strong_fit["slope"],
        },
    )
    plt.close(fig)
    print("wrote fig06_hydrostatic_verification")


if __name__ == "__main__":
    main()
