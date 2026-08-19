"""Figure 07 — inverse-square gravity sweep and scope diagnostics."""

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
    ENRICHED_CAMPAIGN_PATH,
    apply_style,
    campaign_config,
    cases_with_role,
    load_enriched_campaign,
    read_json,
    require_source,
    save_figure,
)


def main() -> None:
    payload = load_enriched_campaign()
    config = campaign_config(payload)
    campaign_g = cases_with_role(payload, "gravity_stress")
    limit_path = require_source(DATA_DIR / "gravity_limit.json", description="gravity-limit JSON")
    limit = read_json(limit_path)

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.2))

    rp = np.array([c["planet_radius_m"] for c in limit["cases"]])
    zrp = np.array([c["max_z_over_rp"] for c in limit["cases"]])
    rms = np.array([c["temperature_rms_vs_isentrope"] for c in limit["cases"]])
    dmax = np.array([c["max_superadiabaticity"] for c in limit["cases"]])
    drift = np.array([c["apparent_enthalpy_drift"] for c in limit["cases"]])
    ez = np.array([c["E_z"] for c in limit["cases"]])
    dgg = np.array([c["max_abs_dg_over_g0"] for c in limit["cases"]])
    stress = [c["extreme_stress"] for c in limit["cases"]]

    ax = axes[0, 0]
    ax.set_title("Formal equilibrium gates")
    ax.loglog(rp, rms, marker="o", color="C0", label="coupled sweep RMS")
    ax.loglog(rp, dmax, marker="s", color="C1", label=r"coupled sweep $\max\Delta\nabla_s^+$")
    ax.axhline(config["isentrope_rms_tolerance"], color="C0", ls=":", lw=0.8)
    ax.axhline(config["epsilon_gradient"], color="C1", ls=":", lw=0.8)
    ax.set_ylim(1.0e-10, 3.0e-6)
    ax.set_xlabel(r"$R_p$ [m]")

    ax = axes[0, 1]
    ax.set_title("Scope: column extent")
    ax.loglog(rp, zrp, marker="o", color="C3", label="coupled sweep")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    for r, z, flag in zip(rp, zrp, stress):
        if flag:
            ax.annotate("extreme stress", (r, z), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel(r"$R_p$ [m]")
    ax.set_ylabel(r"$\max(z/R_p)$")

    ax = axes[0, 2]
    ax.set_title("Diagnostic only: apparent drift")
    ax.loglog(zrp, drift, marker="o", color="0.3")
    ax.set_xlabel(r"$\max(z/R_p)$")
    ax.set_ylabel("apparent enthalpy drift")

    ax = axes[1, 0]
    ax.set_title(r"$\max|(g-g_0)/g_0|$")
    ax.loglog(rp, dgg, marker="o")
    ax.set_xlabel(r"$R_p$ [m]")

    ax = axes[1, 1]
    ax.set_title(r"$E_z=\max|z_{\mathrm{var}}-z_{\mathrm{const}}|/\max|z_{\mathrm{const}}|$")
    ax.loglog(rp, ez, marker="o")
    ax.set_xlabel(r"$R_p$ [m]")

    ax = axes[1, 2]
    profile = limit.get("thin_atmosphere_profile")
    if profile is None:
        ax.set_title(r"Thin-atmosphere signed $(g-g_0)/g_0\simeq-2z/R_p$")
        ax.text(0.5, 0.5, "no thin-atmosphere member", ha="center", va="center", transform=ax.transAxes)
    else:
        z = np.asarray(profile["z_edges_m"])
        signed = np.asarray(profile["dg_over_g0_signed"])
        approx = np.asarray(profile["approx_signed_minus_2z_over_rp"])
        ax.plot(z, signed, color="C0", lw=1.5, label="coupled final g")
        ax.plot(z, approx, color="C1", ls="--", lw=1.4, label=r"$-2z/R_p$")
        ax.set_xlabel("z [m]")
        ax.legend(frameon=False, fontsize=7)
        rp_exp = int(round(np.log10(float(profile["planet_radius_m"]))))
        ax.set_title(rf"Signed approx at $R_p=10^{{{rp_exp}}}$ m")

    campaign_rp = np.array([c["planet_radius"] for c in campaign_g], dtype=float)
    campaign_rms = np.array([c["temperature_rms_vs_isentrope"] for c in campaign_g], dtype=float)
    campaign_zrp = np.array([c["max_z_over_rp"] for c in campaign_g], dtype=float)
    axes[0, 0].scatter(
        campaign_rp,
        campaign_rms,
        facecolors="none",
        edgecolors="k",
        zorder=5,
        s=70,
        linewidths=1.3,
        label="production campaign RMS",
    )
    axes[0, 1].scatter(
        campaign_rp,
        campaign_zrp,
        facecolors="none",
        edgecolors="k",
        zorder=5,
        s=70,
        linewidths=1.3,
        label="production campaign",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)

    fig.suptitle("Figure 07 — Inverse-square gravity: gates vs scope diagnostics")
    save_figure(
        fig,
        "fig07_inverse_square",
        source_files=[ENRICHED_CAMPAIGN_PATH, limit_path],
        tolerances=config,
        cases_included=[c["case_id"] for c in campaign_g],
        extra={
            "radii_m": limit["planet_radii_m"],
            "height_norm": "E_z column-scale",
            "rms_definition": "mass-weighted relative T vs rebuilt numerical isentrope",
        },
    )
    plt.close(fig)
    print("wrote fig07_inverse_square")


if __name__ == "__main__":
    main()
