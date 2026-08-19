"""Figure 01 — thermodynamic provider audit."""

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
    load_enriched_campaign,
    read_json,
    require_source,
    save_figure,
)


def main() -> None:
    path = require_source(DATA_DIR / "thermo_audit.json", description="thermo audit JSON")
    data = read_json(path)
    campaign = load_enriched_campaign()
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    mixes = data["mixtures"]
    colors = {"0.0": "C0", "0.1": "C1", "0.10": "C1", "0.25": "C2"}
    labels = {"0.0": r"$x_{\mathrm{He}}=0$", "0.1": r"$x_{\mathrm{He}}=0.1$", "0.10": r"$x_{\mathrm{He}}=0.1$", "0.25": r"$x_{\mathrm{He}}=0.25$"}

    ax = axes[0, 0]
    for key, curve in mixes.items():
        t = np.asarray(curve["temperature_k"])
        ax.plot(t, np.asarray(curve["cp_J_per_kg_K"]), color=colors.get(key, "k"), label=labels.get(key, key), lw=1.5)
    ax.axvline(1000.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("T [K]")
    ax.set_ylabel(r"$c_p$ [J kg$^{-1}$ K$^{-1}$]")
    ax.set_title(r"$c_p(T)$")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for key, curve in mixes.items():
        t = np.asarray(curve["temperature_k"])
        ax.plot(t, np.asarray(curve["nabla_ad"]), color=colors.get(key, "k"), lw=1.5)
    ax.axvline(1000.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("T [K]")
    ax.set_ylabel(r"$\nabla_{\mathrm{ad}} = R/c_p$")
    ax.set_title("Adiabatic gradient")

    ax = axes[1, 0]
    for key, curve in mixes.items():
        t = np.asarray(curve["temperature_k"])
        res = np.asarray(curve["dh_dT_relative_residual"])
        ax.semilogy(t, np.clip(res, 1e-18, None), color=colors.get(key, "k"), lw=1.2)
    ax.axvline(1000.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("T [K]")
    ax.set_ylabel(r"$|dh/dT - c_p|/c_p$")
    ax.set_title("Interval-aware derivative residual\n(exact 1000 K join excluded)")

    ax = axes[1, 1]
    for key, curve in mixes.items():
        t = np.asarray(curve["temperature_k"])
        ax.semilogy(t, np.clip(np.asarray(curve["T_from_h_relative_error"]), 1e-18, None), color=colors.get(key, "k"), lw=1.2, label=r"$T\to h\to T$")
        ax.semilogy(t, np.clip(np.asarray(curve["T_from_psi_relative_error"]), 1e-18, None), color=colors.get(key, "k"), lw=1.0, ls=":", label=r"$T\to\Psi\to T$")
    ax.axvline(1000.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("relative inversion error")
    ax.set_title("Inverse-function error\n(exact 1000 K join excluded)")
    handles, labs = ax.get_legend_handles_labels()
    by = dict(zip(labs, handles))
    ax.legend(by.values(), by.keys(), frameon=False)

    fig.suptitle("Figure 01 — Thermodynamic provider audit (NASA H2 / H2–He)")
    save_figure(
        fig,
        "fig01_thermo_audit",
        source_files=[path, "production_campaign_enriched.json"],
        tolerances=campaign["campaign_config"],
        cases_included=[],
        extra={
            "breakpoint_k": 1000.0,
            "t_ref_k": data["providers"]["nasa_h2"]["t_ref_k"],
            "exact_join_excluded": True,
        },
    )
    plt.close(fig)
    print("wrote fig01_thermo_audit")


if __name__ == "__main__":
    main()
