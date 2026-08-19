"""Figure 04 — resolution / composition / grid robustness."""

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
    ENRICHED_CAMPAIGN_PATH,
    apply_style,
    campaign_config,
    load_enriched_campaign,
    parameter_matrix_cases,
    save_figure,
)

COLORS = {0.0: "C0", 0.1: "C1", 0.10: "C1", 0.25: "C2"}


def main() -> None:
    payload = load_enriched_campaign()
    config = campaign_config(payload)
    cases = parameter_matrix_cases(payload)
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=True)

    series = {}
    for case in cases:
        key = (float(case["x_he"]), bool(case["irregular_grid"]))
        series.setdefault(key, []).append(case)
    for key, items in series.items():
        items.sort(key=lambda c: c["n_layers"])
        n = np.array([c["n_layers"] for c in items], dtype=float)
        x_he, irregular = key
        color = COLORS[x_he]
        marker = "o" if not irregular else "o"
        mfc = color if not irregular else "none"
        ls = "-" if not irregular else "--"
        label = rf"$x_{{\mathrm{{He}}}}={x_he:g}$ " + ("regular" if not irregular else "irregular")
        axes[0].semilogy(
            n,
            [c["temperature_rms_vs_isentrope"] / config["isentrope_rms_tolerance"] for c in items],
            color=color,
            ls=ls,
            marker=marker,
            mfc=mfc,
            label=label,
        )
        axes[1].plot(
            n,
            [c["max_superadiabaticity"] / config["epsilon_gradient"] for c in items],
            color=color,
            ls=ls,
            marker=marker,
            mfc=mfc,
        )
        axes[2].semilogy(
            n,
            [c["enthalpy_drift"] / config["enthalpy_drift_tolerance"] for c in items],
            color=color,
            ls=ls,
            marker=marker,
            mfc=mfc,
        )

    for ax in axes:
        ax.axhline(1.0, color="k", ls=":", lw=1.0)
        ax.set_xlabel("N")
        ax.set_xticks([25, 50, 100, 200])
    axes[0].set_ylabel(r"$S_{\mathrm{RMS}}=\epsilon_{\mathrm{RMS}}/10^{-6}$")
    axes[1].set_ylabel(r"$S_{\Delta}=\max\Delta\nabla_s^+/10^{-8}$")
    axes[2].set_ylabel(r"$S_H=\mathrm{drift}/10^{-12}$")
    axes[1].set_ylim(0.994, 1.002)
    axes[0].legend(frameon=False, fontsize=7, ncol=1)
    fig.suptitle("Figure 04 — Tolerance-limited grid independence (not spatial convergence)")
    save_figure(
        fig,
        "fig04_resolution_robustness",
        source_files=[ENRICHED_CAMPAIGN_PATH],
        tolerances=config,
        cases_included=[c["case_id"] for c in cases],
        extra={"claim": "tolerance-limited grid independence"},
    )
    plt.close(fig)
    print("wrote fig04_resolution_robustness")


if __name__ == "__main__":
    main()
