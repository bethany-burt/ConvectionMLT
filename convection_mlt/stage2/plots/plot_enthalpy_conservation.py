"""Figure 05 — constant-gravity enthalpy conservation."""

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
    load_enriched_campaign,
    parameter_matrix_cases,
    save_figure,
)

COLORS = {0.0: "C0", 0.1: "C1", 0.10: "C1", 0.25: "C2"}


def main() -> None:
    payload = load_enriched_campaign()
    config = campaign_config(payload)
    cases = parameter_matrix_cases(payload)
    history_path = DATA_DIR / "representative_column.json"
    has_history = history_path.exists()

    apply_style()
    ncols = 3 if has_history else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.4))

    series = {}
    for case in cases:
        series.setdefault((float(case["x_he"]), bool(case["irregular_grid"])), []).append(case)
    for key, items in series.items():
        items.sort(key=lambda c: c["n_layers"])
        n = np.array([c["n_layers"] for c in items], dtype=float)
        x_he, irregular = key
        color = COLORS[x_he]
        mfc = color if not irregular else "none"
        ls = "-" if not irregular else "--"
        label = rf"$x_{{\mathrm{{He}}}}={x_he:g}$ " + ("reg" if not irregular else "irr")
        drift = np.array([c["enthalpy_drift"] for c in items])
        axes[0].semilogy(n, drift, color=color, ls=ls, marker="o", mfc=mfc, label=label)
        axes[1].semilogy(n, drift / config["enthalpy_drift_tolerance"], color=color, ls=ls, marker="o", mfc=mfc)
    axes[0].axhline(config["enthalpy_drift_tolerance"], color="k", ls=":", lw=1.0)
    axes[1].axhline(1.0, color="k", ls=":", lw=1.0)
    axes[0].set_title("Terminal relative drift")
    axes[1].set_title(r"Drift / $10^{-12}$ gate")
    axes[0].set_ylabel(r"$|H-H_0|/H_{\mathrm{scale}}$")
    axes[1].set_ylabel(r"$S_H$")
    for ax in axes[:2]:
        ax.set_xlabel("N")
        ax.set_xticks([25, 50, 100, 200])
    axes[0].legend(frameon=False, fontsize=7, loc="center left")
    axes[0].text(
        0.98,
        0.06,
        r"$H_{\mathrm{scale}}=\max(|H_0|,\,1\;\mathrm{J\,m^{-2}})$",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )

    if has_history:
        from common import read_json

        hist = read_json(history_path)["history"]
        steps = np.array([row["accepted_step"] for row in hist], dtype=float)
        drift = np.array([abs(row["signed_enthalpy_drift"]) for row in hist])
        axes[2].semilogy(np.maximum(steps, 1.0), np.clip(drift, 1e-18, None), color="C0")
        axes[2].axhline(config["enthalpy_drift_tolerance"], color="k", ls=":", lw=1.0)
        axes[2].set_xlabel("accepted step")
        axes[2].set_ylabel(r"$|H-H_0|/H_{\mathrm{scale}}$")
        axes[2].set_title("Representative enthalpy-drift history")
        axes[2].annotate(
            r"gate $=10^{-12}$",
            xy=(0.97, config["enthalpy_drift_tolerance"]),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=7,
        )

    fig.suptitle("Figure 05 — Constant-g enthalpy conservation (inverse-square excluded)")
    sources = [ENRICHED_CAMPAIGN_PATH]
    if has_history:
        sources.append(history_path)
    save_figure(
        fig,
        "fig05_enthalpy_conservation",
        source_files=sources,
        tolerances=config,
        cases_included=[c["case_id"] for c in cases],
        extra={"history_panel": has_history},
    )
    plt.close(fig)
    print("wrote fig05_enthalpy_conservation")


if __name__ == "__main__":
    main()
