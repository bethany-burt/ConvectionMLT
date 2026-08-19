"""Figure 08 — accepted-step scaling."""

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
    fit_log_log_slope,
    load_enriched_campaign,
    parameter_matrix_cases,
    save_figure,
)

COLORS = {0.0: "C0", 0.1: "C1", 0.10: "C1", 0.25: "C2"}


def main() -> None:
    payload = load_enriched_campaign()
    config = campaign_config(payload)
    cases = parameter_matrix_cases(payload)
    h2_regular = [
        c
        for c in cases
        if c["x_he"] == 0.0 and not c["irregular_grid"]
    ]
    h2_regular.sort(key=lambda c: c["n_layers"])
    n_fit = np.array([c["n_layers"] for c in h2_regular], dtype=float)
    steps_fit = np.array([c["steps"] for c in h2_regular], dtype=float)
    fit = fit_log_log_slope(n_fit, steps_fit)

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    series = {}
    for case in cases:
        series.setdefault((float(case["x_he"]), bool(case["irregular_grid"])), []).append(case)
    for key, items in series.items():
        items.sort(key=lambda c: c["n_layers"])
        n = np.array([c["n_layers"] for c in items], dtype=float)
        st = np.array([c["steps"] for c in items], dtype=float)
        x_he, irregular = key
        color = COLORS[x_he]
        mfc = color if not irregular else "none"
        ls = "none"
        label = rf"$x_{{\mathrm{{He}}}}={x_he:g}$ " + ("reg" if not irregular else "irr")
        ax.loglog(n, st, marker="o", mfc=mfc, color=color, ls=ls, label=label)

    n_line = np.array([25.0, 200.0])
    intercept = np.log(steps_fit[0]) - fit["slope"] * np.log(n_fit[0])
    ax.loglog(
        n_line,
        np.exp(intercept + fit["slope"] * np.log(n_line)),
        color="k",
        lw=1.4,
        label=rf"H2 regular fit $N^{{{fit['slope']:.2f}}}$",
    )
    ref = steps_fit[0] * (n_line / n_fit[0]) ** 2
    ax.loglog(n_line, ref, color="0.45", ls="--", lw=1.1, label=r"$N^2$ reference")
    ax.set_xlabel("N")
    ax.set_ylabel("accepted steps")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Figure 08 — Accepted-step scaling (not wall time)")
    save_figure(
        fig,
        "fig08_step_scaling",
        source_files=[ENRICHED_CAMPAIGN_PATH],
        tolerances=config,
        cases_included=[c["case_id"] for c in cases],
        extra={"fit_sequence": "pure_H2_regular", "slope": fit["slope"]},
    )
    plt.close(fig)
    print("wrote fig08_step_scaling")


if __name__ == "__main__":
    main()
