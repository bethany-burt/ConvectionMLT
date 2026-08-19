"""Figure 03 — production robustness matrix."""

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
from matplotlib.colors import BoundaryNorm, ListedColormap

from common import (
    ENRICHED_CAMPAIGN_PATH,
    apply_style,
    campaign_config,
    cases_with_role,
    format_score_ratio,
    format_significant,
    load_enriched_campaign,
    parameter_matrix_cases,
    save_figure,
)


def _cell_color(score: float | None) -> str:
    if score is None:
        return "#d9d9d9"
    return "#b7e1a1" if score < 1.0 else "#f2b4b4"


def main() -> None:
    payload = load_enriched_campaign()
    config = campaign_config(payload)
    matrix = parameter_matrix_cases(payload)
    if len(matrix) != 24:
        raise ValueError(f"expected 24 parameter_matrix cases, found {len(matrix)}")

    rows = []
    for case in matrix:
        s_h = case["enthalpy_drift"] / config["enthalpy_drift_tolerance"]
        s_rms = case["temperature_rms_vs_isentrope"] / config["isentrope_rms_tolerance"]
        s_d = case["max_superadiabaticity"] / config["epsilon_gradient"]
        grid = "irr" if case["irregular_grid"] else "reg"
        rows.append(
            {
                "label": f"N={case['n_layers']}  xHe={case['x_he']:g}  {grid}",
                "case_id": case["case_id"],
                "status": case["status"],
                "S_H": s_h,
                "H": case["enthalpy_drift"],
                "S_rms": s_rms,
                "rms": case["temperature_rms_vs_isentrope"],
                "S_d": s_d,
                "dmax": case["max_superadiabaticity"],
                "steps": case["steps"],
            }
        )

    apply_style()
    fig = plt.figure(figsize=(11.5, 9.2))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.22, 0.32, 0.74, 0.62])
    col_labels = ["status", r"$S_H$", r"$S_{\mathrm{RMS}}$", r"$S_{\Delta}$", "steps"]
    table_text = []
    cell_colours = []
    for row in rows:
        table_text.append(
            [
                row["status"],
                f"{format_score_ratio(row['S_H'])}\n{format_significant(row['H'])}",
                f"{format_score_ratio(row['S_rms'])}\n{format_significant(row['rms'])}",
                f"{row['S_d']:.6f}\n{row['dmax']:.6e}",
                f"{row['steps']}",
            ]
        )
        cell_colours.append(
            [
                _cell_color(0.0 if row["status"] == "converged" else 2.0),
                _cell_color(row["S_H"]),
                _cell_color(row["S_rms"]),
                _cell_color(row["S_d"]),
                "white",
            ]
        )
    table = ax.table(
        cellText=table_text,
        rowLabels=[row["label"] for row in rows],
        colLabels=col_labels,
        cellColours=cell_colours,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.35)
    ax.axis("off")
    ax.set_title(
        "Figure 03 — Constant-g parameter matrix (24 cases)\n"
        r"S = observed / tolerance; green $S<1$. RMS tolerance $=10^{-6}$ (raw value shown)."
    )

    canonical = next(
        c
        for c in matrix
        if c["n_layers"] == 50
        and c["x_he"] == 0.0
        and not c["irregular_grid"]
    )
    case27 = cases_with_role(payload, "pressure_range_check")[0]
    ax_in = fig.add_axes([0.08, 0.03, 0.84, 0.24])
    ax_in.axis("off")
    inset = [
        ["field", "canonical N=50 H2 regular", "case 27 pressure-range check"],
        ["case_id", str(canonical["case_id"]), str(case27["case_id"])],
        ["status", canonical["status"], case27["status"]],
        ["P_bottom [Pa]", f"{canonical['pressure_bottom']:.3e}", f"{case27['pressure_bottom']:.3e}"],
        ["P_top [Pa]", f"{canonical['pressure_top']:.3e}", f"{case27['pressure_top']:.3e}"],
        ["steps", str(canonical["steps"]), str(case27["steps"])],
        ["enthalpy drift", f"{canonical['enthalpy_drift']:.3e}", f"{case27['enthalpy_drift']:.3e}"],
        ["T RMS", f"{canonical['temperature_rms_vs_isentrope']:.3e}", f"{case27['temperature_rms_vs_isentrope']:.3e}"],
        [
            "max superadiabaticity",
            f"{canonical['max_superadiabaticity']:.6e}",
            f"{case27['max_superadiabaticity']:.6e}",
        ],
        ["campaign_role", canonical["campaign_role"], case27["campaign_role"]],
    ]
    table2 = ax_in.table(cellText=inset[1:], colLabels=inset[0], loc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(7.5)
    table2.scale(1.0, 1.2)
    ax_in.set_title("Case 27 is a different pressure range (domain coverage), not a repeatability test")

    save_figure(
        fig,
        "fig03_robustness_matrix",
        source_files=[ENRICHED_CAMPAIGN_PATH],
        tolerances=config,
        cases_included=[c["case_id"] for c in matrix] + [case27["case_id"]],
        extra={"n_parameter_matrix": 24, "pressure_range_case_id": case27["case_id"]},
    )
    plt.close(fig)
    print("wrote fig03_robustness_matrix")


if __name__ == "__main__":
    main()
