"""Figure 09 — deterministic audit table."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PLOTS_ROOT = Path(__file__).resolve().parent
if str(PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTS_ROOT))

import matplotlib.pyplot as plt

from common import (
    DATA_DIR,
    GENERATED_DIR,
    apply_style,
    load_enriched_campaign,
    read_json,
    require_source,
    save_figure,
)


def _format_cell(value, *, boolean: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if boolean and value in (0.0, 1.0):
        return "True" if value == 1.0 else "False"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    path = require_source(DATA_DIR / "audit.json", description="audit JSON")
    data = read_json(path)
    campaign = load_enriched_campaign()
    rows = data["rows"]
    if not rows:
        raise ValueError("audit.json rows[] is empty")

    csv_path = GENERATED_DIR / "fig09_audit_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "observed", "tolerance", "comparison", "status", "units", "source_case", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    apply_style()
    fig_height = 0.30 * len(rows) + 1.0
    fig = plt.figure(figsize=(16.0, fig_height))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.93])
    ax.axis("off")
    table_text = []
    colours = []
    for row in rows:
        status = row["status"]
        colour = {"PASS": "#b7e1a1", "FAIL": "#f2b4b4", "N/A": "#d9d9d9"}.get(status, "white")
        boolean = row.get("units") == "boolean"
        table_text.append(
            [
                row["metric"],
                _format_cell(row.get("observed"), boolean=boolean),
                _format_cell(row.get("tolerance"), boolean=boolean),
                row["comparison"],
                row.get("units", ""),
                status,
                row["source_case"],
            ]
        )
        colours.append(["white", "white", "white", "white", "white", colour, "white"])
    table = ax.table(
        cellText=table_text,
        colLabels=["metric", "observed", "tolerance", "criterion", "units", "status", "source"],
        cellColours=colours,
        loc="center",
        cellLoc="left",
        colWidths=[0.30, 0.12, 0.10, 0.07, 0.10, 0.06, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.28)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.PAD = 0.02
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#efefef")
        if col_idx == 0:
            cell.set_text_props(ha="left")
        if col_idx == 5:
            cell.set_text_props(ha="center", weight="bold")
    ax.set_title("Figure 09 — Stage 2 deterministic audit", pad=8)
    save_figure(
        fig,
        "fig09_audit_table",
        source_files=[path],
        tolerances=campaign["campaign_config"],
        cases_included=list(range(1, 28)),
        extra={"n_rows": len(rows), "csv": str(csv_path)},
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"wrote fig09_audit_table and {csv_path}")


if __name__ == "__main__":
    main()
