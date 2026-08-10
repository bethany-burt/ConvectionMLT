"""Deterministic invariant audit table figure and CSV export."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    DATA_DIR,
    GENERATED_DIR,
    apply_style,
    ensure_dirs,
    read_json,
    require_finite,
    save_figure,
)


def _format_value(value: float, boolean_hint: bool) -> str:
    if boolean_hint and value in (0.0, 1.0):
        return f"{int(value)} (bool)"
    return f"{value:.6g}"


def _write_csv(rows: list[dict]) -> None:
    ensure_dirs()
    path = GENERATED_DIR / "validation_table.csv"
    fieldnames = ["name", "expected", "observed", "error", "tolerance", "pass", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"wrote {path}")


def main() -> None:
    path = DATA_DIR / "invariant_audit.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("invariant_audit.json: rows[] is empty")

    table_rows = []
    for row in rows:
        name = row["name"]
        expected = require_finite(f"{name}.expected", row["expected"])
        observed = require_finite(f"{name}.observed", row["observed"])
        error = require_finite(f"{name}.error", row["error"])
        tolerance = require_finite(f"{name}.tolerance", row["tolerance"])
        passed = bool(row.get("pass", False))
        notes = str(row.get("notes", ""))
        boolean_hint = "boolean" in notes.lower() or (
            expected in (0.0, 1.0)
            and observed in (0.0, 1.0)
            and tolerance == 0.0
            and name.startswith(
                (
                    "manufactured_positive",
                    "update_sign",
                    "alpha_zero",
                    "rejected_state",
                    "status_",
                )
            )
        )
        table_rows.append(
            {
                "name": name,
                "expected": expected,
                "observed": observed,
                "error": error,
                "tolerance": tolerance,
                "pass": passed,
                "notes": notes,
                "boolean_hint": boolean_hint,
            }
        )

    _write_csv(table_rows)

    apply_style()
    fig_height = max(6.0, 0.40 * len(table_rows) + 2.2)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.axis("off")

    col_labels = ["Invariant", "Expected", "Observed", "Error", "Tolerance", "Pass", "Notes"]
    cell_text = []
    cell_colours = []
    for row in table_rows:
        cell_text.append(
            [
                row["name"],
                _format_value(row["expected"], row["boolean_hint"]),
                _format_value(row["observed"], row["boolean_hint"]),
                f"{row['error']:.6g}",
                f"{row['tolerance']:.6g}",
                "PASS" if row["pass"] else "FAIL",
                row["notes"],
            ]
        )
        bg = "#d4edda" if row["pass"] else "#f8d7da"
        cell_colours.append([bg] * len(col_labels))

    # Widen first (name) and last (notes) columns.
    col_widths = [0.30, 0.09, 0.09, 0.09, 0.09, 0.06, 0.28]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        loc="center",
        cellLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
        if col_idx in (0, 6):
            cell.set_fontsize(7.0)

    n_pass = sum(1 for r in table_rows if r["pass"])
    fig.suptitle(
        f"Deterministic invariant audit ({n_pass}/{len(table_rows)} pass); "
        "Expected/Observed = 1 means boolean true where noted",
        fontsize=11,
        y=0.995,
    )
    out = save_figure(fig, "08_invariant_table.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
