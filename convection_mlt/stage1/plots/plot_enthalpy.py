"""Enthalpy conservation figure: signed drift histories and max drift vs resolution."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATA_DIR,
    GENERATED_DIR,
    apply_style,
    ensure_dirs,
    read_json,
    require_finite,
    save_figure,
)


def _history_arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    history = record.get("history", [])
    if not history:
        raise ValueError(f"enthalpy record N={record['n_layers']} α={record['alpha']}: empty history")
    times = []
    drifts = []
    for item in history:
        times.append(require_finite("simulated_time_s", item["simulated_time_s"]))
        drifts.append(require_finite("signed_enthalpy_drift", item["signed_enthalpy_drift"]))
    return np.asarray(times), np.asarray(drifts)


def _write_telescoping_csv(data: dict) -> None:
    rows = []
    for record in data.get("records", []):
        audit = record.get("conservation_audit")
        if not isinstance(audit, dict):
            raise ValueError(
                f"enthalpy record N={record['n_layers']} α={record['alpha']} "
                "is missing conservation_audit"
            )
        rows.append(
            {
                "n_layers": record["n_layers"],
                "alpha": record["alpha"],
                "telescoping_residual_w_m2": require_finite(
                    "telescoping_residual_w_m2",
                    audit["telescoping_residual_w_m2"],
                ),
                "telescoping_scale_w_m2": require_finite(
                    "telescoping_scale_w_m2",
                    audit["telescoping_scale_w_m2"],
                ),
                "bottom_boundary_flux_w_m2": require_finite(
                    "bottom_boundary_flux_w_m2",
                    audit["bottom_boundary_flux_w_m2"],
                ),
                "top_boundary_flux_w_m2": require_finite(
                    "top_boundary_flux_w_m2",
                    audit["top_boundary_flux_w_m2"],
                ),
            }
        )
    if not rows:
        raise ValueError("enthalpy conservation table has no rows")
    ensure_dirs()
    path = GENERATED_DIR / "enthalpy_telescoping.csv"
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    path = DATA_DIR / "enthalpy.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    records = data.get("records", [])
    if not records:
        raise ValueError("enthalpy.json: records[] is empty")

    apply_style()
    fig, (ax_time, ax_max) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.plasma

    max_points: list[tuple[int, float, float, float]] = []

    for idx, record in enumerate(records):
        n_layers = int(record["n_layers"])
        alpha = require_finite("alpha", record["alpha"])
        if alpha <= 0.0:
            raise ValueError(f"enthalpy record {idx}: alpha must be positive")
        tolerances = record["tolerances"]
        tol = require_finite("enthalpy_drift_tolerance", tolerances["enthalpy_drift"])
        max_drift = require_finite("max_abs_enthalpy_drift", record["max_abs_enthalpy_drift"])

        times, drifts = _history_arrays(record)
        color = cmap(0.15 + 0.75 * idx / max(len(records) - 1, 1))
        label = f"N={n_layers}, α={alpha:g}"

        ax_time.plot(times, drifts, color=color, lw=1.4, label=label)
        ax_time.axhline(tol, color=color, ls=":", lw=0.9, alpha=0.7)
        ax_time.axhline(-tol, color=color, ls=":", lw=0.9, alpha=0.7)

        max_points.append((n_layers, alpha, max_drift, tol))

    ax_time.set_xlabel("simulated time [s]")
    ax_time.set_ylabel("signed enthalpy drift")
    ax_time.set_yscale("symlog", linthresh=1.0e-20)
    ax_time.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax_time.legend(fontsize=7, loc="best")
    ax_time.set_title("Signed enthalpy drift vs time (± tolerance)")

    ns = np.array([p[0] for p in max_points], dtype=float)
    alphas = np.array([p[1] for p in max_points], dtype=float)
    max_drifts = np.array([p[2] for p in max_points], dtype=float)
    tols = np.array([p[3] for p in max_points], dtype=float)

    scatter = ax_max.scatter(
        ns,
        max_drifts,
        c=alphas,
        cmap="plasma",
        s=70,
        edgecolors="k",
        linewidths=0.4,
    )
    for n, drift, tol in zip(ns, max_drifts, tols):
        ax_max.axhline(tol, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax_max.set_xscale("log", base=2)
    ax_max.set_yscale("log")
    ax_max.set_xlabel("N (layers)")
    ax_max.set_ylabel("max |enthalpy drift| over run")
    ax_max.set_title("Maximum absolute drift vs resolution")
    cbar = fig.colorbar(scatter, ax=ax_max, fraction=0.046, pad=0.04)
    cbar.set_label("α")

    fig.suptitle("Enthalpy conservation", fontsize=12)
    out = save_figure(fig, "03_enthalpy_conservation.png")
    plt.close(fig)
    print(f"wrote {out}")

    _write_telescoping_csv(data)


if __name__ == "__main__":
    main()
