"""Figure 02 — representative numerical isentrope."""

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
    pressure_axis,
    read_json,
    require_source,
    save_figure,
)


def main() -> None:
    path = require_source(
        DATA_DIR / "representative_column.json", description="representative column JSON"
    )
    data = read_json(path)
    campaign = load_enriched_campaign()
    p = np.asarray(data["pressure_centres_pa"])
    t0 = np.asarray(data["temperature_initial_k"])
    t1 = np.asarray(data["temperature_final_k"])
    tref = np.asarray(data["temperature_isentrope_k"])
    s1 = np.asarray(data["entropy_final"])
    hist = data["history"]
    steps = np.array([row["accepted_step"] for row in hist], dtype=float)
    span = np.array([row["entropy_span"] if row["entropy_span"] is not None else np.nan for row in hist])
    dmax = np.array([row["max_superadiabaticity"] for row in hist])
    flux = np.array([np.abs(row["convective_flux_max"]) for row in hist])

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))

    ax = axes[0, 0]
    ax.plot(t0, p, color="0.55", lw=1.4, label="initial")
    ax.plot(tref, p, color="C1", lw=1.6, ls="--", label="isentrope")
    ax.plot(t1, p, color="C0", lw=1.8, label="final")
    pressure_axis(ax)
    ax.set_xlabel("T [K]")
    ax.set_title("T(P)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(t1 - tref, p, color="C0", lw=1.5)
    pressure_axis(ax)
    ax.set_xlabel(r"$T_{\mathrm{final}} - T_{\mathrm{isentrope}}$ [K]")
    rel_rms = data.get("metrics", {}).get("temperature_rms")
    if rel_rms is None:
        ax.set_title("Final residual")
    else:
        ax.set_title(rf"Final residual (relative T RMS $={rel_rms:.2e}$)")

    ax = axes[1, 0]
    s_bar = float(np.mean(s1))
    ax.plot(s1 - s_bar, p, color="C0", lw=1.6)
    pressure_axis(ax)
    ax.set_xlabel(r"$s-\bar{s}$ [J kg$^{-1}$ K$^{-1}$]")
    ax.set_title("Final entropy residual")

    ax = axes[1, 1]
    span0 = next((v for v in span if np.isfinite(v) and v > 0.0), np.nan)
    flux0 = next((v for v in flux if np.isfinite(v) and v > 0.0), np.nan)
    eps = campaign["campaign_config"]["epsilon_gradient"]
    ax.loglog(
        np.maximum(steps, 1.0),
        np.clip(span / span0, 1e-18, None),
        label=r"$\Delta s/\Delta s_0$",
    )
    ax.loglog(
        np.maximum(steps, 1.0),
        np.clip(dmax / eps, 1e-18, None),
        label=r"$\max\Delta\nabla_s^+/10^{-8}$",
    )
    ax.loglog(
        np.maximum(steps, 1.0),
        np.clip(flux / flux0, 1e-18, None),
        label=r"$\max|F|/F_0$",
    )
    ax.set_xlabel("accepted step")
    ax.set_ylabel("normalized diagnostic")
    ax.set_title("Relaxation history")
    ax.legend(frameon=False, loc="best")

    fig.suptitle("Figure 02 — Representative N=100 NASA H2 isentrope")
    save_figure(
        fig,
        "fig02_representative_isentrope",
        source_files=[path],
        tolerances=campaign["campaign_config"],
        cases_included=[],
        extra={
            "n_layers": data["n_layers"],
            "steps": data["steps"],
            "flux_semantics": data["flux_semantics"],
        },
    )
    plt.close(fig)
    print("wrote fig02_representative_isentrope")


if __name__ == "__main__":
    main()
