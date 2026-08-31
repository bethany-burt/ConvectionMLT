#!/usr/bin/env python3
"""User-facing nested-τ radiative–convective equilibrium runner.

Usage:
  cd convection_mlt && PYTHONPATH=src python examples/rce/run_rce.py --config examples/rce/cfg_demo.py
  cd convection_mlt && PYTHONPATH=src python examples/rce/run_rce.py --write-example-config my_case.py

Exit codes:
  0 CONVERGED
  1 NOT CONVERGED
  2 INVALID INPUT
  3 unexpected runtime error
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG / "src"))
sys.path.insert(0, str(_PKG / "stage4" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_cfg import ConfigError, copy_config_to_run_dir, load_and_validate, write_example_config
from outputs import OutputDirError, RunLog, prepare_output_dir, write_run_artifacts

from convection_mlt.production_rce import run_production_rce


EXIT_CONVERGED = 0
EXIT_NOT_CONVERGED = 1
EXIT_INVALID = 2
EXIT_ERROR = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Python case file (.py) with user inputs",
    )
    ap.add_argument(
        "--write-example-config",
        type=Path,
        default=None,
        help="write the demo case file template to PATH",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override output_dir/out_name from the case file",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite a non-empty output directory",
    )
    args = ap.parse_args(argv)

    if args.write_example_config is not None:
        try:
            write_example_config(args.write_example_config)
            print(f"wrote {args.write_example_config}")
            return EXIT_CONVERGED
        except Exception:
            traceback.print_exc()
            return EXIT_ERROR

    if args.config is None:
        print("INVALID INPUT: --config is required (or use --write-example-config)")
        return EXIT_INVALID

    cfg_path = args.config.resolve()
    try:
        cfg = load_and_validate(cfg_path)
    except ConfigError as exc:
        print(f"INVALID INPUT: {exc}")
        return EXIT_INVALID
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(cfg.resolved_output_dir)
    )
    force = bool(args.force or cfg.overwrite)
    try:
        out_dir = prepare_output_dir(out_dir, force=force)
    except OutputDirError as exc:
        print(f"INVALID INPUT: {exc}")
        return EXIT_INVALID
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR

    log = RunLog(out_dir / "run.log")
    started = _utc_now()
    wall0 = time.perf_counter()
    try:
        log.write(f"start_utc={started}")
        log.write(f"config={cfg_path}")
        log.write(f"config_checksum={cfg.config_checksum_sha256}")
        log.write(
            f"procedure={cfg.procedure} seed={cfg.seed} "
            f"N={cfg.n_layers} alpha={cfg.alpha} x_he={cfg.x_he}"
        )
        log.write(f"validation_envelope={cfg.envelope_status}")
        for w in cfg.envelope_warnings:
            log.write(f"ENVELOPE WARNING: {w}")
        log.write(
            "Flux sign: F_int upward at bottom; F_irr downward at top; "
            "profile fluxes positive upward."
        )

        copy_config_to_run_dir(cfg_path, out_dir)

        run = run_production_rce(
            n_layers=cfg.n_layers,
            alpha=cfg.alpha,
            f_int=cfg.f_int,
            f_irr=cfg.f_irr,
            gravity=cfg.gravity,
            p_bottom=cfg.p_bottom,
            p_top=cfg.p_top,
            x_he=cfg.x_he,
            seed=cfg.seed,
            procedure=cfg.procedure,
            controls=cfg.controls(),
            log=log.write,
        )
        ended = _utc_now()
        wall = time.perf_counter() - wall0
        verdict, record = write_run_artifacts(
            out_dir,
            cfg,
            run,
            started_utc=started,
            ended_utc=ended,
            wall_s=wall,
        )
        log.write(f"end_utc={ended} wall_s={wall:.3f}")
        log.write(f"verdict={verdict}")
        log.write(
            f"flatness={record.get('flux_flatness')} "
            f"tendency={record.get('tendency_norm')} "
            f"rcb={record.get('primary_rcb_log10p')}"
        )
        print(verdict)
        print(
            f"  convergence gates / topology / envelope: "
            f"see {out_dir / 'status.json'}"
        )
        print(f"  output_dir={out_dir}")
        return EXIT_CONVERGED if verdict == "CONVERGED" else EXIT_NOT_CONVERGED
    except ConfigError as exc:
        print(f"INVALID INPUT: {exc}")
        return EXIT_INVALID
    except Exception as exc:
        log.write(f"RUNTIME ERROR: {exc}")
        log.write(traceback.format_exc())
        print("RUNTIME ERROR")
        traceback.print_exc()
        return EXIT_ERROR
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
