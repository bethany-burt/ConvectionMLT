"""Fail the gated HELIOS sequence unless a result JSON reports PASS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not args.result.exists():
        print(f"{args.label}: missing {args.result}", file=sys.stderr)
        raise SystemExit(2)
    payload = json.loads(args.result.read_text())
    status = payload.get("status")
    print(json.dumps({
        "label": args.label,
        "path": str(args.result),
        "status": status,
        "failures": payload.get("failures"),
        "kappa_max_rel": payload.get("kappa_max_rel"),
        "helios_kappa_vs_P_exponent": payload.get("helios_kappa_vs_P_exponent"),
        "helios_dtau_vs_P_exponent": payload.get("helios_dtau_vs_P_exponent"),
        "gates": payload.get("gates") or payload.get("failed_gates"),
    }, indent=2))
    if status != "PASS":
        raise SystemExit(f"{args.label} status={status}")


if __name__ == "__main__":
    main()
