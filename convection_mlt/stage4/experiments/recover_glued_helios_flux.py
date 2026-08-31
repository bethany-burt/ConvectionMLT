"""One-off recovery of the glued F_net/F_dir constant-κ flux file. Not an audit artifact."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT.parent / "src"))

from convection_mlt.adapters.helios import format_integrated_flux_row, parse_flux_token

# .17e mantissa glued to a trailing F_dir=0 written with %g.
_GLUED_FNET_FDIR0 = re.compile(r"^([+-]?\d\.\d+e[+-]\d{2,3})0$")


def recover_fields(fields: list[str]) -> list[str]:
    if len(fields) >= 5:
        match = _GLUED_FNET_FDIR0.fullmatch(fields[4])
        if match:
            fields = fields[:4] + [match.group(1), "0.0"] + fields[5:]
    return fields


def recover_file(src: Path, dest: Path) -> dict:
    lines_out = [
        "This file contains the integrated total and net fluxes at each interface resp. layer.",
        "Fluxes given in [erg s^-1 cm^-2].",
        "Recovered from glued %.17e F_net+F_dir=0. Not a final audit artifact.",
        "interface press.[10^-6bar] F_down F_up F_net F_dir delta_F_net F_net_conv F_add_heat F_intern",
    ]
    n_rows = 0
    n_split = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            interface = int(float(fields[0]))
        except ValueError:
            continue
        recovered = recover_fields(fields)
        if recovered != fields:
            n_split += 1
        # After split: interface, P, Fd, Fu, Fnet, Fdir, delta, conv, heat, [intern]
        while len(recovered) < 9:
            recovered.append("not_avail.")
        intern = parse_flux_token(recovered[9]) if len(recovered) > 9 else float("nan")
        add_heat = parse_flux_token(recovered[8])
        delta = parse_flux_token(recovered[6])
        lines_out.append(
            format_integrated_flux_row(
                interface,
                parse_flux_token(recovered[1]),
                parse_flux_token(recovered[2]),
                parse_flux_token(recovered[3]),
                parse_flux_token(recovered[4]),
                f_dir=parse_flux_token(recovered[5]),
                delta_f_net=None if not np_isfinite(delta) else delta,
                f_net_conv=parse_flux_token(recovered[7]),
                f_add_heat=None if not np_isfinite(add_heat) else add_heat,
                f_intern=None if not np_isfinite(intern) else intern,
            )
        )
        n_rows += 1
    dest.write_text("\n".join(lines_out) + "\n")
    return {"src": str(src), "dest": str(dest), "n_rows": n_rows, "n_glued_split": n_split}


def np_isfinite(value: float) -> bool:
    return value == value and value != float("inf") and value != float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(recover_file(args.src, args.dest), indent=2))


if __name__ == "__main__":
    main()
