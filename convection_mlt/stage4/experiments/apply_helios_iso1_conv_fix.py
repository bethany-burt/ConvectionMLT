"""Apply minimal HELIOS ISO1 convective-adjustment fix (versioned).

Stock b0800f9 skips conv_check when isothermal_layers=yes, leaving
conv_unstable=None and crashing convection_loop. This patch enables
conv_check with constant-κ interface κ so iso=yes + convective adjustment
can run as a labelled counterfactual. Also writes conv flags in TP when
convection is on (stock blanks them for iso==1). Does not change radiation
kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "helios"
DEFAULT_PATCH = FIXTURES / "helios_iso1_conv_check_b0800f9.patch"
PINNED_HELIOS_COMMIT = "b0800f9ea4366263241c13bb926e8ca68f266cc5"
MARKER = "Minimal ISO1 fix: stock HELIOS skips conv_check when iso==1"
WRITE_MARKER = "Minimal ISO1 fix: write conv flags when convection==1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_provenance(patch_path: Path = DEFAULT_PATCH) -> dict:
    return {
        "helios_commit": PINNED_HELIOS_COMMIT,
        "patch_file": patch_path.name,
        "patch_sha256": _sha256(patch_path),
        "purpose": "iso1_conv_check_minimal",
        "note": (
            "Enables convective adjustment under isothermal_layers=yes for the "
            "source-treatment RCB counterfactual. Also writes conv flags in TP "
            "when convection is on. Not a HELIOS upstream claim."
        ),
    }


def _apply_computation(helios_root: Path) -> str:
    target = helios_root / "source" / "computation.py"
    text = target.read_text()
    if MARKER in text:
        return "already_patched"
    old = (
        "            if quant.iso == 0:\n"
        "                quant.kappa_int = quant.dev_kappa_int.get()\n"
        "                hsfunc.conv_check(quant)\n"
        "                hsfunc.mark_convective_layers(quant, stitching=0)\n"
        "\n"
        "            condition = sum(quant.conv_unstable) > 0\n"
    )
    new = (
        "            if quant.iso == 0:\n"
        "                quant.kappa_int = quant.dev_kappa_int.get()\n"
        "            else:\n"
        "                # Minimal ISO1 fix: stock HELIOS skips conv_check when iso==1,\n"
        "                # leaving conv_unstable=None and crashing on sum(...).\n"
        "                # Constant-κ cases never allocate kappa_int under iso==1 in read.py;\n"
        "                # build interfaces from layer κ so convective adjustment can run\n"
        "                # with isothermal radiation kernels. Radiation path unchanged.\n"
        "                if quant.kappa_int is None:\n"
        "                    import numpy as np\n"
        "                    quant.kappa_int = (\n"
        "                        np.ones(quant.ninterface, dtype=quant.kappa_lay.dtype)\n"
        "                        * float(quant.kappa_lay[0])\n"
        "                    )\n"
        "            hsfunc.conv_check(quant)\n"
        "            hsfunc.mark_convective_layers(quant, stitching=0)\n"
        "\n"
        "            condition = sum(quant.conv_unstable) > 0\n"
    )
    if old not in text:
        raise SystemExit(
            f"ISO1 patch anchor not found in {target}. "
            f"Expected pinned HELIOS {PINNED_HELIOS_COMMIT}."
        )
    target.write_text(text.replace(old, new, 1))
    if MARKER not in target.read_text():
        raise SystemExit("ISO1 computation.py patch failed to insert marker")
    return "patched"


def _apply_write(helios_root: Path) -> str:
    """Stock write.py blanks conv flags whenever iso==1; emit them if convection on."""
    target = helios_root / "source" / "write.py"
    text = target.read_text()
    if WRITE_MARKER in text:
        return "already_patched"
    old = (
        "                if quant.iso == 0 and quant.convection == 1:\n"
        "                    file.writelines(\"{:<30g}{:<32g}\".format("
        "quant.conv_unstable[quant.nlayer], quant.conv_layer[quant.nlayer]))\n"
        "\n"
        "                if quant.iso == 1 or quant.convection == 0:\n"
        "                    file.writelines(\"{:<30}{:<32}\".format("
        "\"not_calculated\", \"not_calculated\"))\n"
    )
    new = (
        "                # Minimal ISO1 fix: write conv flags when convection==1\n"
        "                # (stock blanks them for iso==1 even if adjustment ran).\n"
        "                if quant.convection == 1:\n"
        "                    file.writelines(\"{:<30g}{:<32g}\".format("
        "quant.conv_unstable[quant.nlayer], quant.conv_layer[quant.nlayer]))\n"
        "                else:\n"
        "                    file.writelines(\"{:<30}{:<32}\".format("
        "\"not_calculated\", \"not_calculated\"))\n"
    )
    old2 = (
        "                    if quant.iso == 0 and quant.convection == 1:\n"
        "                        file.writelines(\"{:<30g}{:<32g}\".format("
        "quant.conv_unstable[i], quant.conv_layer[i]))\n"
        "                    if quant.iso == 1 or quant.convection == 0:\n"
        "                        file.writelines(\"{:<30}{:<32}\".format("
        "\"not_calculated\", \"not_calculated\"))\n"
    )
    new2 = (
        "                    if quant.convection == 1:\n"
        "                        file.writelines(\"{:<30g}{:<32g}\".format("
        "quant.conv_unstable[i], quant.conv_layer[i]))\n"
        "                    else:\n"
        "                        file.writelines(\"{:<30}{:<32}\".format("
        "\"not_calculated\", \"not_calculated\"))\n"
    )
    if old not in text or old2 not in text:
        raise SystemExit(f"ISO1 write.py patch anchors not found in {target}")
    text = text.replace(old, new, 1).replace(old2, new2, 1)
    target.write_text(text)
    if WRITE_MARKER not in target.read_text():
        raise SystemExit("ISO1 write.py patch failed to insert marker")
    return "patched"


def apply(helios_root: Path, *, patch_path: Path = DEFAULT_PATCH) -> str:
    del patch_path  # content is applied from inlined anchors; file is provenance
    c = _apply_computation(helios_root)
    w = _apply_write(helios_root)
    if c == "already_patched" and w == "already_patched":
        return "already_patched"
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helios-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument(
        "--require-patch-checksum",
        action="store_true",
        help="Refuse if patch file sha256 != sidecar JSON",
    )
    args = parser.parse_args()
    meta_path = args.patch.with_suffix(args.patch.suffix + ".json")
    if not meta_path.exists():
        meta_path = Path(str(args.patch) + ".json")
    if args.require_patch_checksum:
        if not meta_path.exists():
            raise SystemExit(f"missing patch sidecar {meta_path}")
        meta = json.loads(meta_path.read_text())
        got = _sha256(args.patch)
        want = meta.get("patch_sha256")
        if got != want:
            raise SystemExit(f"patch sha256={got} != pinned {want}")
    status = apply(args.helios_root, patch_path=args.patch)
    print(json.dumps({"status": status, **patch_provenance(args.patch)}, indent=2))


if __name__ == "__main__":
    main()
