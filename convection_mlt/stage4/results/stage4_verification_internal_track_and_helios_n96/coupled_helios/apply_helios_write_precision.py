"""Patch HELIOS write_integrated_flux to space-delimited full precision after checkout."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "helios"
DEFAULT_PATCH = FIXTURES / "helios_write_integrated_flux_b0800f9.patch"
PINNED_HELIOS_COMMIT = "b0800f9ea4366263241c13bb926e8ca68f266cc5"

MARKER = "space-delimited HELIOS flux writer"

ANCHOR = "    @staticmethod\n    def write_integrated_flux(quant, read):"
NEXT = "    @staticmethod\n    def write_upward_spectral_flux(quant, read):"

NEW_METHOD = '''    @staticmethod
    def write_integrated_flux(quant, read):
        """ writes the integrated total and net fluxes to a file """
        try:
            with open(read.output_path + quant.name+"/" + quant.name + "_integrated_flux.dat", "w") as file:
                file.writelines("This file contains the integrated total and net fluxes at each interface resp. "
                                "layer. \\nFluxes given in [erg s^-1 cm^-2].")
                # space-delimited HELIOS flux writer
                file.writelines(
                    "\\n" + " ".join(
                        ["interface", "press.[10^-6bar]", "F_down", "F_up", "F_net", "F_dir",
                         "delta_F_net", "F_net_conv", "F_add_heat", "F_intern"]
                    )
                )
                for i in range(quant.ninterface):
                    if quant.singlewalk == 0 and i < quant.nlayer:
                        delta_f_net_text = "{:.17e}".format(quant.F_net_diff[i])
                    else:
                        delta_f_net_text = "not_avail."
                    if i < quant.nlayer:
                        f_add_heat_text = "{:.17e}".format(quant.F_add_heat_lay[i])
                    else:
                        f_add_heat_text = "not_avail."
                    fields = [
                        str(i),
                        "{:.17e}".format(quant.p_int[i]),
                        "{:.17e}".format(quant.F_down_tot[i]),
                        "{:.17e}".format(quant.F_up_tot[i]),
                        "{:.17e}".format(quant.F_net[i]),
                        "{:.17e}".format(quant.F_dir_tot[i]),
                        delta_f_net_text,
                        "{:.17e}".format(quant.F_net_conv[i]),
                        f_add_heat_text,
                    ]
                    if i == 0:
                        fields.append("{:.17e}".format(quant.F_intern))
                    file.writelines("\\n" + " ".join(fields))

        except TypeError:
            print("Integrated flux-file generation corrupted. You might want to look into it!")

'''


def _replace_method(text: str) -> str:
    start = text.find(ANCHOR)
    nxt = text.find(NEXT)
    if start < 0 or nxt < 0 or nxt <= start:
        raise SystemExit("could not locate write_integrated_flux / write_upward_spectral_flux in HELIOS write.py")
    return text[:start] + NEW_METHOD + text[nxt:]


def unified_write_py_diff(original: str, patched: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/source/write.py {PINNED_HELIOS_COMMIT}",
            tofile="b/source/write.py space-delimited-.17e",
        )
    )


def verify_applied_diff(diff: str, patch_path: Path = DEFAULT_PATCH) -> dict:
    """Fail unless the live unified diff matches the versioned patch checksum."""
    sidecar = patch_provenance(patch_path)
    digest = hashlib.sha256(diff.encode()).hexdigest()
    expected = sidecar.get("sha256")
    if expected is None:
        raise SystemExit(f"missing versioned write.py patch checksum at {patch_path}")
    if digest != expected:
        raise SystemExit(
            f"applied write.py diff sha256={digest} != pinned {expected} "
            f"({sidecar.get('patch_file')} @ {sidecar.get('helios_commit')})"
        )
    return sidecar


def apply(write_py: Path, *, require_patch_checksum: bool = False) -> str:
    text = write_py.read_text()
    if MARKER in text:
        ast.parse(text)
        return "already_patched"
    patched = _replace_method(text)
    ast.parse(patched)
    sidecar = DEFAULT_PATCH.with_suffix(".patch.json")
    if require_patch_checksum or sidecar.exists():
        verify_applied_diff(unified_write_py_diff(text, patched))
    write_py.write_text(patched)
    return "patched"


def emit_patch(write_py: Path, patch_path: Path = DEFAULT_PATCH) -> dict:
    """Write a unified diff of write_integrated_flux plus a checksum sidecar."""
    original = write_py.read_text()
    if MARKER in original:
        raise SystemExit("write.py is already patched; emit the diff from the pinned original")
    patched = _replace_method(original)
    ast.parse(patched)
    diff = unified_write_py_diff(original, patched)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(diff)
    digest = hashlib.sha256(diff.encode()).hexdigest()
    sidecar = {
        "helios_commit": PINNED_HELIOS_COMMIT,
        "patch_file": patch_path.name,
        "sha256": digest,
        "applies_to": "source/write.py write_integrated_flux",
        "format": "unified_diff",
        "purpose": "space-delimited {:.17e} integrated-flux writer after git checkout of the pinned commit",
    }
    patch_path.with_suffix(".patch.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def patch_provenance(patch_path: Path = DEFAULT_PATCH) -> dict:
    sidecar = patch_path.with_suffix(".patch.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    if patch_path.exists():
        return {
            "helios_commit": PINNED_HELIOS_COMMIT,
            "patch_file": patch_path.name,
            "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        }
    return {"helios_commit": PINNED_HELIOS_COMMIT, "patch_file": None, "sha256": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-py", type=Path, default=None)
    parser.add_argument("--emit-patch", action="store_true")
    parser.add_argument("--patch-out", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--require-patch-checksum", action="store_true")
    args = parser.parse_args()
    if args.emit_patch:
        if args.write_py is None:
            raise SystemExit("--write-py is required with --emit-patch")
        payload = emit_patch(args.write_py, args.patch_out)
        print(json.dumps(payload, indent=2))
        return
    if args.write_py is None:
        raise SystemExit("--write-py is required")
    status = apply(args.write_py, require_patch_checksum=args.require_patch_checksum)
    print(json.dumps({"status": status, **patch_provenance(args.patch_out)}))


if __name__ == "__main__":
    main()
