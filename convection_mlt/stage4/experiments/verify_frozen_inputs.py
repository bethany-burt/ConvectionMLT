"""Verify checksummed HELIOS inputs before a live scoring run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "helios"
MANIFEST = FIXTURES / "frozen_input_manifest.json"
COUPLED_MANIFEST = FIXTURES / "coupled_input_manifest.json"

RADIATION_ONLY_FILES = {
    "analytic_grey_nested.h5": FIXTURES / "analytic_grey_nested.h5",
    "constant_grey.h5": FIXTURES / "constant_grey.h5",
    "pressure_tagged.h5": FIXTURES / "pressure_tagged.h5",
    "radiation_only_tolerances.json": FIXTURES / "radiation_only_tolerances.json",
    "helios_write_integrated_flux_b0800f9.patch": (
        FIXTURES / "helios_write_integrated_flux_b0800f9.patch"
    ),
    "helios_write_integrated_flux_b0800f9.patch.json": (
        FIXTURES / "helios_write_integrated_flux_b0800f9.patch.json"
    ),
}

COUPLED_FILES = {
    "coupled_rce_benchmark_tolerances.json": FIXTURES / "coupled_rce_benchmark_tolerances.json",
    "helios_coupled_n96_runtime_config.json": FIXTURES / "helios_coupled_n96_runtime_config.json",
    "helios_param_template.dat": FIXTURES / "helios_param_template.dat",
    "beam_contract.json": FIXTURES / "beam_contract.json",
    "analytic_grey_nested.json": FIXTURES / "analytic_grey_nested.json",
    "helios_write_integrated_flux_b0800f9.patch": (
        FIXTURES / "helios_write_integrated_flux_b0800f9.patch"
    ),
    "helios_write_integrated_flux_b0800f9.patch.json": (
        FIXTURES / "helios_write_integrated_flux_b0800f9.patch.json"
    ),
    "analytic_grey_nested.h5": FIXTURES / "analytic_grey_nested.h5",
    "mlt_nested_tau_n96_firr0.json": FIXTURES / "mlt_nested_tau_n96_firr0.json",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text())


def verify(manifest: dict, *, fixtures: Path = FIXTURES) -> dict:
    results = {}
    ok = True
    for name, expected in (manifest.get("files") or {}).items():
        path = fixtures / name
        if not path.exists():
            results[name] = {"status": "MISSING", "path": str(path)}
            ok = False
            continue
        digest = file_sha256(path)
        match = digest == expected
        results[name] = {
            "status": "PASS" if match else "MISMATCH",
            "expected": expected,
            "observed": digest,
            "path": str(path),
        }
        ok = ok and match
    return {"status": "PASS" if ok else "FAIL", "files": results}


def write_manifest(files: dict[str, Path], *, out: Path = MANIFEST) -> dict:
    payload = {
        "purpose": "Checksums of frozen HELIOS inputs. Live wrappers must verify, never rebuild.",
        "files": {name: file_sha256(path) for name, path in files.items()},
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--write", action="store_true", help="Rebuild the manifest from current fixtures.")
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="Subset of manifest file names that must be present and matching.",
    )
    args = parser.parse_args()
    if args.write:
        coupled = args.manifest.name == "coupled_input_manifest.json"
        files = COUPLED_FILES if coupled else RADIATION_ONLY_FILES
        existing = {k: v for k, v in files.items() if v.exists()}
        previous = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
        hashed = {name: file_sha256(path) for name, path in existing.items()}
        if coupled:
            merged = dict(previous.get("files") or {})
            merged.update(hashed)
            payload = {
                "purpose": (
                    "Checksums for coupled-HELIOS inputs. Live wrappers must verify "
                    "this manifest, never rebuild the opacity table."
                ),
                "helios_commit": previous.get("helios_commit")
                or "b0800f9ea4366263241c13bb926e8ca68f266cc5",
                "opacity_table_note": (
                    "analytic_grey_nested.h5 lives on the cluster fixtures directory. "
                    "checksum_sha256 in analytic_grey_nested.json is the HDF5 file digest."
                ),
                "files": merged,
            }
            args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
        else:
            payload = write_manifest(existing, out=args.manifest)
        print(json.dumps(payload, indent=2))
        return payload
    if not args.manifest.exists():
        raise SystemExit(f"missing frozen input manifest: {args.manifest}")
    payload = verify(load_manifest(args.manifest))
    if args.require:
        for name in args.require:
            status = (payload["files"].get(name) or {}).get("status")
            if status != "PASS":
                payload["status"] = "FAIL"
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("frozen input checksum verification failed")
    return payload


if __name__ == "__main__":
    main()
