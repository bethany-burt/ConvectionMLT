#!/usr/bin/env bash
# Sync corrected coupled-HELIOS inputs into a staging tree, verify the coupled
# manifest on the cluster, then optionally submit N=96.
#
# Staging keeps an older production convection_mlt/ tree untouched until the
# verifier PASSes. HELIOS case output uses a timestamped directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLUSTER_HOST="${CLUSTER_HOST:-Bethany.Burt@cluster.hpc.physik.uni-muenchen.de}"
CLUSTER_PARENT="${CLUSTER_PARENT:-/project/ls-heng/Bethany.Burt}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_NAME="${STAGE_NAME:-convection_mlt_coupled_n96_${STAMP}}"
# Prefer an explicit staging tree. A pre-exported CLUSTER_DIR that points at the
# production convection_mlt/ package is refused unless ALLOW_PRODUCTION_SYNC=1.
DEFAULT_STAGE="${CLUSTER_PARENT}/${STAGE_NAME}"
if [[ -n "${CLUSTER_DIR:-}" && "${CLUSTER_DIR}" == */convection_mlt && "${ALLOW_PRODUCTION_SYNC:-0}" != "1" ]]; then
  echo "Refusing to sync into production CLUSTER_DIR=${CLUSTER_DIR}." >&2
  echo "Unset CLUSTER_DIR to use staging ${DEFAULT_STAGE}, or set ALLOW_PRODUCTION_SYNC=1." >&2
  exit 2
fi
CLUSTER_DIR="${CLUSTER_DIR:-${DEFAULT_STAGE}}"
SUBMIT="${SUBMIT:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

if [[ "${CLUSTER_DIR}" == /Users/* ]]; then
  echo "error: CLUSTER_DIR='${CLUSTER_DIR}' looks like a local macOS path." >&2
  exit 1
fi

RSYNC_FLAGS=(-az --exclude '.venv' --exclude 'venv' --exclude '__pycache__'
  --exclude '.pytest_cache' --exclude '*.pyc' --exclude '.git'
  --exclude 'stage4/results/' --exclude 'stage4/results/**'
  --exclude 'stage4/plots/generated/' --exclude 'stage4/plots/generated/**')
# Always ship the frozen F_irr=0 MLT reference and negative diagnostic index.
# The large diagnostic JSON bodies stay local unless INCLUDE_DIAGNOSTICS=1.
if [[ "${INCLUDE_DIAGNOSTICS:-0}" == "1" ]]; then
  :
fi

echo "rsync ${PKG_ROOT}/ -> ${CLUSTER_HOST}:${CLUSTER_DIR}/"
ssh "${CLUSTER_HOST}" "mkdir -p '${CLUSTER_DIR}/stage4/results' '${CLUSTER_DIR}/stage4/fixtures/helios'"
rsync "${RSYNC_FLAGS[@]}" "${PKG_ROOT}/" "${CLUSTER_HOST}:${CLUSTER_DIR}/"

# Explicitly sync the frozen MLT reference (excluded from results/ but required in fixtures).
rsync -az \
  "${PKG_ROOT}/stage4/fixtures/helios/mlt_nested_tau_n96_firr0.json" \
  "${PKG_ROOT}/stage4/fixtures/helios/coupled_input_manifest.json" \
  "${PKG_ROOT}/stage4/fixtures/helios/coupled_rce_benchmark_tolerances.json" \
  "${PKG_ROOT}/stage4/fixtures/helios/helios_coupled_n96_runtime_config.json" \
  "${PKG_ROOT}/stage4/fixtures/helios/analytic_grey_nested.json" \
  "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/fixtures/helios/"

# Reuse the production opacity HDF5 if staging does not yet contain a copy.
ssh "${CLUSTER_HOST}" bash -s <<REMOTE
set -euo pipefail
STAGING='${CLUSTER_DIR}'
PROD='${CLUSTER_PARENT}/convection_mlt'
OPAC_ST="\${STAGING}/stage4/fixtures/helios/analytic_grey_nested.h5"
OPAC_PROD="\${PROD}/stage4/fixtures/helios/analytic_grey_nested.h5"
if [[ ! -f "\${OPAC_ST}" ]]; then
  if [[ -f "\${OPAC_PROD}" ]]; then
    ln -f "\${OPAC_PROD}" "\${OPAC_ST}" 2>/dev/null || cp -a "\${OPAC_PROD}" "\${OPAC_ST}"
  else
    echo "missing analytic_grey_nested.h5 in staging and production" >&2
    exit 2
  fi
fi
cd "\${STAGING}"
export PYTHONPATH="\${STAGING}/src\${PYTHONPATH:+:\$PYTHONPATH}"
python3 stage4/experiments/verify_frozen_inputs.py \
  --manifest stage4/fixtures/helios/coupled_input_manifest.json
python3 - <<'PY'
import hashlib, json, re
from pathlib import Path
fix = Path("stage4/fixtures/helios")
h5 = fix / "analytic_grey_nested.h5"
digest = hashlib.sha256(h5.read_bytes()).hexdigest()
want = "9505247e1104c9d11500944975a2d26b82d55c4e3c7c66f579a5a9c08334cd3c"
assert digest == want, (digest, want)
man = json.loads((fix / "coupled_input_manifest.json").read_text())
assert man["files"]["analytic_grey_nested.h5"] == want
mlt = json.loads((fix / "mlt_nested_tau_n96_firr0.json").read_text())
assert mlt["status"] == "converged"
assert float(mlt["f_irr"]) == 0.0
assert float(mlt["flux_flatness"]) < 1.0e-3
assert mlt["profile_checksum_sha256"].startswith("b5eb3508")
print(json.dumps({
    "coupled_manifest": "PASS",
    "h5_sha256": digest,
    "mlt_profile": mlt["profile_checksum_sha256"],
    "mlt_flatness": mlt["flux_flatness"],
    "staging": str(Path(".").resolve()),
}, indent=2))
PY
REMOTE

echo "staging CLUSTER_DIR=${CLUSTER_DIR}"
if [[ "${VERIFY_ONLY}" == "1" ]]; then
  echo "VERIFY_ONLY=1; not submitting"
  exit 0
fi
if [[ "${SUBMIT}" != "1" ]]; then
  echo "Manifest PASS. Re-run with SUBMIT=1 to sbatch N=96 into this staging tree."
  exit 0
fi

OUT_ROOT="${HELIOS_COUPLED_OUT:-${CLUSTER_PARENT}/helios_stage4_coupled_${STAMP}}"
echo "submitting coupled N=96; HELIOS_COUPLED_OUT=${OUT_ROOT}"
ssh "${CLUSTER_HOST}" \
  "cd '${CLUSTER_DIR}' && \
   STAGE4_ROOT='${CLUSTER_DIR}' HELIOS_COUPLED_OUT='${OUT_ROOT}' \
   sbatch --export=ALL,STAGE4_ROOT='${CLUSTER_DIR}',HELIOS_COUPLED_OUT='${OUT_ROOT}' \
   stage4/cluster/stage4_helios_coupled_n96.slurm"
