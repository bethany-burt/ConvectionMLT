#!/usr/bin/env bash
# Sync and submit HELIOS N=8 contract smoke, then optionally N96-A / N192.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLUSTER_HOST="${CLUSTER_HOST:-Bethany.Burt@cluster.hpc.physik.uni-muenchen.de}"
CLUSTER_DIR="${CLUSTER_DIR:?Set CLUSTER_DIR to the absolute remote path of convection_mlt/}"
JOB="${HELIOS_JOB:-smoke}"

if [[ "${CLUSTER_DIR}" == /Users/* ]]; then
  echo "error: CLUSTER_DIR='${CLUSTER_DIR}' looks like a local macOS path." >&2
  exit 1
fi

RSYNC_FLAGS=(-az --exclude '.venv' --exclude 'venv' --exclude '__pycache__'
  --exclude '.pytest_cache' --exclude '*.pyc' --exclude '.git'
  --exclude 'stage4/results/' --exclude 'stage4/results/**')
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_FLAGS+=(--delete)
fi

echo "rsync ${PKG_ROOT}/ -> ${CLUSTER_HOST}:${CLUSTER_DIR}/"
ssh "${CLUSTER_HOST}" "mkdir -p '${CLUSTER_DIR}/stage4/results' '${CLUSTER_DIR}/stage4/fixtures/helios'"
rsync "${RSYNC_FLAGS[@]}" "${PKG_ROOT}/" "${CLUSTER_HOST}:${CLUSTER_DIR}/"

SBATCH_EXPORT="NONE"
case "${JOB}" in
  smoke)
    SLURM="stage4/cluster/stage4_helios_contract_smoke.slurm"
    ;;
  n192|192)
    SLURM="stage4/cluster/stage4_helios_gated_n192.slurm"
    ;;
  constant|const|diag)
    SLURM="stage4/cluster/stage4_helios_opacity_diag.slurm"
    SBATCH_EXPORT="HELIOS_DIAG_KIND=constant,HELIOS_FROZEN_LAYERS=96"
    ;;
  tagged)
    SLURM="stage4/cluster/stage4_helios_opacity_diag.slurm"
    SBATCH_EXPORT="HELIOS_DIAG_KIND=tagged,HELIOS_FROZEN_LAYERS=96"
    ;;
  analytic)
    SLURM="stage4/cluster/stage4_helios_opacity_diag.slurm"
    SBATCH_EXPORT="HELIOS_DIAG_KIND=analytic,HELIOS_FROZEN_LAYERS=96"
    ;;
  gated)
    SLURM="stage4/cluster/stage4_helios_gated_diag.slurm"
    ;;
  n96|96)
    echo "N=96 radiation-only already PASSed. Use HELIOS_JOB=n192." >&2
    exit 2
    ;;
  *)
    echo "Unknown HELIOS_JOB=${JOB}; use smoke|constant|tagged|analytic|gated|n192" >&2
    exit 1
    ;;
esac

echo "submitting ${JOB} on ${CLUSTER_HOST}"
EXCLUDE_FLAG=()
if [[ -n "${SBATCH_EXCLUDE:-th-cl-naples02}" ]]; then
  EXCLUDE_FLAG=(--exclude="${SBATCH_EXCLUDE:-th-cl-naples02}")
fi
if [[ "${SBATCH_EXPORT}" == "NONE" ]]; then
  ssh "${CLUSTER_HOST}" "cd '${CLUSTER_DIR}' && sbatch ${EXCLUDE_FLAG[*]} ${SLURM}"
else
  ssh "${CLUSTER_HOST}" "cd '${CLUSTER_DIR}' && sbatch ${EXCLUDE_FLAG[*]} --export=${SBATCH_EXPORT} ${SLURM}"
fi
