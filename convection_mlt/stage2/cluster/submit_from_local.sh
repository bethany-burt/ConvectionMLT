#!/usr/bin/env bash
# Sync this package to the Munich physics HPC login node and submit the
# Stage 2 production SLURM job.
#
# CLUSTER_DIR must be the remote path of *convection_mlt itself*
# (not the parent ConvectionMLT repo). Example layout on the cluster:
#   /project/ls-heng/Bethany.Burt/convection_mlt/
#
# Required:
#   CLUSTER_HOST   e.g. Bethany.Burt@cluster.hpc.physik.uni-muenchen.de
#   CLUSTER_DIR    absolute remote path ending in .../convection_mlt
#
# Optional:
#   CLUSTER_PARTITION / CLUSTER_ACCOUNT
#   RSYNC_DELETE=1
#
# Example:
#   export CLUSTER_HOST=Bethany.Burt@cluster.hpc.physik.uni-muenchen.de
#   export CLUSTER_DIR=/project/ls-heng/Bethany.Burt/convection_mlt
#   ./stage2/cluster/submit_from_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLUSTER_HOST="${CLUSTER_HOST:-Bethany.Burt@cluster.hpc.physik.uni-muenchen.de}"
CLUSTER_DIR="${CLUSTER_DIR:?Set CLUSTER_DIR to the absolute remote path of convection_mlt/}"

if [[ "${CLUSTER_DIR}" == /Users/* ]]; then
  echo "error: CLUSTER_DIR='${CLUSTER_DIR}' looks like a local macOS path." >&2
  echo "Point it at the cluster copy of convection_mlt, e.g.:" >&2
  echo "  export CLUSTER_DIR=/project/ls-heng/Bethany.Burt/convection_mlt" >&2
  exit 1
fi

if [[ "$(basename "${CLUSTER_DIR}")" != "convection_mlt" ]]; then
  echo "error: CLUSTER_DIR should be the remote convection_mlt directory," >&2
  echo "got: ${CLUSTER_DIR}" >&2
  echo "example: /project/ls-heng/Bethany.Burt/convection_mlt" >&2
  exit 1
fi

RSYNC_FLAGS=(-az --exclude '.venv' --exclude 'venv' --exclude '__pycache__'
  --exclude '.pytest_cache' --exclude '*.pyc'
  --exclude 'stage1/plots/generated' --exclude 'stage1/results'
  --exclude '.git')
if [[ "${RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_FLAGS+=(--delete)
fi

echo "rsync ${PKG_ROOT}/ -> ${CLUSTER_HOST}:${CLUSTER_DIR}/"
ssh "${CLUSTER_HOST}" "mkdir -p '${CLUSTER_DIR}/stage2/results'"
rsync "${RSYNC_FLAGS[@]}" "${PKG_ROOT}/" "${CLUSTER_HOST}:${CLUSTER_DIR}/"

SBATCH_EXTRA=()
if [[ -n "${CLUSTER_PARTITION:-}" ]]; then
  SBATCH_EXTRA+=(--partition="${CLUSTER_PARTITION}")
fi
if [[ -n "${CLUSTER_ACCOUNT:-}" ]]; then
  SBATCH_EXTRA+=(--account="${CLUSTER_ACCOUNT}")
fi

echo "submitting on ${CLUSTER_HOST} from ${CLUSTER_DIR}"
ssh "${CLUSTER_HOST}" "cd '${CLUSTER_DIR}' && mkdir -p stage2/results && sbatch ${SBATCH_EXTRA[*]:-} stage2/cluster/production_campaign.slurm"
