#!/usr/bin/env bash
# Sync attribution tooling + pilot final TP/flux, then submit post-proc HELIOS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLUSTER_HOST="${CLUSTER_HOST:-Bethany.Burt@cluster.hpc.physik.uni-muenchen.de}"
CLUSTER_PARENT="${CLUSTER_PARENT:-/project/ls-heng/Bethany.Burt}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_NAME="${STAGE_NAME:-convection_mlt_rcb_attrib_${STAMP}}"
CLUSTER_DIR="${CLUSTER_DIR:-${CLUSTER_PARENT}/${STAGE_NAME}}"
SUBMIT="${SUBMIT:-1}"

echo "rsync ${PKG_ROOT}/ -> ${CLUSTER_HOST}:${CLUSTER_DIR}/"
ssh "${CLUSTER_HOST}" "mkdir -p '${CLUSTER_DIR}/stage4/results' '${CLUSTER_DIR}/stage4/fixtures/helios' '${CLUSTER_DIR}/stage4/experiments' '${CLUSTER_DIR}/stage4/cluster' '${CLUSTER_DIR}/src'"

RSYNC_FLAGS=(-az --exclude '.venv' --exclude 'venv' --exclude '__pycache__'
  --exclude '.pytest_cache' --exclude '*.pyc' --exclude '.git'
  --exclude 'stage4/plots/generated/')

rsync "${RSYNC_FLAGS[@]}" \
  "${PKG_ROOT}/src/" "${CLUSTER_HOST}:${CLUSTER_DIR}/src/"
rsync "${RSYNC_FLAGS[@]}" \
  "${PKG_ROOT}/stage4/experiments/" "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/experiments/"
rsync "${RSYNC_FLAGS[@]}" \
  "${PKG_ROOT}/stage4/cluster/" "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/cluster/"
rsync "${RSYNC_FLAGS[@]}" \
  "${PKG_ROOT}/stage4/fixtures/helios/" "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/fixtures/helios/"

# Pilot finals required for final-T post-proc (results/ is often excluded elsewhere).
ssh "${CLUSTER_HOST}" "mkdir -p '${CLUSTER_DIR}/stage4/results/helios_coupled_n96_job16015698_debug/iterative'"
rsync -az \
  "${PKG_ROOT}/stage4/results/helios_coupled_n96_job16015698_debug/iterative/final_tp.dat" \
  "${PKG_ROOT}/stage4/results/helios_coupled_n96_job16015698_debug/iterative/final_integrated_flux.dat" \
  "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/results/helios_coupled_n96_job16015698_debug/iterative/"

# Opacity HDF5 from production if missing in staging.
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
    echo "missing analytic_grey_nested.h5" >&2
    exit 2
  fi
fi
REMOTE

echo "staging CLUSTER_DIR=${CLUSTER_DIR}"
if [[ "${SUBMIT}" == "1" ]]; then
  ssh "${CLUSTER_HOST}" bash -s <<REMOTE
set -euo pipefail
cd '${CLUSTER_DIR}'
export STAGE4_ROOT='${CLUSTER_DIR}'
JOB=\$(sbatch --export=ALL,STAGE4_ROOT='${CLUSTER_DIR}' stage4/cluster/stage4_helios_rcb_attribution_postproc.slurm)
echo "\$JOB"
REMOTE
else
  echo "SUBMIT=0; staging only"
fi
