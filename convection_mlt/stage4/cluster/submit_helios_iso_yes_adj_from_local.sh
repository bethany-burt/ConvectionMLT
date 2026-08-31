#!/usr/bin/env bash
# Sync ISO1 counterfactual tooling and submit N=96 iso=yes+adj job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLUSTER_HOST="${CLUSTER_HOST:-Bethany.Burt@cluster.hpc.physik.uni-muenchen.de}"
CLUSTER_PARENT="${CLUSTER_PARENT:-/project/ls-heng/Bethany.Burt}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_NAME="${STAGE_NAME:-convection_mlt_iso1_n96_${STAMP}}"
CLUSTER_DIR="${CLUSTER_DIR:-${CLUSTER_PARENT}/${STAGE_NAME}}"
SUBMIT="${SUBMIT:-1}"

echo "rsync -> ${CLUSTER_HOST}:${CLUSTER_DIR}/"
ssh "${CLUSTER_HOST}" "mkdir -p '${CLUSTER_DIR}/stage4/results' '${CLUSTER_DIR}/stage4/fixtures/helios' '${CLUSTER_DIR}/stage4/experiments' '${CLUSTER_DIR}/stage4/cluster' '${CLUSTER_DIR}/src'"

rsync -az --exclude '__pycache__' --exclude '*.pyc' \
  "${PKG_ROOT}/src/" "${CLUSTER_HOST}:${CLUSTER_DIR}/src/"
rsync -az --exclude '__pycache__' \
  "${PKG_ROOT}/stage4/experiments/" "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/experiments/"
rsync -az \
  "${PKG_ROOT}/stage4/cluster/run_stage4_helios_iso_yes_adj_n96.sh" \
  "${PKG_ROOT}/stage4/cluster/stage4_helios_iso_yes_adj_n96.slurm" \
  "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/cluster/"
rsync -az \
  "${PKG_ROOT}/stage4/fixtures/helios/" "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/fixtures/helios/"

# Opacity + attribution JSON for merge
ssh "${CLUSTER_HOST}" bash -s <<REMOTE
set -euo pipefail
STAGING='${CLUSTER_DIR}'
PROD='${CLUSTER_PARENT}/convection_mlt'
OPAC_ST="\${STAGING}/stage4/fixtures/helios/analytic_grey_nested.h5"
OPAC_PROD="\${PROD}/stage4/fixtures/helios/analytic_grey_nested.h5"
if [[ ! -f "\${OPAC_ST}" && -f "\${OPAC_PROD}" ]]; then
  ln -f "\${OPAC_PROD}" "\${OPAC_ST}" 2>/dev/null || cp -a "\${OPAC_PROD}" "\${OPAC_ST}"
fi
REMOTE

rsync -az \
  "${PKG_ROOT}/stage4/results/helios_coupled_n96_rcb_attribution.json" \
  "${CLUSTER_HOST}:${CLUSTER_DIR}/stage4/results/" 2>/dev/null || true

echo "staging CLUSTER_DIR=${CLUSTER_DIR}"
if [[ "${SUBMIT}" == "1" ]]; then
  ssh "${CLUSTER_HOST}" bash -s <<REMOTE
set -euo pipefail
cd '${CLUSTER_DIR}'
chmod +x stage4/cluster/run_stage4_helios_iso_yes_adj_n96.sh
JOB=\$(sbatch --export=ALL,STAGE4_ROOT='${CLUSTER_DIR}' stage4/cluster/stage4_helios_iso_yes_adj_n96.slurm)
echo "\$JOB"
REMOTE
else
  echo "SUBMIT=0; staging only"
fi
