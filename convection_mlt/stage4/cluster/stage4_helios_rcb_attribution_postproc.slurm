#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-rcb-attrib
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-helios-rcb-attrib-n96-%j.out
#SBATCH --error=slurm-helios-rcb-attrib-n96-%j.err

set -euo pipefail
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "${ROOT}/stage4/results"
cd "${ROOT}"
bash stage4/cluster/run_stage4_helios_rcb_attribution_postproc.sh
