#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-opac-diag
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project/ls-heng/Bethany.Burt/helios_stage4_frozen/slurm-helios-opac-diag-%j.out
#SBATCH --error=/project/ls-heng/Bethany.Burt/helios_stage4_frozen/slurm-helios-opac-diag-%j.err

set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
KIND="${HELIOS_DIAG_KIND:-constant}"
LAYERS="${HELIOS_FROZEN_LAYERS:-96}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

bash stage4/cluster/run_stage4_helios_opacity_diag.sh "${KIND}" "${LAYERS}"
