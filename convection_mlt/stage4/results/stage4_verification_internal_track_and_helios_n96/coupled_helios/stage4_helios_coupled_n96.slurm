#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-coupled-n96
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-helios-coupled-n96-%j.out
#SBATCH --error=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-helios-coupled-n96-%j.err

set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

bash stage4/cluster/run_stage4_helios_coupled_n96.sh 96
