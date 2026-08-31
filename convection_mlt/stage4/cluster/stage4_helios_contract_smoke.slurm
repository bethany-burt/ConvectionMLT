#!/bin/bash
#SBATCH --job-name=helios_smoke_n8
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=helios_contract_smoke_n8_%j.out
#SBATCH --error=helios_contract_smoke_n8_%j.err

set -euo pipefail
ROOT="/project/ls-heng/Bethany.Burt/convection_mlt"
bash "${ROOT}/stage4/cluster/run_stage4_helios_contract_smoke.sh"
