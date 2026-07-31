#!/bin/bash
#SBATCH --job-name=vscode-tunnel
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6t/vbrekke/vscode_tunnel.log

source activate /projects/u6t/vbrekke/envs/japan-model

/projects/u6t/vbrekke/code tunnel --accept-server-license-terms
