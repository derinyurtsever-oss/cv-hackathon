#!/bin/bash
#SBATCH --job-name=cv-main-frames
#SBATCH --output=logs/main_frames_%A_%a.out
#SBATCH --error=logs/main_frames_%A_%a.err
#SBATCH --array=1-11%4
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

set -euo pipefail

cd /cluster/home/liadler/projects/cv-hackathon
mkdir -p logs

python3 scripts/extract_one_video.py --kind main --channel "$SLURM_ARRAY_TASK_ID"
