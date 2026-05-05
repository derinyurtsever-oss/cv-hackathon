#!/bin/bash
#SBATCH --job-name=cv-bonus-frames
#SBATCH --output=logs/bonus_frames_%A_%a.out
#SBATCH --error=logs/bonus_frames_%A_%a.err
#SBATCH --array=1-20%4
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

set -euo pipefail

cd /cluster/home/liadler/projects/cv-hackathon
mkdir -p logs

python3 scripts/extract_one_video.py --kind bonus --channel "$SLURM_ARRAY_TASK_ID"
