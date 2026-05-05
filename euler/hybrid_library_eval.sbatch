#!/bin/bash
#SBATCH --job-name=cv-hybrid-eval
#SBATCH --output=logs/hybrid_eval_%j.out
#SBATCH --error=logs/hybrid_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

cd /cluster/home/liadler/projects/cv-hackathon
mkdir -p logs

module load stack/2024-06
module load gcc
module load python

source venv/bin/activate

# Comma-separated lists do not survive `sbatch --export=...` reliably on Euler.
# Pass them with spaces and normalize here.
TRAIN_VIDEOS="${TRAIN_VIDEOS:-1 2 3 4 8 9}"
PREDICT_VIDEOS="${PREDICT_VIDEOS:-10 11}"
FORCE_ANNULAR="${FORCE_ANNULAR:-0}"

ANNULAR_FLAG=""
if [ "$FORCE_ANNULAR" = "1" ]; then
  ANNULAR_FLAG="--force-annular"
fi

TRAIN_VIDEOS="${TRAIN_VIDEOS// /,}"
PREDICT_VIDEOS="${PREDICT_VIDEOS// /,}"

python mySolution/hybrid_library_eval.py \
  --train-videos "$TRAIN_VIDEOS" \
  --predict-videos "$PREDICT_VIDEOS" \
  $ANNULAR_FLAG
