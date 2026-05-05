#!/bin/bash
#SBATCH --job-name=cv-ensemble-eval
#SBATCH --output=logs/ensemble_eval_%j.out
#SBATCH --error=logs/ensemble_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

cd /cluster/home/liadler/projects/cv-hackathon
mkdir -p logs

module load stack/2024-06
module load gcc
module load python

source venv/bin/activate

TRAIN_VIDEOS="${TRAIN_VIDEOS:-1 2 3 4 8 9}"
PREDICT_VIDEOS="${PREDICT_VIDEOS:-10 11}"
TRAIN_VIDEOS="${TRAIN_VIDEOS// /,}"
PREDICT_VIDEOS="${PREDICT_VIDEOS// /,}"

python mySolution/ensemble_hybrid_eval.py \
  --train-videos "$TRAIN_VIDEOS" \
  --predict-videos "$PREDICT_VIDEOS"
