#!/bin/bash
set -euo pipefail

PROJECT_NAME="cv-hackathon"

module load stack/2024-06
module load gcc
module load python

if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python-headless matplotlib tqdm scikit-image pandas

pip freeze > requirements.txt

mkdir -p "$SCRATCH/$PROJECT_NAME/data"
mkdir -p "$SCRATCH/$PROJECT_NAME/outputs"
mkdir -p "$SCRATCH/$PROJECT_NAME/logs"
mkdir -p "$SCRATCH/$PROJECT_NAME/checkpoints"

echo "Euler environment ready."
echo "Activate with: source venv/bin/activate"
