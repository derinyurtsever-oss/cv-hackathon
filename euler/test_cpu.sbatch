#!/bin/bash
#SBATCH --job-name=cv_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --output=/cluster/scratch/%u/cv-hackathon/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/cv-hackathon/logs/%x-%j.err

set -euo pipefail

cd /cluster/home/$USER/projects/cv-hackathon

module load stack/2024-06
module load gcc
module load python

source venv/bin/activate

mkdir -p "$SCRATCH/cv-hackathon/logs"
mkdir -p "$SCRATCH/cv-hackathon/outputs"

python - <<'PY'
from pathlib import Path
import sys

print("Euler Slurm test job works.")
print(f"Python version: {sys.version}")
Path("euler_test_output.txt").write_text("Euler test job completed.\n")
PY
