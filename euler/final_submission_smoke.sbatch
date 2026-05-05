#!/bin/bash
#SBATCH --job-name=cv-final-smoke
#SBATCH --output=logs/final_smoke_%j.out
#SBATCH --error=logs/final_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

cd /cluster/home/liadler/projects/cv-hackathon
mkdir -p logs

module load stack/2024-06
module load gcc
module load python

source venv/bin/activate

VIDEO="${VIDEO:-10}"

python - <<'PY'
import os
from mySolution.MovementPathEstimator import MovementPathEstimator

video = int(os.environ["VIDEO"])
est = MovementPathEstimator(video, False)
est.execute_estimations()
mp = est.calculated_movement_paths[video]
print("video", video)
print("frames", len(mp.movement_path))
print("turn", mp.turning_point)
print("min", float(mp.movement_path.min()))
print("max", float(mp.movement_path.max()))
print("dir_values", sorted(set(float(x) for x in mp.movement_direction[:50])))
PY
