#!/bin/bash
#SBATCH --job-name=cv-extract-frames
#SBATCH --output=logs/extract_frames_%j.out
#SBATCH --error=logs/extract_frames_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G

set -euo pipefail

cd /cluster/home/liadler/projects/cv-hackathon
mkdir -p logs

python3 SETUP_frame_images_extraction/extract_frame_images.py
python3 SETUP_frame_images_extraction/extract_bonus_frame_images.py
