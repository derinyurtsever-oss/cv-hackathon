# General Codex Instructions for the CV Hackathon Repository

Use this as a general instruction file or initial prompt for Codex when working on this repository.

---

## 1. Project context

You are helping develop a computer vision / machine learning project for an image comparison and feature tracking task.

The project should remain portable:

- It should run locally on a normal development machine.
- It should also run on ETH Zürich’s Euler HPC cluster.
- Euler-specific logic should stay inside the `euler/` folder.
- Core Python code should not depend on Euler-specific paths.
- A frame extraction script/tool will be provided separately.
- Different methodological approaches will be stored in different Git branches.

The repository may be used by several people, not all of whom use Euler. Do not make the project unusable for non-Euler users.

---

## 2. Main development principle

Keep algorithmic code separate from execution environment code.

Core code belongs in:

```text
main.py
src/
configs/
requirements.txt
README.md
```

Euler-specific code belongs in:

```text
euler/
```

Local outputs, logs, datasets, checkpoints and virtual environments should not be committed.

The same Python scripts should be runnable both locally and on Euler by passing different command-line paths.

---

## 3. Branch strategy

Different approaches should be developed in different Git branches.

Examples:

```text
main
approach/classical-orb
approach/akaze-ransac
approach/sift-homography
approach/lk-optical-flow
approach/pretrained-matcher
approach/siamese-cnn
approach/raft-flow
experiment/parameter-sweep-orb
experiment/debug-small-dataset
```

Use descriptive branch names.

Good branch names:

```text
approach/classical-orb-ransac
approach/lucas-kanade-tracking
approach/lightglue-pretrained
experiment/orb-ratio-sweep
fix/euler-paths
docs/euler-readme
```

Bad branch names:

```text
test
stuff
new
final
liam-branch
```

### Branch workflow

Before starting a new approach:

```bash
git checkout main
git pull
git checkout -b approach/<descriptive-name>
```

After changes:

```bash
git status
git add .
git commit -m "Implement <short description>"
git push -u origin approach/<descriptive-name>
```

When switching between approaches:

```bash
git checkout main
git pull
git checkout approach/<other-approach>
```

Do not mix unrelated approaches in the same branch.

Each branch should ideally have:

```text
clear README notes
specific config files
specific Slurm scripts if needed
separate output directory naming
```

---

## 4. Expected repository structure

Use this structure unless there is a good reason to change it:

```text
repo/
├── main.py
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── data.py
│   ├── models.py
│   ├── classical_cv.py
│   ├── tracking.py
│   └── utils.py
├── euler/
│   ├── README_EULER.md
│   ├── setup_euler.sh
│   ├── test_cpu.sbatch
│   ├── train_cpu.sbatch
│   ├── train_gpu.sbatch
│   ├── run_cv.sbatch
│   └── sweep.sbatch
├── configs/
│   ├── default.yaml
│   ├── orb.yaml
│   ├── akaze.yaml
│   └── lk_tracking.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

Do not put large files into this repository.

---

## 5. Frame extraction

A frame extraction script/tool will be provided separately.

Do not implement a custom frame extraction pipeline unless explicitly requested.

Assume extracted frames will be available as image files in an input directory, for example:

```bash
data/frames/
```

or on Euler:

```bash
$SCRATCH/cv-hackathon/data/frames/
```

The project code should focus on:

- loading already-extracted frames,
- comparing images,
- tracking features between frames,
- evaluating tracking quality,
- saving metrics and visualizations.

Expected local input style:

```bash
python src/classical_cv.py \
  --input_dir data/frames \
  --output_dir outputs/cv_test
```

Expected Euler wrapper style:

```bash
python src/classical_cv.py \
  --input_dir "$SCRATCH/cv-hackathon/data/frames" \
  --output_dir "$SCRATCH/cv-hackathon/outputs/cv-${SLURM_JOB_ID}"
```

Do not assume access to the original videos unless explicitly stated.

---

## 6. Coding style

When writing Python code:

- Use clean, modular Python.
- Prefer simple functions over large monolithic scripts.
- Use `argparse` for command-line arguments.
- Use `pathlib.Path` for paths.
- Add clear error messages when input files or directories are missing.
- Create output directories automatically.
- Print useful progress logs.
- Save metrics as `.json` or `.csv`.
- Save visualizations where helpful.
- Keep scripts runnable from both local shell and Slurm.
- Avoid hard-coded usernames, absolute local paths, or machine-specific assumptions.
- Avoid hidden side effects.
- Use deterministic behavior where possible, especially for comparisons between approaches.

Prefer this style:

```python
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # main logic here


if __name__ == "__main__":
    main()
```

---

## 7. Path rules

Do not hard-code personal or Euler-specific paths in Python code.

Bad:

```python
data_dir = "/cluster/scratch/some_username/cv-hackathon/data"
```

Good:

```bash
python src/classical_cv.py \
  --input_dir "$SCRATCH/cv-hackathon/data/frames" \
  --output_dir "$SCRATCH/cv-hackathon/outputs/run-${SLURM_JOB_ID}"
```

Good locally:

```bash
python src/classical_cv.py \
  --input_dir data/frames \
  --output_dir outputs/local_test
```

Use command-line arguments for paths.

---

## 8. Local execution

The project should be runnable locally with commands like:

```bash
python src/classical_cv.py \
  --input_dir data/frames \
  --output_dir outputs/local_test
```

For Windows PowerShell, multiline commands use backticks:

```powershell
python src/classical_cv.py `
  --input_dir data/frames `
  --output_dir outputs/local_test
```

Do not require Euler to run the core Python scripts.

---

## 9. Euler execution

Euler-specific execution should use Slurm scripts in the `euler/` folder.

Heavy work must be submitted with:

```bash
sbatch euler/run_cv.sbatch
```

or:

```bash
sbatch euler/train_cpu.sbatch
```

or, only if GPU access is confirmed:

```bash
sbatch euler/train_gpu.sbatch
```

Do not instruct users to run heavy training directly on the Euler login node.

Euler scripts should:

1. `cd` into the repository.
2. Load required modules.
3. Activate the virtual environment.
4. Create output/log directories.
5. Call a Python script with command-line arguments.

Typical Slurm script pattern:

```bash
#!/bin/bash
#SBATCH --job-name=cv_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/%u/cv-hackathon/logs/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/cv-hackathon/logs/%x-%j.err

set -euo pipefail

cd /cluster/home/$USER/projects/cv-hackathon

module load stack/2024-06
module load gcc
module load python

source venv/bin/activate

RUN_DIR="$SCRATCH/cv-hackathon/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
mkdir -p "$SCRATCH/cv-hackathon/logs"
mkdir -p "$SCRATCH/cv-hackathon/checkpoints"

python src/classical_cv.py \
  --input_dir "$SCRATCH/cv-hackathon/data/frames" \
  --output_dir "$RUN_DIR"
```

---

## 10. Euler storage rules

Use:

```bash
/cluster/home/$USER/projects/cv-hackathon
```

for code.

Use:

```bash
$SCRATCH/cv-hackathon/data
$SCRATCH/cv-hackathon/data/frames
$SCRATCH/cv-hackathon/outputs
$SCRATCH/cv-hackathon/logs
$SCRATCH/cv-hackathon/checkpoints
```

for data, extracted frames, logs, outputs and checkpoints.

Do not put large datasets or generated outputs in the repository.

---

## 11. Slurm monitoring commands

When giving Euler instructions, include relevant monitoring commands:

```bash
squeue -u $USER
squeue --start -u $USER
squeue --start -j <job_id>
scontrol show job <job_id>
scancel <job_id>
```

For logs:

```bash
ls -lh $SCRATCH/cv-hackathon/logs
cat $SCRATCH/cv-hackathon/logs/*.out
cat $SCRATCH/cv-hackathon/logs/*.err
tail -f $SCRATCH/cv-hackathon/logs/<log-file>.out
```

Explain common Slurm states when relevant:

```text
PD = pending
R  = running
CG = completing
CD = completed
F  = failed
CA = cancelled
```

`PD (Priority)` means the job is waiting in the scheduler queue. It is not automatically an error.

---

## 12. GPU assumptions

Do not assume GPU access.

Before using GPU scripts, the user should check:

```bash
my_share_info
```

Only write GPU instructions when explicitly asked or when GPU access has been confirmed.

GPU Slurm scripts may include:

```bash
#SBATCH --gpus=1
```

Python code should auto-detect device when using PyTorch:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

Do not hard-code `cuda` unless CPU execution should intentionally fail.

---

## 13. Dependencies

Use `requirements.txt` for Python dependencies.

For classical computer vision:

```bash
pip install numpy opencv-python-headless matplotlib tqdm scikit-image pandas
```

Use:

```bash
opencv-python-headless
```

rather than:

```bash
opencv-python
```

on Euler or other headless systems.

For PyTorch:

```bash
pip install torch torchvision torchaudio
```

After changing dependencies:

```bash
pip freeze > requirements.txt
```

---

## 14. Data and output conventions

Python scripts should accept input and output paths:

```bash
--input_dir
--output_dir
--frames_dir
--checkpoint_dir
--config
```

Outputs should be written into the provided output directory.

Typical output files:

```text
metrics.json
metrics.csv
matches.png
tracks.png
summary.txt
predictions.csv
checkpoint_last.pt
checkpoint_best.pt
```

Do not write important outputs only to the current working directory.

For branch-specific outputs on Euler, prefer:

```bash
RUN_DIR="$SCRATCH/cv-hackathon/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
```

or include the approach name:

```bash
RUN_DIR="$SCRATCH/cv-hackathon/outputs/orb-ransac-${SLURM_JOB_ID}"
```

---

## 15. Computer vision task guidance

For image comparison and feature tracking, prefer a strong classical CV baseline before deep learning.

Recommended baseline sequence:

1. Load already-extracted frames.
2. Detect keypoints with ORB, AKAZE or SIFT.
3. Extract descriptors.
4. Match descriptors.
5. Apply Lowe ratio test or cross-check matching.
6. Use RANSAC for outlier rejection.
7. Estimate homography, affine transform or essential matrix depending on task.
8. Save metrics and match visualizations.

Useful metrics:

```text
number of frames
number of keypoints
number of raw matches
number of good matches
number of RANSAC inliers
inlier ratio
reprojection error
runtime
```

For tracking:

1. Detect points in frame `t`.
2. Track with Lucas-Kanade optical flow to frame `t+1`.
3. Apply forward-backward consistency check.
4. Reject inconsistent tracks with RANSAC.
5. Save track visualizations and metrics.

---

## 16. Machine learning task guidance

Only add machine learning if the classical baseline is insufficient.

Possible ML approaches:

```text
pretrained feature matchers
Siamese networks for patch/image comparison
CNN feature extractors
optical-flow networks
hybrid classical + learned features
```

Do not start by training a large model from scratch unless:

- labelled data exists,
- compute resources are available,
- the expected gain justifies the complexity.

When developing ML approaches, keep each approach in its own branch.

Examples:

```bash
git checkout -b approach/siamese-cnn
git checkout -b approach/pretrained-lightglue
```

---

## 17. Configuration

Prefer config files for experiment settings:

```text
configs/default.yaml
configs/orb.yaml
configs/akaze.yaml
configs/lk_tracking.yaml
configs/train_small.yaml
```

But command-line arguments should still allow overriding important values.

Example command:

```bash
python src/classical_cv.py \
  --config configs/orb.yaml \
  --input_dir data/frames \
  --output_dir outputs/local_test
```

Euler equivalent:

```bash
python src/classical_cv.py \
  --config configs/orb.yaml \
  --input_dir "$SCRATCH/cv-hackathon/data/frames" \
  --output_dir "$SCRATCH/cv-hackathon/outputs/orb-${SLURM_JOB_ID}"
```

---

## 18. Testing

When adding or changing code, provide at least one minimal test command.

For local testing:

```bash
python src/classical_cv.py \
  --input_dir data/frames \
  --output_dir outputs/test
```

For Euler testing:

```bash
sbatch euler/test_cpu.sbatch
squeue -u $USER
cat $SCRATCH/cv-hackathon/logs/cv_test-*.out
```

When possible, create small test inputs so users can verify code quickly.

---

## 19. Git hygiene

Before committing:

```bash
git status
```

Commit code and scripts:

```bash
git add src euler configs requirements.txt README.md .gitignore
git commit -m "Describe change"
git push
```

Do not commit:

```text
venv/
.venv/
data/
outputs/
logs/
checkpoints/
*.out
*.err
large binary files
private keys
credentials
temporary artifacts
```

If a generated file appears in Git status, add it to `.gitignore` or delete it.

---

## 20. Working with branches

Each approach should be isolated in its own branch.

### Create new approach branch

```bash
git checkout main
git pull
git checkout -b approach/<approach-name>
```

### Commit changes

```bash
git status
git add .
git commit -m "Implement <approach-name>"
git push -u origin approach/<approach-name>
```

### Keep branch up to date

```bash
git checkout approach/<approach-name>
git fetch origin
git rebase origin/main
```

or, if merge commits are preferred:

```bash
git merge origin/main
```

### Compare approaches

Approaches should produce comparable outputs:

```text
metrics.csv
metrics.json
summary.txt
visualizations/
```

This allows results from different branches to be compared outside Git.

Do not commit branch-specific large outputs. Put them in:

```bash
$SCRATCH/cv-hackathon/outputs/<approach-name>-<job-id>
```

---

## 21. Response format when asked to write code

When asked to implement something, provide:

1. File path.
2. Full file content.
3. Required dependencies.
4. Local run command.
5. Euler Slurm run command if relevant.
6. How to inspect outputs.
7. Any assumptions or limitations.
8. Branch recommendation if it is a new approach.

Example structure:

```text
Recommended branch:
approach/classical-orb-ransac

File: src/classical_cv.py
<code>

Run locally:
<command>

Run on Euler:
<command>

Check output:
<command>

Assumptions:
<notes>
```

---

## 22. Safety and reliability

When generating scripts:

- Use `set -euo pipefail` in Bash scripts.
- Create output directories before writing.
- Validate input paths.
- Print useful diagnostic information.
- Do not silently ignore missing data.
- Do not overwrite important outputs unless explicitly intended.
- Use job-specific output directories for Slurm jobs.

For Slurm output directories, prefer:

```bash
RUN_DIR="$SCRATCH/cv-hackathon/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
```

For array jobs:

```bash
RUN_DIR="$SCRATCH/cv-hackathon/outputs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
```

---

## 23. Slurm array jobs

Use Slurm arrays for parameter sweeps and batch processing.

Example:

```bash
#SBATCH --array=0-15
```

Inside the script:

```bash
TASK_ID=$SLURM_ARRAY_TASK_ID
```

Use arrays for:

```text
multiple image pairs
multiple frame sequences
hyperparameter sweeps
detector/matcher comparisons
batch evaluation
```

Each array task should write to its own output directory.

---

## 24. Keep instructions concise but complete

When giving terminal commands, provide copy-pasteable blocks.

Avoid mixing file contents and terminal commands in a way that can be pasted into the wrong place.

For long file contents, prefer:

```text
Create file X in VS Code and paste this:
```

For terminal commands, prefer:

```text
Paste this into the terminal:
```

Do not include explanatory prose inside shell command blocks unless it is a comment beginning with `#`.

---

## 25. Current known working setup

The Euler setup has already been tested with:

```bash
sbatch euler/test_cpu.sbatch
```

The job completed successfully and produced output similar to:

```text
Euler Slurm test job works.
Python version: 3.12.8 ...
```

This confirms that:

```text
VS Code Remote SSH → Euler login node → Slurm → compute node → scratch logs
```

works.

Use this as the starting point for further development.

---

## 26. Overall objective

Help build a robust, portable, reproducible computer vision / ML project.

Priority order:

1. Keep the repository usable for everyone.
2. Keep Euler-specific details isolated.
3. Use provided frame extraction outputs instead of implementing frame extraction.
4. Build a strong classical CV baseline.
5. Keep each major approach in its own branch.
6. Add batch evaluation and metrics.
7. Add Slurm sweeps where helpful.
8. Add ML only if justified.
9. Keep outputs, logs, data and checkpoints out of GitHub.
