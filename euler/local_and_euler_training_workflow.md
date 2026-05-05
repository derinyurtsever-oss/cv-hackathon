# Training Models Locally and on ETH Euler

This document explains two workflows:

1. **Local training** on your own computer.
2. **Euler training** on ETH Zürich’s Euler HPC cluster through VS Code Remote SSH and Slurm.

The goal is to keep the project portable: everyone can run the code locally, while Euler users can submit the same code to the cluster through `sbatch`.

---

## 1. Core idea

The Python code should be general and not depend on Euler-specific paths.

Good:

```bash
python src/train.py --data_dir data --output_dir outputs --epochs 10
```

Also good on Euler:

```bash
python src/train.py \
  --data_dir "$SCRATCH/cv-hackathon/data" \
  --output_dir "$SCRATCH/cv-hackathon/outputs/run-${SLURM_JOB_ID}" \
  --epochs 10
```

Bad:

```python
data_dir = "/cluster/scratch/my_username/cv-hackathon/data"
```

Do not hard-code usernames or cluster paths inside Python code. Use command-line arguments.

---

## 2. Recommended project structure

```text
cv-hackathon/
├── main.py
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── data.py
│   ├── models.py
│   ├── classical_cv.py
│   └── utils.py
├── euler/
│   ├── README_EULER.md
│   ├── setup_euler.sh
│   ├── test_cpu.sbatch
│   ├── train_cpu.sbatch
│   ├── train_gpu.sbatch
│   └── run_cv.sbatch
├── configs/
│   └── default.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

General code belongs in:

```text
main.py
src/
configs/
requirements.txt
```

Euler-specific helper files belong in:

```text
euler/
```

Do not put Euler-specific assumptions into `main.py` or the core `src/` code.

---

## 3. What should be committed to GitHub

Commit:

```text
main.py
src/
euler/
configs/
requirements.txt
README.md
.gitignore
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
large images
model weights
temporary test files
```

Typical `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# Local data and generated outputs
data/
outputs/
logs/
checkpoints/
*.out
*.err

# Test artifacts
euler_test_output.txt

# OS / editor
.DS_Store
.vscode/
```

---

# Part A — Local training

## 4. When to train locally

Train locally when:

- the dataset is small,
- the model is small,
- you are debugging,
- you are testing whether the code runs,
- you are doing classical CV on a few images,
- you do not need many CPU cores or GPUs.

Local training is faster to iterate because there is no Slurm queue.

---

## 5. Clone the repository locally

On your local computer:

```bash
git clone git@github.com:<owner>/<repo>.git
cd <repo>
```

Example:

```bash
git clone git@github.com:derinyurtsever-oss/cv-hackathon.git
cd cv-hackathon
```

If SSH is not set up locally, use the HTTPS clone URL from GitHub.

---

## 6. Create a local Python environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Local folder layout for data and outputs

Create:

```bash
mkdir -p data outputs logs checkpoints
```

On Windows PowerShell:

```powershell
mkdir data
mkdir outputs
mkdir logs
mkdir checkpoints
```

These folders should stay local and should not be committed to GitHub.

---

## 8. Run a local training job

Example:

```bash
python src/train.py \
  --data_dir data \
  --output_dir outputs/local_test \
  --epochs 5
```

On Windows PowerShell, multiline commands use the backtick:

```powershell
python src/train.py `
  --data_dir data `
  --output_dir outputs/local_test `
  --epochs 5
```

A good training script should print progress and write checkpoints/metrics to the output directory.

Example expected output:

```text
Training started
Data directory: data
Output directory: outputs/local_test
Epochs: 5
Epoch 1/5
Epoch 2/5
...
Training finished
```

---

## 9. Run local classical CV

Example:

```bash
python src/classical_cv.py \
  --img1 data/img1.png \
  --img2 data/img2.png \
  --out outputs/matches.png
```

Windows PowerShell:

```powershell
python src/classical_cv.py `
  --img1 data/img1.png `
  --img2 data/img2.png `
  --out outputs/matches.png
```

---

## 10. Local debugging rules

Use local runs for:

```text
syntax errors
argument parsing
small examples
checking output files
fast iteration
basic visualizations
```

Do not use local training for large experiments if your machine is too slow or lacks memory/GPU resources. Move those runs to Euler.

---

# Part B — Euler training

## 11. Euler mental model

Euler is not just a remote Linux computer for direct training.

The correct workflow is:

```text
VS Code Remote SSH
  → Euler login node
      → edit files, install environment, submit jobs
          → Slurm scheduler
              → compute node runs training
                  → logs and outputs written to $SCRATCH
```

Important:

```text
VS Code = editor
Euler login node = setup and job submission
Slurm compute node = actual training
```

Do not run heavy training directly on the Euler login node.

Bad for heavy training:

```bash
python src/train.py --data_dir ... --output_dir ...
```

Good:

```bash
sbatch euler/train_cpu.sbatch
```

or:

```bash
sbatch euler/train_gpu.sbatch
```

---

## 12. Connect VS Code to Euler

In VS Code:

```text
Ctrl + Shift + P
→ Remote-SSH: Connect to Host
→ euler.ethz.ch
```

Open the project folder:

```text
File → Open Folder
```

Path:

```bash
/cluster/home/$USER/projects/cv-hackathon
```

If VS Code asks for the remote platform, choose:

```text
Linux
```

---

## 13. Clone the repository on Euler

If the repository is not cloned yet:

```bash
cd ~/projects
git clone git@github.com:<owner>/<repo>.git
cd <repo>
```

Example:

```bash
cd ~/projects
git clone git@github.com:derinyurtsever-oss/cv-hackathon.git
cd cv-hackathon
```

---

## 14. Euler storage convention

Use:

```bash
/cluster/home/$USER/projects/cv-hackathon
```

for code.

Use:

```bash
$SCRATCH/cv-hackathon/data
$SCRATCH/cv-hackathon/outputs
$SCRATCH/cv-hackathon/logs
$SCRATCH/cv-hackathon/checkpoints
```

for data, logs, outputs, and checkpoints.

Create the scratch folders:

```bash
mkdir -p "$SCRATCH/cv-hackathon/data"
mkdir -p "$SCRATCH/cv-hackathon/outputs"
mkdir -p "$SCRATCH/cv-hackathon/logs"
mkdir -p "$SCRATCH/cv-hackathon/checkpoints"
```

---

## 15. Set up Python on Euler

From the repository root:

```bash
cd ~/projects/cv-hackathon
bash euler/setup_euler.sh
source venv/bin/activate
```

The setup script should do roughly this:

```bash
module load stack/2024-06
module load gcc
module load python

if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet, install manually and then freeze:

```bash
pip install numpy opencv-python-headless matplotlib tqdm scikit-image pandas
pip freeze > requirements.txt
```

For PyTorch:

```bash
pip install torch torchvision torchaudio
pip freeze > requirements.txt
```

---

## 16. Test Slurm on Euler

Submit the test job:

```bash
sbatch euler/test_cpu.sbatch
```

Check status:

```bash
squeue -u $USER
```

States:

```text
PD = pending / waiting in queue
R  = running
CG = completing
```

After the job disappears from `squeue`, read the logs:

```bash
cat $SCRATCH/cv-hackathon/logs/cv_test-*.out
cat $SCRATCH/cv-hackathon/logs/cv_test-*.err
```

A harmless `.err` message may look like:

```text
Many modules are hidden in this stack...
```

That is not a failure by itself.

A successful test output looks like:

```text
Euler Slurm test job works.
Python version: ...
```

---

## 17. CPU training on Euler

Create:

```text
euler/train_cpu.sbatch
```

Example:

```bash
#!/bin/bash
#SBATCH --job-name=train_cpu
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
mkdir -p "$SCRATCH/cv-hackathon/checkpoints"
mkdir -p "$SCRATCH/cv-hackathon/logs"

python src/train.py \
  --data_dir "$SCRATCH/cv-hackathon/data" \
  --output_dir "$RUN_DIR" \
  --epochs 10
```

Submit:

```bash
sbatch euler/train_cpu.sbatch
```

Monitor:

```bash
squeue -u $USER
```

Read logs:

```bash
cat $SCRATCH/cv-hackathon/logs/train_cpu-*.out
cat $SCRATCH/cv-hackathon/logs/train_cpu-*.err
```

---

## 18. GPU training on Euler

Only use this if your account has GPU access.

Check:

```bash
my_share_info
```

If you do not have GPU shareholder access, do not assume GPU jobs will work.

Create:

```text
euler/train_gpu.sbatch
```

Example:

```bash
#!/bin/bash
#SBATCH --job-name=train_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
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
mkdir -p "$SCRATCH/cv-hackathon/checkpoints"
mkdir -p "$SCRATCH/cv-hackathon/logs"

python src/train.py \
  --data_dir "$SCRATCH/cv-hackathon/data" \
  --output_dir "$RUN_DIR" \
  --epochs 50 \
  --batch_size 64
```

Submit:

```bash
sbatch euler/train_gpu.sbatch
```

---

## 19. Python code should auto-select device

Inside `src/train.py`, use:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

Move tensors and models to the device:

```python
model = model.to(device)

for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)
```

Do not hard-code `cuda` unless you intentionally want CPU runs to fail.

---

## 20. Upload data to Euler

From local Windows PowerShell, not from inside Euler:

```powershell
scp -r C:\path\to\data <eth-username>@euler.ethz.ch:/cluster/scratch/<eth-username>/cv-hackathon/data/
```

Example:

```powershell
scp -r C:\Users\<local-user>\Desktop\contest_images <eth-username>@euler.ethz.ch:/cluster/scratch/<eth-username>/cv-hackathon/data/
```

On Euler, verify:

```bash
ls -lh $SCRATCH/cv-hackathon/data
```

---

## 21. Download outputs from Euler

From local Windows PowerShell:

```powershell
scp -r <eth-username>@euler.ethz.ch:/cluster/scratch/<eth-username>/cv-hackathon/outputs C:\Users\<local-user>\Desktop\euler_outputs
```

Or a single file:

```powershell
scp <eth-username>@euler.ethz.ch:/cluster/scratch/<eth-username>/cv-hackathon/outputs/result.png C:\Users\<local-user>\Desktop\
```

---

## 22. Monitoring commands on Euler

Use these frequently:

```bash
# show your jobs
squeue -u $USER

# show estimated start time
squeue --start -u $USER

# show estimated start for one job
squeue --start -j <job_id>

# show detailed job information
scontrol show job <job_id>

# cancel job
scancel <job_id>

# list logs
ls -lh $SCRATCH/cv-hackathon/logs

# read logs
cat $SCRATCH/cv-hackathon/logs/*.out
cat $SCRATCH/cv-hackathon/logs/*.err

# follow a running log
tail -f $SCRATCH/cv-hackathon/logs/<log-file>.out

# inspect outputs
ls -lh $SCRATCH/cv-hackathon/outputs

# inspect scratch data
ls -lh $SCRATCH/cv-hackathon/data

# check share/GPU access
my_share_info

# check storage quota
lquota
```

---

## 23. What `PD (Priority)` means

If `squeue` shows:

```text
ST = PD
NODELIST(REASON) = (Priority)
```

that usually means:

```text
The job was accepted.
It is waiting in the scheduler queue.
Nothing is necessarily broken.
```

Wait, or request fewer resources for small tests:

```bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
```

---

## 24. What `CG` means

If `squeue` shows:

```text
ST = CG
```

that means:

```text
Completing
```

The job has run and is finishing cleanup/log writing. Wait a few seconds, then check logs.

---

# Part C — Code design for both local and Euler training

## 25. Use command-line arguments

`src/train.py` should accept paths and hyperparameters:

```python
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # training logic here


if __name__ == "__main__":
    main()
```

This allows local and Euler execution with the same code.

---

## 26. Save metrics and checkpoints

A real training script should save:

```text
metrics.json
metrics.csv
checkpoint_last.pt
checkpoint_best.pt
config_used.yaml or args.json
```

Example:

```python
import json
from pathlib import Path

metrics = {
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
}

Path(output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
```

For PyTorch checkpoints:

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    },
    output_dir / "checkpoint_last.pt",
)
```

---

## 27. Classical CV baseline should also use arguments

Example:

```bash
python src/classical_cv.py \
  --input_dir data \
  --output_dir outputs/cv_test \
  --method orb \
  --nfeatures 3000 \
  --ratio 0.75 \
  --ransac_threshold 5.0
```

On Euler:

```bash
python src/classical_cv.py \
  --input_dir "$SCRATCH/cv-hackathon/data" \
  --output_dir "$SCRATCH/cv-hackathon/outputs/cv-${SLURM_JOB_ID}" \
  --method orb \
  --nfeatures 3000 \
  --ratio 0.75 \
  --ransac_threshold 5.0
```

---

## 28. Keep Euler files thin

Euler scripts should mostly do:

```text
load modules
activate venv
create output folders
call Python with correct paths
```

They should not contain the main algorithm.

Good:

```bash
python src/train.py --data_dir ... --output_dir ...
```

Bad:

```bash
# hundreds of lines of model logic inside .sbatch
```

---

# Part D — Git workflow

## 29. Local Git workflow

```bash
git status
git pull
# edit files
git add .
git commit -m "Describe change"
git push
```

---

## 30. Euler Git workflow

Inside VS Code connected to Euler:

```bash
cd ~/projects/cv-hackathon
git status
git pull
# edit files
git add .
git commit -m "Describe change"
git push
```

If other teammates are also pushing:

```bash
git pull --rebase
```

Then resolve any conflicts and push.

---

## 31. What to avoid in Git

Do not commit:

```text
training outputs
logs
datasets
large generated files
venv
temporary output files
personal credentials
private keys
```

---

# Part E — Recommended development workflow

## 32. Practical workflow

1. Develop locally or in VS Code Remote SSH.
2. Test the script on a tiny example locally.
3. Test the same script on Euler with a tiny Slurm job.
4. Increase resources and dataset size.
5. Save outputs to `$SCRATCH`.
6. Commit only code and scripts to GitHub.
7. Keep data and model outputs out of GitHub.

---

## 33. Suggested daily routine

### Local

```bash
git pull
source .venv/bin/activate
python src/train.py --data_dir data --output_dir outputs/local_test --epochs 1
git add .
git commit -m "Improve training script"
git push
```

### Euler

```bash
git pull
source venv/bin/activate
sbatch euler/train_cpu.sbatch
squeue -u $USER
cat $SCRATCH/cv-hackathon/logs/train_cpu-*.out
```

---

## 34. Minimal checklist before submitting an Euler job

Before running:

```bash
sbatch euler/train_cpu.sbatch
```

check:

```bash
pwd
ls
source venv/bin/activate
python --version
ls -lh $SCRATCH/cv-hackathon/data
```

Make sure:

```text
you are in the correct repo
venv exists
requirements are installed
data exists in scratch
Slurm script uses correct paths
output/log directories exist or are created by the script
```

---

## 35. Summary

Use local training for debugging and small experiments.

Use Euler training for:

```text
larger datasets
longer training
parallel parameter sweeps
CPU-heavy preprocessing
GPU training, if access exists
batch evaluation
```

Keep the code portable. Pass all paths and hyperparameters through command-line arguments. Keep Euler-specific setup inside the `euler/` folder. Use Slurm for heavy work. Use GitHub for code, not data or generated outputs.

The central commands are:

Local:

```bash
python src/train.py --data_dir data --output_dir outputs/local_test --epochs 5
```

Euler:

```bash
sbatch euler/train_cpu.sbatch
squeue -u $USER
cat $SCRATCH/cv-hackathon/logs/train_cpu-*.out
```
