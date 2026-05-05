# Physical AI Hackathon — Relevant Summary

## 1. Context

The hackathon is a **Physical AI Hackathon** organized/presented by **Helbling**, with EESTEC LC Zurich and other student organizations visible in the event material.

Helbling frames **Physical AI** as AI systems that can autonomously make decisions and execute tasks in dynamic, physical environments. The relevant loop is:

```text
sense → think → act → physical world → sense
```

The challenge combines several disciplines:

- Computer vision
- Visual odometry / SLAM
- Data science
- Machine learning / AI
- Robotics / physical systems
- System integration
- Data acquisition

The industrial context is sewer/channel inspection in collaboration with **Enz Technik**. The customer need is:

- Clean
- Document
- Integrate & optimize

## 2. Challenge Goal

Build a method called:

```python
MovementPathEstimator
```

For each input video, the method must estimate:

```text
movement_path → position in meters per frame
turning_point → frame where the direction changes
```

The object/camera moves through a channel of known length. The task is to reconstruct its motion from visual input.

## 3. Data

The dataset consists of:

- **11 videos**
- Videos are provided as **frame images**
- The object moves through a channel of known length
- Some videos are labelled
- Some labels are hidden and used for scoring

A separate **frame extraction script/tool will be provided**.

## 4. Main Required Outputs

For every video, output:

### `movement_path`

Estimated travelled position for each frame.

Expected conceptual format:

```text
frame_0  → position_m_0
frame_1  → position_m_1
frame_2  → position_m_2
...
```

The path should be in meters per frame.

### `turning_point`

The frame index where the camera/object changes direction.

Example:

```python
turning_point = 438
```

## 5. Evaluation / Scoring

The solution is scored on the following criteria:

| Criterion | Description | Points |
|---|---:|---:|
| Accuracy | Calculated track closely matches the path labels | 0–40 |
| Informativeness | Clear, detailed, accurate documentation | 0–10 |
| Presentation | Clear presentation, e.g. slides, video, or document | 0–10 |
| Novelty | Creative, new, or surprising approach | 0–10 |
| Efficiency | Runs with low requirements and cost | 0–10 |
| Bonus: Accuracy | Additional detected features are accurate | 0–10 |
| Bonus: Novelty | Creative or innovative bonus ideas | 0–10 |
| **Total** | Maximum score | **100** |

Core scoring metrics mentioned:

- Turning point error
- Normalized position MAE

## 6. Bonus Tasks

Optional bonus objectives:

- Automatically classify sewer condition per frame
- Highlight further useful information

Possible classes / features:

- Gravel
- Lime deposit
- Sludge
- Sand
- Roots
- Grease
- Concrete
- No vision
- Lens covered
- Splash water
- Under water

## 7. Suggested Technical Approaches

The organizers explicitly mention the following possible methods:

- Classical computer vision techniques
- Visual odometry
- SLAM
- LLMs
- Multimodal AI
- Foundation models

Practical candidate approaches:

### A. Classical CV baseline

Use frame-to-frame image similarity or optical flow to estimate relative displacement.

Possible tools:

- OpenCV
- Feature matching: ORB / SIFT / AKAZE
- Optical flow: Lucas-Kanade or Farnebäck
- Homography / affine transform estimation
- Frame similarity / phase correlation

Potential pipeline:

```text
load frames
→ preprocess
→ extract visual features
→ match features frame-to-frame
→ estimate relative motion
→ integrate relative motion over time
→ normalize to known channel length
→ detect turning point from motion sign change
```

### B. Visual odometry approach

Estimate camera movement from visual changes across frames.

Useful when texture and visibility are sufficient.

Potential weakness:

- Sewer videos may have poor texture
- Water, dirt, blur, lens obstruction, or repeated pipe patterns can break feature matching

### C. Learned / foundation model approach

Use a pretrained vision model to extract frame embeddings, then estimate progression through the pipe by comparing temporal similarity.

Possible approach:

```text
frame → visual embedding
→ temporal ordering / trajectory regression
→ movement_path
→ turning point detection
```

This may help when classical features are unstable, but it must remain efficient and easy to run.

### D. Hybrid approach

Most robust likely approach:

```text
classical CV motion cues
+ frame-level quality / obstruction detection
+ smoothing / physical constraints
+ known channel length normalization
```

Physical constraints to exploit:

- Motion is one-dimensional along the pipe
- The travelled path is bounded by the known channel length
- Direction changes once at the turning point
- Motion should be smooth except for stops, acceleration changes, or the turn

## 8. Starting Point

The provided demo code includes a **dummy triangle-wave model**.

The task is to replace that dummy model with a real computer vision solution.

A reasonable baseline can still preserve the triangle-wave prior:

```text
movement starts at one end
→ moves forward through channel
→ reaches turning point
→ returns / changes direction
```

Then refine it using image evidence.

## 9. Working Setup

The organizers recommend **BYO infrastructure**:

- Laptop
- Cloud VM
- Google Colab
- Any cloud service

Kaggle is also supported.

## 10. Collaboration / Submission Modes

Two workflow options are mentioned.

### Option 1 — BYO infrastructure, recommended

Use:

- Your own laptop / VM / cloud
- GitHub for collaboration
- Swisstransfer or Kaggle for inputs

Submission:

- Invite `philipp.huber@helbling.ch` to the GitHub repository
- Ensure all resources are accessible
- Include `requirements.txt`
- Include a presentation summarizing the submission
- Code must run

### Option 2 — Kaggle

Use:

- Kaggle team workflow
- Kaggle inputs
- Kaggle notebook

Submission:

- Kaggle writeup
- Attached notebook

## 11. Deadline

Submission deadline:

```text
20:00
```

Office hours with André, Philipp, and Damian are available until 20:00.

## 12. Repository / Branching Plan

Different approaches should be stored in different Git branches.

Recommended branch layout:

```text
main
├── baseline-triangle
├── classical-cv-optical-flow
├── classical-cv-feature-matching
├── embedding-regression
├── hybrid-cv-smoothing
└── bonus-condition-classification
```

Recommended workflow:

```bash
git checkout -b classical-cv-optical-flow
# implement approach
git add .
git commit -m "Implement optical-flow movement estimator"
git push -u origin classical-cv-optical-flow
```

Keep `main` stable and only merge working approaches.

## 13. Recommended Repository Structure

```text
cv-hackathon/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── frames/
│   └── labels/
├── src/
│   ├── movement_path_estimator.py
│   ├── preprocessing.py
│   ├── visual_odometry.py
│   ├── smoothing.py
│   └── bonus_classification.py
├── notebooks/
├── outputs/
│   ├── predictions/
│   └── figures/
├── presentation/
└── scripts/
```

## 14. Euler / Local Training Notes

### Local development

Use local development for:

- Debugging
- Testing on a few videos
- Visualizing frames and predictions
- Fast iteration

Typical local commands:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python src/movement_path_estimator.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\movement_path_estimator.py
```

### Euler / cluster training

Use Euler for:

- Heavier experiments
- Batch processing many videos
- Training models
- Running expensive embedding extraction

General Euler workflow:

```bash
ssh <username>@euler.ethz.ch
cd ~/projects/cv-hackathon
module load stack
module load python
python --version
```

Submit jobs through Slurm. A previous test job confirmed that Slurm worked and produced output similar to:

```text
Euler Slurm test job works.
Python version: 3.12.8
```

Example Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=cv_hackathon
#SBATCH --output=$SCRATCH/cv-hackathon/logs/%x-%j.out
#SBATCH --error=$SCRATCH/cv-hackathon/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load stack
module load python

cd ~/projects/cv-hackathon

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/movement_path_estimator.py
```

Submit:

```bash
sbatch scripts/run_euler.sh
```

Check queue:

```bash
squeue -u $USER
```

Check logs:

```bash
cat $SCRATCH/cv-hackathon/logs/<job-name>-<job-id>.out
cat $SCRATCH/cv-hackathon/logs/<job-name>-<job-id>.err
```

## 15. Minimum Submission Checklist

Before submitting, ensure:

- `MovementPathEstimator` is implemented
- Predictions are generated for every required video
- `movement_path` is output in meters per frame
- `turning_point` is output as a frame number
- Code runs from a clean environment
- `requirements.txt` is complete
- README explains the method clearly
- Submission includes a short presentation / slides / document
- All resources are accessible
- GitHub repo access is granted to `philipp.huber@helbling.ch`
- Submission is completed before 20:00

## 16. Presentation Content

The final presentation should be short and practical.

Suggested structure:

1. Problem statement
2. Data overview
3. Chosen method
4. Why this method should work
5. Implementation details
6. Results / plots
7. Failure cases
8. Bonus features, if implemented
9. Runtime / efficiency
10. How to run the code

## 17. README Content

The README should include:

```text
Project title
Short challenge description
Method summary
Installation instructions
How to run
Expected input/output format
Repository structure
Known limitations
Bonus features
Team members
```

## 18. Codex / AI Coding Instructions

When using Codex or another coding assistant:

- Keep changes small and reviewable
- Work on one branch per approach
- Preserve the expected API / class name: `MovementPathEstimator`
- Do not change input/output contracts unless necessary
- Add comments only where they clarify non-obvious logic
- Prefer robust, simple code over clever but brittle code
- Add a quick test or sanity-check script for every approach
- Always update `requirements.txt` when adding dependencies
- Avoid hardcoded absolute paths
- Keep data out of git unless explicitly allowed
- Write outputs to `outputs/`

Suggested instruction prompt:

```text
Implement a MovementPathEstimator for the Physical AI Hackathon.

Requirements:
- Input consists of video frames from sewer/channel inspection videos.
- Output movement_path as position in meters per frame.
- Output turning_point as the frame number where direction changes.
- Preserve the expected class/API.
- Use robust computer vision methods.
- Include smoothing and normalization to known channel length.
- Keep dependencies minimal and update requirements.txt.
- Write predictions to outputs/predictions.
- Add clear README instructions.
```

## 19. Key Strategic Priorities

To score well:

1. **Accuracy first**  
   Optimize movement path reconstruction and turning point detection.

2. **Make the solution run reliably**  
   A clever method that fails at submission is worse than a simple robust baseline.

3. **Document clearly**  
   Informativeness is worth 10 points.

4. **Prepare a concise presentation**  
   Presentation is worth 10 points.

5. **Use physical priors**  
   The problem is constrained: one-dimensional movement, known channel length, one turning point.

6. **Add bonus only after core output works**  
   Bonus classification can add points, but only if the main path estimator is solid.
