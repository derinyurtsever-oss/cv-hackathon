# Sewer Probe Distance Estimation — Documentation

## Overview

This project estimates how far a sewer-inspection probe has travelled inside a pipe channel, using only the video recorded by the probe's on-board camera.

The probe enters a pipe, travels to the far end, then returns to the start.  
The goal is to produce a per-frame distance curve (in metres) and identify the exact frame at which the probe reversed direction.
---

## File Structure

```
cv-hackathon/
├── mySolution/
│   └── MovementPathEstimator.py   # Core estimation logic (this is the submission)
├── compare_estimators.py          # Run all videos and score against ground truth
├── quick_check.py                 # Fast single-video sanity check with score
├── run_baseline.py                # Single-video run with path plot + frame preview
├── EstimationRater.py             # Scoring framework (provided by organisers)
├── MovementPath.py                # Result container (provided by organisers)
├── channel_lengths.npy            # Known physical channel lengths per video [m]
├── distance_labels/               # Ground-truth distance curves for scoring
└── frame_images/                  # Extracted PNG frames, one subfolder per video
```

---

## How to Run

**All videos (scored against ground truth):**
```bash
python compare_estimators.py
```

**Single video with path plot and frame preview:**
```bash
python run_baseline.py 4
```

**Single video, fast score only:**
```bash
python quick_check.py 4
```

Optical flow is cached to `mySolution/flow_cache/` after the first run.  
Subsequent runs on the same video load from cache and are much faster.

---

## How the Estimator Works

### 1. Initialization

`MovementPathEstimator.__init__` prepares:

| Attribute | Purpose |
|---|---|
| `channel_lengths` | Physical length of each channel loaded from `channel_lengths.npy` |
| `path_to_videos` | Root directory for extracted PNG frames (`frame_images/`) |
| `video_search_dirs` | Fallback directories for raw MP4 files (`data/`, `Videos/`, `videos/`) |
| `_dis` | OpenCV DIS dense optical-flow engine (FAST preset) |
| `_clahe` | CLAHE contrast enhancer for murky frames |
| `calculated_movement_paths` | Output dictionary, keyed by video number |

---

### 2. Main Estimation Flow

`calculate_movement_path_and_turning_point(video_number, channel_length)` runs in this order:

```
raw frames
    │
    ▼
_flow_signal_for_video()          ← per-frame signed radial flow + confidence
    │
    ▼
_apply_confidence_gating()        ← repair obvious bad spikes
    │
    ├──► _stabilize_flow_signal()  ← smooth + threshold  (for turn detection)
    │         │
    │         ▼
    │    _estimate_turning_point() ← find reversal frame
    │
    ├──► _stabilize_flow_signal()  ← smooth + threshold  (for path quality)
    │         │
    │         ▼
    │    _integrate_path()         ← build distance curve
    │         │
    │         ▼
    │    _enforce_unimodal()       ← enforce physically valid shape
    │
    └──► _path_to_direction()      ← derive +1 / -1 / 0 direction labels
```

---

### 3. Reading the Motion Signal

`_flow_signal_for_video` returns two arrays of length N (one entry per frame transition):

- **`flow`** — signed radial motion value per frame pair
- **`confidence`** — trustworthiness of each value (0–1)

**Cache behaviour:** checks `mySolution/flow_cache/annulus_v4_video_<n>.npz` first.  
If missing, reads from `frame_images/<n>/` (PNG) or falls back to the raw MP4.  
After computing, saves to the cache for all future runs.

---

### 4. Frame Preprocessing

`_prepare_gray_frame` applies three steps to every grayscale frame before flow:

1. **Resize** to 160 × 120 — keeps flow computation fast and consistent
2. **CLAHE** — stabilises contrast in dirty or dark frames
3. **Gaussian blur (3×3)** — reduces pixel noise

---

### 5. Pipe-Wall Weighting

`_radial_weight_map` builds a per-pixel weight mask shaped like a Gaussian annulus centred on the pipe cross-section.

- Weight is strongest at ~58% of the image radius (the pipe wall)
- The lower 28% of the frame is zeroed out (water / sludge region)
- Top-left and top-right corners are zeroed out (overlay artefacts)

It also returns **radial** `(rdx, rdy)` and **tangential** `(tdx, tdy)` unit vectors for every pixel.

---

### 6. Per-Frame Signed Flow (`_estimate_signed_radial_flow`)

Converts one frame pair into a single scalar + confidence:

| Step | What happens |
|---|---|
| Valid mask | Annulus weight AND both frames non-black (eroded) |
| Dense flow | DIS optical flow → per-pixel `(flow_x, flow_y)` |
| Drift removal | Subtract weighted-mean flow to cancel global camera translation |
| Projection | `radial = flow_x·rdx + flow_y·rdy` |
| Texture weight | min(Sobel prev, Sobel curr) × annulus × temporal stability |
| Filtering | Keep top-textured pixels; cap 92nd-percentile motion outliers |
| Trimming | Clip radial values to 10th–90th percentile |
| Collapse | Weighted average → `signed_flow` |
| Confidence | f(coverage, sign coherence, radial/tangential ratio, stability) |

---

### 7. Confidence Gating (`_apply_confidence_gating`)

Conservative spike repair only — does **not** globally reshape the signal.

Replaces a value only if it is **both**:
- below the 10th-percentile confidence threshold, **and**
- a large residual from a locally interpolated trend

All other values are left exactly as computed.

---

### 8. Turning-Point Detection

`_stabilize_flow_signal` smooths, median-centres, and thresholds the signal.  
`_estimate_turning_point` then scores every candidate turn frame:

```
score = (forward motion before turn)
      + (backward motion after turn)
      − 0.75 × (wrong-sign motion before turn)
      − 0.75 × (wrong-sign motion after turn)
      − 0.25 × balance_penalty × total_motion
```

The frame with the highest score is the estimated reversal point.

---

### 9. Path Reconstruction (`_integrate_path`)

1. Separate signal into positive (outbound) and negative (inbound) components
2. Apply gamma exponents to compress large spikes  
   (`gamma_outbound = 0.18`, `gamma_inbound = 0.002`)
3. Cumulative-sum both components independently
4. Scale outbound cumsum to `[0, channel_length]` up to the turn frame
5. Scale inbound cumsum from `channel_length` back to `0` after the turn
6. Apply moving-average smoothing
7. `_enforce_unimodal` forces strict monotone increase then decrease,  
   and pins `path[0] = 0`, `path[turn] = channel_length`, `path[-1] = 0`

---

### 10. Direction Labels (`_path_to_direction`)

Derived from the final smoothed path:

- `+1` where the path is rising (probe moving forward)
- `-1` where the path is falling (probe returning)
- `0` where the path is flat, and in a quiet zone around the turning point

---

## Outputs

`execute_estimations()` stores results in `self.calculated_movement_paths[video_number]`  
as a `MovementPath` object containing:

| Field | Type | Description |
|---|---|---|
| `movement_path` | `np.ndarray (N,)` | Position in `[0, channel_length]` metres per frame |
| `turning_point` | `float` | Frame index of the probe reversal |
| `movement_direction` | `np.ndarray (N,)` | `+1` / `-1` / `0` per frame |

---

## Key Design Choices

- **Radial projection** isolates in/out motion and suppresses pipe rotation noise
- **Annulus masking** focuses on the pipe wall where texture is richest
- **Texture weighting** makes the signal robust to murky water and featureless sections
- **Confidence gating** is intentionally conservative — only obvious spikes are repaired
- **Separate tuning** for turn detection vs. path integration (different smoothing windows and thresholds)
- **Unimodal enforcement** guarantees a physically valid output shape even when the raw integrated curve wiggles
- **Disk caching** of the flow signal makes repeated runs fast (only compute flow once per video)
