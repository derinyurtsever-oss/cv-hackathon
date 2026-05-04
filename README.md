# CV Hackathon — Image Comparison & Feature Tracking

## Setup

```bash
# Create and activate a virtual environment (uv recommended)
uv venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
uv pip install -r requirements.txt
```

Or with plain pip:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
cv-hackathon/
├── main.py                  # CLI entry point
├── requirements.txt
├── src/
│   ├── extract_frames.py    # Video → frames
│   ├── feature_matching.py  # ORB / SIFT / AKAZE + BF/FLANN + homography
│   ├── tracking.py          # Lucas-Kanade & Farneback optical flow
│   ├── comparison.py        # SSIM, MSE, PSNR, histogram, template match
│   └── utils.py             # I/O & visualisation helpers
├── data/                    # Put videos and frames here
└── output/                  # Visualisation outputs
```

---

## CLI Commands

### 1. Extract Frames from Video(s)

```bash
# Single video, 5 FPS
python main.py extract --input data/my_video.mp4 --output data/frames/my_video --fps 5

# Whole directory of videos, every 10th frame
python main.py extract --input data/videos/ --output data/frames/ --every-n 10
```

### 2. Feature Matching (ORB / SIFT / AKAZE)

```bash
python main.py match \
  --img1 data/frames/vid1/frame_000000.jpg \
  --img2 data/frames/vid1/frame_000100.jpg \
  --detector sift \
  --save output/match_vis.jpg
```

### 3. Optical Flow Tracking

```bash
# Sparse Lucas-Kanade (fast)
python main.py track --frames data/frames/vid1/ --method lk --save-dir output/flow/

# Dense Farneback
python main.py track --frames data/frames/vid1/ --method dense
```

### 4. Image Similarity Metrics

```bash
python main.py compare \
  --img1 data/frames/vid1/frame_000000.jpg \
  --img2 data/frames/vid2/frame_000000.jpg \
  --save-diff output/diff.jpg
```

Outputs: **SSIM**, **MSE**, **PSNR**, **histogram correlation**.

---

## Using Modules Directly

```python
from src.feature_matching import compare_images
from src.comparison import full_comparison
from src.tracking import track_video_frames, lucas_kanade_flow
from src.utils import load_image, show_images

img1 = load_image("data/frames/vid1/frame_000000.jpg")
img2 = load_image("data/frames/vid1/frame_000050.jpg")

# Feature matching
result = compare_images(img1, img2, method="sift")
print(f"Good matches: {len(result.good_matches)}, match ratio: {result.match_ratio:.3f}")

# Similarity
cmp = full_comparison(img1, img2)
print(cmp.summary())
```

---

## Key Algorithms

| Task | Algorithm | Module |
|------|-----------|--------|
| Feature detection | ORB, SIFT, AKAZE | `feature_matching.py` |
| Feature matching | BFMatcher + Lowe ratio | `feature_matching.py` |
| Geometric verification | RANSAC homography | `feature_matching.py` |
| Sparse tracking | Lucas-Kanade optical flow | `tracking.py` |
| Dense motion | Farneback optical flow | `tracking.py` |
| Background removal | MOG2 / KNN subtractor | `tracking.py` |
| Pixel similarity | SSIM, MSE, PSNR | `comparison.py` |
| Colour similarity | Histogram comparison | `comparison.py` |
| Template search | `cv2.matchTemplate` | `comparison.py` |
