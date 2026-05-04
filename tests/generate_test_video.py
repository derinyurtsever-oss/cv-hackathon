"""
generate_test_video.py
----------------------
Generates a synthetic pipe-inspection-like video for testing.

Simulates a camera moving forward through a textured tunnel:
- Background texture scrolls forward (increasing Y) to mimic forward motion
- Known per-frame displacement so we can verify odometry accuracy

Usage
-----
python tests/generate_test_video.py
# Writes: data/videos/synthetic_pipe.mp4
# Prints the ground-truth total distance in pixels (use with --gt-distance after calibration)
"""

import cv2
import numpy as np
from pathlib import Path

# --- Config ---
WIDTH, HEIGHT = 640, 480
FPS = 25
DURATION_SEC = 10
PIXELS_PER_FRAME = 8        # true forward displacement per frame (pixels)
OUTPUT_PATH = "data/videos/synthetic_pipe.mp4"

def make_pipe_texture(h: int, w: int, scale: float = 4.0) -> np.ndarray:
    """Generate a repeating rough stone/concrete texture for the pipe wall."""
    rng = np.random.default_rng(42)
    noise = rng.integers(40, 200, size=(h * 2, w), dtype=np.uint8)
    # Blur to make it look like a surface
    texture = cv2.GaussianBlur(noise, (7, 7), 0)
    # Add some horizontal streaks (pipe joints)
    for y in range(0, h * 2, 60):
        texture[y:y+4, :] = np.clip(texture[y:y+4, :].astype(int) - 60, 0, 255).astype(np.uint8)
    return texture


def draw_pipe_frame(texture: np.ndarray, offset: int, h: int, w: int) -> np.ndarray:
    """Crop a window from the scrolling texture and add a circular pipe mask."""
    # Wrap offset within texture height
    tex_h = texture.shape[0]
    offset = offset % tex_h
    # Tile if needed
    if offset + h > tex_h:
        strip = np.vstack([texture[offset:], texture[:h - (tex_h - offset)]])
    else:
        strip = texture[offset: offset + h]

    frame_gray = strip.copy()
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Darken edges to simulate circular pipe cross-section
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    radius = min(cx, cy) - 10
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    frame_bgr = (frame_bgr * mask[:, :, np.newaxis]).astype(np.uint8)

    # Add faint crosshair (camera centre marker)
    cv2.line(frame_bgr, (cx - 10, cy), (cx + 10, cy), (0, 80, 80), 1)
    cv2.line(frame_bgr, (cx, cy - 10), (cx, cy + 10), (0, 80, 80), 1)

    return frame_bgr


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    texture = make_pipe_texture(HEIGHT, WIDTH)

    total_frames = FPS * DURATION_SEC
    true_total_px = PIXELS_PER_FRAME * total_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(total_frames):
        offset = i * PIXELS_PER_FRAME
        frame = draw_pipe_frame(texture, offset, HEIGHT, WIDTH)
        writer.write(frame)

    writer.release()
    print(f"Saved synthetic video to '{OUTPUT_PATH}'")
    print(f"Frames     : {total_frames}")
    print(f"FPS        : {FPS}")
    print(f"Motion     : {PIXELS_PER_FRAME} px/frame  ({PIXELS_PER_FRAME * FPS} px/sec)")
    print(f"Total disp : {true_total_px} pixels  (use as --gt-distance after converting to metres)")
    print()
    print("Run the pipeline:")
    print(f"  python main.py pipeline --video {OUTPUT_PATH} --output output/annotated_synthetic.mp4 --gt-distance {true_total_px}")


if __name__ == "__main__":
    main()
