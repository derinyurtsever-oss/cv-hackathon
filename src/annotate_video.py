"""
annotate_video.py
-----------------
Produce an annotated output video that overlays the estimated distance
(and optionally a progress bar) on each frame of the inspection video.
"""

import cv2
import numpy as np
from pathlib import Path

from src.visual_odometry import OdometryResult, OdometryFrame


# -------------------------------------------------------------------
# Drawing helpers
# -------------------------------------------------------------------

def _draw_distance_overlay(
    frame: np.ndarray,
    distance_m: float,
    frame_idx: int,
    total_frames: int,
    n_tracked: int,
    show_progress: bool = True,
    show_tracker_count: bool = True,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent dark banner at top
    banner_h = 60
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    # Distance text
    dist_text = f"Distance: {distance_m:.2f} m"
    cv2.putText(out, dist_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2, cv2.LINE_AA)

    # Tracker count
    if show_tracker_count:
        track_text = f"Tracked pts: {n_tracked}"
        cv2.putText(out, track_text, (w - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar at bottom
    if show_progress and total_frames > 0:
        bar_h = 8
        bar_y = h - bar_h
        progress = frame_idx / max(total_frames - 1, 1)
        filled_w = int(w * progress)
        cv2.rectangle(out, (0, bar_y), (w, h), (40, 40, 40), -1)
        cv2.rectangle(out, (0, bar_y), (filled_w, h), (0, 200, 80), -1)

    return out


# -------------------------------------------------------------------
# Main annotation function
# -------------------------------------------------------------------

def annotate_video(
    video_path: str,
    odometry: OdometryResult,
    output_path: str,
    every_n_frames: int = 1,
    show_progress_bar: bool = True,
    show_tracker_count: bool = True,
) -> str:
    """
    Write an annotated video with distance overlay to output_path.

    Parameters
    ----------
    video_path : str
        Path to the original inspection video.
    odometry : OdometryResult
        Result from `visual_odometry.estimate_distance_from_video()`.
    output_path : str
        Where to save the annotated video (e.g. 'output/annotated.mp4').
    every_n_frames : int
        Must match the value used during odometry estimation.
    show_progress_bar : bool
        Draw a progress bar at the bottom of each frame.
    show_tracker_count : bool
        Show number of tracked feature points in the corner.

    Returns
    -------
    str
        Path to the saved output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps / every_n_frames, (width, height))

    # Build frame_idx → OdometryFrame lookup
    odo_map: dict[int, OdometryFrame] = {f.frame_idx: f for f in odometry.frames}
    total_odo_frames = len(odometry.frames) + 1

    raw_idx = 0
    processed_idx = 0
    current_distance = 0.0
    current_tracked = 0

    print(f"Writing annotated video to '{output_path}' ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_idx % every_n_frames == 0:
            odo_frame = odo_map.get(processed_idx + 1)
            if odo_frame:
                current_distance = odo_frame.cumulative_distance_m
                current_tracked = odo_frame.n_tracked_points

            annotated = _draw_distance_overlay(
                frame,
                distance_m=current_distance,
                frame_idx=processed_idx,
                total_frames=total_odo_frames,
                n_tracked=current_tracked,
                show_progress=show_progress_bar,
                show_tracker_count=show_tracker_count,
            )
            writer.write(annotated)
            processed_idx += 1

        raw_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Total distance: {odometry.total_distance_m:.3f} m  |  Frames written: {processed_idx}")
    return output_path


# -------------------------------------------------------------------
# Distance plot helper (saves a matplotlib figure)
# -------------------------------------------------------------------

def save_distance_plot(odometry: OdometryResult, output_path: str) -> str:
    """Save a matplotlib plot of cumulative distance vs frame index."""
    import matplotlib.pyplot as plt

    frames = [f.frame_idx for f in odometry.frames]
    distances = [f.cumulative_distance_m for f in odometry.frames]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, distances, color="royalblue", linewidth=1.5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Cumulative distance (m)")
    ax.set_title("Estimated camera distance along pipe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved distance plot to '{output_path}'")
    return output_path
