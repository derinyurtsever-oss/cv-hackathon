"""
visual_odometry.py
------------------
Estimate cumulative distance travelled by an inspection camera through a pipe.

Approach
--------
1. Detect Shi-Tomasi corners in each frame.
2. Track them with Lucas-Kanade optical flow to the next frame.
3. Compute the median displacement magnitude (pixels) between consecutive frames.
4. Multiply by a calibration scale (metres/pixel) to get per-frame distance.
5. Accumulate to produce a running distance estimate.

The calibration scale can be set manually or estimated automatically from
ground-truth annotations via `calibrate_scale()`.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field

from src.tracking import detect_corners, lucas_kanade_flow


@dataclass
class OdometryFrame:
    frame_idx: int
    pixel_displacement: float
    distance_delta_m: float
    cumulative_distance_m: float
    n_tracked_points: int


@dataclass
class OdometryResult:
    frames: list[OdometryFrame] = field(default_factory=list)
    scale_m_per_px: float = 1.0

    @property
    def total_distance_m(self) -> float:
        return self.frames[-1].cumulative_distance_m if self.frames else 0.0

    @property
    def cumulative_distances(self) -> list[float]:
        return [f.cumulative_distance_m for f in self.frames]

    @property
    def pixel_displacements(self) -> list[float]:
        return [f.pixel_displacement for f in self.frames]


def calibrate_scale(
    pixel_displacements: list[float],
    ground_truth_distance_m: float,
) -> float:
    """
    Estimate metres-per-pixel scale from total pixel displacement and known distance.

    Parameters
    ----------
    pixel_displacements : list[float]
        Per-frame pixel displacements (output of a first odometry pass with scale=1.0).
    ground_truth_distance_m : float
        Known total distance the camera travelled (from annotations).

    Returns
    -------
    float
        Scale factor in metres per pixel.
    """
    total_pixels = sum(pixel_displacements)
    if total_pixels == 0:
        raise ValueError("Total pixel displacement is zero — cannot calibrate scale.")
    return ground_truth_distance_m / total_pixels


def _frame_displacement(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray | None,
    min_points: int = 10,
    redetect_threshold: int = 20,
) -> tuple[float, np.ndarray | None]:
    """
    Compute median optical-flow displacement between two grayscale frames.

    Returns (median_displacement_pixels, updated_points_for_next_frame).
    """
    if prev_pts is None or len(prev_pts) < redetect_threshold:
        prev_pts = detect_corners(prev_gray)

    if len(prev_pts) == 0:
        return 0.0, None

    result = lucas_kanade_flow(prev_gray, curr_gray, prev_pts)
    good_prev = result.good_prev
    good_curr = result.good_curr

    if len(good_curr) < min_points:
        return 0.0, None

    displacements = np.linalg.norm(good_curr - good_prev, axis=1)
    median_disp = float(np.median(displacements))

    return median_disp, good_curr.reshape(-1, 1, 2)


def estimate_distance(
    frames: list[np.ndarray],
    scale_m_per_px: float = 1.0,
    redetect_interval: int = 30,
    min_points: int = 10,
) -> OdometryResult:
    """
    Estimate cumulative distance across a list of BGR frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        Consecutive BGR frames from the inspection video.
    scale_m_per_px : float
        Conversion factor. Use 1.0 first, then calibrate with `calibrate_scale()`.
    redetect_interval : int
        Force corner re-detection every N frames.
    min_points : int
        Minimum tracked points before re-detecting.

    Returns
    -------
    OdometryResult
    """
    result = OdometryResult(scale_m_per_px=scale_m_per_px)
    cumulative = 0.0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts: np.ndarray | None = None

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        force_redetect = (i % redetect_interval == 0)
        if force_redetect:
            prev_pts = None

        disp_px, prev_pts = _frame_displacement(prev_gray, curr_gray, prev_pts, min_points)
        dist_m = disp_px * scale_m_per_px
        cumulative += dist_m

        n_pts = len(prev_pts) if prev_pts is not None else 0
        result.frames.append(OdometryFrame(
            frame_idx=i,
            pixel_displacement=disp_px,
            distance_delta_m=dist_m,
            cumulative_distance_m=cumulative,
            n_tracked_points=n_pts,
        ))

        prev_gray = curr_gray

    return result


def estimate_distance_from_video(
    video_path: str,
    scale_m_per_px: float = 1.0,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    redetect_interval: int = 30,
) -> OdometryResult:
    """
    Convenience wrapper: run visual odometry directly on a video file.

    Parameters
    ----------
    video_path : str
        Path to the inspection video.
    scale_m_per_px : float
        Metres per pixel scale factor.
    every_n_frames : int
        Process every N-th frame (1 = all frames).
    max_frames : int, optional
        Stop after this many frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    print(f"Loaded {len(frames)} frames from '{video_path}'")
    return estimate_distance(frames, scale_m_per_px=scale_m_per_px, redetect_interval=redetect_interval)
