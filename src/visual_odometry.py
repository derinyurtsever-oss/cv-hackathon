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
from scipy.ndimage import median_filter

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


def _make_wall_mask(h: int, w: int, bottom_fraction: float) -> np.ndarray:
    """
    Create a mask that zeros out the bottom `bottom_fraction` of the frame.
    Keeps only the pipe wall region (top and sides) for feature tracking,
    excluding water/sediment in the bottom of the pipe.
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cutoff = int(h * (1.0 - bottom_fraction))
    mask[cutoff:, :] = 0
    return mask


def _frame_displacement(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray | None,
    min_points: int = 10,
    redetect_threshold: int = 20,
    wall_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray | None]:
    """
    Estimate forward displacement between two frames by fitting an affine
    transform and extracting only the translation magnitude.
    wall_mask excludes water/sediment in the bottom of the pipe.
    """
    if prev_pts is None or len(prev_pts) < redetect_threshold:
        prev_pts = detect_corners(prev_gray, mask=wall_mask)

    if len(prev_pts) == 0:
        return 0.0, None

    result = lucas_kanade_flow(prev_gray, curr_gray, prev_pts)
    good_prev = result.good_prev
    good_curr = result.good_curr

    if len(good_curr) < min_points:
        return 0.0, None

    # Fit a partial affine transform (rotation + scale + translation)
    # RANSAC rejects outliers caused by independently moving objects or noise
    transform, inliers = cv2.estimateAffinePartial2D(
        good_prev.reshape(-1, 1, 2),
        good_curr.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if transform is None:
        return 0.0, good_curr.reshape(-1, 1, 2)

    # ------------------------------------------------------------------
    # Radial flow: project each tracked point's motion vector onto the
    # outward-radial direction from the image centre.
    # Forward camera motion = flow vectors point OUTWARD (expanding field).
    # Lateral/rotational motion = tangential → radial projection ≈ 0.
    # This is robust to varying pipe width and camera acceleration.
    # ------------------------------------------------------------------
    h, w = prev_gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    radial_components = []
    pts0 = good_prev.reshape(-1, 2)
    pts1 = good_curr.reshape(-1, 2)
    for p0, p1 in zip(pts0, pts1):
        x0, y0 = p0
        x1, y1 = p1
        # Vector from image centre to point
        rx, ry = x0 - cx, y0 - cy
        r = np.sqrt(rx ** 2 + ry ** 2)
        if r < 5:          # too close to centre — skip
            continue
        # Unit outward radial direction
        nx, ny = rx / r, ry / r
        # Flow vector
        dx, dy = x1 - x0, y1 - y0
        # Radial component (positive = outward = forward motion)
        radial = dx * nx + dy * ny
        radial_components.append(radial)

    if not radial_components:
        return 0.0, good_curr.reshape(-1, 1, 2)

    # Median of positive radial components (forward motion only)
    radial_arr = np.array(radial_components)
    forward_px = float(np.median(radial_arr[radial_arr > 0])) if np.any(radial_arr > 0) else 0.0

    return forward_px, good_curr.reshape(-1, 1, 2)


def _find_steady_start(
    displacements: list[float],
    window: int = 15,
    threshold_multiplier: float = 1.5,
    min_steady_frames: int = 20,
) -> int:
    """
    Find the first frame index where the camera has settled into
    steady inspection speed, ignoring the fast initial insertion phase.

    Returns the index into `displacements` from which to start accumulating.
    Returns 0 if no fast insertion is detected.
    """
    arr = np.array(displacements, dtype=np.float64)
    overall_median = float(np.median(arr))
    if overall_median == 0:
        return 0

    # Rolling median
    rolling = median_filter(arr, size=window, mode="nearest")

    # Find first run of `min_steady_frames` consecutive frames all below threshold
    threshold = overall_median * threshold_multiplier
    below = rolling < threshold
    count = 0
    for i, b in enumerate(below):
        if b:
            count += 1
            if count >= min_steady_frames:
                return max(0, i - min_steady_frames + 1)
        else:
            count = 0
    return 0


def _smooth_displacements(
    displacements: list[float],
    window: int = 5,
    max_multiplier: float = 3.0,
) -> list[float]:
    """
    Clean raw per-frame displacements before accumulation:
    1. Cap outlier spikes > max_multiplier * rolling median
    2. Apply a median filter to smooth remaining noise
    """
    arr = np.array(displacements, dtype=np.float64)

    # Cap spikes: any value > max_multiplier * local median is clipped
    smoothed = median_filter(arr, size=window, mode="nearest")
    cap = smoothed * max_multiplier
    arr = np.minimum(arr, cap)

    # Final smoothing pass
    arr = median_filter(arr, size=window, mode="nearest")
    return arr.tolist()


def estimate_distance(
    frames: list[np.ndarray],
    scale_m_per_px: float = 1.0,
    redetect_interval: int = 30,
    min_points: int = 10,
    water_mask_fraction: float = 0.0,
    smooth_window: int = 5,
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
    water_mask_fraction : float
        Fraction of frame height to mask from the bottom (0.0 = no mask,
        0.3 = bottom 30% excluded). Use when water flows in the bottom
        of the pipe and corrupts feature tracking.
    smooth_window : int
        Window size for median smoothing of raw displacements (odd number).
        Larger values = smoother but less responsive to real speed changes.

    Returns
    -------
    OdometryResult
    """
    h_frame, w_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).shape[:2]
    wall_mask = _make_wall_mask(h_frame, w_frame, water_mask_fraction) if water_mask_fraction > 0 else None

    # --- Pass 1: collect raw per-frame displacements ---
    raw_disps: list[float] = []
    tracked_pts: list[int] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts: np.ndarray | None = None

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        if i % redetect_interval == 0:
            prev_pts = None
        disp_px, prev_pts = _frame_displacement(prev_gray, curr_gray, prev_pts, min_points, wall_mask=wall_mask)
        raw_disps.append(disp_px)
        tracked_pts.append(len(prev_pts) if prev_pts is not None else 0)
        prev_gray = curr_gray

    # --- Pass 2: smooth & cap outlier spikes ---
    smooth_disps = _smooth_displacements(raw_disps, window=smooth_window)

    result = OdometryResult(scale_m_per_px=scale_m_per_px)
    cumulative = 0.0
    for i, (disp_px, n_pts) in enumerate(zip(smooth_disps, tracked_pts)):
        dist_m = disp_px * scale_m_per_px
        cumulative += dist_m
        result.frames.append(OdometryFrame(
            frame_idx=i + 1,
            pixel_displacement=disp_px,
            distance_delta_m=dist_m,
            cumulative_distance_m=cumulative,
            n_tracked_points=n_pts,
        ))

    return result


def estimate_distance_from_video(
    video_path: str,
    scale_m_per_px: float = 1.0,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    redetect_interval: int = 30,
    water_mask_fraction: float = 0.0,
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
    water_mask_fraction : float
        Fraction of frame height to ignore from the bottom (e.g. 0.3 = bottom 30%).
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
    return estimate_distance(
        frames,
        scale_m_per_px=scale_m_per_px,
        redetect_interval=redetect_interval,
        water_mask_fraction=water_mask_fraction,
    )
