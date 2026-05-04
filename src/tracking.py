"""
tracking.py
-----------
Optical flow and object tracking utilities:
  - Lucas-Kanade sparse optical flow
  - Farneback dense optical flow
  - Background subtraction
  - Simple centroid tracker helper
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


# -------------------------------------------------------------------
# Sparse Optical Flow (Lucas-Kanade)
# -------------------------------------------------------------------

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

SHI_TOMASI_PARAMS = dict(
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7,
)


@dataclass
class FlowResult:
    prev_pts: np.ndarray
    curr_pts: np.ndarray
    status: np.ndarray
    errors: np.ndarray | None = None

    @property
    def good_prev(self) -> np.ndarray:
        return self.prev_pts[self.status.ravel() == 1]

    @property
    def good_curr(self) -> np.ndarray:
        return self.curr_pts[self.status.ravel() == 1]

    @property
    def motion_vectors(self) -> np.ndarray:
        return self.good_curr - self.good_prev

    @property
    def mean_displacement(self) -> float:
        vecs = self.motion_vectors
        if len(vecs) == 0:
            return 0.0
        return float(np.mean(np.linalg.norm(vecs, axis=1)))


def detect_corners(gray: np.ndarray, params: dict | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Detect Shi-Tomasi corners for seeding Lucas-Kanade tracking."""
    p = params or SHI_TOMASI_PARAMS
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **p)
    return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)


def lucas_kanade_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray | None = None,
    lk_params: dict | None = None,
) -> FlowResult:
    """
    Compute sparse LK optical flow.

    If prev_pts is None, Shi-Tomasi corners are detected automatically.
    """
    params = lk_params or LK_PARAMS
    if prev_pts is None or len(prev_pts) == 0:
        prev_pts = detect_corners(prev_gray)

    if len(prev_pts) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return FlowResult(empty, empty, np.empty((0, 1), dtype=np.uint8))

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **params)
    return FlowResult(prev_pts, curr_pts, status, err)


# -------------------------------------------------------------------
# Dense Optical Flow (Farneback)
# -------------------------------------------------------------------

@dataclass
class DenseFlowResult:
    flow: np.ndarray  # shape (H, W, 2)

    @property
    def magnitude(self) -> np.ndarray:
        mag, _ = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        return mag

    @property
    def angle(self) -> np.ndarray:
        _, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        return ang

    def to_hsv_vis(self) -> np.ndarray:
        """Encode flow as an HSV colour image for visualisation."""
        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        hsv = np.zeros((*self.flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 90 / np.pi
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def farneback_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> DenseFlowResult:
    """Compute dense Farneback optical flow between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0,
    )
    return DenseFlowResult(flow)


# -------------------------------------------------------------------
# Background Subtraction
# -------------------------------------------------------------------

def create_bg_subtractor(method: str = "mog2", history: int = 500, threshold: float = 16.0):
    """
    Create a background subtractor.

    method : 'mog2' | 'knn'
    """
    if method == "mog2":
        return cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold, detectShadows=True)
    elif method == "knn":
        return cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=threshold, detectShadows=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_bg_subtraction(subtractor, frame: np.ndarray, learning_rate: float = -1) -> np.ndarray:
    """Return foreground mask for a frame."""
    return subtractor.apply(frame, learningRate=learning_rate)


# -------------------------------------------------------------------
# Full tracking loop helper
# -------------------------------------------------------------------

def track_video_frames(
    frames: list[np.ndarray],
    method: str = "lk",
    redetect_interval: int = 30,
) -> list[FlowResult | DenseFlowResult]:
    """
    Run optical flow tracking across a list of BGR frames.

    method : 'lk' (sparse Lucas-Kanade) | 'dense' (Farneback)
    redetect_interval : for LK, re-detect corners every N frames.
    """
    results = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts: np.ndarray | None = None

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        if method == "lk":
            if prev_pts is None or i % redetect_interval == 0 or (results and len(results[-1].good_curr) < 20):
                prev_pts = detect_corners(prev_gray)
            result = lucas_kanade_flow(prev_gray, curr_gray, prev_pts)
            prev_pts = result.good_curr.reshape(-1, 1, 2) if len(result.good_curr) > 0 else None
        else:
            result = farneback_flow(prev_gray, curr_gray)

        results.append(result)
        prev_gray = curr_gray

    return results
