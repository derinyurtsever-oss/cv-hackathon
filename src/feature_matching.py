"""
feature_matching.py
-------------------
Feature detection, description, and matching using classical algorithms:
  - ORB  (fast, no license restrictions)
  - SIFT (best quality)
  - AKAZE (good balance)
  - Brute-Force and FLANN matchers
  - Homography estimation with RANSAC
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    keypoints1: list
    keypoints2: list
    descriptors1: np.ndarray | None
    descriptors2: np.ndarray | None
    matches: list
    good_matches: list
    homography: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    match_ratio: float = 0.0


def get_detector(method: str = "orb", **kwargs):
    """
    Create a feature detector/descriptor.

    Parameters
    ----------
    method : str
        One of 'orb', 'sift', 'akaze'.
    """
    method = method.lower()
    if method == "orb":
        nfeatures = kwargs.get("nfeatures", 2000)
        return cv2.ORB_create(nFeatures=nfeatures)
    elif method == "sift":
        nfeatures = kwargs.get("nfeatures", 0)
        return cv2.SIFT_create(nfeatures=nfeatures)
    elif method == "akaze":
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown detector method: {method}. Choose 'orb', 'sift', or 'akaze'.")


def detect_and_compute(img: np.ndarray, detector, mask: np.ndarray | None = None):
    """Run detect + compute on a grayscale or BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return detector.detectAndCompute(gray, mask)


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str = "orb",
    lowe_ratio: float = 0.75,
    use_flann: bool = False,
) -> tuple[list, list]:
    """
    Match descriptors and apply Lowe's ratio test.

    Returns (all_matches, good_matches).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return [], []

    if use_flann:
        if method == "orb" or method == "akaze":
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        desc1_f = desc1.astype(np.float32) if desc1.dtype != np.float32 else desc1
        desc2_f = desc2.astype(np.float32) if desc2.dtype != np.float32 else desc2
        all_matches = matcher.knnMatch(desc1_f, desc2_f, k=2)
    else:
        norm = cv2.NORM_HAMMING if method in ("orb", "akaze") else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        all_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in all_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)

    return all_matches, good_matches


def estimate_homography(
    kp1: list,
    kp2: list,
    good_matches: list,
    ransac_threshold: float = 5.0,
    min_matches: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate homography from matched keypoints using RANSAC.

    Returns (H, mask) or (None, None) if not enough matches.
    """
    if len(good_matches) < min_matches:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    return H, mask


def compare_images(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "orb",
    lowe_ratio: float = 0.75,
    compute_homography: bool = True,
) -> MatchResult:
    """
    Full pipeline: detect -> describe -> match -> (optional) homography.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images (BGR or grayscale).
    method : str
        Feature detector: 'orb', 'sift', or 'akaze'.
    lowe_ratio : float
        Lowe's ratio test threshold.
    compute_homography : bool
        Whether to estimate homography from inliers.

    Returns
    -------
    MatchResult
    """
    detector = get_detector(method)
    kp1, desc1 = detect_and_compute(img1, detector)
    kp2, desc2 = detect_and_compute(img2, detector)

    all_matches, good_matches = match_features(desc1, desc2, method=method, lowe_ratio=lowe_ratio)

    match_ratio = len(good_matches) / max(len(kp1), 1)

    H, mask = None, None
    if compute_homography:
        H, mask = estimate_homography(kp1, kp2, good_matches)

    return MatchResult(
        keypoints1=kp1,
        keypoints2=kp2,
        descriptors1=desc1,
        descriptors2=desc2,
        matches=all_matches,
        good_matches=good_matches,
        homography=H,
        inlier_mask=mask,
        match_ratio=match_ratio,
    )


def batch_compare(
    frames: list[np.ndarray],
    method: str = "orb",
    consecutive_only: bool = True,
) -> list[MatchResult]:
    """
    Compare frames pairwise.

    If consecutive_only=True, compare frame[i] vs frame[i+1].
    Otherwise, compare every pair (O(n²)).
    """
    results = []
    if consecutive_only:
        for i in range(len(frames) - 1):
            results.append(compare_images(frames[i], frames[i + 1], method=method))
    else:
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                results.append(compare_images(frames[i], frames[j], method=method))
    return results
