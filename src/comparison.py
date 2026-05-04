"""
comparison.py
-------------
Image similarity and comparison metrics:
  - SSIM  (structural similarity)
  - MSE / PSNR
  - Histogram comparison (multiple methods)
  - Template matching
  - Difference image generation
"""

import cv2
import numpy as np
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


@dataclass
class ComparisonResult:
    ssim_score: float
    mse: float
    psnr_db: float
    hist_correlation: float
    hist_chi_square: float
    hist_intersection: float
    hist_bhattacharyya: float

    def summary(self) -> str:
        return (
            f"SSIM={self.ssim_score:.4f}  "
            f"MSE={self.mse:.2f}  "
            f"PSNR={self.psnr_db:.2f}dB  "
            f"Hist-corr={self.hist_correlation:.4f}"
        )


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error between two same-size images."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return float(np.mean((img1 - img2) ** 2))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).
    Automatically handles grayscale and colour images.
    """
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g1, g2 = img1, img2
    score, _ = ssim(g1, g2, full=True)
    return float(score)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio. Returns inf if images are identical."""
    return float(psnr(img1, img2, data_range=255))


def compute_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute normalised colour histogram (BGR channels concatenated)."""
    hists = []
    if len(img.shape) == 2:
        h = cv2.calcHist([img], [0], None, [bins], [0, 256])
        cv2.normalize(h, h)
        return h.flatten()
    for ch in range(img.shape[2]):
        h = cv2.calcHist([img], [ch], None, [bins], [0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    return np.concatenate(hists)


def compare_histograms(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> dict[str, float]:
    """Return all four OpenCV histogram comparison metrics."""
    if len(img1.shape) == 2:
        channels = [(img1, img2)]
    else:
        channels = list(zip(
            cv2.split(img1),
            cv2.split(img2),
        ))

    scores = {m: 0.0 for m in ("correlation", "chi_square", "intersection", "bhattacharyya")}
    method_map = {
        "correlation": cv2.HISTCMP_CORREL,
        "chi_square": cv2.HISTCMP_CHISQR,
        "intersection": cv2.HISTCMP_INTERSECT,
        "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    }
    for ch1, ch2 in channels:
        h1 = cv2.calcHist([ch1], [0], None, [bins], [0, 256])
        h2 = cv2.calcHist([ch2], [0], None, [bins], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        for name, flag in method_map.items():
            scores[name] += cv2.compareHist(h1, h2, flag)
    n = len(channels)
    return {k: v / n for k, v in scores.items()}


def difference_image(img1: np.ndarray, img2: np.ndarray, amplify: float = 1.0) -> np.ndarray:
    """Return absolute difference image, optionally amplified."""
    diff = cv2.absdiff(img1, img2)
    if amplify != 1.0:
        diff = np.clip(diff.astype(np.float32) * amplify, 0, 255).astype(np.uint8)
    return diff


def template_match(
    image: np.ndarray,
    template: np.ndarray,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """
    Find template in image.

    Returns (score, top_left, bottom_right).
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    result = cv2.matchTemplate(gray_img, gray_tpl, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    h, w = gray_tpl.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return max_val, top_left, bottom_right


def full_comparison(img1: np.ndarray, img2: np.ndarray) -> ComparisonResult:
    """
    Run all comparison metrics between two same-size images.

    Images are resized to match if they differ.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    hist_scores = compare_histograms(img1, img2)
    mse_val = compute_mse(img1, img2)

    try:
        psnr_val = compute_psnr(img1, img2)
    except Exception:
        psnr_val = float("inf")

    return ComparisonResult(
        ssim_score=compute_ssim(img1, img2),
        mse=mse_val,
        psnr_db=psnr_val,
        hist_correlation=hist_scores["correlation"],
        hist_chi_square=hist_scores["chi_square"],
        hist_intersection=hist_scores["intersection"],
        hist_bhattacharyya=hist_scores["bhattacharyya"],
    )
