"""
utils.py
--------
I/O helpers and visualisation utilities.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Load an image from disk. Returns BGR (or grayscale) numpy array."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def load_images_from_dir(directory: str, extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg")) -> list[tuple[str, np.ndarray]]:
    """Return (filename, image) pairs from a directory, sorted by name."""
    result = []
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() in extensions:
            result.append((p.name, cv2.imread(str(p))))
    return result


def save_image(img: np.ndarray, path: str) -> None:
    """Save an image (BGR numpy array) to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)


def show_images(images: list[np.ndarray], titles: list[str] | None = None, cols: int = 2, figsize_per: tuple[int, int] = (6, 4)) -> None:
    """Display a list of images using matplotlib (converts BGR -> RGB)."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows))
    axes = np.array(axes).flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def draw_matches_side_by_side(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    matches: list,
    max_matches: int = 50,
) -> np.ndarray:
    """Draw feature matches between two images."""
    top_matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    return cv2.drawMatches(
        img1, kp1, img2, kp2, top_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_optical_flow(frame: np.ndarray, prev_pts: np.ndarray, curr_pts: np.ndarray, status: np.ndarray | None = None) -> np.ndarray:
    """Draw optical flow vectors on a frame copy."""
    out = frame.copy()
    for i, (p0, p1) in enumerate(zip(prev_pts, curr_pts)):
        if status is not None and not status[i]:
            continue
        x0, y0 = map(int, p0.ravel())
        x1, y1 = map(int, p1.ravel())
        cv2.arrowedLine(out, (x0, y0), (x1, y1), (0, 255, 0), 1, tipLength=0.3)
        cv2.circle(out, (x1, y1), 3, (0, 0, 255), -1)
    return out


def resize_if_large(img: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    """Downscale image if its largest dimension exceeds max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))
