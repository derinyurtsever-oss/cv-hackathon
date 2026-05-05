from pathlib import Path

import cv2
import numpy as np

from .localization_model import feature_cache_dir, project_root


ANNULAR_VERSION = "annular-unwrap-fft-seams-v2"


def _frame_paths(video_number: int) -> list[Path]:
    video_dir = project_root() / "frame_images" / str(video_number)
    paths = [p for p in video_dir.glob("*.png") if p.stem.isdigit()]
    return sorted(paths, key=lambda p: int(p.stem))


def _read_annular_descriptor(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame image: {path}")

    h, w = image.shape[:2]
    crop = image[: int(h * 0.80), int(w * 0.04) : int(w * 0.96)]
    small = cv2.resize(crop, (192, 144), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray_f = gray.astype(np.float32) / 255.0

    hh, ww = gray_f.shape[:2]
    center = (ww * 0.5, hh * 0.52)
    max_radius = min(ww, hh) * 0.50
    unwrap = cv2.warpPolar(
        gray_f,
        (56, 192),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
    )

    gx = cv2.Sobel(unwrap, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(unwrap, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    radial_edge = np.abs(cv2.Sobel(unwrap, cv2.CV_32F, 0, 1, ksize=3))
    angular_edge = np.abs(cv2.Sobel(unwrap, cv2.CV_32F, 1, 0, ksize=3))

    r0 = int(unwrap.shape[1] * 0.22)
    r1 = int(unwrap.shape[1] * 0.88)
    unwrap = unwrap[:, r0:r1]
    grad = grad[:, r0:r1]
    radial_edge = radial_edge[:, r0:r1]
    angular_edge = angular_edge[:, r0:r1]

    unwrap = (unwrap - unwrap.mean(axis=1, keepdims=True)) / (unwrap.std(axis=1, keepdims=True) + 1e-6)
    grad = (grad - grad.mean(axis=1, keepdims=True)) / (grad.std(axis=1, keepdims=True) + 1e-6)
    radial_edge = radial_edge / (radial_edge.mean() + 1e-6)
    angular_edge = angular_edge / (angular_edge.mean() + 1e-6)

    fft_gray = np.abs(np.fft.rfft(unwrap, axis=0))[1:17]
    fft_grad = np.abs(np.fft.rfft(grad, axis=0))[1:17]
    radial_gray = unwrap.mean(axis=0)
    radial_grad = grad.mean(axis=0)
    seam_profile = radial_edge.mean(axis=0).astype(np.float32)
    seam_profile /= seam_profile.sum() + 1e-6
    angular_profile = angular_edge.mean(axis=1).astype(np.float32)
    angular_profile /= angular_profile.sum() + 1e-6

    band_stats = []
    for arr in (seam_profile, radial_gray.astype(np.float32), radial_grad.astype(np.float32)):
        for bins in (8, 12, 16):
            splits = np.array_split(arr, bins)
            band_stats.extend(float(chunk.mean()) for chunk in splits)
            band_stats.extend(float(chunk.max()) for chunk in splits)

    peak_order = np.argsort(seam_profile)[-6:]
    peak_idx = np.sort(peak_order).astype(np.float32) / max(1, len(seam_profile) - 1)
    peak_val = seam_profile[peak_order].astype(np.float32)

    desc = np.concatenate(
        [
            fft_gray.reshape(-1),
            fft_grad.reshape(-1),
            radial_gray.astype(np.float32),
            radial_grad.astype(np.float32),
            seam_profile,
            angular_profile,
            np.array(band_stats, dtype=np.float32),
            peak_idx,
            peak_val,
        ]
    ).astype(np.float32)
    desc /= np.linalg.norm(desc) + 1e-6
    return desc


def extract_annular_features(video_number: int, force: bool = False):
    paths = _frame_paths(video_number)
    if not paths:
        raise FileNotFoundError(f"No frame PNGs found for video {video_number}")
    frame_numbers = np.array([int(p.stem) for p in paths], dtype=np.int32)
    cache = feature_cache_dir() / f"video_{video_number}_{ANNULAR_VERSION}.npz"

    if cache.exists() and not force:
        data = np.load(cache, allow_pickle=False)
        cached_version = data["version"].item() if "version" in data.files else ""
        if cached_version == ANNULAR_VERSION and np.array_equal(data["frame_numbers"], frame_numbers):
            return frame_numbers, data["features"].astype(np.float32)

    features = []
    bad_frames = []
    for path in paths:
        try:
            features.append(_read_annular_descriptor(path))
        except FileNotFoundError:
            features.append(None)
            bad_frames.append(path.name)

    first_good = next((feat for feat in features if feat is not None), None)
    if first_good is None:
        raise FileNotFoundError(f"Could not read any frame images for video {video_number}")

    last_good = None
    for idx, feat in enumerate(features):
        if feat is None:
            features[idx] = last_good.copy() if last_good is not None else first_good.copy()
        else:
            last_good = feat

    if bad_frames:
        names = ", ".join(bad_frames[:5])
        suffix = "" if len(bad_frames) <= 5 else f", ... ({len(bad_frames)} total)"
        print(f"Warning: substituted unreadable annular frames in video {video_number}: {names}{suffix}")

    feature_array = np.vstack(features).astype(np.float32)
    np.savez_compressed(
        cache,
        version=np.array(ANNULAR_VERSION),
        frame_numbers=frame_numbers,
        features=feature_array,
    )
    return frame_numbers, feature_array
