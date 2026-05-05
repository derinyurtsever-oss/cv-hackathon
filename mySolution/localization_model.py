import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression


MODEL_VERSION = "pipe-localizer-v1"
FEATURE_VERSION = "crop80-hsv-grid-thumb-v1"
LABEL_VIDEOS = (1, 2, 3, 4, 8, 9, 10, 11)

# The labels were supplied for the original extraction window.  We use the
# original frame numbers that remain in frame_images/{video}/*.png to align
# labels after trimming irrelevant leading footage.
ORIGINAL_FIRST_KEPT_FRAME = {
    1: 180,
    2: 56,
    3: 58,
    4: 30,
    5: 55,
    6: 58,
    7: 330,
    8: 100,
    9: 35,
    10: 42,
    11: 314,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    path = project_root() / "mySolution" / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path() -> Path:
    return results_dir() / "localization_model.joblib"


def feature_cache_dir() -> Path:
    path = results_dir() / "feature_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _frame_paths(video_number: int):
    video_dir = project_root() / "frame_images" / str(video_number)
    paths = [p for p in video_dir.glob("*.png") if p.stem.isdigit()]
    return sorted(paths, key=lambda p: int(p.stem))


def _read_image_features(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame image: {path}")

    h, w = image.shape[:2]
    # Ignore the lower water-flow band and a thin side margin.  Lens dirt is
    # usually soft/out-of-focus, so the descriptor leans on coarse appearance,
    # contrast, and structure rather than fragile keypoints.
    crop = image[: int(h * 0.80), int(w * 0.04) : int(w * 0.96)]
    small = cv2.resize(crop, (96, 72), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray_f = gray.astype(np.float32) / 255.0

    thumb = cv2.resize(gray_f, (32, 20), interpolation=cv2.INTER_AREA).reshape(-1)
    thumb_z = (thumb - thumb.mean()) / (thumb.std() + 1e-6)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist_parts = []
    for channel, bins, max_value in ((0, 18, 180), (1, 8, 256), (2, 8, 256)):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, max_value]).reshape(-1)
        hist = hist / (hist.sum() + 1e-6)
        hist_parts.append(hist.astype(np.float32))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0

    grid_features = []
    for y0 in np.linspace(0, gray_f.shape[0], 5, dtype=int)[:-1]:
        y1 = y0 + gray_f.shape[0] // 4
        for x0 in np.linspace(0, gray_f.shape[1], 5, dtype=int)[:-1]:
            x1 = x0 + gray_f.shape[1] // 4
            cell = gray_f[y0:y1, x0:x1]
            cell_mag = mag[y0:y1, x0:x1]
            cell_edges = edges[y0:y1, x0:x1]
            grid_features.extend(
                [
                    float(cell.mean()),
                    float(cell.std()),
                    float(np.percentile(cell, 10)),
                    float(np.percentile(cell, 90)),
                    float(cell_mag.mean()),
                    float(cell_edges.mean()),
                ]
            )

    global_features = np.array(
        [
            float(gray_f.mean()),
            float(gray_f.std()),
            float(mag.mean()),
            float(mag.std()),
            float(lap.var()),
            float(edges.mean()),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            thumb.astype(np.float32),
            thumb_z.astype(np.float32),
            *hist_parts,
            np.array(grid_features, dtype=np.float32),
            global_features,
        ]
    )


def extract_video_features(video_number: int, force: bool = False):
    paths = _frame_paths(video_number)
    if not paths:
        raise FileNotFoundError(f"No frame PNGs found for video {video_number}")
    frame_numbers = np.array([int(p.stem) for p in paths], dtype=np.int32)
    cache = feature_cache_dir() / f"video_{video_number}_{FEATURE_VERSION}.npz"

    if cache.exists() and not force:
        data = np.load(cache, allow_pickle=False)
        cached_version = data["version"].item() if "version" in data.files else ""
        if (
            cached_version == FEATURE_VERSION
            and np.array_equal(data["frame_numbers"], frame_numbers)
        ):
            return frame_numbers, data["features"].astype(np.float32)

    features_list = []
    bad_frames = []
    for path in paths:
        try:
            features_list.append(_read_image_features(path))
        except FileNotFoundError:
            features_list.append(None)
            bad_frames.append(path)

    first_good = next((feat for feat in features_list if feat is not None), None)
    if first_good is None:
        raise FileNotFoundError(f"Could not read any frame images for video {video_number}")

    last_good = None
    for idx, feat in enumerate(features_list):
        if feat is None:
            features_list[idx] = last_good.copy() if last_good is not None else first_good.copy()
        else:
            last_good = feat

    if bad_frames:
        names = ", ".join(path.name for path in bad_frames[:5])
        suffix = "" if len(bad_frames) <= 5 else f", ... ({len(bad_frames)} total)"
        print(f"Warning: substituted unreadable frames in video {video_number}: {names}{suffix}")

    features = np.vstack(features_list).astype(np.float32)
    np.savez_compressed(
        cache,
        version=np.array(FEATURE_VERSION),
        frame_numbers=frame_numbers,
        features=features,
    )
    return frame_numbers, features


def aligned_labels(video_number: int, frame_numbers: np.ndarray) -> np.ndarray | None:
    label_path = project_root() / "distance_labels" / f"{video_number}.npy"
    if not label_path.exists():
        return None

    labels = np.load(label_path).astype(np.float32)
    if labels.size == 0:
        return None

    old_first = ORIGINAL_FIRST_KEPT_FRAME.get(video_number, int(frame_numbers[0]))
    old_last = int(frame_numbers[-1])
    denom = max(1, old_last - old_first)
    label_x = (frame_numbers.astype(np.float32) - old_first) / denom
    label_x = np.clip(label_x, 0.0, 1.0) * (len(labels) - 1)
    return np.interp(label_x, np.arange(len(labels)), labels).astype(np.float32)


def _augment_features(features: np.ndarray, frame_numbers: np.ndarray, channel_length: float):
    n = max(1, len(frame_numbers))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    frame_span = max(1, int(frame_numbers[-1]) - int(frame_numbers[0]))
    f = (frame_numbers.astype(np.float32) - float(frame_numbers[0])) / float(frame_span)
    triangle_prior = 1.0 - np.abs(2.0 * t - 1.0)
    context = np.column_stack(
        [
            t,
            t * t,
            np.sqrt(np.clip(t, 0.0, 1.0)),
            1.0 - t,
            triangle_prior,
            f,
            np.full(n, float(channel_length) / 100.0, dtype=np.float32),
            np.full(n, float(n) / 5000.0, dtype=np.float32),
        ]
    ).astype(np.float32)
    return np.hstack([features.astype(np.float32), context])


def _available_label_videos():
    return [
        video
        for video in LABEL_VIDEOS
        if (project_root() / "distance_labels" / f"{video}.npy").exists()
        and (project_root() / "frame_images" / str(video)).exists()
    ]


def train_model(exclude_video: int | None = None, stride: int = 3, save: bool = True):
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    x_parts = []
    y_parts = []
    trained_videos = []

    for video in _available_label_videos():
        if exclude_video is not None and video == exclude_video:
            continue
        frame_numbers, features = extract_video_features(video)
        labels = aligned_labels(video, frame_numbers)
        if labels is None:
            continue

        idx = np.arange(0, len(frame_numbers), max(1, stride), dtype=np.int32)
        channel_length = float(channel_lengths[video - 1])
        x_parts.append(_augment_features(features[idx], frame_numbers[idx], channel_length))
        y_parts.append(np.clip(labels[idx] / channel_length, 0.0, 1.0))
        trained_videos.append(video)

    if not x_parts:
        raise RuntimeError("No labeled videos available for training.")

    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)

    model = ExtraTreesRegressor(
        n_estimators=220,
        max_features=0.35,
        min_samples_leaf=2,
        bootstrap=False,
        n_jobs=-1,
        random_state=17,
    )
    model.fit(x_train, y_train)
    bundle = {
        "version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "model": model,
        "trained_videos": trained_videos,
        "stride": stride,
    }

    if save:
        joblib.dump(bundle, model_path())
    return bundle


def load_or_train_model():
    path = model_path()
    if path.exists():
        bundle = joblib.load(path)
        if (
            bundle.get("version") == MODEL_VERSION
            and bundle.get("feature_version") == FEATURE_VERSION
        ):
            return bundle
    return train_model(save=True)


def _physical_postprocess(raw_norm: np.ndarray, channel_length: float):
    raw = np.nan_to_num(raw_norm.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    raw = np.clip(raw, 0.0, 1.0)
    n = len(raw)
    if n <= 3:
        path = np.clip(raw, 0.0, 1.0) * channel_length
        return path.astype(np.float32), float(np.argmax(path))

    sigma = max(2.0, n / 350.0)
    smooth = gaussian_filter1d(raw, sigma=sigma)
    lo = min(n - 1, max(1, int(0.10 * n)))
    hi = min(n, max(lo + 1, int(0.95 * n)))
    turn = int(lo + np.argmax(smooth[lo:hi]))

    x = np.arange(n, dtype=np.float32)
    left = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0).fit_transform(
        x[: turn + 1], smooth[: turn + 1]
    )
    right = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0).fit_transform(
        x[turn:], smooth[turn:]
    )
    shaped = np.concatenate([left[:-1], right])
    peak = float(np.max(shaped))
    if peak > 1e-4:
        shaped = shaped / peak
    shaped = gaussian_filter1d(shaped, sigma=max(1.0, n / 900.0))
    shaped = np.clip(shaped, 0.0, 1.0)

    path = shaped * float(channel_length)
    turn = int(np.argmax(path))
    return path.astype(np.float32), float(turn)


def predict_video_path(video_number: int, channel_length: float, bundle=None):
    if bundle is None:
        bundle = load_or_train_model()
    frame_numbers, features = extract_video_features(video_number)
    x = _augment_features(features, frame_numbers, float(channel_length))
    raw_norm = bundle["model"].predict(x)
    movement_path, turning_point = _physical_postprocess(raw_norm, float(channel_length))

    diffs = np.diff(movement_path, append=movement_path[-1])
    eps = max(0.01, float(channel_length) * 1e-4)
    movement_direction = np.sign(diffs)
    movement_direction[np.abs(diffs) < eps] = 0
    return movement_path, turning_point, movement_direction.astype(np.float32)
