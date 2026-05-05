import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression

try:
    from .annular_unwrap_place_eval import extract_annular_features
    from .localization_model import extract_video_features, project_root
    from .place_recognition_direction_test import _sequence_place_predict
except ImportError:
    from annular_unwrap_place_eval import extract_annular_features
    from localization_model import extract_video_features, project_root
    from place_recognition_direction_test import _sequence_place_predict


BASE_ANNULAR_DIM = 1258
_LIBRARY_CACHE = None


def _available_label_videos() -> tuple[int, ...]:
    root = project_root() / "distance_labels"
    videos = []
    for path in sorted(root.glob("*.npy"), key=lambda p: int(p.stem)):
        if path.stem.isdigit():
            videos.append(int(path.stem))
    if not videos:
        raise RuntimeError("No labeled videos found for hybrid library model.")
    return tuple(videos)


def _aligned_labels(video: int, frame_numbers: np.ndarray) -> np.ndarray | None:
    try:
        from .localization_model import aligned_labels
    except ImportError:
        from localization_model import aligned_labels

    return aligned_labels(video, frame_numbers)


def _windowed_features(features: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(features) <= 2:
        return features
    out = np.empty_like(features)
    for idx in range(len(features)):
        lo = max(0, idx - radius)
        hi = min(len(features), idx + radius + 1)
        out[idx] = features[lo:hi].mean(axis=0)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-6
    return out.astype(np.float32)


def _pool_descriptor(features: np.ndarray) -> np.ndarray:
    pooled = features.mean(axis=0).astype(np.float32)
    pooled /= np.linalg.norm(pooled) + 1e-6
    return pooled


def _hybrid_features(video: int, mode: str):
    frame_numbers0, base = extract_video_features(video)
    frame_numbers1, ann_full = extract_annular_features(video, force=False)
    if not np.array_equal(frame_numbers0, frame_numbers1):
        raise RuntimeError(f"Frame mismatch in video {video}")
    base = base.astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-6
    if mode == "base":
        ann = ann_full[:, :BASE_ANNULAR_DIM].astype(np.float32)
    elif mode == "seam":
        ann = ann_full.astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    ann /= np.linalg.norm(ann, axis=1, keepdims=True) + 1e-6
    feats = np.hstack([base, 0.85 * ann]).astype(np.float32)
    return frame_numbers0, _windowed_features(feats, radius=2)


def _reference_library(mode: str):
    cache_path = project_root() / "mySolution" / "results" / f"runtime_ref_library_{mode}_v1.joblib"
    videos = _available_label_videos()
    if cache_path.exists():
        payload = joblib.load(cache_path)
        if payload.get("mode") == mode and tuple(payload.get("videos", ())) == tuple(videos):
            return payload["refs"], float(payload["turn_frac"])

    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    refs = []
    turn_fracs = []
    for video in videos:
        frame_numbers, features = _hybrid_features(video, mode)
        labels = _aligned_labels(video, frame_numbers)
        if labels is None:
            continue
        channel_length = float(channel_lengths[video - 1])
        norm_labels = np.clip(labels / channel_length, 0.0, 1.0).astype(np.float32)
        turn = int(np.argmax(labels))
        turn_fracs.append(turn / max(1, len(labels) - 1))
        inward = np.arange(0, turn + 1, dtype=np.int32)
        ret_rev = np.arange(turn, len(labels), dtype=np.int32)[::-1]
        refs.append((features[inward], norm_labels[inward], _pool_descriptor(features[inward])))
        refs.append((features[ret_rev], norm_labels[ret_rev], _pool_descriptor(features[ret_rev])))
    if not refs:
        raise RuntimeError(f"No reference videos available for mode {mode}.")
    turn_frac = float(np.median(turn_fracs))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "mode": mode,
            "videos": tuple(videos),
            "turn_frac": turn_frac,
            "refs": refs,
        },
        cache_path,
    )
    return refs, turn_frac


def _get_library():
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        _LIBRARY_CACHE = {
            "base": _reference_library("base"),
            "seam": _reference_library("seam"),
        }
    return _LIBRARY_CACHE


def _ensemble_predict(refs, query_features, top_k: int = 3, prefilter_k: int = 4):
    query_pool = _pool_descriptor(query_features)
    shortlist = []
    for ref_features, ref_labels, ref_pool in refs:
        shortlist.append((float(np.dot(query_pool, ref_pool)), ref_features, ref_labels))
    shortlist.sort(key=lambda item: item[0], reverse=True)
    shortlist = shortlist[: min(prefilter_k, len(shortlist))]

    candidates = []
    for _coarse, ref_features, ref_labels in shortlist:
        pred_norm, _matched, cost = _sequence_place_predict(ref_features, ref_labels, query_features)
        candidates.append((float(cost), pred_norm))
    candidates.sort(key=lambda item: item[0])
    chosen = candidates[: min(top_k, len(candidates))]
    costs = np.array([item[0] for item in chosen], dtype=np.float32)
    weights = np.exp(-(costs - costs.min()) / 0.02)
    weights /= weights.sum() + 1e-6
    pred = np.zeros_like(chosen[0][1], dtype=np.float32)
    for weight, (_cost, cur_pred) in zip(weights, chosen):
        pred += float(weight) * cur_pred.astype(np.float32)
    margin = float(chosen[1][0] - chosen[0][0]) if len(chosen) > 1 else 0.0
    mean_cost = float(np.mean(costs))
    return pred, mean_cost, margin


def _downsample(seq: np.ndarray, stride: int) -> np.ndarray:
    return seq if stride <= 1 else seq[::stride]


def _candidate_turns(n: int, center_frac: float) -> list[int]:
    lo = max(1, int(round((center_frac - 0.18) * (n - 1))))
    hi = min(n - 2, int(round((center_frac + 0.18) * (n - 1))))
    coarse = np.linspace(lo, hi, 11, dtype=int)
    mids = np.linspace(max(1, int(0.34 * (n - 1))), min(n - 2, int(0.66 * (n - 1))), 5, dtype=int)
    return sorted(set(int(x) for x in np.concatenate([coarse, mids])))


def _choose_turn(refs, features: np.ndarray, center_frac: float) -> int:
    n = len(features)
    stride = 8 if n > 1400 else 6
    coarse_features = _downsample(features, stride)
    best = None
    for turn in _candidate_turns(n, center_frac):
        turn_ds = max(1, min(len(coarse_features) - 2, int(round(turn / stride))))
        inward = np.arange(0, turn_ds + 1, dtype=np.int32)
        ret = np.arange(turn_ds, len(coarse_features), dtype=np.int32)
        ret_rev = ret[::-1]
        _pred_in, cost_in, margin_in = _ensemble_predict(refs, coarse_features[inward], top_k=2, prefilter_k=3)
        _pred_ret, cost_ret, margin_ret = _ensemble_predict(refs, coarse_features[ret_rev], top_k=2, prefilter_k=3)
        balance = abs(len(inward) - len(ret)) / max(1, len(coarse_features))
        score = cost_in + cost_ret + 0.02 * balance - 0.05 * (margin_in + margin_ret)
        if best is None or score < best[0]:
            best = (score, turn)
    return int(best[1])


def _fit_physical_path(pred: np.ndarray, channel_length: float, suggested_turn: int) -> np.ndarray:
    n = len(pred)
    if n <= 3:
        return np.clip(pred, 0.0, channel_length).astype(np.float32)
    smooth = gaussian_filter1d(pred.astype(np.float32), sigma=max(1.0, n / 250.0))
    turn_lo = max(1, suggested_turn - max(8, n // 18))
    turn_hi = min(n - 2, suggested_turn + max(8, n // 18))
    peak_idx = int(turn_lo + np.argmax(smooth[turn_lo : turn_hi + 1]))

    x = np.arange(n, dtype=np.float32)
    left = IsotonicRegression(increasing=True, y_min=0.0, y_max=float(channel_length)).fit_transform(
        x[: peak_idx + 1], smooth[: peak_idx + 1]
    )
    right = IsotonicRegression(increasing=False, y_min=0.0, y_max=float(channel_length)).fit_transform(
        x[peak_idx:], smooth[peak_idx:]
    )
    shaped = np.concatenate([left[:-1], right]).astype(np.float32)
    shaped[0] = 0.0
    shaped[-1] = 0.0
    peak = float(np.max(shaped))
    if peak > 1e-6:
        target_peak = min(float(channel_length), max(0.88 * float(channel_length), peak))
        shaped *= target_peak / peak
    shaped = gaussian_filter1d(shaped, sigma=max(0.8, n / 900.0))
    shaped = np.clip(shaped, 0.0, float(channel_length))
    shaped[0] = 0.0
    shaped[-1] = 0.0
    return shaped.astype(np.float32)


def _predict_branch(mode: str, video_number: int, channel_length: float):
    refs, turn_frac = _get_library()[mode]
    frame_numbers, features = _hybrid_features(video_number, mode)
    turn = _choose_turn(refs, features, turn_frac)

    inward = np.arange(0, turn + 1, dtype=np.int32)
    ret = np.arange(turn, len(features), dtype=np.int32)
    ret_rev = ret[::-1]

    pred_in_norm, in_cost, in_margin = _ensemble_predict(refs, features[inward], top_k=3, prefilter_k=4)
    pred_ret_norm_rev, ret_cost, ret_margin = _ensemble_predict(refs, features[ret_rev], top_k=3, prefilter_k=4)

    pred = np.zeros(len(features), dtype=np.float32)
    pred[inward] = pred_in_norm * float(channel_length)
    pred[ret] = pred_ret_norm_rev[::-1] * float(channel_length)
    pred = np.clip(pred, 0.0, float(channel_length))
    return {
        "frame_numbers": frame_numbers,
        "pred": pred,
        "turn": turn,
        "inward": inward,
        "ret": ret,
        "in_conf": float(in_margin - 0.15 * in_cost),
        "ret_conf": float(ret_margin - 0.15 * ret_cost),
    }


def _mix_halves(base_info, seam_info, channel_length: float):
    n = len(base_info["pred"])
    split_turn = int(round(0.5 * (base_info["turn"] + seam_info["turn"])))
    inward = np.arange(0, split_turn + 1, dtype=np.int32)
    ret = np.arange(split_turn, n, dtype=np.int32)

    def seam_weight(base_conf: float, seam_conf: float) -> float:
        delta = np.clip((seam_conf - base_conf) / 0.08, -1.0, 1.0)
        w = 0.30 + 0.20 * delta
        return float(np.clip(w, 0.10, 0.50))

    w_seam_in = seam_weight(base_info["in_conf"], seam_info["in_conf"])
    w_seam_ret = seam_weight(base_info["ret_conf"], seam_info["ret_conf"])

    mixed = np.zeros(n, dtype=np.float32)
    mixed[inward] = (1.0 - w_seam_in) * base_info["pred"][inward] + w_seam_in * seam_info["pred"][inward]
    mixed[ret] = (1.0 - w_seam_ret) * base_info["pred"][ret] + w_seam_ret * seam_info["pred"][ret]
    mixed = _fit_physical_path(np.clip(mixed, 0.0, float(channel_length)), float(channel_length), split_turn)
    return mixed


def predict_video_path(video_number: int, channel_length: float):
    base_info = _predict_branch("base", int(video_number), float(channel_length))
    seam_info = _predict_branch("seam", int(video_number), float(channel_length))
    if not np.array_equal(base_info["frame_numbers"], seam_info["frame_numbers"]):
        raise RuntimeError(f"Frame mismatch between branches for video {video_number}")

    movement_path = _mix_halves(base_info, seam_info, float(channel_length))
    turning_point = float(np.argmax(movement_path))
    diffs = np.diff(movement_path, append=movement_path[-1])
    eps = max(0.01, float(channel_length) * 1e-4)
    movement_direction = np.sign(diffs)
    movement_direction[np.abs(diffs) < eps] = 0
    return movement_path.astype(np.float32), turning_point, movement_direction.astype(np.float32)
