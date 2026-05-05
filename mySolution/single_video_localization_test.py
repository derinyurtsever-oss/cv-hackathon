import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from localization_model import (
    _augment_features,
    _physical_postprocess,
    aligned_labels,
    extract_video_features,
    project_root,
    results_dir,
)


def _direction(path: np.ndarray, channel_length: float) -> np.ndarray:
    diffs = np.diff(path, append=path[-1])
    eps = max(0.01, float(channel_length) * 1e-4)
    direction = np.sign(diffs)
    direction[np.abs(diffs) < eps] = 0
    return direction.astype(np.float32)


def _metrics(name: str, prediction: np.ndarray, label: np.ndarray, channel_length: float):
    n = min(len(prediction), len(label))
    prediction = prediction[:n]
    label = label[:n]
    mae = float(np.mean(np.abs(prediction - label)))
    true_turn = float(np.argmax(label))
    pred_turn = float(np.argmax(prediction))
    turn_error = abs(pred_turn - true_turn)
    print(
        f"{name}: MAE={mae:.3f} m, normalized={mae / channel_length:.4f}, "
        f"turn true={true_turn:.0f}, pred={pred_turn:.0f}, err={turn_error:.0f} frames"
    )
    return mae, mae / channel_length, true_turn, pred_turn, turn_error


def _naive_triangle(n: int, channel_length: float) -> np.ndarray:
    half = max(1, n // 2)
    return np.concatenate(
        [
            np.linspace(0.0, channel_length, half, dtype=np.float32),
            np.linspace(channel_length, 0.0, n - half, dtype=np.float32),
        ]
    )


def run(video: int, train_stride: int) -> None:
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    channel_length = float(channel_lengths[video - 1])
    frame_numbers, features = extract_video_features(video)
    label = aligned_labels(video, frame_numbers)
    if label is None:
        raise FileNotFoundError(f"No distance label available for video {video}")

    x = _augment_features(features, frame_numbers, channel_length)
    y = np.clip(label / channel_length, 0.0, 1.0)

    train_idx = np.arange(0, len(y), max(1, train_stride), dtype=np.int32)
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_features=0.35,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=23,
    )
    model.fit(x[train_idx], y[train_idx])
    model_path = results_dir() / f"single_video_{video}_model.joblib"
    joblib.dump(
        {
            "video": video,
            "train_stride": train_stride,
            "frame_numbers": frame_numbers,
            "channel_length": channel_length,
            "model": model,
        },
        model_path,
    )

    raw = model.predict(x)
    pred, turning_point = _physical_postprocess(raw, channel_length)
    triangle = _naive_triangle(len(label), channel_length)

    print(f"Video {video}: {len(frame_numbers)} frames, trained on {len(train_idx)} sampled frames")
    ml_metrics = _metrics("single-video ML", pred, label, channel_length)
    baseline_metrics = _metrics("naive triangle", triangle, label, channel_length)

    out_dir = results_dir()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(label, label="label", linewidth=1.6)
    ax.plot(pred[: len(label)], label="single-video ML", linewidth=1.2)
    ax.plot(triangle, label="naive triangle", linewidth=1.0, alpha=0.75)
    ax.axvline(ml_metrics[2], color="black", linewidth=0.8, alpha=0.5, label="true turn")
    ax.axvline(ml_metrics[3], color="tab:red", linewidth=0.8, alpha=0.6, label="ML turn")
    ax.set_title(f"Video {video} localization smoke test")
    ax.set_xlabel("trimmed frame index")
    ax.set_ylabel("distance [m]")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / f"single_video_{video}_localization.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    csv_path = out_dir / f"single_video_{video}_localization.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame_number,label_m,pred_m,triangle_m,ml_direction\n")
        direction = _direction(pred, channel_length)
        for frame, label_m, pred_m, tri_m, dir_val in zip(frame_numbers, label, pred, triangle, direction):
            f.write(f"{frame},{label_m:.6f},{pred_m:.6f},{tri_m:.6f},{dir_val:.0f}\n")

    print(f"Saved plot: {plot_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=int, default=11)
    parser.add_argument("--train-stride", type=int, default=4)
    args = parser.parse_args()
    run(args.video, args.train_stride)
