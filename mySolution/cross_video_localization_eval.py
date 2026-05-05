import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def _parse_video_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _triangle(n: int, channel_length: float) -> np.ndarray:
    half = max(1, n // 2)
    return np.concatenate(
        [
            np.linspace(0.0, channel_length, half, dtype=np.float32),
            np.linspace(channel_length, 0.0, n - half, dtype=np.float32),
        ]
    )


def _direction(path: np.ndarray, channel_length: float) -> np.ndarray:
    diffs = np.diff(path, append=path[-1])
    eps = max(0.01, channel_length * 1e-4)
    direction = np.sign(diffs)
    direction[np.abs(diffs) < eps] = 0
    return direction.astype(np.float32)


def _metrics(name: str, pred: np.ndarray, labels: np.ndarray, channel_length: float):
    n = min(len(pred), len(labels))
    pred = pred[:n]
    labels = labels[:n]
    err = np.abs(pred - labels)
    mae = float(err.mean())
    p90 = float(np.percentile(err, 90))
    true_turn = float(np.argmax(labels))
    pred_turn = float(np.argmax(pred))
    turn_err = abs(pred_turn - true_turn)
    print(
        f"{name}: MAE={mae:.3f} m, norm={mae / channel_length:.4f}, "
        f"p90={p90:.3f} m, turn_err={turn_err:.0f}"
    )
    return {
        "mae_m": mae,
        "norm_mae": mae / channel_length,
        "p90_m": p90,
        "true_turn": true_turn,
        "pred_turn": pred_turn,
        "turn_error": turn_err,
    }


def train(train_videos: list[int], stride: int):
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    x_parts = []
    y_parts = []
    used = []

    for video in train_videos:
        frame_numbers, features = extract_video_features(video)
        labels = aligned_labels(video, frame_numbers)
        if labels is None:
            print(f"Skipping train video {video}: no label")
            continue
        idx = np.arange(0, len(frame_numbers), max(1, stride), dtype=np.int32)
        channel_length = float(channel_lengths[video - 1])
        x_parts.append(_augment_features(features[idx], frame_numbers[idx], channel_length))
        y_parts.append(np.clip(labels[idx] / channel_length, 0.0, 1.0))
        used.append(video)

    if not x_parts:
        raise RuntimeError("No labeled training videos were available.")

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    model = ExtraTreesRegressor(
        n_estimators=260,
        max_features=0.35,
        min_samples_leaf=2,
        bootstrap=False,
        n_jobs=-1,
        random_state=59,
    )
    print(f"Training on videos {used}, samples={len(y)}, feature_dim={x.shape[1]}")
    model.fit(x, y)
    return {"model": model, "train_videos": used, "stride": stride}


def predict_video(bundle, video: int):
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    channel_length = float(channel_lengths[video - 1])
    frame_numbers, features = extract_video_features(video)
    x = _augment_features(features, frame_numbers, channel_length)
    raw_norm = bundle["model"].predict(x)
    pred, turning_point = _physical_postprocess(raw_norm, channel_length)
    return frame_numbers, pred, turning_point, _direction(pred, channel_length)


def run(train_videos: list[int], predict_videos: list[int], stride: int):
    out_dir = results_dir()
    model_name = "cross_video_model_train_" + "-".join(map(str, train_videos)) + ".joblib"
    model_file = out_dir / model_name
    if model_file.exists():
        bundle = joblib.load(model_file)
        print(f"Loaded existing model: {model_file}")
    else:
        bundle = train(train_videos, stride)
        model_name = "cross_video_model_train_" + "-".join(map(str, bundle["train_videos"])) + ".joblib"
        model_file = out_dir / model_name
        joblib.dump(bundle, model_file)
        print(f"Saved model: {model_file}")

    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    eval_rows = []
    labeled_plots = []

    for video in predict_videos:
        channel_length = float(channel_lengths[video - 1])
        frame_numbers, pred, turning_point, direction = predict_video(bundle, video)
        labels = aligned_labels(video, frame_numbers)
        triangle = _triangle(len(pred), channel_length)

        csv_file = out_dir / f"cross_video_prediction_{video}.csv"
        with csv_file.open("w", encoding="utf-8") as f:
            f.write("frame_number,pred_m,direction,triangle_m,label_m\n")
            for i, frame in enumerate(frame_numbers):
                label_val = "" if labels is None else f"{labels[i]:.6f}"
                f.write(
                    f"{frame},{pred[i]:.6f},{direction[i]:.0f},{triangle[i]:.6f},{label_val}\n"
                )
        print(f"Saved prediction CSV for video {video}: {csv_file}")

        if labels is not None:
            print(f"\nHeld-out labeled video {video}:")
            ml = _metrics("cross-video ML", pred, labels, channel_length)
            base = _metrics("triangle", triangle, labels, channel_length)
            eval_rows.append((video, len(frame_numbers), ml, base))
            labeled_plots.append((video, frame_numbers, labels, pred, triangle, ml, base))
        else:
            print(
                f"Predicted unlabeled video {video}: frames={len(frame_numbers)}, "
                f"turn={turning_point:.0f}"
            )

    eval_file = out_dir / "cross_video_eval.csv"
    with eval_file.open("w", encoding="utf-8") as f:
        f.write(
            "video,frames,ml_mae_m,ml_norm_mae,ml_p90_m,ml_turn_error,"
            "triangle_mae_m,triangle_norm_mae,triangle_p90_m,triangle_turn_error\n"
        )
        for video, frames, ml, base in eval_rows:
            f.write(
                f"{video},{frames},{ml['mae_m']:.6f},{ml['norm_mae']:.6f},"
                f"{ml['p90_m']:.6f},{ml['turn_error']:.3f},"
                f"{base['mae_m']:.6f},{base['norm_mae']:.6f},"
                f"{base['p90_m']:.6f},{base['turn_error']:.3f}\n"
            )
    print(f"Saved eval CSV: {eval_file}")

    if labeled_plots:
        fig, axes = plt.subplots(len(labeled_plots), 1, figsize=(11, 4.2 * len(labeled_plots)))
        if len(labeled_plots) == 1:
            axes = [axes]
        for ax, (video, _frames, labels, pred, triangle, ml, base) in zip(axes, labeled_plots):
            ax.plot(labels, label="label", linewidth=1.5)
            ax.plot(pred[: len(labels)], label="cross-video ML", linewidth=1.2)
            ax.plot(triangle[: len(labels)], label="triangle", linewidth=1.0, alpha=0.7)
            ax.axvline(ml["true_turn"], color="black", linewidth=0.8, alpha=0.5)
            ax.axvline(ml["pred_turn"], color="tab:red", linewidth=0.8, alpha=0.6)
            ax.set_title(
                f"Video {video}: ML MAE {ml['mae_m']:.2f} m vs triangle {base['mae_m']:.2f} m"
            )
            ax.set_ylabel("distance [m]")
            ax.legend()
        axes[-1].set_xlabel("trimmed frame index")
        fig.tight_layout()
        plot_file = out_dir / "cross_video_labeled_predictions.png"
        fig.savefig(plot_file, dpi=160)
        plt.close(fig)
        print(f"Saved labeled comparison plot: {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-videos", default="1,2,3,4,8,9")
    parser.add_argument("--predict-videos", default="5,6,7,10,11")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    run(_parse_video_list(args.train_videos), _parse_video_list(args.predict_videos), args.stride)
