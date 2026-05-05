import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression

from localization_model import aligned_labels, extract_video_features, project_root, results_dir


def _fit_model(features: np.ndarray, labels_m: np.ndarray, channel_length: float, seed: int):
    model = ExtraTreesRegressor(
        n_estimators=320,
        max_features=0.35,
        min_samples_leaf=2,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features, np.clip(labels_m / channel_length, 0.0, 1.0))
    return model


def _side_smooth(raw_m: np.ndarray, increasing: bool, channel_length: float) -> np.ndarray:
    raw = np.clip(np.nan_to_num(raw_m, nan=0.0), 0.0, channel_length)
    if len(raw) < 4:
        return raw
    smooth = gaussian_filter1d(raw, sigma=max(1.0, len(raw) / 180.0))
    x = np.arange(len(raw), dtype=np.float32)
    iso = IsotonicRegression(increasing=increasing, y_min=0.0, y_max=channel_length)
    shaped = iso.fit_transform(x, smooth)
    return np.asarray(shaped, dtype=np.float32)


def _metrics(name: str, prediction: np.ndarray, labels: np.ndarray, channel_length: float):
    err = np.abs(prediction - labels)
    mae = float(err.mean())
    p90 = float(np.percentile(err, 90))
    print(f"{name}: MAE={mae:.3f} m, norm={mae / channel_length:.4f}, p90={p90:.3f} m")
    return mae, mae / channel_length, p90


def _triangle(n: int, channel_length: float):
    half = max(1, n // 2)
    return np.concatenate(
        [
            np.linspace(0.0, channel_length, half, dtype=np.float32),
            np.linspace(channel_length, 0.0, n - half, dtype=np.float32),
        ]
    )


def run(video: int, gap: int):
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    channel_length = float(channel_lengths[video - 1])
    frame_numbers, features = extract_video_features(video)
    labels = aligned_labels(video, frame_numbers)
    if labels is None:
        raise FileNotFoundError(f"No distance label available for video {video}")

    turn = int(np.argmax(labels))
    in_idx = np.arange(0, max(1, turn - gap + 1), dtype=np.int32)
    out_idx = np.arange(min(len(labels) - 1, turn + gap), len(labels), dtype=np.int32)

    in_model = _fit_model(features[in_idx], labels[in_idx], channel_length, seed=31)
    out_model = _fit_model(features[out_idx], labels[out_idx], channel_length, seed=37)

    raw_in_to_out = in_model.predict(features[out_idx]) * channel_length
    raw_out_to_in = out_model.predict(features[in_idx]) * channel_length
    mono_in_to_out = _side_smooth(raw_in_to_out, increasing=False, channel_length=channel_length)
    mono_out_to_in = _side_smooth(raw_out_to_in, increasing=True, channel_length=channel_length)
    baseline = _triangle(len(labels), channel_length)

    print(f"Video {video}: frames={len(labels)}, true turn={turn}, gap={gap}")
    print(f"Train inward side: {len(in_idx)} frames; test return side: {len(out_idx)} frames")
    in_raw = _metrics("inward->return raw", raw_in_to_out, labels[out_idx], channel_length)
    in_mono = _metrics("inward->return monotone", mono_in_to_out, labels[out_idx], channel_length)
    base_out = _metrics("triangle on return side", baseline[out_idx], labels[out_idx], channel_length)
    print(f"Train return side: {len(out_idx)} frames; test inward side: {len(in_idx)} frames")
    out_raw = _metrics("return->inward raw", raw_out_to_in, labels[in_idx], channel_length)
    out_mono = _metrics("return->inward monotone", mono_out_to_in, labels[in_idx], channel_length)
    base_in = _metrics("triangle on inward side", baseline[in_idx], labels[in_idx], channel_length)

    pred_in_to_out = np.full_like(labels, np.nan, dtype=np.float32)
    pred_out_to_in = np.full_like(labels, np.nan, dtype=np.float32)
    pred_in_to_out[out_idx] = mono_in_to_out
    pred_out_to_in[in_idx] = mono_out_to_in

    out_dir = results_dir()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(labels, label="label", linewidth=1.5)
    axes[0].plot(pred_in_to_out, label="train inward -> predict return", linewidth=1.2)
    axes[0].plot(baseline, label="triangle", linewidth=1.0, alpha=0.7)
    axes[0].axvline(turn, color="black", linewidth=0.8, alpha=0.55)
    axes[0].set_title(f"Video {video}: direction split, inward to return")
    axes[0].set_ylabel("distance [m]")
    axes[0].legend()

    axes[1].plot(labels, label="label", linewidth=1.5)
    axes[1].plot(pred_out_to_in, label="train return -> predict inward", linewidth=1.2)
    axes[1].plot(baseline, label="triangle", linewidth=1.0, alpha=0.7)
    axes[1].axvline(turn, color="black", linewidth=0.8, alpha=0.55)
    axes[1].set_title(f"Video {video}: direction split, return to inward")
    axes[1].set_xlabel("trimmed frame index")
    axes[1].set_ylabel("distance [m]")
    axes[1].legend()
    fig.tight_layout()

    plot_path = out_dir / f"direction_split_video_{video}.png"
    csv_path = out_dir / f"direction_split_video_{video}.csv"
    summary_path = out_dir / f"direction_split_video_{video}_summary.txt"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame_number,label_m,triangle_m,inward_to_return_m,return_to_inward_m\n")
        for frame, label, tri, pred_a, pred_b in zip(
            frame_numbers, labels, baseline, pred_in_to_out, pred_out_to_in
        ):
            f.write(f"{frame},{label:.6f},{tri:.6f},{pred_a:.6f},{pred_b:.6f}\n")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"video={video}\nturn={turn}\ngap={gap}\n")
        f.write(f"inward_to_return_raw_mae={in_raw[0]:.6f}\n")
        f.write(f"inward_to_return_monotone_mae={in_mono[0]:.6f}\n")
        f.write(f"triangle_return_mae={base_out[0]:.6f}\n")
        f.write(f"return_to_inward_raw_mae={out_raw[0]:.6f}\n")
        f.write(f"return_to_inward_monotone_mae={out_mono[0]:.6f}\n")
        f.write(f"triangle_inward_mae={base_in[0]:.6f}\n")

    print(f"Saved plot: {plot_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=int, default=11)
    parser.add_argument("--gap", type=int, default=0)
    args = parser.parse_args()
    run(args.video, args.gap)
