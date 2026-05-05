from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from localization_model import (
    aligned_labels,
    extract_video_features,
    predict_video_path,
    results_dir,
    train_model,
    project_root,
)


def _metrics(prediction, labels):
    max_len = min(len(prediction), len(labels))
    prediction = prediction[:max_len]
    labels = labels[:max_len]
    mae = float(np.mean(np.abs(prediction - labels)))
    true_turn = float(np.argmax(labels))
    pred_turn = float(np.argmax(prediction))
    return mae, true_turn, pred_turn, abs(pred_turn - true_turn)


def evaluate():
    channel_lengths = np.load(project_root() / "channel_lengths.npy")
    videos = [1, 2, 3, 4, 8, 9, 10, 11]
    rows = []

    print("Running leave-one-video-out evaluation...")
    for video in videos:
        bundle = train_model(exclude_video=video, stride=3, save=False)
        frame_numbers, _ = extract_video_features(video)
        labels = aligned_labels(video, frame_numbers)
        pred, turn, _ = predict_video_path(video, channel_lengths[video - 1], bundle=bundle)
        mae, true_turn, pred_turn, turn_err = _metrics(pred, labels)
        rows.append(
            {
                "mode": "leave_one_out",
                "video": video,
                "frames": len(frame_numbers),
                "mae_m": mae,
                "norm_mae": mae / float(channel_lengths[video - 1]),
                "true_turn": true_turn,
                "pred_turn": pred_turn,
                "turn_error": turn_err,
            }
        )
        print(
            f"LOO video {video}: MAE={mae:.3f} m, "
            f"norm={mae / float(channel_lengths[video - 1]):.4f}, turn_err={turn_err:.1f}"
        )

    print("\nTraining final model on all labeled videos...")
    final_bundle = train_model(exclude_video=None, stride=2, save=True)

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=False)
    for ax, video in zip(axes.ravel(), videos):
        frame_numbers, _ = extract_video_features(video)
        labels = aligned_labels(video, frame_numbers)
        pred, turn, _ = predict_video_path(video, channel_lengths[video - 1], bundle=final_bundle)
        mae, true_turn, pred_turn, turn_err = _metrics(pred, labels)
        rows.append(
            {
                "mode": "train_all",
                "video": video,
                "frames": len(frame_numbers),
                "mae_m": mae,
                "norm_mae": mae / float(channel_lengths[video - 1]),
                "true_turn": true_turn,
                "pred_turn": pred_turn,
                "turn_error": turn_err,
            }
        )
        ax.plot(labels, label="label", linewidth=1.2)
        ax.plot(pred[: len(labels)], label="prediction", linewidth=1.0)
        ax.axvline(true_turn, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(pred_turn, color="tab:red", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Video {video}: MAE {mae:.2f} m, turn {turn_err:.0f} frames")
        ax.set_ylabel("m")
    axes.ravel()[0].legend()
    fig.tight_layout()

    out_dir = results_dir()
    fig.savefig(out_dir / "localization_predictions_labeled.png", dpi=160)
    plt.close(fig)

    csv_path = out_dir / "localization_eval.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("mode,video,frames,mae_m,norm_mae,true_turn,pred_turn,turn_error\n")
        for row in rows:
            f.write(
                f"{row['mode']},{row['video']},{row['frames']},"
                f"{row['mae_m']:.6f},{row['norm_mae']:.6f},"
                f"{row['true_turn']:.3f},{row['pred_turn']:.3f},{row['turn_error']:.3f}\n"
            )

    loo = [row for row in rows if row["mode"] == "leave_one_out"]
    train_all = [row for row in rows if row["mode"] == "train_all"]
    summary_path = out_dir / "localization_eval_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Localization Model Evaluation\n\n")
        f.write("Features ignore the bottom 20% of each image and use coarse visual texture/color/structure descriptors.\n\n")
        for name, group in (("Leave-one-video-out", loo), ("Train-all labeled fit", train_all)):
            f.write(f"## {name}\n\n")
            f.write(
                f"- Mean MAE: {np.mean([r['mae_m'] for r in group]):.3f} m\n"
                f"- Mean normalized MAE: {np.mean([r['norm_mae'] for r in group]):.4f}\n"
                f"- Mean turn error: {np.mean([r['turn_error'] for r in group]):.1f} frames\n\n"
            )
    print(f"\nSaved {csv_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {out_dir / 'localization_predictions_labeled.png'}")


if __name__ == "__main__":
    evaluate()
