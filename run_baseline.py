import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


video_num_to_test = 11
preview_frames = 6


def list_frame_paths(video_number):
    frame_dir = os.path.join("frame_images", str(video_number))
    frame_names = [
        name for name in os.listdir(frame_dir)
        if name.lower().endswith(".png") and os.path.splitext(name)[0].isdigit()
    ]
    frame_names.sort(key=lambda name: int(os.path.splitext(name)[0]))
    return [os.path.join(frame_dir, name) for name in frame_names]


def build_preview_indices(num_frames, count):
    if num_frames == 0:
        return []
    if count <= 1:
        return [num_frames // 2]
    return np.linspace(0, num_frames - 1, count, dtype=int).tolist()


def summary_text(video_number, result, measured, rater):
    lines = [
        f"Video {video_number}",
        f"Predicted turning point: {result.turning_point:.1f}",
    ]
    if measured is not None and rater is not None:
        lines.extend(
            [
                f"Actual turning point: {measured.turning_point:.1f}",
                f"Turning-point error: {rater.turning_point_scores[0]:.1f} frames",
                f"Path MAE: {rater.movement_path_scores[0]:.3f} m",
                f"Direction accuracy: {rater.movement_direction_scores[0]:.3f}",
            ]
        )
    else:
        lines.append("No ground-truth label available.")
    return "\n".join(lines)


def main():
    video_number = int(sys.argv[1]) if len(sys.argv) > 1 else video_num_to_test

    estimator = MovementPathEstimator(video_number, False)
    estimator.execute_estimations()
    result = estimator.calculated_movement_paths[video_number]

    ground_truth_path = os.path.join("distance_labels", f"{video_number}.npy")
    measured = None
    rater = None
    if os.path.exists(ground_truth_path):
        measured = MovementPath(video_number, np.load(ground_truth_path))
        rater = EstimationRater(
            {0: result},
            {0: measured},
            do_print=True,
            do_plot=False,
            estimator_name=type(estimator).__name__,
            video_num=video_number,
        )
        rater.rate()
    else:
        print(f"Video {video_number}: no ground-truth label available, so no MAE can be computed.")

    frame_paths = list_frame_paths(video_number)
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, preview_frames, height_ratios=[2.2, 1.2])
    path_ax = fig.add_subplot(grid[0, :])

    path_ax.plot(result.movement_path, label="Predicted", linewidth=2.2, color="tab:blue")
    path_ax.axvline(result.turning_point, linestyle="--", linewidth=1.5, color="tab:blue", alpha=0.75)
    if measured is not None:
        path_ax.plot(measured.movement_path, label="Actual", linewidth=2.0, color="tab:orange")
        path_ax.axvline(measured.turning_point, linestyle="--", linewidth=1.5, color="tab:orange", alpha=0.75)
    path_ax.set_title("Distance Along Pipe vs Frame")
    path_ax.set_xlabel("Frame")
    path_ax.set_ylabel("Distance [m]")
    path_ax.grid(alpha=0.25)
    path_ax.legend(loc="upper left")
    path_ax.text(
        1.01,
        0.98,
        summary_text(video_number, result, measured, rater),
        transform=path_ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    for ax_index, frame_index in enumerate(build_preview_indices(len(frame_paths), preview_frames)):
        axis = fig.add_subplot(grid[1, ax_index])
        frame = mpimg.imread(frame_paths[frame_index])
        axis.imshow(frame)
        axis.set_title(f"Frame {frame_index}", fontsize=9)

        predicted_distance = result.movement_path[frame_index]
        if measured is not None and frame_index < len(measured.movement_path):
            actual_distance = measured.movement_path[frame_index]
            axis.set_xlabel(
                f"pred {predicted_distance:.1f} m\nactual {actual_distance:.1f} m",
                fontsize=8,
            )
        else:
            axis.set_xlabel(f"pred {predicted_distance:.1f} m", fontsize=8)

        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle("Baseline Run: Path Estimate and Frame Preview", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
