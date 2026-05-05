import argparse
import os
from pathlib import Path

import numpy as np

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the submission estimator on one video or all videos."
    )
    parser.add_argument(
        "--video",
        type=int,
        default=None,
        help="Video number to run. Omit to run all videos found in frame_images/.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip scoring even when distance_labels are available.",
    )
    return parser.parse_args()


def load_measured_paths() -> dict[int, MovementPath]:
    measured = {}
    label_dir = Path("distance_labels")
    if not label_dir.exists():
        return measured
    for path in sorted(label_dir.glob("*.npy"), key=lambda p: int(p.stem)):
        if path.stem.isdigit():
            video_num = int(path.stem)
            measured[video_num] = MovementPath(video_num, np.load(path))
    return measured


def main():
    args = parse_args()
    repo_root = Path(".")
    if not (repo_root / "mySolution").exists():
        raise FileNotFoundError("Run this from the repository root.")
    if not (repo_root / "frame_images").exists():
        raise FileNotFoundError("Missing frame_images/ next to main.py.")
    if not (repo_root / "channel_lengths.npy").exists():
        raise FileNotFoundError("Missing channel_lengths.npy next to main.py.")

    run_all = args.video is None
    estimator = MovementPathEstimator(args.video or -1, run_all)
    estimator.execute_estimations()

    if args.no_score:
        print("Finished estimation. Scoring skipped.")
        return

    measured = load_measured_paths()
    if not measured:
        print("Finished estimation. No distance_labels/ found, so scoring was skipped.")
        return

    if run_all:
        scored = {
            video: path
            for video, path in estimator.calculated_movement_paths.items()
            if video in measured
        }
        rater = EstimationRater(
            scored,
            measured,
            do_print=True,
            do_plot=True,
            estimator_name=type(estimator).__name__,
            video_num=-1,
        )
        rater.rate()
    else:
        video = int(args.video)
        if video not in measured:
            print(f"Finished estimation for video {video}. No label available for scoring.")
            return
        predicted = {0: estimator.calculated_movement_paths[video]}
        expected = {0: measured[video]}
        rater = EstimationRater(
            predicted,
            expected,
            do_print=True,
            do_plot=True,
            estimator_name=type(estimator).__name__,
            video_num=video,
        )
        rater.rate()


if __name__ == "__main__":
    main()
