import os
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MovementPath import MovementPath

try:
    from .hybrid_library_model import predict_video_path
except ImportError:
    from hybrid_library_model import predict_video_path


class MovementPathEstimator:
    """
    Hackathon template: estimate the movement path and turning point
    of an object moving through a channel, based on video frames.

    The framework calls `execute_estimations()` which in turn calls
    `calculate_movement_path_and_turning_point()` for each video.
    Your job is to implement that one method.

    Inputs available inside `calculate_movement_path_and_turning_point`:
      - video_number   : int, which video (1-based)
      - channel_length : float, the physical length of the channel [m]
      - path_to_video  : str, folder containing the frame images (0.png, 1.png, ...)

    Outputs to return (as a tuple):
      - movement_path      : np.ndarray shape (N,)  position in [0, channel_length] per frame
      - turning_point      : float                  frame index where the object reverses
      - movement_direction : np.ndarray shape (N,)  +1 forward, -1 backward, 0 stationary per frame
    """

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        # Always points to whichever folder this file lives in,
        # so the estimator works regardless of the folder name.
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep

        self.calculated_movement_paths = {}

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        """
        Estimate the movement path for a single video.
        """
        return predict_video_path(int(video_number), float(channel_length))

    # ------------------------------------------------------------------ #
    #  Framework boilerplate – you should not need to change this          #
    # ------------------------------------------------------------------ #

    def execute_estimations(self):
        if self.test_all_videos:
            if not os.path.exists(self.path_to_videos):
                raise FileNotFoundError(f"The folder '{self.path_to_videos}' does not exist.")
            for entry in os.listdir(self.path_to_videos):
                if entry.isdigit():
                    self._run_single(int(entry))
        else:
            self._run_single(self.video_num_to_test)

    def _run_single(self, video_number):
        try:
            channel_length = self.channel_lengths[video_number - 1]
        except Exception:
            print("Cannot load channel length, using 100 m")
            channel_length = 100
        movement_path, turning_point, movement_direction = \
            self.calculate_movement_path_and_turning_point(int(video_number), channel_length)
        self.calculated_movement_paths[int(video_number)] = MovementPath(
            int(video_number), movement_path, movement_direction, turning_point
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the submitted MovementPathEstimator directly."
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
        help="Skip scoring even if distance_labels/ exists.",
    )
    return parser.parse_args()


def _load_measured_paths():
    measured = {}
    label_dir = Path("distance_labels")
    if not label_dir.exists():
        return measured
    for path in sorted(label_dir.glob("*.npy"), key=lambda p: int(p.stem)):
        if path.stem.isdigit():
            video_num = int(path.stem)
            measured[video_num] = MovementPath(video_num, np.load(path))
    return measured


def _main():
    args = _parse_args()
    if not Path("channel_lengths.npy").exists():
        raise FileNotFoundError("Run this from the repository root so channel_lengths.npy is available.")
    if not Path("frame_images").exists():
        raise FileNotFoundError("Missing frame_images/ next to the repository root.")

    run_all = args.video is None
    estimator = MovementPathEstimator(args.video or -1, run_all)
    estimator.execute_estimations()

    if args.no_score:
        print("Finished estimation. Scoring skipped.")
        return

    measured = _load_measured_paths()
    if not measured:
        print("Finished estimation. No distance_labels/ found, so scoring was skipped.")
        return

    from EstimationRater import EstimationRater

    if run_all:
        predicted = {
            video: path
            for video, path in estimator.calculated_movement_paths.items()
            if video in measured
        }
        rater = EstimationRater(
            predicted,
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
        rater = EstimationRater(
            {0: estimator.calculated_movement_paths[video]},
            {0: measured[video]},
            do_print=True,
            do_plot=True,
            estimator_name=type(estimator).__name__,
            video_num=video,
        )
        rater.rate()


if __name__ == "__main__":
    _main()
