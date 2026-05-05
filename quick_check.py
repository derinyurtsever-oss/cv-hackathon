import os
import sys

import numpy as np

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


# Fastest labeled default video for a quick sanity check.
video_num_to_test = 11


def main():
    video_number = int(sys.argv[1]) if len(sys.argv) > 1 else video_num_to_test

    estimator = MovementPathEstimator(video_number, False)
    estimator.execute_estimations()
    result = estimator.calculated_movement_paths[video_number]

    ground_truth_path = os.path.join("distance_labels", f"{video_number}.npy")
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
        print(f"Video {video_number}: estimator ran, but no ground-truth label is available.")


if __name__ == "__main__":
    main()
