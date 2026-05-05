import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# ---- configure which video to run ----
video_num_to_test = int(os.environ.get("VIDEO_NUM_TO_TEST", "11"))
if len(sys.argv) > 1:
    video_num_to_test = int(sys.argv[1])
# --------------------------------------

do_plot = os.environ.get("SHOW_PLOTS", "0") == "1"

estimator = MovementPathEstimator(video_num_to_test, False)
estimator.execute_estimations()

result = estimator.calculated_movement_paths[video_num_to_test]

# EstimationRater expects key 0 in single-video mode
ground_truth_path = f'distance_labels/{video_num_to_test}.npy'
measured = {}
if os.path.exists(ground_truth_path):
    measured = {0: MovementPath(video_num_to_test, np.load(ground_truth_path))}
else:
    print(f"No ground-truth label for video {video_num_to_test} — skipping scoring.")
rater = EstimationRater(
    {0: result},
    measured,
    do_print=bool(measured),
    do_plot=bool(measured) and do_plot,
    estimator_name=type(estimator).__name__,
    video_num=video_num_to_test,
)
rater.rate()
