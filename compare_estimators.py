import numpy as np
import os
import sys

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

test_all_videos = os.environ.get("TEST_ALL_VIDEOS", "1") != "0"
do_plot = os.environ.get("SHOW_PLOTS", "0") == "1"
if len(sys.argv) > 1:
    test_all_videos = sys.argv[1].lower() != "single"

estimator = MovementPathEstimator(-1, test_all_videos)
estimator.execute_estimations()

# load ground-truth smoothed measurements
measured_movement_paths = {}
for fname in os.listdir('distance_labels/'):
    if not fname.endswith('.npy'):
        continue
    video_num = int(fname.split('.')[0])
    measured_movement_paths[video_num] = MovementPath(
        video_num, np.load(f'distance_labels/{fname}')
    )

rater = EstimationRater(
    estimator.calculated_movement_paths,
    measured_movement_paths,
    do_print=True,
    do_plot=do_plot,
    estimator_name=type(estimator).__name__,
    video_num=-1,
)
rater.rate()
