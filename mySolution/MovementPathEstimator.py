import numpy as np
import os
import cv2
from scipy.ndimage import uniform_filter1d

from MovementPath import MovementPath


# Video filenames as used by the organizers' extraction script
_VIDEOS = {
    1:  "20230810_081214_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test1.mp4",
    2:  "20230810_083027_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test2.mp4",
    3:  "20230810_084551_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test3.mp4",
    4:  "20230810_090123_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test4neuerKanal.mp4",
    5:  "20230810_091120_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test5gleicherKanalwie4.mp4",
    6:  "20230810_092329_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test6neuerKanal.mp4",
    7:  "20230810_100306_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test7neuerSchacht.mp4",
    8:  "20230810_101624_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test8gleicherKanal.mp4",
    9:  "20230810_103619_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test9gleicherKanal.mp4",
    10: "20230810_104524_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test10gleicherKanal.mp4",
    11: "20230810_105533_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test11gleicherKanal.mp4",
}

# First frame index to keep per video (frames before this are trimmed)
_FIRST_KEPT_FRAME = {
    1: 180, 2: 56, 3: 58, 4: 30, 5: 55, 6: 58,
    7: 330, 8: 100, 9: 35, 10: 42, 11: 314,
}

# Last frame index to keep per video (None = keep all)
_LAST_KEPT_FRAME = {8: 1600}

# Resize every frame to this resolution before computing optical flow (speed vs. accuracy)
_FLOW_W, _FLOW_H = 160, 120

# Cache directory for computed flow arrays (avoids recomputing on every run)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flow_cache')


class MovementPathEstimator:

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        # data/ sits one level above mySolution/
        self.data_dir = os.path.normpath(os.path.join(self.current_folder, '..', 'data'))

        # DIS flow: Dense Inverse Search — 3-5× faster than Farneback, similar accuracy
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_FAST)
        self._dis.setUseSpatialPropagation(True)

        self.calculated_movement_paths = {}

    # ------------------------------------------------------------------ #
    #  Core estimator                                                      #
    # ------------------------------------------------------------------ #

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        raw_flow = self._signed_flow_for_video(video_number)

        if raw_flow is None or len(raw_flow) < 2:
            half = 1
            return (
                np.array([0.0, channel_length, 0.0]),
                1.0,
                np.array([0.0, 1.0, -1.0]),
            )

        # Smooth to suppress per-frame noise
        window = max(5, min(51, len(raw_flow) // 20))
        smoothed = uniform_filter1d(raw_flow.astype(float), size=window)

        # Cumulative integral → raw position curve (index 0 = 0 m)
        cum = np.concatenate([[0.0], np.cumsum(smoothed)])

        # Turning point = frame with maximum accumulated forward distance
        tp = int(np.argmax(cum))

        # Two-phase scaling: forward and return halves independently anchored
        forward_total = cum[tp]
        return_total  = cum[tp] - cum[-1]  # total backward flow magnitude after TP

        fwd_scale = channel_length / forward_total if forward_total > 0.0 else 1.0
        if return_total > 0.05 * forward_total:
            # Cap the asymmetry ratio to [0.6, 1.67] to guard against water-flow bias
            ret_total_capped = np.clip(return_total, 0.6 * forward_total, 1.67 * forward_total)
            ret_scale = channel_length / ret_total_capped
        else:
            ret_scale = fwd_scale

        movement_path = np.empty(len(cum))
        movement_path[:tp + 1] = cum[:tp + 1] * fwd_scale
        movement_path[tp:] = channel_length - (cum[tp] - cum[tp:]) * ret_scale
        movement_path = np.clip(movement_path, 0.0, channel_length)

        # Enforce physical monotonicity: forward-only before TP, backward-only after
        movement_path[:tp + 1] = np.maximum.accumulate(movement_path[:tp + 1])
        movement_path[tp:] = np.minimum.accumulate(movement_path[tp:])

        direction = np.concatenate([[0.0], np.sign(smoothed)])

        return movement_path, float(tp), direction

    # ------------------------------------------------------------------ #
    #  Optical flow pipeline                                               #
    # ------------------------------------------------------------------ #

    def _signed_flow_for_video(self, video_number):
        """Return a 1-D array of signed radial optical-flow values, one per frame transition."""
        # Check cache first
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(_CACHE_DIR, f'video_{video_number}.npy')
        if os.path.exists(cache_file):
            print(f"  [video {video_number}] loading cached flow")
            return np.load(cache_file)

        # Prefer pre-extracted frames (no decode overhead)
        frame_dir = os.path.join(self.path_to_videos, str(video_number))
        if os.path.exists(frame_dir) and any(f.endswith('.png') for f in os.listdir(frame_dir)):
            result = self._flow_from_images(frame_dir)
        else:
            vid_name = _VIDEOS.get(video_number)
            vid_path = os.path.join(self.data_dir, vid_name) if vid_name else None
            if vid_path and os.path.exists(vid_path):
                result = self._flow_from_mp4(vid_path, video_number)
            else:
                print(f"  [video {video_number}] no source found in frame_images/ or data/ — skipping")
                return None

        if result is not None:
            np.save(cache_file, result)
        return result

    @staticmethod
    def _radial_weight_map(h, w):
        """
        Per-pixel radial unit vectors and weights for an h×w frame.
        Weight is 0 at the centre (often a dark void), rises linearly to 1
        at the half-diagonal, then is clamped — so the pipe walls dominate.
        """
        cx, cy = w / 2.0, h / 2.0
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = x - cx, y - cy
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        rdx, rdy = dx / r, dy / r
        weight = np.clip(r / (min(h, w) / 2.0), 0.0, 1.0)
        # Mask out bottom 30% — flowing water produces spurious optical flow
        weight[int(h * 0.70):, :] = 0.0
        return rdx, rdy, weight

    def _flow_from_mp4(self, video_path, video_number):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        first_kept = _FIRST_KEPT_FRAME.get(video_number, 0)
        last_kept  = _LAST_KEPT_FRAME.get(video_number, None)

        # Skip leading frames (grab is faster than read — no decode)
        for _ in range(first_kept):
            if not cap.grab():
                cap.release()
                return None

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        prev = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (_FLOW_W, _FLOW_H))
        rdx, rdy, weight = self._radial_weight_map(_FLOW_H, _FLOW_W)
        flow_values = []
        idx = first_kept + 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if last_kept is not None and idx > last_kept:
                break

            curr = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (_FLOW_W, _FLOW_H))
            fv = self._dis.calc(prev, curr, None)
            # Positive radial flow = features diverge from centre = probe moving forward
            radial = fv[..., 0] * rdx + fv[..., 1] * rdy
            # Masked median: more robust to water/reflections than weighted mean
            valid = radial[weight > 0.1]
            flow_values.append(float(np.median(valid)) if valid.size else 0.0)
            prev = curr
            idx += 1

        cap.release()
        return np.array(flow_values, dtype=float) if flow_values else None

    def _flow_from_images(self, frame_dir):
        files = sorted(
            [f for f in os.listdir(frame_dir) if f.endswith('.png')],
            key=lambda f: int(os.path.splitext(f)[0]),
        )
        if len(files) < 2:
            return None

        first_img = cv2.imread(os.path.join(frame_dir, files[0]), cv2.IMREAD_GRAYSCALE)
        if first_img is None:
            return None

        prev = cv2.resize(first_img, (_FLOW_W, _FLOW_H))
        rdx, rdy, weight = self._radial_weight_map(_FLOW_H, _FLOW_W)
        flow_values = []

        for fname in files[1:]:
            img = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            curr = cv2.resize(img, (_FLOW_W, _FLOW_H))
            fv = self._dis.calc(prev, curr, None)
            radial = fv[..., 0] * rdx + fv[..., 1] * rdy
            valid = radial[weight > 0.1]
            flow_values.append(float(np.median(valid)) if valid.size else 0.0)
            prev = curr

        return np.array(flow_values, dtype=float) if flow_values else None

    # ------------------------------------------------------------------ #
    #  Framework boilerplate                                               #
    # ------------------------------------------------------------------ #

    def execute_estimations(self):
        if self.test_all_videos:
            available = []
            for n in range(1, 12):
                vname = _VIDEOS.get(n, '')
                has_mp4 = os.path.exists(os.path.join(self.data_dir, vname))
                fdir = os.path.join(self.path_to_videos, str(n))
                has_frames = os.path.exists(fdir) and any(
                    f.endswith('.png') for f in os.listdir(fdir)
                )
                if has_mp4 or has_frames:
                    available.append(n)
            for n in available:
                print(f"Processing video {n} ...")
                self._run_single(n)
        else:
            self._run_single(self.video_num_to_test)

    def _run_single(self, video_number):
        try:
            channel_length = self.channel_lengths[video_number - 1]
        except Exception:
            print("Cannot load channel length, using 100 m")
            channel_length = 100.0
        movement_path, turning_point, movement_direction = \
            self.calculate_movement_path_and_turning_point(int(video_number), channel_length)
        self.calculated_movement_paths[int(video_number)] = MovementPath(
            int(video_number), movement_path, movement_direction, turning_point
        )
