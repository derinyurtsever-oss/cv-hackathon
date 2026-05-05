import os
import sys

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MovementPath import MovementPath


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

_FIRST_KEPT_FRAME = {
    1: 180, 2: 56, 3: 58, 4: 30, 5: 55, 6: 58,
    7: 330, 8: 100, 9: 35, 10: 42, 11: 314,
}

_LAST_KEPT_FRAME = {8: 1600}

_FLOW_W, _FLOW_H = 160, 120

_FLOW_CACHE_DIR = os.path.join(CURRENT_DIR, "flow_cache")
_FLOW_CACHE_VERSION = "annulus_v4"

_TURN_SIGNAL_MAX_WINDOW = 47
_TURN_SIGNAL_DIVISOR = None
_TURN_THRESHOLD_PERCENTILE = 10.0
_TURN_THRESHOLD_MULTIPLIER = 0.30

_PATH_SIGNAL_WINDOW = 41
_PATH_THRESHOLD_PERCENTILE = 26.0
_PATH_THRESHOLD_MULTIPLIER = 0.35
_PATH_GAMMA_OUTBOUND = 0.18
_PATH_GAMMA_INBOUND = 0.002
_PATH_SMOOTH_WINDOW = 17


class MovementPathEstimator:
    """
    Estimates the movement path and turning point of a sewer-inspection probe
    as it travels through a pipe channel and returns.

    The pipeline is:
      1. Load or compute a per-frame signed radial-flow signal.
      2. Clean obvious bad frames via confidence gating.
      3. Detect the turning-point frame from the flow sign pattern.
      4. Integrate forward and backward motion into a distance curve.
      5. Enforce a physically valid unimodal shape.
      6. Derive per-frame direction labels from the final path.

    Public entry point: execute_estimations()
    Results written to:  self.calculated_movement_paths[video_number]
    """

    def __init__(self, video_num_to_test, test_all_videos):
        """
        Prepare file paths and image-processing utilities.

        Sets up:
          - channel_lengths      : known physical channel lengths per video
          - path_to_videos       : expected extracted-frames root (frame_images/)
          - video_search_dirs    : fallback raw-video directories (data/, Videos/, videos/)
          - _dis                 : OpenCV DIS dense optical-flow engine
          - _clahe               : CLAHE contrast enhancer
          - calculated_movement_paths : output dictionary keyed by video number
        """
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.video_search_dirs = [
            os.path.join(PROJECT_ROOT, "data"),
            os.path.join(PROJECT_ROOT, "Videos"),
            os.path.join(PROJECT_ROOT, "videos"),
        ]

        self._dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_FAST)
        self._dis.setUseSpatialPropagation(True)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.calculated_movement_paths = {}

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        """
        Core estimation method called by the framework for each video.

        Steps:
          1. Retrieve the raw signed-flow and per-frame confidence signals.
          2. Gate obvious bad spikes using confidence (_apply_confidence_gating).
          3. Estimate the reversal frame from the raw flow sign pattern.
          4. Build the distance path by integrating the gated signal around the turn.
          5. Derive direction labels from the final path.

        Returns:
          movement_path      (np.ndarray, shape N)  position in [0, channel_length] per frame
          turning_point      (float)                frame index of the reversal
          movement_direction (np.ndarray, shape N)  +1 forward / -1 backward / 0 stationary
        """
        raw_flow, flow_confidence = self._flow_signal_for_video(video_number)

        if raw_flow is None or len(raw_flow) < 2:
            return (
                np.array([0.0, channel_length, 0.0]),
                1.0,
                np.array([0.0, 1.0, -1.0]),
            )

        gated_flow = self._apply_confidence_gating(raw_flow, flow_confidence)

        turn_signal, turn_threshold = self._stabilize_flow_signal(
            raw_flow,
            window=self._window_from_length(len(raw_flow), _TURN_SIGNAL_DIVISOR, _TURN_SIGNAL_MAX_WINDOW),
            percentile=_TURN_THRESHOLD_PERCENTILE,
            threshold_multiplier=_TURN_THRESHOLD_MULTIPLIER,
        )
        turn = int(self._estimate_turning_point(turn_signal, turn_threshold))

        path_signal, path_threshold = self._stabilize_flow_signal(
            gated_flow,
            window=self._window_from_length(len(raw_flow), None, _PATH_SIGNAL_WINDOW),
            percentile=_PATH_THRESHOLD_PERCENTILE,
            threshold_multiplier=_PATH_THRESHOLD_MULTIPLIER,
        )
        movement_path = self._integrate_path(
            path_signal,
            turn,
            path_threshold,
            channel_length,
            gamma_outbound=_PATH_GAMMA_OUTBOUND,
            gamma_inbound=_PATH_GAMMA_INBOUND,
            smooth_window=self._window_from_length(len(raw_flow) + 1, None, _PATH_SMOOTH_WINDOW),
        )

        movement_direction = self._path_to_direction(movement_path, turn, channel_length)
        return movement_path, float(turn), movement_direction

    @staticmethod
    def _window_from_length(signal_length, divisor, max_window):
        """Return an odd smoothing-window size clamped to [3, signal_length]."""
        if signal_length <= 1:
            return 1
        if divisor is None:
            window = max_window
        else:
            window = max(5, signal_length // divisor)
            window = min(window, max_window)
        window = max(3, min(window, signal_length))
        if window % 2 == 0:
            window = max(3, window - 1)
        return window

    @staticmethod
    def _frame_image_files(frame_dir):
        """Return numerically sorted list of PNG filenames inside frame_dir."""
        if not os.path.isdir(frame_dir):
            return []

        files = [name for name in os.listdir(frame_dir) if name.endswith(".png")]
        files.sort(key=lambda name: int(os.path.splitext(name)[0]))
        return files

    def _flow_signal_for_video(self, video_number):
        """
        Return (flow, confidence) arrays for a video, using disk cache when available.

        Cache path: mySolution/flow_cache/annulus_v4_video_<n>.npz

        If not cached, reads extracted PNGs from frame_images/<n>/ first;
        falls back to the raw MP4 in data/, Videos/, or videos/.
        After computing, saves the result to the cache for future runs.
        """
        os.makedirs(_FLOW_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(_FLOW_CACHE_DIR, f"{_FLOW_CACHE_VERSION}_video_{video_number}.npz")
        if os.path.exists(cache_file):
            print(f"  [video {video_number}] loading cached flow")
            cached = np.load(cache_file)
            return cached["flow"], cached["confidence"]

        frame_dir = os.path.join(self.path_to_videos, str(video_number))
        frame_files = self._frame_image_files(frame_dir)
        if frame_files:
            result = self._flow_from_images(frame_dir)
        else:
            video_path = self._resolve_video_path(video_number)
            if video_path is not None:
                result = self._flow_from_mp4(video_path, video_number)
            else:
                print(f"  [video {video_number}] no source found in frame_images/, data/, or Videos/ - skipping")
                return None, None

        if result is not None:
            flow_values, confidence_values = result
            np.savez_compressed(cache_file, flow=flow_values, confidence=confidence_values)
        return result

    def _resolve_video_path(self, video_number):
        """Return the absolute path to the raw MP4 for video_number, or None."""
        video_name = _VIDEOS.get(video_number)
        if not video_name:
            return None

        for directory in self.video_search_dirs:
            candidate = os.path.join(directory, video_name)
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _radial_weight_map(h, w):
        """
        Build per-pixel radial/tangential unit vectors and a Gaussian annulus weight mask.

        The weight peaks at ~58% of the image radius (the pipe-wall band) and falls
        off toward the centre and edges.  Three corner/bottom regions are zeroed out
        to suppress water flow at the bottom and overlay artefacts in the top corners.

        Returns: rdx, rdy   radial unit vectors
                 tdx, tdy   tangential unit vectors (perpendicular to radial)
                 weight      per-pixel importance mask (0 outside useful annulus)
        """
        cx, cy = w / 2.0, h / 2.0
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = x - cx, y - cy
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        rdx, rdy = dx / r, dy / r
        tdx, tdy = -rdy, rdx

        radius = min(h, w) / 2.0
        inner = 0.18 * radius
        outer = 0.92 * radius
        annulus = (r >= inner) & (r <= outer)

        mid_radius = 0.58 * radius
        spread = 0.22 * radius + 1e-6
        weight = np.exp(-((r - mid_radius) ** 2) / (2.0 * spread ** 2)) * annulus.astype(np.float32)

        weight[int(h * 0.72):, :] = 0.0
        weight[:int(h * 0.16), :int(w * 0.24)] = 0.0
        weight[:int(h * 0.12), int(w * 0.76):] = 0.0
        return rdx, rdy, tdx, tdy, weight

    def _prepare_gray_frame(self, gray_frame):
        """
        Resize to 160x120, apply CLAHE, then Gaussian blur.

        Why:
          - fixed resolution keeps flow computation fast and consistent
          - CLAHE stabilises contrast in murky or dark frames
          - blur reduces pixel noise before dense optical flow
        """
        resized = cv2.resize(gray_frame, (_FLOW_W, _FLOW_H))
        equalized = self._clahe.apply(resized)
        return cv2.GaussianBlur(equalized, (3, 3), 0)

    def _estimate_signed_radial_flow(self, prev, curr, rdx, rdy, tdx, tdy, base_weight):
        """
        Reduce one frame pair to a single signed motion scalar and a confidence score.

        Steps:
          1. Build dynamic_valid mask: annulus weight AND both frames non-black (eroded).
          2. Compute DIS dense optical flow over the valid region.
          3. Subtract weighted-mean flow to remove global camera drift.
          4. Project flow onto radial and tangential axes.
          5. Weight pixels by texture (min Sobel gradient) x annulus x temporal stability.
          6. Keep only the top-textured pixels; cap extreme motion outliers.
          7. Trim radial values to 10th-90th percentile to remove remaining spikes.
          8. Compute weighted-average radial value  → signed_flow.
          9. Derive confidence from coverage, sign coherence, axis ratio, stability.

        Returns: (signed_flow, confidence)  both floats.
        """
        base_valid = base_weight > 0.0
        non_black = (prev > 10) & (curr > 10)
        non_black = cv2.erode(non_black.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        dynamic_valid = base_valid & non_black
        if np.count_nonzero(dynamic_valid) < 64:
            return 0.0, 0.0

        flow = self._dis.calc(prev, curr, None)

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        mean_x = np.average(flow_x[dynamic_valid], weights=base_weight[dynamic_valid])
        mean_y = np.average(flow_y[dynamic_valid], weights=base_weight[dynamic_valid])
        flow_x = flow_x - mean_x
        flow_y = flow_y - mean_y

        radial = flow_x * rdx + flow_y * rdy
        tangential = flow_x * tdx + flow_y * tdy

        grad_x = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3)
        gradient_prev = cv2.magnitude(grad_x, grad_y)
        grad_x_curr = cv2.Sobel(curr, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_curr = cv2.Sobel(curr, cv2.CV_32F, 0, 1, ksize=3)
        gradient_curr = cv2.magnitude(grad_x_curr, grad_y_curr)
        gradient = np.minimum(gradient_prev, gradient_curr)

        diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
        diff_scale = max(float(np.percentile(diff[dynamic_valid], 75)), 6.0)
        temporal_weight = np.exp(-np.square(diff / diff_scale))

        texture_weight = gradient * base_weight * temporal_weight
        gradient_values = texture_weight[dynamic_valid]
        if gradient_values.size == 0 or float(np.max(gradient_values)) <= 1e-6:
            return 0.0, 0.0

        gradient_threshold = np.percentile(gradient_values, 60)
        textured = dynamic_valid & (texture_weight >= gradient_threshold)
        if np.count_nonzero(textured) < 64:
            textured = dynamic_valid & (texture_weight > 0.0)
            if np.count_nonzero(textured) < 64:
                return 0.0, 0.0

        flow_mag = np.sqrt(flow_x ** 2 + flow_y ** 2)
        motion_cap = np.percentile(flow_mag[textured], 92)
        filtered = textured & (flow_mag <= motion_cap)
        if np.count_nonzero(filtered) < 64:
            filtered = textured

        values = radial[filtered]
        tangential_values = tangential[filtered]
        weights = texture_weight[filtered]
        if values.size == 0:
            return 0.0, 0.0

        lo, hi = np.percentile(values, [10, 90])
        trimmed = (values >= lo) & (values <= hi)
        if np.any(trimmed):
            values = values[trimmed]
            tangential_values = tangential_values[trimmed]
            weights = weights[trimmed]

        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            signed_flow = float(np.mean(values))
            radial_strength = float(np.mean(np.abs(values)))
            tangential_strength = float(np.mean(np.abs(tangential_values)))
        else:
            signed_flow = float(np.sum(values * weights) / weight_sum)
            radial_strength = float(np.sum(np.abs(values) * weights) / weight_sum)
            tangential_strength = float(np.sum(np.abs(tangential_values) * weights) / weight_sum)

        sign_coherence = abs(signed_flow) / (radial_strength + 1e-6)
        axis_ratio = radial_strength / (radial_strength + tangential_strength + 1e-6)
        coverage = np.count_nonzero(filtered) / max(np.count_nonzero(base_valid), 1)
        stability = float(np.mean(temporal_weight[filtered]))
        confidence = np.clip(
            (coverage ** 0.6)
            * (0.25 + 0.75 * sign_coherence)
            * (0.25 + 0.75 * axis_ratio)
            * (0.35 + 0.65 * stability),
            0.0,
            1.0,
        )
        return signed_flow, float(confidence)

    def _apply_confidence_gating(self, raw_flow, confidence):
        """
        Conservative spike repair pass.  Does NOT globally reshape the signal.

        For steps that are both low-confidence AND a large residual from the local
        interpolated trend, replace the value with the interpolated estimate.
        High-confidence steps and small residuals are always left untouched.
        """
        raw_flow = np.asarray(raw_flow, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        if raw_flow.size < 5 or confidence.shape != raw_flow.shape:
            return raw_flow

        confidence = np.clip(confidence, 0.0, 1.0)
        if float(np.max(confidence)) <= 1e-6:
            return raw_flow

        x = np.arange(raw_flow.size, dtype=float)
        spike_threshold = max(0.08, float(np.percentile(confidence, 10)))
        anchor_threshold = max(0.20, float(np.percentile(confidence, 45)))
        anchors = confidence >= anchor_threshold
        if np.count_nonzero(anchors) < 3:
            return raw_flow

        interpolated = np.interp(x, x[anchors], raw_flow[anchors])
        local_trend = uniform_filter1d(
            interpolated,
            size=self._window_from_length(len(raw_flow), 30, 7),
            mode="nearest",
        )
        residual = np.abs(raw_flow - local_trend)
        residual_threshold = max(float(np.percentile(residual, 90)), float(np.median(residual)) * 4.0 + 1e-6)
        hard_replace = (confidence < spike_threshold) & (residual > residual_threshold)
        if not np.any(hard_replace):
            return raw_flow

        gated = raw_flow.copy()
        gated[hard_replace] = interpolated[hard_replace]
        return gated

    def _stabilize_flow_signal(self, raw_flow, window, percentile, threshold_multiplier):
        """
        Smooth, median-centre, and threshold the raw flow signal.

        1. Apply moving-average smoothing of given window size.
        2. Subtract the median so the baseline is zero.
        3. Compute a percentile-based noise floor and zero out values below it.

        Returns: (smoothed_signal, threshold)
        """
        smoothed = uniform_filter1d(raw_flow.astype(float), size=window)
        smoothed = smoothed - np.median(smoothed)
        magnitude = np.abs(smoothed)
        threshold = max(float(np.percentile(magnitude, percentile)) * threshold_multiplier, 1e-6)
        smoothed[magnitude < threshold] = 0.0
        return smoothed, threshold

    def _estimate_turning_point(self, smoothed, threshold):
        """
        Find the frame where the probe reverses direction.

        For every candidate turn frame the scorer measures:
          - forward motion accumulated before the turn  (rewarded)
          - backward motion accumulated after the turn  (rewarded)
          - wrong-sign motion before the turn           (penalised)
          - wrong-sign motion after the turn            (penalised)
          - imbalance between outbound and inbound halves (penalised)

        Returns the frame index with the highest score.
        """
        positive = np.maximum(smoothed - threshold, 0.0)
        negative = np.maximum(-smoothed - threshold, 0.0)

        pos_cum = np.concatenate([[0.0], np.cumsum(positive)])
        neg_cum = np.concatenate([[0.0], np.cumsum(negative)])

        total_pos = pos_cum[-1] + 1e-6
        total_neg = neg_cum[-1] + 1e-6

        start = max(10, len(pos_cum) // 7)
        stop = min(len(pos_cum) - 10, (6 * len(pos_cum)) // 7)
        best_score = -np.inf
        best_turn = len(pos_cum) // 2

        for turn in range(start, stop):
            outbound = pos_cum[turn]
            inbound = neg_cum[-1] - neg_cum[turn]
            wrong_before = neg_cum[turn]
            wrong_after = pos_cum[-1] - pos_cum[turn]

            balance_penalty = abs((outbound / total_pos) - (inbound / total_neg))
            score = outbound + inbound
            score -= 0.75 * wrong_before
            score -= 0.75 * wrong_after
            score -= 0.25 * balance_penalty * (outbound + inbound)

            if score > best_score:
                best_score = score
                best_turn = turn

        return int(best_turn)

    def _integrate_path(
        self,
        smoothed,
        turning_point,
        threshold,
        channel_length,
        gamma_outbound=1.0,
        gamma_inbound=1.0,
        smooth_window=None,
    ):
        """
        Convert the stabilised flow signal into a physical distance curve.

        1. Separate positive (outbound) and negative (inbound) components;
           apply gamma exponents to compress large spikes.
        2. Cumulative-sum both components independently.
        3. Scale outbound cumsum to [0, channel_length] up to the turn frame.
        4. Scale inbound cumsum to [channel_length, 0] from the turn frame onward.
        5. Smooth the joined curve and enforce a unimodal shape.
        """
        positive = np.maximum(smoothed - threshold, 0.0) ** gamma_outbound
        negative = np.maximum(-smoothed - threshold, 0.0) ** gamma_inbound

        pos_cum = np.concatenate([[0.0], np.cumsum(positive)])
        neg_cum = np.concatenate([[0.0], np.cumsum(negative)])

        outbound_total = max(pos_cum[turning_point], 1e-6)
        inbound_total = max(neg_cum[-1] - neg_cum[turning_point], 1e-6)

        movement_path = np.empty(len(pos_cum), dtype=float)
        movement_path[:turning_point + 1] = pos_cum[:turning_point + 1] * (channel_length / outbound_total)

        inbound_progress = neg_cum[turning_point:] - neg_cum[turning_point]
        movement_path[turning_point:] = channel_length - inbound_progress * (channel_length / inbound_total)

        if smooth_window is None:
            smooth_window = self._window_from_length(len(movement_path), 40, 19)
        movement_path = uniform_filter1d(movement_path, size=smooth_window, mode="nearest")
        movement_path = np.clip(movement_path, 0.0, channel_length)
        return self._enforce_unimodal(movement_path, turning_point, channel_length)

    def _enforce_unimodal(self, movement_path, turning_point, channel_length):
        """
        Force the path to increase monotonically to the turn then decrease monotonically.

        Also pins: path[0] = 0, path[turn] = channel_length, path[-1] = 0.
        """
        path = np.asarray(movement_path, dtype=float).copy()
        turn = int(np.clip(turning_point, 1, max(len(path) - 2, 1)))
        path[:turn + 1] = np.maximum.accumulate(path[:turn + 1])
        path[turn:] = np.minimum.accumulate(path[turn:])
        path = np.clip(path, 0.0, channel_length)
        path[0] = 0.0
        path[turn] = channel_length
        path[-1] = 0.0
        return path

    def _path_to_direction(self, movement_path, turning_point, channel_length):
        """
        Derive per-frame direction labels (+1 / -1 / 0) from the final distance path.

        1. Lightly smooth the path to reduce label chatter.
        2. Compute framewise finite differences.
        3. Threshold: +1 if rising, -1 if falling, 0 if flat.
        4. Zero out a small quiet zone around the turning point.
        """
        smooth_window = max(3, min(13, len(movement_path) // 60))
        if smooth_window % 2 == 0:
            smooth_window += 1
        smoothed_path = uniform_filter1d(np.asarray(movement_path, dtype=float), size=smooth_window, mode="nearest")
        delta = np.diff(smoothed_path, prepend=smoothed_path[0])

        threshold = max(channel_length / max(len(movement_path) * 15.0, 1.0), 1e-4)
        direction = np.zeros(len(movement_path), dtype=float)
        direction[delta > threshold] = 1.0
        direction[delta < -threshold] = -1.0

        quiet_radius = max(2, len(direction) // 200)
        start = max(0, int(turning_point) - quiet_radius)
        stop = min(len(direction), int(turning_point) + quiet_radius + 1)
        direction[start:stop] = 0.0
        return direction

    def _flow_from_mp4(self, video_path, video_number):
        """
        Compute flow and confidence arrays directly from a raw MP4 file.

        Skips the first _FIRST_KEPT_FRAME frames (header / stabilisation period)
        and stops at _LAST_KEPT_FRAME if defined.  Each frame pair is preprocessed
        and passed to _estimate_signed_radial_flow.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        first_kept = _FIRST_KEPT_FRAME.get(video_number, 0)
        last_kept = _LAST_KEPT_FRAME.get(video_number, None)

        for _ in range(first_kept):
            if not cap.grab():
                cap.release()
                return None

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        prev = self._prepare_gray_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rdx, rdy, tdx, tdy, weight = self._radial_weight_map(_FLOW_H, _FLOW_W)
        flow_values = []
        confidence_values = []
        idx = first_kept + 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if last_kept is not None and idx > last_kept:
                break

            curr = self._prepare_gray_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            flow_value, confidence_value = self._estimate_signed_radial_flow(prev, curr, rdx, rdy, tdx, tdy, weight)
            flow_values.append(flow_value)
            confidence_values.append(confidence_value)
            prev = curr
            idx += 1

        cap.release()
        if not flow_values:
            return None
        return np.asarray(flow_values, dtype=float), np.asarray(confidence_values, dtype=float)

    def _flow_from_images(self, frame_dir):
        """
        Compute flow and confidence arrays from a folder of numbered PNG frames.

        Processes numerically sorted PNGs in frame_dir.  Each consecutive pair is
        preprocessed and passed to _estimate_signed_radial_flow.
        """
        files = self._frame_image_files(frame_dir)
        if len(files) < 2:
            return None

        first_img = cv2.imread(os.path.join(frame_dir, files[0]), cv2.IMREAD_GRAYSCALE)
        if first_img is None:
            return None

        prev = self._prepare_gray_frame(first_img)
        rdx, rdy, tdx, tdy, weight = self._radial_weight_map(_FLOW_H, _FLOW_W)
        flow_values = []
        confidence_values = []

        for file_name in files[1:]:
            image = cv2.imread(os.path.join(frame_dir, file_name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            curr = self._prepare_gray_frame(image)
            flow_value, confidence_value = self._estimate_signed_radial_flow(prev, curr, rdx, rdy, tdx, tdy, weight)
            flow_values.append(flow_value)
            confidence_values.append(confidence_value)
            prev = curr

        if not flow_values:
            return None
        return np.asarray(flow_values, dtype=float), np.asarray(confidence_values, dtype=float)

    def execute_estimations(self):
        """
        Framework entry point.  Runs estimation for one or all videos.

        When test_all_videos=True iterates over every digit-named sub-folder
        inside frame_images/.  Otherwise runs only video_num_to_test.
        Results are stored in self.calculated_movement_paths.
        """
        if self.test_all_videos:
            if not os.path.exists(self.path_to_videos):
                raise FileNotFoundError(f"The folder '{self.path_to_videos}' does not exist.")
            for entry in os.listdir(self.path_to_videos):
                if entry.isdigit():
                    self._run_single(int(entry))
        else:
            self._run_single(self.video_num_to_test)

    def _run_single(self, video_number):
        """
        Estimate one video and store the result as a MovementPath object.

        Looks up the known channel length, calls
        calculate_movement_path_and_turning_point, wraps the outputs in a
        MovementPath, and writes it to self.calculated_movement_paths.
        """
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


if __name__ == "__main__":
    print("MovementPathEstimator.py defines the estimator class.")
    print("For a quick sanity check, run: python ..\\quick_check.py")
    print("For one labeled video with optional plot, run: python ..\\estimate_single_movement_path.py [video_num]")
    print("For plot plus frame preview, run: python ..\\run_baseline.py [video_num]")
    print("For video playback plus live path plot, run: python visualize.py [video_num] [speed]")
