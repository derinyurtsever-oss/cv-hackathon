import numpy as np
import os
import cv2
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

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

# Resize every frame to this resolution before computing optical flow
_FLOW_W, _FLOW_H = 160, 120

# Cache directory for computed flow arrays
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flow_cache')


class MovementPathEstimator:

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.data_dir = os.path.normpath(os.path.join(self.current_folder, '..', 'data'))

        # DIS flow: Dense Inverse Search — fast and accurate
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_FAST)
        self._dis.setUseSpatialPropagation(True)

        self.calculated_movement_paths = {}

    # ------------------------------------------------------------------ #
    #  Core estimator                                                      #
    # ------------------------------------------------------------------ #

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        raw_flow = self._flow_for_video(video_number)

        if raw_flow is None or len(raw_flow) < 2:
            return (
                np.array([0.0, channel_length, 0.0]),
                1.0,
                np.array([0.0, 1.0, -1.0]),
            )

        # Smooth → cumsum → turning point
        # Gaussian for TP: sharper cumsum peak, better argmax accuracy.
        # Uniform for paths/DTW: preserves the profile shapes DTW relies on.
        window      = max(5, min(51, len(raw_flow) // 20))
        sm_gaussian = gaussian_filter1d(raw_flow.astype(float), sigma=max(1.0, window / 3.46))
        smoothed    = uniform_filter1d(raw_flow.astype(float), size=window)
        tp          = int(np.argmax(np.concatenate([[0.0], np.cumsum(sm_gaussian)])))
        cum         = np.concatenate([[0.0], np.cumsum(smoothed)])

        # Forward path: two-phase cumsum scaling
        fwd = float(cum[tp])
        ret = float(cum[tp] - cum[-1])
        fwd_scale = channel_length / fwd if fwd > 1e-9 else 1.0
        ret_scale = channel_length / ret if ret > 0.05 * max(fwd, 1e-9) else fwd_scale

        movement_path = np.empty(len(cum), dtype=np.float32)
        movement_path[:tp + 1] = cum[:tp + 1] * fwd_scale
        movement_path[:tp + 1] = np.clip(movement_path[:tp + 1], 0.0, channel_length)
        movement_path[:tp + 1] = np.maximum.accumulate(movement_path[:tp + 1])

        # Return path: DTW → mirrored-forward fallback → two-phase
        ret_path = self._dtw_return_path(smoothed, movement_path[:tp + 1], tp, channel_length)
        if ret_path is None:
            ret_path = self._mirrored_return_path(movement_path[:tp + 1], len(cum) - 1 - tp, channel_length)
        if ret_path is not None:
            movement_path[tp:] = ret_path
        else:
            movement_path[tp:] = channel_length - (cum[tp] - cum[tp:]) * ret_scale
            movement_path[tp:] = np.clip(movement_path[tp:], 0.0, channel_length)

        movement_path[tp:] = np.minimum.accumulate(movement_path[tp:])

        diffs     = np.diff(movement_path, prepend=movement_path[0])
        eps       = max(0.01, channel_length * 1e-4)
        direction = np.sign(diffs)
        direction[np.abs(diffs) < eps] = 0.0

        return movement_path, float(tp), direction.astype(float)

    # ------------------------------------------------------------------ #
    #  DTW forward/return flow profile alignment                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dtw_return_path(smoothed, fwd_path, tp, channel_length):
        """
        Align the forward flow-magnitude profile with the time-reversed return
        profile using DTW.  The warp path transfers known forward positions onto
        the return timeline, correcting for non-uniform probe speed.

        smoothed  : uniform-filtered flow signal, length N
        fwd_path  : forward movement path, shape (tp+1,)
        tp        : turning-point index
        Returns   : return path array of length (N - tp + 1), or None on failure.
        """
        N     = len(smoothed)        # number of flow transitions (= len(raw_flow))
        n_ret = N - tp
        if tp < 10 or n_ret < 10:
            return None

        # Flow magnitude profiles
        fwd_mag = np.abs(smoothed[:tp])           # length tp
        ret_mag = np.abs(smoothed[tp:])[::-1]     # reversed, length n_ret

        # Downsample both to length L so DTW is fast
        L   = min(250, tp, n_ret)
        t_f = np.linspace(0, tp - 1,    L)
        t_r = np.linspace(0, n_ret - 1, L)
        a   = np.interp(t_f, np.arange(tp),    fwd_mag).astype(np.float64)
        b   = np.interp(t_r, np.arange(n_ret), ret_mag).astype(np.float64)

        # Normalise so shape (not amplitude) drives the alignment
        a /= (a.mean() + 1e-8)
        b /= (b.mean() + 1e-8)

        # ---- DTW DP ----
        D = np.full((L + 1, L + 1), np.inf, dtype=np.float64)
        D[0, 0] = 0.0
        for i in range(1, L + 1):
            for j in range(1, L + 1):
                cost    = (a[i - 1] - b[j - 1]) ** 2
                D[i, j] = cost + min(D[i-1, j-1], D[i-1, j], D[i, j-1])

        # ---- Traceback ----
        i, j   = L, L
        pi, pj = [], []
        while i > 0 or j > 0:
            pi.append(i - 1)
            pj.append(j - 1)
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                best = min(D[i-1, j-1], D[i-1, j], D[i, j-1])
                if best == D[i-1, j-1]:
                    i -= 1; j -= 1
                elif best == D[i-1, j]:
                    i -= 1
                else:
                    j -= 1
        pi = np.array(pi[::-1])   # downsampled fwd indices   (0 … L-1)
        pj = np.array(pj[::-1])   # downsampled ret indices   (0 … L-1)

        # Map downsampled indices back to original frame indices
        fwd_orig = np.interp(pi.astype(float), np.arange(L), t_f).astype(int)
        ret_orig = np.interp(pj.astype(float), np.arange(L), t_r).astype(int)
        # ret_orig is in reversed-return space; flip to real return index
        ret_real = n_ret - 1 - ret_orig

        # For each warp pair, fwd frame fwd_orig[k] ↔ return frame ret_real[k]
        # fwd_path[fwd_orig + 1] is the known position at that forward frame
        fwd_pos = fwd_path[np.clip(fwd_orig + 1, 0, len(fwd_path) - 1)]

        # Build dense return position array by interpolating anchor pairs
        # Sort by ret_real to make np.interp happy
        order       = np.argsort(ret_real)
        ret_sorted  = ret_real[order]
        pos_sorted  = fwd_pos[order]

        # Deduplicate (multiple warp steps can map to same return frame)
        _, unique   = np.unique(ret_sorted, return_index=True)
        ret_anch    = ret_sorted[unique]
        pos_anch    = pos_sorted[unique]

        ret_positions = np.interp(np.arange(n_ret), ret_anch, pos_anch)

        # Smooth and enforce monotone decrease
        ret_positions = uniform_filter1d(ret_positions, size=max(5, n_ret // 20))
        ret_positions = np.clip(ret_positions, 0.0, channel_length)
        ret_positions = np.minimum.accumulate(ret_positions)

        # DTW cost gate: too similar (cost < 0.05) or too chaotic (cost > 0.30)
        dtw_cost = D[L, L] / max(1, 2 * L - 1)
        if not (0.05 < dtw_cost < 0.30):
            return None

        # Geometry gate: path must start near channel_length and end near 0
        if ret_positions[0] < channel_length * 0.50:
            return None
        if ret_positions[-1] > channel_length * 0.25:
            return None

        ret_full    = np.empty(n_ret + 1, dtype=np.float32)
        ret_full[0] = channel_length
        ret_full[1:] = ret_positions
        return ret_full

    @staticmethod
    def _mirrored_return_path(fwd_path, n_ret, channel_length):
        """
        Use the time-reversed forward speed profile as the return path template.
        Works when the operator moves the probe at comparable speed in both directions
        (forward and return flow profiles are too similar for DTW to add value).
        """
        tp1 = len(fwd_path)          # tp + 1 points
        if tp1 < 10 or n_ret < 10:
            return None

        # Normalise forward path 0→1, mirror it 1→0, resample to return length
        fwd_norm    = fwd_path / (channel_length + 1e-9)
        mirrored    = (1.0 - fwd_norm[::-1]).astype(np.float64)
        t_src       = np.linspace(0.0, 1.0, tp1)
        t_dst       = np.linspace(0.0, 1.0, n_ret + 1)
        ret_path    = np.interp(t_dst, t_src, mirrored) * channel_length

        # Geometry gate
        if ret_path[0] < channel_length * 0.85:
            return None
        if ret_path[-1] > channel_length * 0.15:
            return None

        ret_path = np.minimum.accumulate(ret_path)
        return ret_path.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Optical flow pipeline                                               #
    # ------------------------------------------------------------------ #

    def _flow_for_video(self, video_number):
        """Return flow array — cached to disk."""
        os.makedirs(_CACHE_DIR, exist_ok=True)
        flow_file = os.path.join(_CACHE_DIR, f'video_{video_number}.npy')

        if os.path.exists(flow_file):
            return np.load(flow_file)

        frame_dir = os.path.join(self.path_to_videos, str(video_number))
        if os.path.exists(frame_dir) and any(f.endswith('.png') for f in os.listdir(frame_dir)):
            flow = self._flow_from_images(frame_dir)
        else:
            vid_name = _VIDEOS.get(video_number)
            vid_path = os.path.join(self.data_dir, vid_name) if vid_name else None
            if vid_path and os.path.exists(vid_path):
                flow = self._flow_from_mp4(vid_path, video_number)
            else:
                print(f"  [video {video_number}] no source found in frame_images/ or data/ — skipping")
                return None

        if flow is not None:
            np.save(flow_file, flow)
        return flow

    @staticmethod
    def _radial_weight_map(h, w):
        """Per-pixel radial unit vectors and weights for an h×w frame.
        Weight rises linearly from centre; bottom 30% masked to exclude water flow."""
        cx, cy = w / 2.0, h / 2.0
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = x - cx, y - cy
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        rdx, rdy = dx / r, dy / r
        weight = np.clip(r / (min(h, w) / 2.0), 0.0, 1.0)
        weight[int(h * 0.70):, :] = 0.0
        return rdx, rdy, weight

    def _flow_from_mp4(self, video_path, video_number):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        first_kept = _FIRST_KEPT_FRAME.get(video_number, 0)
        last_kept  = _LAST_KEPT_FRAME.get(video_number, None)

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
            fv = dis.calc(prev, curr, None)
            radial = fv[..., 0] * rdx + fv[..., 1] * rdy
            valid  = radial[weight > 0.1]
            flow_values.append(float(np.median(valid)) if valid.size else 0.0)
            prev = curr
            idx += 1

        cap.release()
        if not flow_values:
            return None
        return np.array(flow_values, dtype=float)

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

        dis   = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        prev  = cv2.resize(first_img, (_FLOW_W, _FLOW_H))
        rdx, rdy, weight = self._radial_weight_map(_FLOW_H, _FLOW_W)
        flow_values = []

        for fname in files[1:]:
            img = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            curr = cv2.resize(img, (_FLOW_W, _FLOW_H))
            fv = dis.calc(prev, curr, None)
            radial = fv[..., 0] * rdx + fv[..., 1] * rdy
            valid  = radial[weight > 0.1]
            flow_values.append(float(np.median(valid)) if valid.size else 0.0)
            prev = curr

        if not flow_values:
            return None
        return np.array(flow_values, dtype=float)

    # ------------------------------------------------------------------ #
    #  Framework boilerplate                                               #
    # ------------------------------------------------------------------ #

    def execute_estimations(self):
        if self.test_all_videos:
            available = []
            for n in range(1, 12):
                vname = _VIDEOS.get(n, '')
                has_mp4    = os.path.exists(os.path.join(self.data_dir, vname))
                fdir       = os.path.join(self.path_to_videos, str(n))
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
