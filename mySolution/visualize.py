"""
Visualization: video frame (top) + real-time distance plot (bottom).

Usage (from project root):
    python mySolution/visualize.py [video_num] [speed]

    video_num : 1-11  (default 1)
    speed     : frames to advance per display tick (default 5)
                press +/- while running to adjust live
                press q or ESC to quit
"""
import sys
import os
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mySolution.MovementPathEstimator import MovementPathEstimator, _VIDEOS, _FIRST_KEPT_FRAME, _LAST_KEPT_FRAME
from EstimationRater import EstimationRater
from MovementPath import MovementPath

DISPLAY_W = 1024
VIDEO_H   = 576
PLOT_H    = 260


class PlotRenderer:
    """Renders the distance plot to a BGR numpy image (reuses the figure for speed)."""

    def __init__(self, n_frames: int, channel_length: float, has_measured: bool):
        self.fig = Figure(figsize=(DISPLAY_W / 100, PLOT_H / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_subplot(111)
        self.ax = ax

        ax.set_facecolor("#12122a")
        self.fig.patch.set_facecolor("#12122a")

        self.est_line, = ax.plot([], [], color="#4fc3f7", linewidth=2, label="Schätzung")
        self.meas_line = None
        if has_measured:
            self.meas_line, = ax.plot([], [], color="#ffb74d", linewidth=2, label="Messung")

        self.vline = ax.axvline(x=0, color="#ef5350", linewidth=1.5, alpha=0.85)
        ax.set_xlim(0, n_frames)
        ax.set_ylim(-1, channel_length * 1.08)
        ax.set_xlabel("Frame", color="#bbbbbb", fontsize=9)
        ax.set_ylabel("Distanz [m]", color="#bbbbbb", fontsize=9)
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.legend(loc="upper right", facecolor="#1e1e3e",
                  labelcolor="#cccccc", edgecolor="#444466", fontsize=9)
        self.fig.tight_layout(pad=1.0)

    def render(self, estimated: np.ndarray, measured, frame_idx: int) -> np.ndarray:
        x = np.arange(frame_idx + 1)
        self.est_line.set_data(x, estimated[: frame_idx + 1])
        if self.meas_line is not None and measured is not None:
            ml = min(frame_idx + 1, len(measured))
            self.meas_line.set_data(np.arange(ml), measured[:ml])
        self.vline.set_xdata([frame_idx, frame_idx])
        self.canvas.draw()
        img = np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8).copy()
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def main():
    video_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    speed     = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    data_dirs = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "Videos"),
        os.path.join(project_root, "videos"),
    ]

    channel_lengths = np.load("channel_lengths.npy")
    channel_length  = float(channel_lengths[video_num - 1])

    print(f"Estimating movement path for video {video_num}...")
    estimator = MovementPathEstimator(video_num, False)
    estimator.execute_estimations()
    estimated = estimator.calculated_movement_paths[video_num].movement_path

    gt_path  = f"distance_labels/{video_num}.npy"
    measured = np.load(gt_path) if os.path.exists(gt_path) else None
    if measured is not None:
        rater = EstimationRater(
            {0: estimator.calculated_movement_paths[video_num]},
            {0: MovementPath(video_num, measured)},
            do_print=True,
            do_plot=False,
            estimator_name=type(estimator).__name__,
            video_num=video_num,
        )
        rater.rate()
    else:
        print(f"No ground-truth label available for video {video_num}.")

    n_frames = len(estimated)
    renderer = PlotRenderer(n_frames, channel_length, measured is not None)

    # Prefer pre-extracted PNGs (faster) over live MP4 decode
    frame_dir = os.path.join(project_root, "frame_images", str(video_num))
    use_images = os.path.exists(frame_dir)

    if use_images:
        files = sorted(
            [f for f in os.listdir(frame_dir) if f.endswith(".png")],
            key=lambda f: int(os.path.splitext(f)[0]),
        )
        cap = None
    else:
        vid_path = None
        for data_dir in data_dirs:
            candidate = os.path.join(data_dir, _VIDEOS.get(video_num, ""))
            if os.path.exists(candidate):
                vid_path = candidate
                break
        if vid_path is None:
            raise FileNotFoundError(
                f"Could not find video {video_num} in frame_images/, data/, Videos/, or videos/."
            )
        cap = cv2.VideoCapture(vid_path)
        for _ in range(_FIRST_KEPT_FRAME.get(video_num, 0)):
            cap.grab()

    win = f"Video {video_num}  |  channel {channel_length:.1f} m  |  [+/-] speed  [q] quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_W, VIDEO_H + PLOT_H)

    frame_idx = 0
    last_kept = _LAST_KEPT_FRAME.get(video_num, None)

    while frame_idx < n_frames:
        # ── Read frame ──────────────────────────────────────────────────
        if use_images:
            if frame_idx >= len(files):
                break
            bgr = cv2.imread(os.path.join(frame_dir, files[frame_idx]))
            if bgr is None:
                frame_idx += speed
                continue
        else:
            ret, bgr = cap.read()
            if not ret:
                break
            abs_idx = _FIRST_KEPT_FRAME.get(video_num, 0) + frame_idx
            if last_kept is not None and abs_idx > last_kept:
                break

        # ── Video panel ─────────────────────────────────────────────────
        video_panel = cv2.resize(bgr, (DISPLAY_W, VIDEO_H))

        dist = float(estimated[min(frame_idx, n_frames - 1)])
        label = f"{dist:.1f} m"

        # Black background pill behind the text
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        pad = 10
        cv2.rectangle(video_panel, (12 - pad, 12 - pad), (12 + tw + pad, 12 + th + bl + pad),
                      (0, 0, 0), -1)
        cv2.putText(video_panel, label, (12, 12 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

        # ── Plot panel ──────────────────────────────────────────────────
        plot_panel = renderer.render(estimated, measured, frame_idx)

        # ── Stack & display ─────────────────────────────────────────────
        composite = np.vstack([video_panel, plot_panel])
        cv2.imshow(win, composite)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("+"), ord("=")):
            speed = min(speed + 1, 50)
            print(f"Speed: {speed}")
        elif key == ord("-"):
            speed = max(speed - 1, 1)
            print(f"Speed: {speed}")

        frame_idx += speed

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
