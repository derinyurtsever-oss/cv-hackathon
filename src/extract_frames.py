"""
extract_frames.py
-----------------
Extract frames from video files at a given FPS or frame interval.
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float | None = None,
    every_n_frames: int | None = None,
    max_frames: int | None = None,
    img_format: str = "jpg",
) -> list[str]:
    """
    Extract frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    output_dir : str
        Directory where extracted frames will be saved.
    fps : float, optional
        Target extraction rate in frames per second.
        If None and every_n_frames is None, every frame is extracted.
    every_n_frames : int, optional
        Save one frame every N frames (overrides fps if both given).
    max_frames : int, optional
        Stop after extracting this many frames.
    img_format : str
        Output image format, e.g. 'jpg', 'png'.

    Returns
    -------
    list[str]
        Sorted list of saved frame paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if every_n_frames is None:
        if fps is not None and video_fps > 0:
            every_n_frames = max(1, round(video_fps / fps))
        else:
            every_n_frames = 1

    saved_paths = []
    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc=f"Extracting {Path(video_path).name}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n_frames == 0:
                out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.{img_format}")
                cv2.imwrite(out_path, frame)
                saved_paths.append(out_path)
                saved_count += 1
                if max_frames is not None and saved_count >= max_frames:
                    break
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}'")
    return sorted(saved_paths)


def extract_frames_from_dir(
    video_dir: str,
    output_root: str,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    **kwargs,
) -> dict[str, list[str]]:
    """
    Extract frames from all videos in a directory.

    Returns a dict mapping video filename -> list of saved frame paths.
    """
    results = {}
    video_dir = Path(video_dir)
    for video_file in sorted(video_dir.iterdir()):
        if video_file.suffix.lower() in extensions:
            out_dir = os.path.join(output_root, video_file.stem)
            results[video_file.name] = extract_frames(str(video_file), out_dir, **kwargs)
    return results
