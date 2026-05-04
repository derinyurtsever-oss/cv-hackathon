"""
main.py
-------
Entry point for the CV Hackathon pipeline.

Commands
--------
extract   – Extract frames from video(s)
match     – Feature matching between two images or across a frame directory
track     – Optical flow tracking across a frame directory
compare   – Image similarity metrics between two images

Examples
--------
python main.py extract  --input data/videos/ --output data/frames/ --fps 5
python main.py match    --img1 data/frames/vid1/frame_000000.jpg --img2 data/frames/vid1/frame_000100.jpg
python main.py track    --frames data/frames/vid1/ --method lk
python main.py compare  --img1 data/frames/vid1/frame_000000.jpg --img2 data/frames/vid2/frame_000000.jpg
python main.py pipeline --video data/videos/inspection.mp4 --output output/annotated.mp4
python main.py pipeline --video data/videos/inspection.mp4 --output output/annotated.mp4 --gt-distance 45.5 --every-n 2
"""

import argparse
import os
import sys

import cv2
import numpy as np


def cmd_extract(args):
    from src.extract_frames import extract_frames, extract_frames_from_dir

    if os.path.isdir(args.input):
        results = extract_frames_from_dir(
            args.input, args.output,
            fps=args.fps,
            every_n_frames=args.every_n,
            max_frames=args.max_frames,
        )
        for vid, frames in results.items():
            print(f"  {vid}: {len(frames)} frames")
    else:
        frames = extract_frames(
            args.input, args.output,
            fps=args.fps,
            every_n_frames=args.every_n,
            max_frames=args.max_frames,
        )
        print(f"Extracted {len(frames)} frames.")


def cmd_match(args):
    from src.feature_matching import compare_images
    from src.utils import load_image, draw_matches_side_by_side, save_image, resize_if_large

    img1 = load_image(args.img1)
    img2 = load_image(args.img2)
    img1 = resize_if_large(img1, max_dim=args.max_dim)
    img2 = resize_if_large(img2, max_dim=args.max_dim)

    result = compare_images(img1, img2, method=args.detector, lowe_ratio=args.ratio)
    print(f"Keypoints: {len(result.keypoints1)} / {len(result.keypoints2)}")
    print(f"Good matches: {len(result.good_matches)}")
    print(f"Match ratio: {result.match_ratio:.4f}")
    if result.homography is not None:
        inliers = int(result.inlier_mask.sum()) if result.inlier_mask is not None else 0
        print(f"Homography inliers: {inliers}")

    if args.save:
        vis = draw_matches_side_by_side(img1, result.keypoints1, img2, result.keypoints2, result.good_matches)
        save_image(vis, args.save)
        print(f"Saved visualisation to '{args.save}'")


def cmd_track(args):
    from src.utils import load_images_from_dir, draw_optical_flow, save_image
    from src.tracking import track_video_frames

    print(f"Loading frames from '{args.frames}' ...")
    pairs = load_images_from_dir(args.frames)
    frames = [img for _, img in pairs]
    names = [name for name, _ in pairs]

    if len(frames) < 2:
        print("Need at least 2 frames.")
        sys.exit(1)

    print(f"Tracking {len(frames)} frames with method='{args.method}' ...")
    results = track_video_frames(frames, method=args.method)

    displacements = []
    for i, r in enumerate(results):
        if args.method == "lk":
            d = r.mean_displacement
        else:
            d = float(np.mean(r.magnitude))
        displacements.append(d)
        if (i + 1) % 50 == 0:
            print(f"  Frame {i + 1}/{len(results)}: mean displacement={d:.2f}px")

    avg = np.mean(displacements)
    print(f"\nAverage displacement over all frames: {avg:.2f} px")

    if args.save_dir and args.method == "lk":
        os.makedirs(args.save_dir, exist_ok=True)
        for i, (r, name) in enumerate(zip(results, names[1:])):
            vis = draw_optical_flow(
                frames[i + 1],
                r.prev_pts if len(r.prev_pts) > 0 else np.empty((0, 1, 2)),
                r.curr_pts if len(r.curr_pts) > 0 else np.empty((0, 1, 2)),
                r.status,
            )
            save_image(vis, os.path.join(args.save_dir, name))
        print(f"Saved flow visualisations to '{args.save_dir}'")


def cmd_pipeline(args):
    from src.visual_odometry import estimate_distance_from_video, calibrate_scale
    from src.annotate_video import annotate_video, save_distance_plot

    print(f"=== Pipe Inspection Pipeline ===")
    print(f"Video : {args.video}")
    print(f"Output: {args.output}")

    # --- Pass 1: run odometry with scale=1.0 to get raw pixel displacements ---
    print("\n[1/3] Estimating camera motion (scale=1.0) ...")
    raw_result = estimate_distance_from_video(
        args.video,
        scale_m_per_px=1.0,
        every_n_frames=args.every_n,
        max_frames=args.max_frames,
        water_mask_fraction=args.water_mask,
    )

    # --- Calibrate scale if ground-truth distance provided ---
    if args.gt_distance is not None:
        from src.visual_odometry import calibrate_scale
        scale = calibrate_scale(raw_result.pixel_displacements, args.gt_distance)
        print(f"\n[2/3] Calibrated scale: {scale:.6f} m/px  (gt={args.gt_distance} m)")
    else:
        scale = args.scale
        print(f"\n[2/3] Using manual scale: {scale} m/px  (pass --gt-distance to auto-calibrate)")

    # --- Pass 2: re-run with calibrated scale (fast: just multiply) ---
    for f in raw_result.frames:
        f.distance_delta_m = f.pixel_displacement * scale
    cumulative = 0.0
    for f in raw_result.frames:
        cumulative += f.distance_delta_m
        f.cumulative_distance_m = cumulative
    raw_result.scale_m_per_px = scale

    print(f"     Total estimated distance: {raw_result.total_distance_m:.3f} m")

    # --- Annotate video ---
    print("\n[3/3] Writing annotated video ...")
    annotate_video(
        args.video,
        raw_result,
        args.output,
        every_n_frames=args.every_n,
    )

    # --- Optional distance plot ---
    if args.plot:
        plot_path = args.plot
        save_distance_plot(raw_result, plot_path)

    print(f"\nDone. Annotated video saved to '{args.output}'")


def cmd_compare(args):
    from src.comparison import full_comparison, difference_image
    from src.utils import load_image, resize_if_large, save_image

    img1 = load_image(args.img1)
    img2 = load_image(args.img2)
    img1 = resize_if_large(img1, max_dim=args.max_dim)
    img2 = resize_if_large(img2, max_dim=args.max_dim)

    result = full_comparison(img1, img2)
    print(result.summary())

    if args.save_diff:
        diff = difference_image(img1, img2, amplify=args.amplify)
        save_image(diff, args.save_diff)
        print(f"Saved difference image to '{args.save_diff}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CV Hackathon Pipeline")
    sub = parser.add_subparsers(dest="command")

    p_ext = sub.add_parser("extract", help="Extract frames from video(s)")
    p_ext.add_argument("--input", required=True, help="Video file or directory of videos")
    p_ext.add_argument("--output", required=True, help="Output directory for frames")
    p_ext.add_argument("--fps", type=float, default=None, help="Target extraction FPS")
    p_ext.add_argument("--every-n", type=int, default=None, dest="every_n", help="Save every N-th frame")
    p_ext.add_argument("--max-frames", type=int, default=None, dest="max_frames")

    p_match = sub.add_parser("match", help="Feature match two images")
    p_match.add_argument("--img1", required=True)
    p_match.add_argument("--img2", required=True)
    p_match.add_argument("--detector", default="orb", choices=["orb", "sift", "akaze"])
    p_match.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    p_match.add_argument("--max-dim", type=int, default=1280, dest="max_dim")
    p_match.add_argument("--save", default=None, help="Path to save visualisation image")

    p_track = sub.add_parser("track", help="Optical flow tracking across frames")
    p_track.add_argument("--frames", required=True, help="Directory of frame images")
    p_track.add_argument("--method", default="lk", choices=["lk", "dense"])
    p_track.add_argument("--save-dir", default=None, dest="save_dir", help="Save flow visualisations here")

    p_cmp = sub.add_parser("compare", help="Image similarity metrics")
    p_cmp.add_argument("--img1", required=True)
    p_cmp.add_argument("--img2", required=True)
    p_cmp.add_argument("--max-dim", type=int, default=1280, dest="max_dim")
    p_cmp.add_argument("--save-diff", default=None, dest="save_diff", help="Save difference image")
    p_cmp.add_argument("--amplify", type=float, default=3.0, help="Amplify difference image contrast")

    p_pipe = sub.add_parser("pipeline", help="Full pipe inspection: video → annotated video with distance")
    p_pipe.add_argument("--video", required=True, help="Input inspection video")
    p_pipe.add_argument("--output", required=True, help="Output annotated video path (.mp4)")
    p_pipe.add_argument("--gt-distance", type=float, default=None, dest="gt_distance",
                        help="Ground-truth total distance (m) for auto scale calibration")
    p_pipe.add_argument("--scale", type=float, default=0.001,
                        help="Manual metres/pixel scale (used if --gt-distance not provided)")
    p_pipe.add_argument("--every-n", type=int, default=1, dest="every_n",
                        help="Process every N-th frame (speeds up long videos)")
    p_pipe.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    p_pipe.add_argument("--water-mask", type=float, default=0.0, dest="water_mask",
                        help="Mask bottom fraction of frame to exclude water (e.g. 0.3 = bottom 30%%))")
    p_pipe.add_argument("--plot", default=None, help="Save cumulative distance plot to this path")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "extract": cmd_extract,
        "match": cmd_match,
        "track": cmd_track,
        "compare": cmd_compare,
        "pipeline": cmd_pipeline,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)
