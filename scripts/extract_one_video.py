#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "SETUP_frame_images_extraction"
sys.path.insert(0, str(SETUP))

import extract_bonus_frame_images
import extract_frame_images


def extract_main(channel: int) -> None:
    out_dir = extract_frame_images.OUT_DIR / str(channel)
    first_kept = extract_frame_images.FIRST_KEPT_FRAME[channel]
    if (out_dir / f"{first_kept}.png").exists():
        print(f"Main channel {channel} already extracted, skipping.")
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ffmpeg = extract_frame_images.find_ffmpeg()
    extract_frame_images.extract_frames(ffmpeg, channel)
    extract_frame_images.delete_leading_frames(channel)
    extract_frame_images.delete_trailing_frames(channel)


def extract_bonus(channel: int) -> None:
    out_dir = extract_bonus_frame_images.OUT_DIR / str(channel)
    if out_dir.exists() and any(out_dir.glob("*.png")):
        print(f"Bonus channel {channel} already extracted, skipping.")
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ffmpeg = extract_frame_images.find_ffmpeg()
    extract_bonus_frame_images.extract_frames(ffmpeg, channel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("main", "bonus"), required=True)
    parser.add_argument("--channel", type=int, required=True)
    args = parser.parse_args()

    if args.kind == "main":
        extract_main(args.channel)
    else:
        extract_bonus(args.channel)


if __name__ == "__main__":
    main()
