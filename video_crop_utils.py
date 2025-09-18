import os
import sys
from typing import Optional, Tuple

import cv2


FIXED_X = 10
FIXED_Y = 272
FIXED_WIDTH = 1909
FIXED_HEIGHT = 651


def select_roi_from_video(video_path: str) -> Optional[Tuple[int, int, int, int]]:
    """Show the first frame and let the user draw a rectangle.

    Returns (x, y, width, height) or None if cancelled/invalid.
    """
    if not os.path.exists(video_path):
        print(f"Error: Input video not found: {video_path}", file=sys.stderr)
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {video_path}", file=sys.stderr)
        return None

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Error: Could not read first frame.", file=sys.stderr)
        return None

    # Interactive selection using cv2.selectROI for simplicity
    window_name = "Select ROI (press Enter or Space to confirm, c to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    rect = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def crop_video(
    input_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
    output_path: Optional[str] = None,
    codec: str = "mp4v",
) -> int:
    """Crop a rectangle from each frame and write a new video.

    Returns 0 on success, non-zero on error.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {input_path}", file=sys.stderr)
        return 1

    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        # Validate bounds
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_width or y + height > frame_height:
            print(
                f"Error: Crop rectangle (x={x}, y={y}, w={width}, h={height}) exceeds frame bounds ({frame_width}x{frame_height}).",
                file=sys.stderr,
            )
            return 1

        # Ensure even dimensions for video encoders
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        # Derive default output path
        if output_path is None or output_path == "":
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_cropped.mp4"
        else:
            # Normalize: allow directory path or '.' and ensure .mp4 extension
            output_path = output_path.strip().strip('"').strip("'")
            if output_path in (".", "./", ".\\") or os.path.isdir(output_path) or output_path.endswith(("/", "\\")):
                directory = os.path.abspath(output_path) if output_path != "." else os.getcwd()
                base, _ = os.path.splitext(os.path.basename(input_path))
                output_path = os.path.join(directory, f"{base}_cropped.mp4")
            _, ext = os.path.splitext(output_path)
            if ext == "":
                output_path = f"{output_path}.mp4"

        parent_dir = os.path.dirname(os.path.abspath(output_path))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Prefer XVID for wide Win compatibility if requested codec fails later
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # Retry with XVID as fallback
            fallback = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(output_path, fallback, fps, (width, height))
            if not writer.isOpened():
                print(f"Error: Failed to open output video for writing: {output_path}", file=sys.stderr)
                return 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            roi = frame[y : y + height, x : x + width]
            writer.write(roi)
    finally:
        cap.release()
        try:
            writer.release()  # type: ignore[name-defined]
        except Exception:
            pass

    return 0


def crop_video_fixed(
    input_path: str,
    output_path: Optional[str] = None,
    codec: str = "mp4v",
) -> int:
    """Crop using fixed rectangle x=7, y=184, w=1905, h=603."""
    return crop_video(
        input_path=input_path,
        x=FIXED_X,
        y=FIXED_Y,
        width=FIXED_WIDTH,
        height=FIXED_HEIGHT,
        output_path=output_path,
        codec=codec,
    )

