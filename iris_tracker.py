#!/usr/bin/env python3
"""
iris_tracker.py — All-in-one video iris/eye tracking pipeline.

Consolidates the full workflow from this repository into a single file:
  1. Scan folder(s) for .mp4 videos
  2. Crop each video with a fixed ROI (in-memory, no temp files)
  3. Split each cropped frame into left/right eye halves
  4. Run YOLO segmentation model to detect iris in each half
  5. Compute position (x, y) and velocity per frame
  6. Save per-video CSV results (optionally merged with annotation CSVs)

Usage examples:
  # Single folder
  python iris_tracker.py ./videos --model ./model/segment.pt

  # Batch mode (folders listed in a text file, one per line)
  python iris_tracker.py --folders folders.txt --model ./model/segment.pt

  # Custom output directory + annotation CSV fallback
  python iris_tracker.py ./videos --model ./model/segment.pt --out ./output --csv annotations.csv

  # Quick test (process only the first video)
  python iris_tracker.py ./videos --model ./model/segment.pt --test
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────
# Fixed crop ROI constants (pixels)
# ─────────────────────────────────────────────────────────
FIXED_ROI_X = 7
FIXED_ROI_Y = 184
FIXED_ROI_W = 1905
FIXED_ROI_H = 603


# ─────────────────────────────────────────────────────────
# Video cropping utilities
# ─────────────────────────────────────────────────────────

def crop_frames_in_memory(video_path: str,
                          roi: Tuple[int, int, int, int] = (FIXED_ROI_X, FIXED_ROI_Y, FIXED_ROI_W, FIXED_ROI_H)):
    """Generator that yields cropped frames from *video_path* using the given ROI.

    Parameters
    ----------
    video_path : str
        Path to the input .mp4 video.
    roi : tuple of int
        (x, y, width, height) rectangle to crop from each frame.

    Yields
    ------
    numpy.ndarray
        BGR cropped frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    x, y, w, h = roi
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame[y:y + h, x:x + w]
    finally:
        cap.release()


def get_video_fps(video_path: str) -> float:
    """Return the FPS of *video_path*, falling back to 30.0."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else 30.0


def crop_video_to_file(input_path: str,
                       output_path: Optional[str] = None,
                       roi: Tuple[int, int, int, int] = (FIXED_ROI_X, FIXED_ROI_Y, FIXED_ROI_W, FIXED_ROI_H),
                       codec: str = "mp4v") -> int:
    """Crop *input_path* with the given ROI and write a new video file.

    Returns 0 on success, non-zero on error.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {input_path}", file=sys.stderr)
        return 1

    x, y, w, h = roi
    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > frame_w or y + h > frame_h:
            print(f"Error: Crop ROI ({x},{y},{w},{h}) exceeds frame bounds ({frame_w}x{frame_h}).", file=sys.stderr)
            return 1

        # Ensure even dimensions for video encoders
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1

        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_cropped.mp4"

        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
            if not writer.isOpened():
                print(f"Error: Failed to open output video for writing: {output_path}", file=sys.stderr)
                return 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame[y:y + h, x:x + w])
    finally:
        cap.release()
        try:
            writer.release()
        except Exception:
            pass

    return 0


# ─────────────────────────────────────────────────────────
# YOLO iris detection helpers
# ─────────────────────────────────────────────────────────

def _parse_iris_result(result) -> Tuple[int, int, float, Optional[Tuple[int, int, int, int]]]:
    """Parse a single YOLO result into (cx, cy, size, bbox). Returns (-1, -1, -1.0, None) if no detection."""
    if len(result.boxes) == 0:
        return -1, -1, -1.0, None
    boxes = result.boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    size = round(((x2 - x1) + (y2 - y1)) / 2, 2)
    return cx, cy, size, (x1, y1, x2, y2)


def detect_iris(frame: np.ndarray, model, device: str, conf_threshold: float = 0.5):
    """Run YOLO prediction on *frame* and return (cx, cy, size, bbox).

    Returns (-1, -1, -1.0, None) when no detection is found.
    """
    use_half = device == "cuda"
    results = model.predict(frame, verbose=False, conf=conf_threshold, device=device, half=use_half)[0]
    return _parse_iris_result(results)


def detect_iris_batch(frames: list, model, device: str, conf_threshold: float = 0.5):
    """Run YOLO on a batch of images in one GPU call. Returns list of (cx, cy, size, bbox) per frame."""
    if not frames:
        return []
    use_half = device == "cuda"
    results_list = model.predict(frames, verbose=False, conf=conf_threshold, device=device, half=use_half)
    return [_parse_iris_result(r) for r in results_list]


# ─────────────────────────────────────────────────────────
# Core tracking — CSV-only (no video output)
# ─────────────────────────────────────────────────────────

def track_irises_from_frames(cropped_frames, model, device: str, fps: float = 30.0, start_frame: int = 0):
    """Track iris positions in left/right halves of each cropped frame.

    Parameters
    ----------
    cropped_frames : iterable of numpy.ndarray
        Generator or list of cropped BGR frames (both eyes visible, side-by-side).
    model : ultralytics.YOLO
        Loaded YOLO model.
    device : str
        ``"cuda"`` or ``"cpu"``.
    fps : float
        Frames per second (used for velocity calculation).
    start_frame : int
        Absolute frame number offset for the ``Frame#`` column.

    Returns
    -------
    list[dict]
        One row per frame with keys: Frame#, left_x, left_y, left_velocity,
        right_x, right_y, right_velocity.
    """
    frame_idx = 0
    prev_lx, prev_ly = None, None
    prev_rx, prev_ry = None, None
    printed_info = False
    results_rows: list[dict] = []
    log_every_n_frames = 500

    for frame in cropped_frames:
        if not printed_info:
            print(f"  [track] frame size: {frame.shape[1]}x{frame.shape[0]}, fps={fps}")
            printed_info = True
        if frame_idx > 0 and frame_idx % log_every_n_frames == 0:
            print(f"  [track] processed {frame_idx} frames …")

        width = frame.shape[1]
        left_half = frame[:, :width // 2]
        right_half = frame[:, width // 2:]

        # Batch left + right in one GPU call (faster than two separate inferences)
        (lx, ly, lsize, lbox), (rx, ry, rsize, rbox) = detect_iris_batch(
            [left_half, right_half], model, device, conf_threshold=0.5
        )
        if prev_lx is not None and lx != -1 and ly != -1:
            lvel = round(((lx - prev_lx) ** 2 + (ly - prev_ly) ** 2) ** 0.5 * fps, 2)
        else:
            lvel = 0.0
        if prev_rx is not None and rx != -1 and ry != -1:
            rvel = round(((rx - prev_rx) ** 2 + (ry - prev_ry) ** 2) ** 0.5 * fps, 2)
        else:
            rvel = 0.0
        prev_lx, prev_ly = lx, ly
        prev_rx, prev_ry = rx, ry

        results_rows.append({
            "Frame#": int(start_frame + frame_idx),
            "left_x": int(lx) if lx != -1 else -1,
            "left_y": int(ly) if ly != -1 else -1,
            "left_velocity": float(lvel),
            "left_size": float(lsize) if lsize != -1.0 else -1.0,
            "right_x": int(rx) if rx != -1 else -1,
            "right_y": int(ry) if ry != -1 else -1,
            "right_velocity": float(rvel),
            "right_size": float(rsize) if rsize != -1.0 else -1.0,
        })
        frame_idx += 1

    if results_rows:
        print(f"  [track] done — {len(results_rows)} frames.")
    return results_rows


# ─────────────────────────────────────────────────────────
# Optional: save annotated combined video
# ─────────────────────────────────────────────────────────

def save_combined_traced_video(cropped_frames, model, device: str, output_path: str, fps: float = 30.0):
    """For each cropped frame, detect irises, draw bounding boxes, combine side-by-side, and save."""
    writer = None
    for frame in cropped_frames:
        width = frame.shape[1]
        left_frame = frame[:, :width // 2].copy()
        right_frame = frame[:, width // 2:].copy()

        # Left
        lx, ly, _, lbox = detect_iris(left_frame, model, device)
        if lbox:
            x1, y1, x2, y2 = lbox
            cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(left_frame, (lx, ly), 4, (0, 0, 255), -1)

        # Right
        rx, ry, _, rbox = detect_iris(right_frame, model, device)
        if rbox:
            x1, y1, x2, y2 = rbox
            cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(right_frame, (rx, ry), 4, (0, 0, 255), -1)

        combined = cv2.hconcat([left_frame, right_frame])

        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        writer.write(combined)

    if writer:
        writer.release()


# ─────────────────────────────────────────────────────────
# Single-video pipeline
# ─────────────────────────────────────────────────────────

def process_video(video_path: str,
                  model_path: str,
                  output_dir: str = ".",
                  annotation_csv_path: Optional[str] = None) -> dict:
    """Full pipeline for one video file.

    Steps:
      1. Crop with fixed ROI (in-memory — no intermediate file)
      2. Split each frame into left / right halves
      3. Run YOLO detection on each half, compute centre + velocity
      4. Write left & right CSV files (merged with annotations if provided)

    Returns
    -------
    dict
        ``{"left_csv": <path>, "right_csv": <path>}``
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [pipeline] Device: {device}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- Crop frames in memory ---
    print(f"  [pipeline] Cropping in-memory: {video_path}")
    cropped_frames = crop_frames_in_memory(video_path)

    # --- Load YOLO model ---
    print(f"  [pipeline] Loading model: {model_path} …")
    model = YOLO(model_path)
    print(f"  [pipeline] Model loaded.")
    if device == "cuda":
        print(f"  [pipeline] Warming up GPU (one-time) …")
        dummy = np.zeros((FIXED_ROI_H // 2, FIXED_ROI_W // 2, 3), dtype=np.uint8)
        detect_iris_batch([dummy, dummy], model, device)
        print(f"  [pipeline] GPU warmup done.")

    # --- Read annotation CSV (optional) ---
    start_frame = 0
    annotations_df = None
    if annotation_csv_path and os.path.exists(annotation_csv_path):
        try:
            annotations_df = pd.read_csv(annotation_csv_path)
            if "Frame#" in annotations_df.columns:
                start_frame = int(annotations_df["Frame#"].iloc[0])
        except Exception as e:
            print(f"  [pipeline] Warning: failed reading annotation CSV '{annotation_csv_path}': {e}")

    # --- Track irises ---
    fps = get_video_fps(video_path)
    print(f"  [pipeline] Starting iris tracking …")
    results_list = track_irises_from_frames(
        cropped_frames=cropped_frames,
        model=model,
        device=device,
        fps=fps,
        start_frame=start_frame,
    )

    # --- Write CSVs ---
    left_csv_path = os.path.join(output_dir, f"{base_name}_left.csv")
    right_csv_path = os.path.join(output_dir, f"{base_name}_right.csv")

    if results_list:
        df_all = pd.DataFrame(results_list)

        # Build left/right DataFrames with columns: Frame#, annotation, x_position, y_position, seconds, ellipse_size
        left_cols = ["Frame#", "left_x", "left_y", "left_size"]
        right_cols = ["Frame#", "right_x", "right_y", "right_size"]
        df_left = df_all[left_cols].copy()
        df_right = df_all[right_cols].copy()
        df_left["seconds"] = (df_left["Frame#"] - start_frame) / fps
        df_right["seconds"] = (df_right["Frame#"] - start_frame) / fps
        df_left = df_left.rename(columns={"left_x": "x_position", "left_y": "y_position", "left_size": "ellipse_size"})
        df_right = df_right.rename(columns={"right_x": "x_position", "right_y": "y_position", "right_size": "ellipse_size"})
        # Column order: Frame#, annotation, x_position, y_position, seconds, ellipse_size
        df_left = df_left.reindex(columns=["Frame#", "x_position", "y_position", "seconds", "ellipse_size"])
        df_right = df_right.reindex(columns=["Frame#", "x_position", "y_position", "seconds", "ellipse_size"])

        # Merge with annotations so every row has an annotation label (tracking rows preserved)
        if annotations_df is not None and "Frame#" in annotations_df.columns:
            ann_cols = ["Frame#"]
            label_col = None
            if "Annotation" in annotations_df.columns:
                label_col = "Annotation"
            elif "Label" in annotations_df.columns:
                label_col = "Label"
            if label_col:
                ann_cols.append(label_col)
            ann_df = annotations_df[ann_cols].rename(columns={label_col: "annotation"}) if label_col else annotations_df[["Frame#"]].copy()
            if label_col:
                df_left = pd.merge(df_left, ann_df, on="Frame#", how="left")
                df_right = pd.merge(df_right, ann_df, on="Frame#", how="left")
            else:
                df_left["annotation"] = ""
                df_right["annotation"] = ""
        else:
            df_left["annotation"] = ""
            df_right["annotation"] = ""
        # Ensure annotation is second column
        df_left = df_left.reindex(columns=["Frame#", "annotation", "x_position", "y_position", "seconds", "ellipse_size"])
        df_right = df_right.reindex(columns=["Frame#", "annotation", "x_position", "y_position", "seconds", "ellipse_size"])

        df_left.to_csv(left_csv_path, index=False)
        df_right.to_csv(right_csv_path, index=False)
        print(f"  [pipeline] Wrote CSVs: {left_csv_path}, {right_csv_path}")
    else:
        print("  [pipeline] Warning: no tracking results produced.")

    return {
        "left_csv": left_csv_path,
        "right_csv": right_csv_path,
    }


# ─────────────────────────────────────────────────────────
# Folder processing
# ─────────────────────────────────────────────────────────

def find_videos(folder: str) -> list[str]:
    """Return sorted list of full paths to .mp4 files. Layout: data/subject01/video.mp4 (data then subject subdirs)."""
    if not os.path.isdir(folder):
        return []
    search_base = os.path.join(folder, "data")
    if not os.path.isdir(search_base):
        search_base = folder
    paths = []
    for subdir_name in sorted(os.listdir(search_base)):
        subdir = os.path.join(search_base, subdir_name)
        if not os.path.isdir(subdir):
            continue
        for name in sorted(os.listdir(subdir)):
            if (name.lower().endswith(".mp4")
                    and "_cropped" not in name
                    and "_combined_traced" not in name
                    and "_left" not in name
                    and "_right" not in name):
                paths.append(os.path.join(subdir, name))
    return paths


def process_folder(folder: str, model_path: str, output_dir: str = ".",
                   fallback_csv: Optional[str] = None, test: bool = False) -> int:
    """Process every .mp4 in *folder*. Returns count of successfully processed videos."""
    if not os.path.isdir(folder):
        print(f"Error: Not a directory: {folder}", file=sys.stderr)
        return 0

    os.makedirs(output_dir, exist_ok=True)
    video_paths = find_videos(folder)
    if not video_paths:
        print(f"No .mp4 files found in: {folder} (layout: data/<subject>/video.mp4)")
        return 0

    processed = 0
    for video_path in video_paths:
        name = os.path.basename(video_path)
        base_name = os.path.splitext(name)[0]
        video_dir = os.path.dirname(video_path)

        # Annotation CSV next to the video (same folder), fall back to global --csv
        per_video_csv = os.path.join(video_dir, f"{base_name}.csv")
        annotation_csv = per_video_csv if os.path.exists(per_video_csv) else fallback_csv

        try:
            results = process_video(video_path, model_path, output_dir=output_dir,
                                    annotation_csv_path=annotation_csv)
            print(f"[✓] Done: {name}")
            print(f"    left_csv:  {results['left_csv']}")
            print(f"    right_csv: {results['right_csv']}")
            processed += 1
            if test:
                print("[test mode] Only processed the first video.")
                break
        except Exception as e:
            print(f"[✗] Failed: {name} → {e}", file=sys.stderr)

    return processed


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="All-in-one iris tracking: crop, split, detect, and export CSVs for .mp4 eye videos."
    )
    parser.add_argument("folder", nargs="?",
                        help="Folder containing .mp4 files to process")
    parser.add_argument("--folders",
                        help="Text file listing folders to process (one path per line)")
    parser.add_argument("--model", required=True,
                        help="Path to YOLO model file (e.g. model/segment.pt)")
    parser.add_argument("--out", default=".",
                        help="Output directory for CSV files (default: current dir)")
    parser.add_argument("--csv",
                        help="Fallback annotation CSV (Frame# + Annotation columns)")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first video (quick sanity check)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    total = 0

    if args.folders:
        # Batch mode — read folder paths from a text file
        if not os.path.isfile(args.folders):
            print(f"Error: Not a file: {args.folders}", file=sys.stderr)
            return 1
        with open(args.folders, "r", encoding="utf-8") as f:
            folders = [line.strip() for line in f if line.strip()]
        for folder in folders:
            print(f"\n{'═' * 60}")
            print(f"[batch] Processing folder: {folder}")
            print(f"{'═' * 60}")
            total += process_folder(folder, args.model, output_dir=args.out,
                                    fallback_csv=args.csv, test=args.test)
            if args.test:
                break
    else:
        if not args.folder:
            print("Error: Provide a folder path or use --folders <file>", file=sys.stderr)
            return 1
        total = process_folder(args.folder, args.model, output_dir=args.out,
                               fallback_csv=args.csv, test=args.test)

    print(f"\nDone — {total} video(s) processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
