import os
import cv2
import numpy as np
import pandas as pd
import torch
from video_crop_utils import crop_video_fixed
from ultralytics import YOLO

def crop_video_fixed_in_memory(video_path, roi=(7,184,1905,603)):
    """
    Generator that yields cropped frames from the video using the fixed ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    x, y, w, h = roi
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        cropped = frame[y:y+h, x:x+w]
        yield cropped
    cap.release()

def process_video_pipeline(video_path: str, model_path: str, output_dir: str = ".", annotation_csv_path: str | None = None) -> dict:
    """
    1) Crop input video with fixed ROI (x=7,y=184,w=1905,h=603)
    2) Split cropped video into left/right halves
    3) Run YOLO tracking on each half and annotate
    4) Save left/right annotated videos + CSVs and combined side-by-side video
    Returns dict with output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Crop with fixed ROI (in-memory, do not save cropped video)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[pipeline] Cropping in-memory: {video_path}")
    try:
        cropped_frames = crop_video_fixed_in_memory(video_path)
    except Exception as e:
        raise RuntimeError(f"Cropping failed for {video_path}: {e}")

    print(f"[pipeline] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[pipeline] Tracking L/R from cropped frames (in-memory)")
    start_frame = 0
    annotations_df = None
    if annotation_csv_path and os.path.exists(annotation_csv_path):
        try:
            annotations_df = pd.read_csv(annotation_csv_path)
            if 'Frame#' in annotations_df.columns:
                start_frame = int(annotations_df['Frame#'].iloc[0])
        except Exception as e:
            print(f"[pipeline] Warning: failed reading annotation CSV '{annotation_csv_path}': {e}")

    results_list = process_video_csv_only_frames(
        cropped_frames=cropped_frames,
        model=model,
        device=device,
        start_frame=start_frame,
    )

    left_csv_path = os.path.join(output_dir, f"{base_name}_left.csv")
    right_csv_path = os.path.join(output_dir, f"{base_name}_right.csv")
    # combined_video_path = os.path.join(output_dir, f"{base_name}_combined_traced.mp4")  # REMOVE

    try:
        if results_list is not None:
            df_metrics = pd.DataFrame(results_list)
            if annotations_df is not None and 'Frame#' in annotations_df.columns:
                if 'Annotation' in annotations_df.columns:
                    merged = pd.merge(annotations_df[['Frame#','Annotation']], df_metrics, on='Frame#', how='left')
                else:
                    merged = pd.merge(annotations_df[['Frame#']], df_metrics, on='Frame#', how='left')
                merged.to_csv(left_csv_path, index=False)
                merged.to_csv(right_csv_path, index=False)
            else:
                df_metrics.to_csv(left_csv_path, index=False)
                df_metrics.to_csv(right_csv_path, index=False)
    except Exception as e:
        print(f"[pipeline] Warning: failed writing CSV files: {e}")

    # --- REMOVE video saving ---
    # cropped_frames_for_video = crop_video_fixed_in_memory(video_path)
    # save_combined_traced_video(
    #     cropped_frames=cropped_frames_for_video,
    #     model=model,
    #     device=device,
    #     output_path=combined_video_path,
    # )

    # Only return the two required outputs
    return {
        "left_csv": left_csv_path,
        "right_csv": right_csv_path,
        # "combined_traced_video": combined_video_path,  # REMOVE
    }


def _combine_side_by_side(left_video_path: str, right_video_path: str, output_path: str) -> None:
    cap_l = cv2.VideoCapture(left_video_path)
    cap_r = cv2.VideoCapture(right_video_path)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Failed to open left/right video for combining")

    fps_l = cap_l.get(cv2.CAP_PROP_FPS)
    fps_r = cap_r.get(cv2.CAP_PROP_FPS)
    fps = fps_l if fps_l and fps_l > 0 else (fps_r if fps_r and fps_r > 0 else 30.0)
    lw = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    lh = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize right to left height if needed for hconcat
    target_h = min(lh, rh)
    def resize_to_h(frame, target_h):
        h, w = frame.shape[:2]
        if h == target_h:
            return frame
        new_w = int(w * (target_h / h))
        return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Determine final width after possible resize
    dummy_left = target_h != lh
    out_w = None  # compute on first frame
    writer = None

    while True:
        ok_l, f_l = cap_l.read()
        ok_r, f_r = cap_r.read()
        if not ok_l or not ok_r:
            break
        f_lr = resize_to_h(f_l, target_h)
        f_rr = resize_to_h(f_r, target_h)
        try:
            concat = cv2.hconcat([f_lr, f_rr])
        except cv2.error as e:
            cap_l.release(); cap_r.release()
            raise RuntimeError(f"hconcat failed: {e}")

        if writer is None:
            out_h = concat.shape[0]
            out_w = concat.shape[1]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                cap_l.release(); cap_r.release()
                raise RuntimeError(f"Failed to open output for combined video: {output_path}")

        try:
            writer.write(concat)
        except cv2.error as e:
            cap_l.release(); cap_r.release(); writer.release()
            raise RuntimeError(f"writer.write failed for combined video: {e}")

    cap_l.release()
    cap_r.release()
    if writer is not None:
        writer.release()


def process_video_csv_only(video_path, model, device, start_frame: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_lx, prev_ly = None, None
    prev_rx, prev_ry = None, None

    def detect(frame, conf_threshold=0.5):
        results = model.predict(frame, verbose=False, conf=conf_threshold, device=device)[0]
        if len(results.boxes) == 0:
            return -1, -1, -1.0, None
        boxes = results.boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        size = round(((x2 - x1) + (y2 - y1)) / 2, 2)
        return cx, cy, size, (x1, y1, x2, y2)

    printed_info = False
    results_rows: list[dict] = []
    while cap.isOpened():
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            cap.release()
            raise RuntimeError(f"[track] cv2.read failed: {e}")
        if not ret or frame is None:
            break

        if not printed_info:
            print(f"[track] input frame size: {frame.shape[1]}x{frame.shape[0]} fps={fps}")
            printed_info = True

        try:
            left_frame = frame[:, : width // 2]
            right_frame = frame[:, width // 2 :]
        except Exception as e:
            cap.release()
            raise RuntimeError(f"[track] slicing L/R failed: {e}")

        try:
            lx, ly, lsize, lbox = detect(left_frame)
        except Exception as e:
            cap.release()
            raise RuntimeError(f"[track] detect left failed: {e}")
        if prev_lx is not None and lx != -1 and ly != -1:
            lvelocity = round(((lx - prev_lx) ** 2 + (ly - prev_ly) ** 2) ** 0.5 * fps, 2)
        else:
            lvelocity = 0.0

        try:
            rx, ry, rsize, rbox = detect(right_frame)
        except Exception as e:
            cap.release()
            raise RuntimeError(f"[track] detect right failed: {e}")
        if prev_rx is not None and rx != -1 and ry != -1:
            rvelocity = round(((rx - prev_rx) ** 2 + (ry - prev_ry) ** 2) ** 0.5 * fps, 2)
        else:
            rvelocity = 0.0

        prev_lx, prev_ly = lx, ly
        prev_rx, prev_ry = rx, ry
        frame_idx += 1
        # Record metrics row aligned to absolute frame number
        results_rows.append({
            'Frame#': int(start_frame + frame_idx - 1),
            'x': int(lx) if lx != -1 else -1,
            'y': int(ly) if ly != -1 else -1,
            'velocity': float(lvelocity),
        })

    cap.release()
    return results_rows


def process_video_csv_only_frames(cropped_frames, model, device, start_frame: int = 0):
    """
    Like process_video_csv_only, but operates on an iterable of cropped frames (in-memory).
    """
    # Assume cropped_frames is a generator or list of frames (numpy arrays)
    frame_idx = 0
    prev_lx, prev_ly = None, None
    prev_rx, prev_ry = None, None

    # Try to get fps from the original video if possible (optional: pass as argument)
    fps = 30.0  # Default fallback
    # If cropped_frames has attribute 'fps', use it (optional)
    if hasattr(cropped_frames, 'fps'):
        fps = cropped_frames.fps

    def detect(frame, conf_threshold=0.5):
        results = model.predict(frame, verbose=False, conf=conf_threshold, device=device)[0]
        if len(results.boxes) == 0:
            return -1, -1, -1.0, None
        boxes = results.boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        size = round(((x2 - x1) + (y2 - y1)) / 2, 2)
        return cx, cy, size, (x1, y1, x2, y2)

    printed_info = False
    results_rows: list[dict] = []
    for frame in cropped_frames:
        if not printed_info:
            print(f"[track] input frame size: {frame.shape[1]}x{frame.shape[0]} fps={fps}")
            printed_info = True

        width = frame.shape[1]
        try:
            left_frame = frame[:, : width // 2]
            right_frame = frame[:, width // 2 :]
        except Exception as e:
            raise RuntimeError(f"[track] slicing L/R failed: {e}")

        try:
            lx, ly, lsize, lbox = detect(left_frame)
        except Exception as e:
            raise RuntimeError(f"[track] detect left failed: {e}")
        if prev_lx is not None and lx != -1 and ly != -1:
            lvelocity = round(((lx - prev_lx) ** 2 + (ly - prev_ly) ** 2) ** 0.5 * fps, 2)
        else:
            lvelocity = 0.0

        try:
            rx, ry, rsize, rbox = detect(right_frame)
        except Exception as e:
            raise RuntimeError(f"[track] detect right failed: {e}")
        if prev_rx is not None and rx != -1 and ry != -1:
            rvelocity = round(((rx - prev_rx) ** 2 + (ry - prev_ry) ** 2) ** 0.5 * fps, 2)
        else:
            rvelocity = 0.0

        prev_lx, prev_ly = lx, ly
        prev_rx, prev_ry = rx, ry
        frame_idx += 1
        # Record metrics row aligned to absolute frame number
        results_rows.append({
            'Frame#': int(start_frame + frame_idx - 1),
            'x': int(lx) if lx != -1 else -1,
            'y': int(ly) if ly != -1 else -1,
            'velocity': float(lvelocity),
        })

    return results_rows


def save_combined_traced_video(cropped_frames, model, device, output_path, fps=30.0):
    """
    For each cropped frame, detect object in left/right, draw boxes, combine, and save as video.
    """
    # Try to get fps from cropped_frames if available
    if hasattr(cropped_frames, 'fps'):
        fps = cropped_frames.fps

    writer = None
    for frame in cropped_frames:
        width = frame.shape[1]
        left_frame = frame[:, : width // 2].copy()
        right_frame = frame[:, width // 2 :].copy()

        # Detect and draw on left
        lx, ly, lsize, lbox = detect_for_video(left_frame, model, device)
        if lbox:
            x1, y1, x2, y2 = lbox
            cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(left_frame, (lx, ly), 4, (0,0,255), -1)

        # Detect and draw on right
        rx, ry, rsize, rbox = detect_for_video(right_frame, model, device)
        if rbox:
            x1, y1, x2, y2 = rbox
            cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(right_frame, (rx, ry), 4, (0,0,255), -1)

        combined = cv2.hconcat([left_frame, right_frame])

        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        writer.write(combined)
    if writer:
        writer.release()


def detect_for_video(frame, model, device, conf_threshold=0.5):
    results = model.predict(frame, verbose=False, conf=conf_threshold, device=device)[0]
    if len(results.boxes) == 0:
        return -1, -1, -1.0, None
    boxes = results.boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    size = round(((x2 - x1) + (y2 - y1)) / 2, 2)
    return cx, cy, size, (x1, y1, x2, y2)