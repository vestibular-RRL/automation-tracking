# automation-tracking

## Overview

This tool processes all `.mp4` videos in a folder (or multiple folders), crops each video to a fixed region of interest (ROI), splits the cropped video into left and right halves, runs YOLO tracking on each half, and saves the tracking results as CSV files.

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- pandas
- numpy
- torch
- [ultralytics](https://github.com/ultralytics/ultralytics) (for YOLO)
- Your YOLO model file (e.g., `model/segment.pt`)

Install dependencies (example):