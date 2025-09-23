# automation-tracking

## program workflow

![](media/diagram(2).png)

## How to run the video processing program (main.py)

A step-by-step template to get the `main.py` CLI running. Paste this into a `README.md` or `RUN_INSTRUCTIONS.md` in your project and adapt paths as needed.

---

## Overview

This project processes `.mp4` videos using `process_video.process_video_pipeline`. The CLI `main.py` scans a folder (or many folders), runs the pipeline on each video, and writes output CSVs and processed video files to an output directory.

## Prerequisites

* Python 3.8+ (3.10 recommended)
* A working `process_video.py` (must expose `process_video_pipeline`) in the same repo or on `PYTHONPATH`
* A trained YOLO model file (e.g. `model/best.pt`) — set path with `--model`
* `conda` or `python -m venv` for an isolated environment

---

## 1. Project layout (recommended)

```
project-root/
├─ main.py
├─ process_video.py
├─ folder_processor.py
├─ video_crop_utils.py
├─ requirements.txt        # (optional) pip requirements
├─ folders.txt             # (optional) list of folders for batch processing
└─ videos/                 # folder with .mp4 files
   └─ sample_video.mp4
└─ model/              
   └─ segmentation_model.pt # need to be download
```

---

## 2. Create & activate environment

### Using conda (recommended)

```bash
conda create -n videoproc python=3.10 -y
conda activate videoproc
# then install requirements
pip install -r requirements.txt
```

### Using venv (alternate)

```bash
python -m venv venv
# mac / linux
source venv/bin/activate
# windows (PowerShell)
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, see section **"Generate requirements.txt"** below.

---

## 3. Generate a `requirements.txt` (if you need one)

If you've already installed packages in your environment and want to export them:

#### Export pip packages (simple)

```bash
pip freeze > requirements.txt
```

#### Export complete conda environment (recommended for reproducibility)

```bash
conda env export > environment.yml
```

> Tip: `pip freeze` lists everything installed in the active Python environment. `conda env export` produces a YAML that can recreate the conda environment (preferred if you used conda packages).

---

## 4. Example `requirements.txt` (minimal template)

Fill this with the libs your project uses. Replace or extend as needed.

```
# example requirements.txt (edit to match your project)
torch>=1.13
numpy
pandas
opencv-python
tqdm
scipy
ultralytics   # or yolov5 / your YOLO dependency
```

---

## 5. How to run `main.py`

Run from the project root (so `process_video` imports from the same folder work):

### Batch mode (multiple folders listed in a file)

Create `folders.txt` with one folder path per line, then:

#### Windows PowerShell example

```powershell
python .\main.py "C:\Data\videos" --model "C:\models\best.pt" --out "C:\Data\processed"
```

---

## 6. What the script prints / outputs

* The script prints a success line for each processed video: `[✓] Done: <filename>`
* After each processed video it prints the returned CSV paths, for example:

  * `- left_csv: /path/to/output/video_left.csv`
  * `- right_csv: /path/to/output/video_right.csv`
* `--out` sets the directory where generated files are stored (default = current working directory `.`)

The `process_video_pipeline` function is expected to return a dictionary like `{'left_csv': ..., 'right_csv': ...}` — the script prints these values.

---



### Overview

This tool processes all `.mp4` videos in a folder (or multiple folders), crops each video to a fixed region of interest (ROI), splits the cropped video into left and right halves, runs YOLO tracking on each half, and saves the tracking results as CSV files.

### Requirements

- Python 3.8+
- OpenCV (`cv2`)
- pandas
- numpy
- torch
- [ultralytics](https://github.com/ultralytics/ultralytics) (for YOLO)
- Your YOLO model file (e.g., `model/segment.pt`)

Install dependencies (example):
