# automation-tracking

## program workflow

![alt text](media\diagram(2).png)

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
Important: `--model` is required. You must provide either the positional `folder` argument or `--folders`.

### Show help

```powershell
python .\main.py --help
```

### Create & activate a venv (PowerShell)

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If script execution is blocked, run PowerShell with an execution policy bypass for the session:

```powershell
powershell -ExecutionPolicy Bypass -NoProfile -Command ". .\venv\Scripts\Activate.ps1"
```

### Common run examples (PowerShell)

- Single folder (process all .mp4 in `.\media`):

```powershell
python .\main.py --model .\model\segment.pt .\media
```

- Specify an output directory and a fallback CSV:

```powershell
python .\main.py --model .\model\segment.pt --out .\output --csv .\annotations.csv .\media
```

- Batch mode using a `folders.txt` file (one folder path per line):

```powershell
python .\main.py --model .\model\segment.pt --folders .\folders.txt
```

- Quick test run (only process the first video):

```powershell
python .\main.py --model .\model\segment.pt --test .\media
```

### CLI arguments summary

- `folder` (positional, optional): Folder containing `.mp4` files to process (provide this OR `--folders`).
- `--folders`: Path to a text file with a list of folders to process (one per line).
- `--model` (required): Path to the YOLO model file (e.g., `model\\segment.pt`).
- `--out`: Output directory (default: current working directory `.`).
- `--csv`: Path to an annotation CSV file with `Frame#` and `Annotation` columns — used when a same-named per-video CSV is not present.
- `--test`: Flag; when present the script processes only the first video (useful for quick checks).

### What the script prints / outputs

- On success the script prints a line like: `[✓] Done: <filename>`
- After each processed video it prints the returned CSV paths, e.g. `- left_csv: <path>` and `- right_csv: <path>`.
- `--out` sets where generated files are written (default `.`).

The `process_video_pipeline` function should return a dictionary containing at least `left_csv` and `right_csv` entries — the script prints those values.

---

### Minimal dependency install (if you don't have `requirements.txt`)

```powershell
pip install torch numpy pandas opencv-python tqdm ultralytics
```

Adjust packages to match your environment and CUDA/CPU requirements.
