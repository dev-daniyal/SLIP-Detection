# Slip Detector — Ultimate Frisbee

Automatically detects moments where players fully fall or slip to the ground in drone footage of Ultimate Frisbee games. Outputs a total slip count and per-event timestamps to a CSV file.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [macOS Setup Guide (Step-by-Step)](#macos-setup-guide-step-by-step)
- [Windows Setup Guide](#windows-setup-guide)
- [Linux Setup Guide](#linux-setup-guide)
- [Running the App](#running-the-app)
- [Output](#output)
- [Configuration](#configuration)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## What It Does

Given a drone video of an Ultimate Frisbee game (MP4 or MOV), the app:

1. Opens a simple desktop window where you pick the video file
2. Runs YOLOv8 pose estimation on every 3rd frame to detect all players simultaneously
3. Tracks each player across frames using BoTSORT and monitors how their body collapses
4. Flags a "slip" whenever a player transitions rapidly from standing to fallen
5. Saves the results to `output/slip_events.csv` — one row per detected slip

---

## How It Works

### Pose Estimation

The app uses **YOLOv8n-pose** (`yolov8n-pose.pt`), a lightweight neural network that detects people and estimates 17 body keypoints per person (the COCO keypoint set: nose, shoulders, hips, knees, ankles, etc.). The model weights (~6 MB) are downloaded automatically on first run from the Ultralytics CDN.

### Multi-Person Tracking

YOLOv8's built-in **BoTSORT tracker** assigns a consistent integer ID to each player and maintains it across frames. This means the slip logic is evaluated independently for each player, so two players falling simultaneously are both counted.

### Slip Heuristic

For every tracked player, the detector computes a **height ratio**: the vertical pixel span of their key joints (shoulders, hips, ankles) divided by their bounding-box height. A standing player typically has a height ratio of 0.40–0.70. A player lying on the ground collapses toward 0.

Over a rolling ~0.75-second window, the detector compares the current height ratio against the recent maximum. If the ratio drops sharply (standing → collapsed) and the magnitude of collapse exceeds the confidence threshold, a `SlipEvent` is recorded.

A per-player cooldown (default 2 seconds) prevents the same fall from being counted multiple times.

### Optical Flow Stabilisation

Between sampled frames the detector runs Lucas-Kanade optical flow to estimate camera translation/rotation from the drone. Frames are warped to remove this drift before pose estimation, reducing false positives caused by camera movement.

### Frame Sampling

Only every 3rd frame is decoded (the rest use `grab()` which advances the codec without full decompression). This roughly triples processing speed with negligible impact on detection quality for typical drone footage at 30 fps.

### Device Selection

The app automatically picks the best available compute backend:

| Priority | Backend | When used |
|----------|---------|-----------|
| 1 | CUDA | NVIDIA GPU present |
| 2 | MPS | Apple Silicon (M1/M2/M3/M4) |
| 3 | CPU | Fallback |

---

## Prerequisites

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.10 | 3.11 or 3.12 recommended |
| pip | 23+ | Bundled with Python 3.10+ |
| Git | any | To clone the repo |
| Disk space | ~2 GB | Python packages + YOLO model weights |
| RAM | 4 GB | 8 GB+ recommended for long videos |

No GPU is required. An Apple Silicon Mac will use the MPS backend automatically, which is noticeably faster than CPU-only.

---

## macOS Setup Guide (Step-by-Step)

This section walks you through every step from a clean Mac with nothing installed.

### 1. Install Homebrew

Homebrew is the standard package manager for macOS. Open **Terminal** (`Cmd+Space`, type "Terminal", press Enter) and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen prompts. It will ask for your macOS password. After it finishes, follow any instructions it prints about adding Homebrew to your PATH (especially on Apple Silicon where it installs to `/opt/homebrew`).

Verify Homebrew works:

```bash
brew --version
```

### 2. Install Python

macOS ships with an outdated Python. Install a current version via Homebrew:

```bash
brew install python@3.11
```

After installation, confirm the version:

```bash
python3.11 --version
# Python 3.11.x
```

### 3. Install Git (if not already installed)

```bash
brew install git
```

Verify:

```bash
git --version
```

### 4. Clone the Repository

Navigate to where you want the project to live, then clone it:

```bash
cd ~/Documents          # or any folder you prefer
git clone https://github.com/dev-daniyal/slip-detection.git
cd slip-detection
```

### 5. Create a Virtual Environment

A virtual environment keeps this project's packages isolated from the rest of your system. This is strongly recommended.

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Your terminal prompt will change to show `(.venv)` at the start. **You must activate the environment every time you open a new terminal window to work on this project.**

### 6. Upgrade pip

```bash
pip install --upgrade pip
```

### 7. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- **opencv-python** — video decoding and optical flow
- **ultralytics** — YOLOv8 framework (also installs PyTorch automatically)
- **numpy** — numerical operations

The first install downloads PyTorch (~200 MB) and other packages. This may take a few minutes depending on your internet connection.

### 8. (Apple Silicon only) Verify MPS is Available

If you have an M1/M2/M3/M4 Mac, PyTorch should automatically use the MPS GPU backend. You can verify:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# True
```

If this prints `False`, make sure you installed the standard `ultralytics` package (which pulls in the correct PyTorch version for macOS). Re-running `pip install -r requirements.txt` should fix it.

### 9. Run the App

```bash
python main.py
```

A small window will appear. See [Running the App](#running-the-app) for next steps.

---

## Windows Setup Guide

### 1. Install Python

Download the latest Python 3.11 or 3.12 installer from https://www.python.org/downloads/windows/

During installation, **check the box "Add Python to PATH"** before clicking Install.

Verify in Command Prompt:

```cmd
python --version
```

### 2. Install Git

Download Git for Windows from https://git-scm.com/download/win and run the installer with default settings.

### 3. Clone the Repository

Open **Command Prompt** or **Git Bash**:

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/dev-daniyal/slip-detection.git
cd slip-detection
```

### 4. Create and Activate a Virtual Environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 5. Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the App

```cmd
python main.py
```

---

## Linux Setup Guide

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git python3-tk -y

git clone https://github.com/dev-daniyal/slip-detection.git
cd slip-detection

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

> **Note:** `python3-tk` is required because the UI uses Tkinter, which is not bundled with Python on some Linux distributions.

---

## Running the App

```bash
# Make sure your virtual environment is active first
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

python main.py
```

### Step-by-Step Walkthrough

**1. The app window opens**

A 520×340 desktop window appears titled "Slip Detector — Ultimate Frisbee". It has four sections: Video File, controls, Progress, and Results.

**2. Select your video**

Click **Browse…** and navigate to your MP4 or MOV drone footage file. The filename will appear in the window. The **Run Detection** button becomes active.

**3. Start detection**

Click **Run Detection**. The app immediately begins processing in the background so the window stays responsive. The progress bar fills as frames are processed, and the status line shows the current frame count and percentage.

In your terminal you will also see live output:

```
  Using device: mps
  [SLIP] t=23.40s  frame=702  player=3  conf=0.812
  Progress: 1 min of footage processed (18s elapsed, 1 slips so far, 7 active tracks)
  [SLIP] t=87.10s  frame=2613  player=11  conf=0.743
  ...
  ==================================================
  Detection complete.
  Total slips: 4
  Processing time: 142.3s
  Average confidence: 0.776
  ==================================================
```

**4. Cancel (optional)**

If you want to stop early, click **Cancel**. Any slips detected so far are saved to the CSV.

**5. View results**

When processing finishes, the Results section shows the total slip count and processing time. Click **Open CSV** to open `output/slip_events.csv` in your default spreadsheet app (Numbers on macOS, Excel on Windows).

### First Run Note

On the very first run, `yolov8n-pose.pt` (~6 MB) is downloaded automatically from the Ultralytics CDN. This happens once and is cached for all future runs.

---

## Output

Results are saved to `output/slip_events.csv` (the `output/` folder is created automatically next to wherever you launch the script from).

```
timestamp,frame_number,confidence
23.4,702,0.812
87.1,2613,0.743
```

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Seconds into the video when the slip occurred |
| `frame_number` | integer | Exact frame index (useful for jumping to in a video editor) |
| `confidence` | float (0–1) | How confident the model is that this is a genuine slip |

A higher confidence value means a larger, cleaner collapse was detected. Values near the threshold (0.65) are borderline; values above 0.80 are strong detections.

---

## Configuration

The key parameters are defined at the top of the `run_detection()` function in `detector.py`. You can adjust them directly in the source, or modify `ui.py` to expose them as UI controls.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | `0.65` | Minimum confidence score to record a slip. Raise to `0.70`–`0.75` to reduce false positives; lower to `0.55` to catch borderline falls. |
| `frame_skip` | `3` | Process every Nth frame. Lower values (e.g. `2`) are more sensitive but slower. Higher values (e.g. `5`) speed up processing but may miss very fast falls. |
| `cooldown_sec` | `2.0` | Minimum seconds between two slip events for the same player. Prevents one fall being counted multiple times. |

**To change a parameter**, open `detector.py` and edit the function signature defaults:

```python
def run_detection(
    video_path: str,
    confidence_threshold: float = 0.65,   # <-- change here
    frame_skip: int = 3,                   # <-- change here
    cooldown_sec: float = 2.0,             # <-- change here
    ...
```

---

## Performance Notes

| Setup | Approximate speed |
|-------|-----------------|
| Apple M2 (MPS) | ~3–6× real-time for 1080p 30fps footage |
| Intel CPU (no GPU) | ~0.5–1× real-time |
| NVIDIA GPU (CUDA) | ~8–15× real-time |

"Real-time" means processing speed equals footage duration. 3× real-time means a 10-minute video takes ~3–4 minutes to process.

Processing speed scales with resolution. 4K footage is significantly slower than 1080p. If speed is a concern, consider downscaling the video to 1080p before processing.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tkinter'`

On Linux, Tkinter is a separate package:

```bash
sudo apt install python3-tk   # Ubuntu/Debian
sudo dnf install python3-tkinter  # Fedora
```

On macOS, this should not happen if you installed Python via Homebrew. If it does:

```bash
brew reinstall python@3.11
```

### `No module named 'cv2'` or `No module named 'ultralytics'`

Your virtual environment is not active. Run:

```bash
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows
```

Then re-run `python main.py`.

### YOLOv8 model download fails

The model is downloaded from the internet on first run. If the download fails (network error, corporate firewall):

1. Manually download `yolov8n-pose.pt` from `https://github.com/ultralytics/assets/releases`
2. Place it in the root of the project directory (same folder as `main.py`)

### `Could not open video: <path>`

- Check the file path has no unusual characters
- Make sure the file is a valid MP4 or MOV (try playing it in VLC first)
- On macOS, if you moved the file after selecting it, re-browse to it

### MPS not available on Apple Silicon

Make sure PyTorch was installed correctly. Uninstall and reinstall:

```bash
pip uninstall torch torchvision
pip install -r requirements.txt
```

Then verify:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

### Too many false positives

Raise `confidence_threshold` in `detector.py`:

```python
confidence_threshold: float = 0.70,  # was 0.65
```

Common sources of false positives: players diving (intentional), players jumping and landing awkwardly, or very low-altitude drone footage where bounding boxes are large and noisy.

### Slips are being missed

Lower `confidence_threshold` or `frame_skip`:

```python
confidence_threshold: float = 0.55,
frame_skip: int = 2,
```

This increases sensitivity and processes more frames, at the cost of more false positives and slower runtime.

### App window does not appear (macOS)

If running in a remote shell or headless environment, Tkinter cannot open a display. Run the app from a local terminal session, not SSH.

---

## Project Structure

```
slip-detection/
├── main.py          # Entry point — instantiates and runs SlipDetectorApp
├── detector.py      # All detection logic: pose estimation, slip heuristic, CSV export
├── ui.py            # Tkinter GUI: file picker, progress bar, results display
├── requirements.txt # Python dependencies
├── README.md
└── output/          # Auto-created on first run; stores slip_events.csv
```

### Module responsibilities

**`main.py`** — Six lines. Imports `SlipDetectorApp` from `ui.py` and calls `.run()`.

**`detector.py`** — Core logic:
- `_select_device()` — picks CUDA / MPS / CPU
- `_PlayerTracker` — per-player rolling history and slip check
- `_compute_height_ratio()` — extracts the vertical compression metric from keypoints
- `_stabilise_frame()` — optical flow camera drift compensation
- `run_detection()` — main loop: reads frames, calls YOLO, updates trackers, emits events
- `save_csv()` — writes `DetectionResult` to CSV

**`ui.py`** — `SlipDetectorApp` Tkinter class:
- `_build_ui()` — constructs the window layout
- `_pick_file()` — opens the OS file dialog
- `_start()` / `_run_detection()` — spawns the detection thread
- `_on_complete()` / `_on_error()` — updates UI on the main thread after detection finishes
- `_open_csv()` — opens the CSV with the OS default app (`open` on macOS, `startfile` on Windows, `xdg-open` on Linux)
