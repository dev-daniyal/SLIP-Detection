# Slip Detector — Ultimate Frisbee

Detects moments where players fully fall/slip to the ground in drone footage of Ultimate Frisbee games. Outputs a total slip count and timestamps to CSV.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

```bash
python main.py
```

1. A file picker opens — select an MP4 or MOV drone footage file
2. Click **Run Detection**
3. Watch the progress bar; console prints updates every 60s of footage
4. When done, view the slip count in the UI and click **Open CSV** for details

## Output

Results are saved to `output/slip_events.csv` with columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Seconds into the video |
| `frame_number` | Exact frame index |
| `confidence` | Detection confidence score |

## How It Works

- **YOLOv8 Pose** (`yolov8n-pose.pt`) detects and tracks ALL players simultaneously with per-person bounding boxes and 17 COCO keypoints
- Built-in BoTSORT tracker maintains consistent player IDs across frames
- Per-player **temporal slip detection**: monitors the vertical compression of shoulder/hip/ankle keypoints over a ~0.75s window; a rapid collapse flags a fall
- **Optical flow stabilisation** compensates for minor drone drift
- Every 3rd frame is sampled (skipped frames use `grab()` for speed)
- Per-player 2-second cooldown prevents duplicate detections of the same fall

## Configuration

Key parameters in `detector.py` → `run_detection()`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `confidence_threshold` | 0.65 | Raise to reduce false positives |
| `frame_skip` | 3 | Higher = faster but may miss quick falls |
| `cooldown_sec` | 2.0 | Min gap between reported slips |

## Project Structure

```
├── main.py          # Entry point, launches UI
├── detector.py      # Pose estimation + slip detection logic
├── ui.py            # Tkinter file picker and progress window
├── output/          # Auto-created, stores CSV results
├── requirements.txt
└── README.md
```
