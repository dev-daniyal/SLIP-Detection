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

- **MediaPipe Pose** (full model, complexity 2) estimates body landmarks per frame
- A vertical-collapse heuristic flags when a player's upper body drops to ground level
- **Optical flow stabilisation** compensates for minor drone drift
- Every 3rd frame is sampled to keep processing time reasonable on ~1 hour footage
- A 2-second cooldown between events prevents duplicate detections of the same fall

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
