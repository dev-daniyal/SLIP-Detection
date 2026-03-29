"""
Slip detection using YOLOv8 multi-person pose estimation.

Detects all players in each frame simultaneously using the YOLOv8-pose model
with built-in tracking (BoTSORT). For each tracked player, monitors the
vertical compression of body keypoints over time. A rapid collapse within
~0.75 seconds flags a slip event.
"""

import csv
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Data classes — public API consumed by ui.py
# ---------------------------------------------------------------------------

@dataclass
class SlipEvent:
    """A single detected slip event."""
    timestamp: float       # seconds into the video
    frame_number: int
    confidence: float


@dataclass
class DetectionResult:
    """Final result of processing a video."""
    slips: list = field(default_factory=list)
    total_frames: int = 0
    processing_time: float = 0.0
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3 placeholders — scaffold only, do not build yet
# ---------------------------------------------------------------------------

# TODO [Phase 3]: Annotated output video with slip counter overlay
# def write_annotated_video(input_path, output_path, slip_events):
#     """Re-encode the video with a running slip counter and markers at each event."""
#     pass

# TODO [Phase 3]: Bounding boxes around slip events
# def draw_slip_bounding_boxes(frame, detections):
#     """Draw bounding boxes around the player(s) involved in a slip."""
#     pass

# TODO [Phase 3]: Highlight clip extraction
# def extract_highlight_clips(input_path, output_dir, slip_events, padding_sec=2):
#     """Cut short clips around each slip event for quick review."""
#     pass


# ---------------------------------------------------------------------------
# COCO 17-keypoint indices (used by YOLOv8-pose)
# ---------------------------------------------------------------------------

_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

# Keypoints we rely on for the slip heuristic
_KEY_JOINTS = [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE]


# ---------------------------------------------------------------------------
# Per-player tracking state
# ---------------------------------------------------------------------------

class _PlayerTracker:
    """
    Maintains recent pose history for a single tracked player.

    Stores the "height ratio" — the vertical spread of body keypoints
    normalised by bounding-box height.  A standing player has a high ratio
    (~0.4–0.7); a fallen player approaches 0.
    """

    def __init__(self, max_history: int = 20):
        # Each entry: (timestamp, height_ratio, avg_keypoint_confidence)
        self.history: deque[tuple[float, float, float]] = deque(maxlen=max_history)
        self.last_slip_time: float = -999.0
        self.last_seen: float = 0.0       # for stale-tracker cleanup

    def record(self, timestamp: float, height_ratio: float, kp_conf: float):
        self.history.append((timestamp, height_ratio, kp_conf))
        self.last_seen = timestamp

    def check_slip(
        self,
        confidence_threshold: float,
        cooldown_sec: float,
        look_back_sec: float = 0.75,
    ) -> tuple[bool, float]:
        """
        Check whether this player just slipped.

        Compares the current height_ratio against the recent maximum.  A large
        rapid drop (standing → collapsed within *look_back_sec*) is flagged.
        """
        if len(self.history) < 3:
            return False, 0.0

        curr_ts, curr_ratio, curr_conf = self.history[-1]

        # Per-player cooldown — avoid re-flagging the same fall
        if curr_ts - self.last_slip_time < cooldown_sec:
            return False, 0.0

        # Find the max height_ratio in the look-back window
        max_standing_ratio = 0.0
        for ts, ratio, conf in self.history:
            if curr_ts - ts <= look_back_sec and conf > 0.3:
                max_standing_ratio = max(max_standing_ratio, ratio)

        # Need evidence of a clear standing pose recently
        if max_standing_ratio < 0.25:
            return False, 0.0

        # Current pose must look collapsed
        if curr_ratio > 0.18:
            return False, 0.0

        # Confidence = magnitude of collapse, weighted by keypoint quality
        drop = (max_standing_ratio - curr_ratio) / max_standing_ratio
        confidence = drop * min(curr_conf / 0.5, 1.0)

        if confidence >= confidence_threshold:
            self.last_slip_time = curr_ts
            return True, round(confidence, 4)

        return False, 0.0


# ---------------------------------------------------------------------------
# Pose metric extraction
# ---------------------------------------------------------------------------

def _compute_height_ratio(kp_xy, kp_conf, bbox):
    """
    Compute the vertical compression ratio for one detected person.

    Returns (height_ratio, avg_keypoint_confidence) or (None, conf) when
    keypoint quality is too low to make a reliable measurement.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1

    if bbox_h < 10:
        return None, 0.0

    # Filter to key joints with sufficient confidence
    confs = kp_conf[_KEY_JOINTS]
    avg_conf = float(np.mean(confs))
    visible = confs > 0.3

    if visible.sum() < 4:
        return None, avg_conf

    # Vertical span of visible key joints
    y_coords = kp_xy[_KEY_JOINTS, 1][visible]
    vertical_span = float(np.max(y_coords) - np.min(y_coords))

    height_ratio = vertical_span / bbox_h
    return height_ratio, avg_conf


# ---------------------------------------------------------------------------
# Camera drift compensation
# ---------------------------------------------------------------------------

def _stabilise_frame(prev_gray, curr_gray, frame):
    """
    Compensate for minor camera drift between consecutive frames using
    optical-flow-based alignment.  Returns (stabilised_frame, gray_for_next).
    """
    if prev_gray is None:
        return frame, curr_gray

    features = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3,
    )
    if features is None or len(features) < 10:
        return frame, curr_gray

    matched, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    if matched is None:
        return frame, curr_gray

    good_old = features[status.flatten() == 1]
    good_new = matched[status.flatten() == 1]

    if len(good_old) < 6:
        return frame, curr_gray

    transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    if transform is None:
        return frame, curr_gray

    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(transform)
    stabilised = cv2.warpAffine(frame, inv, (w, h))
    stabilised_gray = cv2.cvtColor(stabilised, cv2.COLOR_BGR2GRAY)
    return stabilised, stabilised_gray


# ---------------------------------------------------------------------------
# Core detection loop
# ---------------------------------------------------------------------------

def run_detection(
    video_path: str,
    confidence_threshold: float = 0.65,
    frame_skip: int = 3,
    cooldown_sec: float = 2.0,
    progress_callback=None,
    cancel_check=None,
) -> DetectionResult:
    """
    Process a video file and detect slip events using YOLOv8 pose estimation.

    Args:
        video_path: Path to MP4/MOV file.
        confidence_threshold: Minimum confidence to flag a slip (0.65–0.70).
        frame_skip: Process every Nth frame (default 3).
        cooldown_sec: Per-player cooldown between reported slips.
        progress_callback: Optional callable(current_frame, total_frames).
        cancel_check: Optional callable() returning True to abort.

    Returns:
        DetectionResult with slip events and metadata.
    """
    result = DetectionResult()
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result.warnings.append(f"Could not open video: {video_path}")
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result.total_frames = total_frames

    # YOLOv8 nano pose model — auto-downloads on first run (~6 MB)
    model = YOLO("yolov8n-pose.pt")

    # Per-player state keyed by YOLO track ID
    players: dict[int, _PlayerTracker] = defaultdict(
        lambda: _PlayerTracker(max_history=20)
    )

    frame_idx = 0
    prev_gray = None
    last_log_sec = -1
    consecutive_failures = 0
    max_consecutive_failures = 30  # ~1 second at 30 fps

    while True:
        if cancel_check and cancel_check():
            result.warnings.append("Processing cancelled by user.")
            break

        # For skipped frames, use grab() — advances the codec without decoding.
        if frame_idx % frame_skip != 0:
            if not cap.grab():
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                frame_idx += 1
                continue
            consecutive_failures = 0
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            # Tolerate a run of failures before stopping so a few corrupt
            # frames don't end processing prematurely.
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                if frame_idx < total_frames - max_consecutive_failures:
                    result.warnings.append(
                        f"Video read failed at frame {frame_idx} "
                        f"(expected {total_frames}), possible corruption.")
                break
            frame_idx += 1
            continue
        consecutive_failures = 0

        timestamp = frame_idx / fps
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Stabilise against drone drift
        stabilised, curr_gray_out = _stabilise_frame(prev_gray, curr_gray, frame)
        prev_gray = curr_gray_out

        # --- YOLOv8 multi-person pose tracking ---
        yolo_results = model.track(
            stabilised,
            persist=True,       # maintain track IDs across frames
            conf=0.3,           # detection confidence floor
            iou=0.5,
            verbose=False,
        )

        detections = yolo_results[0]

        # track() returns None for boxes.id when no objects are tracked
        if detections.boxes.id is not None and detections.keypoints is not None:
            track_ids = detections.boxes.id.cpu().numpy().astype(int)
            boxes = detections.boxes.xyxy.cpu().numpy()        # (N, 4)
            kp_xy = detections.keypoints.xy.cpu().numpy()      # (N, 17, 2)
            kp_conf = detections.keypoints.conf.cpu().numpy()  # (N, 17)

            for i, track_id in enumerate(track_ids):
                height_ratio, avg_conf = _compute_height_ratio(
                    kp_xy[i], kp_conf[i], boxes[i],
                )

                if height_ratio is None:
                    # Keypoints too unreliable for this person — skip
                    continue

                tracker = players[track_id]
                tracker.record(timestamp, height_ratio, avg_conf)

                slip, confidence = tracker.check_slip(
                    confidence_threshold, cooldown_sec,
                )
                if slip:
                    event = SlipEvent(
                        timestamp=round(timestamp, 2),
                        frame_number=frame_idx,
                        confidence=confidence,
                    )
                    result.slips.append(event)
                    print(
                        f"  [SLIP] t={event.timestamp:.1f}s  "
                        f"frame={event.frame_number}  "
                        f"player={track_id}  "
                        f"conf={event.confidence:.3f}"
                    )

        # Periodically prune stale trackers to bound memory
        if frame_idx % (frame_skip * 300) == 0:
            stale_ids = [
                tid for tid, trk in players.items()
                if timestamp - trk.last_seen > 10.0
            ]
            for tid in stale_ids:
                del players[tid]

        # Console progress every ~60 seconds of footage
        if int(timestamp) % 60 == 0 and int(timestamp) > 0 and int(timestamp) != last_log_sec:
            last_log_sec = int(timestamp)
            elapsed = time.time() - start_time
            print(
                f"  Progress: {timestamp / 60:.0f} min of footage processed "
                f"({elapsed:.0f}s elapsed, {len(result.slips)} slips so far, "
                f"{len(players)} active tracks)"
            )

        # UI progress callback
        if progress_callback:
            progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()

    result.processing_time = round(time.time() - start_time, 2)

    # Final console summary
    avg_conf = (
        float(np.mean([s.confidence for s in result.slips]))
        if result.slips
        else 0.0
    )
    print(f"\n{'=' * 50}")
    print(f"  Detection complete.")
    print(f"  Total slips: {len(result.slips)}")
    print(f"  Processing time: {result.processing_time:.1f}s")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"{'=' * 50}\n")

    return result


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(result: DetectionResult, output_dir: str = "output") -> str:
    """
    Save slip events to a CSV file.

    Returns the path to the written CSV.
    """
    # TODO: output_dir is relative to CWD, not the script's directory.
    # If the user launches from a different working directory, the CSV
    # will be written there.  Consider using Path(__file__).parent / output_dir.
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "slip_events.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_number", "confidence"])
        for event in result.slips:
            writer.writerow([event.timestamp, event.frame_number, event.confidence])

    print(f"  CSV saved to {csv_path}")
    return csv_path
