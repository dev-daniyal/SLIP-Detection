"""
Slip detection logic using MediaPipe Pose estimation.

Analyzes video frames to detect when players fall to the ground by tracking
vertical collapse of pose landmarks (shoulders/hips dropping to knee/ankle level).
"""

import csv
import os
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np


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
# Core detection
# ---------------------------------------------------------------------------

# Landmark indices used for vertical-ratio slip heuristic
_SHOULDER_L = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
_SHOULDER_R = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
_HIP_L = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
_HIP_R = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
_KNEE_L = mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
_KNEE_R = mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value
_ANKLE_L = mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
_ANKLE_R = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value


def _mean_y(landmarks, indices):
    """Return the mean normalised y-coordinate for the given landmark indices."""
    return np.mean([landmarks[i].y for i in indices])


def _mean_visibility(landmarks, indices):
    """Return the mean visibility score for the given landmark indices."""
    return np.mean([landmarks[i].visibility for i in indices])


def _is_slip(landmarks, confidence_threshold: float = 0.65) -> tuple[bool, float]:
    """
    Determine whether the pose represents a slip/fall.

    A player is considered fallen when the vertical distance between their
    upper body (shoulders) and lower body (ankles) is very small — i.e. the
    torso has collapsed to near ground level.

    Returns (is_slip, confidence).
    """
    key_indices = [_SHOULDER_L, _SHOULDER_R, _HIP_L, _HIP_R,
                   _KNEE_L, _KNEE_R, _ANKLE_L, _ANKLE_R]
    avg_vis = _mean_visibility(landmarks, key_indices)

    # If pose confidence is too low, we can't make a reliable call
    if avg_vis < 0.4:
        return False, avg_vis

    shoulder_y = _mean_y(landmarks, [_SHOULDER_L, _SHOULDER_R])
    hip_y = _mean_y(landmarks, [_HIP_L, _HIP_R])
    ankle_y = _mean_y(landmarks, [_ANKLE_L, _ANKLE_R])

    # In normalised coords, y increases downward. When standing, shoulders are
    # well above ankles (shoulder_y << ankle_y). When fallen, shoulders drop
    # close to ankle level.
    total_height = abs(ankle_y - shoulder_y)

    # Very small vertical span → person is horizontal / on the ground
    if total_height < 0.01:
        return False, avg_vis  # degenerate, skip

    # Ratio of upper-body drop: how close hips are to ankles relative to total height
    # When standing this is ~0.4-0.5; when fallen it approaches 0.0-0.15
    torso_ratio = abs(ankle_y - hip_y) / total_height

    # Confidence that this is a slip: low torso_ratio + low total_height
    # Combined heuristic score
    height_score = max(0.0, 1.0 - (total_height / 0.35))  # 0.35 is typical standing span
    ratio_score = max(0.0, 1.0 - (torso_ratio / 0.45))

    confidence = 0.5 * height_score + 0.5 * ratio_score
    # Weight by visibility so low-quality detections score lower
    confidence *= min(avg_vis / 0.7, 1.0)

    return confidence >= confidence_threshold, confidence


def _stabilise_frame(prev_gray, curr_gray, frame):
    """
    Compensate for minor camera drift between consecutive frames using
    optical-flow-based alignment. Returns the stabilised frame.
    """
    if prev_gray is None:
        return frame, curr_gray

    # Find feature points in the previous frame
    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                       minDistance=30, blockSize=3)
    if features is None or len(features) < 10:
        return frame, curr_gray

    # Calculate optical flow
    matched, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    if matched is None:
        return frame, curr_gray

    # Filter valid matches
    good_old = features[status.flatten() == 1]
    good_new = matched[status.flatten() == 1]

    if len(good_old) < 6:
        return frame, curr_gray

    # Estimate rigid transform (translation + rotation only)
    transform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    if transform is None:
        return frame, curr_gray

    h, w = frame.shape[:2]
    # Invert the motion to stabilise
    inv = cv2.invertAffineTransform(transform)
    stabilised = cv2.warpAffine(frame, inv, (w, h))
    stabilised_gray = cv2.cvtColor(stabilised, cv2.COLOR_BGR2GRAY)
    return stabilised, stabilised_gray


def run_detection(
    video_path: str,
    confidence_threshold: float = 0.65,
    frame_skip: int = 3,
    cooldown_sec: float = 2.0,
    progress_callback=None,
    cancel_check=None,
) -> DetectionResult:
    """
    Process a video file and detect slip events.

    Args:
        video_path: Path to MP4/MOV file.
        confidence_threshold: Minimum confidence to flag a slip (0.65–0.70).
        frame_skip: Process every Nth frame (default 3).
        cooldown_sec: Minimum seconds between reported slips to avoid duplicates.
        progress_callback: Optional callable(current_frame, total_frames) for UI updates.
        cancel_check: Optional callable() that returns True to abort processing.

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

    # MediaPipe Pose — single-person model applied per detection window
    # MediaPipe's base Pose solution processes one person at a time, but by
    # scanning the full frame it will lock onto the most prominent pose.
    # For multi-person coverage we rely on frequent sampling and the fact that
    # slips are visually dominant events that MediaPipe tends to pick up.
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,        # full / heavy model
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    last_slip_time = -999.0  # timestamp of last recorded slip
    prev_gray = None
    last_log_sec = -1  # for console progress logging

    while True:
        if cancel_check and cancel_check():
            result.warnings.append("Processing cancelled by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for performance
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Stabilise against drone drift
        stabilised, curr_gray_out = _stabilise_frame(prev_gray, curr_gray, frame)
        prev_gray = curr_gray_out

        # Run pose estimation
        rgb = cv2.cvtColor(stabilised, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            slip_detected, confidence = _is_slip(landmarks, confidence_threshold)

            if slip_detected and (timestamp - last_slip_time) >= cooldown_sec:
                event = SlipEvent(
                    timestamp=round(timestamp, 2),
                    frame_number=frame_idx,
                    confidence=round(confidence, 4),
                )
                result.slips.append(event)
                last_slip_time = timestamp
                print(f"  [SLIP] t={event.timestamp:.1f}s  frame={event.frame_number}  "
                      f"conf={event.confidence:.3f}")
        else:
            # Pose confidence too low to detect anyone
            vis_key = "low"
            if int(timestamp) % 60 == 0 and int(timestamp) != last_log_sec:
                result.warnings.append(
                    f"No pose detected at t={timestamp:.1f}s (may be transitional)")

        # Console progress every ~60 seconds of footage
        current_minute = int(timestamp // 60)
        if current_minute > 0 and int(timestamp) % 60 == 0 and int(timestamp) != last_log_sec:
            last_log_sec = int(timestamp)
            elapsed = time.time() - start_time
            print(f"  Progress: {timestamp/60:.0f} min of footage processed "
                  f"({elapsed:.0f}s elapsed, {len(result.slips)} slips so far)")

        # UI progress callback
        if progress_callback:
            progress_callback(frame_idx, total_frames)

        frame_idx += 1

    pose.close()
    cap.release()

    result.processing_time = round(time.time() - start_time, 2)

    # Final summary to console
    avg_conf = (np.mean([s.confidence for s in result.slips])
                if result.slips else 0.0)
    print(f"\n{'='*50}")
    print(f"  Detection complete.")
    print(f"  Total slips: {len(result.slips)}")
    print(f"  Processing time: {result.processing_time:.1f}s")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"{'='*50}\n")

    return result


def save_csv(result: DetectionResult, output_dir: str = "output") -> str:
    """
    Save slip events to a CSV file.

    Returns the path to the written CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "slip_events.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_number", "confidence"])
        for event in result.slips:
            writer.writerow([event.timestamp, event.frame_number, event.confidence])

    print(f"  CSV saved to {csv_path}")
    return csv_path
