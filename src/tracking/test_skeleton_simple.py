#!/usr/bin/env python3
"""
RealSense Skeleton + Breathing Detection

Uses IR stream (no projector dots) + MediaPipe pose + depth fusion.
Median + EMA smoothing for responsive, clean breathing signal.
Peak detection for BPM calculation.

Run with: sudo ./venv/bin/python src/tracking/test_skeleton_simple.py
Controls: q=quit, r=reset, s=screenshot
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import time
from collections import deque

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pip install pyrealsense2-macosx")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("Error: pip install mediapipe")
    sys.exit(1)


class BreathingDetector:
    """
    Breathing detector: median filter + EMA smoothing + peak-based BPM.
    Responsive and accurate - no heavy filters that add latency.
    """

    def __init__(self):
        self.raw_buffer = deque(maxlen=200)     # ~10s at 20fps
        self.smoothed = deque(maxlen=200)
        self.timestamps = deque(maxlen=200)

        # State
        self.phase = "calibrating"
        self.signal = 0.0
        self.display_signal = 0.0   # Lerp-interpolated for smooth circle
        self.bpm = 0.0
        self.breath_count = 0

        # Phase detection with debounce
        self._last_raw_phase = "hold"
        self._phase_hold_count = 0
        self._confirmed_phase = "hold"

        # Peak detection for BPM
        self._peak_times = deque(maxlen=20)
        self._trough_times = deque(maxlen=20)
        self._prev_smoothed = None
        self._prev_derivative = 0.0
        self._was_rising = False

    def update(self, depth_mm, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        self.raw_buffer.append(depth_mm)
        self.timestamps.append(timestamp)

        if len(self.raw_buffer) < 10:
            self.phase = "calibrating"
            return

        # Step 1: Median of last 7 samples (kills spike noise)
        median_val = np.median(list(self.raw_buffer)[-7:])

        # Step 2: EMA on top (smooth but responsive)
        if len(self.smoothed) > 0:
            alpha = 0.15
            avg = alpha * median_val + (1 - alpha) * self.smoothed[-1]
        else:
            avg = median_val
        self.smoothed.append(avg)

        if len(self.smoothed) < 8:
            self.phase = "calibrating"
            return

        # --- Derivative for phase detection (8 frames apart) ---
        recent = list(self.smoothed)
        derivative = recent[-1] - recent[-8]

        # Phase detection: threshold at 2mm to ignore noise
        if derivative < -2.0:
            raw_phase = "inhale"
        elif derivative > 2.0:
            raw_phase = "exhale"
        else:
            raw_phase = "hold"

        # Debounce: 3 consistent frames to switch
        if raw_phase == self._last_raw_phase:
            self._phase_hold_count += 1
        else:
            self._phase_hold_count = 1
            self._last_raw_phase = raw_phase

        if self._phase_hold_count >= 3 and raw_phase != self._confirmed_phase:
            old_phase = self._confirmed_phase
            self._confirmed_phase = raw_phase

            # Count breath on inhale->exhale
            if old_phase == "inhale" and raw_phase == "exhale":
                self.breath_count += 1
                self._peak_times.append(timestamp)

            # Also track exhale->inhale for BPM
            if old_phase == "exhale" and raw_phase == "inhale":
                self._trough_times.append(timestamp)

        self.phase = self._confirmed_phase

        # --- BPM from breath cycle intervals ---
        self._update_bpm()

        # --- Normalized signal for circle (-1 to +1) ---
        if len(self.smoothed) >= 30:
            buf = list(self.smoothed)[-90:]
            min_d, max_d = min(buf), max(buf)
            range_d = max_d - min_d if max_d > min_d else 1
            # Invert: closer = inhale = positive signal
            self.signal = -((avg - min_d) / range_d * 2 - 1)
            self.signal = max(-1.0, min(1.0, self.signal))

        # Smooth lerp for circle animation
        self.display_signal += (self.signal - self.display_signal) * 0.25

    def _update_bpm(self):
        """BPM from inhale->exhale transition intervals."""
        times = list(self._peak_times)
        if len(times) >= 2:
            recent = times[-6:]
            intervals = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            valid = [iv for iv in intervals if 1.5 < iv < 12.0]
            if valid:
                self.bpm = 60.0 / np.mean(valid)

    def reset(self):
        self.raw_buffer.clear()
        self.smoothed.clear()
        self.timestamps.clear()
        self.phase = "calibrating"
        self.signal = 0.0
        self.display_signal = 0.0
        self.bpm = 0.0
        self.breath_count = 0
        self._confirmed_phase = "hold"
        self._last_raw_phase = "hold"
        self._phase_hold_count = 0
        self._peak_times.clear()
        self._trough_times.clear()


def draw_breathing_circle(display, breath, x, y):
    """Animated breathing circle with glow."""
    base_r = 55
    radius = int(base_r + breath.display_signal * 25)
    radius = max(20, radius)

    # Phase colors (BGR)
    colors = {
        "inhale":      ((200, 255, 100), (100, 200, 50)),
        "exhale":      ((180, 140, 255), (120, 80, 200)),
        "calibrating": ((120, 120, 120), (60, 60, 60)),
        "hold":        ((200, 200, 180), (100, 100, 90)),
    }
    primary, glow = colors.get(breath.phase, colors["hold"])

    # Glow rings
    cv2.circle(display, (x, y), radius + 12, glow, 2, cv2.LINE_AA)
    cv2.circle(display, (x, y), radius + 6, primary, 2, cv2.LINE_AA)

    # Filled circle with transparency
    overlay = display.copy()
    cv2.circle(overlay, (x, y), radius, primary, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Inner highlight
    inner_r = max(5, radius - 20)
    highlight = tuple(min(255, c + 60) for c in primary)
    cv2.circle(display, (x - 5, y - 5), inner_r // 2, highlight, -1, cv2.LINE_AA)

    # Phase label
    label = breath.phase.upper()
    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(display, label, (x - ts[0]//2, y + radius + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, primary, 2, cv2.LINE_AA)

    # BPM or breath count above
    if breath.bpm > 0:
        text = f"{breath.bpm:.0f} BPM"
    else:
        text = f"Breaths: {breath.breath_count}"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(display, text, (x - ts[0]//2, y - radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_waveform(display, breath, x, y, w, h):
    """Waveform showing smoothed signal with filled area."""
    # Background
    cv2.rectangle(display, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.rectangle(display, (x, y), (x + w, y + h), (60, 60, 60), 1)

    mid_y = y + h // 2
    cv2.line(display, (x, mid_y), (x + w, mid_y), (40, 40, 40), 1)

    if len(breath.smoothed) < 3:
        return

    data = list(breath.smoothed)
    mean_d = np.mean(data)
    max_dev = max(max(abs(d - mean_d) for d in data), 0.5)

    # Build points
    n = len(data)
    points = []
    for i, val in enumerate(data):
        px = x + int((i / max(n - 1, 1)) * w)
        norm = (val - mean_d) / max_dev
        py = mid_y - int(norm * h * 0.4)
        py = max(y + 2, min(y + h - 2, py))
        points.append((px, py))

    # Filled area
    if len(points) >= 2:
        fill_pts = [(points[0][0], mid_y)] + points + [(points[-1][0], mid_y)]
        overlay = display.copy()
        cv2.fillPoly(overlay, [np.array(fill_pts, dtype=np.int32)], (0, 80, 40))
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

    # Main line (teal)
    for i in range(1, len(points)):
        cv2.line(display, points[i-1], points[i], (0, 220, 180), 2, cv2.LINE_AA)

    # Raw signal faintly
    if len(breath.raw_buffer) > 2:
        raw = list(breath.raw_buffer)
        raw_mean = np.mean(raw)
        raw_range = max(max(raw) - min(raw), 1)
        nr = len(raw)
        for i in range(1, nr):
            px1 = x + int(((i-1) / max(nr-1, 1)) * w)
            px2 = x + int((i / max(nr-1, 1)) * w)
            n1 = (raw[i-1] - raw_mean) / raw_range
            n2 = (raw[i] - raw_mean) / raw_range
            py1 = mid_y - int(n1 * h * 0.35)
            py2 = mid_y - int(n2 * h * 0.35)
            py1 = max(y+2, min(y+h-2, py1))
            py2 = max(y+2, min(y+h-2, py2))
            cv2.line(display, (px1, py1), (px2, py2), (50, 50, 50), 1)

    # Label + range info
    range_mm = max(data) - min(data)
    cv2.putText(display, f"Chest Depth (smoothed) | Range: {range_mm:.1f}mm", (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 150), 1, cv2.LINE_AA)


def draw_info_panel(display, fps, chest_depth, breath, stream_mode):
    """Semi-transparent HUD."""
    pw, ph = 200, 130
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.rectangle(display, (10, 10), (10 + pw, 10 + ph), (60, 60, 60), 1)

    phase_colors = {"inhale": (200, 255, 100), "exhale": (180, 140, 255),
                    "hold": (200, 200, 180), "calibrating": (120, 120, 120)}

    lines = [
        (f"FPS: {fps:.0f}", (0, 255, 0)),
        (f"Depth: {chest_depth:.0f}mm", (0, 255, 255)),
        (f"Phase: {breath.phase}", phase_colors.get(breath.phase, (200, 200, 200))),
    ]
    if breath.bpm > 0:
        lines.append((f"BPM: {breath.bpm:.1f}", (255, 200, 100)))
    else:
        lines.append((f"Breaths: {breath.breath_count}", (150, 150, 150)))

    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (20, 32 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    mode = "IR" if stream_mode == "infrared" else "RGB"
    cv2.putText(display, f"{mode} | q:quit r:reset s:save", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)


def main():
    print("\n" + "=" * 50)
    print("  RealSense Skeleton + Breathing Detection")
    print("=" * 50)

    # --- Start RealSense ---
    configs = [
        ("Depth + IR (recommended)", [
            (rs.stream.depth, 640, 480, rs.format.z16, 30),
            (rs.stream.infrared, 640, 480, rs.format.y8, 30),
        ], "infrared"),
        ("Depth + Color 424x240", [
            (rs.stream.depth, 424, 240, rs.format.z16, 30),
            (rs.stream.color, 424, 240, rs.format.bgr8, 30),
        ], "color"),
        ("Depth + Color 640x480", [
            (rs.stream.depth, 640, 480, rs.format.z16, 30),
            (rs.stream.color, 640, 480, rs.format.bgr8, 30),
        ], "color"),
    ]

    pipeline = None
    profile = None
    stream_mode = None

    for name, streams, mode in configs:
        print(f"\nTrying: {name}...")
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            for stream_args in streams:
                config.enable_stream(*stream_args)
            profile = pipeline.start(config)
            stream_mode = mode
            print(f"  SUCCESS! Using: {name}")
            break
        except RuntimeError as e:
            print(f"  Failed: {str(e)[:60]}")
            try:
                pipeline.stop()
            except:
                pass
            pipeline = None
            profile = None

    if profile is None:
        print("\nAll configurations failed!")
        print("Try: unplug camera, restart Mac, replug, run with sudo")
        return

    depth_sensor = profile.get_device().first_depth_sensor()

    if stream_mode == "infrared":
        try:
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            print("  IR projector disabled (clean image for pose detection)")
        except Exception as e:
            print(f"  Warning: Could not disable IR emitter: {e}")

    align = rs.align(rs.stream.infrared if stream_mode == "infrared" else rs.stream.color)

    # --- MediaPipe ---
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

    # --- Breathing ---
    breath = BreathingDetector()

    cv2.namedWindow("Skeleton", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Skeleton", 960, 720)

    print("\nRunning! Press 'q' to quit, 'r' to reset, 's' to save screenshot\n")

    frame_count = 0
    start_time = time.time()
    consecutive_failures = 0
    actual_fps = 20.0
    last_fps_time = time.time()
    fps_frames = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                consecutive_failures = 0
            except RuntimeError:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("\nToo many frame failures. Camera needs reset.")
                    break
                continue

            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            if not depth_frame:
                continue

            if stream_mode == "infrared":
                ir_frame = aligned.get_infrared_frame()
                if not ir_frame:
                    continue
                ir_image = np.asanyarray(ir_frame.get_data())
                color_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            else:
                color_frame = aligned.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())

            frame_count += 1
            fps_frames += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                actual_fps = fps_frames / (now - last_fps_time)
                fps_frames = 0
                last_fps_time = now

            depth_image = np.asanyarray(depth_frame.get_data())
            h, w = color_image.shape[:2]

            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            chest_depth = 0
            display = color_image.copy()

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
                )

                lm = results.pose_landmarks.landmark
                left_sh, right_sh = lm[11], lm[12]

                if left_sh.visibility > 0.5 and right_sh.visibility > 0.5:
                    cx = int((left_sh.x + right_sh.x) / 2 * w)
                    cy = int((left_sh.y + right_sh.y) / 2 * h)

                    # Sample chest depth (31x31 grid, step 2)
                    depths = []
                    for dy in range(-15, 16, 2):
                        for dx in range(-15, 16, 2):
                            px = max(0, min(w-1, cx + dx))
                            py = max(0, min(h-1, cy + dy))
                            d = depth_frame.get_distance(px, py)
                            if 0.1 < d < 5.0:
                                depths.append(d)

                    if depths:
                        chest_depth = np.median(depths) * 1000
                        breath.update(chest_depth, now)

                        # Chest crosshair
                        cv2.circle(display, (cx, cy), 12, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.line(display, (cx-18, cy), (cx+18, cy), (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.line(display, (cx, cy-18), (cx, cy+18), (0, 255, 255), 1, cv2.LINE_AA)

            # --- UI ---
            draw_breathing_circle(display, breath, w - 110, h // 2 - 30)
            draw_waveform(display, breath, 20, h - 105, w - 40, 90)
            draw_info_panel(display, actual_fps, chest_depth, breath, stream_mode)

            # Depth preview
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_small = cv2.resize(depth_color, (140, 105))
            display[10:115, w-150:w-10] = depth_small
            cv2.rectangle(display, (w-150, 10), (w-10, 115), (60, 60, 60), 1)

            cv2.imshow("Skeleton", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                breath.reset()
                frame_count = 0
                start_time = time.time()
                print("Reset!")
            elif key == ord('s'):
                os.makedirs("outputs", exist_ok=True)
                fname = f"outputs/skeleton_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    except KeyboardInterrupt:
        pass
    finally:
        pose.close()
        pipeline.stop()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nSession: {frame_count} frames in {elapsed:.0f}s ({frame_count/max(elapsed,1):.1f} FPS)")
    if breath.bpm > 0:
        print(f"Breathing: {breath.bpm:.1f} BPM, {breath.breath_count} breaths detected")


if __name__ == "__main__":
    main()
