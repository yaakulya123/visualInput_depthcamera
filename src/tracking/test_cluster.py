#!/usr/bin/env python3
"""
RealSense People Clustering - 3D Proximity Grouping

Standalone app that detects multiple people with the RealSense D435,
computes real-world 3D distances between them, and groups nearby people
(within a configurable threshold) into clusters.

- YOLO11n-pose + ByteTrack for multi-person detection with stable IDs
- RealSense depth deprojection for true 3D world coordinates
- Union-find clustering by Euclidean distance
- Polished HUD with distance lines, cluster panel, depth preview

Run with: sudo ./venv/bin/python src/tracking/test_cluster.py
Controls: q=quit, r=reset, s=screenshot, t=cycle threshold
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pip install pyrealsense2-macosx")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: pip install ultralytics>=8.1.0")
    sys.exit(1)

try:
    import supervision as sv
except ImportError:
    print("Error: pip install supervision>=0.19.0")
    sys.exit(1)

from src.tracking.person_tracker import SKELETON_CONNECTIONS
from src.tracking.keypoint_adapter import get_shoulder_center
from src.tracking.person_state import PERSON_COLORS
from src.tracking.cluster_detector import (
    PeopleClusterDetector, PersonPosition, ClusterResult, PairDistance,
)


# ---------------------------------------------------------------------------
#  Per-person info (lightweight, no breathing/audio)
# ---------------------------------------------------------------------------

@dataclass
class PersonInfo:
    """Per-person tracking state for clustering."""
    person_id: int
    color: Tuple[int, int, int]
    last_seen: float = field(default_factory=time.time)
    shoulder_depth_mm: float = 0.0
    world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_valid: bool = False


# ---------------------------------------------------------------------------
#  Multi-Person RealSense Tracker (clustering-focused)
# ---------------------------------------------------------------------------

class ClusteringTracker:
    """
    YOLO11n-pose + ByteTrack on RealSense IR/color stream.
    Tracks people, computes 3D positions, delegates clustering.
    """

    STALE_TIMEOUT = 5.0

    def __init__(self, device: str = "mps"):
        print("[ClusteringTracker] Loading YOLO11n-pose...")
        self.model = YOLO("yolo11n-pose.pt")
        self.device = device
        print(f"[ClusteringTracker] Model loaded. Device: {device}")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        self.persons: Dict[int, PersonInfo] = {}
        self._color_index = 0
        self.cluster_detector = PeopleClusterDetector(threshold_meters=1.0)
        self.last_cluster_result = ClusterResult()

    def process_frame(self, frame: np.ndarray, depth_frame, intrinsics):
        """
        Detect + track all people, compute 3D positions, cluster.

        Returns:
            list of (track_id, bbox, keypoints, kpt_confs, PersonInfo)
        """
        h, w = frame.shape[:2]
        now = time.time()

        results = self.model(
            frame, device=self.device, conf=0.5, verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            self._cleanup_stale(now)
            self.last_cluster_result = ClusterResult()
            return []

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        box_confs = results.boxes.conf.cpu().numpy()

        all_keypoints = None
        all_kpt_confs = None
        if results.keypoints is not None and results.keypoints.xy is not None:
            all_keypoints = results.keypoints.xy.cpu().numpy()
            all_kpt_confs = results.keypoints.conf.cpu().numpy()

        sv_detections = sv.Detections(xyxy=boxes_xyxy, confidence=box_confs)
        tracked = self.tracker.update_with_detections(sv_detections)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            self._cleanup_stale(now)
            self.last_cluster_result = ClusterResult()
            return []

        detections = []
        positions: List[PersonPosition] = []

        for i in range(len(tracked.xyxy)):
            track_id = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i].astype(int)

            kpts = None
            kpt_confs = None
            if all_keypoints is not None:
                matched_idx = self._match_box(bbox, boxes_xyxy)
                if matched_idx is not None:
                    kpts = all_keypoints[matched_idx]
                    kpt_confs = all_kpt_confs[matched_idx]

            person = self._get_or_create(track_id)
            person.last_seen = now

            # Compute 3D world position from shoulder center + depth
            pixel_x, pixel_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            person.depth_valid = False

            if kpts is not None and kpt_confs is not None:
                shoulder_center = get_shoulder_center(kpts, kpt_confs, 0.3)
                if shoulder_center is not None:
                    sx, sy = int(shoulder_center[0]), int(shoulder_center[1])
                    pixel_x, pixel_y = float(sx), float(sy)
                    depth_m = self._sample_depth(depth_frame, sx, sy, w, h)
                    if depth_m > 0:
                        person.shoulder_depth_mm = depth_m * 1000
                        # Deproject to 3D world coordinates
                        point_3d = rs.rs2_deproject_pixel_to_point(
                            intrinsics, [float(sx), float(sy)], depth_m
                        )
                        person.world_pos = (point_3d[0], point_3d[1], point_3d[2])
                        person.depth_valid = True

            positions.append(PersonPosition(
                track_id=track_id,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                world_x=person.world_pos[0],
                world_y=person.world_pos[1],
                world_z=person.world_pos[2],
                depth_valid=person.depth_valid,
            ))

            detections.append((track_id, bbox, kpts, kpt_confs, person))

        # Cluster by 3D proximity
        self.last_cluster_result = self.cluster_detector.compute(positions)

        # Update person colors based on cluster assignment
        self._assign_cluster_colors()

        self._cleanup_stale(now)
        return detections

    def _assign_cluster_colors(self):
        """Assign same color to all people in the same cluster."""
        result = self.last_cluster_result
        cluster_ids = sorted(result.clusters.keys())
        for ci, cluster_id in enumerate(cluster_ids):
            color = PERSON_COLORS[ci % len(PERSON_COLORS)]
            for tid in result.clusters[cluster_id]:
                if tid in self.persons:
                    self.persons[tid].color = color

    def _sample_depth(self, depth_frame, cx, cy, w, h, grid_half=2):
        """Sample median depth around a pixel."""
        depths = []
        for dy in range(-grid_half, grid_half + 1):
            for dx in range(-grid_half, grid_half + 1):
                px = max(0, min(w - 1, cx + dx))
                py = max(0, min(h - 1, cy + dy))
                d = depth_frame.get_distance(px, py)
                if 0.1 < d < 5.0:
                    depths.append(d)
        return float(np.median(depths)) if depths else 0.0

    def _get_or_create(self, track_id: int) -> PersonInfo:
        if track_id in self.persons:
            return self.persons[track_id]
        color = PERSON_COLORS[self._color_index % len(PERSON_COLORS)]
        self._color_index += 1
        person = PersonInfo(person_id=track_id, color=color)
        self.persons[track_id] = person
        return person

    def _cleanup_stale(self, now: float):
        stale = [pid for pid, p in self.persons.items()
                 if (now - p.last_seen) > self.STALE_TIMEOUT]
        for pid in stale:
            del self.persons[pid]

    def _match_box(self, tracked_box, original_boxes):
        best_iou = 0.0
        best_idx = None
        for i, orig in enumerate(original_boxes):
            x1 = max(tracked_box[0], orig[0])
            y1 = max(tracked_box[1], orig[1])
            x2 = min(tracked_box[2], orig[2])
            y2 = min(tracked_box[3], orig[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            if intersection == 0:
                continue
            area1 = (tracked_box[2] - tracked_box[0]) * (tracked_box[3] - tracked_box[1])
            area2 = (orig[2] - orig[0]) * (orig[3] - orig[1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx if best_iou > 0.3 else None

    def reset(self):
        self.persons.clear()
        self._color_index = 0
        self.last_cluster_result = ClusterResult()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )


# ---------------------------------------------------------------------------
#  Drawing functions
# ---------------------------------------------------------------------------

def draw_skeleton_coco(display, kpts, kpt_confs, color, thickness=2):
    """Draw COCO 17-keypoint skeleton."""
    for idx in range(17):
        if kpt_confs[idx] > 0.3:
            x, y = int(kpts[idx][0]), int(kpts[idx][1])
            cv2.circle(display, (x, y), 3, color, -1)
    for i, j in SKELETON_CONNECTIONS:
        if kpt_confs[i] > 0.3 and kpt_confs[j] > 0.3:
            pt1 = (int(kpts[i][0]), int(kpts[i][1]))
            pt2 = (int(kpts[j][0]), int(kpts[j][1]))
            cv2.line(display, pt1, pt2, color, thickness)


def draw_distance_lines(display, detections, cluster_result, tracker):
    """
    Draw lines between people showing distance.
    - Bright colored line for clustered pairs (same cluster)
    - Faint gray line for near-but-not-clustered pairs (< 2m)
    - Nothing for far pairs
    """
    # Build pixel position map from detections
    pixel_map: Dict[int, Tuple[int, int]] = {}
    for track_id, bbox, kpts, kpt_confs, person in detections:
        if kpts is not None and kpt_confs is not None:
            sc = get_shoulder_center(kpts, kpt_confs, 0.3)
            if sc is not None:
                pixel_map[track_id] = (int(sc[0]), int(sc[1]))
                continue
        # Fallback to bbox center
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        pixel_map[track_id] = (cx, cy)

    for pair in cluster_result.pairwise_distances:
        if pair.id_a not in pixel_map or pair.id_b not in pixel_map:
            continue
        if pair.distance_m == float('inf') or pair.distance_m > 2.0:
            continue

        pt_a = pixel_map[pair.id_a]
        pt_b = pixel_map[pair.id_b]
        mid_x = (pt_a[0] + pt_b[0]) // 2
        mid_y = (pt_a[1] + pt_b[1]) // 2

        # Check if same cluster
        same_cluster = (
            cluster_result.person_to_cluster.get(pair.id_a) ==
            cluster_result.person_to_cluster.get(pair.id_b)
        )

        if same_cluster and pair.distance_m <= tracker.cluster_detector.threshold_meters:
            # Bright colored line
            cluster_id = cluster_result.person_to_cluster.get(pair.id_a, 0)
            cluster_ids = sorted(cluster_result.clusters.keys())
            ci = cluster_ids.index(cluster_id) if cluster_id in cluster_ids else 0
            color = PERSON_COLORS[ci % len(PERSON_COLORS)]
            cv2.line(display, pt_a, pt_b, color, 2, cv2.LINE_AA)
            # Distance label with background
            label = f"{pair.distance_m:.2f}m"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(display,
                          (mid_x - ts[0]//2 - 3, mid_y - ts[1] - 3),
                          (mid_x + ts[0]//2 + 3, mid_y + 3),
                          (0, 0, 0), -1)
            cv2.putText(display, label, (mid_x - ts[0]//2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        else:
            # Faint gray line for near but not clustered
            cv2.line(display, pt_a, pt_b, (60, 60, 60), 1, cv2.LINE_AA)
            label = f"{pair.distance_m:.2f}m"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(display, label, (mid_x - ts[0]//2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)


def draw_cluster_panel(display, cluster_result, tracker, x, y):
    """Draw cluster info panel (bottom-left)."""
    clusters = cluster_result.clusters
    if not clusters:
        return

    num_clusters = len(clusters)
    panel_w = 200
    panel_h = 25 + num_clusters * 28
    overlay = display.copy()
    cv2.rectangle(overlay, (x, y - panel_h), (x + panel_w, y), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.rectangle(display, (x, y - panel_h), (x + panel_w, y), (60, 60, 60), 1)

    cv2.putText(display, f"CLUSTERS ({num_clusters})", (x + 5, y - panel_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    cluster_ids = sorted(clusters.keys())
    for i, cid in enumerate(cluster_ids):
        members = clusters[cid]
        color = PERSON_COLORS[i % len(PERSON_COLORS)]
        row_y = y - panel_h + 28 + i * 28

        # Color swatch
        cv2.rectangle(display, (x + 8, row_y - 8), (x + 22, row_y + 6), color, -1)
        cv2.rectangle(display, (x + 8, row_y - 8), (x + 22, row_y + 6), (200, 200, 200), 1)

        # Cluster info
        member_str = ", ".join(f"P{m}" for m in members)
        if len(member_str) > 20:
            member_str = member_str[:17] + "..."
        label = f"C{cid}: {len(members)}p - {member_str}"
        cv2.putText(display, label, (x + 28, row_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def draw_info_panel(display, fps, person_count, cluster_count, threshold, stream_mode):
    """Semi-transparent info HUD (top-left)."""
    pw, ph = 230, 130
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.rectangle(display, (10, 10), (10 + pw, 10 + ph), (60, 60, 60), 1)

    lines = [
        (f"FPS: {fps:.0f}", (0, 255, 0)),
        (f"People: {person_count}", (255, 200, 50)),
        (f"Clusters: {cluster_count}", (0, 255, 255)),
        (f"Threshold: {threshold:.1f}m", (200, 180, 255)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (20, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    mode = "IR" if stream_mode == "infrared" else "RGB"
    cv2.putText(display, f"{mode} | q:quit r:reset s:save t:threshold",
                (20, 10 + ph - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1, cv2.LINE_AA)


def draw_depth_crosshair(display, detections, tracker):
    """Draw depth crosshair at each person's shoulder center."""
    for track_id, bbox, kpts, kpt_confs, person in detections:
        if kpts is None or kpt_confs is None:
            continue
        sc = get_shoulder_center(kpts, kpt_confs, 0.3)
        if sc is None:
            continue
        sx, sy = int(sc[0]), int(sc[1])
        color = person.color
        cv2.circle(display, (sx, sy), 10, color, 2, cv2.LINE_AA)
        cv2.line(display, (sx - 16, sy), (sx + 16, sy), color, 1, cv2.LINE_AA)
        cv2.line(display, (sx, sy - 16), (sx, sy + 16), color, 1, cv2.LINE_AA)
        if person.depth_valid:
            depth_label = f"{person.shoulder_depth_mm:.0f}mm"
            cv2.putText(display, depth_label, (sx + 14, sy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

THRESHOLD_OPTIONS = [0.5, 1.0, 1.5, 2.0]


def main():
    print("\n" + "=" * 60)
    print("  RealSense People Clustering (3D Proximity)")
    print("=" * 60)

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

    align_stream = rs.stream.infrared if stream_mode == "infrared" else rs.stream.color
    align = rs.align(align_stream)

    # Get intrinsics for 3D deprojection
    stream_profile = profile.get_stream(align_stream).as_video_stream_profile()
    intrinsics = stream_profile.get_intrinsics()
    print(f"  Intrinsics: {intrinsics.width}x{intrinsics.height}, fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    # --- Clustering tracker ---
    tracker = ClusteringTracker(device="mps")

    # --- Threshold cycling ---
    threshold_idx = 1  # Start at 1.0m

    cv2.namedWindow("Clustering", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Clustering", 960, 720)

    print(f"\nRunning! Threshold: {THRESHOLD_OPTIONS[threshold_idx]:.1f}m")
    print("Controls: q=quit, r=reset, s=screenshot, t=cycle threshold\n")

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

            # --- Multi-person detection + clustering ---
            detections = tracker.process_frame(color_image, depth_frame, intrinsics)
            cluster_result = tracker.last_cluster_result

            display = color_image.copy()

            # --- Draw distance lines (behind everything else) ---
            draw_distance_lines(display, detections, cluster_result, tracker)

            # --- Draw all people ---
            for track_id, bbox, kpts, kpt_confs, person in detections:
                color = person.color
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              color, 2)

                if kpts is not None and kpt_confs is not None:
                    draw_skeleton_coco(display, kpts, kpt_confs, color, thickness=2)

                # Label
                label_y = max(bbox[1] - 10, 20)
                cluster_id = cluster_result.person_to_cluster.get(track_id)
                if cluster_id is not None:
                    label = f"P{track_id} [C{cluster_id}]"
                else:
                    label = f"P{track_id}"
                if person.depth_valid:
                    label += f" {person.shoulder_depth_mm:.0f}mm"
                cv2.putText(display, label, (bbox[0], label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # --- Depth crosshairs ---
            draw_depth_crosshair(display, detections, tracker)

            # --- UI overlays ---
            draw_info_panel(display, actual_fps, len(detections),
                            cluster_result.cluster_count,
                            tracker.cluster_detector.threshold_meters,
                            stream_mode)

            # Cluster panel (bottom-left)
            draw_cluster_panel(display, cluster_result, tracker, 10, h - 15)

            # Depth preview (top-right corner)
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_small = cv2.resize(depth_color, (140, 105))
            display[10:115, w - 150:w - 10] = depth_small
            cv2.rectangle(display, (w - 150, 10), (w - 10, 115), (60, 60, 60), 1)

            cv2.imshow("Clustering", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                tracker.reset()
                frame_count = 0
                start_time = time.time()
                print("Reset all!")
            elif key == ord('s'):
                os.makedirs("outputs", exist_ok=True)
                fname = f"outputs/cluster_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")
            elif key == ord('t'):
                threshold_idx = (threshold_idx + 1) % len(THRESHOLD_OPTIONS)
                new_thresh = THRESHOLD_OPTIONS[threshold_idx]
                tracker.cluster_detector.threshold_meters = new_thresh
                print(f"Threshold: {new_thresh:.1f}m")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nSession: {frame_count} frames in {elapsed:.0f}s ({frame_count / max(elapsed, 1):.1f} FPS)")
    print(f"People tracked: {len(tracker.persons)}")


if __name__ == "__main__":
    main()
