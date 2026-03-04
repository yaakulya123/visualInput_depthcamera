#!/usr/bin/env python3
"""
RealSense Multi-Person + Audio: Stillness-Driven Sound + Clustering

Combines multi-person tracking (YOLO + ByteTrack) on RealSense IR stream
with a multi-layer audio engine driven by GROUP AVERAGE jitter/stillness,
and 3D proximity clustering via depth deprojection.

- All detected people get colored bounding boxes + skeletons
- Each person runs an independent StillnessDetector
- Group average jitter score (0.0-1.0) drives 5 audio layers:
    Still  (0.0) = Base theta drone only
    Gentle (0.2) = +Layer 1
    Active (0.4) = +Layer 2
    Restless(0.6)= +Layer 3
    Chaotic(0.8) = All 5 layers
- 3D proximity clustering groups nearby people (union-find on deprojected coords)
- Distance lines + cluster panel HUD

Run with: sudo ./venv/bin/python src/tracking/test_realsense_audio.py
Controls: q=quit, r=reset, s=screenshot, a=toggle audio, t=cycle cluster threshold
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
from src.tracking.keypoint_adapter import get_shoulder_center, coco_to_mediapipe
from src.tracking.person_state import PERSON_COLORS
from src.tracking.cluster_detector import (
    PeopleClusterDetector, PersonPosition, ClusterResult, PairDistance,
)
from src.stillness.stillness_detector import StillnessDetector, StillnessState, create_stillness_detector
from src.audio.sound_engine import MultiLayerSoundEngine
from src.network.data_server import DataServer
from collections import deque
import math


# ---------------------------------------------------------------------------
#  Per-person info with StillnessDetector
# ---------------------------------------------------------------------------

@dataclass
class PersonInfo:
    """Per-person tracking state with independent stillness detection."""
    person_id: int
    color: Tuple[int, int, int]
    stillness_detector: StillnessDetector = field(default=None, repr=False)
    jitter_score: float = 0.0
    stillness_state: Optional[StillnessState] = None  # Full stillness data
    last_seen: float = field(default_factory=time.time)
    shoulder_depth_mm: float = 0.0
    world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_valid: bool = False
    # Depth profile for this person's bbox
    depth_min_mm: float = 0.0
    depth_max_mm: float = 0.0
    depth_mean_mm: float = 0.0
    body_pixel_count: int = 0  # Number of valid depth pixels (body surface area proxy)


# ---------------------------------------------------------------------------
#  Multi-Person RealSense Tracker (with per-person stillness)
# ---------------------------------------------------------------------------

class MultiPersonRealSense:
    """
    YOLO11n-pose + ByteTrack on RealSense IR stream.
    Per-person stillness detection. Closest person = primary for breathing.
    """

    STALE_TIMEOUT = 5.0
    HYSTERESIS_MM = 50.0

    def __init__(self, device: str = "mps"):
        print("[MultiPersonRealSense] Loading YOLO11n-pose...")
        self.model = YOLO("yolo11n-pose.pt")
        self.device = device
        print(f"[MultiPersonRealSense] Model loaded. Device: {device}")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        self.persons: Dict[int, PersonInfo] = {}
        self._color_index = 0
        self.primary_id: Optional[int] = None
        self.intrinsics = None  # Set from main() after pipeline starts
        self.cluster_detector = PeopleClusterDetector(threshold_meters=1.0)
        self.last_cluster_result = ClusterResult()

    def process_frame(self, frame: np.ndarray, depth_frame, depth_image: np.ndarray):
        """
        Detect + track all people, run stillness per person, select closest as primary,
        compute 3D positions and cluster by proximity.

        Returns:
            list of (track_id, bbox, keypoints, kpt_confs, PersonInfo)
        """
        h, w = frame.shape[:2]
        now = time.time()

        results = self.model(
            frame, device=self.device, conf=0.3, verbose=False,
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
            person.depth_valid = False

            # Default pixel position from bbox center
            pixel_x = float((bbox[0] + bbox[2]) / 2)
            pixel_y = float((bbox[1] + bbox[3]) / 2)

            # Sample shoulder depth + run stillness detector + 3D deprojection
            if kpts is not None and kpt_confs is not None:
                shoulder_center = get_shoulder_center(kpts, kpt_confs, 0.3)
                if shoulder_center is not None:
                    sx, sy = int(shoulder_center[0]), int(shoulder_center[1])
                    pixel_x, pixel_y = float(sx), float(sy)
                    depth_mm = self._sample_depth(depth_frame, sx, sy, w, h)
                    if depth_mm > 0:
                        person.shoulder_depth_mm = depth_mm
                        # Deproject to 3D world coordinates
                        if self.intrinsics is not None:
                            depth_m = depth_mm / 1000.0
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                self.intrinsics, [float(sx), float(sy)], depth_m
                            )
                            person.world_pos = (point_3d[0], point_3d[1], point_3d[2])
                            person.depth_valid = True

                # Run stillness detection via adapted landmarks
                # Normalize by bounding box size (not frame size) so distance
                # from camera doesn't affect sensitivity — a person far away
                # has a smaller bbox but their movements relative to their
                # body size remain proportionally the same.
                bbox_w = max(bbox[2] - bbox[0], 1)
                bbox_h = max(bbox[3] - bbox[1], 1)
                # Shift keypoints to bbox-relative coords before normalizing
                kpts_relative = kpts.copy()
                kpts_relative[:, 0] -= bbox[0]
                kpts_relative[:, 1] -= bbox[1]
                landmarks = coco_to_mediapipe(kpts_relative, kpt_confs, bbox_w, bbox_h)
                if landmarks is not None:
                    state = person.stillness_detector.update(frame, landmarks=landmarks)
                    person.jitter_score = state.jitter_score
                    person.stillness_state = state

            # Compute depth profile within person's bounding box
            bx1, by1, bx2, by2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
            if bx2 > bx1 and by2 > by1:
                person_depth_roi = depth_image[by1:by2, bx1:bx2].astype(np.float32)
                valid_mask = (person_depth_roi > 200) & (person_depth_roi < 5000)
                valid_count = np.count_nonzero(valid_mask)
                if valid_count > 10:
                    valid_depths = person_depth_roi[valid_mask]
                    person.depth_min_mm = float(np.min(valid_depths))
                    person.depth_max_mm = float(np.max(valid_depths))
                    person.depth_mean_mm = float(np.mean(valid_depths))
                    person.body_pixel_count = valid_count

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

        self._select_primary(detections)
        self._cleanup_stale(now)
        return detections

    def get_group_jitter(self) -> float:
        """Max jitter across all tracked people.

        Uses max instead of mean so that ANY person moving drives
        the audio layers up — regardless of how many others are still.
        Layers only fade down when ALL skeletons are static.
        """
        if not self.persons:
            return 0.0
        scores = [p.jitter_score for p in self.persons.values()]
        return float(np.max(scores))

    def get_group_stillness_stats(self) -> dict:
        """Compute group-level stillness statistics."""
        if not self.persons:
            return {"avg_jitter": 0.0, "min_jitter": 0.0, "max_jitter": 0.0,
                    "still_count": 0, "moving_count": 0, "sync_score": 0.0,
                    "avg_stillness_duration": 0.0, "max_stillness_duration": 0.0}

        scores = []
        durations = []
        still_count = 0
        moving_count = 0
        for p in self.persons.values():
            scores.append(p.jitter_score)
            if p.stillness_state:
                durations.append(p.stillness_state.stillness_duration)
                if p.stillness_state.motion_type == "still":
                    still_count += 1
                else:
                    moving_count += 1

        # Sync score: how similar are everyone's jitter levels?
        # Low variance = high sync (everyone similar), high variance = low sync
        sync_score = 0.0
        if len(scores) >= 2:
            variance = float(np.var(scores))
            # Map variance 0-0.1 to sync 1.0-0.0
            sync_score = max(0.0, 1.0 - variance * 10)

        return {
            "avg_jitter": float(np.mean(scores)),
            "min_jitter": float(np.min(scores)),
            "max_jitter": float(np.max(scores)),
            "still_count": still_count,
            "moving_count": moving_count,
            "sync_score": round(sync_score, 4),
            "avg_stillness_duration": float(np.mean(durations)) if durations else 0.0,
            "max_stillness_duration": float(np.max(durations)) if durations else 0.0,
        }

    def _sample_depth(self, depth_frame, cx, cy, w, h, grid_half=2):
        depths = []
        for dy in range(-grid_half, grid_half + 1):
            for dx in range(-grid_half, grid_half + 1):
                px = max(0, min(w - 1, cx + dx))
                py = max(0, min(h - 1, cy + dy))
                d = depth_frame.get_distance(px, py)
                if 0.1 < d < 5.0:
                    depths.append(d * 1000)
        return np.median(depths) if depths else 0

    def _select_primary(self, detections):
        candidates = []
        for track_id, bbox, kpts, kpt_confs, person in detections:
            if person.shoulder_depth_mm > 0:
                candidates.append((track_id, person.shoulder_depth_mm))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[1])
        closest_id, closest_depth = candidates[0]
        if self.primary_id is None:
            self.primary_id = closest_id
            return
        current_depth = None
        for cid, cdepth in candidates:
            if cid == self.primary_id:
                current_depth = cdepth
                break
        if current_depth is None:
            self.primary_id = closest_id
            return
        if closest_id != self.primary_id and closest_depth < current_depth - self.HYSTERESIS_MM:
            self.primary_id = closest_id

    def _get_or_create(self, track_id: int) -> PersonInfo:
        if track_id in self.persons:
            return self.persons[track_id]
        color = PERSON_COLORS[self._color_index % len(PERSON_COLORS)]
        self._color_index += 1
        person = PersonInfo(
            person_id=track_id,
            color=color,
            stillness_detector=create_stillness_detector("normal"),
        )
        self.persons[track_id] = person
        return person

    def _cleanup_stale(self, now: float):
        stale = [pid for pid, p in self.persons.items()
                 if (now - p.last_seen) > self.STALE_TIMEOUT]
        for pid in stale:
            del self.persons[pid]
            if self.primary_id == pid:
                self.primary_id = None

    def _match_box(self, tracked_box, original_boxes):
        best_iou = 0.0
        best_idx = None
        for i, orig in enumerate(original_boxes):
            iou = self._compute_iou(tracked_box, orig)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx if best_iou > 0.3 else None

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def reset(self):
        self.persons.clear()
        self._color_index = 0
        self.primary_id = None
        self.last_cluster_result = ClusterResult()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
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


def draw_chaos_meter(display, chaos_score, x, y, w, h):
    """Vertical chaos/jitter meter with gradient fill."""
    # Background
    cv2.rectangle(display, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(display, (x, y), (x + w, y + h), (80, 80, 80), 1)

    # Fill height
    fill_h = int(chaos_score * h)
    fill_y = y + h - fill_h

    # Gradient fill (green → yellow → orange → red)
    for i in range(fill_h):
        row_y = fill_y + i
        ratio = 1.0 - (i / max(h, 1))  # 1.0 at top, 0.0 at bottom
        if ratio < 0.33:
            color = (0, 255, int(255 * ratio * 3))           # Green → Yellow
        elif ratio < 0.66:
            r = int(255 * (ratio - 0.33) * 3)
            color = (0, int(255 * (1 - (ratio - 0.33) * 3)), 255)  # Yellow → Red-ish
        else:
            color = (0, 50, 255)                               # Red
        cv2.line(display, (x + 2, row_y), (x + w - 2, row_y), color, 1)

    # Marker line
    marker_y = y + h - int(chaos_score * h)
    cv2.line(display, (x - 3, marker_y), (x + w + 3, marker_y), (255, 255, 255), 2)

    # Value
    cv2.putText(display, f"{chaos_score:.2f}", (x - 2, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(display, "JITTER", (x - 5, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)


def draw_audio_layers(display, layer_info, x, y, w):
    """Draw active audio layers panel."""
    num_layers = len(layer_info)
    panel_h = 20 + num_layers * 22
    overlay = display.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.rectangle(display, (x, y), (x + w, y + panel_h), (60, 60, 60), 1)

    cv2.putText(display, "AUDIO LAYERS", (x + 5, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    for i, layer in enumerate(layer_info):
        ly = y + 22 + i * 22
        active = layer['active']
        vol = layer['volume']
        name = layer['name']
        if len(name) > 28:
            name = name[:25] + "..."

        # Volume bar background
        bar_x = x + 5
        bar_w = w - 60
        cv2.rectangle(display, (bar_x, ly), (bar_x + bar_w, ly + 14), (40, 40, 40), -1)

        # Volume bar fill
        if active:
            fill_w = int(bar_w * vol)
            color = (0, 200, 100) if vol > 0.5 else (0, 150, 80)
            cv2.rectangle(display, (bar_x, ly), (bar_x + fill_w, ly + 14), color, -1)

        # Layer label
        text_color = (200, 200, 200) if active else (80, 80, 80)
        cv2.putText(display, f"L{i}", (bar_x + 3, ly + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

        # Volume text
        cv2.putText(display, f"{vol:.0%}", (bar_x + bar_w + 5, ly + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)


def draw_distance_lines(display, detections, cluster_result, tracker):
    """
    Draw lines between people showing distance.
    - Bright colored line for clustered pairs (same cluster)
    - Faint gray line for near-but-not-clustered pairs (< 2m)
    """
    pixel_map: Dict[int, Tuple[int, int]] = {}
    for track_id, bbox, kpts, kpt_confs, person in detections:
        if kpts is not None and kpt_confs is not None:
            sc = get_shoulder_center(kpts, kpt_confs, 0.3)
            if sc is not None:
                pixel_map[track_id] = (int(sc[0]), int(sc[1]))
                continue
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

        same_cluster = (
            cluster_result.person_to_cluster.get(pair.id_a) ==
            cluster_result.person_to_cluster.get(pair.id_b)
        )

        if same_cluster and pair.distance_m <= tracker.cluster_detector.threshold_meters:
            cluster_id = cluster_result.person_to_cluster.get(pair.id_a, 0)
            cluster_ids = sorted(cluster_result.clusters.keys())
            ci = cluster_ids.index(cluster_id) if cluster_id in cluster_ids else 0
            color = PERSON_COLORS[ci % len(PERSON_COLORS)]
            cv2.line(display, pt_a, pt_b, color, 2, cv2.LINE_AA)
            label = f"{pair.distance_m:.2f}m"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(display,
                          (mid_x - ts[0]//2 - 3, mid_y - ts[1] - 3),
                          (mid_x + ts[0]//2 + 3, mid_y + 3),
                          (0, 0, 0), -1)
            cv2.putText(display, label, (mid_x - ts[0]//2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        else:
            cv2.line(display, pt_a, pt_b, (60, 60, 60), 1, cv2.LINE_AA)
            label = f"{pair.distance_m:.2f}m"
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(display, label, (mid_x - ts[0]//2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)


def draw_cluster_panel(display, cluster_result, x, y):
    """Draw cluster info panel."""
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

        cv2.rectangle(display, (x + 8, row_y - 8), (x + 22, row_y + 6), color, -1)
        cv2.rectangle(display, (x + 8, row_y - 8), (x + 22, row_y + 6), (200, 200, 200), 1)

        member_str = ", ".join(f"P{m}" for m in members)
        if len(member_str) > 20:
            member_str = member_str[:17] + "..."
        label = f"C{cid}: {len(members)}p - {member_str}"
        cv2.putText(display, label, (x + 28, row_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


def draw_info_panel(display, fps, person_count, primary_id, group_jitter,
                    audio_on, stream_mode, cluster_count=0, cluster_threshold=1.0):
    """Semi-transparent HUD."""
    pw, ph = 230, 170
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.rectangle(display, (10, 10), (10 + pw, 10 + ph), (60, 60, 60), 1)

    lines = [
        (f"FPS: {fps:.0f}", (0, 255, 0)),
        (f"People: {person_count}", (255, 200, 50)),
        (f"Clusters: {cluster_count} ({cluster_threshold:.1f}m)", (200, 180, 255)),
        (f"Primary: P{primary_id}" if primary_id is not None else "Primary: --", (0, 255, 255)),
        (f"Group Jitter: {group_jitter:.2f}", _jitter_color(group_jitter)),
        (f"Audio: {'ON' if audio_on else 'OFF'}", (0, 255, 0) if audio_on else (0, 0, 255)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (20, 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    mode = "IR" if stream_mode == "infrared" else "RGB"
    cv2.putText(display, f"{mode} | q:quit r:reset s:save a:audio t:threshold",
                (20, 10 + ph - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1, cv2.LINE_AA)


def _jitter_color(jitter):
    """Color for jitter value: green (still) → yellow → red (restless)."""
    if jitter < 0.2:
        return (0, 255, 0)
    elif jitter < 0.5:
        return (0, 255, 255)
    elif jitter < 0.8:
        return (0, 165, 255)
    else:
        return (0, 0, 255)


def draw_per_person_jitter(display, detections, tracker):
    """Draw small jitter bar next to each person's bounding box."""
    for track_id, bbox, kpts, kpt_confs, person in detections:
        jitter = person.jitter_score
        x2 = int(bbox[2])
        y1 = int(bbox[1])
        y2 = int(bbox[3])

        # Small vertical bar on right side of bbox
        bar_x = x2 + 4
        bar_w = 8
        bar_h = min(y2 - y1, 80)
        bar_y = y1

        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                       (40, 40, 40), -1)
        fill_h = int(bar_h * min(jitter, 1.0))
        fill_y = bar_y + bar_h - fill_h
        color = _jitter_color(jitter)
        if fill_h > 0:
            cv2.rectangle(display, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h),
                           color, -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                       (80, 80, 80), 1)


# ---------------------------------------------------------------------------
#  Motion Waveform Visualization
# ---------------------------------------------------------------------------

class MotionWaveform:
    """
    Seismograph-style waveform driven by jitter/motion.
    Flat line when still, chaotic waves when moving.
    Maintains per-person history for overlaid waveforms.
    """

    def __init__(self, width=400, history_len=200):
        self.width = width
        self.history_len = history_len
        # Group waveform history
        self.group_history = deque([0.0] * history_len, maxlen=history_len)
        # Per-person waveform histories (track_id -> deque)
        self.person_histories: Dict[int, deque] = {}
        # Phase accumulators for smooth wave generation
        self.group_phase = 0.0
        self.person_phases: Dict[int, float] = {}
        # Smoothed amplitude for organic feel
        self._smoothed_amp = 0.0

    def update(self, group_jitter: float, person_jitters: Dict[int, float] = None):
        """Push new jitter values into history buffers."""
        # Smooth the amplitude for organic wave feel
        self._smoothed_amp += (group_jitter - self._smoothed_amp) * 0.3

        # Generate wave sample from jitter — more jitter = bigger + faster waves
        amp = self._smoothed_amp
        freq = 0.15 + amp * 0.6  # Faster oscillation when moving
        self.group_phase += freq
        # Combine sine waves at different frequencies for organic look
        sample = amp * (
            0.6 * math.sin(self.group_phase) +
            0.25 * math.sin(self.group_phase * 2.3 + 0.7) +
            0.15 * math.sin(self.group_phase * 4.1 + 1.3)
        )
        # Add noise proportional to jitter
        if amp > 0.05:
            sample += np.random.normal(0, amp * 0.15)
        self.group_history.append(sample)

        # Per-person histories
        if person_jitters:
            active_ids = set(person_jitters.keys())
            # Remove stale
            for pid in list(self.person_histories.keys()):
                if pid not in active_ids:
                    del self.person_histories[pid]
                    self.person_phases.pop(pid, None)
            # Update each person
            for pid, jitter in person_jitters.items():
                if pid not in self.person_histories:
                    self.person_histories[pid] = deque([0.0] * self.history_len, maxlen=self.history_len)
                    self.person_phases[pid] = np.random.uniform(0, 2 * math.pi)
                phase = self.person_phases.get(pid, 0.0)
                phase += 0.12 + jitter * 0.5
                self.person_phases[pid] = phase
                p_sample = jitter * (
                    0.7 * math.sin(phase) +
                    0.3 * math.sin(phase * 2.7 + pid * 0.5)
                )
                if jitter > 0.05:
                    p_sample += np.random.normal(0, jitter * 0.1)
                self.person_histories[pid].append(p_sample)


def draw_motion_waveform(display, waveform: MotionWaveform, group_jitter: float,
                         person_colors: Dict[int, Tuple[int, int, int]],
                         x: int, y: int, w: int, h: int):
    """
    Draw the motion waveform panel.
    - Dark background with grid lines
    - Per-person thin waveforms in their colors
    - Group waveform as thick bright line on top
    - Glow effect on the group line
    """
    # Semi-transparent background
    overlay = display.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
    cv2.rectangle(display, (x, y), (x + w, y + h), (40, 40, 50), 1)

    # Grid lines (subtle horizontal)
    mid_y = y + h // 2
    for offset in [-h // 4, 0, h // 4]:
        gy = mid_y + offset
        cv2.line(display, (x + 1, gy), (x + w - 1, gy), (25, 25, 35), 1)

    # Subtle vertical grid lines
    for gx_offset in range(0, w, w // 8):
        cv2.line(display, (x + gx_offset, y + 1), (x + gx_offset, y + h - 1), (20, 20, 30), 1)

    # Center line (zero line)
    cv2.line(display, (x + 1, mid_y), (x + w - 1, mid_y), (40, 40, 55), 1)

    history_len = waveform.history_len
    scale_y = (h // 2) * 0.85  # Max amplitude fills 85% of half-height

    def history_to_points(history, color_override=None):
        """Convert a history deque to polyline points."""
        pts = []
        data = list(history)
        step = max(1, len(data) / w)
        for i in range(w):
            idx = int(i * step)
            if idx >= len(data):
                idx = len(data) - 1
            val = data[idx]
            px = x + i
            py = int(mid_y - val * scale_y)
            py = max(y + 2, min(y + h - 2, py))
            pts.append((px, py))
        return pts

    # Draw per-person waveforms (thin, colored)
    for pid, hist in waveform.person_histories.items():
        color = person_colors.get(pid, (100, 100, 100))
        pts = history_to_points(hist)
        if len(pts) > 1:
            # Dim version of person color
            dim_color = tuple(max(20, c // 3) for c in color)
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_arr], False, dim_color, 1, cv2.LINE_AA)

    # Draw group waveform (main, thick, with glow)
    group_pts = history_to_points(waveform.group_history)
    if len(group_pts) > 1:
        pts_arr = np.array(group_pts, dtype=np.int32).reshape((-1, 1, 2))

        # Color based on jitter level
        if group_jitter < 0.15:
            wave_color = (180, 255, 180)  # Soft green — calm
            glow_color = (60, 120, 60)
        elif group_jitter < 0.4:
            wave_color = (0, 255, 255)    # Cyan — gentle movement
            glow_color = (0, 100, 100)
        elif group_jitter < 0.7:
            wave_color = (0, 180, 255)    # Orange — active
            glow_color = (0, 80, 120)
        else:
            wave_color = (80, 80, 255)    # Red — chaotic
            glow_color = (40, 40, 140)

        # Glow layer (wider, dimmer)
        cv2.polylines(display, [pts_arr], False, glow_color, 4, cv2.LINE_AA)
        # Main line
        cv2.polylines(display, [pts_arr], False, wave_color, 2, cv2.LINE_AA)

        # Bright tip at the latest point (rightmost)
        tip = group_pts[-1]
        cv2.circle(display, tip, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(display, tip, 7, wave_color, 1, cv2.LINE_AA)

    # Label
    label_color = (150, 150, 170)
    cv2.putText(display, "MOTION", (x + 6, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

    # Jitter value on the right
    jitter_str = f"{group_jitter:.2f}"
    jitter_color = (180, 255, 180) if group_jitter < 0.2 else (0, 255, 255) if group_jitter < 0.5 else (0, 180, 255) if group_jitter < 0.7 else (80, 80, 255)
    cv2.putText(display, jitter_str, (x + w - 40, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, jitter_color, 1, cv2.LINE_AA)

    # State label
    if group_jitter < 0.1:
        state_text = "STILL"
        state_color = (100, 200, 100)
    elif group_jitter < 0.3:
        state_text = "CALM"
        state_color = (100, 200, 180)
    elif group_jitter < 0.6:
        state_text = "ACTIVE"
        state_color = (0, 200, 255)
    else:
        state_text = "CHAOTIC"
        state_color = (80, 80, 255)
    ts = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
    cv2.putText(display, state_text, (x + w // 2 - ts[0] // 2, y + h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, state_color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  Micro-Motion Heatmap
# ---------------------------------------------------------------------------

class MicroMotionHeatmap:
    """
    Per-pixel depth differencing to create a real-time motion heatmap.
    Shows exactly where on the body (or scene) movement is happening.
    """

    def __init__(self, decay: float = 0.7, noise_threshold_mm: int = 8):
        self.prev_depth = None
        self.accumulated = None  # Accumulated heatmap with decay
        self.decay = decay       # How fast old motion fades (0=instant, 1=never)
        self.noise_threshold_mm = noise_threshold_mm  # Ignore changes smaller than this

    def update(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Compute micro-motion heatmap from depth frame.

        Args:
            depth_image: uint16 depth image (values in mm)

        Returns:
            Heatmap as float32 array (0.0 = still, 1.0 = max motion), same size as input
        """
        depth_f = depth_image.astype(np.float32)

        if self.prev_depth is None:
            self.prev_depth = depth_f.copy()
            self.accumulated = np.zeros_like(depth_f)
            return self.accumulated

        # Compute absolute depth difference
        diff = np.abs(depth_f - self.prev_depth)

        # Mask out invalid pixels (where either frame has zero depth)
        valid = (depth_f > 100) & (self.prev_depth > 100)
        diff[~valid] = 0

        # Threshold to remove sensor noise
        diff[diff < self.noise_threshold_mm] = 0

        # Normalize: 8mm-80mm range mapped to 0-1
        motion = np.clip(diff / 80.0, 0.0, 1.0)

        # Accumulate with decay (creates trailing glow effect)
        self.accumulated = self.accumulated * self.decay + motion * (1 - self.decay)

        self.prev_depth = depth_f.copy()
        return self.accumulated

    def get_colored_heatmap(self, heatmap: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """
        Convert heatmap to a colored BGR image for display.

        Args:
            heatmap: float32 heatmap from update()
            size: Optional (width, height) to resize

        Returns:
            BGR uint8 image with cool-to-hot colormap
        """
        # Scale to 0-255
        vis = np.clip(heatmap * 255 * 3, 0, 255).astype(np.uint8)

        # Apply colormap (INFERNO: black → purple → red → yellow → white)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

        if size is not None:
            colored = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)

        return colored

    def get_motion_score(self, heatmap: np.ndarray) -> float:
        """Get overall motion score from heatmap (0.0 = still, 1.0 = lots of motion)."""
        valid = heatmap[heatmap > 0.01]
        if len(valid) == 0:
            return 0.0
        return float(np.clip(np.mean(valid) * 5, 0, 1))

    def reset(self):
        self.prev_depth = None
        self.accumulated = None


def draw_stillness_panel(display, detections, group_stats, x, y, w):
    """Draw per-person stillness details + group sync score."""
    if not detections:
        return

    num_people = len(detections)
    panel_h = 50 + num_people * 26
    overlay = display.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.rectangle(display, (x, y), (x + w, y + panel_h), (60, 60, 60), 1)

    # Header with sync score
    sync = group_stats.get("sync_score", 0.0)
    sync_color = (0, 255, 0) if sync > 0.7 else (0, 255, 255) if sync > 0.4 else (0, 100, 255)
    cv2.putText(display, "STILLNESS", (x + 5, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, f"SYNC:{sync:.0%}", (x + w - 75, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, sync_color, 1, cv2.LINE_AA)

    # Group still/moving count
    still_c = group_stats.get("still_count", 0)
    move_c = group_stats.get("moving_count", 0)
    cv2.putText(display, f"Still:{still_c}  Moving:{move_c}", (x + 5, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
    max_dur = group_stats.get("max_stillness_duration", 0)
    cv2.putText(display, f"Best:{max_dur:.0f}s", (x + w - 65, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    # Per-person rows
    for i, (tid, bbox, kpts, kpt_confs, person) in enumerate(detections):
        row_y = y + 42 + i * 26

        # Quality badge
        quality = person.stillness_detector.get_stillness_quality()
        quality_colors = {
            "settling": (100, 100, 100), "focused": (200, 200, 0),
            "deep_focus": (255, 150, 0), "transcendent": (0, 215, 255),
            "moving": (100, 100, 180), "unknown": (80, 80, 80),
        }
        q_color = quality_colors.get(quality, (80, 80, 80))

        # Stillness duration bar
        duration = person.stillness_state.stillness_duration if person.stillness_state else 0.0
        bar_x = x + 45
        bar_w = w - 90
        bar_fill = min(duration / 30.0, 1.0)  # 30s = full bar
        cv2.rectangle(display, (bar_x, row_y), (bar_x + bar_w, row_y + 16), (30, 30, 30), -1)
        if bar_fill > 0:
            fill_color = q_color
            cv2.rectangle(display, (bar_x, row_y),
                          (bar_x + int(bar_w * bar_fill), row_y + 16), fill_color, -1)
        cv2.rectangle(display, (bar_x, row_y), (bar_x + bar_w, row_y + 16), (50, 50, 50), 1)

        # Person ID
        cv2.putText(display, f"P{tid}", (x + 5, row_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, person.color, 1, cv2.LINE_AA)

        # Duration text inside bar
        dur_str = f"{duration:.0f}s" if duration > 0 else "--"
        cv2.putText(display, dur_str, (bar_x + 3, row_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)

        # Quality label
        q_short = quality[:4].upper() if quality != "unknown" else "--"
        cv2.putText(display, q_short, (bar_x + bar_w + 5, row_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, q_color, 1, cv2.LINE_AA)


def draw_heatmap_overlay(display, heatmap_colored, x, y, w, h, motion_score):
    """Draw the micro-motion heatmap as a panel on the HUD."""
    resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)

    # Blend with display area
    roi = display[y:y + h, x:x + w]
    if roi.shape[:2] == resized.shape[:2]:
        blended = cv2.addWeighted(resized, 0.85, roi, 0.15, 0)
        display[y:y + h, x:x + w] = blended

    # Border
    cv2.rectangle(display, (x, y), (x + w, y + h), (60, 60, 60), 1)

    # Label
    cv2.putText(display, "MICRO-MOTION", (x + 4, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
    score_str = f"{motion_score:.2f}"
    cv2.putText(display, score_str, (x + w - 35, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  RealSense Multi-Person + Audio (Stillness-Driven)")
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

    # Hardware reset: prevent segfault on macOS when camera was used recently
    print("\nResetting RealSense hardware...")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            for dev in devices:
                dev.hardware_reset()
                print(f"  Reset: {dev.get_info(rs.camera_info.name)}")
            import time as _time
            _time.sleep(3)  # Wait for USB re-enumeration after reset
            print("  Reset complete, waiting for USB...")
        else:
            print("  No RealSense devices found!")
            return
    except Exception as e:
        print(f"  Reset warning: {e}")
        import time as _time
        _time.sleep(1)

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

    # Get intrinsics for 3D deprojection (needed for clustering)
    stream_profile = profile.get_stream(align_stream).as_video_stream_profile()
    intrinsics = stream_profile.get_intrinsics()
    print(f"  Intrinsics: {intrinsics.width}x{intrinsics.height}, fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    # --- Multi-person tracker ---
    tracker = MultiPersonRealSense(device="mps")
    tracker.intrinsics = intrinsics

    # --- Cluster threshold cycling ---
    THRESHOLD_OPTIONS = [0.5, 1.0, 1.5, 2.0]
    threshold_idx = 1  # Start at 1.0m

    # --- Audio engine ---
    audio_on = True
    try:
        audio_engine = MultiLayerSoundEngine()
        audio_started = audio_engine.start()
        if not audio_started:
            print("[!] Audio engine failed to start, continuing without audio")
            audio_on = False
    except Exception as e:
        print(f"[!] Audio engine error: {e}")
        audio_engine = None
        audio_on = False

    # --- Live data server (HTTP + WebSocket to VPS relay) ---
    data_server = DataServer(port=8765, ws_url="ws://82.112.226.90:3000")
    data_server.start()

    # --- Motion waveform ---
    motion_waveform = MotionWaveform(width=400, history_len=200)

    # --- Micro-motion heatmap ---
    micro_motion = MicroMotionHeatmap(decay=0.7, noise_threshold_mm=8)

    # --- Smoothed group jitter for audio ---
    smoothed_group_jitter = 0.0
    # Higher alpha = more responsive to rapid movement
    # StillnessDetector already smooths internally (alpha=0.15), so we use a high
    # alpha here to avoid double-dampening. 0.7 means 70% new value, 30% old.
    JITTER_EMA_ALPHA = 0.7

    cv2.namedWindow("Audio+Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Audio+Tracking", 960, 720)

    print(f"\nRunning! Cluster threshold: {THRESHOLD_OPTIONS[threshold_idx]:.1f}m")
    print("Controls: q=quit, r=reset, s=screenshot, a=toggle audio, t=cycle threshold\n")

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

            # --- Multi-person detection + stillness ---
            detections = tracker.process_frame(color_image, depth_frame, depth_image)

            # --- Group jitter → Audio ---
            raw_group_jitter = tracker.get_group_jitter()
            smoothed_group_jitter = (JITTER_EMA_ALPHA * raw_group_jitter +
                                     (1 - JITTER_EMA_ALPHA) * smoothed_group_jitter)

            if audio_engine and audio_on:
                audio_engine.update(smoothed_group_jitter)

            # --- Update motion waveform ---
            person_jitters = {tid: person.jitter_score
                              for tid, bbox, kpts, kpt_confs, person in detections}
            motion_waveform.update(smoothed_group_jitter, person_jitters)

            # --- Micro-motion heatmap ---
            heatmap = micro_motion.update(depth_image)
            heatmap_colored = micro_motion.get_colored_heatmap(heatmap)
            heatmap_score = micro_motion.get_motion_score(heatmap)

            # --- Cluster result ---
            cluster_result = tracker.last_cluster_result

            # --- Group stats ---
            group_stats = tracker.get_group_stillness_stats()

            # --- Push live data to server (clean, focused payload) ---
            # Group: person_count, group_jitter, active_layers
            # Per-person: id, jitter, stillness, depth_mm
            active_layers = audio_engine.get_active_count() if audio_on else 0
            data_server.update({
                "person_count": len(detections),
                "group_jitter": round(smoothed_group_jitter, 3),
                "active_layers": active_layers,
                "persons": [
                    {
                        "id": tid,
                        "jitter": round(person.jitter_score, 3),
                        "stillness": round(
                            person.stillness_state.stillness_duration, 1
                        ) if person.stillness_state else 0.0,
                        "depth_mm": round(person.shoulder_depth_mm, 0),
                    }
                    for tid, bbox, kpts, kpt_confs, person in detections
                ],
            })

            display = color_image.copy()

            # --- Draw distance lines (behind everything else) ---
            draw_distance_lines(display, detections, cluster_result, tracker)

            # --- Draw all people ---
            for track_id, bbox, kpts, kpt_confs, person in detections:
                is_primary = (track_id == tracker.primary_id)
                color = person.color
                box_thickness = 3 if is_primary else 2

                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              color, box_thickness)

                if kpts is not None and kpt_confs is not None:
                    draw_skeleton_coco(display, kpts, kpt_confs, color,
                                       thickness=2 if is_primary else 1)

                label_y = max(bbox[1] - 10, 20)
                cid = cluster_result.person_to_cluster.get(track_id)
                cluster_tag = f" [C{cid}]" if cid is not None else ""
                if is_primary:
                    label = f"P{track_id}{cluster_tag} [PRIMARY] {person.shoulder_depth_mm:.0f}mm"
                    cv2.putText(display, label, (bbox[0], label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                else:
                    label = f"P{track_id}{cluster_tag}"
                    if person.depth_valid:
                        label += f" {person.shoulder_depth_mm:.0f}mm"
                    cv2.putText(display, label, (bbox[0], label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # --- Per-person jitter bars ---
            draw_per_person_jitter(display, detections, tracker)

            # --- UI overlays (left column: stacked top-to-bottom) ---
            # 1. Info panel (top-left, fixed)
            draw_info_panel(display, actual_fps,
                            len(detections), tracker.primary_id,
                            smoothed_group_jitter, audio_on, stream_mode,
                            cluster_result.cluster_count,
                            tracker.cluster_detector.threshold_meters)
            left_y = 190  # Info panel ends at y=180, +10px gap

            # 2. Stillness panel (below info panel)
            stillness_panel_h = 50 + max(len(detections), 1) * 26
            draw_stillness_panel(display, detections, group_stats, 10, left_y, 220)
            left_y += stillness_panel_h + 8  # gap

            # 3. Audio layers panel (below stillness panel)
            if audio_engine:
                layer_info = audio_engine.get_layer_info()
                draw_audio_layers(display, layer_info, 10, left_y, 220)

            # --- Right column ---
            # Depth preview (top-right corner)
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_small = cv2.resize(depth_color, (140, 105))
            display[10:115, w-150:w-10] = depth_small
            cv2.rectangle(display, (w-150, 10), (w-10, 115), (60, 60, 60), 1)

            # Micro-motion heatmap (below depth preview)
            draw_heatmap_overlay(display, heatmap_colored,
                                  w - 150, 120, 140, 105, heatmap_score)

            # Chaos meter (right side, mid-screen)
            draw_chaos_meter(display, smoothed_group_jitter, w - 50, 240, 30, 120)

            # Cluster panel (right side, above waveform)
            draw_cluster_panel(display, cluster_result, w - 220, h - 100)

            # --- Motion waveform (bottom of screen, full width) ---
            waveform_h = 80
            waveform_y = h - waveform_h - 5
            waveform_x = 5
            waveform_w = w - 10
            person_color_map = {tid: person.color
                                for tid, bbox, kpts, kpt_confs, person in detections}
            draw_motion_waveform(display, motion_waveform, smoothed_group_jitter,
                                 person_color_map, waveform_x, waveform_y,
                                 waveform_w, waveform_h)

            cv2.imshow("Audio+Tracking", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                tracker.reset()
                smoothed_group_jitter = 0.0
                motion_waveform = MotionWaveform(width=400, history_len=200)
                micro_motion.reset()
                if audio_engine:
                    audio_engine.reset()
                frame_count = 0
                start_time = time.time()
                print("Reset all!")
            elif key == ord('s'):
                os.makedirs("outputs", exist_ok=True)
                fname = f"outputs/audio_tracking_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")
            elif key == ord('a'):
                audio_on = not audio_on
                if not audio_on and audio_engine:
                    audio_engine.update(0.0)  # Fade to base only
                print(f"Audio: {'ON' if audio_on else 'OFF'}")
            elif key == ord('t'):
                threshold_idx = (threshold_idx + 1) % len(THRESHOLD_OPTIONS)
                new_thresh = THRESHOLD_OPTIONS[threshold_idx]
                tracker.cluster_detector.threshold_meters = new_thresh
                print(f"Cluster threshold: {new_thresh:.1f}m")

    except KeyboardInterrupt:
        pass
    finally:
        if audio_engine:
            audio_engine.stop()
        data_server.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nSession: {frame_count} frames in {elapsed:.0f}s ({frame_count/max(elapsed,1):.1f} FPS)")
    print(f"People tracked: {len(tracker.persons)}")


if __name__ == "__main__":
    main()
