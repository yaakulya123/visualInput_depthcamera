#!/usr/bin/env python3
"""
Person Tracker - Multi-person tracking orchestrator using YOLO11n-pose + ByteTrack.

Per frame:
1. YOLO11n-pose inference (all people + keypoints in one pass)
2. ByteTrack assigns stable person IDs
3. For each person: adapt keypoints, run independent breathing + stillness detectors
4. Aggregate into MultiPersonState with group statistics
"""

import time
import numpy as np
from typing import Optional, Tuple

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import supervision as sv
    SV_AVAILABLE = True
except ImportError:
    SV_AVAILABLE = False

from .keypoint_adapter import coco_to_mediapipe
from .person_state import PersonManager, PersonState, MultiPersonState


# YOLO COCO skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # Arms
    (5, 11), (6, 12),                         # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15), (12, 14), (14, 16), # Legs
]


class PersonTracker:
    """
    Multi-person tracking with independent breathing and stillness detection.

    Uses YOLO11n-pose for multi-person keypoint detection and ByteTrack
    for stable person ID assignment across frames.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        confidence_threshold: float = 0.5,
        device: str = "mps",  # Apple Silicon GPU
        stale_timeout: float = 5.0,
    ):
        """
        Args:
            model_name: YOLO pose model name (downloaded automatically).
            confidence_threshold: Minimum detection confidence.
            device: Inference device ('mps' for Apple Silicon, 'cpu', 'cuda').
            stale_timeout: Seconds before removing untracked persons.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics>=8.1.0"
            )
        if not SV_AVAILABLE:
            raise ImportError(
                "supervision is required. Install with: pip install supervision>=0.19.0"
            )

        self.confidence_threshold = confidence_threshold
        self.device = device

        # Load YOLO model
        print(f"[PersonTracker] Loading {model_name}...")
        self.model = YOLO(model_name)
        print(f"[PersonTracker] Model loaded. Device: {device}")

        # ByteTrack for stable IDs
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        # Person state management
        self.person_manager = PersonManager(stale_timeout=stale_timeout)

        # Frame info
        self._frame_count = 0
        self._fps_timer = time.time()
        self._fps = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[MultiPersonState, dict]:
        """
        Process a single frame: detect, track, and analyze all persons.

        Args:
            frame: BGR frame from camera.

        Returns:
            (MultiPersonState, raw_detections_dict)
            raw_detections_dict contains 'boxes', 'keypoints', 'track_ids'
            for visualization.
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        # Update FPS
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self._fps = self._frame_count / (now - self._fps_timer)
            self._frame_count = 0
            self._fps_timer = now

        # 1. YOLO inference
        results = self.model(
            frame,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]

        # Extract detections
        raw_info = {
            'boxes': [],
            'keypoints': [],
            'confidences': [],
            'track_ids': [],
            'kpt_confidences': [],
        }

        if results.boxes is None or len(results.boxes) == 0:
            self.person_manager.cleanup_stale()
            return MultiPersonState(timestamp=now), raw_info

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        box_confs = results.boxes.conf.cpu().numpy()

        # Get keypoints if available
        if results.keypoints is not None and results.keypoints.xy is not None:
            all_keypoints = results.keypoints.xy.cpu().numpy()  # (N, 17, 2)
            all_kpt_confs = results.keypoints.conf.cpu().numpy()  # (N, 17)
        else:
            all_keypoints = None
            all_kpt_confs = None

        # 2. ByteTrack for stable IDs
        sv_detections = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=box_confs,
        )
        tracked = self.tracker.update_with_detections(sv_detections)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            self.person_manager.cleanup_stale()
            return MultiPersonState(timestamp=now), raw_info

        # 3. Process each tracked person
        persons = {}

        for i in range(len(tracked.xyxy)):
            track_id = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i].astype(int)

            # Match back to original detection for keypoints
            # Find closest original box to tracked box
            kpts = None
            kpt_confs = None
            if all_keypoints is not None:
                matched_idx = self._match_box(bbox, boxes_xyxy)
                if matched_idx is not None:
                    kpts = all_keypoints[matched_idx]
                    kpt_confs = all_kpt_confs[matched_idx]

            # Get or create person state
            person = self.person_manager.get_or_create(track_id)
            person.bbox = tuple(bbox)

            # Store raw info for visualization
            raw_info['boxes'].append(bbox)
            raw_info['track_ids'].append(track_id)
            raw_info['keypoints'].append(kpts)
            raw_info['kpt_confidences'].append(kpt_confs)

            # Convert keypoints and run detectors
            if kpts is not None and kpt_confs is not None:
                landmarks = coco_to_mediapipe(kpts, kpt_confs, w, h)

                if landmarks is not None:
                    # Run breathing detector
                    person.breathing_state = person.breathing_detector.update(
                        frame, landmarks=landmarks
                    )

                    # Run stillness detector
                    person.stillness_state = person.stillness_detector.update(
                        frame, landmarks=landmarks
                    )

                    person.frame_count += 1

            persons[track_id] = person

        # Cleanup stale persons
        self.person_manager.cleanup_stale()

        multi_state = MultiPersonState(persons=persons, timestamp=now)
        return multi_state, raw_info

    def _match_box(self, tracked_box: np.ndarray, original_boxes: np.ndarray) -> Optional[int]:
        """Find the original detection index closest to a tracked box (IoU)."""
        best_iou = 0.0
        best_idx = None

        for i, orig in enumerate(original_boxes):
            iou = self._compute_iou(tracked_box, orig)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx if best_iou > 0.3 else None

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
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

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self):
        """Reset all tracking state."""
        self.person_manager.reset()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self._frame_count = 0

    @property
    def person_count(self) -> int:
        return self.person_manager.person_count
