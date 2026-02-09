#!/usr/bin/env python3
"""
RealSense Skeleton Tracker - MediaPipe + Depth Fusion

Combines MediaPipe Pose (accurate 2D landmarks) with RealSense depth
for true 3D skeleton tracking. This is more accurate than running
pose detection on depth colormaps.

Key approach:
1. Use RGB stream for MediaPipe pose detection (accurate 2D)
2. Align depth to color frame
3. Sample depth at landmark positions for true 3D coordinates
4. Apply One-Euro filtering for smooth tracking

Run with: sudo python src/tracking/test_realsense_skeleton.py
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from collections import deque
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available")


@dataclass
class Landmark3D:
    """A single 3D landmark with confidence."""
    x: float  # X position in meters (camera space)
    y: float  # Y position in meters
    z: float  # Z depth in meters (distance from camera)
    visibility: float  # MediaPipe visibility [0-1]
    name: str = ""

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def is_valid(self) -> bool:
        return self.visibility > 0.5 and self.z > 0


@dataclass
class Skeleton3D:
    """Full 3D skeleton with all landmarks."""
    landmarks: List[Landmark3D]
    timestamp: float
    frame_id: int = 0

    # Body region indices (MediaPipe Pose)
    HEAD_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Face landmarks
    TORSO_INDICES = [11, 12, 23, 24]  # Shoulders + Hips
    ARM_LEFT_INDICES = [11, 13, 15, 17, 19, 21]  # Left arm
    ARM_RIGHT_INDICES = [12, 14, 16, 18, 20, 22]  # Right arm
    LEG_LEFT_INDICES = [23, 25, 27, 29, 31]  # Left leg
    LEG_RIGHT_INDICES = [24, 26, 28, 30, 32]  # Right leg

    # Key landmarks for breathing detection
    CHEST_INDICES = [11, 12]  # Shoulders

    def get_chest_center(self) -> Optional[Landmark3D]:
        """Get center point between shoulders for breathing."""
        if len(self.landmarks) < 13:
            return None

        left_shoulder = self.landmarks[11]
        right_shoulder = self.landmarks[12]

        if not left_shoulder.is_valid or not right_shoulder.is_valid:
            return None

        return Landmark3D(
            x=(left_shoulder.x + right_shoulder.x) / 2,
            y=(left_shoulder.y + right_shoulder.y) / 2,
            z=(left_shoulder.z + right_shoulder.z) / 2,
            visibility=min(left_shoulder.visibility, right_shoulder.visibility),
            name="chest_center"
        )

    def get_region_depth(self, indices: List[int]) -> float:
        """Get average depth for a body region."""
        valid_depths = []
        for i in indices:
            if i < len(self.landmarks) and self.landmarks[i].is_valid:
                valid_depths.append(self.landmarks[i].z)

        return np.mean(valid_depths) if valid_depths else 0.0


class OneEuroFilter3D:
    """One-Euro filter for 3D landmark smoothing."""

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.5,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # Derivative
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filtered value
        alpha = self._alpha(np.mean(cutoff), dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat

        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class RealSenseSkeletonTracker:
    """
    3D Skeleton tracker using MediaPipe Pose + RealSense depth fusion.

    Usage:
        tracker = RealSenseSkeletonTracker()
        if tracker.start():
            while True:
                skeleton, color_image, depth_image = tracker.get_frame()
                if skeleton:
                    chest = skeleton.get_chest_center()
                    print(f"Chest depth: {chest.z:.3f}m")
    """

    # MediaPipe landmark names
    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_smoothing: bool = True,
        depth_sample_radius: int = 3
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_smoothing = enable_smoothing
        self.depth_sample_radius = depth_sample_radius

        # RealSense
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001  # Default, will be updated
        self.depth_intrinsics = None

        # MediaPipe
        self.pose = None

        # Smoothing filters (one per landmark dimension)
        self.filters: Dict[int, OneEuroFilter3D] = {}

        # State
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.color_format = "bgr8"  # Will be updated on start

    def start(self) -> bool:
        """Initialize RealSense and MediaPipe."""
        if not REALSENSE_AVAILABLE:
            print("Error: pyrealsense2 not installed")
            print("Install with: pip install pyrealsense2-macosx")
            return False

        if not MEDIAPIPE_AVAILABLE:
            print("Error: mediapipe not installed")
            return False

        # Try different configurations in order of preference
        # IR first: same sensor module as depth, no extra USB power on macOS
        configs_to_try = [
            ("infrared", rs.format.y8, rs.stream.infrared),
            ("bgr8", rs.format.bgr8, rs.stream.color),
            ("rgb8", rs.format.rgb8, rs.stream.color),
        ]

        profile = None
        for format_name, color_format, stream_type in configs_to_try:
            try:
                print(f"Trying {format_name} format...")

                # Create fresh pipeline and config for each attempt
                self.pipeline = rs.pipeline()
                config = rs.config()

                # Enable depth stream
                config.enable_stream(rs.stream.depth, self.width, self.height,
                                   rs.format.z16, self.fps)

                # Enable color/infrared stream
                if stream_type == rs.stream.infrared:
                    config.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                                       color_format, self.fps)
                else:
                    config.enable_stream(stream_type, self.width, self.height,
                                       color_format, self.fps)

                profile = self.pipeline.start(config)
                self.color_format = format_name
                print(f"Pipeline started with {format_name} format!")
                break

            except RuntimeError as e:
                print(f"  {format_name} failed: {str(e)[:50]}...")
                try:
                    self.pipeline.stop()
                except:
                    pass
                self.pipeline = None
                continue

        if profile is None:
            print("\nError: Could not start any stream configuration")
            print("TIP: Try unplugging and replugging the camera, then run with sudo")
            return False

        try:

            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth scale: {self.depth_scale}")

            # Disable IR projector if using IR stream (dot pattern confuses MediaPipe)
            if self.color_format == "infrared":
                try:
                    depth_sensor.set_option(rs.option.emitter_enabled, 0)
                    print("IR projector disabled (clean image for pose detection)")
                except Exception as e:
                    print(f"Warning: Could not disable IR emitter: {e}")

            # Get depth intrinsics for 3D projection
            depth_profile = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

            # Create alignment object (align depth to color)
            if self.color_format != "infrared":
                self.align = rs.align(rs.stream.color)
            else:
                self.align = rs.align(rs.stream.infrared)

            # Initialize MediaPipe Pose
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                static_image_mode=False,
                enable_segmentation=False
            )

            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0

            print("RealSense Skeleton Tracker started successfully!")
            return True

        except RuntimeError as e:
            print(f"Error starting RealSense: {e}")
            if "failed to set power state" in str(e):
                print("\nTIP: Run with sudo on macOS:")
                print("  sudo ./venv/bin/python your_script.py")
                print("\nIf still failing, try unplugging and replugging the camera.")
            return False

    def stop(self):
        """Clean up resources."""
        if self.pose:
            self.pose.close()
            self.pose = None

        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None

        self.is_running = False
        print("RealSense Skeleton Tracker stopped")

    def get_frame(self) -> Tuple[Optional[Skeleton3D], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture frame and detect skeleton.

        Returns:
            (skeleton, color_image, depth_image) or (None, None, None) on failure
        """
        if not self.is_running:
            return None, None, None

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)

            # Align depth to color/infrared
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()

            # Get color or infrared frame based on mode
            if self.color_format == "infrared":
                ir_frame = aligned_frames.get_infrared_frame()
                if not depth_frame or not ir_frame:
                    return None, None, None
                # Convert IR to 3-channel grayscale for MediaPipe
                ir_image = np.asanyarray(ir_frame.get_data())
                color_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            else:
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    return None, None, None
                color_image = np.asanyarray(color_frame.get_data())
                # Convert RGB8 to BGR for OpenCV display
                if self.color_format == "rgb8":
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())

            self.frame_count += 1

            # Run MediaPipe pose detection on color image
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)

            if not results.pose_landmarks:
                return None, color_image, depth_image

            # Convert landmarks to 3D with depth
            skeleton = self._create_skeleton_3d(
                results.pose_landmarks.landmark,
                depth_frame,
                color_image.shape
            )

            return skeleton, color_image, depth_image

        except RuntimeError as e:
            print(f"Frame capture error: {e}")
            return None, None, None

    def _create_skeleton_3d(
        self,
        mp_landmarks,
        depth_frame,
        image_shape: Tuple[int, int, int]
    ) -> Skeleton3D:
        """Convert MediaPipe landmarks to 3D using depth data."""
        height, width = image_shape[:2]
        landmarks_3d = []
        current_time = time.time()

        for i, lm in enumerate(mp_landmarks):
            # Get 2D pixel coordinates
            px = int(lm.x * width)
            py = int(lm.y * height)

            # Sample depth at landmark position (with radius for robustness)
            depth_m = self._sample_depth(depth_frame, px, py)

            # Store PIXEL coordinates for drawing (normalized 0-1)
            # and depth in meters for breathing detection
            x_norm = lm.x  # Keep normalized for easy drawing
            y_norm = lm.y
            z_m = depth_m if depth_m > 0 else 0

            # Apply smoothing to depth only
            if self.enable_smoothing and depth_m > 0:
                if i not in self.filters:
                    self.filters[i] = OneEuroFilter3D(min_cutoff=0.3, beta=0.3, d_cutoff=1.0)
                z_arr = np.array([z_m])
                z_smoothed = self.filters[i].filter(z_arr, current_time)
                z_m = float(z_smoothed[0])

            landmark = Landmark3D(
                x=x_norm,  # Normalized 0-1 for easy drawing
                y=y_norm,  # Normalized 0-1
                z=z_m,     # Depth in meters
                visibility=lm.visibility,
                name=self.LANDMARK_NAMES[i] if i < len(self.LANDMARK_NAMES) else f"point_{i}"
            )
            landmarks_3d.append(landmark)

        return Skeleton3D(
            landmarks=landmarks_3d,
            timestamp=current_time,
            frame_id=self.frame_count
        )

    def _sample_depth(self, depth_frame, x: int, y: int) -> float:
        """Sample depth at a point with neighborhood averaging."""
        h, w = self.height, self.width
        r = self.depth_sample_radius

        # Clamp coordinates
        x = max(r, min(x, w - r - 1))
        y = max(r, min(y, h - r - 1))

        # Sample neighborhood
        depths = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = depth_frame.get_distance(x + dx, y + dy)
                if d > 0.1 and d < 10.0:  # Valid range
                    depths.append(d)

        if not depths:
            return 0.0

        # Use median for robustness against outliers
        return float(np.median(depths))

    def _pixel_to_3d(self, px: int, py: int, depth_m: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates + depth to 3D camera coordinates."""
        if self.depth_intrinsics is None:
            return float(px), float(py), depth_m

        # Deproject using camera intrinsics
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics,
            [px, py],
            depth_m
        )
        return point_3d[0], point_3d[1], point_3d[2]

    def _smooth_landmark(
        self,
        idx: int,
        x: float,
        y: float,
        z: float,
        t: float
    ) -> Tuple[float, float, float]:
        """Apply One-Euro filter to landmark position."""
        if idx not in self.filters:
            self.filters[idx] = OneEuroFilter3D(
                min_cutoff=0.5,
                beta=0.5,
                d_cutoff=1.0
            )

        pos = np.array([x, y, z])
        smoothed = self.filters[idx].filter(pos, t)
        return smoothed[0], smoothed[1], smoothed[2]

    def reset_filters(self):
        """Reset all smoothing filters."""
        for f in self.filters.values():
            f.reset()
        self.filters.clear()

    def get_fps(self) -> float:
        """Get current frame rate."""
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0


def draw_skeleton_3d(
    image: np.ndarray,
    skeleton: Skeleton3D,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw 3D skeleton on image with depth coloring."""

    if skeleton is None:
        return image

    h, w = image.shape[:2]

    # Connection pairs for skeleton (MediaPipe Pose)
    CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
        (9, 10),  # Mouth
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]

    # Get depth range for coloring
    valid_depths = [lm.z for lm in skeleton.landmarks if lm.visibility > 0.5 and lm.z > 0]
    if valid_depths:
        min_depth = min(valid_depths)
        max_depth = max(valid_depths)
        depth_range = max(max_depth - min_depth, 0.05)
    else:
        min_depth, depth_range = 0.5, 0.5

    # Draw connections first (behind points)
    for start_idx, end_idx in CONNECTIONS:
        if start_idx >= len(skeleton.landmarks) or end_idx >= len(skeleton.landmarks):
            continue

        start_lm = skeleton.landmarks[start_idx]
        end_lm = skeleton.landmarks[end_idx]

        # Need visibility > 0.3 to draw
        if start_lm.visibility < 0.3 or end_lm.visibility < 0.3:
            continue

        # Convert normalized coordinates to pixels
        start_x = int(start_lm.x * w)
        start_y = int(start_lm.y * h)
        end_x = int(end_lm.x * w)
        end_y = int(end_lm.y * h)

        # Color based on depth (green = close, blue = far)
        avg_depth = (start_lm.z + end_lm.z) / 2 if (start_lm.z > 0 and end_lm.z > 0) else min_depth
        depth_norm = (avg_depth - min_depth) / depth_range
        depth_norm = max(0, min(1, depth_norm))

        # Green for close, blue for far
        g = int(255 * (1 - depth_norm))
        b = int(255 * depth_norm)
        line_color = (b, g, 100)

        cv2.line(image, (start_x, start_y), (end_x, end_y), line_color, thickness)

    # Draw landmarks
    for i, lm in enumerate(skeleton.landmarks):
        if lm.visibility < 0.3:
            continue

        x = int(lm.x * w)
        y = int(lm.y * h)

        # Size based on visibility
        radius = int(3 + 5 * lm.visibility)

        # Color based on depth
        if lm.z > 0:
            depth_norm = (lm.z - min_depth) / depth_range
            depth_norm = max(0, min(1, depth_norm))
            g = int(255 * (1 - depth_norm))
            b = int(255 * depth_norm)
            point_color = (b, g, 100)
        else:
            point_color = (100, 100, 100)  # Gray for unknown depth

        cv2.circle(image, (x, y), radius, point_color, -1)
        cv2.circle(image, (x, y), radius, (255, 255, 255), 1)

    return image


if __name__ == "__main__":
    import cv2

    print("RealSense Skeleton Tracker - MediaPipe + Depth Fusion")
    print("=" * 50)
    print("Run the full test with:")
    print("  sudo ./venv/bin/python src/tracking/test_realsense_skeleton.py")
    print()

    # Quick test
    tracker = RealSenseSkeletonTracker()
    if tracker.start():
        print("\nCapturing 10 test frames...")
        for i in range(10):
            skeleton, color, depth = tracker.get_frame()
            if skeleton:
                chest = skeleton.get_chest_center()
                if chest:
                    print(f"Frame {i}: Chest depth = {chest.z:.3f}m")
                else:
                    print(f"Frame {i}: No chest detected")
            else:
                print(f"Frame {i}: No skeleton")

        tracker.stop()
    else:
        print("Failed to start tracker")
