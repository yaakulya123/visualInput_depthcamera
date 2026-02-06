#!/usr/bin/env python3
"""
Multi-Person Tracking Test - Interactive test for multi-person breathing and stillness.

Controls:
    q / ESC  - Quit
    r        - Reset all tracking + recalibrate
    s        - Save screenshot to outputs/
    a        - Toggle analytics window
    c        - Change camera
    d        - Export session data to CSV

Run: python -m src.tracking.test_multi_person
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tracking.person_tracker import PersonTracker
from src.analytics.session_analytics import SessionAnalytics
from src.analytics.analytics_dashboard import AnalyticsDashboard


def get_saved_camera() -> int:
    """Load saved camera selection."""
    try:
        path = os.path.join(PROJECT_ROOT, "outputs", "selected_camera.txt")
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def select_camera_interactive() -> int:
    """Interactive camera selection with preview."""
    from src.diagnostics.list_cameras import list_available_cameras, select_camera

    cameras = list_available_cameras(max_test=5)
    if not cameras:
        print("No cameras found. Defaulting to index 0.")
        return 0

    selected = select_camera(cameras)
    return selected if selected is not None else 0


def run_multi_person_test(camera_index: int = None):
    """Main test loop for multi-person tracking."""

    print("\n" + "=" * 60)
    print("  LIQUID STILLNESS - Multi-Person Tracking Test")
    print("=" * 60)

    # Camera selection
    if camera_index is None:
        saved = get_saved_camera()
        print(f"\n  Saved camera index: {saved}")
        use_saved = input(f"  Use camera {saved}? [Y/n]: ").strip().lower()
        if use_saved == 'n':
            camera_index = select_camera_interactive()
        else:
            camera_index = saved

    print(f"\n  Using camera index: {camera_index}")

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"  ERROR: Could not open camera {camera_index}")
        print("  Trying camera 0...")
        camera_index = 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ERROR: No camera available.")
            return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera resolution: {actual_w}x{actual_h}")

    # Initialize components
    print("\n  Initializing tracker (downloading model if needed)...")
    try:
        tracker = PersonTracker(
            model_name="yolo11n-pose.pt",
            confidence_threshold=0.5,
            device="mps",  # Apple Silicon
        )
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        print("  Install dependencies: pip install ultralytics>=8.1.0 supervision>=0.19.0")
        cap.release()
        return
    except Exception as e:
        # Fallback to CPU if MPS fails
        print(f"  MPS device failed ({e}), falling back to CPU...")
        tracker = PersonTracker(
            model_name="yolo11n-pose.pt",
            confidence_threshold=0.5,
            device="cpu",
        )

    analytics = SessionAnalytics(sample_interval=0.1)
    dashboard = AnalyticsDashboard(analytics)

    # Ensure outputs directory exists
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

    print("\n  Ready! Controls:")
    print("    q/ESC - Quit")
    print("    r     - Reset tracking")
    print("    s     - Save screenshot")
    print("    a     - Toggle analytics window")
    print("    c     - Change camera")
    print("    d     - Export CSV data")
    print("-" * 60)

    window_name = "Multi-Person Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Frame read failed, retrying...")
                time.sleep(0.1)
                continue

            # Process frame through tracker
            multi_state, raw_info = tracker.process_frame(frame)

            # Record analytics
            analytics.record(multi_state)

            # Draw main view
            display = dashboard.draw_main_view(
                frame, multi_state, raw_info, tracker.fps
            )

            cv2.imshow(window_name, display)

            # Draw analytics window if enabled
            dashboard.draw_analytics_window(multi_state)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break

            elif key == ord('r'):
                print("  Resetting tracking...")
                tracker.reset()
                analytics.reset()
                print("  Reset complete.")

            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(PROJECT_ROOT, "outputs",
                                     f"multi_person_{timestamp}.png")
                cv2.imwrite(path, display)
                print(f"  Screenshot saved: {path}")

            elif key == ord('a'):
                dashboard.toggle_analytics()
                state = "ON" if dashboard.show_analytics else "OFF"
                print(f"  Analytics window: {state}")

            elif key == ord('c'):
                print("  Changing camera...")
                cap.release()
                camera_index = select_camera_interactive()
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    print(f"  ERROR: Could not open camera {camera_index}")
                    break
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                tracker.reset()
                print(f"  Switched to camera {camera_index}")

            elif key == ord('d'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(PROJECT_ROOT, "outputs",
                                         f"session_data_{timestamp}.csv")
                analytics.export_csv(csv_path)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        # Print session summary
        summary = analytics.get_summary()
        print(f"\n{summary}")

        cap.release()
        dashboard.cleanup()
        cv2.destroyAllWindows()
        print("\n  Done.\n")


if __name__ == "__main__":
    # Allow passing camera index as argument
    cam_idx = None
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            pass

    run_multi_person_test(camera_index=cam_idx)
