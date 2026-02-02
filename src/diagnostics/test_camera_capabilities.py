#!/usr/bin/env python3
"""
Camera Capabilities Diagnostic
Tests what RealSense D435 features are available on this system.

Run: python src/diagnostics/test_camera_capabilities.py
"""

import sys
import time

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_result(name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} | {name}")
    if details:
        print(f"         {details}")

def test_opencv():
    """Test OpenCV availability and camera access."""
    print_header("TEST 1: OpenCV + RGB Camera")

    try:
        import cv2
        print_result("OpenCV imported", True, f"Version: {cv2.__version__}")
    except ImportError as e:
        print_result("OpenCV imported", False, str(e))
        return False, None

    # Try to find RealSense camera
    cap = None
    working_index = None

    for idx in range(3):  # Check indices 0, 1, 2
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print_result(f"Camera index {idx}", True, f"Resolution: {w}x{h}")
                working_index = idx
                cap.release()
                break
        cap.release()

    if working_index is None:
        print_result("Camera access", False, "No camera found at indices 0-2")
        return False, None

    return True, working_index

def test_realsense_sdk():
    """Test pyrealsense2 SDK and depth stream."""
    print_header("TEST 2: Intel RealSense SDK (Depth)")

    try:
        import pyrealsense2 as rs
        print_result("pyrealsense2 imported", True, f"Version: {rs.__version__ if hasattr(rs, '__version__') else 'N/A'}")
    except ImportError as e:
        print_result("pyrealsense2 imported", False, str(e))
        print("\n  💡 Install with: pip install pyrealsense2")
        return False, None

    # Check for connected devices
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print_result("RealSense device found", False, "No devices connected")
        return False, None

    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print_result(f"Device {i}", True, f"{name} (S/N: {serial})")

    # Try to start depth stream
    print("\n  Attempting to start depth stream...")
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        # Configure depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start pipeline
        profile = pipeline.start(config)
        print_result("Depth stream started", True)

        # Try to get frames
        print("  Waiting for depth frames (5 second timeout)...")
        start_time = time.time()
        frames_received = 0

        while time.time() - start_time < 5:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                frames_received += 1
                if frames_received == 1:
                    w = depth_frame.get_width()
                    h = depth_frame.get_height()
                    print_result("Depth frames received", True, f"Resolution: {w}x{h}")
                if frames_received >= 10:
                    break

        pipeline.stop()

        if frames_received > 0:
            print_result("Depth capture working", True, f"Got {frames_received} frames")
            return True, "depth"
        else:
            print_result("Depth capture working", False, "No frames received")
            return False, None

    except Exception as e:
        print_result("Depth stream", False, str(e))
        try:
            pipeline.stop()
        except:
            pass
        return False, None

def test_mediapipe():
    """Test MediaPipe pose detection as fallback."""
    print_header("TEST 3: MediaPipe Pose (Fallback)")

    try:
        import mediapipe as mp
        print_result("MediaPipe imported", True, f"Version: {mp.__version__}")
    except ImportError as e:
        print_result("MediaPipe imported", False, str(e))
        return False

    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print_result("Pose detector initialized", True)
        pose.close()
        return True
    except Exception as e:
        print_result("Pose detector", False, str(e))
        return False

def main():
    print("\n" + "="*60)
    print("  LIQUID STILLNESS - Camera Diagnostics")
    print("  Intel RealSense D435 Capability Test")
    print("="*60)

    results = {
        "opencv_rgb": False,
        "realsense_depth": False,
        "mediapipe_pose": False,
        "camera_index": None
    }

    # Test 1: OpenCV RGB
    success, cam_idx = test_opencv()
    results["opencv_rgb"] = success
    results["camera_index"] = cam_idx

    # Test 2: RealSense Depth SDK
    success, mode = test_realsense_sdk()
    results["realsense_depth"] = success

    # Test 3: MediaPipe (fallback)
    results["mediapipe_pose"] = test_mediapipe()

    # Summary and Recommendations
    print_header("SUMMARY & RECOMMENDATIONS")

    if results["realsense_depth"]:
        print("""
  🎉 EXCELLENT! Full depth access is working!

  Recommended approach:
  → Use pyrealsense2 for depth-based breathing detection
  → Track chest Z-axis with 5-10mm precision
  → This is the ideal setup for Liquid Stillness
        """)
        recommended = "depth"
    elif results["opencv_rgb"] and results["mediapipe_pose"]:
        print("""
  ⚠️  Depth access blocked (known macOS issue)
  ✅ RGB + MediaPipe available as fallback

  Recommended approach:
  → Use MediaPipe Pose to track shoulder landmarks
  → Detect breathing from Y-axis shoulder movement
  → Less precise but functional for breathing detection

  For full depth access, consider:
  → Running on Linux (native or VM with USB passthrough)
        """)
        recommended = "mediapipe"
    elif results["opencv_rgb"]:
        print("""
  ⚠️  Limited capabilities
  ✅ RGB camera works
  ❌ No depth or pose tracking

  Recommended:
  → Install MediaPipe: pip install mediapipe
  → Then re-run this diagnostic
        """)
        recommended = "limited"
    else:
        print("""
  ❌ No camera access detected

  Troubleshooting:
  1. Check USB connection
  2. Grant camera permissions in System Settings
  3. Try: pip install opencv-python pyrealsense2 mediapipe
        """)
        recommended = "none"

    print(f"\n  Camera Index: {results['camera_index']}")
    print(f"  Recommended Mode: {recommended.upper()}")

    # Save results
    import json
    results["recommended_mode"] = recommended
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    output_path = "outputs/diagnostic_results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")
    except:
        pass

    print("\n" + "="*60 + "\n")

    return results

if __name__ == "__main__":
    main()
