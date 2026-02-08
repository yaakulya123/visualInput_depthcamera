#!/usr/bin/env python3
"""
Minimal RealSense test - just depth, no color, no pose detection.
This is the simplest possible test to check if the camera works.

Run: sudo ./venv/bin/python src/diagnostics/test_camera_minimal.py
"""

import sys
import time

try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)


def main():
    print("\n" + "=" * 50)
    print("  MINIMAL REALSENSE TEST (depth only)")
    print("=" * 50)

    # Check for devices first
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("\n  ERROR: No RealSense camera detected!")
        print("  - Unplug camera")
        print("  - Restart your Mac (full restart, not sleep)")
        print("  - Plug camera back in")
        print("  - Run with sudo")
        return

    print(f"\n  Found {len(devices)} device(s)")
    for dev in devices:
        print(f"  - {dev.get_info(rs.camera_info.name)}")

    # Try depth-only (minimal bandwidth)
    print("\n  Starting depth-only stream...")

    pipeline = rs.pipeline()
    config = rs.config()

    # Only depth - minimal bandwidth requirement
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        print("  Pipeline started!")
    except RuntimeError as e:
        print(f"\n  ERROR starting pipeline: {e}")
        if "power state" in str(e):
            print("\n  SOLUTION:")
            print("  1. Unplug the RealSense camera")
            print("  2. RESTART YOUR MAC completely")
            print("  3. After Mac restarts, plug camera back in")
            print("  4. Run: sudo ./venv/bin/python this_script.py")
        return

    # Warmup - let camera stabilize
    print("  Warming up (3 seconds)...")
    time.sleep(3)

    # Try to get frames
    print("  Attempting to capture frames...")

    success_count = 0
    fail_count = 0

    for i in range(30):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            depth = frames.get_depth_frame()
            if depth:
                center_dist = depth.get_distance(320, 240)
                print(f"    Frame {i}: center depth = {center_dist:.3f}m")
                success_count += 1
            else:
                print(f"    Frame {i}: no depth data")
                fail_count += 1
        except RuntimeError as e:
            print(f"    Frame {i}: TIMEOUT - {e}")
            fail_count += 1
            if fail_count >= 5:
                print("\n  Too many failures!")
                break

    pipeline.stop()

    print(f"\n  Results: {success_count} successes, {fail_count} failures")

    if success_count > 0:
        print("\n  DEPTH STREAM WORKS!")
        print("  The issue may be with color stream or bandwidth.")
        print("  Try running the depth-only breathing test.")
    else:
        print("\n  CAMERA NOT WORKING PROPERLY")
        print("  SOLUTION: Restart your Mac completely, then try again.")


if __name__ == "__main__":
    main()
