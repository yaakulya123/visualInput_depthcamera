#!/usr/bin/env python3
"""
RealSense Depth Camera Test

Tests if the Intel RealSense D435 depth stream works on macOS.
May need to run with sudo on macOS due to USB power state bug.

Run: sudo python src/diagnostics/test_realsense_depth.py
"""

import sys
import time

try:
    import pyrealsense2 as rs
    print("pyrealsense2 imported successfully")
except ImportError:
    print("ERROR: pyrealsense2 not installed")
    print("Install with: pip install pyrealsense2-macosx")
    sys.exit(1)

try:
    import numpy as np
    import cv2
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)


def test_device_detection():
    """Test if RealSense device is detected."""
    print("\n" + "=" * 50)
    print("  STEP 1: Device Detection")
    print("=" * 50)

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("  ERROR: No RealSense devices found!")
        print("  - Check USB connection (use USB 3.0 port)")
        print("  - Try unplugging and replugging the camera")
        return None

    print(f"  Found {len(devices)} device(s)")

    for i, dev in enumerate(devices):
        try:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            fw = dev.get_info(rs.camera_info.firmware_version)
            print(f"\n  Device {i}: {name}")
            print(f"    Serial: {serial}")
            print(f"    Firmware: {fw}")
        except RuntimeError as e:
            print(f"  ERROR getting device info: {e}")
            print("  TIP: Try running with sudo")
            return None

    return devices[0]


def test_depth_stream():
    """Test depth stream capture."""
    print("\n" + "=" * 50)
    print("  STEP 2: Depth Stream Test")
    print("=" * 50)

    pipeline = rs.pipeline()
    config = rs.config()

    # Configure depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        print("  Starting depth pipeline...")
        profile = pipeline.start(config)
        print("  Depth pipeline started!")

        # Get device info
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"  Depth scale: {depth_scale} meters per unit")

        # Capture a few frames
        print("\n  Capturing depth frames...")
        for i in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                print(f"    Frame {i}: No depth data")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            center_depth = depth_frame.get_distance(320, 240)

            print(f"    Frame {i}: shape={depth_image.shape}, "
                  f"center_depth={center_depth:.3f}m, "
                  f"min={depth_image.min()}, max={depth_image.max()}")

        pipeline.stop()
        print("\n  SUCCESS: Depth stream is working!")
        return True

    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        if "failed to set power state" in str(e):
            print("\n  This is a known macOS issue.")
            print("  TIP: Run this script with sudo:")
            print("       sudo python src/diagnostics/test_realsense_depth.py")
        return False
    finally:
        try:
            pipeline.stop()
        except:
            pass


def test_depth_and_color():
    """Test both depth and color streams with live preview."""
    print("\n" + "=" * 50)
    print("  STEP 3: Live Preview (Depth + Color)")
    print("=" * 50)
    print("  Press 'q' to quit preview")

    pipeline = rs.pipeline()
    config = rs.config()

    # Configure streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
        print("  Streams started!")

        # Create alignment object
        align = rs.align(rs.stream.color)

        cv2.namedWindow("RealSense Depth Test", cv2.WINDOW_NORMAL)

        frame_count = 0
        start_time = time.time()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            frame_count += 1

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Colorize depth for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Stack side by side
            display = np.hstack((color_image, depth_colormap))

            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Get center depth
            center_depth = depth_frame.get_distance(320, 240)

            # Add text overlay
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Center: {center_depth:.2f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Color", (280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Depth", (920, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("RealSense Depth Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n  Captured {frame_count} frames at {fps:.1f} FPS")
        print("  SUCCESS: Both streams working!")
        return True

    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        return False
    finally:
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
        except:
            pass


def main():
    print("\n" + "=" * 50)
    print("  INTEL REALSENSE D435 - macOS DEPTH TEST")
    print("=" * 50)
    print("  Testing pyrealsense2-macosx package")

    # Step 1: Device detection
    device = test_device_detection()
    if device is None:
        print("\n  FAILED: Could not detect device properly")
        print("  Try: sudo python src/diagnostics/test_realsense_depth.py")
        sys.exit(1)

    # Step 2: Depth stream test
    if not test_depth_stream():
        print("\n  FAILED: Depth stream not working")
        sys.exit(1)

    # Step 3: Live preview
    print("\n  Would you like to see a live preview? [Y/n]: ", end="")
    try:
        response = input().strip().lower()
        if response != 'n':
            test_depth_and_color()
    except EOFError:
        # Non-interactive mode, skip preview
        pass

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED!")
    print("  Your RealSense D435 depth stream is working on macOS!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
