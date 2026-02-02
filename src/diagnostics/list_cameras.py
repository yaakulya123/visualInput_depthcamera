#!/usr/bin/env python3
"""
Camera Listing Utility
Lists all available cameras and their properties.

Run: python src/diagnostics/list_cameras.py
"""

import cv2
import sys


def list_available_cameras(max_test=10):
    """
    Test camera indices and return list of available cameras.

    Args:
        max_test: Maximum number of indices to test

    Returns:
        List of dicts with camera info
    """
    print("\n" + "="*60)
    print("  Scanning for cameras...")
    print("="*60)

    available_cameras = []

    for index in range(max_test):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # Try to read a frame to verify it actually works
            ret, frame = cap.read()

            if ret and frame is not None:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()

                # Identify camera type based on resolution patterns
                camera_type = "Unknown"
                camera_icon = "📷"

                if width == 1920 and height == 1080:
                    camera_type = "RealSense D435 (likely)"
                    camera_icon = "🎯"
                elif width == 1280 and height == 720:
                    camera_type = "Mac FaceTime HD (likely)"
                    camera_icon = "💻"
                elif width == 640 and height == 480:
                    camera_type = "iPhone Continuity (likely)"
                    camera_icon = "📱"
                elif width >= 1280:
                    camera_type = "High-res camera"
                    camera_icon = "🎥"
                else:
                    camera_type = "Standard camera"
                    camera_icon = "📷"

                camera_info = {
                    "index": index,
                    "name": camera_type,
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "backend": backend,
                    "width": width,
                    "height": height,
                    "icon": camera_icon
                }

                available_cameras.append(camera_info)

                # Print camera info
                print(f"\n  [{index}] {camera_icon} {camera_type}")
                print(f"      Resolution: {width}x{height} @ {fps}fps")
                print(f"      Backend: {backend}")

            cap.release()

    if not available_cameras:
        print("\n  ❌ No cameras found!")
        print("  Troubleshooting:")
        print("    1. Check USB connections")
        print("    2. Grant camera permissions in System Settings")
        print("    3. Restart the application")
    else:
        print("\n" + "="*60)
        print(f"  Found {len(available_cameras)} camera(s)")
        print("="*60)

    return available_cameras


def select_camera(cameras):
    """
    Prompt user to select a camera.

    Args:
        cameras: List of camera dicts from list_available_cameras()

    Returns:
        Selected camera index, or None if cancelled
    """
    if not cameras:
        return None

    if len(cameras) == 1:
        print(f"\n  Using camera {cameras[0]['index']} (only one available)")
        return cameras[0]['index']

    print("\n" + "-"*60)
    print("  SELECT CAMERA:")
    for cam in cameras:
        icon = cam.get('icon', '📷')
        print(f"    [{cam['index']}] {icon} {cam['name']}")
        print(f"         {cam['resolution']} @ {cam['fps']}fps")
    print("    [q] Quit")
    print("-"*60)

    while True:
        try:
            choice = input("\n  Enter camera index: ").strip().lower()

            if choice == 'q':
                print("  Cancelled.")
                return None

            index = int(choice)

            # Validate index
            if any(cam['index'] == index for cam in cameras):
                selected = next(cam for cam in cameras if cam['index'] == index)
                print(f"\n  ✅ Selected: Camera {index} ({selected['resolution']})")
                return index
            else:
                print(f"  ❌ Invalid index. Please choose from: {[cam['index'] for cam in cameras]}")

        except ValueError:
            print("  ❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            return None


def test_camera_preview(camera_index):
    """
    Show a preview of the selected camera.

    Args:
        camera_index: Camera index to preview
    """
    print(f"\n  Opening preview for camera {camera_index}...")
    print("  Press 'q' or ESC to close preview")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"  ❌ Could not open camera {camera_index}")
        return

    # Set to higher resolution if available
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Preview resolution: {actual_w}x{actual_h}")

    window_name = f"Camera {camera_index} Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("  ❌ Failed to read frame")
                break

            # Add text overlay
            cv2.putText(
                frame,
                f"Camera {camera_index} - Press 'q' to close",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("  Preview closed")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("  LIQUID STILLNESS - Camera Selection")
    print("="*60)

    # List all available cameras
    cameras = list_available_cameras(max_test=10)

    if not cameras:
        sys.exit(1)

    # Let user select camera
    selected_index = select_camera(cameras)

    if selected_index is None:
        sys.exit(0)

    # Ask if they want to preview
    print("\n" + "-"*60)
    preview = input("  Show camera preview? [Y/n]: ").strip().lower()

    if preview != 'n':
        test_camera_preview(selected_index)

    # Save selection
    try:
        with open("outputs/selected_camera.txt", "w") as f:
            f.write(str(selected_index))
        print(f"\n  ✅ Camera selection saved to outputs/selected_camera.txt")
        print(f"     The breathing detection script will use camera {selected_index}")
    except:
        pass

    print("\n" + "="*60)
    print("  Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
