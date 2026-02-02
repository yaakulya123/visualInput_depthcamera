#!/bin/bash
# Quick camera list without preview

echo ""
echo "Scanning for available cameras..."
echo ""

source venv/bin/activate

python3 2>/dev/null << 'EOF'
import cv2

print("Available cameras:")
print("")

found = False
for idx in range(10):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Camera 0 is usually the built-in Mac camera
            if idx == 0:
                cam_type = "💻 Built-in Camera (likely Mac)"
            elif w == 640 and h == 480:
                cam_type = "📱 Secondary Camera (likely iPhone)"
            else:
                cam_type = "📷 External Camera"

            print(f"  Camera {idx}: {cam_type} - {w}x{h}")
            found = True
        cap.release()

if not found:
    print("  No cameras found!")
print("")
EOF

echo "Usually your Mac's front camera is Camera 0"
echo ""
