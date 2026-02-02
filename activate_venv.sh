#!/bin/bash
# Activate virtual environment for Liquid Stillness project

echo "Activating Liquid Stillness virtual environment..."
source venv/bin/activate

echo ""
echo "✅ Virtual environment activated!"
echo ""
echo "Available commands:"
echo "  ./select_camera.sh                                     # Preview & select camera (NEW!)"
echo "  python src/breathing/test_breathing_detection.py      # Test breathing detection"
echo "  python src/stillness/test_stillness_detection.py      # Test stillness/jitter detection"
echo "  ./reset_camera.sh                                      # Reset camera selection"
echo ""
echo "Tip: Use ./select_camera.sh to see which camera is which!"
echo ""
echo "To deactivate: deactivate"
echo ""
