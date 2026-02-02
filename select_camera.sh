#!/bin/bash
# Quick camera selection with preview

echo ""
echo "================================"
echo "  Camera Selection Tool"
echo "================================"
echo ""
echo "This will help you choose the right camera."
echo ""

source venv/bin/activate
python src/diagnostics/list_cameras.py

echo ""
echo "Camera selection saved!"
echo "Now you can run:"
echo "  python src/breathing/test_breathing_detection.py"
echo "  python src/stillness/test_stillness_detection.py"
echo ""
