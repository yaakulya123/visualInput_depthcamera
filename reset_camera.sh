#!/bin/bash
# Reset camera selection - forces camera picker on next run

echo ""
echo "Resetting camera selection..."
rm -f outputs/selected_camera.txt
echo "✅ Done! You'll be prompted to select camera on next run."
echo ""
