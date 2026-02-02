# Camera Selection Guide

## Camera Types

When you run the breathing or stillness tests, cameras are automatically identified:

| Icon | Type | Resolution | Notes |
|------|------|------------|-------|
| 🎯 | RealSense D435 | 1920x1080 | High-quality depth camera |
| 💻 | Mac FaceTime HD | 1280x720 | Built-in Mac camera |
| 📱 | iPhone Continuity | 640x480 | iPhone via Continuity Camera |
| 📷 | Other Camera | Various | Generic camera |

## How to Select Camera

### Method 1: Run a test with PREVIEW (Recommended)
```bash
python src/breathing/test_breathing_detection.py
# or
python src/stillness/test_stillness_detection.py
```

You'll see:
```
Scanning for cameras...
  [0] 💻 Mac FaceTime HD (likely) - 1280x720
  [1] 📱 iPhone Continuity (likely) - 640x480

Options:
  [number] - Preview & select that camera
  [p] - Preview all cameras one by one
  [q] - Quit

Your choice: 0
```

**NEW!** The script will show you a preview window so you can SEE which camera is which before selecting!

When you type a number (like `0`), it will:
1. Open a preview window showing what that camera sees
2. Ask if you want to use that camera
3. Save your selection so you don't have to choose again

### Method 2: List all cameras with preview
```bash
python src/diagnostics/list_cameras.py
```

This shows detailed info and lets you preview each camera.

## Reset Camera Selection

If you want to choose a different camera:

```bash
./reset_camera.sh
```

Or manually delete:
```bash
rm outputs/selected_camera.txt
```

## Troubleshooting

### Wrong camera opens
- Run `./reset_camera.sh`
- Re-run test and carefully check the camera type icons
- Look at resolution to identify which is which

### Camera indices change
This can happen when you:
- Connect/disconnect cameras
- Restart computer
- Change USB ports

Solution: Just run reset_camera.sh and re-select

## Tips

- **For development**: Use your Mac FaceTime camera (💻)
- **For testing**: Use whatever camera is convenient
- **For production**: Use RealSense D435 (🎯) ceiling-mounted
