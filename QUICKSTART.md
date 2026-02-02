# Quick Start Guide - Liquid Stillness

## Test Breathing Detection (Now!)

```bash
# 1. Activate environment
source activate_venv.sh

# 2. Run breathing test
python src/breathing/test_breathing_detection.py
```

On first run, you'll be prompted to select a camera:
```
Scanning for cameras...
  [0] 💻 Mac FaceTime HD (likely) - 1280x720
  [1] 📱 iPhone Continuity (likely) - 640x480

Options:
  [number] - Preview & select that camera
  [p] - Preview all cameras one by one
  [q] - Quit

Your choice:
```

**NEW! Preview Feature:**
- Type `p` to preview ALL cameras one by one
- Type a number (like `0`) to preview THAT specific camera
- A window will pop up showing what the camera sees
- Confirm your selection when you see the right camera

**Camera icons:**
- 🎯 RealSense D435 (1920x1080)
- 💻 Mac FaceTime HD (1280x720)
- 📱 iPhone Continuity (640x480)

## What to Expect

The test window will show:

1. **Live camera feed** with your pose skeleton
2. **Yellow box** around shoulders (detection area)
3. **Breathing waveform** at bottom
4. **Pulsing circle** (top-right) that grows/shrinks with breath
5. **Metrics panel** showing:
   - Phase: INHALE / EXHALE / HOLD
   - Signal: -1 (exhale) to +1 (inhale)
   - Breath Rate: BPM
   - Confidence: Detection quality

## Controls

| Key | Action |
|-----|--------|
| `q` or ESC | Quit |
| `c` | **Change camera on-the-fly!** (NEW!) |
| `r` | Reset & recalibrate |
| `s` | Save screenshot |
| SPACE | Toggle waveform |

**NEW Feature:** Press `c` at any time to switch cameras without restarting!

## Tips for Best Results

1. **Stand/sit facing camera** (or lie down for ceiling mount test)
2. **Wait 2-3 seconds** for calibration
3. **Try slow, deep breaths** → watch waveform smooth out
4. **Try fast, shallow breaths** → watch waveform get choppy
5. If detection is poor, press `r` to recalibrate

## Troubleshooting

### "No cameras found"
- Check USB connections
- Grant camera permissions in System Settings

### "MediaPipe not found" or "has no attribute solutions"
```bash
./venv/bin/pip install mediapipe==0.10.9
```

### Camera selection not working
```bash
# Use dedicated camera list tool
python src/diagnostics/list_cameras.py
```

### Want to change camera
Reset the camera selection:
```bash
./reset_camera.sh
```
Then run the test again to re-select.

See `CAMERA_SELECTION.md` for detailed camera info.

---

## What's Next?

Once breathing detection works well:
1. Add stillness/jitter detection
2. Build fluid simulation (p5.js)
3. Integrate everything
4. Test with ceiling-mounted camera + projector

See `CLAUDE.md` for full technical details.
