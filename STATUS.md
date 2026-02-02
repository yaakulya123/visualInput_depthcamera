# Liquid Stillness - Project Status

**Last Updated:** 2026-01-28 Evening
**Current Phase:** Input Systems Complete ✅

---

## What's Working Now

### ✅ Breathing Detection
- **Method:** MediaPipe Pose (shoulder Y-axis tracking)
- **Output:** Signal -1 (exhale) to +1 (inhale)
- **Metrics:** Phase, BPM, amplitude, confidence
- **Visualization:** Real-time waveform, pulsing circle
- **Status:** Tested and working great

### ✅ Stillness Detection
- **Method:** One-Euro filtered landmark tracking + optical flow fallback
- **Output:** Jitter score 0 (still) to 1 (restless)
- **Features:** Regional body analysis (head, torso, arms, legs)
- **Quality Levels:** Settling → Focused → Deep Focus → Transcendent (30s+)
- **Status:** Tested and working great

### ✅ Supporting Systems
- Multi-camera selection (Mac, iPhone, RealSense)
- Virtual environment with all dependencies
- Real-time visualization frameworks

---

## Next Session Goals

### Fluid Simulation 🎨

**Research Phase:**
- Survey Navier-Stokes implementation approaches
- Choose technology: p5.js vs Processing vs custom shaders
- Determine Python ↔ JavaScript integration strategy

**Implementation Phase:**
1. Setup canvas (fullscreen, projection-ready)
2. Implement fluid dynamics solver
3. Add body silhouette as obstacle
4. Connect breathing signal → pulse/brightness
5. Connect jitter score → turbulence
6. Implement color grading (Purple → Cyan → Gold)

**Data Flow:**
```
Python (Breathing + Stillness)
         ↓
   OSC / WebSocket / File
         ↓
p5.js (Fluid Simulation)
         ↓
   Projector Output
```

---

## File Structure

```
visualInput_depthCamera/
├── src/
│   ├── breathing/
│   │   ├── breath_detector.py          ✅ Working
│   │   └── test_breathing_detection.py ✅ Working
│   ├── stillness/
│   │   ├── one_euro_filter.py          ✅ Working
│   │   ├── stillness_detector.py       ✅ Working
│   │   └── test_stillness_detection.py ✅ Working
│   ├── diagnostics/
│   │   └── list_cameras.py             ✅ Working
│   └── visualization/
│       └── [NEXT: fluid simulation]
├── CLAUDE.md      # Full technical context
├── README.md      # Project overview
└── STATUS.md      # This file
```

---

## Technical Notes

### Why MediaPipe Instead of Depth?
- Depth stream blocked on macOS (USB bug)
- MediaPipe provides sufficient precision for biofeedback
- Falls back to optical flow if pose detection fails

### Key Algorithms
- **One-Euro Filter:** Adaptive smoothing (smooth when still, responsive when moving)
- **Regional Weighting:** Arms 1.5x, torso 1.0x, head 0.5x
- **Temporal Analysis:** 30-frame buffer for sustained stillness detection

### Performance
- Target: 30 FPS for both detection systems
- Current: Achievable on M-series Macs
- Bottleneck: Fluid simulation (TBD)

---

## Installation Recap

```bash
# Already done ✅
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:**
- opencv-python 4.13.0
- numpy 2.4.1
- mediapipe 0.10.9
- scipy, matplotlib

---

## Next Time Checklist

- [ ] Research fluid simulation libraries
- [ ] Choose integration approach (OSC/WebSocket/file-based)
- [ ] Prototype basic fluid dynamics
- [ ] Test with breathing signal input
- [ ] Test with jitter signal input
- [ ] Implement color system

---

**Ready to build the visual magic! 🌊✨**
