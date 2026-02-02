# Liquid Stillness

An immersive biofeedback meditation installation that tracks a user's breathing and body stillness using depth sensing, then drives a generative fluid simulation projected onto the ceiling.

## Overview

Liquid Stillness creates a closed-loop biofeedback experience: the user lies on the floor beneath a ceiling-mounted camera and projector. Their breathing patterns and physical stillness are captured in real time and translated into a fluid simulation overhead. Calm, rhythmic breathing produces smooth, luminous waves; restless movement produces turbulent, chaotic flow. The system rewards sustained meditation with a visual transition from deep indigo to luminescent gold.

### Core Interaction Model

| User Behavior | Visual Response |
|---------------|-----------------|
| Deep, slow breathing | Smooth pulsing waves, bright highlights |
| Shallow, fast breathing | Choppy, fragmented fluid motion |
| Physical stillness | Laminar flow, reduced turbulence |
| Restless movement | High-velocity ripples, scattered light |
| Sustained calm (30s+) | "Golden State" -- viscosity increases, palette shifts to gold |

## System Architecture

```
                    Intel RealSense D435 (Ceiling-Mounted)
                                |
                          Depth / RGB Feed
                                |
                    +-----------+-----------+
                    |                       |
            Breathing Detector       Stillness Detector
          (MediaPipe Pose-Based)   (One-Euro Filtered, Hybrid)
                    |                       |
             Signal [-1, +1]         Jitter Score [0, 1]
             Phase, BPM              Regional Motion, Duration
                    |                       |
                    +-----------+-----------+
                                |
                      Fluid Simulation Engine
                       (Navier-Stokes Solver)
                                |
                        Ceiling Projection
```

### Breathing Detection

The breathing module tracks chest rise and fall by monitoring shoulder landmark positions on the Y-axis via MediaPipe Pose estimation. The raw positional signal passes through exponential smoothing and is normalized to a `[-1, +1]` range representing the full inhale-to-exhale cycle.

**Output metrics:**
- Normalized breathing signal
- Breathing phase (inhale / exhale / hold)
- Breaths per minute (BPM)
- Amplitude and confidence scores

**Algorithm pipeline:**
1. MediaPipe Pose extracts shoulder landmarks (indices 11, 12)
2. Y-axis position is averaged and smoothed (exponential, alpha = 0.3)
3. 100-frame buffer enables waveform analysis
4. Phase detection via signal derivative with threshold gating
5. BPM calculated from inhale-to-exhale transition intervals

### Stillness Detection

The stillness module employs a hybrid approach: primary detection uses MediaPipe Pose landmarks filtered through a One-Euro adaptive filter, with dense optical flow (Farneback) as a fallback when pose confidence drops below threshold.

**Output metrics:**
- Jitter score (0.0 = still, 1.0 = restless)
- Motion classification (still / fidgeting / moving / restless)
- Sustained stillness duration
- Per-region motion breakdown (head, torso, arms, legs)

**One-Euro Filter:**
An adaptive low-pass filter that adjusts its cutoff frequency based on signal velocity. When the user is still, the filter applies heavy smoothing to eliminate sensor noise. When the user moves, the filter becomes responsive to preserve real motion. This approach is based on Casiez et al. (CHI 2012).

**Regional body weighting:**

| Region | Weight | Rationale |
|--------|--------|-----------|
| Torso | 1.0x | Core stability indicator |
| Arms | 1.5x | Most visible movement when supine |
| Legs | 1.2x | Moderate contribution |
| Head | 0.5x | Least relevant for meditation tracking |

**Meditation quality progression:**

| Duration | State | Description |
|----------|-------|-------------|
| 0 -- 5s | Settling | User adjusting position |
| 5 -- 15s | Focused | Meditation establishing |
| 15 -- 30s | Deep Focus | Sustained meditative state |
| 30s+ | Transcendent | Golden State visual reward triggered |

## Hardware Requirements

- **Camera:** Intel RealSense D435 (ceiling-mounted, top-down view, 2.0--2.5m height)
- **Projector:** Short-throw ceiling projector
- **Environment:** Functions in complete darkness using the D435's infrared emitter
- **Compute:** Any system capable of running Python 3.11+ with OpenCV and MediaPipe at 30 FPS

## Software Dependencies

- Python 3.11+
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- MediaPipe 0.10.9 (pinned for `mp.solutions` API compatibility)
- SciPy >= 1.11.0

Optional (for RealSense depth stream on Linux):
- pyrealsense2

## Project Structure

```
visualInput_depthCamera/
├── src/
│   ├── breathing/
│   │   ├── breath_detector.py            # Core breathing detection engine
│   │   └── test_breathing_detection.py   # Interactive test with real-time visualization
│   ├── stillness/
│   │   ├── one_euro_filter.py            # One-Euro adaptive smoothing filter
│   │   ├── stillness_detector.py         # Hybrid stillness/jitter detection engine
│   │   └── test_stillness_detection.py   # Interactive test with jitter meter and HUD
│   ├── diagnostics/
│   │   ├── list_cameras.py               # Multi-camera enumeration and selection
│   │   └── test_camera_capabilities.py   # Hardware capability diagnostics
│   └── visualization/                    # Fluid simulation (in development)
├── prd.txt                               # Product Requirements Document
├── requirements.txt                      # Python dependencies
└── activate_venv.sh                      # Virtual environment activation
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yaakulya123/visualInput_depthcamera.git
cd visualInput_depthcamera

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# List available cameras (useful for multi-camera setups)
python src/diagnostics/list_cameras.py

# Run breathing detection
python src/breathing/test_breathing_detection.py

# Run stillness detection
python src/stillness/test_stillness_detection.py
```

## Interactive Controls

### Breathing Detection Test

| Key | Action |
|-----|--------|
| `q` / ESC | Quit |
| `c` | Switch camera |
| `r` | Reset and recalibrate |
| `s` | Save screenshot |
| SPACE | Toggle waveform display |

### Stillness Detection Test

| Key | Action |
|-----|--------|
| `q` / ESC | Quit |
| `c` | Switch camera |
| `r` | Reset and recalibrate |
| `s` | Save screenshot |
| `+` / `-` | Adjust sensitivity |

## Platform Notes

**macOS:** The RealSense D435 depth stream is currently blocked on macOS due to a USB interface limitation (see [librealsense issue #9916](https://github.com/IntelRealSense/librealsense/issues/9916)). The system falls back to MediaPipe Pose-based detection, which tracks shoulder Y-axis movement as a proxy for breathing. This approach provides sufficient precision for driving visual feedback.

**Linux:** Full depth stream access is available natively. For macOS users requiring true depth data, running Linux in a VM with USB passthrough is a viable alternative.

## Planned Development

- Navier-Stokes fluid dynamics engine (GPU-accelerated)
- Breathing signal mapped to fluid pulse and brightness
- Jitter score mapped to fluid turbulence
- Color grading system (Deep Purple to Cyan to Gold)
- Viscosity control driven by breathing rhythm consistency
- Body silhouette integration as fluid obstacle
- Full-system integration with ceiling projection output

## Visual Design Language

| State | Palette | Fluid Behavior |
|-------|---------|----------------|
| Anxious / Moving | Dark purple, deep indigo | High turbulence, choppy waves |
| Calm / Breathing | Cyan, teal, soft white | Smooth pulsing, laminar flow |
| Deep Meditation | Gold, amber, warm white | Honey-like viscosity, slow drift |

## Performance Targets

- Input-to-visual latency: < 200ms
- Breathing resolution: distinguishes shallow from deep breaths (5--10mm chest displacement)
- Frame rate: 30 FPS minimum across the full pipeline

## References

- Casiez, G., Roussel, N., and Vogel, D. (2012). "1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems." *Proceedings of CHI 2012*. [https://gery.casiez.net/1euro/](https://gery.casiez.net/1euro/)
- Intel RealSense D435 Documentation. [https://www.intelrealsense.com/depth-camera-d435/](https://www.intelrealsense.com/depth-camera-d435/)
- MediaPipe Pose Estimation. [https://developers.google.com/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

## License

This project was developed as part of an academic installation. All rights reserved.
