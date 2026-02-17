# Liquid Stillness - TouchDesigner Integration Guide

## What's Happening

We have a RealSense D435 depth camera tracking people's **breathing** and **body stillness** in real-time. This data is being sent over WebSocket to a relay server on our VPS, so you can receive it from anywhere.

The tracking system detects:
- **Breathing** — chest rise and fall via depth sensing (inhale, exhale, hold phases)
- **Stillness / Jitter** — how still or restless the person's body is
- **Multiple people** — each person tracked independently, closest person = "primary"

All of this streams as JSON over WebSocket at ~12-30 updates per second.


## How to Receive the Data

Connect to this WebSocket address:

```
ws://82.112.226.90:3000
```

That's it. No login, no VPN, no setup. As long as the tracking system is running on our end, data flows through.


## TouchDesigner Setup

### 1. Create a Web Client DAT
- **Protocol**: WebSocket
- **Address**: `82.112.226.90`
- **Port**: `3000`
- Toggle **Active** on

You should immediately start seeing JSON messages arrive.

### 2. Parse the Data

Attach a callback to the Web Client DAT to extract the values you need:

```python
import json

def onReceiveText(dat, rowIndex, message, bytes, peer):
    data = json.loads(message)

    # --- Primary person (closest to camera) ---
    breathing_signal = data['primary']['breathing_signal']   # -1.0 to +1.0
    breathing_phase  = data['primary']['breathing_phase']    # "inhale" / "exhale" / "hold"
    bpm              = data['primary']['bpm']                # breaths per minute (~8-20)
    chest_depth      = data['primary']['chest_depth_mm']     # raw depth in millimeters

    # --- Group (all people combined) ---
    jitter        = data['group']['jitter']          # 0.0 (still) to 1.0 (restless)
    person_count  = data['group']['person_count']    # how many people detected
    audio_layers  = data['group']['audio_layers']    # how many audio layers active

    # --- Route to your CHOPs / TOPs ---
    # Example: push values into Constant CHOPs
    op('breathing').par.value0 = breathing_signal
    op('jitter').par.value0 = jitter
```

### 3. Quick Test (Browser)

If you want to verify data is flowing before opening TouchDesigner, open any browser console (`Cmd+Option+J` on Mac, `F12` on Windows) and paste:

```javascript
ws = new WebSocket("ws://82.112.226.90:3000");
ws.onmessage = e => console.log(JSON.parse(e.data));
```

You should see objects streaming in the console.


## Data Reference

### What each message looks like

```json
{
  "timestamp": 1771360404.908,
  "fps": 12.1,
  "group": {
    "person_count": 2,
    "jitter": 0.34,
    "audio_layers": 3
  },
  "primary": {
    "id": 1,
    "breathing_phase": "inhale",
    "breathing_signal": 0.62,
    "bpm": 14.2,
    "chest_depth_mm": 850.5
  },
  "persons": [
    { "id": 1, "jitter": 0.22, "depth_mm": 850.5, "is_primary": true },
    { "id": 3, "jitter": 0.45, "depth_mm": 1120.0, "is_primary": false }
  ]
}
```

### Key fields for the fluid simulation

| Field | Range | What it means | Use for |
|-------|-------|---------------|---------|
| `primary.breathing_signal` | -1.0 to +1.0 | Normalized breath position | Fluid pulse / brightness |
| `primary.breathing_phase` | inhale / exhale / hold | Current breath phase | Phase-based transitions |
| `primary.bpm` | ~8 - 20 | Breaths per minute | Rhythm / viscosity |
| `group.jitter` | 0.0 - 1.0 | How restless the group is | Fluid turbulence |
| `group.person_count` | 0+ | Number of people in frame | Scale effects |
| `primary.chest_depth_mm` | ~500 - 1500 | Raw chest distance from camera | Direct depth mapping |

### Mapping to fluid visuals (from the project spec)

| Body State | Jitter Range | Visual |
|------------|-------------|--------|
| Still / Meditating | 0.0 - 0.2 | Smooth, honey-like, slow waves |
| Gentle movement | 0.2 - 0.4 | Soft ripples |
| Active | 0.4 - 0.6 | Medium turbulence |
| Restless | 0.6 - 0.8 | Choppy, noisy |
| Chaotic | 0.8 - 1.0 | Full turbulence |

| Calm Duration | Color |
|---------------|-------|
| Starting / Moving | Dark Purple, Deep Indigo |
| Calming down | Cyan, Teal |
| Deep calm (30s+) | Gold, Amber ("Golden State") |


## Files on Our End (for reference)

These are on the tracking machine, you don't need them — just the WebSocket:

| File | What it does |
|------|-------------|
| `src/network/data_server.py` | HTTP dashboard + WebSocket sender to VPS |
| `src/tracking/test_realsense_audio.py` | Main tracking app (camera + YOLO + breathing + stillness) |
| `src/tracking/keypoint_adapter.py` | Converts pose keypoints between formats |
| `src/stillness/stillness_detector.py` | Per-person body stillness scoring |
| `src/audio/sound_engine.py` | Audio layers driven by jitter (runs on tracking machine) |

The VPS relay server at `82.112.226.90:3000` just passes messages through — anything our tracking system sends, you receive.


## Troubleshooting

**No data coming through?**
- The tracking system might not be running on our end. Check with us.
- Make sure you're connecting to `ws://82.112.226.90:3000` (not https, not a different port).

**Data is choppy or slow?**
- Expected rate is ~12-30 Hz depending on camera performance.
- Network latency between you and the VPS adds a few ms, shouldn't be noticeable.

**Connection drops?**
- The relay server auto-accepts reconnections. Just reconnect your Web Client DAT.
- Our tracking system also auto-reconnects if the relay restarts.
