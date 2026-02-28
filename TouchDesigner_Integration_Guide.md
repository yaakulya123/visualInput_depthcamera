# Liquid Stillness - TouchDesigner Integration Guide

## What's Happening

We have a RealSense D435 depth camera mounted on the ceiling, tracking people in real-time from a top-down view. The tracking system detects:

- **Multiple people** — each person tracked independently with stable IDs across frames
- **Stillness / Jitter** — how still or restless each person's body is (per-person + group average)
- **Micro-motion** — pixel-level depth differencing that catches subtle movement skeleton tracking misses (finger twitches, weight shifts, clothing movement)
- **3D Clustering** — groups nearby people by real-world distance using depth deprojection
- **Stillness duration** — per-person timer of how long they've been continuously still (resets on movement)

All of this streams as JSON over WebSocket at ~12-30 updates per second.

> **Note:** Breathing detection was removed — inhale/exhale tracking was unreliable from the ceiling-mounted depth camera. If needed in the future, the depth data per person is still available for experimentation.


## How to Receive the Data

Connect to this WebSocket address:

```
ws://82.112.226.90:3000
```

No login, no VPN, no setup. As long as the tracking system is running on our end, data flows through.


## TouchDesigner Setup

### 1. Create a Web Client DAT
- **Protocol**: WebSocket
- **Address**: `82.112.226.90`
- **Port**: `3000`
- Toggle **Active** on

You should immediately start seeing JSON messages arrive.

### 2. Parse the Data

Attach a callback to the Web Client DAT:

```python
import json

def onReceiveText(dat, rowIndex, message, bytes, peer):
    data = json.loads(message)

    # --- Group level ---
    person_count  = data['person_count']      # int: how many people in view
    group_jitter  = data['group_jitter']      # 0.0 - 1.0
    micro_motion  = data['micro_motion']      # 0.0 - 1.0
    cluster_count = data['cluster_count']     # int: number of proximity groups

    # --- Per person ---
    for p in data['persons']:
        pid        = p['id']           # stable int ID
        jitter     = p['jitter']       # 0.0 - 1.0
        stillness  = p['stillness']    # seconds (0.0+, resets on movement)
        depth      = p['depth_mm']     # millimeters from camera
        cluster    = p['cluster_id']   # which group they belong to

    # --- Route to CHOPs ---
    op('group_jitter').par.value0 = group_jitter
    op('micro_motion').par.value0 = micro_motion
    op('person_count').par.value0 = person_count
```

### 3. Quick Test (Browser)

Verify data is flowing before opening TouchDesigner — open browser console (`Cmd+Option+J` on Mac, `F12` on Windows):

```javascript
ws = new WebSocket("ws://82.112.226.90:3000");
ws.onmessage = e => console.log(JSON.parse(e.data));
```


## Data Reference

### Message format

```json
{
  "person_count": 3,
  "group_jitter": 0.342,
  "micro_motion": 0.15,
  "cluster_count": 2,
  "persons": [
    { "id": 1, "jitter": 0.12, "stillness": 14.5, "depth_mm": 1250, "cluster_id": 1 },
    { "id": 2, "jitter": 0.45, "stillness": 0.0,  "depth_mm": 1500, "cluster_id": 1 },
    { "id": 3, "jitter": 0.08, "stillness": 22.3, "depth_mm": 1800, "cluster_id": 2 }
  ]
}
```

### Field reference with ranges

#### Group level (top-level fields)

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `person_count` | int | 0+ | Number of people currently in camera view |
| `group_jitter` | float | **0.0 - 1.0** | Max motion across all people. If ANY person is moving, this reflects their motion level. 0 = everyone still, 1 = someone chaotic |
| `micro_motion` | float | **0.0 - 1.0** | Pixel-level scene motion from depth frame differencing. Catches subtle stuff like finger twitches, weight shifts — things skeleton tracking misses |
| `cluster_count` | int | 0+ | Number of proximity groups (people within ~1m of each other get grouped) |

#### Per-person (inside `persons` array)

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `id` | int | 1+ | Stable person ID — persists across frames as long as the person is visible |
| `jitter` | float | **0.0 - 1.0** | This person's motion level. 0 = perfectly still, 1 = highly restless |
| `stillness` | float | **0.0+** (seconds) | How long this person has been continuously still. Resets to 0 the moment they move. Reaches 30+ for deep meditation |
| `depth_mm` | float | **~500 - 2500** | Distance from camera in millimeters. Closer = smaller number. Useful for proximity-based effects |
| `cluster_id` | int or null | 1+ | Which proximity group this person belongs to. People near each other share the same cluster_id. `null` if depth data unavailable |

### Mapping guide for visuals

#### Jitter → Fluid turbulence

| Jitter Range | Body State | Suggested Visual |
|-------------|------------|-----------------|
| 0.0 - 0.15 | Still / meditating | Smooth, honey-like, barely moving |
| 0.15 - 0.35 | Gentle movement | Soft ripples, slow waves |
| 0.35 - 0.6 | Active / fidgeting | Medium turbulence |
| 0.6 - 0.8 | Restless | Choppy, noisy |
| 0.8 - 1.0 | Chaotic | Full turbulence, high energy |

#### Stillness duration → Color / reward states

| Stillness (seconds) | State | Suggested Color |
|---------------------|-------|----------------|
| 0 - 5 | Settling in | Dark Purple, Deep Indigo |
| 5 - 15 | Focused | Cyan, Teal |
| 15 - 30 | Deep focus | Blue, Soft White |
| 30+ | Transcendent ("Golden State") | Gold, Amber, Warm White |

#### Micro-motion → Fine detail

| Micro-motion Range | What's happening | Suggested Use |
|-------------------|-----------------|---------------|
| 0.0 - 0.05 | Scene is dead still | Minimal particle emission |
| 0.05 - 0.2 | Subtle shifts | Light shimmer / sparkle |
| 0.2 - 0.5 | Noticeable movement | Particle bursts, ripple sources |
| 0.5 - 1.0 | Lots of motion | Heavy distortion, noise textures |

#### Cluster count → Social/spatial effects

| Cluster Count | Meaning | Suggested Use |
|--------------|---------|---------------|
| 0 | No one in view | Idle / screensaver state |
| 1 | Everyone is close together | Unified visual field |
| 2+ | People spread apart in groups | Split visual zones, different palettes per cluster |


## Files on Our End (for reference)

These are on the tracking machine — you only need the WebSocket:

| File | What it does |
|------|-------------|
| `src/tracking/test_realsense_audio.py` | Main app — camera + YOLO + stillness + clustering + WebSocket |
| `src/network/data_server.py` | HTTP dashboard + WebSocket sender to VPS |
| `src/stillness/stillness_detector.py` | Per-person body stillness/jitter scoring |
| `src/tracking/cluster_detector.py` | 3D proximity clustering via depth deprojection |
| `src/audio/sound_engine.py` | Audio layers driven by group jitter (runs on tracking machine only) |


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

**person_count is 0?**
- No one is in the camera's field of view, or the person is too far / too close.
- Camera range is roughly 0.5m - 3m for reliable detection.
