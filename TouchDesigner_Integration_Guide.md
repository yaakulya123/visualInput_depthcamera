# Liquid Stillness - TouchDesigner Integration Guide

## What's Happening

We have a RealSense D435 depth camera mounted on the ceiling, tracking people in real-time from a top-down view. The tracking system detects:

- **Multiple people** — each person tracked independently with stable IDs across frames
- **Stillness / Jitter** — how still or restless each person's body is (per-person + group max)
- **Micro-motion** — pixel-level depth differencing that catches subtle movement skeleton tracking misses (finger twitches, weight shifts, clothing movement)
- **3D Clustering** — groups nearby people by real-world distance using depth deprojection
- **Stillness duration** — per-person timer of how long they've been continuously still (resets on movement)

All of this streams as JSON over WebSocket at ~10-30 updates per second.


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

    # --- Group level (4 values) ---
    person_count  = data['person_count']      # int: 0+
    group_jitter  = data['group_jitter']      # float: 0.0 - 1.0
    micro_motion  = data['micro_motion']      # float: 0.0 - 1.0
    cluster_count = data['cluster_count']     # int: 0+

    # --- Per person (5 values each) ---
    for p in data['persons']:
        pid        = p['id']           # int: stable ID (1+)
        jitter     = p['jitter']       # float: 0.0 - 1.0
        stillness  = p['stillness']    # float: seconds (0.0+)
        depth      = p['depth_mm']     # float: ~500 - 3000
        cluster    = p['cluster_id']   # int or None

    # --- Route to CHOPs ---
    op('group_jitter').par.value0 = group_jitter
    op('micro_motion').par.value0 = micro_motion
    op('person_count').par.value0 = person_count
    op('cluster_count').par.value0 = cluster_count
```

### 3. Quick Test (Browser)

Verify data is flowing before opening TouchDesigner — open browser console (`Cmd+Option+J` on Mac, `F12` on Windows):

```javascript
ws = new WebSocket("ws://82.112.226.90:3000");
ws.onmessage = e => console.log(JSON.parse(e.data));
```


## Data Reference

### Full message format (exactly what you receive)

```json
{
  "person_count": 1,
  "group_jitter": 0.052,
  "micro_motion": 1.0,
  "cluster_count": 1,
  "persons": [
    { "id": 3, "jitter": 0.074, "stillness": 9.7, "depth_mm": 545.0, "cluster_id": 3 }
  ]
}
```

That's it — **4 group-level values + 5 values per person**. Nothing else is sent.

### Real-world example (actual TouchDesigner capture)

Here's what the data looks like in a real session with 1 person standing still, then slightly moving:

```
Line  person_count  group_jitter  micro_motion  id  jitter  stillness  depth_mm  cluster_id
───── ──────────── ──────────── ──────────── ──── ─────── ────────── ──────── ──────────
 1        1            0.0          1.0        3    0.0      9.3      522.0       3
 2        1            0.0          1.0        3    0.0      9.4      536.0       3
 3        1            0.0          1.0        3    0.0      9.5      528.0       3
 4        1            0.0          1.0        3    0.0      9.6      526.0       3
 5        1            0.052        1.0        3    0.074    9.7      545.0       3    ← slight movement
 6        1            0.058        1.0        3    0.061    9.8      530.0       3
 7        1            0.052        1.0        3    0.049    9.9      542.0       3
 8        1            0.036        1.0        3    0.03     10.0     537.0       3    ← settling back
 9        1            0.023        1.0        3    0.018    10.1     553.0       3
10        1            0.014        1.0        3    0.011    10.1     539.0       3    ← nearly still again
```

**What you can see:**
- `person_count` = 1 the whole time (one person in view)
- `group_jitter` jumped from 0.0 to 0.052 on line 5 (person moved slightly), then faded back down
- `jitter` (per-person) mirrors this — peaked at 0.074, decayed to 0.011
- `stillness` kept counting up (9.3 → 10.1 seconds) because the movement was too small to reset it
- `depth_mm` hovers around 522-553mm (person ~0.5m below the ceiling camera)
- `micro_motion` stayed at 1.0 (high scene-level motion from depth noise at this distance)
- `cluster_id` = 3 (this person's cluster, only one group since only one person)


### Field reference — Group level (top-level fields)

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `person_count` | int | **0+** | Number of people currently detected in the camera's field of view. 0 means nobody is visible. |
| `group_jitter` | float | **0.0 - 1.0** | The **maximum** jitter across all detected people. If ANY single person is moving, this value reflects that person's motion level. Only drops to 0 when ALL people are completely still. This is what drives the audio layers on our end. |
| `micro_motion` | float | **0.0 - 1.0** | Pixel-level scene motion from raw depth frame differencing (no skeleton involved). Catches very subtle stuff that skeleton tracking misses — finger twitches, weight shifts, clothing movement, even someone swaying slightly. Think of it as a "scene vibration" sensor. |
| `cluster_count` | int | **0+** | Number of spatial proximity groups. People standing within ~1 meter of each other are grouped into the same cluster. If everyone is close together, this is 1. If spread out in separate spots, this increases. |

### Field reference — Per-person (inside `persons` array)

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `id` | int | **1+** | Stable person ID that persists across frames as long as the person stays in view. If someone leaves and comes back, they get a new ID. Use this to track individuals over time. |
| `jitter` | float | **0.0 - 1.0** | This specific person's motion level. 0.0 = perfectly still (not moving at all), 1.0 = highly restless/chaotic movement. Calculated from skeleton keypoint displacement, weighted by body region (arms weigh more than head). |
| `stillness` | float | **0.0+** (seconds) | How many seconds this person has been continuously still. Starts at 0, counts up as long as they don't move. **Resets back to 0 the instant they move.** Can reach 30+ seconds for deep meditation. This is the key value for color/reward transitions. |
| `depth_mm` | float | **~500 - 3000** | Distance from the camera to this person's shoulders, in millimeters. Camera is mounted on the ceiling looking down, so this is roughly how far below the camera they are. Closer to camera = smaller number. Can use for proximity-based visual effects or layering. |
| `cluster_id` | int or null | **1+** or **null** | Which proximity group this person belongs to. People standing near each other share the same `cluster_id`. Use this to give different visual zones or color palettes to different groups. `null` means depth data wasn't available for this person (rare). |


## Mapping Guide for Visuals

### Jitter → Fluid turbulence

| Jitter Range | What the person is doing | Suggested Visual |
|-------------|--------------------------|-----------------|
| **0.0 - 0.1** | Completely still, meditating | Smooth, honey-like, barely moving. Glass-like surface. |
| **0.1 - 0.2** | Very slight movement, breathing | Gentle undulation, slow drift |
| **0.2 - 0.4** | Gentle movement, small adjustments | Soft ripples, slow waves |
| **0.4 - 0.6** | Active, fidgeting, shifting weight | Medium turbulence, visible currents |
| **0.6 - 0.8** | Restless, frequent movement | Choppy, noisy, agitated surface |
| **0.8 - 1.0** | Chaotic, walking, waving arms | Full turbulence, high energy, breaking waves |

### Stillness duration → Color / reward states

| Stillness (seconds) | State | Suggested Color | Notes |
|---------------------|-------|----------------|-------|
| **0 - 5** | Just arrived / settling in | Dark Purple, Deep Indigo | Person just stopped moving or just entered |
| **5 - 15** | Focused | Cyan, Teal | They're actively holding still |
| **15 - 30** | Deep focus | Blue, Soft White | Sustained commitment to stillness |
| **30+** | Transcendent ("Golden State") | Gold, Amber, Warm White | This is the reward — hard to reach, feels earned |

**Important:** `stillness` resets to 0.0 instantly when the person moves. So a person at 28 seconds who fidgets drops back to 0 and has to earn it again. This creates natural tension and reward cycles.

### Micro-motion → Fine detail / texture

| Micro-motion Range | What's happening in the scene | Suggested Use |
|-------------------|------------------------------|---------------|
| **0.0 - 0.05** | Scene is dead still, nobody moving at all | Minimal particle emission, calm ambient |
| **0.05 - 0.15** | Very subtle shifts (weight transfer, breathing) | Light shimmer, gentle sparkle overlay |
| **0.15 - 0.3** | Small movements happening | Particle trails, soft ripple sources |
| **0.3 - 0.5** | Noticeable movement in the scene | Particle bursts, visible distortion |
| **0.5 - 1.0** | Lots of motion (people walking, arms moving) | Heavy distortion, noise textures, energy waves |

**Tip:** `micro_motion` is great for adding texture/detail on top of the main `group_jitter` turbulence. Jitter drives the big fluid behavior, micro-motion drives fine particles/shimmer.

### Cluster count → Social/spatial effects

| Cluster Count | Meaning | Suggested Use |
|--------------|---------|---------------|
| **0** | No one in view | Idle / screensaver / ambient state |
| **1** | Everyone is close together (within ~1m) | Unified visual field, shared colors, collective energy |
| **2+** | People spread apart in separate groups | Split visual zones, different color palettes per cluster, independent fluid regions |

### Depth → Layering / proximity effects

| Depth Range (mm) | What it means | Suggested Use |
|------------------|---------------|---------------|
| **500 - 1000** | Very close to camera (unusual) | Intense, close-up effects |
| **1000 - 1500** | Normal standing distance below ceiling camera | Standard visual treatment |
| **1500 - 2000** | Further from camera | Slightly softer/distant effects |
| **2000 - 3000** | Far from camera, near edge of detection | Fade-out zone, subtle presence |

### Person count → Scene intensity

| Person Count | Suggested Use |
|-------------|---------------|
| **0** | Idle state — ambient visuals, waiting mode |
| **1** | Solo experience — personal, intimate, responsive to one person |
| **2-3** | Small group — can show individual responses or blend into group |
| **4+** | Crowd — more chaotic energy, collective behavior dominates |


## How the Motion Detection Works (Technical)

For anyone curious about what's under the hood:

```
Camera frame (IR 640x480 @ 30fps)
    ↓
YOLO11n-pose → detects all people + 17 skeleton keypoints each
    ↓
ByteTrack → assigns stable IDs across frames
    ↓
Per person:
    Keypoints shifted to bounding-box-relative coordinates
    → Normalized by bbox size (distance-independent)
    → One-Euro adaptive filter (removes sensor noise)
    → Frame-to-frame displacement per keypoint
    → Regional weighting (arms ×1.5, legs ×1.2, torso ×1.0, head ×0.5)
    → Temporal smoothing (EMA α=0.4)
    → Piecewise linear mapping → jitter score (0-1)
    ↓
Group jitter = MAX of all per-person jitter scores
    → Smoothed with EMA (α=0.7)
    → Sent as group_jitter

Micro-motion (separate from skeleton):
    Raw depth frame differencing (pixel-by-pixel)
    → Noise thresholding → decay accumulation
    → Normalized to 0-1 → sent as micro_motion
```

**Key detail:** Motion sensitivity is distance-independent. A person 3 meters from the camera registers the same jitter as someone 1 meter away for the same body movement, because keypoints are normalized by bounding box size, not frame size.


## Troubleshooting

**No data coming through?**
- The tracking system might not be running on our end. Check with us.
- Make sure you're connecting to `ws://82.112.226.90:3000` (not https, not a different port).

**Data is choppy or slow?**
- Expected rate is ~10-30 Hz depending on how many people are in view.
- Network latency between you and the VPS adds a few ms, shouldn't be noticeable.

**Connection drops?**
- The relay server auto-accepts reconnections. Just reconnect your Web Client DAT.
- Our tracking system also auto-reconnects if the relay restarts.

**person_count is 0?**
- No one is in the camera's field of view.
- Camera detection range is roughly 0.5m - 3m from the lens.
- People at the edge of frame or very far away may not be detected.

**group_jitter stays at 0 even when people move?**
- Each new person has a ~0.3 second calibration period when first detected.
- After that, any movement should register immediately.

**Values seem stuck or not updating?**
- Check that the WebSocket connection is still open (green status in Web Client DAT).
- Try disconnecting and reconnecting.
