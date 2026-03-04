# Liquid Stillness - TouchDesigner Integration Guide

## What's Happening

RealSense D435 depth camera on the ceiling, tracking people in real-time from a top-down view. Streams data over WebSocket.

## How to Receive the Data

```
ws://82.112.226.90:3000
```

No login, no VPN, no setup. Data flows as long as the tracking system is running.


## TouchDesigner Setup

### 1. Create a Web Client DAT
- **Protocol**: WebSocket
- **Address**: `82.112.226.90`
- **Port**: `3000`
- Toggle **Active** on

### 2. Parse the Data

```python
import json

def onReceiveText(dat, rowIndex, message, bytes, peer):
    data = json.loads(message)

    # --- Group level (3 values) ---
    person_count  = data['person_count']      # int: 0+
    group_jitter  = data['group_jitter']      # float: 0.0 - 1.0
    active_layers = data['active_layers']     # int: 0 - 5

    # --- Per person (4 values each) ---
    for p in data['persons']:
        pid        = p['id']           # int: stable ID (1+)
        jitter     = p['jitter']       # float: 0.0 - 1.0
        stillness  = p['stillness']    # float: seconds (0.0+)
        depth      = p['depth_mm']     # float: ~500 - 3000

    # --- Route to CHOPs ---
    op('group_jitter').par.value0 = group_jitter
    op('active_layers').par.value0 = active_layers
    op('person_count').par.value0 = person_count
```

### 3. Quick Test (Browser)

```javascript
ws = new WebSocket("ws://82.112.226.90:3000");
ws.onmessage = e => console.log(JSON.parse(e.data));
```


## Data Reference

### Message format

```json
{
  "person_count": 2,
  "group_jitter": 0.35,
  "active_layers": 3,
  "persons": [
    { "id": 1, "jitter": 0.35, "stillness": 0.0, "depth_mm": 1250.0 },
    { "id": 3, "jitter": 0.08, "stillness": 22.3, "depth_mm": 1800.0 }
  ]
}
```

**3 group-level values + 4 values per person.**


### Group level fields

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `person_count` | int | **0 - 10+** | How many people the camera can see right now |
| `group_jitter` | float | **0.0 - 1.0** | The **maximum** motion level across all people. If ANY person is moving, this goes up. Only drops to 0 when EVERYONE is completely still. |
| `active_layers` | int | **0 - 5** | How many audio layers are currently playing. 1 = base drone only (calm), 5 = all layers active (chaotic). This directly tells you how intense the audio is right now. |

### Per-person fields (inside `persons` array)

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `id` | int | **1+** | Stable person ID that stays the same across frames. Use this to track individuals over time. New ID if they leave and come back. |
| `jitter` | float | **0.0 - 1.0** | This person's motion level. **0.0** = perfectly still, not moving at all. **1.0** = chaotic movement (waving arms, walking). Calculated from skeleton keypoint movement, so it tracks actual body motion. |
| `stillness` | float | **0.0 - 60+** (seconds) | How many seconds this person has been continuously still. Counts up while they're not moving. **Resets to 0 instantly** the moment they move. Can reach 30+ for deep meditation. |
| `depth_mm` | float | **~500 - 3000** | Distance from camera to this person in millimeters. Camera is on the ceiling, so this is how far below the camera they are. |


## Mapping Guide

### `active_layers` → Visual intensity

| Active Layers | Audio State | Suggested Visual |
|--------------|-------------|-----------------|
| **1** | Base theta drone only | Calm, minimal, ambient glow |
| **2** | +Theta layer | Gentle movement, soft ripples |
| **3** | +Bass layer | Medium energy, visible currents |
| **4** | +Melody layer | High energy, active turbulence |
| **5** | All layers playing | Full chaos, maximum turbulence |

### `group_jitter` → Fluid turbulence

| Jitter Range | Body State | Suggested Visual |
|-------------|------------|-----------------|
| **0.0 - 0.1** | Completely still | Smooth, glass-like, honey viscosity |
| **0.1 - 0.3** | Slight movement | Gentle ripples, slow waves |
| **0.3 - 0.6** | Active, fidgeting | Medium turbulence, visible currents |
| **0.6 - 1.0** | Restless / chaotic | Full turbulence, high energy |

### `stillness` → Color transitions

| Stillness (seconds) | State | Suggested Color |
|---------------------|-------|----------------|
| **0 - 5** | Settling in | Dark Purple, Deep Indigo |
| **5 - 15** | Focused | Cyan, Teal |
| **15 - 30** | Deep focus | Blue, Soft White |
| **30+** | Transcendent ("Golden State") | Gold, Amber, Warm White |

Stillness resets to 0 instantly when the person moves — they have to earn it again.

### `person_count` → Scene mode

| Count | Suggested Use |
|-------|---------------|
| **0** | Idle / screensaver state |
| **1** | Solo experience, personal and intimate |
| **2-3** | Small group, can show individual responses |
| **4+** | Crowd energy, collective behavior |


## Troubleshooting

**No data?** — Check the tracking system is running on our end. Connect to `ws://82.112.226.90:3000` (not https).

**person_count is 0?** — No one in camera view. Detection range is ~0.5m - 3m.

**Connection drops?** — Just reconnect. Both sides auto-reconnect.
