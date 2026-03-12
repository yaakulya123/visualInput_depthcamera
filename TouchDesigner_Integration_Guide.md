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

def onReceiveText(dat, rowIndex, message, *args):
    # 1. Debugging: Update text1 with raw JSON
    if op('text1'):
        op('text1').text = message
    
    try:
        # 2. Parse the JSON
        data = json.loads(message)
        persons = data.get('persons', [])
        
        # 3. Define the Target Table
        target = op('table1')
        if not target:
            return

        target.clear()
        
        # 4. Set Headers
        # We include global data plus individual person data
        headers = [
            'person_count', 'group_jitter', 'active_layers', 
            'p_id', 'p_jitter', 'p_stillness', 'p_depth'
        ]
        target.appendRow(headers)
        
        # 5. Global data values
        p_count = data.get('person_count', 0)
        g_jitter = data.get('group_jitter', 0)
        a_layers = data.get('active_layers', 0)

        # 6. Logic: If persons exist, create a row for each. If not, one row with 0s.
        if persons:
            for p in persons:
                row = [
                    p_count, 
                    g_jitter, 
                    a_layers,
                    p.get('id', 0),
                    p.get('jitter', 0),
                    p.get('stillness', 0),
                    p.get('depth_mm', 0)
                ]
                target.appendRow(row)
        else:
            # Placeholder row when no one is detected
            target.appendRow([p_count, g_jitter, a_layers, 0, 0, 0, 0])

    except Exception as e:
        print(f"WebSocket Parsing Error: {e}")
```

### 3. Create a table DAT

The code above works by parsing the JSON Data and appending it to a DAT Table called `table1`, all columns in row 0 correspond to the header names and values below it. You can then feed these data values into a `DAT To CHOP` and use them as you see fit, for example using a ` for arbitrary choosing.

### 4. Quick Test (Browser)

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
