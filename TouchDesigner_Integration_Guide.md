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

### 4. Optional - Quick Test via Browser

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
