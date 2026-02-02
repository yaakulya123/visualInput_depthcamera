# Next Session - Fluid Simulation 🌊

**Goal:** Build the visual output that responds to breathing and stillness

---

## What We Have

✅ **Breathing Signal**
- Range: -1 (exhale) → +1 (inhale)
- Smooth sine wave pattern
- BPM, phase detection

✅ **Jitter Score**
- Range: 0 (still) → 1 (restless)
- Regional body tracking
- Sustained stillness duration

---

## What We Need to Build

### 1. Fluid Simulation Engine

**Research Questions:**
- Technology choice: p5.js vs Processing vs custom WebGL/GLSL?
- Real-time performance: GPU shaders vs CPU?
- Python ↔ JavaScript integration: OSC? WebSocket? File-based?

**Core Features:**
- Navier-Stokes fluid dynamics
- Body silhouette as obstacle in fluid
- Responsive to external parameters

### 2. Visual Mappings

```
BREATHING → Fluid Pulse & Brightness
  -1 (exhale) = contract, darken
   0 (neutral) = baseline
  +1 (inhale)  = expand, brighten

JITTER → Fluid Turbulence
   0 (still)    = laminar flow, smooth
   0.5 (moving) = ripples, waves
   1 (restless) = chaos, high velocity

STILLNESS DURATION → Visual Rewards
   0-5s   = Purple/Indigo (anxious)
   5-15s  = Cyan/Teal (calm)
   15-30s = Blue deepening (focused)
   30s+   = GOLD (transcendent state)
```

### 3. Color Palette System

**State Colors:**
- **Anxious/Moving:** Dark Purple (#4A148C), Deep Indigo (#1A237E)
- **Calm/Breathing:** Cyan (#00BCD4), Teal (#009688), Soft White highlights
- **Deep Meditation:** Gold (#FFD700), Amber (#FFC107), Warm White

**Transitions:**
- Smooth gradient interpolation
- Duration-based (not instant)
- Persist through brief movements

### 4. Viscosity Control

**Implementation:**
- Low jitter (<0.2) → Increase viscosity (honey-like)
- High jitter (>0.6) → Decrease viscosity (water-like)
- Creates satisfying "reward" for stillness

---

## Technical Architecture Options

### Option A: p5.js (Recommended)
**Pros:** Easy, web-based, good community support
**Cons:** Performance may be limited for complex fluid dynamics
**Integration:** Python writes to JSON file → p5.js reads via fetch()

### Option B: Processing (Python Mode)
**Pros:** Native Python, direct integration
**Cons:** Slower, limited shader support
**Integration:** Direct API calls

### Option C: Custom GLSL Shaders
**Pros:** Maximum performance, full control
**Cons:** Complex, requires OpenGL knowledge
**Integration:** PyOpenGL or Pyglet

### Option D: TouchDesigner
**Pros:** Built for projection mapping, GPU-accelerated
**Cons:** Proprietary, steeper learning curve
**Integration:** OSC protocol from Python

---

## Implementation Plan

### Phase 1: Basic Fluid (2-3 hours)
- [ ] Choose technology stack
- [ ] Setup canvas (fullscreen)
- [ ] Implement basic Navier-Stokes solver
- [ ] Test with static parameters

### Phase 2: Input Integration (1-2 hours)
- [ ] Setup data bridge (Python → Visualization)
- [ ] Connect breathing signal to pulse
- [ ] Connect jitter score to turbulence
- [ ] Real-time testing

### Phase 3: Visual Polish (2-3 hours)
- [ ] Implement color palette system
- [ ] Add viscosity control
- [ ] Body silhouette integration
- [ ] Glow/bloom effects

### Phase 4: Performance Optimization
- [ ] Target 30+ FPS
- [ ] Test on target hardware
- [ ] Optimize for ceiling projection

---

## Data Bridge Examples

### JSON File (Simple)
```python
# Python writes:
{
  "breathing": 0.75,
  "jitter": 0.15,
  "stillness_duration": 23.5,
  "timestamp": 1234567890
}

# p5.js reads every frame
```

### WebSocket (Real-time)
```python
# Python: websocket.send(json.dumps(data))
# JavaScript: ws.onmessage = (msg) => { ... }
```

### OSC (Professional)
```python
# Python: osc_client.send_message("/breathing", 0.75)
# p5.js/TouchDesigner: receives at /breathing
```

---

## Reference Resources

**Fluid Simulation:**
- Navier-Stokes implementation tutorials
- GPU particle systems
- Real-time fluid dynamics papers

**Visual Inspiration:**
- Bioluminescent plankton
- Ink in water
- Aurora borealis effects

---

**Estimated Total Time:** 6-10 hours for complete fluid system

**Start Point:** Research + prototype basic fluid with hardcoded values
