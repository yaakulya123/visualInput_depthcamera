#!/usr/bin/env python3
"""
Multi-Layer Sound Engine for Liquid Stillness

Adapted from the Depthcamera_testing "Harmonic Stack" audio system.
Plays multiple WAV files simultaneously with smooth volume transitions
driven by a chaos/jitter score (0.0 - 1.0).

Architecture:
- Base layer always playing (theta drone)
- 4 additional layers fade in/out based on jitter/chaos score
- Volume slewing for smooth 1-second transitions
- Master volume boost at high chaos (+0-20%)
- Soft tanh clipping prevents distortion

Original: /Users/yaakulyasabbani/Documents/GitHub/Depthcamera_testing/demos/audio_engine_multilayer.py
"""

import numpy as np
import threading
import os
from typing import Dict, List, Optional

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


# Default sound files directory (from reference repo)
DEFAULT_SOUND_DIR = "/Users/yaakulyasabbani/Documents/GitHub/Depthcamera_testing/soundfiles_capstone"

# Layer configuration: (filename, chaos_threshold)
DEFAULT_LAYERS = [
    ("ES_Theta Waves 144 hz - Syntropy.wav", 0.0),              # Base - always playing
    ("ES_Koan I (Theta 5 Hz) - Syntropy.wav", 0.2),             # Layer 1 - theta
    ("ES_Trillion (Sub Theta) STEMS BASS - Autonomic Sensations.wav", 0.4),  # Layer 2 - bass
    ("ES_Halcyon Daydream (Theta Drone L108Hz R113Hz) STEMS MELODY - Ookean.wav", 0.6),  # Layer 3 - melody
    ("ES_Halcyon Daydream (Theta Drone L108Hz R113Hz) - Ookean.wav", 0.8),   # Layer 4 - full
]


class MultiLayerSoundEngine:
    """
    Real-time multi-layer audio playback driven by a chaos/jitter score.

    Usage:
        engine = MultiLayerSoundEngine()
        engine.start()
        # In your loop:
        engine.update(jitter_score)  # 0.0 = still (base only), 1.0 = all layers
        # When done:
        engine.stop()
    """

    def __init__(
        self,
        sound_dir: str = DEFAULT_SOUND_DIR,
        layers: Optional[List[tuple]] = None,
        sample_rate: int = 48000,
        block_size: int = 2048,
    ):
        if not SD_AVAILABLE:
            raise ImportError("sounddevice required: pip install sounddevice>=0.4.6")
        if not SF_AVAILABLE:
            raise ImportError("soundfile required: pip install soundfile>=0.12.1")

        self.sound_dir = sound_dir
        self.layer_config = layers or DEFAULT_LAYERS
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Audio buffers
        self.audio_data: List[np.ndarray] = []
        self.layer_positions: List[int] = []
        self.layer_names: List[str] = []
        self.chaos_thresholds: List[float] = []

        # Volume control
        num_layers = len(self.layer_config)
        self.current_volumes = np.zeros(num_layers)
        self.target_volumes = np.zeros(num_layers)

        # Slew rate for smooth 1-second fades
        time_per_block = block_size / sample_rate
        self.slew_rate = time_per_block / 1.0  # 1 second fade

        # Master volume
        self.master_volume = 1.0
        self.chaos_score = 0.0

        # Thread control
        self.stream: Optional[sd.OutputStream] = None
        self.is_running = False
        self.lock = threading.Lock()

        # Load audio files
        self._load_audio_files()

        # Base layer always active
        self.target_volumes[0] = 1.0
        self.current_volumes[0] = 1.0

    def _load_audio_files(self):
        """Load all WAV files into memory."""
        print("\n[SoundEngine] Loading audio files...")

        for filename, threshold in self.layer_config:
            filepath = os.path.join(self.sound_dir, filename)

            if not os.path.exists(filepath):
                print(f"  [!] Not found: {filename}")
                # Silent fallback
                self.audio_data.append(np.zeros((self.sample_rate * 10, 2), dtype='float32'))
            else:
                try:
                    data, sr = sf.read(filepath, dtype='float32')

                    # Mono to stereo
                    if data.ndim == 1:
                        data = np.column_stack([data, data])

                    # Resample if needed
                    if sr != self.sample_rate:
                        from scipy import signal
                        num_samples = int(len(data) * self.sample_rate / sr)
                        resampled = np.zeros((num_samples, 2), dtype='float32')
                        for ch in range(2):
                            resampled[:, ch] = signal.resample(data[:, ch], num_samples)
                        data = resampled

                    self.audio_data.append(data)
                    duration = len(data) / self.sample_rate
                    print(f"  Loaded: {filename[:50]}... ({duration:.0f}s, threshold={threshold})")

                except Exception as e:
                    print(f"  [!] Error loading {filename}: {e}")
                    self.audio_data.append(np.zeros((self.sample_rate * 10, 2), dtype='float32'))

            self.layer_positions.append(0)
            self.layer_names.append(filename)
            self.chaos_thresholds.append(threshold)

        print(f"[SoundEngine] {len(self.audio_data)} layers loaded\n")

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Audio callback (runs in separate thread). Mixes active layers."""
        if status:
            print(f"[SoundEngine] callback status: {status}")

        with self.lock:
            outdata.fill(0)

            for i in range(len(self.audio_data)):
                # Volume slewing
                if self.current_volumes[i] < self.target_volumes[i]:
                    self.current_volumes[i] = min(
                        self.target_volumes[i],
                        self.current_volumes[i] + self.slew_rate
                    )
                elif self.current_volumes[i] > self.target_volumes[i]:
                    self.current_volumes[i] = max(
                        self.target_volumes[i],
                        self.current_volumes[i] - self.slew_rate
                    )

                if self.current_volumes[i] < 0.001:
                    continue

                # Read audio chunk
                layer_data = self.audio_data[i]
                layer_len = len(layer_data)
                pos = self.layer_positions[i]

                remaining = layer_len - pos
                to_read = min(frames, remaining)

                chunk = layer_data[pos:pos + to_read]
                outdata[:to_read] += chunk * self.current_volumes[i]

                # Loop
                self.layer_positions[i] = (pos + to_read) % layer_len

                if to_read < frames:
                    remaining_frames = frames - to_read
                    chunk2 = layer_data[:remaining_frames]
                    outdata[to_read:] += chunk2 * self.current_volumes[i]

            # Master volume boost (+0-20% at high chaos)
            master = 1.0 + (self.chaos_score * 0.2)
            outdata *= master

            # Soft clipping
            outdata[:] = np.tanh(outdata)

    def start(self) -> bool:
        """Start the audio engine."""
        if self.is_running:
            return True

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=2,
                callback=self._audio_callback,
                dtype='float32',
            )
            self.stream.start()
            self.is_running = True
            print("[SoundEngine] Started")
            return True
        except Exception as e:
            print(f"[SoundEngine] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the audio engine."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        print("[SoundEngine] Stopped")

    def update(self, chaos_score: float):
        """Update layer volumes based on chaos/jitter score (0.0 - 1.0)."""
        with self.lock:
            self.chaos_score = np.clip(chaos_score, 0.0, 1.0)

            for i in range(len(self.layer_config)):
                threshold = self.chaos_thresholds[i]
                if self.chaos_score >= threshold:
                    self.target_volumes[i] = 1.0
                else:
                    self.target_volumes[i] = 0.0

    def get_layer_info(self) -> List[Dict]:
        """Get current state of all layers for visualization."""
        with self.lock:
            info = []
            for i, name in enumerate(self.layer_names):
                short = name.replace("ES_", "").replace(" - Syntropy", "").replace(" - Ookean", "")
                if len(short) > 40:
                    short = short[:37] + "..."
                info.append({
                    'index': i,
                    'name': short,
                    'threshold': self.chaos_thresholds[i],
                    'volume': float(self.current_volumes[i]),
                    'target': float(self.target_volumes[i]),
                    'active': self.current_volumes[i] > 0.01,
                })
            return info

    def get_active_count(self) -> int:
        """Number of currently active layers."""
        with self.lock:
            return int(np.sum(self.current_volumes > 0.01))

    def reset(self):
        """Reset all layers to base only."""
        with self.lock:
            self.chaos_score = 0.0
            self.target_volumes[:] = 0.0
            self.target_volumes[0] = 1.0
            # Immediate reset of positions
            for i in range(len(self.layer_positions)):
                self.layer_positions[i] = 0
