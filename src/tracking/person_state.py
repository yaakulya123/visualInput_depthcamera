#!/usr/bin/env python3
"""
Person State - Per-person tracking state and multi-person management.

Each tracked person gets their own BreathingDetector and StillnessDetector
instances, ensuring independent metrics and calibration.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..breathing.breath_detector import BreathingDetector, BreathingState, DetectionMode
from ..stillness.stillness_detector import StillnessDetector, StillnessState, create_stillness_detector


# Stable color palette for tracked persons (BGR format for OpenCV)
PERSON_COLORS = [
    (255, 100, 50),    # Blue-ish
    (50, 220, 100),    # Green-ish
    (50, 100, 255),    # Red-ish
    (255, 200, 50),    # Cyan-ish
    (200, 50, 255),    # Magenta-ish
    (50, 255, 255),    # Yellow-ish
    (255, 150, 150),   # Light blue
    (150, 255, 150),   # Light green
]


@dataclass
class PersonState:
    """State for a single tracked person."""
    person_id: int
    color: Tuple[int, int, int]
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2

    # Detector instances (created by PersonManager)
    breathing_detector: Optional[BreathingDetector] = field(default=None, repr=False)
    stillness_detector: Optional[StillnessDetector] = field(default=None, repr=False)

    # Latest states
    breathing_state: Optional[BreathingState] = field(default=None, repr=False)
    stillness_state: Optional[StillnessState] = field(default=None, repr=False)

    # Tracking metadata
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 0

    @property
    def tracking_duration(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def jitter_score(self) -> float:
        if self.stillness_state:
            return self.stillness_state.jitter_score
        return 0.0

    @property
    def breathing_signal(self) -> float:
        if self.breathing_state:
            return self.breathing_state.signal
        return 0.0

    @property
    def bpm(self) -> float:
        if self.breathing_state:
            return self.breathing_state.breath_rate
        return 0.0

    @property
    def stillness_duration(self) -> float:
        if self.stillness_state:
            return self.stillness_state.stillness_duration
        return 0.0


@dataclass
class MultiPersonState:
    """Aggregate state for all tracked persons."""
    persons: Dict[int, PersonState] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def avg_jitter(self) -> float:
        if not self.persons:
            return 0.0
        return np.mean([p.jitter_score for p in self.persons.values()])

    @property
    def avg_bpm(self) -> float:
        bpms = [p.bpm for p in self.persons.values() if p.bpm > 0]
        if not bpms:
            return 0.0
        return np.mean(bpms)

    @property
    def calmest_person(self) -> Optional[PersonState]:
        if not self.persons:
            return None
        return min(self.persons.values(), key=lambda p: p.jitter_score)

    @property
    def most_restless_person(self) -> Optional[PersonState]:
        if not self.persons:
            return None
        return max(self.persons.values(), key=lambda p: p.jitter_score)

    def get_sorted_by_calmness(self) -> List[PersonState]:
        return sorted(self.persons.values(), key=lambda p: p.jitter_score)


class PersonManager:
    """
    Creates and manages per-person detector instances.

    Assigns stable colors, creates/reuses detectors, and cleans up
    stale persons that haven't been seen recently.
    """

    def __init__(self, stale_timeout: float = 5.0):
        """
        Args:
            stale_timeout: Remove persons not seen for this many seconds.
        """
        self.stale_timeout = stale_timeout
        self._persons: Dict[int, PersonState] = {}
        self._color_index = 0

    def get_or_create(self, person_id: int) -> PersonState:
        """Get existing PersonState or create a new one with fresh detectors."""
        if person_id in self._persons:
            self._persons[person_id].last_seen = time.time()
            return self._persons[person_id]

        # Create new person with independent detectors
        color = PERSON_COLORS[self._color_index % len(PERSON_COLORS)]
        self._color_index += 1

        person = PersonState(
            person_id=person_id,
            color=color,
            breathing_detector=BreathingDetector(mode=DetectionMode.POSE),
            stillness_detector=create_stillness_detector("normal"),
        )

        self._persons[person_id] = person
        return person

    def cleanup_stale(self) -> List[int]:
        """Remove persons not seen for stale_timeout seconds. Returns removed IDs."""
        now = time.time()
        stale_ids = [
            pid for pid, person in self._persons.items()
            if (now - person.last_seen) > self.stale_timeout
        ]
        for pid in stale_ids:
            del self._persons[pid]
        return stale_ids

    def get_all(self) -> Dict[int, PersonState]:
        return self._persons

    def reset(self):
        """Clear all tracked persons."""
        self._persons.clear()
        self._color_index = 0

    @property
    def person_count(self) -> int:
        return len(self._persons)
