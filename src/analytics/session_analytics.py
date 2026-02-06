#!/usr/bin/env python3
"""
Session Analytics - Time-series recording and session summary for multi-person tracking.

Records per-person metrics at configurable intervals (default 10Hz) and provides
end-of-session summaries with CSV export.
"""

import csv
import os
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..tracking.person_state import PersonState, MultiPersonState


@dataclass
class PersonTimeSeries:
    """Time-series data for a single person."""
    person_id: int
    timestamps: List[float] = field(default_factory=list)
    jitter_scores: List[float] = field(default_factory=list)
    breathing_signals: List[float] = field(default_factory=list)
    bpm_values: List[float] = field(default_factory=list)
    stillness_durations: List[float] = field(default_factory=list)
    motion_types: List[str] = field(default_factory=list)

    def append(self, person: PersonState, timestamp: float):
        self.timestamps.append(timestamp)
        self.jitter_scores.append(person.jitter_score)
        self.breathing_signals.append(person.breathing_signal)
        self.bpm_values.append(person.bpm)
        self.stillness_durations.append(person.stillness_duration)
        motion_type = "unknown"
        if person.stillness_state:
            motion_type = person.stillness_state.motion_type
        self.motion_types.append(motion_type)

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)

    @property
    def avg_jitter(self) -> float:
        return float(np.mean(self.jitter_scores)) if self.jitter_scores else 0.0

    @property
    def min_jitter(self) -> float:
        return float(np.min(self.jitter_scores)) if self.jitter_scores else 0.0

    @property
    def max_stillness(self) -> float:
        return float(np.max(self.stillness_durations)) if self.stillness_durations else 0.0

    @property
    def avg_bpm(self) -> float:
        valid = [b for b in self.bpm_values if b > 0]
        return float(np.mean(valid)) if valid else 0.0

    def get_rolling_jitter(self, window: int = 30) -> List[float]:
        """Get rolling average jitter (for plotting)."""
        if len(self.jitter_scores) < window:
            return self.jitter_scores[:]
        arr = np.array(self.jitter_scores)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid').tolist()


@dataclass
class SessionSummary:
    """End-of-session statistics."""
    session_duration: float
    total_people_seen: int
    max_simultaneous: int
    per_person: Dict[int, dict] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  SESSION SUMMARY",
            "=" * 50,
            f"  Duration: {self.session_duration:.1f}s",
            f"  Total people tracked: {self.total_people_seen}",
            f"  Max simultaneous: {self.max_simultaneous}",
            "-" * 50,
        ]
        for pid, stats in self.per_person.items():
            lines.append(f"  Person {pid}:")
            lines.append(f"    Avg jitter: {stats['avg_jitter']:.3f}")
            lines.append(f"    Best stillness: {stats['max_stillness']:.1f}s")
            lines.append(f"    Avg BPM: {stats['avg_bpm']:.1f}")
            lines.append(f"    Golden state: {'Yes' if stats['golden_state'] else 'No'}")
            lines.append(f"    Samples: {stats['samples']}")
        lines.append("=" * 50)
        return "\n".join(lines)


class SessionAnalytics:
    """
    Records multi-person metrics over time and generates session summaries.

    Samples at a configurable rate (default 10Hz) to save memory during
    long sessions.
    """

    def __init__(self, sample_interval: float = 0.1):
        """
        Args:
            sample_interval: Minimum seconds between recorded samples (0.1 = 10Hz).
        """
        self.sample_interval = sample_interval
        self.session_start = time.time()
        self._last_sample_time = 0.0
        self._person_series: Dict[int, PersonTimeSeries] = {}
        self._all_person_ids: set = set()
        self._max_simultaneous = 0

        # Rolling group stats (for dashboard timeline)
        self._group_jitter_history = deque(maxlen=600)  # 60s at 10Hz
        self._group_timestamps = deque(maxlen=600)

    def record(self, multi_state: MultiPersonState):
        """
        Record a multi-person state snapshot if enough time has elapsed.

        Args:
            multi_state: Current MultiPersonState from PersonTracker.
        """
        now = time.time()
        if (now - self._last_sample_time) < self.sample_interval:
            return

        self._last_sample_time = now
        self._max_simultaneous = max(self._max_simultaneous, multi_state.person_count)

        for pid, person in multi_state.persons.items():
            self._all_person_ids.add(pid)

            if pid not in self._person_series:
                self._person_series[pid] = PersonTimeSeries(person_id=pid)

            self._person_series[pid].append(person, now)

        # Record group stats
        if multi_state.person_count > 0:
            self._group_jitter_history.append(float(multi_state.avg_jitter))
            self._group_timestamps.append(now)

    def get_person_series(self, person_id: int) -> Optional[PersonTimeSeries]:
        return self._person_series.get(person_id)

    def get_all_series(self) -> Dict[int, PersonTimeSeries]:
        return self._person_series

    def get_group_jitter_history(self) -> List[float]:
        return list(self._group_jitter_history)

    def get_group_timestamps(self) -> List[float]:
        return list(self._group_timestamps)

    def get_summary(self) -> SessionSummary:
        """Generate end-of-session summary."""
        duration = time.time() - self.session_start

        per_person = {}
        for pid, series in self._person_series.items():
            per_person[pid] = {
                'avg_jitter': series.avg_jitter,
                'min_jitter': series.min_jitter,
                'max_stillness': series.max_stillness,
                'avg_bpm': series.avg_bpm,
                'golden_state': series.max_stillness >= 30.0,
                'samples': series.sample_count,
            }

        return SessionSummary(
            session_duration=duration,
            total_people_seen=len(self._all_person_ids),
            max_simultaneous=self._max_simultaneous,
            per_person=per_person,
        )

    def export_csv(self, filepath: str):
        """
        Export all per-person time-series data to CSV.

        Args:
            filepath: Output CSV file path.
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        rows = []
        for pid, series in self._person_series.items():
            for i in range(series.sample_count):
                rows.append({
                    'timestamp': series.timestamps[i],
                    'elapsed': series.timestamps[i] - self.session_start,
                    'person_id': pid,
                    'jitter_score': series.jitter_scores[i],
                    'breathing_signal': series.breathing_signals[i],
                    'bpm': series.bpm_values[i],
                    'stillness_duration': series.stillness_durations[i],
                    'motion_type': series.motion_types[i],
                })

        # Sort by timestamp then person_id
        rows.sort(key=lambda r: (r['timestamp'], r['person_id']))

        if not rows:
            print("[Analytics] No data to export.")
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"[Analytics] Exported {len(rows)} rows to {filepath}")

    def reset(self):
        """Reset all analytics data."""
        self.session_start = time.time()
        self._last_sample_time = 0.0
        self._person_series.clear()
        self._all_person_ids.clear()
        self._max_simultaneous = 0
        self._group_jitter_history.clear()
        self._group_timestamps.clear()
