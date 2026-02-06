#!/usr/bin/env python3
"""
Analytics Dashboard - OpenCV-based multi-person visualization.

Two windows:
1. Main View (camera feed): color-coded bboxes, skeleton overlay, mini-HUDs
2. Analytics Window (1280x720): waveforms, jitter bars, timeline, leaderboard
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

from ..tracking.person_state import PersonState, MultiPersonState, PERSON_COLORS
from ..tracking.person_tracker import SKELETON_CONNECTIONS
from .session_analytics import SessionAnalytics


class AnalyticsDashboard:
    """
    Renders the main camera overlay and a separate analytics window.
    """

    ANALYTICS_WIDTH = 1280
    ANALYTICS_HEIGHT = 720

    def __init__(self, analytics: SessionAnalytics):
        self.analytics = analytics
        self._show_analytics = False

        # Breathing waveform buffers per person (for main view mini-HUD)
        self._breathing_history: Dict[int, List[float]] = {}
        self._max_waveform_len = 100

    @property
    def show_analytics(self) -> bool:
        return self._show_analytics

    @show_analytics.setter
    def show_analytics(self, value: bool):
        self._show_analytics = value
        if not value:
            cv2.destroyWindow("Analytics")

    def toggle_analytics(self):
        self.show_analytics = not self.show_analytics

    # ------------------------------------------------------------------
    #  Main camera view overlay
    # ------------------------------------------------------------------

    def draw_main_view(
        self,
        frame: np.ndarray,
        multi_state: MultiPersonState,
        raw_info: dict,
        fps: float,
    ) -> np.ndarray:
        """
        Draw bounding boxes, skeletons, and mini-HUDs on the camera frame.

        Returns annotated frame (modified in-place).
        """
        overlay = frame.copy()

        # Draw each person
        for i, track_id in enumerate(raw_info.get('track_ids', [])):
            person = multi_state.persons.get(track_id)
            if person is None:
                continue

            color = person.color
            bbox = raw_info['boxes'][i]
            kpts = raw_info['keypoints'][i]
            kpt_confs = raw_info['kpt_confidences'][i]

            # Bounding box
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Skeleton
            if kpts is not None and kpt_confs is not None:
                self._draw_skeleton(overlay, kpts, kpt_confs, color)

            # Mini HUD above bbox
            self._draw_person_hud(overlay, person, bbox, color)

            # Track breathing history
            if track_id not in self._breathing_history:
                self._breathing_history[track_id] = []
            self._breathing_history[track_id].append(person.breathing_signal)
            if len(self._breathing_history[track_id]) > self._max_waveform_len:
                self._breathing_history[track_id] = self._breathing_history[track_id][-self._max_waveform_len:]

        # Clean up stale breathing histories
        active_ids = set(raw_info.get('track_ids', []))
        stale = [k for k in self._breathing_history if k not in active_ids]
        for k in stale:
            del self._breathing_history[k]

        # Group stats bar at bottom
        self._draw_group_bar(overlay, multi_state, fps)

        return overlay

    def _draw_skeleton(self, frame: np.ndarray, kpts: np.ndarray,
                       confs: np.ndarray, color: Tuple[int, int, int]):
        """Draw COCO skeleton on frame."""
        for idx in range(17):
            if confs[idx] > 0.3:
                x, y = int(kpts[idx][0]), int(kpts[idx][1])
                cv2.circle(frame, (x, y), 3, color, -1)

        for i, j in SKELETON_CONNECTIONS:
            if confs[i] > 0.3 and confs[j] > 0.3:
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)

    def _draw_person_hud(self, frame: np.ndarray, person: PersonState,
                         bbox: np.ndarray, color: Tuple[int, int, int]):
        """Draw mini-HUD above bounding box for a person."""
        x1, y1 = int(bbox[0]), int(bbox[1])
        hud_y = max(y1 - 60, 10)

        # Background
        cv2.rectangle(frame, (x1, hud_y), (x1 + 180, hud_y + 55),
                       (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, hud_y), (x1 + 180, hud_y + 55),
                       color, 1)

        # Person ID
        cv2.putText(frame, f"Person {person.person_id}",
                     (x1 + 5, hud_y + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Breathing phase indicator
        phase = "---"
        if person.breathing_state:
            phase = person.breathing_state.phase
        phase_color = (100, 255, 100) if phase == "inhale" else (100, 100, 255) if phase == "exhale" else (180, 180, 180)
        cv2.putText(frame, f"Breath: {phase}",
                     (x1 + 5, hud_y + 32),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)

        # Jitter bar
        jitter = person.jitter_score
        bar_x = x1 + 5
        bar_y = hud_y + 40
        bar_w = 170
        bar_h = 10

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                       (50, 50, 50), -1)

        # Filled portion
        fill_w = int(bar_w * min(jitter, 1.0))
        if jitter < 0.2:
            bar_color = (100, 255, 100)   # Green = calm
        elif jitter < 0.5:
            bar_color = (50, 200, 255)    # Yellow = moderate
        else:
            bar_color = (50, 50, 255)     # Red = restless
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                       bar_color, -1)

    def _draw_group_bar(self, frame: np.ndarray, multi_state: MultiPersonState,
                        fps: float):
        """Draw group statistics bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        bar_h = 35
        bar_y = h - bar_h

        # Semi-transparent background
        overlay_roi = frame[bar_y:h, 0:w]
        dark = np.zeros_like(overlay_roi)
        cv2.addWeighted(overlay_roi, 0.4, dark, 0.6, 0, overlay_roi)

        # Text
        n = multi_state.person_count
        avg_j = multi_state.avg_jitter
        avg_b = multi_state.avg_bpm

        info = f"People: {n}  |  Avg Jitter: {avg_j:.2f}  |  Avg BPM: {avg_b:.1f}  |  FPS: {fps:.0f}"
        cv2.putText(frame, info, (10, h - 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Analytics toggle hint
        hint = "[A] Analytics"
        cv2.putText(frame, hint, (w - 150, h - 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ------------------------------------------------------------------
    #  Analytics window
    # ------------------------------------------------------------------

    def draw_analytics_window(self, multi_state: MultiPersonState):
        """Draw the separate analytics window if enabled."""
        if not self._show_analytics:
            return

        canvas = np.zeros((self.ANALYTICS_HEIGHT, self.ANALYTICS_WIDTH, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 25)  # Dark background

        # Title
        cv2.putText(canvas, "LIQUID STILLNESS - Analytics",
                     (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Layout: 4 panels
        # Top-left: Breathing waveforms (640x280)
        # Top-right: Jitter bars (640x280)
        # Bottom-left: Session timeline (640x380)
        # Bottom-right: Leaderboard (640x380)

        self._draw_breathing_waveforms(canvas, multi_state, 10, 55, 620, 270)
        self._draw_jitter_bars(canvas, multi_state, 650, 55, 620, 270)
        self._draw_session_timeline(canvas, 10, 340, 620, 360)
        self._draw_leaderboard(canvas, multi_state, 650, 340, 620, 360)

        cv2.imshow("Analytics", canvas)

    def _draw_breathing_waveforms(self, canvas: np.ndarray,
                                   multi_state: MultiPersonState,
                                   x: int, y: int, w: int, h: int):
        """Draw overlaid breathing waveforms, color-coded per person."""
        # Panel background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 65), 1)
        cv2.putText(canvas, "Breathing Waveforms", (x + 10, y + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Center line
        center_y = y + h // 2
        cv2.line(canvas, (x + 10, center_y), (x + w - 10, center_y),
                  (50, 50, 55), 1)

        # Draw each person's waveform
        plot_x_start = x + 10
        plot_x_end = x + w - 10
        plot_w = plot_x_end - plot_x_start
        plot_y_top = y + 35
        plot_y_bot = y + h - 10
        plot_h = (plot_y_bot - plot_y_top) // 2

        for pid, history in self._breathing_history.items():
            person = multi_state.persons.get(pid)
            if person is None or len(history) < 2:
                continue

            color = person.color
            n = len(history)
            points = []
            for i, val in enumerate(history):
                px = plot_x_start + int(i * plot_w / max(n - 1, 1))
                py = center_y - int(val * plot_h)
                py = max(plot_y_top, min(plot_y_bot, py))
                points.append((px, py))

            for j in range(1, len(points)):
                cv2.line(canvas, points[j - 1], points[j], color, 1)

            # Label at end
            if points:
                lx, ly = points[-1]
                cv2.putText(canvas, f"P{pid}", (lx + 3, ly - 3),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_jitter_bars(self, canvas: np.ndarray,
                           multi_state: MultiPersonState,
                           x: int, y: int, w: int, h: int):
        """Draw horizontal jitter bars per person."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 65), 1)
        cv2.putText(canvas, "Jitter Scores", (x + 10, y + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        persons = list(multi_state.persons.values())
        if not persons:
            cv2.putText(canvas, "No people detected", (x + 10, y + h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return

        bar_area_top = y + 35
        bar_area_h = h - 45
        bar_h = min(30, bar_area_h // max(len(persons), 1) - 5)
        bar_max_w = w - 120

        for i, person in enumerate(persons):
            by = bar_area_top + i * (bar_h + 5)
            if by + bar_h > y + h - 5:
                break

            # Label
            cv2.putText(canvas, f"P{person.person_id}",
                         (x + 10, by + bar_h - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, person.color, 1)

            # Background bar
            bx = x + 50
            cv2.rectangle(canvas, (bx, by), (bx + bar_max_w, by + bar_h),
                           (50, 50, 55), -1)

            # Filled bar
            jitter = person.jitter_score
            fill_w = int(bar_max_w * min(jitter, 1.0))
            if jitter < 0.2:
                bar_color = (100, 200, 100)
            elif jitter < 0.5:
                bar_color = (50, 180, 230)
            else:
                bar_color = (50, 80, 230)

            if fill_w > 0:
                cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h),
                               bar_color, -1)

            # Value
            cv2.putText(canvas, f"{jitter:.2f}",
                         (bx + bar_max_w + 5, by + bar_h - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    def _draw_session_timeline(self, canvas: np.ndarray,
                                x: int, y: int, w: int, h: int):
        """Draw rolling group calm level timeline (last 60s)."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 65), 1)
        cv2.putText(canvas, "Group Calm Timeline (60s)", (x + 10, y + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        history = self.analytics.get_group_jitter_history()
        if len(history) < 2:
            cv2.putText(canvas, "Collecting data...", (x + 10, y + h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return

        plot_x_start = x + 10
        plot_x_end = x + w - 10
        plot_w = plot_x_end - plot_x_start
        plot_y_top = y + 35
        plot_y_bot = y + h - 30

        # Calm level = 1 - jitter (inverted so higher = calmer)
        n = len(history)
        points = []
        for i, jitter in enumerate(history):
            calm = 1.0 - min(jitter, 1.0)
            px = plot_x_start + int(i * plot_w / max(n - 1, 1))
            py = plot_y_bot - int(calm * (plot_y_bot - plot_y_top))
            points.append((px, py))

        # Fill area under curve
        if len(points) >= 2:
            fill_pts = [(plot_x_start, plot_y_bot)] + points + [(plot_x_end, plot_y_bot)]
            fill_array = np.array(fill_pts, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [fill_array], (80, 50, 30))
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw line
        for j in range(1, len(points)):
            cv2.line(canvas, points[j - 1], points[j], (200, 150, 80), 2)

        # Axis labels
        cv2.putText(canvas, "Calm", (x + 10, plot_y_top + 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
        cv2.putText(canvas, "Restless", (x + 10, plot_y_bot + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    def _draw_leaderboard(self, canvas: np.ndarray,
                           multi_state: MultiPersonState,
                           x: int, y: int, w: int, h: int):
        """Draw leaderboard ranked by calmness."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 65), 1)
        cv2.putText(canvas, "Calmness Leaderboard", (x + 10, y + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        sorted_persons = multi_state.get_sorted_by_calmness()
        if not sorted_persons:
            cv2.putText(canvas, "No people detected", (x + 10, y + h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return

        row_h = 50
        start_y = y + 40

        medals = ["1st", "2nd", "3rd"]

        for i, person in enumerate(sorted_persons):
            ry = start_y + i * row_h
            if ry + row_h > y + h - 10:
                break

            # Rank
            rank_text = medals[i] if i < 3 else f"{i + 1}th"
            rank_color = (50, 215, 255) if i == 0 else (200, 200, 200) if i == 1 else (80, 130, 200)
            if i >= 3:
                rank_color = (150, 150, 150)
            cv2.putText(canvas, rank_text, (x + 15, ry + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, rank_color, 2 if i == 0 else 1)

            # Person info
            cv2.putText(canvas, f"Person {person.person_id}",
                         (x + 70, ry + 15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, person.color, 1)

            # Stats
            jitter_text = f"Jitter: {person.jitter_score:.2f}"
            bpm_text = f"BPM: {person.bpm:.0f}" if person.bpm > 0 else "BPM: --"
            still_text = f"Still: {person.stillness_duration:.0f}s"

            cv2.putText(canvas, jitter_text, (x + 70, ry + 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(canvas, bpm_text, (x + 220, ry + 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(canvas, still_text, (x + 350, ry + 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            # Stillness quality
            quality = "---"
            if person.stillness_state:
                sd = person.stillness_detector
                if sd:
                    quality = sd.get_stillness_quality()
            if quality == "transcendent":
                cv2.putText(canvas, "GOLDEN", (x + 480, ry + 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 215, 255), 2)

            # Separator
            if i < len(sorted_persons) - 1:
                cv2.line(canvas, (x + 10, ry + row_h - 5),
                          (x + w - 10, ry + row_h - 5), (45, 45, 50), 1)

    def cleanup(self):
        """Destroy analytics window."""
        try:
            cv2.destroyWindow("Analytics")
        except Exception:
            pass
