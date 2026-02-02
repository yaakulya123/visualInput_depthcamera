#!/usr/bin/env python3
"""
One-Euro Filter Implementation

A simple but effective filter for noisy signals in real-time applications.
Provides adaptive smoothing that reduces jitter while maintaining responsiveness.

Based on: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
by Géry Casiez, Nicolas Roussel, and Daniel Vogel (CHI 2012)

Reference: https://gery.casiez.net/1euro/
Implementation inspired by: https://github.com/jaantollander/OneEuroFilter
"""

import math
import numpy as np
from typing import Optional, Union


def smoothing_factor(t_e: float, cutoff: float) -> float:
    """
    Calculate smoothing factor alpha.

    Args:
        t_e: Time elapsed since last sample (seconds)
        cutoff: Cutoff frequency (Hz)

    Returns:
        Smoothing factor alpha in [0, 1]
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / t_e)


def exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    """
    Apply exponential smoothing.

    Args:
        a: Smoothing factor alpha
        x: Current value
        x_prev: Previous smoothed value

    Returns:
        Smoothed value
    """
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """
    One-Euro Filter for single values.

    The One-Euro filter is designed to filter noisy signals in real-time.
    It uses an adaptive cutoff frequency based on the signal's rate of change.

    - When the signal is relatively stable (low velocity), it applies more
      smoothing to reduce jitter
    - When the signal changes rapidly (high velocity), it reduces smoothing
      to minimize lag

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing when still.
                    Default 1.0 Hz. Reduce to 0.5-0.1 for very smooth output.
        beta: Speed coefficient. Higher = more responsive to fast movements.
              Default 0.0. Increase to 0.5-1.0 for faster response.
        d_cutoff: Cutoff frequency for derivative calculation. Default 1.0 Hz.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0
    ):
        """
        Initialize the filter.

        Args:
            t0: Initial timestamp
            x0: Initial value
            min_cutoff: Minimum cutoff frequency when still
            beta: Speed coefficient for adaptive cutoff
            d_cutoff: Derivative cutoff frequency
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Initialize state
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def __call__(self, t: float, x: float) -> float:
        """
        Filter a new value.

        Args:
            t: Current timestamp
            x: Current raw value

        Returns:
            Filtered value
        """
        # Time delta
        t_e = t - self.t_prev

        if t_e <= 0:
            return self.x_prev

        # Estimate derivative
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter signal
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class OneEuroFilterMulti:
    """
    One-Euro Filter for multi-dimensional data (e.g., landmarks).

    Applies independent One-Euro filters to each dimension.
    """

    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0
    ):
        """
        Initialize multi-dimensional filter.

        Args:
            t0: Initial timestamp
            x0: Initial values as numpy array
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient
            d_cutoff: Derivative cutoff frequency
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Store shape for validation
        self.shape = x0.shape

        # Initialize state
        self.x_prev = x0.copy()
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Filter new values.

        Args:
            t: Current timestamp
            x: Current raw values as numpy array

        Returns:
            Filtered values as numpy array
        """
        # Validate shape
        if x.shape != self.shape:
            raise ValueError(f"Input shape {x.shape} doesn't match expected {self.shape}")

        # Time delta
        t_e = t - self.t_prev

        if t_e <= 0:
            return self.x_prev.copy()

        # Estimate derivative
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filter signal (element-wise)
        a = np.zeros_like(cutoff)
        for i in range(len(a.flat)):
            tau = 1.0 / (2.0 * math.pi * cutoff.flat[i])
            a.flat[i] = 1.0 / (1.0 + tau / t_e)

        x_hat = a * x + (1.0 - a) * self.x_prev

        # Update state
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t

        return x_hat


class LandmarkSmoother:
    """
    Specialized One-Euro filter for MediaPipe pose landmarks.

    Handles the (33, 3) landmark array structure (x, y, z for each landmark).
    Provides easy-to-use interface for pose smoothing.
    """

    def __init__(
        self,
        num_landmarks: int = 33,
        min_cutoff: float = 0.5,
        beta: float = 0.5,
        d_cutoff: float = 1.0
    ):
        """
        Initialize landmark smoother.

        Args:
            num_landmarks: Number of landmarks (33 for full pose)
            min_cutoff: Minimum cutoff (lower = smoother when still)
            beta: Speed coefficient (higher = more responsive)
            d_cutoff: Derivative cutoff
        """
        self.num_landmarks = num_landmarks
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # Will be initialized on first call
        self.filter: Optional[OneEuroFilterMulti] = None
        self.initialized = False

    def smooth(self, t: float, landmarks: np.ndarray) -> np.ndarray:
        """
        Smooth landmarks.

        Args:
            t: Current timestamp (seconds)
            landmarks: Raw landmarks array (num_landmarks, 3) or (num_landmarks, 4)

        Returns:
            Smoothed landmarks array
        """
        # Ensure 2D array
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 3)

        # Initialize filter on first call
        if not self.initialized:
            self.filter = OneEuroFilterMulti(
                t0=t,
                x0=landmarks,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff
            )
            self.initialized = True
            return landmarks.copy()

        return self.filter(t, landmarks)

    def reset(self):
        """Reset filter state."""
        self.filter = None
        self.initialized = False


# Preset configurations for different use cases
PRESETS = {
    # Very smooth, good for meditation/stillness detection
    "stillness": {"min_cutoff": 0.3, "beta": 0.1, "d_cutoff": 1.0},

    # Balanced smoothing with reasonable responsiveness
    "balanced": {"min_cutoff": 1.0, "beta": 0.5, "d_cutoff": 1.0},

    # Responsive, less smoothing, good for fast movements
    "responsive": {"min_cutoff": 1.5, "beta": 1.0, "d_cutoff": 1.0},

    # MediaPipe recommended starting point
    "mediapipe": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
}


def get_preset(name: str) -> dict:
    """Get filter parameters for a named preset."""
    return PRESETS.get(name, PRESETS["balanced"])
