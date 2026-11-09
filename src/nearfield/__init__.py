"""Nearfield: real-time caller voice isolation. Keeps the voice near the mic; removes noise and background voices."""
from .core import LATENCY_SAMPLES, SR, STRENGTHS, Nearfield, Strength, clean

__version__ = "1.0.0"
__all__ = ["Nearfield", "Strength", "STRENGTHS", "clean", "SR", "LATENCY_SAMPLES", "__version__"]
