"""Nearfield: real-time caller voice isolation. Keeps the voice near the mic; removes noise and background voices."""
from .core import LATENCY_SAMPLES, POSTS, SR, Nearfield, Post, clean

__version__ = "2.0.0"
__all__ = ["Nearfield", "Post", "POSTS", "clean", "SR", "LATENCY_SAMPLES", "__version__"]
