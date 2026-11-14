"""Nearfield: real-time caller voice isolation. Keeps the voice near the mic; removes noise and background voices."""
from .core import LATENCY_SAMPLES, MODELS, POSTS, SR, Nearfield, Post, clean, latency_samples

__version__ = "2.2.0"
__all__ = ["Nearfield", "Post", "POSTS", "clean", "MODELS", "SR", "LATENCY_SAMPLES", "latency_samples", "__version__"]
