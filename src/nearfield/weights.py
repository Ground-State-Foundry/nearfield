"""Where the model weights are: two 2.5 MB ONNX files that ship inside the package."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_FILES = {"fast": "nearfield-v2.3.onnx", "best": "nearfield-v2.3-best.onnx"}
MODEL_SHA256 = {
    "fast": "d3634c3c12fc9c320f10d1665cf50910565852222d2e2c601e3acf9c04250d7c",
    "best": "ce966bc8c050c7a6cbbdab4b2fc831b4e19df259ea8a6bc1d2c0cf90e3c2053f",
}
MODEL_FILE = MODEL_FILES["fast"]


def model_path(model: str = "fast") -> str:
    """Path to the weights of a model. Set NEARFIELD_MODEL to use a different file."""
    return os.environ.get("NEARFIELD_MODEL") or str(Path(__file__).with_name(MODEL_FILES[model]))
