"""Where the model weights are: one 2.5 MB ONNX file that ships inside the package."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_FILE = "nearfield-v2.onnx"
MODEL_SHA256 = "c442b6e87e0a84b0395be4dec208a508d8c876732c2845fb707e736762c01ea3"


def model_path() -> str:
    """Path to the weights. Set NEARFIELD_MODEL to use a different file."""
    return os.environ.get("NEARFIELD_MODEL") or str(Path(__file__).with_name(MODEL_FILE))
