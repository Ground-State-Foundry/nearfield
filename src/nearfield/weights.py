"""Where the model weights are: one 2.5 MB ONNX file that ships inside the package."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_FILE = "nearfield-v2.1.onnx"
MODEL_SHA256 = "0e584ce9555fa645bb1cd3d8340cd7d0e843d4b5f2f283fd614aac2a66e08142"


def model_path() -> str:
    """Path to the weights. Set NEARFIELD_MODEL to use a different file."""
    return os.environ.get("NEARFIELD_MODEL") or str(Path(__file__).with_name(MODEL_FILE))
