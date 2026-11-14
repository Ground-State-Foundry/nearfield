"""Where the model weights are: two 2.5 MB ONNX files that ship inside the package."""
from __future__ import annotations

import os
from pathlib import Path

MODEL_FILES = {"fast": "nearfield-v2.1.onnx", "best": "nearfield-v2.2-best.onnx"}
MODEL_SHA256 = {
    "fast": "0e584ce9555fa645bb1cd3d8340cd7d0e843d4b5f2f283fd614aac2a66e08142",
    "best": "3a64ed4d3be788644b40dcf441152cbc2d66076f88220c66e837ae425cfbc0d4",
}
MODEL_FILE = MODEL_FILES["fast"]


def model_path(model: str = "fast") -> str:
    """Path to the weights of a model. Set NEARFIELD_MODEL to use a different file."""
    return os.environ.get("NEARFIELD_MODEL") or str(Path(__file__).with_name(MODEL_FILES[model]))
