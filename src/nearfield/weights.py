"""Download and cache the model weights (one 11 MB ONNX file, fetched once)."""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_VERSION = "v1.0.0"
MODEL_FILE = "nearfield-v1.onnx"
MODEL_SHA256 = "2e57df28c7b2baf788ef6b7340640568999de16084e8eeafd6ff3ef3db3c1ea8"
MODEL_URL = f"https://github.com/ground-state-foundry/nearfield/releases/download/{MODEL_VERSION}/{MODEL_FILE}"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_path() -> str:
    """Path to the weights. Set NEARFIELD_MODEL to use a local file; NEARFIELD_HOME to change the cache folder."""
    local = os.environ.get("NEARFIELD_MODEL")
    if local:
        return local
    home = Path(os.environ.get("NEARFIELD_HOME", Path.home() / ".cache" / "nearfield"))
    dest = home / MODEL_VERSION / MODEL_FILE
    if dest.exists() and _sha256(dest) == MODEL_SHA256:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        with urllib.request.urlopen(os.environ.get("NEARFIELD_MODEL_URL", MODEL_URL), timeout=60) as r:
            for b in iter(lambda: r.read(1 << 20), b""):
                tmp.write(b)
    got = _sha256(Path(tmp.name))
    if got != MODEL_SHA256:
        os.unlink(tmp.name)
        raise RuntimeError(f"Nearfield weights failed the checksum (got {got[:12]}..., expected {MODEL_SHA256[:12]}...).")
    os.replace(tmp.name, dest)
    return str(dest)
