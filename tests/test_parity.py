"""Checks the packaged runtime against the reference output of the research pipeline, and streaming consistency.
Run: NEARFIELD_MODEL=path/to/nearfield-v1.onnx python tests/test_parity.py <x.bin> <ref_full.bin>"""
import sys, time
import numpy as np
from nearfield import Nearfield, LATENCY_SAMPLES, clean

x = np.fromfile(sys.argv[1], np.float32)
ref = np.fromfile(sys.argv[2], np.float32)

nf = Nearfield(strength="full")
t = time.time(); y = np.concatenate([nf.process(x), nf.flush()]); dt = time.time() - t
y = y[LATENCY_SAMPLES: LATENCY_SAMPLES + len(ref)]
err = np.sum((y - ref) ** 2) / np.sum(ref ** 2)
print(f"parity vs reference: relative error {err:.2e}  | {dt / (len(x) / 48000) * 1000 / 100:.2f} ms per 10 ms frame")
assert err < 1e-6

rng = np.random.default_rng(0); nf.reset(); outs, i = [], 0
while i < len(x):
    n = int(rng.integers(1, 2000)); outs.append(nf.process(x[i:i + n])); i += n
y2 = np.concatenate(outs + [nf.flush()])[LATENCY_SAMPLES: LATENCY_SAMPLES + len(ref)]
print("random chunk sizes match one-shot:", np.abs(y2 - y).max())
assert np.abs(y2 - y).max() < 1e-5

import soxr
for sr in (8000, 16000, 48000):
    xs = soxr.resample(x, 48000, sr); ys = clean(xs, sr)
    assert len(ys) == len(xs) and np.isfinite(ys).all()
    print(f"clean() at {sr} Hz: ok, {len(ys)} samples")
print("ALL OK")
