"""Checks the packaged runtime: streaming in any chunk size gives the same audio as one call, lengths and timing
line up, silence stays silent, and it runs faster than real time.  Run: python tests/test_stream.py [recording.wav]"""
import sys, time
import numpy as np
from nearfield import SR, Nearfield, clean, latency_samples

if len(sys.argv) > 1:
    import soundfile as sf, soxr
    x, sr = sf.read(sys.argv[1], dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
    x = soxr.resample(x, sr, SR).astype(np.float32)
else:
    rng = np.random.default_rng(0); t = np.arange(6 * SR) / SR
    x = (0.1 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)

for model, post in (("fast", "default"), ("fast", "none"), ("best", "default"), ("best", "none")):
    nf = Nearfield(post=post, model=model); LATENCY_SAMPLES = latency_samples(model)
    assert nf.latency_samples == LATENCY_SAMPLES == {"fast": 768, "best": 1280}[model]
    t0 = time.time(); a = nf.process(x); dt = time.time() - t0; y = np.concatenate([a, nf.flush()])
    assert len(y) == len(x) and np.isfinite(y).all(), (len(y), len(x))
    assert len(x) - len(a) <= LATENCY_SAMPLES + 128, len(x) - len(a)
    print(f"model={model} post={post}: one call, {len(y)} samples, {dt / (len(x) / SR):.2f} s of compute per second of audio; {len(x) - len(a)} samples still inside before flush")
    assert dt / (len(x) / SR) < 1.0
    rng = np.random.default_rng(1); nf.reset(); outs, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 700)); outs.append(nf.process(x[i:i + n])); i += n
    y2 = np.concatenate(outs + [nf.flush()])
    assert len(y2) == len(x)
    print("  random chunk sizes match one call:", float(np.abs(y2 - y).max()))
    assert np.abs(y2 - y).max() < 2e-4

nf = Nearfield(post="none"); z = np.concatenate([nf.process(np.zeros(SR, np.float32)), nf.flush()])
assert np.abs(z).max() < 1e-4; print("silence in, silence out:", float(np.abs(z).max()))

import soxr
for sr in (8000, 16000, 48000):
    xs = soxr.resample(x, SR, sr) if sr != SR else x; ys = clean(xs, sr)
    assert len(ys) == len(xs) and np.isfinite(ys).all()
    yb = clean(xs, sr, model="best"); assert len(yb) == len(xs) and np.isfinite(yb).all()
    print(f"clean() at {sr} Hz: ok, {len(ys)} samples")
print("ALL OK")
