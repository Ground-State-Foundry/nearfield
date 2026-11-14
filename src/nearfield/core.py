"""Nearfield v2 streaming runtime: keeps the voice close to the mic, removes noise and background voices.

One small recurrent network, trained from scratch for this job, reads the call as a stream of short-time spectra
(32 ms window, 8 ms hop, 16 kHz) and writes a complex mask for each one. It looks a little ahead before it commits
to a frame. There are two sets of weights, same size and same speed:

    "fast"  looks 16 ms ahead: 48 ms of delay. The default.
    "best"  looks 48 ms ahead: 80 ms of delay, and keeps more of the caller when other people are talking.

A light post-processing stage then steadies the volume and tidies what is left. Everything runs on one CPU thread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
HOP = 128
N_FFT = 512
N_BINS = 257
MODELS = {"fast": 2, "best": 6}                           # frames of look-ahead (8 ms each)
LOOKAHEAD_FRAMES = MODELS["fast"]
LATENCY_SAMPLES = N_FFT + LOOKAHEAD_FRAMES * HOP          # the default model: 768 samples at 16 kHz = 48 ms


def latency_samples(model: str = "fast") -> int:
    """Delay of a model in samples at 16 kHz: 768 (48 ms) for "fast", 1280 (80 ms) for "best"."""
    return N_FFT + MODELS[model] * HOP

_STATE_SHAPE = (4, 64, 128)


@dataclass(frozen=True)
class Post:
    """What happens to the separated caller before it is returned. Every step works on the model's own output:
    nothing from the original recording is mixed back in, and no step adds delay.

    level_db   volume levelling: speech is brought to about this loudness (dBFS) and peaks are limited.
               Quiet callers are raised by up to 18 dB, loud ones lowered by up to 12 dB. None = leave the level alone.
    blips      remove isolated leftovers: a frequency that is suddenly loud while its neighbours and its own
               recent past are quiet.
    pause_db   push the stretches between the caller's words down by up to this many dB, fading over about 80 ms.
               None = off.
    tilt_db    change above 5 kHz, easing in from 3 kHz (negative = softer top end). 0 = off.
    """
    level_db: float | None = -18.0
    blips: bool = True
    pause_db: float | None = 18.0
    tilt_db: float = -5.0


POSTS = {
    "default": Post(),
    "level": Post(blips=False, pause_db=None, tilt_db=0.0),
    "none": None,
}


class _PostState:
    def __init__(self, p: Post):
        self.p = p
        f = np.arange(N_BINS) * (SR / N_FFT)
        self.tilt = 10 ** (p.tilt_db * np.clip((f - 3000.0) / 2000.0, 0, 1) / 20)
        self.gp = np.zeros(N_BINS)
        self.top, self.gate, self.hold = -60.0, 1.0, 0
        self.lvl, self.gain = -26.0, 1.0
        self.up, self.dn = 1 - np.exp(-8 / 150), 1 - np.exp(-8 / 3000)
        self.rel = np.exp(-8 / 80)

    def spec(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Y: separated spectrum of one frame, X: the input spectrum of the same frame."""
        p = self.p
        s = self.tilt
        if p.blips:
            g = np.minimum(2.0, np.abs(Y) / (np.abs(X) + 1e-9))
            gl, gr = np.concatenate([g[:1], g[:-1]]), np.concatenate([g[1:], g[-1:]])
            pl, pr = np.concatenate([self.gp[:1], self.gp[:-1]]), np.concatenate([self.gp[1:], self.gp[-1:]])
            ref = np.maximum.reduce([pl, self.gp, pr, gl, gr]) * 1.5 + 0.03
            s = np.where(g > ref, s * ref / np.maximum(g, 1e-9), s)
            self.gp = g
        Y = Y * s
        if p.pause_db:
            e = float((Y.real ** 2 + Y.imag ** 2).sum())
            db = 10 * np.log10(e / (N_FFT * N_FFT * 0.375) * 2 + 1e-12)
            self.top = db if db > self.top else self.top - 0.004
            thr = max(-70.0, self.top - 32.0)
            want = 1.0
            if db >= thr:
                self.hold = 12
            elif self.hold > 0:
                self.hold -= 1
            else:
                want = 10 ** (max(-p.pause_db, db - thr) / 20)
            self.gate = want if want >= self.gate else self.gate * self.rel + want * (1 - self.rel)
            if self.gate < 1:
                Y = Y * self.gate
        return Y

    def time(self, y: np.ndarray) -> np.ndarray:
        """One hop of output samples, levelled in place."""
        p = self.p
        if p.level_db is None or len(y) == 0:
            return y
        db = 10 * np.log10(float(np.mean(y * y)) + 1e-12)
        if db > max(-55.0, self.lvl - 25.0):
            self.lvl += (self.up if db > self.lvl else self.dn) * (db - self.lvl)
        want = 10 ** (min(18.0, max(-12.0, p.level_db - self.lvl)) / 20)
        g1 = self.gain + (want - self.gain) * 0.08
        v = y * (self.gain + (g1 - self.gain) * np.arange(len(y)) / len(y))
        a = np.abs(v)
        v = np.where(a > 0.5, np.sign(v) * (0.5 + 0.45 * np.tanh((a - 0.5) / 0.45)), v)
        self.gain = g1
        return v


class Nearfield:
    """Streaming cleaner at 16 kHz.

    >>> nf = Nearfield()
    >>> out = nf.process(chunk_16k_float32)    # any chunk size; returns the samples that are ready

    Output sample i is the cleaned version of input sample i. Samples come out ``latency_samples`` after they
    went in (48 ms for "fast", 80 ms for "best"), so early calls return fewer samples than they were given;
    flush() returns the rest.

    model: "fast" (48 ms of delay, the default) or "best" (80 ms, keeps more of the caller over other voices).
           The NEARFIELD_QUALITY environment variable sets it when the argument is left out.
    post:  "default", "level", "none", or a custom ``Post``.
    model_path: a different weights file for the chosen model; by default the one inside the package is used.
    """

    def __init__(self, model_path: str | None = None, post: str | Post | None = "default", threads: int = 1,
                 model: str | None = None):
        import os

        import onnxruntime as ort
        from .weights import model_path as _default_path

        model = model or os.environ.get("NEARFIELD_QUALITY") or "fast"
        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
        self.model = model
        self._look = MODELS[model]
        self.latency_samples = N_FFT + self._look * HOP

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(model_path or _default_path(model), opts, providers=["CPUExecutionProvider"])
        self._post_cfg = POSTS[post] if isinstance(post, str) else post
        self._win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, np.float32)
        self._buf = np.zeros(N_FFT // 2, np.float32)
        self._hist: list[np.ndarray] = []
        self._ola = np.zeros(N_FFT)
        self._skip = N_FFT // 2
        self._frames = 0
        self._n_in = 0
        self._n_out = 0
        self._post = _PostState(self._post_cfg) if self._post_cfg is not None else None

    def set_post(self, post: str | Post | None) -> None:
        self._post_cfg = POSTS[post] if isinstance(post, str) else post
        self._post = _PostState(self._post_cfg) if self._post_cfg is not None else None

    def _run(self) -> np.ndarray:
        n = (len(self._buf) - N_FFT) // HOP + 1 if len(self._buf) >= N_FFT else 0
        if n == 0:
            return np.zeros(0, np.float32)
        fr = np.stack([self._buf[t * HOP:t * HOP + N_FFT] for t in range(n)]) * self._win
        self._buf = self._buf[n * HOP:]
        X = np.fft.rfft(fr, axis=1).astype(np.complex64)
        Xd = np.zeros_like(X)
        for t in range(n):                                  # the model finishes the frame from `look` frames back
            self._hist.append(X[t])
            if len(self._hist) > self._look:
                Xd[t] = self._hist.pop(0)
        ri = lambda Z: np.stack([Z.real, Z.imag], -1).astype(np.float32)
        y, self._state = self._sess.run(None, {"x": ri(X), "xd": ri(Xd), "state": self._state})
        Y = y[..., 0] + 1j * y[..., 1]
        out = []
        for t in range(n):
            self._frames += 1
            if self._frames <= self._look:
                continue
            Yt = self._post.spec(Y[t], Xd[t]) if self._post is not None else Y[t]
            self._ola += np.fft.irfft(Yt, N_FFT) * self._win
            hop = self._ola[:HOP] / 1.5
            self._ola = np.concatenate([self._ola[HOP:], np.zeros(HOP)])
            if self._skip:
                d = min(self._skip, HOP)
                hop = hop[d:]
                self._skip -= d
            out.append(self._post.time(hop) if self._post is not None else hop)
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Feed any number of 16 kHz float samples; returns the cleaned samples that are ready."""
        audio = np.asarray(audio, np.float32)
        self._n_in += len(audio)
        self._buf = np.concatenate([self._buf, audio])
        y = self._run()
        self._n_out += len(y)
        return y

    def flush(self) -> np.ndarray:
        """Push silence through so the samples still inside come out. After this, as many samples have come out as went in."""
        want = self._n_in - self._n_out
        self._buf = np.concatenate([self._buf, np.zeros(self.latency_samples + N_FFT + (-len(self._buf)) % HOP, np.float32)])
        y = self._run()[:max(0, want)]
        self._n_out += len(y)
        return y


def clean(audio: np.ndarray, sample_rate: int, post: str | Post | None = "default", model_path: str | None = None,
          model: str | None = None) -> np.ndarray:
    """Clean a whole recording (mono float). Returns audio at the same rate and length, time-aligned with the input.

    model: "fast" (the default) or "best". The model works at 16 kHz, so the result carries sound up to 8 kHz whatever rate it is returned at.
    """
    import soxr

    x = np.asarray(audio, np.float32)
    x16 = soxr.resample(x, sample_rate, SR) if sample_rate != SR else x
    nf = Nearfield(model_path, post, model=model)
    y = np.concatenate([nf.process(x16), nf.flush()])
    if sample_rate == SR:
        return y
    y = soxr.resample(y, SR, sample_rate).astype(np.float32)
    return np.concatenate([y, np.zeros(max(0, len(x) - len(y)), np.float32)])[:len(x)]
