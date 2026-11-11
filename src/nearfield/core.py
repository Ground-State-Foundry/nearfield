"""Nearfield streaming runtime: keeps the voice close to the mic, removes noise and background voices.

The ONNX model runs one 10 ms frame (480 samples at 48 kHz) at a time and returns, for the frame two steps back,
the noise-suppressed spectrum and a per-band probability that the near (caller) voice owns that band. The keep/cut
rule below turns that into the final spectrum. Latency is 30 ms (20 ms lookahead + one 10 ms hop).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 48000
HOP = 480
N_FFT = 960
N_BINS = 481
LATENCY_SAMPLES = 3 * HOP                         # at 48 kHz

_ERB_WIDTHS = [2] * 13 + [5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67]
_BAND = np.repeat(np.arange(32), _ERB_WIDTHS)
_STATE_LEN = 44728


def initial_state() -> np.ndarray:
    s = np.zeros(_STATE_LEN, np.float32)
    s[:32] = np.linspace(-60.0, -90.0, 32)
    s[32:128] = np.linspace(0.001, 0.0001, 96)
    return s


@dataclass(frozen=True)
class Strength:
    """How hard to cut what isn't the near voice.

    k        sharpening of the ownership decision (1 = as predicted)
    far_db   cut applied to bands owned by the background
    gate_db  extra cut for whole frames where the near voice is silent (None = off)
    keep_db  bands owned by the near voice are never cut below this

    talk_thr       ownership (averaged over talk_avg frames) above which the near voice is 'definitely talking';
                   None switches the whole talk state off
    talk_avg       frames (10 ms each) of averaging before that threshold, so short background bursts don't trigger it
    talk_hold      frames the talk state is held after it last fired
    talk_far_db    cut for background-owned bands while the talk state is held (instead of far_db)
    talk_floor_db  while the talk state is held, no band is cut below this
    """
    k: float = 2.0
    far_db: float = -60.0
    gate_db: float | None = -45.0
    keep_db: float = -6.0
    talk_thr: float | None = None
    talk_avg: int = 10
    talk_hold: int = 150
    talk_far_db: float = -12.0
    talk_floor_db: float = -8.0


STRENGTHS = {
    "gentle": Strength(k=1.0, far_db=-20.0, gate_db=None),
    "balanced": Strength(k=1.0, far_db=-60.0, gate_db=-45.0, talk_thr=0.6),
    "strong": Strength(k=1.0, far_db=-60.0, gate_db=-45.0),
    "full": Strength(k=2.0, far_db=-60.0, gate_db=-45.0),
}


class _Rule:
    def __init__(self, s: Strength):
        self.s = s
        self.keep, self.far = 10 ** (s.keep_db / 20), 10 ** (s.far_db / 20)
        self.floor = None if s.gate_db is None else 10 ** (s.gate_db / 20)
        self.prev, self.hold = 0.0, 0
        self.talk, self.talk_hold = 0.0, 0
        self.hist = np.zeros(max(1, s.talk_avg))
        self.talk_floor = 10 ** (s.talk_floor_db / 20)

    def __call__(self, xn: np.ndarray, xd: np.ndarray, own: np.ndarray) -> np.ndarray:
        s = self.s
        o = own if s.k == 1 else own ** s.k / (own ** s.k + (1 - own) ** s.k)
        G, far, talk = 1.0, self.far, 0.0
        if self.floor is not None or s.talk_thr is not None:
            w = np.bincount(_BAND, weights=np.abs(xd) ** 2, minlength=32) + 1e-12
            stat = float((own * w).sum() / w.sum())                # energy-weighted share of the frame owned by the near voice
        if s.talk_thr is not None:
            self.hist = np.roll(self.hist, -1)
            self.hist[-1] = stat
            t0 = float(np.clip((self.hist.mean() - s.talk_thr) / 0.3, 0, 1))
            if t0 >= self.talk:
                self.talk, self.talk_hold = t0, s.talk_hold
            elif self.talk_hold > 0:
                self.talk_hold -= 1
            else:
                self.talk = max(t0, self.talk * 0.9)
            talk = self.talk
            far = 10 ** ((s.far_db * (1 - talk) + s.talk_far_db * talk) / 20)
        if self.floor is not None:
            g0 = float(np.clip((stat - 0.25) / 0.35, 0, 1))
            if g0 >= self.prev:
                self.prev, self.hold = g0, 15
            elif self.hold > 0:
                self.hold -= 1
            else:
                self.prev = max(g0, self.prev * 0.85)
            G = self.floor + (1 - self.floor) * max(self.prev, talk)
        dg = np.clip(np.abs(xd) / (np.abs(xn) + 1e-9), 0, 1)
        ob = o[_BAND]
        g = (ob * np.maximum(dg, self.keep) + (1 - ob) * np.minimum(dg, far)) * G
        if talk > 0:
            g = np.maximum(g, self.talk_floor * talk)
        return np.where(g >= dg - 1e-6, xn * g, xd * (g / (dg + 1e-9)))


class Nearfield:
    """Streaming cleaner at 48 kHz.

    >>> nf = Nearfield()                       # downloads the weights once
    >>> out = nf.process(chunk_48k_float32)    # any chunk size; returns whole 10 ms frames
    """

    def __init__(self, model_path: str | None = None, strength: str | Strength = "full", threads: int = 1):
        import onnxruntime as ort
        from .weights import model_path as _default_path

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(model_path or _default_path(), opts, providers=["CPUExecutionProvider"])
        self._strength = STRENGTHS[strength] if isinstance(strength, str) else strength
        w = np.sin(0.5 * np.pi * (np.arange(N_FFT) + 0.5) / HOP)
        self._win = np.sin(0.5 * np.pi * w ** 2)
        self._wnorm = 1.0 / (N_FFT ** 2 / (2 * HOP))
        self.reset()

    def reset(self) -> None:
        self._state = initial_state()
        self._amem = np.zeros(HOP, np.float32)
        self._smem = np.zeros(HOP)
        self._hist = [np.zeros(N_BINS, np.complex64), np.zeros(N_BINS, np.complex64)]
        self._buf = np.zeros(0, np.float32)
        self._rule = _Rule(self._strength)
        self.last_own = np.zeros(32, np.float32)

    def set_strength(self, strength: str | Strength) -> None:
        self._strength = STRENGTHS[strength] if isinstance(strength, str) else strength
        self._rule = _Rule(self._strength)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """One 480-sample frame at 48 kHz in, one out (delayed by LATENCY_SAMPLES)."""
        X = np.fft.rfft(np.concatenate([self._amem, frame]) * self._win) * self._wnorm
        self._amem = np.asarray(frame, np.float32)
        spec = np.stack([X.real, X.imag], -1).astype(np.float32)
        _, y_dfn, own, self._state = self._sess.run(None, {"spec": spec, "state": self._state})
        xn = self._hist.pop(0)
        self._hist.append(X.astype(np.complex64))
        xd = y_dfn[:, 0] + 1j * y_dfn[:, 1]
        self.last_own = own
        Y = self._rule(xn, xd, own)
        x = np.fft.irfft(Y, N_FFT) * N_FFT * self._win
        out = x[:HOP] + self._smem
        self._smem = x[HOP:]
        return out.astype(np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Feed any number of 48 kHz float samples; returns all completed output frames."""
        self._buf = np.concatenate([self._buf, np.asarray(audio, np.float32)])
        n = len(self._buf) // HOP
        if n == 0:
            return np.zeros(0, np.float32)
        out = [self.process_frame(self._buf[i * HOP:(i + 1) * HOP]) for i in range(n)]
        self._buf = self._buf[n * HOP:]
        return np.concatenate(out)

    def flush(self) -> np.ndarray:
        """Push silence through so the last LATENCY_SAMPLES of real audio come out."""
        pad = (-len(self._buf)) % HOP + LATENCY_SAMPLES
        return self.process(np.zeros(pad, np.float32))


def clean(audio: np.ndarray, sample_rate: int, strength: str = "full", model_path: str | None = None) -> np.ndarray:
    """Clean a whole recording (mono float). Returns audio at the same rate and length, latency removed."""
    import soxr

    x = np.asarray(audio, np.float32)
    x48 = soxr.resample(x, sample_rate, SR) if sample_rate != SR else x
    nf = Nearfield(model_path, strength)
    y = np.concatenate([nf.process(x48), nf.flush()])[LATENCY_SAMPLES: LATENCY_SAMPLES + len(x48)]
    return soxr.resample(y, SR, sample_rate).astype(np.float32) if sample_rate != SR else y
