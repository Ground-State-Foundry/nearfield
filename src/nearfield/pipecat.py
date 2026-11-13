"""Pipecat audio filter: removes noise and background voices, keeps the caller.

    from nearfield.pipecat import NearfieldFilter
    transport = SomeTransport(params=TransportParams(audio_in_filter=NearfieldFilter()))

Follows Pipecat's RNNoiseFilter: resamples to 16 kHz, cleans, resamples back.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

from .core import SR, Nearfield, Post


class NearfieldFilter(BaseAudioFilter):
    """Real-time near-voice filter for Pipecat transports.

    Args:
        post: "default", "level" or "none", or a custom ``nearfield.Post``: what is done to the separated caller
            (volume levelling, tidying). See ``nearfield.Post``.
        model_path: a different weights file; by default the one inside the package is used.

    The audio that comes back is 48 ms behind the audio that went in.
    """

    def __init__(self, post: str | Post | None = "default", model_path: str | None = None) -> None:
        self._post, self._model_path = post, model_path
        self._filtering = True
        self._sample_rate = 0
        self._nf: Nearfield | None = None
        self._resampler_in = None
        self._resampler_out = None

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate
        try:
            self._nf = Nearfield(self._model_path, self._post)
        except Exception as e:
            logger.error(f"Failed to initialize Nearfield: {e}")
            self._nf = None
            return
        if sample_rate != SR:
            logger.info(f"Nearfield filter enabling resampling: {sample_rate} <-> {SR}")
            self._resampler_in = create_stream_resampler()
            self._resampler_out = create_stream_resampler()

    async def stop(self):
        self._nf = None
        self._resampler_in = None
        self._resampler_out = None

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering or self._nf is None:
            return audio
        in_audio = audio
        if self._sample_rate != SR and self._resampler_in:
            in_audio = await self._resampler_in.resample(audio, self._sample_rate, SR)
        if len(in_audio) == 0:
            return b""
        x = np.frombuffer(in_audio, dtype=np.int16).astype(np.float32) / 32768.0
        y = self._nf.process(x)
        if len(y) == 0:
            return b""
        out = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        if self._sample_rate != SR and self._resampler_out:
            return await self._resampler_out.resample(out, SR, self._sample_rate)
        return out
