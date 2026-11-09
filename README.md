# Nearfield

**Real-time caller voice isolation.** Nearfield keeps the voice that is close to the microphone and removes noise and the people talking in the background: a colleague across the room, a TV, a call-centre floor.

By [Ground State Foundry](https://github.com/ground-state-foundry).

- **Real time:** 10 ms frames, 30 ms latency, about 0.5 ms of CPU per frame in Python (onnxruntime, one thread).
- **One model for 8, 16 and 48 kHz:** phone lines, WebRTC and full-band audio.
- **Small:** one 11 MB ONNX file, downloaded on first use.
- **Drop-in for [Pipecat](https://github.com/pipecat-ai/pipecat)** voice agents.

## How it works

Nearfield is [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) (noise suppression) plus a small caller-ownership head (562k parameters). The head reads DeepFilterNet's internal features, which already encode how far away and how reverberant a voice is. Every 10 ms, for each of 32 frequency bands, it decides whether the sound belongs to the near voice or to the background. A keep/cut rule then removes what isn't the caller, and closes the gaps between the caller's words.

## Install

```bash
pip install "nearfield @ git+https://github.com/ground-state-foundry/nearfield"
```

## Use it in Pipecat

```python
from nearfield.pipecat import NearfieldFilter

transport = DailyTransport(
    room_url, token, "bot",
    DailyParams(audio_in_enabled=True, audio_in_filter=NearfieldFilter()),
)
```

`NearfieldFilter(strength="full")` accepts `"gentle"`, `"strong"` or `"full"`. Send a `FilterEnableFrame(False)` to bypass it at runtime.

## Use it directly

```python
import soundfile as sf
from nearfield import clean

audio, sr = sf.read("call.wav", dtype="float32")
sf.write("call_clean.wav", clean(audio, sr), sr)
```

Streaming at 48 kHz:

```python
from nearfield import Nearfield

nf = Nearfield(strength="full")
for chunk in mic_chunks_48k:          # any chunk size, float32
    out = nf.process(chunk)           # whole 10 ms frames, delayed by 30 ms
```

## Strength

| Setting | Background while the caller is silent | Caller intelligibility (STOI) |
|---|---|---|
| DeepFilterNet3 alone, for reference | −16 dB | 0.878 |
| `gentle` | −22 dB | 0.902 |
| `strong` | −37 dB | 0.893 |
| `full` (default) | −41 dB | 0.882 |

Measured on 48 held-out simulated calls (unseen speakers and rooms) at 8, 16 and 48 kHz. While the caller is talking, background voices are reduced but can stay faintly audible when they overlap the caller in the same frequencies.

## Weights

The model is attached to the [v1.0.0 release](https://github.com/ground-state-foundry/nearfield/releases/tag/v1.0.0) and cached in `~/.cache/nearfield`. Set `NEARFIELD_MODEL=/path/to/nearfield-v1.onnx` to use a local copy (no network), or `NEARFIELD_HOME` to move the cache.

## License

Apache-2.0. Nearfield includes DeepFilterNet3 weights by Hendrik Schröter, used under the MIT license; see [NOTICE](NOTICE).
