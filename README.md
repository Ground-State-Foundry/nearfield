# Nearfield

**Real-time caller voice isolation.** Nearfield keeps the voice that is close to the microphone and removes noise and the people talking in the background: a colleague across the room, a TV, a call-centre floor.

By [Ground State Foundry](https://github.com/ground-state-foundry).

- **Real time on a CPU:** about 0.2 s of compute per second of audio on one thread (onnxruntime, Intel Xeon Platinum 8488C).
- **Two models, one switch:** `fast` (the default) has 48 ms of delay. `best` has 80 ms and keeps more of the caller when other people are talking. Same size, same speed.
- **Small:** each model is one 2.5 MB ONNX file, about 628,000 weights, shipped inside the package. Nothing is downloaded at run time.
- **16 kHz.** Phone lines (8 kHz) and wideband calls both work; audio at other rates is resampled in and out.
- **Drop-in for [Pipecat](https://github.com/pipecat-ai/pipecat)** voice agents.

## What is new in v2

v2 is a new model, trained from scratch by us for this one job. It does not build on any other model. It was trained on simulated calls built from publicly available speech and noise recordings.

v1 asked, for 32 frequency bands every 10 ms, "is this the caller or the room?", and turned bands up or down. That works when the background sits in different frequencies from the caller. When someone else is talking at the same loudness, their voice lands in the same bands, and v1 either let them through or cut the caller.

v2 separates the two voices instead of choosing between bands. Against an equally loud background talker it loses 15.9% of the caller's words where v1 lost about 31%.

**v2.1** is the same design retrained with the look-ahead cut from 96 ms to 16 ms, which takes the delay from 128 ms to 48 ms. It costs a few points of words, shown in the table.

**v2.2** adds a second set of weights next to it, trained with 48 ms of look-ahead (80 ms of delay). It scores the same as v2.0 with 48 ms less delay. The 48 ms model stays the default and is now called `fast`; the new one is `best`. See [Fast or best](#fast-or-best).

Share of the caller's words missing from the transcript (lower is better):

| | Clean | Steady noise | Quieter talker | Equal talker | Babble |
|---|---|---|---|---|---|
| No filter | 2.3% | 3.1% | 17.9% | 41.4% | 38.4% |
| v1 `gentle` | 2.6% | 3.3% | 8.8% | 30.9% | 40.3% |
| v1 `balanced` | 2.4% | 2.8% | 12.5% | 31.3% | 51.2% |
| v1 `strong` | 2.7% | 5.1% | 20.1% | 47.0% | 60.6% |
| v2.0 (128 ms delay) | 2.3% | 4.4% | 5.4% | 13.9% | 32.2% |
| **v2 `fast` (48 ms delay, the default)** | **2.0%** | **3.3%** | **6.1%** | **15.9%** | **37.0%** |
| **v2 `best` (80 ms delay)** | **1.8%** | **3.5%** | **5.4%** | **13.6%** | **30.9%** |

40 callers per condition reading English sentences, mixed with a background talker, babble or steady noise, sent through an 8 kHz phone channel and transcribed with Whisper large-v3-turbo. Gaps under about 5 points are within noise. The no-filter and v1 rows are the ones published with v1.1.0; the v2 rows were scored later on the same clips with the same recogniser, on the packaged versions with the clean-up stage on.

What else was measured, on the same clips:

- **When only the background is talking** (no caller in the clip), the output of `fast` was 45 dB below its input for a background talker and 35 dB for babble; for `best` it was 49 dB and 44 dB. None of those clips produced a transcript of four or more words (v1 `strong`: 38% of the talker clips and 12% of the babble clips did).
- **While the caller is talking**, an equally loud background talker comes out about 16 dB below where it went in, and the caller about 4 dB below (measured on the model output, before the clean-up stage). So the background is still faintly there under the caller; it is not gone.

## How it works

The call is cut into short overlapping slices (32 ms long, one every 8 ms) and each slice is turned into its spectrum: how much sound there is at each frequency. The spectrum is grouped into 64 bands, finer at the low frequencies where voices differ most.

A small recurrent network then reads those bands. It has four blocks. Each block does two things: it reads across the frequencies of one slice, in both directions, to see what belongs together; and it reads along time, forwards only, to follow each voice from one slice to the next. Because it only reads forwards in time, it can run on a live call.

For every slice the network writes a mask: for each frequency, how much of the sound to keep and how to shift its phase. Applying the mask to the input gives the caller alone. The network waits a few slices before it commits to one: 2 slices (16 ms) for `fast`, 6 slices (48 ms) for `best`. With the 32 ms slice itself, that makes 48 ms or 80 ms of delay. The two are the same network with different weights.

After the model, a light clean-up stage runs on the model's output only (nothing from the original recording is mixed back in):

- **Volume levelling.** The caller is brought to a steady loudness and peaks are limited. Quiet callers are raised by up to 18 dB, loud ones lowered by up to 12 dB.
- **Leftover removal.** A frequency that is suddenly loud while its neighbours and its own recent past are quiet is turned down.
- **Quieter pauses.** The stretches between the caller's words are pushed down by up to 18 dB, fading over about 80 ms.
- **Softer top end.** 5 dB less above 5 kHz, easing in from 3 kHz.

The weights are open. The training code and data pipeline are not part of this repository.

## Fast or best

| | Delay | Equal talker | Babble |
|---|---|---|---|
| `fast` (default) | 48 ms | 15.9% | 37.0% |
| `best` | 80 ms | 13.6% | 30.9% |

Pick `fast` when every millisecond of the call matters. Pick `best` when callers are often in a room with other people talking and 32 ms more delay is acceptable. Compute and size are the same.

```python
Nearfield(model="best")
NearfieldFilter(model="best")
clean(audio, sr, model="best")
```

Or, without touching code, set the environment variable `NEARFIELD_QUALITY=best`. An explicit `model=` argument wins over it.

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

Send a `FilterEnableFrame(False)` to bypass it at runtime. The audio that comes back is 48 ms behind the audio that went in (80 ms with `model="best"`).

## Use it directly

A whole recording:

```python
import soundfile as sf
from nearfield import clean

audio, sr = sf.read("call.wav", dtype="float32")     # mono
sf.write("call_clean.wav", clean(audio, sr), sr)      # same rate, same length, time-aligned
```

Streaming at 16 kHz:

```python
from nearfield import Nearfield

nf = Nearfield()
for chunk in mic_chunks_16k:          # any chunk size, float32
    out = nf.process(chunk)           # the samples that are ready
tail = nf.flush()                     # the last 48 ms
```

Output sample `i` is the cleaned version of input sample `i`. Samples come out 48 ms after they went in (80 ms with `model="best"`; `nf.latency_samples` has the exact number), so the first calls return fewer samples than they were given.

## Clean-up settings

```python
from nearfield import Nearfield, Post

Nearfield(post="default")   # everything described above
Nearfield(post="level")     # volume levelling only
Nearfield(post="none")      # the model output as it is
Nearfield(post=Post(level_db=-20, blips=True, pause_db=12, tilt_db=0))
```

## Limits

- **Babble is still hard.** With a whole room talking at the caller's level, `fast` loses about 37 in 100 of the caller's words and `best` about 31. No filter loses 38 and v1 lost 40 to 61. So in babble `fast` silences the room when the caller is not talking but does not rescue more of the caller's words; `best` rescues some.
- **The background is reduced under the caller, not removed.** See the 16 dB figure above.
- **A little more delay than v1.** 48 ms or 80 ms against 30 ms.
- **The shorter delay costs some words.** That is the difference between `fast` and `best` in the table.
- **16 kHz.** The output carries sound up to 8 kHz. v1 worked at 48 kHz.
- **Measured on simulated calls.** Real phone lines and rooms will differ.
- Volume levelling takes a second or two to settle at the start of a call, because it has to hear the caller first.

## v1

v1 (30 ms delay, 48 kHz, built on DeepFilterNet3) is still available:

```bash
pip install "nearfield @ git+https://github.com/ground-state-foundry/nearfield@v1.1.0"
```

## Weights

The models are `src/nearfield/nearfield-v2.1.onnx` (`fast`) and `src/nearfield/nearfield-v2.2-best.onnx` (`best`). Both install with the package and are attached to the [v2.2.0 release](https://github.com/ground-state-foundry/nearfield/releases/tag/v2.2.0). Set `NEARFIELD_MODEL=/path/to/file.onnx` to use a different copy of the chosen model.

## License

Apache-2.0, code and weights. See [NOTICE](NOTICE).
