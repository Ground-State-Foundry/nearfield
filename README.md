# Nearfield

**Real-time caller voice isolation.** Nearfield keeps the voice that is close to the microphone and removes noise and the people talking in the background: a colleague across the room, a TV, a call-centre floor.

By [Ground State Foundry](https://github.com/ground-state-foundry).

- **Real time on a CPU:** about 0.2 s of compute per second of audio on one thread (onnxruntime, Intel Xeon Platinum 8488C).
- **Beats Krisp and Sanas on Krisp's own benchmark:** on Krisp's public set of 265 real recordings, both models make fewer word errors than Krisp VI 2.5 and Sanas and let fewer background words through. See the table below.
- **Two models, one switch:** `fast` (the default) has 48 ms of delay. `best` has 80 ms and lets fewer background words through. Same size, same speed.
- **Small:** each model is one 2.5 MB ONNX file, about 628,000 weights, shipped inside the package. Nothing is downloaded at run time.
- **16 kHz.** Phone lines (8 kHz) and wideband calls both work; audio at other rates is resampled in and out.
- **Drop-in for [Pipecat](https://github.com/pipecat-ai/pipecat)** voice agents.

## Benchmark: real recordings

[Krisp's Voice Isolation benchmark](https://huggingface.co/datasets/Krisp-AI/VoiceIsolation-Benchmark-Dataset): 265 real recordings (headsets at work, call-centre floors, phone calls) with other people talking in the room. Each filter's output is transcribed and compared with what the caller said. Lower is better in both columns.

| # | Model* | Access | Word error rate | Background words let through, per minute |
|---|---|---|---|---|
| 1 | **Nearfield `best` (v2.3, 80 ms)** | **open** | **4.99%** | **0.36** |
| 2 | **Nearfield `fast` (v2.3, 48 ms)** | **open** | **5.01%** | **0.70** |
| 3 | Krisp VI 2.5 Default | paid | 5.30% | 1.52 |
| 4 | Krisp VI 2.5 Lite | paid | 6.21% | 4.52 |
| 5 | ai-coustics Voice Focus 2.2 L | paid | 6.84% | 0.66 |
| 6 | Sanas Voice Isolation | paid | 9.38% | 13.76 |
| 7 | DeepFilterNet3 | open | 13.30% | 22.77 |
| | No filter | | 23.21% | 66.47 |

<sub>* Open models under 50 MB and the paid filters (known sizes: Nearfield 2.5 MB, DeepFilterNet3 about 8 MB, ai-coustics Voice Focus 2.2 L about 20 MB; Krisp and Sanas do not publish one).</sub>

Word error rate by scenario:

| | Work (96 files) | Call centre (106) | Phone calls (63) |
|---|---|---|---|
| No filter | 29.31% | 25.13% | 3.82% |
| Krisp VI 2.5 Default | 5.18% | 5.69% | 4.15% |
| Nearfield `fast` | 4.77% | 5.24% | 4.66% |
| Nearfield `best` | 5.05% | 5.12% | 4.38% |

How to read it:

- **Scoring** follows Krisp's published method: Parakeet TDT 0.6B v2 transcripts, Krisp's text normalisation, word error rate over the whole set. "Background words" are words the recogniser writes during stretches where only a background voice is talking.
- **Who ran what.** We ran the Nearfield, Krisp and no-filter rows (Krisp publishes its own outputs for every file). The ai-coustics, Sanas and DeepFilterNet3 rows are from [The Voice on the Line](https://arxiv.org/abs/2610.35272), which uses the same scoring; our Krisp and no-filter rows reproduce that paper's figures exactly.
- **Gaps of a few tenths of a point are within noise.** We have no error bars for this set. The lead over Krisp Default on words is small; the lead on background words is larger.
- **v2.3 was tuned against this benchmark.** Its recordings were never used for training, but two rounds of changes to our training recipe were judged on it, and the `best` weights are the snapshot that scored best on it. Expect slightly worse on other recordings.
- **Phone calls** have no background talker. No filter at all scores best there, and Krisp is a little ahead of us.

## What is new in v2.3

v2.2 and earlier decided who the caller was by how close they sounded. That was a hidden cutoff: a caller on a car's hands-free, a speakerphone or some headsets sounded "far", and the model muted them, sometimes for the rest of the call. On the benchmark above v2.2 scored 7.83% (`fast`) and 8.37% (`best`), almost all of it from a few callers who were silenced entirely.

v2.3 is the same network retrained with a different rule: **the caller is the nearest voice heard so far in the call, however far away that is.**

- **Far callers are kept.** Word error rate on the phone-call recordings fell from 16.1% to 4.7% (`fast`).
- **Long calls.** Training calls are now 60 seconds long with pauses of up to 40 seconds, so the model holds on to the caller through a long stretch where only the background is talking. v2.2 was trained on clips of a few seconds and could lose the caller after such a stretch.
- **Quiet callers and processed audio.** Training now covers callers down to about -52 dBFS and what phones and headsets do before the audio arrives: noise suppression, automatic gain, noise gates, lost packets.
- **`clean()` on long files** no longer needs gigabytes of memory; a ten-minute recording peaks at about 250 MB.

What it costs: **a voice that speaks before the caller does is let through until the caller starts.** v2.2 silenced it. If a TV or a colleague is audible at the start of a call, you will hear them until the caller's first word. See [Limits](#limits).

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
| v2.2 `fast` (48 ms delay) | 2.0% | 3.3% | 6.1% | 15.9% | 37.0% |
| v2.2 `best` (80 ms delay) | 1.8% | 3.5% | 5.4% | 13.6% | 30.9% |
| **v2.3 `fast` (48 ms delay, the default)** | **2.0%** | **3.3%** | **6.8%** | **19.7%** | **35.9%** |
| **v2.3 `best` (80 ms delay)** | **2.4%** | **4.3%** | **7.2%** | **17.3%** | **34.6%** |

40 callers per condition reading English sentences, mixed with a background talker, babble or steady noise, sent through an 8 kHz phone channel and transcribed with Whisper large-v3-turbo. Gaps under about 5 points are within noise. The no-filter and v1 rows are the ones published with v1.1.0; the v2 rows were scored later on the same clips with the same recogniser, on the packaged versions with the clean-up stage on.

**v2.3 scores worse than v2.2 on this test.** In these clips the background voice is already talking when the caller starts, and under the v2.3 rule that voice is kept until the caller's first word. Against an equally loud talker, v2.3 `fast` loses the caller's first word in 25% of clips where v2.2 lost it in 10%. The real recordings above, where v2.3 is well ahead, are the better guide to a live call.

What else was measured, on the same clips:

- **When only the background is talking and no caller has spoken yet**, v2.3 keeps a single background talker, by design: it is the nearest voice so far. Nearly all of those clips produce a transcript. Babble with no caller is turned down only a little (11 dB for `fast`, 5 dB for `best`, before volume levelling), and 32% and 50% of those clips still produce a transcript of four or more words. v2.2 silenced both (45 to 49 dB down for a talker, 35 to 44 dB for babble).
- **While the caller is talking**, an equally loud background talker comes out about 16 dB below where it went in, and the caller about 4 dB below (measured on the v2.2 model output, before the clean-up stage). So the background is still faintly there under the caller; it is not gone.

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

| | Delay | Word error rate | Background words per minute |
|---|---|---|---|
| `fast` (default) | 48 ms | 5.01% | 0.70 |
| `best` | 80 ms | 4.99% | 0.36 |

Pick `fast` when every millisecond of the call matters. Pick `best` when callers are often in a room with other people talking and 32 ms more delay is acceptable: it lets about half as many background words through. Compute and size are the same.

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

- **A voice that speaks before the caller is let through** until the caller's first word. The model has no way to know a closer voice is coming.
- **Babble is still hard.** With a whole room talking at the caller's level, about 35 in 100 of the caller's words are still lost. No filter loses 38. And a room of babble with no caller yet is no longer silenced.
- **The background is reduced under the caller, not removed.** See the 16 dB figure above.
- **A little more delay than v1.** 48 ms or 80 ms against 30 ms.
- **16 kHz.** The output carries sound up to 8 kHz. v1 worked at 48 kHz.
- **One real benchmark.** The real-recording figures come from a single public set that v2.3 was tuned against; other lines and rooms will differ.
- Volume levelling takes a second or two to settle at the start of a call, because it has to hear the caller first.

## v1

v1 (30 ms delay, 48 kHz, built on DeepFilterNet3) is still available:

```bash
pip install "nearfield @ git+https://github.com/ground-state-foundry/nearfield@v1.1.0"
```

## Weights

The models are `src/nearfield/nearfield-v2.3.onnx` (`fast`) and `src/nearfield/nearfield-v2.3-best.onnx` (`best`). Both install with the package and are attached to the [v2.3.0 release](https://github.com/ground-state-foundry/nearfield/releases/tag/v2.3.0). The v2.2 weights stay available from the `v2.2.0` tag. Set `NEARFIELD_MODEL=/path/to/file.onnx` to use a different copy of the chosen model.

## License

Apache-2.0, code and weights. See [NOTICE](NOTICE).
