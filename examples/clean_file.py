"""Clean a recording: python examples/clean_file.py call.wav call_clean.wav [fast|best] [default|level|none]"""
import sys

import soundfile as sf

from nearfield import clean

audio, sr = sf.read(sys.argv[1], dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(1)
sf.write(sys.argv[2], clean(audio, sr, model=sys.argv[3] if len(sys.argv) > 3 else "fast", post=sys.argv[4] if len(sys.argv) > 4 else "default"), sr)
print("wrote", sys.argv[2])
