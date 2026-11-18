#!/usr/bin/env python3
"""Usage: transcribe.py audio.wav words.json   (16 kHz mono 16-bit wav)
Env: WHISPER_MODEL (default small), WHISPER_LANG (default en)"""
import json, os, sys, wave
import numpy as np
from faster_whisper import WhisperModel

audio, out = sys.argv[1], sys.argv[2]
m = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8",
                 cpu_threads=os.cpu_count() or 2)
wf = wave.open(audio)
a = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
segs, info = m.transcribe(a, language=os.environ.get("WHISPER_LANG", "en"), word_timestamps=True,
                          vad_filter=True, beam_size=1, condition_on_previous_text=False)
words = []
for s in segs:
    for w in (s.words or []):
        words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})
    print("t=%.0f" % s.end, flush=True)
json.dump(words, open(out, "w"))
print("DONE", len(words), flush=True)
