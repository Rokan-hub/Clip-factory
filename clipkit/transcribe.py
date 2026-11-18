#!/usr/bin/env python3
"""Usage: transcribe.py audio.wav words.json   (16 kHz mono 16-bit wav)

1st choice: Groq Whisper large-v3-turbo (free tier, cloud, very fast) if GROQ_API_KEY is set.
2nd choice: Cloudflare Workers AI Whisper (free daily allowance) if CF_ACCOUNT_ID + CF_API_TOKEN are set.
3rd choice: faster-whisper on this machine (free, slower).
Env: WHISPER_MODEL (local, default small), WHISPER_LANG (default en), GROQ_STT_MODEL
"""
import glob, json, os, subprocess, sys, tempfile, time, uuid, urllib.request, urllib.error

LANG = os.environ.get("WHISPER_LANG", "en")

def dur(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True).stdout.strip()
    return float(o or 0)

def groq_chunk(path, key, model):
    b = uuid.uuid4().hex
    fields = {"model": model, "response_format": "verbose_json", "language": LANG,
              "timestamp_granularities[]": "word", "temperature": "0"}
    body = b""
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.mp3"\r\n'
             f"Content-Type: audio/mpeg\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    for attempt in range(4):
        req = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", data=body,
                                     headers={"Authorization": f"Bearer {key}", "User-Agent": "clip-factory",
                                              "Content-Type": f"multipart/form-data; boundary={b}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r).get("words") or []
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(30 * (attempt + 1)); continue
            raise RuntimeError(f"Groq STT HTTP {e.code}: {msg}")
    raise RuntimeError("Groq STT retries exhausted")

def chunks(audio):
    d = tempfile.mkdtemp()
    # 10-minute mp3 chunks (~3.6 MB each, well under the 25 MB free-tier file limit)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio, "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                    "-b:a", "48k", "-f", "segment", "-segment_time", "600", "-reset_timestamps", "1",
                    os.path.join(d, "c%03d.mp3")], check=True)
    return sorted(glob.glob(os.path.join(d, "c*.mp3")))

def cloudflare_chunk(path, acct, token):
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/openai/whisper"
    for attempt in range(3):
        req = urllib.request.Request(url, data=open(path, "rb").read(),
                                     headers={"Authorization": f"Bearer {token}", "User-Agent": "clip-factory",
                                              "Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["result"].get("words") or []
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 503) and attempt < 2:
                time.sleep(20 * (attempt + 1)); continue
            raise RuntimeError(f"Cloudflare STT HTTP {e.code}: {msg}")

def cloud(audio, fn):
    words, off = [], 0.0
    for p in chunks(audio):
        for w in fn(p):
            words.append({"w": w["word"].strip(), "s": round(off + w["start"], 3), "e": round(off + w["end"], 3)})
        off += dur(p)
        print("t=%.0f" % off, flush=True)
    return words

def local(audio):
    import wave
    import numpy as np
    from faster_whisper import WhisperModel
    m = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8",
                     cpu_threads=os.cpu_count() or 2)
    wf = wave.open(audio)
    a = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    segs, info = m.transcribe(a, language=LANG, word_timestamps=True, vad_filter=True,
                              beam_size=1, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})
        print("t=%.0f" % s.end, flush=True)
    return words

if __name__ == "__main__":
    audio, out = sys.argv[1], sys.argv[2]
    words, how = None, "local whisper"
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        try:
            m = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
            words = cloud(audio, lambda p: groq_chunk(p, key, m)); how = "groq cloud whisper"
        except Exception as ex:
            print("WARNING: Groq transcription failed, trying the next option ->", ex, file=sys.stderr)
    acct, tok = os.environ.get("CF_ACCOUNT_ID", "").strip(), os.environ.get("CF_API_TOKEN", "").strip()
    if words is None and acct and tok:
        try:
            words = cloud(audio, lambda p: cloudflare_chunk(p, acct, tok)); how = "cloudflare whisper"
        except Exception as ex:
            print("WARNING: Cloudflare transcription failed, using local Whisper ->", ex, file=sys.stderr)
    if words is None:
        words = local(audio)
    json.dump(words, open(out, "w"))
    print("DONE", len(words), "words via", how, flush=True)
