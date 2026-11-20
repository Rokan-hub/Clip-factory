#!/usr/bin/env python3
"""End-to-end: source video -> transcript -> best moments -> edited 9:16 clips -> titles CSV.

Usage: run_pipeline.py --source input/video.mp4 --channel channel1 [--template auto] [--count 5] [--out out]
       add --words words.json to skip transcription (reuse an existing transcript)

--template auto: every clip gets the best template from the channel's group
  (Gemini/Groq choose it; without AI a keyword rule does).
"""
import argparse, json, os, re, shutil, subprocess, sys, textwrap

K = os.path.dirname(os.path.abspath(__file__))
GROUP = {"channel1": "gta", "channel2": "stream", "channel3": "video"}
DEFAULT_TEMPLATE = {"channel1": "hype", "channel2": "split_live", "channel3": "hype"}
# fallback rule when no AI chose a template (same 4 presets for every channel)
KEYWORDS = [
    ("hype", r"(\$\s?\d|\b(cops?|police|chase|wanted|run|shoot|crash|crashed|rob|robbery|heist|cash|fight|fail|go go|oh my god|bro what|dead|died)\b)"),
    ("cinematic", r"\b(look at|beautiful|graphics|view|sunset|detail|realistic|map|city|story|mom|mother|family|prison|love)\b"),
    ("split_meme", r"\b(lmao|lol|haha|bro|nah|no way|what the|why|crying|i'm dead|wild)\b"),
    ("split_live", r"\b(i think|this is|look|so|actually|you can see|let's|we're gonna|i'm gonna)\b"),
]
NO_SPEECH_HOOKS = {
    "hype": ["THIS WAS INSANE", "WATCH WHAT HAPPENS", "NOBODY SAW THIS COMING", "THE ENDING THOUGH",
             "THIS DID NOT GO TO PLAN", "GTA 6 PHYSICS ARE WILD", "COPS WON'T STOP"],
    "cinematic": ["GTA 6 LOOKS UNREAL", "THIS ISN'T REAL LIFE", "VICE CITY AT NIGHT", "LOOK AT THIS DETAIL"],
    "split_live": ["HIS FIRST REACTION"], "split_meme": ["BRO IS MORE EXCITED THAN THE CHARACTERS"]}

def sh(cmd, **kw): subprocess.run(cmd, check=True, **kw)

def vdur_probe(src):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                       capture_output=True, text=True).stdout.strip()
    return float(o or 0)

def make_sfx():
    d = os.path.join(K, "sfx"); os.makedirs(d, exist_ok=True)
    recipes = {
        "whoosh": ["-f", "lavfi", "-i", "anoisesrc=d=0.45:c=pink:a=0.8:r=48000", "-af",
                   "bandpass=f=1800:w=2500,afade=t=in:d=0.2,afade=t=out:st=0.2:d=0.25,volume=2"],
        "hit":    ["-f", "lavfi", "-i", "aevalsrc='0.9*sin(2*PI*55*t)*exp(-6*t)+0.5*(random(0)*2-1)*exp(-40*t)':d=0.6:s=48000"],
        "pop":    ["-f", "lavfi", "-i", "aevalsrc='0.6*sin(2*PI*(1100-700*t)*t)*exp(-22*t)':d=0.18:s=48000"],
        "riser":  ["-f", "lavfi", "-i", "aevalsrc='0.35*sin(2*PI*(200+900*t*t)*t)*(t/1.2)':d=1.2:s=48000",
                   "-af", "afade=t=out:st=1.1:d=0.1"],
    }
    for name, args in recipes.items():
        p = os.path.join(d, name + ".wav")
        if not os.path.exists(p): sh(["ffmpeg", "-v", "error", "-y", *args, "-ac", "1", p])

def wrap_hook(text, width=17, maxwords=7):
    """Never cut a hook mid-sentence: shorten by whole words, then wrap."""
    words = text.split()
    return "\\N".join(textwrap.wrap(" ".join(words[:maxwords]).upper(), width=width))

def energy_pick(wav, count, mn, mx):
    """No speech? Pick the loudest action windows (explosions, crashes, sirens, screaming)."""
    import wave, numpy as np
    wf = wave.open(wav); sr = wf.getframerate()
    a = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    hop = sr // 2                                    # 0.5 s blocks
    n = len(a) // hop
    if n < 4: return []
    rms = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-9)
    db = 20 * np.log10(rms); base = np.median(db)
    peak = np.clip(db - base, 0, None)               # loudness above the video's normal level
    L = int(((mn + mx) / 2) * 2)                     # window in blocks
    L = min(L, n)
    score = np.convolve(peak, np.ones(L), "valid") + 2 * np.convolve(peak ** 1.5, np.ones(L), "valid") / L
    out, taken = [], np.zeros(n, bool)
    for i in np.argsort(-score):
        if taken[i:i + L].any(): continue
        taken[i:i + L] = True
        # end the clip just after the loudest moment inside the window (the payoff)
        pk = i + int(np.argmax(peak[i:i + L]))
        s = max(0.0, i * 0.5); e = min(n * 0.5, max(s + mn, pk * 0.5 + 2.5))
        out.append({"start": round(s, 2), "end": round(e, 2), "score": round(float(score[i]), 2),
                    "hook": "", "reason": "loudest action (no speech found)"})
        if len(out) >= count: break
    return sorted(out, key=lambda m: -m["score"])

def keyword_template(text, channel):
    hits = {name: len(re.findall(rx, text.lower())) for name, rx in KEYWORDS}
    best = max(hits, key=hits.get)
    return best if hits[best] >= 2 else DEFAULT_TEMPLATE[channel]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--channel", default="channel1", choices=["channel1", "channel2", "channel3"])
    ap.add_argument("--template", default="auto")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min", type=float, default=12); ap.add_argument("--max", type=float, default=24)
    ap.add_argument("--no-watch", action="store_true", help="skip the Gemini video-watching picker")
    ap.add_argument("--out", default="out")
    ap.add_argument("--words", default=None, help="existing words.json, skips transcription")
    ap.add_argument("--required", default="", help='campaign text to put in descriptions, e.g. "@brand #ad"')
    a = ap.parse_args()

    tdefs = json.load(open(os.path.join(K, "templates.json")))
    group = {n: tdefs[n]["desc"] for n in tdefs["_groups"][GROUP[a.channel]]}
    src = os.path.abspath(a.source); out = os.path.abspath(a.out)
    work = os.path.join(out, "_work"); clips_dir = os.path.join(out, "clips")
    os.makedirs(work, exist_ok=True); os.makedirs(clips_dir, exist_ok=True)
    make_sfx()

    # 1. audio + transcript
    print("== 1/5 extracting audio", flush=True)
    wav = os.path.join(work, "audio.wav")
    sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
    words_path = os.path.abspath(a.words) if a.words else os.path.join(work, "words.json")
    if not a.words:
        print("== 2/5 transcribing", flush=True)
        sh([sys.executable, os.path.join(K, "transcribe.py"), wav, words_path])
    words = json.load(open(words_path))

    # 2. moments
    print("== 3/5 picking best moments", flush=True)
    moments_path = os.path.join(out, "moments.json")
    # 3a. best: Gemini WATCHES the video and returns the payoff and tight cut points (falls back on any failure)
    moments, picker = [], None
    if not a.no_watch and os.environ.get("CF_WATCH", "1") != "0" and os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            import watch
            moments = watch.pick(src, a.count + 2, a.min, a.max, group if a.template == "auto" else None,
                                 os.path.join(work, "watch"))
            for m in moments:
                if m.get("template") not in group: m.pop("template", None)
            picker = "gemini-video"
            print(f"   Gemini watched the video: {len(moments)} candidate moments", flush=True)
        except Exception as ex:
            print(f"   !! video watching failed ({str(ex)[:200]}), using the transcript picker instead", flush=True)
            moments = []
        try:
            import llm; llm.save_log(os.path.join(out, "ai_usage.json")); llm.LOG.clear()
        except Exception: pass
    # 3b. fallback / top-up: transcript picker, or loudest action when there is no speech
    if len(moments) < a.count:
        if len(words) >= 40:
            cmd = [sys.executable, os.path.join(K, "pick_moments.py"), words_path, moments_path,
                   "--count", str(a.count), "--min", str(a.min), "--max", str(a.max)]
            if a.template == "auto": cmd += ["--templates", json.dumps(group)]
            sh(cmd)
            extra = json.load(open(moments_path)); how = "transcript"
        else:
            print("   little/no speech found -> picking the loudest action moments instead", flush=True)
            extra = energy_pick(wav, a.count, a.min, a.max); how = "loudest-action"
        for m in extra:
            if not any(not (m["end"] <= o["start"] or m["start"] >= o["end"]) for o in moments):
                m.setdefault("reason", how); moments.append(m)
        picker = (picker + " + " + how) if picker else how
    moments = moments[:a.count + 2]               # a couple of spares in case tightening drops some
    # 3c. tighten every cut locally: drop credits / logo cards / black screens, trim dead lead-in and tail
    has_cam, cam = True, None
    try:
        import watch, engine
        cam = engine.find_cam(src, vdur_probe(src)); has_cam = bool(cam)
        print(f"   streamer facecam in source: {'yes' if cam else 'no'}", flush=True)
        moments = watch.refine(src, moments, a.min, cam[1] if cam else None)
    except Exception as ex:
        print(f"   cut tightening skipped: {str(ex)[:150]}", flush=True)
    moments = sorted(moments, key=lambda m: -m.get("score", 0))[:a.count]
    for i, m in enumerate(moments, 1): m["name"] = f"clip{i:02d}"
    print(f"   moments picked by: {picker}", flush=True)
    vdur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                                capture_output=True, text=True).stdout.strip() or 0)
    if vdur:
        for m in moments: m["end"] = round(min(m["end"], vdur - 0.1), 2)
        moments = [m for m in moments if m["end"] - m["start"] >= 5]
    if not moments: sys.exit("No moments found.")

    # template per clip
    for m in moments:
        if a.template != "auto": m["template"] = a.template
        elif m.get("template") not in group:
            txt = " ".join(w["w"] for w in words if m["start"] <= w["s"] <= m["end"])
            m["template"] = keyword_template(txt + " " + m.get("hook", ""), a.channel)
    if a.template == "auto":                          # choose by content, not for variety
        for m in moments:
            before = m["template"]
            if m["template"].startswith("split"):
                react = m.get("reaction")
                if not has_cam: m["template"] = "hype"                     # no streamer cam in this video
                elif react is not None and react < 4: m["template"] = "hype"   # flat face: show the game instead
                elif m["template"] == "split_meme" and react is not None and react < 7: m["template"] = "split_live"
            if m["template"] not in group: m["template"] = "hype"
            if m["template"] != before: print(f"   {before} -> {m['template']} for {m['start']}s (facecam/reaction check)", flush=True)
    used = {}
    for m in moments:
        if not m.get("hook"):
            opts = NO_SPEECH_HOOKS.get(m["template"], ["WAIT FOR IT..."]); k = used.get(m["template"], 0)
            m["hook"] = opts[k % len(opts)]; used[m["template"]] = k + 1
    json.dump(moments, open(moments_path, "w"), indent=2)

    # 3. render
    cta = json.load(open(os.path.join(K, "channels.json")))[a.channel]["cta"][0]
    print(f"== 4/5 editing {len(moments)} clips", flush=True)
    done = []
    for m in moments:
        out_mp4 = os.path.join(clips_dir, f"{m['name']}_{m['template']}.mp4")
        job = {"template": m["template"], "channel": a.channel, "source": src, "words": words_path, "out": out_mp4,
               "segments": [{"s": m["start"], "e": m["end"]}], "hook": wrap_hook(m["hook"]), "cta": cta}
        if cam: job["cam"] = [[float(v) for v in cam[0]], [float(v) for v in cam[1]]]
        jp = os.path.join(work, m["name"] + ".json"); json.dump(job, open(jp, "w"))
        try:
            print(f"   {m['name']}: {m['template']}  {m['start']}s-{m['end']}s", flush=True)
            sh([sys.executable, os.path.join(K, "engine.py"), jp]); done.append(m)
        except subprocess.CalledProcessError:
            print(f"!! {m['name']} failed to render, skipping", flush=True)
        shutil.rmtree(os.path.splitext(out_mp4)[0] + "_tmp", ignore_errors=True)

    # 4. titles
    print("== 5/5 writing titles", flush=True)
    clips_json = os.path.join(out, "clips.json")
    json.dump([{"name": f"{m['name']}_{m['template']}", "start": m["start"], "end": m["end"], "hook": m.get("hook", ""),
                "template": m["template"]} for m in done], open(clips_json, "w"))
    cmd = [sys.executable, os.path.join(K, "titles.py"), words_path, clips_json, a.channel]
    if a.required: cmd += ["--required", a.required]
    sh(cmd, cwd=out)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nDONE: {len(done)} clips in {clips_dir}, titles in {out}/titles_{a.channel}.csv")

if __name__ == "__main__":
    main()
