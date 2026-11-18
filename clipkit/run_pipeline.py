#!/usr/bin/env python3
"""End-to-end: source video -> transcript -> best moments -> edited 9:16 clips -> titles CSV -> zip.

Usage: run_pipeline.py --source input/video.mp4 --channel channel1 [--template auto] [--count 5] [--out out]
       add --words words.json to skip transcription (reuse an existing transcript)
"""
import argparse, json, os, shutil, subprocess, sys, textwrap

K = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE = {"channel1": "gta_moment", "channel2": "stream_clutch", "channel3": "video_story"}

def sh(cmd, **kw): subprocess.run(cmd, check=True, **kw)

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

def wrap_hook(text, width=17, maxlines=3):
    lines = textwrap.wrap(text.upper(), width=width)[:maxlines]
    return "\\N".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--channel", default="channel1", choices=["channel1", "channel2", "channel3"])
    ap.add_argument("--template", default="auto")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min", type=float, default=15); ap.add_argument("--max", type=float, default=45)
    ap.add_argument("--out", default="out")
    ap.add_argument("--words", default=None, help="existing words.json, skips transcription")
    ap.add_argument("--required", default="", help='campaign text to put in descriptions, e.g. "@brand #ad"')
    a = ap.parse_args()

    template = DEFAULT_TEMPLATE[a.channel] if a.template == "auto" else a.template
    src = os.path.abspath(a.source); out = os.path.abspath(a.out)
    work = os.path.join(out, "_work"); clips_dir = os.path.join(out, "clips")
    os.makedirs(work, exist_ok=True); os.makedirs(clips_dir, exist_ok=True)
    make_sfx()

    # 1. transcript
    words_path = os.path.abspath(a.words) if a.words else os.path.join(work, "words.json")
    if not a.words:
        wav = os.path.join(work, "audio.wav")
        print("== 1/5 extracting audio", flush=True)
        sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav])
        print("== 2/5 transcribing (Whisper, free, runs on this machine)", flush=True)
        sh([sys.executable, os.path.join(K, "transcribe.py"), wav, words_path])
    words = json.load(open(words_path))
    if not words: sys.exit("No speech found in the video, nothing to clip.")

    # 2. moments
    print("== 3/5 picking best moments", flush=True)
    moments_path = os.path.join(out, "moments.json")
    sh([sys.executable, os.path.join(K, "pick_moments.py"), words_path, moments_path,
        "--count", str(a.count), "--min", str(a.min), "--max", str(a.max)])
    moments = json.load(open(moments_path))
    if not moments: sys.exit("No moments found.")

    # 3. render
    cta = json.load(open(os.path.join(K, "channels.json")))[a.channel]["cta"][0]
    print(f"== 4/5 editing {len(moments)} clips with template '{template}'", flush=True)
    done = []
    for m in moments:
        out_mp4 = os.path.join(clips_dir, f"{m['name']}.mp4")
        job = {"template": template, "channel": a.channel, "source": src, "words": words_path, "out": out_mp4,
               "segments": [{"s": m["start"], "e": m["end"]}], "hook": wrap_hook(m["hook"]) if m.get("hook") else "",
               "cta": cta}
        jp = os.path.join(work, m["name"] + ".json"); json.dump(job, open(jp, "w"))
        try:
            sh([sys.executable, os.path.join(K, "engine.py"), jp]); done.append(m)
        except subprocess.CalledProcessError:
            print(f"!! {m['name']} failed to render, skipping", flush=True)
        shutil.rmtree(os.path.join(clips_dir, f"{m['name']}_tmp"), ignore_errors=True)

    # 4. titles
    print("== 5/5 writing titles", flush=True)
    clips_json = os.path.join(out, "clips.json")
    json.dump([{"name": m["name"], "start": m["start"], "end": m["end"], "hook": m.get("hook", "")} for m in done],
              open(clips_json, "w"))
    cmd = [sys.executable, os.path.join(K, "titles.py"), words_path, clips_json, a.channel]
    if a.required: cmd += ["--required", a.required]
    sh(cmd, cwd=out)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nDONE: {len(done)} clips in {clips_dir}, titles in {out}/titles_{a.channel}.csv")

if __name__ == "__main__":
    main()
