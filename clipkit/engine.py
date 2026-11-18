#!/usr/bin/env python3
"""Template clip engine.  Usage: engine.py job.json
job = {
  "template": "gta_moment", "channel": "channel1",
  "source": "/path/src.mp4", "words": "/path/words.json" (optional, for captions),
  "out": "/path/out.mp4",
  "segments": [{"s":12.0,"e":18.5,"speed":1.0,"freeze":0,"freeze_text":"WAIT..."}],
  "hook": "LINE 1\\NLINE 2", "context": "...", "stakes": "$45,000", "cta": "Follow for part 2",
  "labels": [{"src":13.0,"text":"#1","style":"num"|"chat"|"tag"}],
  "shake_src": [15.2], "crop_x": 0.5, "facecam": [x,y,w,h], "game": [x,y,w,h] (fractions of source frame),
  "music": "/path/bed.mp3", "required_text": "", "no_captions": false
}"""
import json, re, subprocess, sys, os, statistics
K = os.path.dirname(os.path.abspath(__file__))
FONTDIRS = "/usr/share/fonts"
STRONG = {"never","worst","best","insane","crazy","impossible","secret","won","lost","paid","free","million",
          "thousand","dollars","shocked","scared","dropped","wait","no","what","wow","dead","killed","crashed",
          "cops","police","boom","bro","yes","gold","clutch","sucks","expensive"}

def run(cmd): subprocess.run(cmd, check=True)
def norm(t): return re.sub(r"[^a-z0-9']", "", t.lower())
def ass_col(h, a="00"): h = h.lstrip("#"); return f"&H{a}{h[4:6]}{h[2:4]}{h[0:2]}"
def ts(t):
    t = max(0, t); h = int(t // 3600); m = int(t % 3600 // 60); return f"{h}:{m:02d}:{t % 60:05.2f}"

def probe(src):
    o = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height",
                        "-of","csv=p=0",src], capture_output=True, text=True).stdout.strip().split(",")
    return int(o[0]), int(o[1])

def face_model():
    p = os.path.join(K, "models", "face.xml")
    if os.path.exists(p): return p
    import cv2
    return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

def find_face(src, times):
    """Median face box (fractions) over sampled frames, or None."""
    import cv2
    cas = cv2.CascadeClassifier(face_model())
    if cas.empty(): return None  # no model available -> caller uses a sensible default crop
    cap = cv2.VideoCapture(src); boxes = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read()
        if not ok: continue
        H, W = fr.shape[:2]; g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        f = cas.detectMultiScale(g, 1.15, 6, minSize=(int(H * 0.08), int(H * 0.08)))
        if len(f): x, y, w, h = max(f, key=lambda b: b[2] * b[3]); boxes.append((x / W, y / H, w / W, h / H))
    if not boxes: return None
    return tuple(statistics.median(b[i] for b in boxes) for i in range(4))

def main():
    job = json.load(open(sys.argv[1]))
    T = json.load(open(os.path.join(K, "templates.json")))[job["template"]]
    ch = json.load(open(os.path.join(K, "channels.json")))[job.get("channel", "channel1")]["style"]
    src, out = job["source"], job["out"]
    work = os.path.splitext(out)[0] + "_tmp"; os.makedirs(work, exist_ok=True)
    W = json.load(open(job["words"])) if job.get("words") else []
    segs = [{"speed": 1.0, "freeze": 0, "freeze_text": "", **s} for s in job["segments"]]

    # ---- 1. render segments (speed / freeze) to uniform 1920x1080 intermediates
    offs, off, parts = [], 0.0, []
    for i, s in enumerate(segs):
        sp, fz = s["speed"], s["freeze"]; dur = (s["e"] - s["s"]) / sp
        at = []; r = sp
        while r < 0.5: at.append("atempo=0.5"); r /= 0.5
        at.append(f"atempo={r:.4f}")
        vf = (f"[0:v]trim=start={s['s']}:end={s['e']},setpts=(PTS-STARTPTS)/{sp},fps=30,"
              f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"
              + (f",tpad=stop_mode=clone:stop_duration={fz}" if fz else "") + "[v]")
        af = (f"[0:a]atrim=start={s['s']}:end={s['e']},asetpts=PTS-STARTPTS,{','.join(at)},aresample=48000"
              + (f",apad=pad_dur={fz}" if fz else "") + "[a]")
        p = f"{work}/seg{i}.mp4"
        run(["ffmpeg","-v","error","-y","-i",src,"-filter_complex",vf+";"+af,"-map","[v]","-map","[a]",
             "-t",f"{dur+fz:.3f}","-c:v","libx264","-preset","ultrafast","-crf","16","-c:a","aac","-b:a","192k",
             "-ar","48000","-ac","2",p])
        parts.append(p); offs.append(off); off += dur + fz
    total = off
    with open(f"{work}/list.txt", "w") as f: f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    joined = f"{work}/joined.mp4"
    run(["ffmpeg","-v","error","-y","-f","concat","-safe","0","-i",f"{work}/list.txt","-c","copy",joined])

    def to_out(t):
        for s, o in zip(segs, offs):
            if s["s"] - 0.05 <= t <= s["e"] + 0.05: return o + (min(max(t, s["s"]), s["e"]) - s["s"]) / s["speed"]
        return None

    # ---- 2. caption words on output timeline
    words = []
    for s, o in zip(segs, offs):
        ws = [dict(w) for w in W if w["s"] >= s["s"] - 0.02 and w["e"] <= s["e"] + 0.05]
        m = []
        for w in ws:
            if m and (re.fullmatch(r"[,.]\d+[.?!,]*", w["w"]) or re.fullmatch(r"[^\w$]+", w["w"])):
                m[-1]["w"] += w["w"]; m[-1]["e"] = w["e"]
            else: m.append(w)
        for w in m:
            raw = w["w"]; disp = re.sub(r"[.,;:]+$", "", raw)
            if norm(disp) in ("shit", "fuck", "fucking"): continue
            words.append((disp.upper(), o + (w["s"] - s["s"]) / s["speed"], o + (w["e"] - s["s"]) / s["speed"],
                          raw[-1:] in "?!."))

    # emphasis -> zooms / shakes / hits
    emph, last = [], -9
    for d, st, en, _ in words:
        if (re.search(r"[$\d]", d) or norm(d) in STRONG) and st > 0.6 and st - last > 2.0:
            emph.append(st); last = st
    shakes = [to_out(t) for t in job.get("shake_src", [])]
    shakes = [t for t in shakes if t is not None] or (emph if T["shake"] else [])

    # ---- 3. ASS overlay
    F, cs = ch["font"], T["cap_size"]
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{F},{cs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,5,50,50,0,1
Style: Box,{F},74,{ass_col(ch['box_fg'])},&H00000000,{ass_col(ch['box'])},&H00000000,-1,0,0,0,100,100,0,0,3,18,0,5,60,60,0,1
Style: Num,{F},150,{ass_col(ch['hl'])},&H00000000,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,10,5,5,60,60,0,1
Style: Wait,{F},130,&H00FFFFFF,&H00000000,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,10,5,5,60,60,0,1
Style: Chat,{F},52,&H00FFFFFF,&H00000000,&H32202020,&H00000000,-1,0,0,0,100,100,0,0,3,20,0,5,80,80,0,1
Style: Stake,{F},72,{ass_col(ch['box_fg'])},&H00000000,{ass_col(ch['box'])},&H00000000,-1,0,0,0,100,100,0,0,3,14,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    def add(st, en, style, x, y, txt, extra="", layer=1):
        ev.append(f"Dialogue: {layer},{ts(st)},{ts(en)},{style},,0,0,0,,{{\\an5\\pos({x},{y}){extra}}}{txt}")
    POP = "\\fscx70\\fscy70\\t(0,120,\\fscx100\\fscy100)"
    hook_end = min(3.2, total)
    if job.get("hook"): add(0, hook_end, "Box", 540, T["hook_y"], job["hook"], "\\fad(80,250)" + POP, 3)
    if job.get("context"): add(hook_end, min(hook_end + 2.8, total), "Box", 540, T["hook_y"], job["context"], "\\fad(120,200)", 3)
    if job.get("cta") and total > 6: add(total - 2.0, total, "Box", 540, T["hook_y"], job["cta"], "\\fad(150,0)" + POP, 3)
    sfx = []  # (file, time, volume)
    if job.get("stakes"):
        val = int(re.sub(r"\D", "", job["stakes"]) or 0); pre = job["stakes"].split(str(val)[0])[0] if val else ""
        t0 = hook_end
        for k in range(13):
            v = int(val * min(1, (k / 12) ** 0.6)); txt = f"{pre}{v:,}"
            add(t0 + k * 0.08, (t0 + (k + 1) * 0.08) if k < 12 else total, "Stake", 540, 200, txt, "", 2)
        sfx.append(("riser", max(0, t0 - 0.2), 0.5))
    for lb in job.get("labels", []):
        t = lb.get("t", to_out(lb["src"]) if "src" in lb else None)
        if t is None: continue
        sty = {"num": "Num", "chat": "Chat", "tag": "Box"}[lb.get("style", "tag")]
        y = T.get("label_y", 560)
        add(t, min(t + lb.get("dur", 2.2), total), sty, 540, y, lb["text"], "\\fad(60,150)" + POP, 2)
        sfx.append(("pop", t, 0.6))
    for s, o in zip(segs, offs):
        if s["freeze"]:
            ft = o + (s["e"] - s["s"]) / s["speed"]
            add(ft, ft + s["freeze"], "Wait", 540, 960, s["freeze_text"] or "WAIT...", POP, 4)
            sfx.append(("hit", ft, 0.8))
    if not job.get("no_captions"):
        groups, cur = [], []
        for w in words:
            if cur and (len(cur) >= T["words_per"] or w[1] - cur[-1][2] > 0.45 or cur[-1][3]
                        or sum(len(x[0]) + 1 for x in cur) + len(w[0]) > 18):
                groups.append(cur); cur = []
            cur.append(w)
        if cur: groups.append(cur)
        hl = ass_col(ch["hl"])
        for gi, g in enumerate(groups):
            nxt = groups[gi + 1][0][1] if gi + 1 < len(groups) else total
            for k, w in enumerate(g):
                st = w[1]; en = g[k + 1][1] if k + 1 < len(g) else min(w[2] + 0.2, nxt)
                en = max(en, st + 0.06)
                txt = " ".join(("{\\c" + hl + "&}" + x[0] + "{\\c&HFFFFFF&}") if j == k else x[0] for j, x in enumerate(g))
                add(st, en, "Cap", 540, T["cap_y"], txt, POP if k == 0 else "", 0)
    ass = f"{work}/overlay.ass"; open(ass, "w").write(hdr + "\n".join(ev) + "\n")

    # ---- 4. layout -> 1080x1920
    sw, shh = 1920, 1080
    L = T["layout"]
    if L == "full_face" or (L == "full" and "crop_x" not in job):
        fb = find_face(joined, [total * f for f in (0.15, 0.35, 0.55, 0.75, 0.9)]) if L == "full_face" else None
        cx = (fb[0] + fb[2] / 2) if fb else 0.5
    else:
        cx = job.get("crop_x", 0.5)
    if L in ("full", "full_face"):
        # 1920x1080 -> height 1920 -> width 3413; centre crop on cx
        sc = 1920 / shh; W2 = int(sw * sc) // 2 * 2
        x = int(min(max(cx * W2 - 540, 0), W2 - 1080))
        base = f"[0:v]scale={W2}:1920,crop=1080:1920:{x}:0[base]"
    elif L == "blur":
        base = ("[0:v]split[b][f];[b]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                "boxblur=40:6,eq=brightness=-0.15:saturation=1.1[bg];[f]scale=1080:-2[fg];[bg][fg]overlay=0:(H-h)/2[base]")
    elif L == "split":
        th = T["top_h"]; bh = 1920 - th
        fc = job.get("facecam")
        if not fc:
            fb = find_face(joined, [total * f for f in (0.2, 0.5, 0.8)])
            if fb:
                ar = 1080 / th; hh = min(1.0, fb[3] * 3.0); ww = min(1.0, hh * shh * ar / sw)
                fx = min(max(fb[0] + fb[2] / 2 - ww / 2, 0), 1 - ww); fy = min(max(fb[1] + fb[3] / 2 - hh / 2, 0), 1 - hh)
                fc = [fx, fy, ww, hh]
            else: fc = [0.3, 0, 0.4, 0.5]
        gm = job.get("game", [0, 0, 1, 1])
        def px(b): return [int(b[0] * sw), int(b[1] * shh), int(b[2] * sw) // 2 * 2, int(b[3] * shh) // 2 * 2]
        a, g = px(fc), px(gm)
        base = (f"[0:v]split[t][g];[t]crop={a[2]}:{a[3]}:{a[0]}:{a[1]},scale=1080:{th}:force_original_aspect_ratio=increase,"
                f"crop=1080:{th}[top];[g]crop={g[2]}:{g[3]}:{g[0]}:{g[1]},scale=-2:{bh},crop=1080:{bh}:(iw-1080)*{cx}:0[bot];"
                f"[top][bot]vstack,drawbox=x=0:y={th-3}:w=1080:h=6:color=#{ch['hl']}:t=fill[base]")
    # ---- 5. zoom / shake
    zt = [f"{T['zoom']}*min(1,max(0,(it-{t-0.05:.2f})/0.1))*between(it,{t-0.05:.2f},{t+0.9:.2f})" for t in emph[:14]] if T["zoom"] else []
    sh = [f"16*sin(it*90)*between(it,{t:.2f},{t+0.28:.2f})" for t in shakes[:10]]
    zb = 1.04 if sh else 1.0
    if zt or sh:
        Z = f"{zb}" + "".join("+" + z for z in zt)
        X = f"(iw-iw/zoom)/2" + "".join("+" + s for s in sh)
        Y = f"(ih-ih/zoom)/2" + "".join("+" + s.replace("sin(it*90)", "cos(it*77)") for s in sh)
        mot = f"[base]zoompan=z='{Z}':x='max(0,min(iw-iw/zoom,{X}))':y='max(0,min(ih-ih/zoom,{Y}))':d=1:s=1080x1920:fps=30[mv]"
    else:
        mot = "[base]null[mv]"
    vchain = f"{base};{mot};[mv]ass={ass}:fontsdir={FONTDIRS}[vout]"

    # ---- 6. audio: voice + sfx (+ music)
    if T.get("sfx_cut"):
        sfx += [(T["sfx_cut"], max(0, o - 0.2), 0.7) for o in offs[1:]]
    if T.get("sfx_emph"):
        sfx += [(T["sfx_emph"], t, 0.7) for t in shakes[:10]]
    if T.get("riser_before_last") and len(offs) > 1:
        sfx.append(("riser", max(0, offs[-1] - 1.2), 0.45))
    inputs = ["-i", joined]; ach = ["[0:a]volume=1.0[v0]"]; mix = ["[v0]"]
    for n, (name, t, vol) in enumerate(sfx[:24], start=1):
        inputs += ["-i", os.path.join(K, "sfx", name + ".wav")]
        ach.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,volume={vol},adelay={int(t*1000)}|{int(t*1000)}[s{n}]")
        mix.append(f"[s{n}]")
    if job.get("music"):
        n = len(sfx[:24]) + 1
        inputs += ["-stream_loop", "-1", "-i", job["music"]]
        ach.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.12,atrim=0:{total:.2f}[mu]"); mix.append("[mu]")
    ach.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,alimiter=limit=0.9,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    run(["ffmpeg","-v","error","-y",*inputs,"-filter_complex",vchain+";"+";".join(ach),"-map","[vout]","-map","[aout]",
         "-t",f"{total:.3f}","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-r","30",
         "-c:a","aac","-b:a","160k","-movflags","+faststart",out])
    print(json.dumps({"out": out, "dur": round(total, 2), "zooms": len(zt), "shakes": len(sh), "sfx": len(sfx)}))

if __name__ == "__main__":
    main()
