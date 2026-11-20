#!/usr/bin/env python3
"""Template clip engine.  Usage: engine.py job.json
job = {
  "template": "hype", "channel": "channel1",
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

def find_cam(src, total, n=14):
    """Streamer facecam = a face that stays in the same spot across the clip (game characters move).
    Returns (face_box, window_box) as fractions, or None. window_box is the face pushed out to its corner."""
    import cv2
    cas = cv2.CascadeClassifier(face_model())
    if cas.empty(): return None
    cap = cv2.VideoCapture(src); hits, got = [], 0
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, total * (i + 0.5) / n * 1000); ok, fr = cap.read()
        if not ok: continue
        got += 1; H, Wd = fr.shape[:2]; g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in cas.detectMultiScale(g, 1.1, 5, minSize=(int(H * 0.06), int(H * 0.06))):
            hits.append((x / Wd, y / H, w / Wd, h / H))
    if not hits or not got: return None
    best, members = None, []
    for c in hits:                                   # cluster by centre position
        cx, cy = c[0] + c[2] / 2, c[1] + c[3] / 2
        m = [b for b in hits if abs(b[0] + b[2] / 2 - cx) < 0.04 and abs(b[1] + b[3] / 2 - cy) < 0.05]
        if len(m) > len(members): members, best = m, c
    if len(members) < max(3, got * 0.35): return None
    fb = tuple(statistics.median(b[i] for b in members) for i in range(4))
    fx, fy, fw, fh = fb; cx, cy = fx + fw / 2, fy + fh / 2
    if fw > 0.35: return None                       # a full-screen face is a talking head, not a corner cam
    x0, x1 = (fx - 0.9 * fw, 1.0) if cx > 0.5 else (0.0, fx + fw * 1.9)
    y0, y1 = (0.0, fy + fh * 2.0) if cy < 0.5 else (fy - fh * 1.2, 1.0)
    # snap to the real cam border: an edge that stays put while the gameplay around it changes
    import numpy as np
    gx_acc, gy_acc, k = None, None, 0
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, total * (i + 0.5) / n * 1000); ok, fr = cap.read()
        if not ok: continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gxm = np.abs(np.diff(g, axis=1)); gym = np.abs(np.diff(g, axis=0))
        gx_acc = [gxm] if gx_acc is None else gx_acc + [gxm]
        gy_acc = [gym] if gy_acc is None else gy_acc + [gym]; k += 1
    if k >= 3:
        gx_acc = np.median(np.array(gx_acc), 0); gy_acc = np.median(np.array(gy_acc), 0)   # edge present in most frames
        H, Wd = gx_acc.shape[0], gy_acc.shape[1]
        r0, r1 = int(fy * H), int(min(1, fy + fh) * H)
        c0, c1 = int(fx * Wd), int(min(1, fx + fw) * Wd)
        if cx > 0.5:   # cam on the right: look for its left edge
            lo, hi = int(max(0, fx - 2.5 * fw) * Wd), int(fx * Wd)
            prof = gx_acc[r0:r1, lo:hi].mean(0)
            if len(prof) and prof.max() > 15: x0 = (lo + int(prof.argmax()) + 1) / Wd
        else:
            lo, hi = int((fx + fw) * Wd), int(min(1, fx + 3.5 * fw) * Wd)
            prof = gx_acc[r0:r1, lo:hi].mean(0)
            if len(prof) and prof.max() > 15: x1 = (lo + int(prof.argmax())) / Wd
        if cy < 0.5:   # cam at the top: look for its bottom edge
            lo, hi = int((fy + fh) * H), int(min(1, fy + 3.5 * fh) * H)
            prof = gy_acc[lo:hi, c0:c1].mean(1)
            if len(prof) and prof.max() > 15: y1 = (lo + int(prof.argmax())) / H
        else:
            lo, hi = int(max(0, fy - 2.5 * fh) * H), int(fy * H)
            prof = gy_acc[lo:hi, c0:c1].mean(1)
            if len(prof) and prof.max() > 15: y0 = (lo + int(prof.argmax()) + 1) / H
        wb = (x0, y0, x1, y1)   # snapped: crop exactly inside the border
        pad = 0.006   # step just inside the border line
        return fb, (min(1, x0 + pad) if cx > 0.5 else x0, max(0, y0 + pad) if cy >= 0.5 else y0,
                    max(0, x1 - pad) if cx <= 0.5 else x1, max(0, y1 - pad) if cy < 0.5 else y1)
    wb = (max(0, x0 - 0.01), max(0, y0 - 0.01), min(1, x1 + 0.01), min(1, y1 + 0.01))  # x0,y0,x1,y1
    return fb, wb

def game_rect(aspect, wb, sw, shh, focus_x=None):
    """Largest rect of `aspect` (w/h) in the frame that avoids the facecam window. Pixels x,y,w,h.
    focus_x (0-1): slide the rect sideways toward where the action is, staying inside the free region."""
    if wb:
        X0, Y0, X1, Y1 = wb[0] * sw, wb[1] * shh, wb[2] * sw, wb[3] * shh
        regions = [(0, 0, X0, shh), (X1, 0, sw - X1, shh), (0, 0, sw, Y0), (0, Y1, sw, shh - Y1)]
    else:
        regions = [(0, 0, sw, shh)]
    best = None
    for rx, ry, rw, rh in regions:
        if rw < sw * 0.3 or rh < shh * 0.3: continue
        w = min(rw, rh * aspect); h = w / aspect
        if best is None or w * h > best[2] * best[3]:
            x = rx + (rw - w) / 2
            if focus_x is not None: x = min(max(focus_x * sw - w / 2, rx), rx + rw - w)
            best = (x, ry + (rh - h) / 2, w, h)
    if best is None:                                 # cam too big to avoid: plain centre crop
        w = min(sw, shh * aspect); h = w / aspect; best = ((sw - w) / 2, (shh - h) / 2, w, h)
    x, y, w, h = best
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    return int(x), int(y), w, h

def action_x(src, total, cam=None, n=24):
    """Where the action is, left to right (0-1): centre of frame-to-frame change, facecam ignored.
    Pulled halfway back to the middle so a tall crop never loses the player character (usually centred)."""
    import cv2, numpy as np
    cap = cv2.VideoCapture(src); prev = None; acc = None
    for i in range(n * 2):
        cap.set(cv2.CAP_PROP_POS_MSEC, total * (i // 2 + 0.5) / n * 1000 + (i % 2) * 150)
        ok, fr = cap.read()
        if not ok: prev = None; continue
        g = cv2.cvtColor(cv2.resize(fr, (192, 108)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if i % 2 and prev is not None:
            d = np.abs(g - prev)
            if cam:
                x0, y0, x1, y1 = cam
                d[int(y0 * 108):int(np.ceil(y1 * 108)), int(x0 * 192):int(np.ceil(x1 * 192))] = 0
            col = d.mean(0); acc = col if acc is None else acc + col
        prev = g
    if acc is None or acc.sum() < 1e-6: return 0.5
    xs = (np.arange(192) + 0.5) / 192
    cx = float((acc * xs).sum() / acc.sum())
    return 0.5 + (cx - 0.5) * 0.5

SWEARS = re.compile(r"^(mother)?f+u+c*k+\w*$|^sh+i+t+\w*$|^bitch\w*$|^dick\w*$|^cocks?$|^puss(y|ies)$|^cunts?$|"
                    r"^whores?$|^sluts?$|^fag\w*$|^nigg\w*$|^asshole\w*$|^bastards?$|^damn$|^goddamn\w*$")

def mask_swear(word):
    """F**K style: keeps captions readable without showing the word (ad-friendlier)."""
    core = re.sub(r"[^A-Za-z]", "", word)
    if not core or not SWEARS.match(core.lower()): return word
    return core[0] + "*" * (len(core) - 2) + core[-1] if len(core) > 2 else core[0] + "*"

def brightness(src, total):
    import cv2
    cap = cv2.VideoCapture(src); vals = []
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        cap.set(cv2.CAP_PROP_POS_MSEC, total * f * 1000); ok, fr = cap.read()
        if ok: vals.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean())
    return statistics.mean(vals) if vals else 100

def main():
    job = json.load(open(sys.argv[1]))
    T = json.load(open(os.path.join(K, "templates.json")))[job["template"]]
    ch = json.load(open(os.path.join(K, "channels.json")))[job.get("channel", "channel1")]["style"]
    src, out = job["source"], job["out"]
    work = os.path.splitext(out)[0] + "_tmp"; os.makedirs(work, exist_ok=True)
    W = json.load(open(job["words"])) if job.get("words") else []
    segs = [{"speed": 1.0, "freeze": 0, "freeze_text": "", **s} for s in job["segments"]]

    # ---- 0. template-driven structure (only when the job gives one plain segment)
    if len(segs) == 1 and not job.get("raw_segments"):
        s0 = dict(segs[0]); L0 = s0["e"] - s0["s"]
        if T.get("speed_ramp") and L0 > 12:            # fast build-up, real-time finish
            cut = s0["s"] + L0 * 0.6
            segs = [{**s0, "e": cut, "speed": T["speed_ramp"]}, {**s0, "s": cut}]
        elif T.get("auto_freeze") and L0 > 10:          # freeze + "WAIT FOR IT" right before the payoff
            cut = s0["e"] - min(4.0, L0 * 0.25)
            segs = [{**s0, "e": cut, "freeze": T["auto_freeze"], "freeze_text": T.get("freeze_text", "WAIT...")},
                    {**s0, "s": cut, "freeze": 0, "freeze_text": ""}]
        if T.get("auto_replay"):                        # slow-mo replay of the payoff
            last = segs[-1]; r = min(T["auto_replay"], last["e"] - last["s"])
            segs.append({"s": last["e"] - r, "e": last["e"], "speed": 0.5, "freeze": 0, "freeze_text": "",
                         "nocap": True, "replay": True})
    if T.get("auto_stakes") and not job.get("stakes"):  # big $ number said in the clip -> counter
        lo, hi = segs[0]["s"], segs[-1]["e"]
        for w in W:
            if lo <= w["s"] <= hi and re.search(r"\$\s?\d", w["w"]):
                job["stakes"] = re.sub(r"[^\d$,]", "", w["w"]); break

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
        if s.get("nocap"): continue
        ws = [dict(w) for w in W if w["s"] >= s["s"] - 0.02 and w["e"] <= s["e"] + 0.05]
        m = []
        for w in ws:
            if m and (re.fullmatch(r"[,.]\d+[.?!,]*", w["w"]) or re.fullmatch(r"[^\w$]+", w["w"])):
                m[-1]["w"] += w["w"]; m[-1]["e"] = w["e"]
            else: m.append(w)
        upper = T.get("cap_upper", True)
        for w in m:
            raw = w["w"].strip(); disp = re.sub(r"[,;:]+$", "", raw) if not upper else re.sub(r"[.,;:]+$", "", raw)
            if not norm(disp): continue
            disp = mask_swear(disp)
            st, en = o + (w["s"] - s["s"]) / s["speed"], o + (w["e"] - s["s"]) / s["speed"]
            en = min(en, st + 1.0)                   # transcripts stretch a word over silence: never hold one > 1 s
            if words and (st < words[-1][1] + 0.03 or (norm(disp) == norm(words[-1][0]) and st - words[-1][2] < 0.15)):
                continue                             # transcript glitch: duplicate / overlapping word
            words.append((disp.upper() if upper else disp, st, en, raw[-1:] in "?!."))

    # emphasis -> zooms / shakes / hits
    emph, last = [], -9
    for d, st, en, _ in words:
        if (re.search(r"[$\d]", d) or norm(d) in STRONG) and st > 0.6 and st - last > 2.0:
            emph.append(st); last = st
    shakes = [to_out(t) for t in job.get("shake_src", [])]
    shakes = [t for t in shakes if t is not None] or (emph if T["shake"] else [])

    # ---- 3. ASS overlay
    F, cs = T.get("cap_font", ch["font"]), T["cap_size"]
    co = T.get("cap_outline", 7); MF, MS = T.get("meme_font", "Anton"), T.get("meme_size", 96)
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{F},{cs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{co},3,5,50,50,0,1
Style: Meme,{MF},{MS},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,9,2,5,40,40,0,1
Style: Box,{F},64,{ass_col(ch['box_fg'])},&H00000000,{ass_col(ch['box'])},&H00000000,-1,0,0,0,100,100,0,0,3,18,0,5,60,60,0,1
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
    HM = T.get("hook_mode", "box")
    if job.get("hook") and HM == "box":
        add(0, hook_end, "Box", 540, T["hook_y"], job["hook"], "\\fad(80,250)" + POP, 3)
    if job.get("hook") and HM == "seam":            # hook box on the facecam/gameplay seam, first seconds only
        hs = min(T.get("hook_secs", 3.2), total)
        add(0, hs, "Box", 540, T["top_h"], job["hook"], "\\fad(80,250)" + POP, 3)
    if job.get("hook") and HM == "meme":            # one meme line on the seam for the whole clip
        import textwrap
        line = re.sub(r"\s+", " ", job["hook"].replace("\\N", " ")).strip().upper()
        fits = T.get("meme_fit") or [[MS, T.get("meme_wrap", 20)]]
        for size, wrap in fits:                      # shrink the font until it fits in 2 lines; never drop words
            parts = textwrap.wrap(line, wrap)
            if len(parts) <= 2: break
        add(0, total, "Meme", 540, T["top_h"], "\\N".join(parts), f"\\fad(0,0)\\fs{size}", 3)
    if job.get("context") and HM == "box": add(hook_end, min(hook_end + 2.8, total), "Box", 540, T["hook_y"], job["context"], "\\fad(120,200)", 3)
    if job.get("cta") and T.get("cta", True) and total > 6:
        add(total - 2.0, total, "Box", 540, T.get("cta_y", T.get("hook_y", 300)), job["cta"], "\\fad(150,0)" + POP, 3)
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
    for s, o in zip(segs, offs):
        if s.get("replay"):
            add(o, min(o + 1.6, total), "Box", 540, T.get("label_y", 560), "SLOW-MO REPLAY", "\\fad(60,200)" + POP, 4)
            sfx.append(("whoosh", max(0, o - 0.15), 0.8))
    if not (job.get("no_captions") or T.get("no_captions")):
        groups, cur = [], []
        for w in words:
            if cur and (len(cur) >= T["words_per"] or w[1] - cur[-1][2] > 0.45 or w[1] - cur[-1][1] > 1.0
                        or cur[-1][3] or sum(len(x[0]) + 1 for x in cur) + len(w[0]) > 18):
                groups.append(cur); cur = []
            cur.append(w)
        if cur: groups.append(cur)
        hlc = T.get("cap_hl", ch["hl"]) if "cap_hl" in T else ch["hl"]
        hl = ass_col(hlc) if hlc else None
        for gi, g in enumerate(groups):
            nxt = groups[gi + 1][0][1] if gi + 1 < len(groups) else total
            for k, w in enumerate(g):
                st = w[1]; en = g[k + 1][1] if k + 1 < len(g) else min(w[2] + 0.2, nxt)
                en = max(min(en, nxt, w[2] + 0.35, st + 1.0), st + 0.06)   # clear the screen in silences
                txt = " ".join(("{\\c" + hl + "&}" + x[0] + "{\\c&HFFFFFF&}") if (hl and j == k) else x[0]
                               for j, x in enumerate(g))
                add(st, en, "Cap", 540, T["cap_y"], txt, POP if k == 0 else "", 0)
    ass = f"{work}/overlay.ass"; open(ass, "w").write(hdr + "\n".join(ev) + "\n")

    # ---- 4. layout -> 1080x1920
    sw, shh = 1920, 1080
    L = T["layout"]
    lift = ""                                        # dark night scenes get lifted so the hook can land
    br = brightness(joined, total)
    if br < 55: lift = ",eq=gamma=1.45:brightness=0.05:contrast=1.05"
    elif br < 80: lift = ",eq=gamma=1.22:brightness=0.03"
    cam = None
    if L in ("wide", "zoom43", "split", "portrait"):
        cam = None if (job.get("facecam") or job.get("cam")) else find_cam(joined, total)
        if job.get("cam"):                           # found on the whole source video (steadier than one short clip)
            cam = (tuple(job["cam"][0]), tuple(job["cam"][1]))
        if job.get("facecam"):
            f = job["facecam"]; cam = (tuple(f), (f[0], f[1], f[0] + f[2], f[1] + f[3]))
        print(f"   facecam: {'found at ' + str([round(float(v), 3) for v in cam[1]]) if cam else 'none'}  brightness {br:.0f}", flush=True)
    if L == "split" and not cam:                     # no streamer cam -> fall back to the hype layout
        hy = json.load(open(os.path.join(K, "templates.json")))["hype"]
        T = {**T, **{k: hy[k] for k in ("layout", "game_ar", "game_y", "cap_y")}, "hook_y": hy["hook_y"]}
        if T.get("hook_mode") in ("seam", "meme", "none"): T["hook_mode"] = "box"
        L = "portrait"
        print("   no facecam found -> using the hype layout", flush=True)
    BGS = {"dark": "eq=brightness=-0.12:saturation=1.2",
           "cool": "eq=brightness=-0.3:saturation=0.9,colorbalance=bs=0.12:rs=-0.08"}
    bgf = (f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=45,"
           f"{BGS.get(T.get('bg', 'dark'), BGS['dark'])}")
    if L == "full_face" or (L == "full" and "crop_x" not in job):
        fb = find_face(joined, [total * f for f in (0.15, 0.35, 0.55, 0.75, 0.9)]) if L == "full_face" else None
        cx = (fb[0] + fb[2] / 2) if fb else 0.5
    else:
        cx = job.get("crop_x", 0.5)
    if L in ("full", "full_face"):
        sc = 1920 / shh; W2 = int(sw * sc) // 2 * 2
        x = int(min(max(cx * W2 - 540, 0), W2 - 1080))
        base = f"[0:v]scale={W2}:1920,crop=1080:1920:{x}:0{lift}[base]"
    elif L == "blur":
        base = ("[0:v]split[b][f];[b]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                f"boxblur=40:6,eq=brightness=-0.15:saturation=1.1[bg];[f]scale=1080:-2{lift}[fg];[bg][fg]overlay=0:(H-h)/2[base]")
    elif L in ("wide", "zoom43", "portrait"):
        ar = T.get("game_ar") or {"wide": 16 / 9, "zoom43": 4 / 3, "portrait": 0.8}[L]
        fx = action_x(joined, total, cam[1] if cam else None) if ar < 1.5 else None
        gx, gy, gw, gh = game_rect(ar, cam[1] if cam else None, sw, shh, fx)
        oh = int(1080 / ar) // 2 * 2
        base = (f"[0:v]split[b][f];[b]{bgf}[bg];[f]crop={gw}:{gh}:{gx}:{gy},scale=1080:{oh}:flags=lanczos,"
                f"unsharp=5:5:0.7{lift}[fg];[bg][fg]overlay=0:{T.get('game_y', (1920 - oh) // 2)}[base]")
    elif L == "split":
        th = T["top_h"]; bh = 1920 - th
        fb, wb = cam
        # top: crop the cam window to the panel shape, centred on the face
        WX0, WY0, WX1, WY1 = wb[0] * sw, wb[1] * shh, wb[2] * sw, wb[3] * shh
        ar = 1080 / th; ww, hh = WX1 - WX0, WY1 - WY0
        if ww / hh > ar: cw, chh = hh * ar, hh
        else: cw, chh = ww, ww / ar
        fcx, fcy = (fb[0] + fb[2] / 2) * sw, (fb[1] + fb[3] / 2) * shh
        cxp = min(max(fcx - cw / 2, WX0), WX1 - cw); cyp = min(max(fcy - chh * 0.45, WY0), WY1 - chh)
        a = [int(cxp), int(cyp), int(cw) // 2 * 2, int(chh) // 2 * 2]
        gx, gy, gw, gh = game_rect(1080 / bh, wb, sw, shh)
        div = (f",drawbox=x=0:y={th - T.get('divider_h', 8) // 2}:w=1080:h={T.get('divider_h', 8)}:color=#{T['divider']}:t=fill"
               if T.get("divider") else "")
        base = (f"[0:v]split[t][g];[t]crop={a[2]}:{a[3]}:{a[0]}:{a[1]},scale=1080:{th}:flags=lanczos,unsharp=5:5:0.6[top];"
                f"[g]crop={gw}:{gh}:{gx}:{gy},scale=1080:{bh}:flags=lanczos,unsharp=5:5:0.6{lift}[bot];"
                f"[top][bot]vstack{div}[base]")
    # ---- 5. zoom / shake
    zt = [f"{T['zoom']}*min(1,max(0,(it-{t-0.05:.2f})/0.1))*between(it,{t-0.05:.2f},{t+0.9:.2f})" for t in emph[:14]] if T["zoom"] else []
    if T.get("push"): zt.append(f"{T['push']}*it/{total:.2f}")   # slow cinematic push-in
    if T.get("pulse"): zt.append(f"{T['pulse']}*abs(sin(it*3.2))")  # beat-like breathing zoom
    sh = [f"16*sin(it*90)*between(it,{t:.2f},{t+0.28:.2f})" for t in shakes[:10]]
    zb = 1.04 if sh else 1.0
    if zt or sh:
        Z = f"{zb}" + "".join("+" + z for z in zt)
        X = f"(iw-iw/zoom)/2" + "".join("+" + s for s in sh)
        Y = f"(ih-ih/zoom)/2" + "".join("+" + s.replace("sin(it*90)", "cos(it*77)") for s in sh)
        mot = f"[base]zoompan=z='{Z}':x='max(0,min(iw-iw/zoom,{X}))':y='max(0,min(ih-ih/zoom,{Y}))':d=1:s=1080x1920:fps=30[mv]"
    else:
        mot = "[base]null[mv]"
    # ---- 5b. look: colour grade, vignette, letterbox, flashes, progress bar
    GRADES = {
        "punchy":      "eq=contrast=1.12:saturation=1.3:gamma=0.97,unsharp=5:5:0.6",
        "teal_orange": "colorbalance=rs=-0.05:bs=0.08:rh=0.08:bh=-0.08,eq=contrast=1.1:saturation=1.15",
        "warm":        "colorbalance=rm=0.06:bm=-0.05:rh=0.05,eq=saturation=1.2:contrast=1.06",
        "neon":        "eq=saturation=1.45:contrast=1.08,colorbalance=rs=0.05:bs=0.08",
    }
    look = []
    if T.get("grade") in GRADES: look.append(GRADES[T["grade"]])
    if T.get("grain"): look.append("noise=alls=6:allf=t")
    if T.get("vignette"): look.append("vignette=PI/5")
    if T.get("letterbox"):
        look.append("drawbox=x=0:y=0:w=iw:h=150:color=black:t=fill,drawbox=x=0:y=ih-150:w=iw:h=150:color=black:t=fill")
    if T.get("flash") == "police":
        for t in shakes[:10]:
            look.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.25:t=fill:enable='between(t,{t:.2f},{t+0.12:.2f})'")
            look.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=blue@0.25:t=fill:enable='between(t,{t+0.12:.2f},{t+0.24:.2f})'")
    elif T.get("flash") == "white":
        for t in (shakes or emph)[:10]:
            look.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.35:t=fill:enable='between(t,{t:.2f},{t+0.07:.2f})'")
    lk = ("," + ",".join(look)) if look else ""
    vchain = f"{base};{mot};[mv]format=yuv420p{lk},ass={ass}:fontsdir={FONTDIRS}"
    if T.get("progress"):
        vchain += (f"[cap];color=c=0x{T.get('bar', ch['hl'])}:s=1080x12:r=30[pb];"
                   f"[cap][pb]overlay=x='-w+w*t/{total:.3f}':y=0:shortest=1[vout]")
    else:
        vchain += "[vout]"

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
