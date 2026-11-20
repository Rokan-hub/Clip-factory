#!/usr/bin/env python3
"""Let Gemini WATCH the video (not just read a transcript) to find clip moments, then tighten every cut locally.

Why: gameplay has little useful speech, so a transcript/loudness picker lands on credits, logo cards and dead time.
Watching the video lets the model find the real payoff and the exact in/out points.

  pick(src, count, mn, mx, templates, work)  -> list of moments {start,end,payoff,score,hook,reason,template}
  refine(src, moments, mn)                   -> trims static lead-in / dead tail, drops mostly-static clips

Needs GEMINI_API_KEY. Any failure raises, and run_pipeline falls back to the old transcript picker.
Env: CF_WATCH=0 turns it off, CF_WATCH_BUDGET (seconds, default 1500) caps total time spent watching,
     GEMINI_BASE overrides the API host (used by tests).
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error
import llm

BASE = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
CHUNK, OVERLAP, MAX_TOTAL = 480, 12, 5400      # watch 8-minute pieces, up to 90 minutes in total
PER_CHUNK = 4
BANNED_KINDS = {"cutscene", "trailer", "menu"}   # Rockstar removes isolated cutscenes; trailers/menus never hold viewers

def log(*a): print("  [watch]", *a, file=sys.stderr, flush=True)

def duration(src):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                       capture_output=True, text=True).stdout.strip()
    return float(o or 0)

def chunks(total):
    total = min(total, MAX_TOTAL); out, s = [], 0.0
    while s < total - 1:
        out.append((s, min(CHUNK, total - s)))
        if s + CHUNK >= total: break
        s += CHUNK - OVERLAP
    return out

def make_proxy(src, start, length, dst):
    """Small copy of one piece: 360p, 4 fps, mono audio. Gemini samples video at 1 fps anyway."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", src,
                    "-map", "0:v:0", "-map", "0:a:0?", "-vf", "fps=4,scale=-2:360",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "33", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "32k", "-movflags", "+faststart", dst], check=True)

def _http(method, url, key, body=None, headers=None, timeout=120):
    h = {"x-goog-api-key": key, "User-Agent": "clip-factory", **(headers or {})}
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    if isinstance(body, dict): h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.headers, (json.loads(raw) if raw.strip() else {})

def upload(path, key):
    """Gemini Files API, resumable upload. Returns (file name, uri)."""
    size = os.path.getsize(path)
    hd, _ = _http("POST", f"{BASE}/upload/v1beta/files", key, {"file": {"display_name": os.path.basename(path)}},
                  {"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                   "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": "video/mp4"})
    url = hd.get("X-Goog-Upload-URL") or hd.get("x-goog-upload-url")
    if not url: raise RuntimeError("upload: no upload URL returned")
    with open(path, "rb") as f: blob = f.read()
    _, d = _http("POST", url, key, blob, {"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                          "X-Goog-Upload-Command": "upload, finalize"}, timeout=300)
    f = d.get("file", d); name, uri = f["name"], f["uri"]
    for _ in range(60):                               # wait until Google finished processing the video
        st = f.get("state", "ACTIVE")
        if st == "ACTIVE": return name, uri
        if st == "FAILED": raise RuntimeError("upload: Google could not process the video")
        time.sleep(3); _, f = _http("GET", f"{BASE}/v1beta/{name}", key)
    raise RuntimeError("upload: processing timed out")

def delete(name, key):
    try: _http("DELETE", f"{BASE}/v1beta/{name}", key, timeout=30)
    except Exception: pass

def parse_t(x):
    if isinstance(x, (int, float)): return float(x)
    parts = [float(p) for p in re.findall(r"\d+(?:\.\d+)?", str(x))]
    if not parts: raise ValueError(f"bad time {x!r}")
    t = 0.0
    for p in parts[-3:]: t = t * 60 + p
    return t

def build_prompt(mn, mx, templates, n=PER_CHUNK):
    tpl = ""
    if templates:
        tpl = ("\nChoose the best editing preset for each moment (exact name):\n" +
               "\n".join(f"- {k}: {v}" for k, v in templates.items()) + '\nPut it in "template".\n')
    return f"""You are the editor of a YouTube Shorts / Instagram Reels channel. You are WATCHING a segment of a longer video
(game footage, a stream, or other). Find up to {n} moments worth cutting into standalone vertical clips.
All times are measured from the start of THIS segment, written MM:SS.s.

Score each moment 1-10 on four pillars: HOOK (stops the scroll in 1-3 s), RETENTION (no dead time, keeps people to the end),
SATISFACTION (a clear payoff: funny line, crash, reveal, win/fail, big reaction) and HUMAN PSYCHOLOGY (curiosity, emotion, shock, relatability).

HARD RULES
- NEVER pick studio logos, "presents" cards, opening/closing credits, title cards, menus, settings or loading screens, black screens,
  still images/slides, or stretches where nothing happens. If a good moment follows such a card, start AFTER the card.
- NEVER pick story cutscenes (scripted non-playable scenes: characters talking in a staged scene, no HUD/controls, camera
  cuts like a film), official trailer or promo footage, in-game TV/radio/movies, or phone, map, inventory, tutorial or pause screens.
  The publisher takes down clips of isolated cutscenes, and trailers were seen by everyone already. Only real PLAYED gameplay
  (or, for non-game videos, the creator's own footage) counts. Report what the moment is in "kind".
- The first 1 second must already show action or a reaction. Never open on a still screen, a menu, or someone looking at a phone.
- Every moment needs a clear PAYOFF that is ON SCREEN (the hit lands, the crash, the reveal, the punchline, the big reaction).
  Give its exact time as "payoff". If the payoff happens off screen or after the moment ends, skip the moment.
- "start" is at most 1.5 s BEFORE the action or line that sets up the payoff. Open on the strongest beat. Never start mid-word.
- "end" is at most 0.8 s AFTER the payoff. Never trail off into dead time.
- Length {mn}-{mx} seconds; 12-20 s is best. Cut waiting, walking and menus out by picking a tighter window. Moments must not overlap.
- "hook": a punchy on-screen headline, max 6 words, that describes ONLY what is actually visible in this moment.
  Do not invent or exaggerate (no "zombie" for a bloody man, no "after robbery" if no robbery is shown). Honest curiosity wins:
  viewers who feel tricked swipe away and the platform shows the channel's next clips to fewer people.
- "facecam": true if a streamer webcam is visible. "reaction": 0-10, how big and visible the streamer's facial reaction is
  in this moment (0 = no cam or a flat face, 10 = jaw drop / crying laughing / rage).
- "reason": one short line.
{tpl}
Return ONLY a JSON array, best first, each item exactly:
{{"start": "MM:SS.s", "end": "MM:SS.s", "payoff": "MM:SS.s", "score": number, "hook": string, "reason": string, "template": string,
  "kind": "gameplay" | "cutscene" | "trailer" | "menu" | "other", "facecam": boolean, "reaction": number}}
If nothing in this segment is usable, return [].
"""

def generate(file_uri, prompt, key):
    """Ask Gemini about one uploaded video piece. Tries the model names in order."""
    last = None
    for m in llm.models("gemini"):
        if ("gemini", m) in llm.BAD: continue
        body = {"contents": [{"parts": [{"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}},
                                        {"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json",
                                     "mediaResolution": "MEDIA_RESOLUTION_LOW"}}
        for attempt in range(3):
            try:
                _, d = _http("POST", f"{BASE}/v1beta/models/{m}:generateContent", key, body, timeout=300)
                txt = d["candidates"][0]["content"]["parts"][0]["text"]
                llm.LOG.append(("watch", "gemini", True, m)); return llm.parse_json(txt)
            except urllib.error.HTTPError as e:
                note = f"{m} HTTP {e.code}: {e.read().decode(errors='ignore')[:140]}"
                llm.LOG.append(("watch", "gemini", False, note)); last = note
                if e.code in (400, 404): llm.BAD.add(("gemini", m)); break          # model name or request not accepted
                if e.code in (429, 500, 502, 503) and attempt < 2: time.sleep(20 * (attempt + 1)); continue
                raise RuntimeError(note)
            except Exception as e:
                last = f"{m} {str(e)[:140]}"; llm.LOG.append(("watch", "gemini", False, last))
                if attempt < 1: time.sleep(5); continue
                break
    raise RuntimeError(last or "no Gemini model worked")

def overlap(a, b):
    i = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return i / max(1e-6, min(a["end"] - a["start"], b["end"] - b["start"]))

def pick(src, count, mn, mx, templates, work):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key: raise RuntimeError("no GEMINI_API_KEY")
    total = duration(src)
    if total <= 0: raise RuntimeError("cannot read video length")
    os.makedirs(work, exist_ok=True)
    budget = float(os.environ.get("CF_WATCH_BUDGET", "1500")); t0 = time.time()
    cand, ok_chunks, parts = [], 0, chunks(total)
    log(f"watching {total / 60:.1f} min of video in {len(parts)} piece(s)")
    for i, (s, ln) in enumerate(parts):
        if time.time() - t0 > budget: log("time budget reached, using what was found"); break
        proxy = os.path.join(work, f"watch{i}.mp4"); name = None
        try:
            make_proxy(src, s, ln, proxy)
            name, uri = upload(proxy, key)
            items = generate(uri, build_prompt(mn, mx, templates), key)
            if isinstance(items, dict): items = items.get("moments") or items.get("clips") or []
            ok_chunks += 1
            for it in items:
                try:
                    a, b = parse_t(it["start"]) + s, parse_t(it["end"]) + s
                    pay = parse_t(it["payoff"]) + s if it.get("payoff") not in (None, "") else None
                except Exception: continue
                if b - a < max(4, mn * 0.6) or b - a > mx * 1.3 or b > total + 0.5: continue
                kind = str(it.get("kind") or "gameplay").lower()
                if kind in BANNED_KINDS:
                    log(f"skip {a:.0f}-{b:.0f}s: {kind}"); continue
                try: react = float(it.get("reaction") or 0)
                except Exception: react = 0.0
                cand.append({"start": round(a, 2), "end": round(min(b, total), 2), "payoff": pay,
                             "score": float(it.get("score") or 0), "hook": it.get("hook", ""),
                             "reason": it.get("reason", ""), "template": it.get("template"),
                             "kind": kind, "facecam": bool(it.get("facecam")), "reaction": react})
            log(f"piece {i + 1}/{len(parts)}: {len(items)} moment(s)")
        except Exception as e:
            log(f"piece {i + 1}/{len(parts)} failed: {str(e)[:160]}")
            llm.LOG.append(("watch", "gemini", False, f"piece {i + 1}: {str(e)[:140]}"))
        finally:
            if name: delete(name, key)
            try: os.remove(proxy)
            except OSError: pass
        if i + 1 < len(parts): time.sleep(4)          # stay under the free per-minute limits
    if not ok_chunks: raise RuntimeError("Gemini could not watch any part of the video")
    out = []
    for m in sorted(cand, key=lambda c: -c["score"]):   # best first, drop duplicates from the piece overlaps
        if any(overlap(m, o) > 0.3 for o in out): continue
        out.append(m)
    return out

# ---------- local tightening ----------
FLAT = 0.52     # share of the frame that is one flat tone -> credits / logo / title card / black or menu screen
                # (measured: cards 0.55-1.0, gameplay and cutscenes 0.1-0.46)

def motion_profile(src, s, e, cam=None, step=0.25):
    """Every `step` s inside [s, e]: (time, frame-to-frame change, flat-tone share), facecam window ignored."""
    import cv2, numpy as np
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, s) * 1000)
    prof, prev, mask, i = [], None, None, 0
    every = max(1, int(round(fps * step))); t0 = max(0, s)
    while True:
        t = t0 + i / fps
        if t > e or not cap.grab(): break
        i += 1
        if (i - 1) % every: continue
        ok, fr = cap.retrieve()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (128, 72)), cv2.COLOR_BGR2GRAY)
        if mask is None:
            mask = np.ones(g.shape, bool)
            if cam:
                x0, y0, x1, y1 = cam
                mask[int(y0 * 72):int(np.ceil(y1 * 72)), int(x0 * 128):int(np.ceil(x1 * 128))] = False
        v = g[mask]; flat = float(np.bincount(v // 12, minlength=22).max() / v.size)
        gf = g.astype(np.float32)
        if prev is not None: prof.append((t, float(np.abs(gf - prev)[mask].mean()), flat))
        prev = gf
    return prof

def refine(src, moments, mn=8, cam=None, still=0.6):
    """Cut a lead-in of credits/cards/still frames and a dead tail, drop clips that are mostly cards or static."""
    out = []
    for m in moments:
        s, e = m["start"], m["end"]
        try: prof = motion_profile(src, s - 1.0, e + 0.5, cam)
        except Exception as ex:
            log(f"motion check skipped: {ex}"); out.append(m); continue
        if len(prof) < 8: out.append(m); continue
        live = [(t, d > still and f < FLAT) for t, d, f in prof]     # a real picture that is actually moving
        inside = [a for t, a in live if s <= t <= e]
        if inside and sum(inside) / len(inside) < 0.35:
            log(f"drop {s:.0f}-{e:.0f}s: mostly cards/static"); continue
        ns, ne = s, e
        first = next((t for k, (t, a) in enumerate(live) if t >= s - 0.5 and a and all(x for _, x in live[k:k + 3])), None)
        if first is not None and first - s > 0.8: ns = max(0.0, first - 0.3)    # never open on a still screen
        pay = m.get("payoff")
        if pay and pay < e and e - pay > 1.0: ne = pay + 0.8          # end just after the payoff
        else:
            last = next((t for t, a in reversed(live) if a and t <= e), None)
            if last is not None and e - last > 1.5: ne = last + 0.8
        if ne - ns < max(4, mn * 0.6): out.append(m); continue          # trimming would leave too little: keep as is
        if (ns, ne) != (s, e): log(f"tightened {s:.1f}-{e:.1f} -> {ns:.1f}-{ne:.1f}")
        out.append({**m, "start": round(ns, 2), "end": round(ne, 2)})
    return out
