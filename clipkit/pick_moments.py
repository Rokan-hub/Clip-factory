#!/usr/bin/env python3
"""Pick the best clip moments from a word-level transcript.

Usage: pick_moments.py words.json moments.json [--count 5] [--min 15] [--max 45]

Uses the free Gemini API (key in env GEMINI_API_KEY, from Google AI Studio) scored on Sam's 4 pillars.
If there is no key, or the API fails, a free local keyword heuristic is used instead so the run never dies.
Env: GEMINI_MODEL (default gemini-flash-latest)
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

STRONG = {"never","worst","best","insane","crazy","impossible","secret","won","lost","paid","free","million",
          "thousand","dollars","shocked","scared","dropped","wait","what","wow","dead","killed","crashed",
          "cops","police","boom","bro","clutch","expensive","nobody","literally","ruined","destroyed"}

RUBRIC = """You are an expert short-form video editor picking clips from a transcript for YouTube Shorts / Instagram Reels.
Judge every candidate moment on these 4 pillars (score each 1-10):
1. HOOK: the first 1-3 seconds must stop the scroll (a bold claim, shock, question, big number, conflict).
2. RETENTION: no dead air, an open loop, fast pacing, keeps people to the end.
3. SATISFACTION: it pays off (punchline, answer, reveal, win/fail) so the viewer feels it was worth it and shares/rewatches.
4. HUMAN PSYCHOLOGY: curiosity, controversy, emotion, relatability, status, fear of missing out.
"""

def lines(words, gap=0.7, maxw=22):
    """Group words into timestamped lines for the prompt."""
    out, cur = [], []
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > gap or len(cur) >= maxw or re.search(r"[.?!]$", cur[-1]["w"])):
            out.append(cur); cur = []
        cur.append(w)
    if cur: out.append(cur)
    return [f"[{c[0]['s']:.1f}] " + " ".join(x["w"] for x in c) for c in out]

def call_gemini(prompt, key, model):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}}).encode()
    for attempt in range(4):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 503) and attempt < 3:
                time.sleep(15 * (attempt + 1)); continue
            raise RuntimeError(f"Gemini HTTP {e.code}: {msg}")
    raise RuntimeError("Gemini retries exhausted")

def snap(words, s, e):
    """Snap a start/end to word boundaries."""
    ws = [w for w in words if w["e"] > s - 0.05 and w["s"] < e + 0.05]
    if not ws: return None
    return round(ws[0]["s"] - 0.1, 2) if ws[0]["s"] > 0.1 else 0.0, round(ws[-1]["e"] + 0.25, 2)

def clean_hook(h):
    h = re.sub(r"[^\w$?!',.%: -]", "", h or "").strip()
    return " ".join(h.split()[:9])

def gemini_pick(words, count, mn, mx, key, model):
    tr = "\n".join(lines(words))
    if len(tr) > 400_000: tr = tr[:400_000]
    prompt = f"""{RUBRIC}
Pick the {count} best NON-OVERLAPPING moments. Rules:
- Each moment is between {mn} and {mx} seconds long.
- Start at the beginning of a sentence/line, end right after the payoff (never mid-sentence).
- Prefer moments whose opening line is the strongest line (we open with it).
- "hook" = a punchy on-screen headline of at most 8 words, in your own words, no emojis, no hashtags.
- Times are in seconds from the transcript timestamps.
Return ONLY a JSON array, best first, each item exactly:
{{"start": number, "end": number, "score": number (1-10 overall), "hook": string, "reason": string (one short line)}}

TRANSCRIPT (format: [seconds] text):
{tr}
"""
    txt = call_gemini(prompt, key, model)
    arr = json.loads(txt)
    if isinstance(arr, dict): arr = arr.get("moments") or arr.get("clips") or []
    return arr

def heuristic_pick(words, count, mn, mx):
    """Free fallback: slide a window and score keywords, numbers, questions."""
    if not words: return []
    starts = [0] + [i + 1 for i, w in enumerate(words[:-1]) if re.search(r"[.?!]$", w["w"])]
    cands = []
    for si in starts:
        s = words[si]["s"]; seg = [w for w in words[si:] if w["e"] <= s + mx]
        if not seg or seg[-1]["e"] - s < mn: continue
        # end at the sentence end closest to the middle of the allowed length
        ends = [i for i, w in enumerate(seg) if re.search(r"[.?!]$", w["w"]) and w["e"] - s >= mn]
        if not ends: continue
        mid = (mn + mx) / 2
        ei = min(ends, key=lambda i: abs((seg[i]["e"] - s) - mid))
        last = seg[ei]
        text = " ".join(w["w"] for w in seg[:ei + 1]); low = text.lower()
        sc = 3 * len(re.findall(r"\$\s?\d", text)) + 2 * sum(1 for t in re.findall(r"[a-z']+", low) if t in STRONG)
        sc += 2 * text.count("?") + 1.5 * text.count("!")
        first = " ".join(w["w"] for w in seg[:12]).lower()
        sc += 3 * sum(1 for t in re.findall(r"[a-z']+", first) if t in STRONG)
        sc = sc / ((last["e"] - s) ** 0.5)
        first_sent = []
        for w in seg[:12]:
            first_sent.append(w["w"])
            if re.search(r"[.?!]$", w["w"]): break
        hook = " ".join(first_sent[:9])
        cands.append({"start": s, "end": last["e"], "score": sc, "hook": hook, "reason": "keyword heuristic"})
    return sorted(cands, key=lambda c: -c["score"])

def finalize(moments, words, count, mn, mx):
    out = []
    for m in moments:
        try: s, e = float(m["start"]), float(m["end"])
        except Exception: continue
        if e - s < mn * 0.6: continue
        if e - s > mx * 1.3: e = s + mx
        sn = snap(words, s, e)
        if not sn: continue
        s, e = sn
        if any(not (e <= o["start"] or s >= o["end"]) for o in out): continue
        out.append({"start": s, "end": e, "score": round(float(m.get("score", 0)), 2),
                    "hook": clean_hook(m.get("hook")), "reason": m.get("reason", "")})
        if len(out) >= count: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words"); ap.add_argument("out")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--min", type=float, default=15); ap.add_argument("--max", type=float, default=45)
    a = ap.parse_args()
    words = json.load(open(a.words))
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    moments, how = [], "heuristic"
    if key:
        try:
            moments = finalize(gemini_pick(words, a.count + 3, a.min, a.max, key, model), words, a.count, a.min, a.max)
            how = f"gemini:{model}"
        except Exception as ex:
            print("WARNING: Gemini failed, using free heuristic instead ->", ex, file=sys.stderr)
    else:
        print("NOTE: no GEMINI_API_KEY set, using the free keyword heuristic.", file=sys.stderr)
    if not moments:
        moments = finalize(heuristic_pick(words, a.count, a.min, a.max), words, a.count, a.min, a.max); how = "heuristic"
    for i, m in enumerate(moments, 1): m["name"] = f"clip{i:02d}"
    json.dump(moments, open(a.out, "w"), indent=2)
    print(f"picked {len(moments)} moments via {how}")

if __name__ == "__main__":
    main()
