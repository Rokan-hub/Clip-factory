#!/usr/bin/env python3
"""Titles, descriptions and hashtags for every clip, separate for YouTube Shorts and Instagram Reels.

AI writes them (each clip goes to a different free provider, see llm.py), and the free pattern-based
writer is the backup, so a run always comes back with titles.
Usage: titles.py words.json clips.json channel [--required "@a #b"]
clips.json: [{"name":"clip01_gta_chase","start":12.0,"end":48.0,"hook":"...","template":"gta_chase"}, ...]
Output: titles_<channel>.csv
"""
import json, re, sys, random, csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm

STRONG = {"never","worst","best","insane","crazy","impossible","secret","won","lost","paid","free","million","thousand","dollars","shocked","scared","angry","disgusting","amazing","incredible","wait","literally","nobody","sucks","hate","regret","gross","scream","dropped","ruined","expensive","rich","poor","fired","broke","destroyed","disrespectful","cheese","gold","blown"}
WEAK = ("it's been a while","how are you","you have","thank you","all right","okay","oh my god","hello","what's up","let's go")
NICHE = {"channel1": "GTA 6 gameplay clips", "channel2": "livestream highlight clips", "channel3": "YouTube video highlight clips"}

def load(p): return json.load(open(p))
def clip_text(words, s, e): return [w for w in words if s <= w["s"] and w["e"] <= e]
def sentences(ws):
    out, cur = [], []
    for w in ws:
        cur.append(w)
        if re.search(r"[.?!]$", w["w"]): out.append(cur); cur = []
    if cur: out.append(cur)
    return [" ".join(x["w"] for x in c) for c in out]
def score(sent):
    s = 0; low = sent.lower()
    s += 3 * len(re.findall(r"\$\s?\d", sent))
    s += 2 * sum(1 for w in re.findall(r"[a-z']+", low) if w in STRONG)
    s += 2 if "?" in sent else 0
    if any(low.startswith(p) or p in low[:22] for p in WEAK): s -= 4
    n = len(sent.split()); s += 2 if 4 <= n <= 12 else (-2 if n > 18 else 0)
    return s
def fix(t): return re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", re.sub(r"\$\s+(\d)", r"$\1", t))
def clean(t): t = fix(t); return re.sub(r"\s+", " ", re.sub(r"[^\w$?!',.%: -]", "", t)).strip(" ,.-")
def tagfix(tags, n):
    out = []
    for t in tags:
        t = "#" + re.sub(r"[^\w]", "", str(t).lstrip("#"))
        if len(t) > 1 and t.lower() not in [x.lower() for x in out]: out.append(t)
    return out[:n]

def basic(c, text_sents, cfg, t, rng, req):
    """Free pattern-based writer (backup)."""
    best = max(text_sents, key=score) if text_sents else ""
    hook = clean(c.get("hook") or best)
    if hook.isupper(): hook = hook.capitalize()
    if len(hook) > 70: hook = hook[:67].rsplit(" ", 1)[0] + "…"
    title = rng.choice(t["title_patterns"]).format(hook=hook)
    tags = list(dict.fromkeys(cfg["base_tags"] + rng.sample(t["hashtags_trending"], min(4, len(t["hashtags_trending"])))))
    cta = rng.choice(cfg["cta"])
    return {"yt_title": (title if len(title) <= 92 else hook)[:92] + " #Shorts",
            "yt_description": f"{hook}\n\n{cta}\n\n{' '.join(tags[:5])}" + (f"\n{req}" if req else ""),
            "ig_caption": f"{rng.choice(cfg['emoji'])} {hook}\n\n{cta}\n\n{' '.join(tags)} {req}".strip(),
            "writer": "basic"}

def ai_prompt(c, transcript, cfg, ch, req):
    return f"""You write viral titles and captions for short vertical videos ({NICHE.get(ch, 'short clips')}).
Channel tone: {cfg.get('tone', 'hype')}. On-screen hook: "{c.get('hook', '')}". Edit style: {c.get('template', '')}.
What is said in the clip: "{transcript[:1500] or '(no speech, gameplay only)'}"

Rules:
- Use curiosity, stakes, emotion. Never lie about what happens. No clickbait the clip does not pay off.
- YouTube title: max 70 characters, no hashtags in it.
- YouTube description: 2-3 short lines, then a call to action.
- YouTube hashtags: 3-5, specific to this clip (e.g. #GTA6 #ViceCity), first one #shorts.
- Instagram caption: different wording from YouTube, casual, 1-3 emojis, ends with a question or call to action to drive comments.
- Instagram hashtags: 8-10, mix of big and niche tags.
Return ONLY JSON:
{{"yt_title": "...", "yt_description": "...", "yt_hashtags": ["#..."], "ig_caption": "...", "ig_hashtags": ["#..."]}}"""

def main():
    words = load(sys.argv[1]); clips = load(sys.argv[2]); ch = sys.argv[3]
    req = sys.argv[sys.argv.index("--required") + 1] if "--required" in sys.argv else ""
    chans = load(os.path.join(HERE, "channels.json")); tr = load(os.path.join(HERE, "trends.json"))
    cfg = chans[ch]; t = tr["channels"].get(ch, tr["channels"]["default"])
    rng = random.Random(hash((ch, tr["updated"])) & 0xffffffff)
    rows = []
    for i, c in enumerate(clips):
        sents = sentences(clip_text(words, c["start"], c["end"]))
        if not sents and not c.get("hook"): continue
        transcript = " ".join(sents)

        def check(d):
            if not isinstance(d, dict) or not d.get("yt_title") or not d.get("ig_caption"): raise ValueError("missing fields")
            yt_tags = tagfix(d.get("yt_hashtags", []), 5); ig_tags = tagfix(d.get("ig_hashtags", []), 10)
            if "#shorts" not in [x.lower() for x in yt_tags]: yt_tags = ["#shorts"] + yt_tags[:4]
            title = clean(d["yt_title"])[:90]
            return {"yt_title": f"{title} #Shorts",
                    "yt_description": f"{d.get('yt_description', '').strip()}\n\n{' '.join(yt_tags)}" + (f"\n{req}" if req else ""),
                    "ig_caption": f"{d['ig_caption'].strip()}\n\n{' '.join(ig_tags)} {req}".strip()}

        res, prov = llm.ask("titles", lambda pv: ai_prompt(c, transcript, cfg, ch, req), rotate=i, check=check)
        if res: res["writer"] = prov
        else: res = basic(c, sents, cfg, t, rng, req)
        rows.append([c["name"], res["yt_title"], res["yt_description"], res["ig_caption"], res["writer"]])
        print(f"   {c['name']}: titles by {res['writer']}", flush=True)

    out = os.path.join(os.getcwd(), f"titles_{ch}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip", "youtube_title", "youtube_description", "instagram_caption", "written_by"]); w.writerows(rows)
    llm.save_log(os.path.join(os.getcwd(), "ai_usage.json"))
    print(out, len(rows), "clips")

main()
