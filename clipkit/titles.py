#!/usr/bin/env python3
"""Free title/description generator. No AI calls.
Usage: titles.py words.json clips.json channel [--required "@a #b"]
clips.json: [{"name":"clip1","start":12.0,"end":48.0}, ...]  (seconds in source video)
Output: titles_<channel>.csv with YouTube + Instagram title/description per clip."""
import json, re, sys, random, csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
STRONG = {"never","worst","best","insane","crazy","impossible","secret","won","lost","paid","free","million","thousand","dollars","shocked","scared","angry","disgusting","amazing","incredible","wait","literally","nobody","sucks","hate","regret","gross","scream","dropped","ruined","expensive","rich","poor","fired","broke","destroyed","disrespectful","cheese","gold","blown"}
WEAK = ("it's been a while","how are you","you have","thank you","all right","okay","oh my god","hello","what's up","let's go")
def load(p): return json.load(open(p))
def clip_text(words, s, e):
    return [w for w in words if s <= w["s"] and w["e"] <= e]
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
    n = len(sent.split())
    s += 2 if 4 <= n <= 12 else (-2 if n > 18 else 0)
    return s
def fix(t): return re.sub(r"(\d)\s+,\s*(\d)", r"\1,\2", re.sub(r"\$\s+(\d)", r"$\1", t))
def clean(t): t = fix(t);  return re.sub(r"\s+", " ", re.sub(r"[^\w$?!',.%: -]", "", t)).strip(" ,.-")
def main():
    words = load(sys.argv[1]); clips = load(sys.argv[2]); ch = sys.argv[3]
    req = sys.argv[sys.argv.index("--required")+1] if "--required" in sys.argv else ""
    chans = load(os.path.join(HERE, "channels.json")); tr = load(os.path.join(HERE, "trends.json"))
    cfg = chans[ch]; t = tr["channels"].get(ch, tr["channels"]["default"])
    rng = random.Random(hash((ch, tr["updated"])) & 0xffffffff)
    rows = []
    for c in clips:
        ws = clip_text(words, c["start"], c["end"]); sents = sentences(ws)
        if not sents and not c.get("hook"): continue
        best = max(sents, key=score) if sents else ""
        hook = clean(c.get("hook") or best)
        if hook.isupper(): hook = hook.capitalize()
        if len(hook) > 70: hook = hook[:67].rsplit(" ", 1)[0] + "…"
        pat = rng.choice(t["title_patterns"]); emoji = rng.choice(cfg["emoji"]); cta = rng.choice(cfg["cta"])
        title = pat.format(hook=hook)
        yt_title = (title if len(title) <= 92 else hook)[:92] + " #Shorts"
        tags = list(dict.fromkeys(cfg["base_tags"] + rng.sample(t["hashtags_trending"], min(4, len(t["hashtags_trending"])))))
        yt_desc = f"{hook}\n\n{cta}\n\n{' '.join(tags[:5])}" + (f"\n{req}" if req else "")
        ig_cap = f"{emoji} {hook}\n\n{cta}\n\n{' '.join(tags)} {req}".strip()
        rows.append([c["name"], yt_title, yt_desc, ig_cap])
    out = os.path.join(os.getcwd(), f"titles_{ch}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "youtube_title", "youtube_description", "instagram_caption"]); w.writerows(rows)
    print(out, len(rows), "clips")
main()
