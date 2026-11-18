#!/usr/bin/env python3
"""One router for every free AI provider. Each job has its own provider order, so the
work is split and no single free tier runs out. If a provider fails or is rate-limited,
the next one takes over. Providers without a key are skipped.

Keys (GitHub secrets, all optional; add as many as you like):
  GEMINI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY
Model overrides: GEMINI_MODEL, GROQ_MODEL, CEREBRAS_MODEL, MISTRAL_MODEL, OPENROUTER_MODEL
"""
import json, os, re, sys, time, urllib.request, urllib.error

PROVIDERS = {
    # name: (key env, base url or None for Gemini, model env, default model, max transcript chars)
    "gemini":     ("GEMINI_API_KEY", None, "GEMINI_MODEL", "gemini-flash-latest", 400_000),
    "groq":       ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "GROQ_MODEL", "llama-3.3-70b-versatile", 24_000),
    "cerebras":   ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "CEREBRAS_MODEL", "gpt-oss-120b", 60_000),
    "mistral":    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "MISTRAL_MODEL", "mistral-small-latest", 80_000),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "OPENROUTER_MODEL",
                   "meta-llama/llama-3.3-70b-instruct:free", 60_000),
}

# Who does what. First = main worker for that job, the rest are backups.
JOBS = {
    "moments": ["gemini", "mistral", "cerebras", "groq", "openrouter"],  # needs the whole transcript -> long context first
    "titles":  ["groq", "cerebras", "mistral", "openrouter", "gemini"],  # many small calls -> rotated per clip
}

LOG = []      # (job, provider, ok, note) for the run summary
DOWN = set()  # providers that hit their free limit this run -> skipped from now on

def available(job):
    return [p for p in JOBS[job] if os.environ.get(PROVIDERS[p][0], "").strip() and p not in DOWN]

def limit(provider):
    return PROVIDERS[provider][4]

def _post(url, body, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "clip-factory", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _call(provider, prompt):
    key_env, base, model_env, default, _ = PROVIDERS[provider]
    key = os.environ[key_env].strip(); model = os.environ.get(model_env, default)
    if provider == "gemini":
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                  {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4}},
                  {"x-goog-api-key": key})
        return d["candidates"][0]["content"]["parts"][0]["text"]
    d = _post(f"{base}/chat/completions",
              {"model": model, "temperature": 0.4, "messages": [{"role": "user", "content": prompt}]},
              {"Authorization": f"Bearer {key}"})
    return d["choices"][0]["message"]["content"]

def parse_json(txt):
    """Pull the first JSON array/object out of a model reply (handles ```json fences and chatter)."""
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try: return json.loads(txt)
    except Exception: pass
    starts = [i for i in (txt.find("["), txt.find("{")) if i >= 0]
    if not starts: raise ValueError("no JSON in reply")
    i = min(starts); close = "]" if txt[i] == "[" else "}"
    return json.loads(txt[i: txt.rfind(close) + 1])

def ask(job, make_prompt, rotate=0, check=None):
    """Try providers for this job in order (rotated by `rotate`). make_prompt(provider) -> prompt text.
    check(parsed) must return the cleaned result or raise. Returns (result, provider) or (None, None)."""
    order = available(job)
    if not order: return None, None
    r = rotate % len(order); order = order[r:] + order[:r]
    for p in order:
        for attempt in range(2):
            try:
                res = parse_json(_call(p, make_prompt(p)))
                if check: res = check(res)
                LOG.append((job, p, True, "")); return res, p
            except urllib.error.HTTPError as e:
                note = f"HTTP {e.code}: {e.read().decode(errors='ignore')[:160]}"
                if e.code in (429, 500, 502, 503) and attempt == 0:
                    time.sleep(12); continue
                if e.code in (401, 403, 429): DOWN.add(p)   # out of free quota or bad key
                LOG.append((job, p, False, note)); break
            except Exception as e:
                LOG.append((job, p, False, str(e)[:160])); break
        print(f"   {job}: {p} failed, passing to the next provider", file=sys.stderr)
    return None, None

def save_log(path):
    old = json.load(open(path)) if os.path.exists(path) else []
    old += [{"job": j, "provider": p, "ok": ok, "note": n} for j, p, ok, n in LOG]
    json.dump(old, open(path, "w"), indent=1)
