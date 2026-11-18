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
    # name: (key env, base url or None for Gemini, model env, max transcript chars)
    "gemini":     ("GEMINI_API_KEY", None, "GEMINI_MODEL", 400_000),
    "groq":       ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "GROQ_MODEL", 24_000),
    "cerebras":   ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "CEREBRAS_MODEL", 60_000),
    "mistral":    ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "MISTRAL_MODEL", 80_000),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "OPENROUTER_MODEL", 60_000),
}

# Free model names change often, so each provider has several to try, best first.
# If a name is rejected (404/400) it is skipped for the rest of the run and the next one is tried.
MODELS = {
    "gemini":     ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-latest"],
    "groq":       ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
    "cerebras":   ["gpt-oss-120b", "llama-3.3-70b"],
    "mistral":    ["mistral-small-latest"],
    "openrouter": ["openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free", "meta-llama/llama-3.3-70b-instruct:free"],
}

# Who does what. First = main worker for that job, the rest are backups.
JOBS = {
    "moments": ["gemini", "mistral", "cerebras", "groq", "openrouter"],  # needs the whole transcript -> long context first
    "titles":  ["groq", "cerebras", "mistral", "openrouter", "gemini"],  # many small calls -> rotated per clip
}

BAD = set()   # (provider, model) pairs the provider rejected this run
TIMEOUT = 60  # seconds to wait for one answer before moving on
LOG = []      # (job, provider, ok, note) for the run summary
DOWN = set()  # providers that hit their free limit this run -> skipped from now on

def available(job):
    return [p for p in JOBS[job] if os.environ.get(PROVIDERS[p][0], "").strip() and p not in DOWN]

def limit(provider):
    return PROVIDERS[provider][3]

def models(provider):
    forced = os.environ.get(PROVIDERS[provider][2], "").strip()
    return ([forced] if forced else []) + [m for m in MODELS[provider] if m != forced]

def _post(url, body, headers, timeout=None):
    timeout = timeout or TIMEOUT
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "clip-factory", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _call(provider, prompt, model):
    key_env, base, _, _ = PROVIDERS[provider]
    key = os.environ[key_env].strip()
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
    """Try providers for this job in order (rotated by `rotate`); inside a provider try its model names.
    make_prompt(provider) -> prompt text. check(parsed) returns the cleaned result or raises.
    Returns (result, provider) or (None, None). A slow or failing provider is dropped quickly."""
    order = available(job)
    if not order: return None, None
    r = rotate % len(order); order = order[r:] + order[:r]
    for p in order:
        tried = 0
        for m in models(p):
            if (p, m) in BAD: continue
            if tried >= 3 or p in DOWN: break
            tried += 1
            try:
                res = parse_json(_call(p, make_prompt(p), m))
                if check: res = check(res)
                LOG.append((job, p, True, m)); return res, p
            except urllib.error.HTTPError as e:
                note = f"{m} HTTP {e.code}: {e.read().decode(errors='ignore')[:140]}"
                LOG.append((job, p, False, note))
                if e.code in (400, 404): BAD.add((p, m)); continue          # model name not accepted -> next name
                if e.code in (500, 502, 503): time.sleep(2); continue        # overloaded -> try another model
                if e.code in (401, 403, 429): DOWN.add(p)                    # bad key or free limit reached
                break
            except Exception as e:                                           # timeout, bad JSON, ...
                LOG.append((job, p, False, f"{m} {str(e)[:140]}"))
                if "timed out" in str(e): DOWN.add(p)                        # don't wait on a slow provider again
                break
        print(f"   {job}: {p} failed, passing to the next provider", file=sys.stderr)
    return None, None

def save_log(path):
    old = json.load(open(path)) if os.path.exists(path) else []
    old += [{"job": j, "provider": p, "ok": ok, "note": n} for j, p, ok, n in LOG]
    json.dump(old, open(path, "w"), indent=1)
