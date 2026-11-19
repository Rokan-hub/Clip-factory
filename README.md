# Free Clip Factory

One long video in → 5, 6 or 7 edited 9:16 reels + YouTube/Instagram titles out. Runs free on GitHub Actions.

## How it works

```
Video link ──► Transcribe ──► Pick best moments ──► Pick template per clip ──► Edit ──► Titles ──► Download zip
               Groq Whisper     Gemini (4 pillars)    AI chooses from the        ffmpeg     YT + IG
               (or local)       → Groq → keywords     channel's templates        engine
                                (no talking? → loudest action moments)
```

## One-time setup: free keys

Add them in GitHub: repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.
**Free with no card (Nov 2026): Gemini, Groq, OpenRouter, Cloudflare.** Cerebras and Mistral now need payment for API access, so skip them unless you pay; the code still supports them if you ever add a key.
Never paste a key into any file in the repo (the repo is public).

| Secret name | Get it at | Job it does |
|---|---|---|
| `GEMINI_API_KEY` | aistudio.google.com/apikey | **Main: picks the best moments** (reads the whole transcript) |
| `GROQ_API_KEY` | console.groq.com/keys | **Main: transcription** (cloud Whisper) + writes titles |
| `CEREBRAS_API_KEY` | cloud.cerebras.ai (paid credits now, optional) | Writes titles, backup moment picker |
| `MISTRAL_API_KEY` | console.mistral.ai (paid plan now, optional) | Writes titles, 2nd moment picker |
| `OPENROUTER_API_KEY` | openrouter.ai/keys (never add credits; only `:free` models are used) | Writes titles, last backup |
| `CF_ACCOUNT_ID` + `CF_API_TOKEN` | dash.cloudflare.com → AI → Workers AI → Use REST API | Backup transcription |

### How the work is split

| Job | Order (first = main worker, rest = backups) |
|---|---|
| Transcription | Groq → Cloudflare → local Whisper on GitHub (free, slower) |
| Picking moments + template | Gemini → Mistral → Cerebras → Groq → OpenRouter → free keyword picker |
| Titles / descriptions / hashtags | **Rotated per clip:** clip 1 Groq, clip 2 Cerebras, clip 3 Mistral, clip 4 OpenRouter, clip 5 Gemini… → free pattern writer |

If a provider fails or runs out of free quota, the next one takes over, and a provider that hit its limit is skipped for the rest of the run.
The run page shows **"Who did the work"**, and `ai_usage.json` in the download has the details.
Change models with secrets such as `GROQ_MODEL`, `CEREBRAS_MODEL`, `MISTRAL_MODEL`, `OPENROUTER_MODEL`, `GEMINI_MODEL` if a provider retires one.

## Easy mode: the control page (recommended)

`docs/index.html` is a small web page: paste the video link, pick channel / reels / style, press **Make reels**, watch live progress, download the zip.

1. Repo → **Settings** → **Pages** → Source **Deploy from a branch** → Branch **main**, folder **/docs** → **Save**.
2. After about a minute it is live at `https://<your-username>.github.io/Clip-factory/` (works on a phone too).
3. First visit: create a GitHub key that can only run this repo's workflow (the page shows the 5 steps) and paste it once. It is stored only in that browser.

## Make reels (the manual way, in GitHub)

1. Repo → **Actions** → **Make clips** → **Run workflow**
2. Fill in:
   - **source_url**: Google Drive share link ("anyone with the link") or a direct .mp4 link
   - **channel**: channel1 (GTA 6), channel2 (stream), channel3 (YouTube video)
   - **template**: `auto` = the AI picks the best template for every clip. Or force one.
   - **count**: 5, 6 or 7 reels
   - **required_text**: campaign tags/mentions for the descriptions (optional)
3. When it's green: open the run → **Artifacts** → download `clips-channelX`.
   Inside: `clip01_gta_chase.mp4` … (the name shows the template used), `titles_channelX.csv` (AI-written YouTube title + description + hashtags and a separate Instagram caption + hashtags, plus which provider wrote them), `moments.json` (why each moment was picked).
4. Watch every clip and check the campaign rules before posting.

## Templates

| Template | Use for | What it does |
|---|---|---|
| `cinematic` | Story, cutscenes, scenery, driving, calm or emotional moments | Widescreen gameplay on a dark blurred fill, teal-orange grade, film grain, vignette, slow push-in, pink word highlight |
| `hype` | Action, chases, robberies, fails, loud chaos | Big 4:3 zoom with a pulsing zoom, punch-ins on key words, big centre captions with yellow highlight |
| `split_live` | Streamer commentating over gameplay | Facecam on top (auto-detected, cropped exactly inside its border), red divider, gameplay below, sentence-case captions |
| `split_meme` | The streamer's reaction is the joke | Big facecam on top, gameplay below, one all-caps meme line on the seam, no word captions |

Every template also has: word-by-word captions with the active word highlighted, hook text for the first 3 s, call-to-action at the end,
a progress bar along the top (keeps people watching), loudness normalised to −14 LUFS (the Shorts/Reels standard), 1080×1920 at 30 fps.

## Files

| File | Job |
|---|---|
| `.github/workflows/make-clips.yml` | The Run button |
| `clipkit/run_pipeline.py` | Runs every step; no-speech fallback; template choice |
| `clipkit/transcribe.py` | Groq Whisper → Cloudflare Whisper → local Whisper |
| `clipkit/llm.py` | The router: which provider does which job, fallbacks, quota skipping |
| `clipkit/pick_moments.py` | Picks moments scored on Hook / Retention / Satisfaction / Psychology |
| `clipkit/engine.py` | The editor (cuts, 9:16, captions, zooms, shake, grades, flashes, replays, SFX, audio) |
| `clipkit/templates.json` | All templates (edit to tweak looks) |
| `clipkit/channels.json` | Per-channel font, colours, emojis, calls to action |
| `clipkit/titles.py` + `trends.json` | AI titles/descriptions/hashtags per platform; `trends.json` feeds the free backup writer |

## Notes

- Free-tier requests to these providers may be used by the providers to improve their products. Don't send private material.
- YouTube blocks downloads from GitHub's servers: download the source yourself and share a Drive link.
- Groq's free tier caps each audio file at 25 MB; the script splits audio into 10-minute pieces automatically.
- In-game radio songs can trigger copyright claims. Prefer footage with the radio off.
- Run locally: `pip install -r requirements.txt`, install ffmpeg, then `python clipkit/run_pipeline.py --source video.mp4 --channel channel1 --count 6`
