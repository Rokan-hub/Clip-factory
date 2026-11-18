# Free Clip Factory

One long video in → 5, 6 or 7 edited 9:16 reels + YouTube/Instagram titles out. Runs free on GitHub Actions.

## How it works

```
Video link ──► Transcribe ──► Pick best moments ──► Pick template per clip ──► Edit ──► Titles ──► Download zip
               Groq Whisper     Gemini (4 pillars)    AI chooses from the        ffmpeg     YT + IG
               (or local)       → Groq → keywords     channel's templates        engine
                                (no talking? → loudest action moments)
```

## One-time setup

1. **Keys (both free, no card):**
   - Gemini: https://aistudio.google.com/apikey → **Create API key**
   - Groq (optional but recommended, makes transcription ~10x faster): https://console.groq.com/keys
2. **Save them in GitHub:** repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
   - `GEMINI_API_KEY` = your Gemini key
   - `GROQ_API_KEY` = your Groq key
   Never paste keys into any file in the repo (the repo is public).

## Make reels

1. Repo → **Actions** → **Make clips** → **Run workflow**
2. Fill in:
   - **source_url**: Google Drive share link ("anyone with the link") or a direct .mp4 link
   - **channel**: channel1 (GTA 6), channel2 (stream), channel3 (YouTube video)
   - **template**: `auto` = the AI picks the best template for every clip. Or force one.
   - **count**: 5, 6 or 7 reels
   - **required_text**: campaign tags/mentions for the descriptions (optional)
3. When it's green: open the run → **Artifacts** → download `clips-channelX`.
   Inside: `clip01_gta_chase.mp4` … (the name shows the template used), `titles_channelX.csv`, `moments.json` (why each moment was picked).
4. Watch every clip and check the campaign rules before posting.

## Templates

| Template | Use for | What it does |
|---|---|---|
| `gta_moment` | Any big gameplay moment | Punch zooms on key words, shake + hit on impacts, whoosh on cuts, punchy colour |
| `gta_react` | Streamer facecam + gameplay | Face on top, gameplay below, reaction-first hook |
| `gta_list` | Hidden details, easter eggs, top 5s | Full frame on blurred fill, big numbers, freeze + WAIT, pop sounds |
| `gta_chase` | Police chases, wanted levels, getaways | 1.15× build-up into real-time finish, red/blue siren flash + shake + hit on impacts, vignette, riser into the ending |
| `gta_heist` | Heists, robberies, big money, expensive cars | Automatic $ counter from what's said, warm Vice City grade, cinematic black bars, slow-mo replay of the payoff |
| `gta_fail` | Fails, ragdolls, funny physics, glitches | Freeze + "WAIT FOR IT..." right before the payoff, then a slow-mo replay, white flash on the hit |
| `gta_cinematic` | Graphics, scenery, sunsets, map showcase | Whole frame visible on blurred fill, teal-orange grade, slow push-in, vignette, calm captions |
| `stream_clutch` | Stream big plays | Facecam split, zoom + shake on the reaction |
| `stream_funny` | Stream funny moments | Context card, chat pop-ups, pop sounds |
| `video_story` | YouTube story payoffs | Blurred fill, $ counter, payoff-first |
| `video_talk` | Talking head | Face-centred crop, big centre captions, punch-ins |

Every template also has: word-by-word captions with the active word highlighted, hook text for the first 3 s, call-to-action at the end,
a progress bar along the top (keeps people watching), loudness normalised to −14 LUFS (the Shorts/Reels standard), 1080×1920 at 30 fps.

## Files

| File | Job |
|---|---|
| `.github/workflows/make-clips.yml` | The Run button |
| `clipkit/run_pipeline.py` | Runs every step; no-speech fallback; template choice |
| `clipkit/transcribe.py` | Groq Whisper (cloud, fast) → local Whisper backup |
| `clipkit/pick_moments.py` | Gemini → Groq → keyword picker, scored on Hook / Retention / Satisfaction / Psychology |
| `clipkit/engine.py` | The editor (cuts, 9:16, captions, zooms, shake, grades, flashes, replays, SFX, audio) |
| `clipkit/templates.json` | All templates (edit to tweak looks) |
| `clipkit/channels.json` | Per-channel font, colours, emojis, calls to action |
| `clipkit/titles.py` + `trends.json` | Titles, descriptions, hashtags. Edit `trends.json` to change hashtags/title styles |

## Notes

- Free Gemini/Groq requests may be used by the providers to improve their products. Don't send private material.
- YouTube blocks downloads from GitHub's servers: download the source yourself and share a Drive link.
- Groq's free tier caps each audio file at 25 MB; the script splits audio into 10-minute pieces automatically.
- In-game radio songs can trigger copyright claims. Prefer footage with the radio off.
- Run locally: `pip install -r requirements.txt`, install ffmpeg, then `python clipkit/run_pipeline.py --source video.mp4 --channel channel1 --count 6`
