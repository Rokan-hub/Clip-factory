# Free Clip Factory

Long video in, 9:16 Shorts/Reels clips + titles out. Runs free on GitHub Actions.

## One-time setup (about 10 minutes)

1. **Create the repo.** On github.com click **New repository**. Name it anything (e.g. `clip-factory`).
   - **Public** = unlimited free Actions minutes, but anyone can see your code and the files in the repo.
   - **Private** = free but capped at 2,000 minutes/month.
2. **Upload these files.** On the new repo page click **uploading an existing file**, drag in everything from this folder
   (keep the folders: `.github/workflows`, `clipkit`, `input`), then **Commit changes**.
   - Tip: the `.github` folder starts with a dot. If your file manager hides it, use the zip -> "Upload files" on the GitHub site, or `git push` from a computer.
3. **Get a free Gemini key.** Go to https://aistudio.google.com/apikey and click **Create API key**.
4. **Save the key in GitHub.** Repo -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**.
   Name: `GEMINI_API_KEY`. Value: your key. Never paste the key into any file in the repo.

## Make clips

1. Repo -> **Actions** tab -> **Make clips** -> **Run workflow**.
2. Fill in:
   - **source_url**: a direct video link, or a Google Drive share link (set sharing to "anyone with the link").
     Or leave it empty and put the file name of a video you uploaded into the `input/` folder in **source_file** (GitHub allows files up to 100 MB).
   - **channel**: channel1 (GTA 6), channel2 (stream), channel3 (video).
   - **template**: `auto` uses that channel's default. Or pick one.
   - **count**: how many clips.
   - **required_text**: campaign tags/mentions that must be in the description (optional).
3. Wait. A 1-hour video takes roughly 20-40 minutes (Whisper runs on CPU).
4. Open the finished run -> scroll to **Artifacts** -> download `clips-channelX`. Inside: the MP4 clips, `titles_channelX.csv`
   (YouTube + Instagram titles/descriptions/hashtags) and `moments.json` (why each moment was picked).
5. **Check the clips before posting.** Check the campaign rules too.

## What each file does

| File | Job |
|---|---|
| `.github/workflows/make-clips.yml` | The button. Starts a free cloud machine and runs everything |
| `clipkit/run_pipeline.py` | Runs the steps in order |
| `clipkit/transcribe.py` | Whisper speech-to-text with word timestamps |
| `clipkit/pick_moments.py` | Gemini picks the best moments using the 4 pillars (hook, retention, satisfaction, psychology). Falls back to a free keyword picker if Gemini fails |
| `clipkit/engine.py` | Cuts, reframes to 9:16, captions, hook text, zooms, shake, sound effects |
| `clipkit/templates.json` | The 8 editing templates |
| `clipkit/channels.json` | Per-channel fonts, colours, emojis, calls to action |
| `clipkit/titles.py` + `trends.json` | Free title/description/hashtag writer. Edit `trends.json` to change hashtags and title styles |

## Notes

- Free Gemini requests can be used by Google to improve its products. Don't send private material.
- YouTube usually blocks downloads from GitHub's servers, so download the video yourself and give a Drive/direct link.
- To run on your own computer instead: `pip install -r requirements.txt`, install ffmpeg, then
  `python clipkit/run_pipeline.py --source video.mp4 --channel channel1`
- If no `GEMINI_API_KEY` is set, it still works using the free keyword picker (lower quality picks).
