# multi-speaker_markup_from_dialog_transcript.py

Generate multi-speaker dialogue audio from plain-text transcripts using
Google Cloud Gemini TTS.

## Quick Start

```bash
# Default: produces .wav + .ogg + .mp3 (lossless from API, high-quality local transcode)
uv run multi-speaker_markup_from_dialog_transcript.py -i dialogue.txt

# With custom voices and style prompt (still 3 files)
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt --voices Charon,Kore \
  -p "Casual conversation between friends"

# Single format from API directly (no local transcoding)
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt -e ogg -o output.ogg

# German dialogue
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i german_dialogue.txt -l de-DE \
  --voices Orus,Aoede \
  -p "Patient teacher with enthusiastic student"
```

## Prerequisites

* Google Cloud project with Text-to-Speech API enabled
* Application Default Credentials: `gcloud auth application-default login`
* Python >=3.11 (or just use `uv run` — dependencies auto-installed via PEP 723)
* **ffmpeg** (for local transcoding to .ogg and .mp3 — optional but recommended)

## Input Format

Plain text with `Speaker: dialogue text` lines:

```
Teacher: How do you say "good morning" in German?
Student: Guten Morgen?
Teacher: Sehr gut! That's perfect.
```

* Markdown formatting on speaker names (`**Speaker**:`) is auto-stripped
* Blank lines are ignored
* Continuation lines (no colon) are appended to the previous speaker's turn

## Audio Quality Guide

### Default behavior (recommended)

By default, the tool requests **lossless WAV** from the API and locally
transcodes via ffmpeg to:

* `.wav` — lossless (LINEAR16), largest files
* `.ogg` — Opus VBR ~96kbps (excellent quality for speech)
* `.mp3` — LAME VBR V2 ~190kbps (high quality, widely compatible)

This gives you the best of all worlds: a lossless master plus high-quality
compressed versions. The API's built-in MP3 is only 32kbps — local
transcoding from WAV is vastly better.

### Single-format mode (-e flag)

Use `-e` to request a specific format from the API directly (no local
transcoding, single output file):

| Encoding | Flag | Quality | Notes |
|----------|------|---------|-------|
| OGG_OPUS | `-e ogg` | Best API lossy | "Considerably higher than MP3 at similar bitrate" |
| LINEAR16 | `-e wav` | Lossless | Uncompressed PCM + WAV header |
| MP3 | `-e mp3` | Low | **Fixed 32kbps** — no bitrate control |
| MULAW | `-e mulaw` | Telephony | 8-bit G.711 mu-law |
| ALAW | `-e alaw` | Telephony | 8-bit G.711 A-law. Not supported by Chirp 3 HD |

### WAV only (no transcoding)

Use `--no-transcode` to get just the lossless WAV from the API without
ffmpeg compression.

### Sample Rate

Default is 24000 Hz (24 kHz). You can change with `--sample-rate`:

```bash
# 48 kHz for higher fidelity WAV
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt -e wav --sample-rate 48000
```

Requesting a sample rate different from the voice's native rate triggers
resampling which "might result in worse audio quality" per the API docs.

### Audio Device Profiles

Apply post-processing optimized for target playback device:

```bash
# Optimize for headphones
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt --audio-profile headphone-class-device

# Multiple profiles (applied in order)
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt \
  --audio-profile headphone-class-device \
  --audio-profile wearable-class-device
```

Available profiles:

* `headphone-class-device` — Headphones, earbuds
* `handset-class-device` — Smartphones
* `small-bluetooth-speaker-class-device` — Small Bluetooth speakers
* `medium-bluetooth-speaker-class-device` — Smart home speakers (Google Home)
* `large-home-entertainment-class-device` — Home entertainment, smart TVs
* `large-automotive-class-device` — Car speakers
* `telephony-class-application` — IVR, phone systems
* `wearable-class-device` — Smartwatches

## Long Dialogues (Chunking)

The Gemini TTS API has an **8000 byte combined limit** (text + prompt). For
dialogues exceeding this, use `--chunk`:

```bash
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i long_dialogue.txt --chunk -e ogg \
  --voices Charon,Kore -p "Debate between colleagues"
```

* Splits at turn boundaries (never mid-sentence)
* Each chunk gets the same prompt and voice config
* Audio is concatenated (cleanest with MP3 or WAV)
* Without `--chunk`, oversized input produces a warning

## Voices

30 Gemini TTS prebuilt voices. Use `--list-voices` to see all, or specify
with `--voices`:

```bash
# List all available voices
uv run multi-speaker_markup_from_dialog_transcript.py --list-voices

# Assign specific voices to speakers (in order of appearance)
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt --voices Zephyr,Puck
```

If `--voices` is omitted, speakers are auto-assigned from the pool.

## Pipeline Integration (JSONL)

Use `--jsonl` for machine-readable output:

```bash
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i dialogue.txt -o out.ogg -e ogg --jsonl \
  | jq -r 'select(.event=="completed") | .output_file'
```

## All Flags

Run `--help` for flag reference, or `--help-llm` for comprehensive
documentation designed for LLM agents (includes input format examples,
JSONL event schema, language codes, etc.).

## Models

* `gemini-2.5-flash-tts` (default) — fast, cost-efficient
* `gemini-2.5-pro-tts` — higher quality, better for podcasts/audiobooks

## API Documentation

Archived in `google_cloud_tts_docs/` with YAML front matter. See
`google_cloud_tts_docs/INDEX.md` for the full index.
