# stt_video

A unified Speech-to-Text wrapper that provides a single interface for transcribing video/audio files using multiple STT backends.

## Features

* **Multiple Backends**: Supports Speechmatics, OpenAI, and AssemblyAI
* **URL Support**: Automatically downloads audio from YouTube and other platforms
* **Unified Interface**: Common flags for language and speaker diarization
* **Pass-Through Support**: Forward additional arguments to backend scripts
* **Smart Download Chain**: Falls back through multiple download tools

## Installation

The script requires Python 3.10+ and uses `uv` for execution:

```bash
# Make executable
chmod +x stt_video

# Ensure it's in your PATH (e.g., symlink to ~/bin)
ln -s /path/to/stt_video ~/bin/stt_video
```

## Quick Start

```bash
# Transcribe local file (default: speechmatics)
stt_video video.mp4

# Transcribe YouTube video
stt_video "https://youtube.com/watch?v=xyz"

# Use OpenAI backend
stt_video -b openai video.mp4

# Specify language and speakers
stt_video -l de -s 3 video.mp4
```

## Backends

| Backend | Script | Diarization | Notes |
|---------|--------|-------------|-------|
| `speechmatics` (default) | `stt_video_using_speechmatics.sh` | Yes | Enhanced operating point, best accuracy |
| `openai` | `stt_video_using_openai.sh` | No | 25MB file limit, auto language detection |
| `assemblyai` | `stt_video_using_assemblyai.sh` | Yes | Good accuracy, reasonable pricing |

## Command-Line Options

```bash
stt_video [options] <file_or_url> [-- backend_args...]
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--backend` | `-b` | STT backend: speechmatics, openai, assemblyai |
| `--language` | `-l` | Language code (e.g., en, de, fr, es, ja) |
| `--speakers` | `-s` | Number of speakers for diarization (0=auto, 1=no diarization) |
| `--output-dir` | `-o` | Output directory for URL downloads |
| `--help` | `-h` | Show help message |

### Pass-Through Arguments

Arguments after `--` are passed directly to the backend script:

```bash
# Pass operating-point to speechmatics
stt_video video.mp4 -- --operating-point enhanced

# Multiple pass-through args
stt_video video.mp4 -- --operating-point enhanced --max-delay 10
```

## Environment Variables

### STT_VIDEO_BACKEND

Sets the default backend (overridden by `--backend` flag):

```bash
export STT_VIDEO_BACKEND=openai
stt_video video.mp4  # Uses OpenAI
```

## URL Download Chain

When given a URL, `stt_video` tries these tools in order:

| Priority | Tool | Usage |
|----------|------|-------|
| 1 | `yt-dlp-priv.py --audio` | Preferred - downloads audio + transcripts + metadata |
| 2 | `yt-dlp -x` | Standard YouTube downloader |
| 3 | `youtube-dl -x` | Legacy YouTube downloader |
| 4 | `curl` / `wget` / `aria2c` | Only for direct media URLs (.mp3, .mp4, etc.) |

### Direct Media URL Detection

The script only uses `curl`/`wget`/`aria2c` for URLs that end with known media extensions:

* Audio: `.mp3`, `.m4a`, `.wav`, `.ogg`, `.flac`, `.aac`, `.opus`
* Video: `.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`, `.wmv`

For platform URLs (YouTube, etc.), install `yt-dlp-priv.py` or `yt-dlp`.

## Examples

### Basic Transcription

```bash
# Local file with default settings
stt_video interview.mp4

# YouTube video
stt_video "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Language and Speaker Settings

```bash
# German video with 2 speakers
stt_video -l de -s 2 german_interview.mp4

# French video, auto-detect speakers
stt_video -l fr -s 0 french_lecture.mp4

# Single speaker (disable diarization)
stt_video -s 1 podcast.mp3
```

### Backend Selection

```bash
# OpenAI (fast, no diarization)
stt_video -b openai short_clip.mp4

# AssemblyAI (good for long files)
stt_video -b assemblyai long_interview.mp4

# Speechmatics with enhanced accuracy
stt_video -b speechmatics video.mp4 -- --operating-point enhanced
```

### Batch Processing

```bash
# Transcribe multiple files
for f in *.mp4; do
    stt_video -b speechmatics -s 2 "$f"
done

# Process URLs from file
while read url; do
    stt_video "$url"
done < urls.txt
```

## Output

The script passes through output from the underlying backend. Typically:

* Transcript printed to stdout
* Status messages printed to stderr
* Transcript saved to `{input_file}.mp3.txt`

Example output locations:

```
video.mp4           → video.mp4.mp3.txt
interview.wav       → interview.wav.mp3.txt
```

## Troubleshooting

### "Backend script not found"

Install the required backend script:

```bash
# Check if script is in PATH
which stt_video_using_speechmatics.sh

# If not, ensure CLIAI/handy_scripts is in your PATH
export PATH="$PATH:/path/to/handy_scripts_CLIAI"
```

### "No suitable download tool found for URL"

Install one of the supported download tools:

```bash
# Recommended: yt-dlp-priv.py (from gw_scripts)
# Or install yt-dlp
pip install yt-dlp
# Or
pipx install yt-dlp
```

### "Download completed but couldn't locate output file"

The download succeeded but the script couldn't find the output file automatically. Check:

* The download tool's default output directory
* For `yt-dlp-priv.py`: `~/Downloads/youtube-dl/youtube-dl-on-btrfs2nd/{uploader}/`

### OpenAI "File too large" error

OpenAI Whisper API has a 25MB limit. Options:

* Use a different backend: `stt_video -b speechmatics video.mp4`
* Split the audio file into smaller chunks
* Compress the audio to lower bitrate

### Speaker diarization not working

* OpenAI does not support diarization
* Use `speechmatics` or `assemblyai` backend
* Ensure `--speakers` is set to 0 (auto) or a number > 1

## Dependencies

* Python 3.10+
* `uv` for script execution
* Backend-specific requirements:
  * `stt_speechmatics.py` for Speechmatics
  * `stt_openai.py` for OpenAI
  * `stt_assemblyai.py` for AssemblyAI
* Optional: `yt-dlp-priv.py`, `yt-dlp`, `youtube-dl` for URL support

## See Also

* `stt_video_using_speechmatics.sh` - Speechmatics backend
* `stt_video_using_openai.sh` - OpenAI backend
* `stt_video_using_assemblyai.sh` - AssemblyAI backend
* `yt-dlp-priv.py` - Enhanced yt-dlp wrapper with --audio support
