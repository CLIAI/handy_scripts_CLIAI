# Transcript Pseudo-Backends for stt_video

**Date:** 2026-01-30
**Status:** Design Complete

## Overview

Add support for downloading pre-existing transcripts/subtitles from video platforms as an alternative to running STT. This saves cost and time when platforms already provide transcripts.

### New Backends

| Backend | Alias | Behavior |
|---------|-------|----------|
| `youtube` | `yt` | Prefer manual, fallback to auto |
| `youtube-sub` | `yt-sub` | Manual subtitles only |
| `youtube-auto` | `yt-auto` | Auto-generated only |
| `transcript` | - | Auto-detect provider, prefer manual |
| `transcript-sub` | - | Auto-detect provider, manual only |
| `transcript-auto` | - | Auto-detect provider, auto only |

### Key Design Decisions

* Implemented directly in `stt_video` (no new backend script)
* Filename encodes source type: `video.youtube.manual.en.txt`
* Language: downloads original + requested (`-l`) + English; `-l all` for everything
* Provider abstraction: designed for multiple platforms, YouTube implemented first
* Enhanced `yt-dlp-priv.py --status --json` to report subtitle type metadata

### Files Changed

* `stt_video` - add transcript backend logic
* `yt-dlp-priv.py` - enhance subtitle handling and JSON output

---

## Backend Detection and Routing

### Backend Configuration

```python
TRANSCRIPT_BACKENDS = {
    # YouTube-specific
    "youtube": {"provider": "youtube", "prefer": "manual", "fallback": "auto"},
    "yt": {"provider": "youtube", "prefer": "manual", "fallback": "auto"},
    "youtube-sub": {"provider": "youtube", "prefer": "manual", "fallback": None},
    "yt-sub": {"provider": "youtube", "prefer": "manual", "fallback": None},
    "youtube-auto": {"provider": "youtube", "prefer": "auto", "fallback": None},
    "yt-auto": {"provider": "youtube", "prefer": "auto", "fallback": None},

    # Generic (auto-detect provider from URL)
    "transcript": {"provider": "auto", "prefer": "manual", "fallback": "auto"},
    "transcript-sub": {"provider": "auto", "prefer": "manual", "fallback": None},
    "transcript-auto": {"provider": "auto", "prefer": "auto", "fallback": None},
}
```

### Provider Auto-Detection

For `transcript*` backends:

```python
def detect_transcript_provider(url: str) -> str | None:
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "vimeo.com" in url:
        return "vimeo"  # Future
    # ... more providers
    return None
```

### Routing Logic

1. If backend in `TRANSCRIPT_BACKENDS` -> handle as transcript download
2. If backend in `BACKENDS` (existing STT) -> route to backend script as before
3. Unknown backend -> error

---

## Filename Conventions

### Output Filename Pattern

```
{base}.{provider}.{type}.{lang}.{ext}
```

Examples:

* `video.youtube.manual.en.txt` - English manual subtitle from YouTube
* `video.youtube.auto.de.txt` - German auto-generated from YouTube
* `video.youtube.manual.es.vtt` - Spanish manual, original VTT format

### Comparison with STT Backends

| Backend | Output filename |
|---------|-----------------|
| `speechmatics` | `video.speechmatics.txt` |
| `openai` | `video.openai.txt` |
| `youtube` | `video.youtube.manual.en.txt` |
| `youtube-auto` | `video.youtube.auto.en.txt` |

### Why Include Language in Transcript Filenames?

* STT backends process one language (specified by `-l`)
* Transcript backends may download multiple languages simultaneously
* Avoids filename collisions when downloading original + requested + English

### Format Handling

* Download as VTT (yt-dlp default)
* Convert to TXT (plain text, timestamps stripped) as primary output
* Keep original VTT alongside for reference

Example output for `stt_video -b youtube -l de URL`:

```
video.youtube.manual.en.vtt
video.youtube.manual.en.txt
video.youtube.auto.de.vtt
video.youtube.auto.de.txt
```

---

## Language Handling

### Language Selection Logic

```python
def determine_languages_to_download(
    requested_lang: str | None,
    original_lang: str | None,  # from video metadata
    available_langs: list[str]
) -> list[str]:

    if requested_lang == "all":
        return available_langs

    # Build priority list (deduplicated)
    wanted = []

    # 1. Original language (highest priority)
    if original_lang and original_lang in available_langs:
        wanted.append(original_lang)

    # 2. Requested language
    if requested_lang and requested_lang in available_langs:
        if requested_lang not in wanted:
            wanted.append(requested_lang)

    # 3. English fallback (always include if available)
    if "en" in available_langs and "en" not in wanted:
        wanted.append("en")

    # 4. If nothing matched, take first available
    if not wanted and available_langs:
        wanted.append(available_langs[0])

    return wanted
```

### Examples

| Video original | `-l` flag | Available | Downloaded |
|----------------|-----------|-----------|------------|
| Spanish | (none) | es, en, de | es, en |
| Spanish | `de` | es, en, de | es, de, en |
| English | `de` | en, de | en, de |
| Japanese | (none) | ja, en | ja, en |
| Spanish | `all` | es, en, de | es, en, de |

### Getting Original Language

Extract from yt-dlp metadata (`original_language` or detect from available subtitles).

---

## yt-dlp-priv.py Changes

### New Flags

```
--transcript-type TYPE   Subtitle type: "auto", "manual", "both" (default: both)
--transcript-lang LANG   Language(s) to download: code, comma-separated, or "all"
```

### Enhanced `--status --json` Output

```json
{
  "url": "https://youtube.com/watch?v=xyz",
  "video_id": "xyz",
  "platform": "youtube",
  "original_language": "es",
  "subtitles": [
    {
      "language": "en",
      "type": "manual",
      "format": "vtt",
      "path": "/path/to/video.en.manual.vtt",
      "size_bytes": 12345
    },
    {
      "language": "en",
      "type": "auto",
      "format": "vtt",
      "path": "/path/to/video.en.auto.vtt",
      "size_bytes": 14567
    },
    {
      "language": "es",
      "type": "manual",
      "format": "vtt",
      "path": "/path/to/video.es.manual.vtt",
      "size_bytes": 11234
    }
  ]
}
```

### Filename Output Template

Modify `download_transcripts()` to use output template that encodes type:

```
%(title)s [%(id)s].{type}.%(subtitles_lang)s.%(ext)s
```

Where `{type}` is `manual` or `auto` based on source.

---

## stt_video Changes

### New Function for Transcript Backends

```python
def run_transcript_backend(
    url: str,
    backend: str,
    language: str | None,
    output_dir: Path,
) -> list[Path]:
    """
    Download transcripts from video platform.
    Returns list of generated .txt files.
    """
    config = TRANSCRIPT_BACKENDS[backend]

    # Auto-detect provider if needed
    provider = config["provider"]
    if provider == "auto":
        provider = detect_transcript_provider(url)
        if not provider:
            raise ValueError(f"Cannot detect transcript provider for: {url}")

    # Build yt-dlp-priv.py command
    cmd = ["yt-dlp-priv.py", "--transcript", "--json"]

    # Add type preference
    if config["fallback"]:
        cmd.extend(["--transcript-type", "both"])
    else:
        cmd.extend(["--transcript-type", config["prefer"]])

    # Add language
    cmd.extend(["--transcript-lang", language or "auto"])
    cmd.append(url)

    # Execute and parse JSON result
    result = subprocess.run(cmd, capture_output=True, text=True)
    # ... parse, filter by preference, convert VTT->TXT, rename files

    return output_files
```

### Integration in Main Flow

```python
def main():
    # ... existing arg parsing ...

    backends = parse_backends(args.backend)

    for backend in backends:
        if backend in TRANSCRIPT_BACKENDS:
            files = run_transcript_backend(url, backend, args.language, output_dir)
        else:
            # Existing STT backend logic
            run_stt(file_path, backend, ...)
```

---

## Error Handling & Edge Cases

### When No Subtitles Available

```python
if not available_subtitles:
    if backend in ("youtube", "yt", "transcript"):
        # Has fallback - warn but don't fail
        print(f"> [{backend}] No subtitles available, skipping", file=sys.stderr)
        return []
    else:
        # Strict mode (youtube-sub, youtube-auto) - fail
        raise RuntimeError(f"No {config['prefer']} subtitles available")
```

### When Preferred Type Unavailable

| Backend | Manual available | Auto available | Result |
|---------|------------------|----------------|--------|
| `youtube` | No | Yes | Use auto |
| `youtube` | Yes | No | Use manual |
| `youtube-sub` | No | Yes | Error |
| `youtube-auto` | Yes | No | Error |

### Non-YouTube URL with YouTube Backend

```python
if backend.startswith(("youtube", "yt")) and "youtube" not in url:
    raise ValueError(f"Backend '{backend}' requires YouTube URL")
```

### VTT to TXT Conversion Failure

* Keep VTT file even if TXT conversion fails
* Log warning, don't fail entire operation
* User can manually convert later

### Rate Limiting / Download Failures

* Inherit yt-dlp-priv.py's existing retry logic
* Report failure in JSON output with error message

---

## Implementation Tasks

### Phase 1: yt-dlp-priv.py Enhancements

1. Add `--transcript-type` flag (auto/manual/both)
2. Add `--transcript-lang` flag (code/comma-separated/all)
3. Modify subtitle filename template to include `.manual.` or `.auto.`
4. Enhance `--status --json` to include subtitle type and original language
5. Add `--list-subs` wrapper to query available subtitles without downloading

### Phase 2: stt_video Transcript Backend

1. Add `TRANSCRIPT_BACKENDS` config dict
2. Add `detect_transcript_provider()` function
3. Add `run_transcript_backend()` function
4. Add VTT->TXT conversion (strip timestamps, clean formatting)
5. Integrate into main flow alongside existing STT backends
6. Update `--status --json` to report transcript files

### Phase 3: Testing & Documentation

1. Test with various YouTube videos (manual only, auto only, both, none)
2. Test language selection logic
3. Update README documentation
4. Update `--help` output

---

## Related: stt-in-batch Bug

During this design session, a bug was identified in `stt-in-batch`:

**Bug 1: Tool naming mismatch**

* Line 383: `tool_name = f"stt-{provider}"` looks for `stt-assemblyai`
* Actual scripts are named `stt_assemblyai.py` (underscore, .py extension)

**Bug 2: Default provider**

* Line 64: `"default_provider": "assemblyai"`
* Expected: `speechmatics` as default

**Fix:** Update `get_stt_tool_path()` to try multiple naming conventions:

1. `stt-{provider}`
2. `stt_{provider}`
3. `stt_{provider}.py`
4. `stt-{provider}.py`
