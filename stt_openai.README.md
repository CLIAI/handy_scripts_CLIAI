# OpenAI Speech-to-Text (STT) Transcription Tool

Transcribe audio files using the OpenAI Whisper API with support for multiple languages, timestamps, and translation.

## Features

* **Multiple Models**: whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe
* **Language Support**: 99+ languages with auto-detection
* **Translation**: Translate any language to English
* **Timestamps**: Word-level and segment-level timing
* **Multiple Output Formats**: JSON, text, SRT, VTT, verbose_json
* **Idempotent**: Skip re-transcription if output already exists
* **META Warning Messages**: Automatic disclaimer about potential transcription errors

## Prerequisites

```bash
# Install dependencies (handled automatically by uv)
# - openai>=1.0

# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"
```

Get your API key at: https://platform.openai.com/api-keys

## Quick Start

```bash
# Basic transcription
./stt_openai.py audio.mp3

# Specify language
./stt_openai.py -l en audio.mp3

# With word timestamps
./stt_openai.py --timestamps audio.mp3

# Translate to English
./stt_openai.py --translate audio.mp3
```

## META Transcript Warning Message

**By default**, all transcript outputs include a META warning message to remind readers that STT transcripts may contain errors.

### Disabling the META Message

**Via command-line flag:**

```bash
./stt_openai.py --no-meta-message audio.mp3
```

**Via environment variable:**

```bash
export STT_META_MESSAGE_DISABLE=1
./stt_openai.py audio.mp3
```

### Custom META Message

```bash
export STT_META_MESSAGE="DRAFT - UNVERIFIED TRANSCRIPT"
./stt_openai.py audio.mp3
```

## Usage Examples

### 1. Basic Transcription

```bash
./stt_openai.py audio.mp3
```

**Creates:**

* `audio.mp3.openai.json` - Full API response
* `audio.mp3.txt` - Plain text transcript

### 2. With Language Specification

```bash
# English
./stt_openai.py -l en audio.mp3

# German
./stt_openai.py -l de audio.mp3

# Japanese
./stt_openai.py -l ja audio.mp3
```

### 3. With Timestamps

```bash
./stt_openai.py --timestamps audio.mp3
```

**JSON output includes:**

```json
{
  "text": "Hello world",
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0}
  ]
}
```

### 4. Translation to English

```bash
# Translate German audio to English
./stt_openai.py --translate german_audio.mp3
```

### 5. With Prompting

```bash
# Guide transcription with technical terms
./stt_openai.py --prompt "OpenAI, GPT-4, Whisper, LLM" audio.mp3
```

### 6. Different Output Formats

```bash
# SRT subtitles
./stt_openai.py --response-format srt audio.mp3

# VTT subtitles
./stt_openai.py --response-format vtt audio.mp3

# Plain text only
./stt_openai.py --response-format text audio.mp3
```

### 7. Custom Output Path

```bash
./stt_openai.py -o transcript.txt audio.mp3
```

### 8. Output to Stdout Only

```bash
./stt_openai.py -o - audio.mp3
```

### 9. Verbose Logging

```bash
# INFO level
./stt_openai.py -v audio.mp3

# DEBUG level
./stt_openai.py -vvvvv audio.mp3
```

## Command-Line Options

### Positional Arguments

* `audio_input` - Path to audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)

### Output Control

* `-o, --output PATH` - Output file path (default: `{audio}.txt`)
* `-q, --quiet` - Suppress status messages

### Language & Model

* `-l, --language CODE` - Language code (default: auto-detect)
* `--model MODEL` - Whisper model (default: `whisper-1`)

### Transcription Options

* `--timestamps` - Include word-level timestamps
* `--response-format FORMAT` - json, text, srt, verbose_json, vtt
* `--translate` - Translate to English instead of transcribing
* `--prompt TEXT` - Guide transcription style
* `--temperature FLOAT` - Sampling temperature (0-1)

### Logging

* `-v, --verbose` - Increase verbosity (use multiple times)

### META Message Control

* `--no-meta-message` - Disable META warning message

## Output Files

### Default File Names

**Input:** `audio.mp3`

**Output:**

* `audio.mp3.openai.json` - Full API response (always created)
* `audio.mp3.txt` - Human-readable transcript (default output)

### JSON Format

```json
{
  "_meta_note": "THIS IS AN AUTOMATED SPEECH-TO-TEXT...",
  "text": "The transcribed text...",
  "task": "transcribe",
  "language": "english",
  "duration": 8.47
}
```

### Verbose JSON Format (with timestamps)

```json
{
  "text": "Hello world",
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "world", "start": 0.5, "end": 1.0}
  ],
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.0,
      "text": "Hello world"
    }
  ]
}
```

## Supported Audio Formats

* MP3, MP4, MPEG, MPGA
* M4A, WAV, WebM
* FLAC, OGG

**Maximum file size:** 25 MB

For larger files, split using ffmpeg or PyDub.

## Idempotent Behavior

```bash
$ ./stt_openai.py audio.mp3
# Transcribes audio, creates files

$ ./stt_openai.py audio.mp3
# Skips transcription, displays existing transcript
SKIPPING: transcription of audio.mp3 as audio.mp3.txt already exists
```

**To force re-transcription:** Delete existing `.txt` file

## Error Handling

### Missing API Key

```
Error: OPENAI_API_KEY environment variable not set.
Get your API key at: https://platform.openai.com/api-keys
```

**Solution:** `export OPENAI_API_KEY="your_key"`

### File Too Large

```
Error: File size exceeds 25MB limit
```

**Solution:** Split audio file into smaller chunks

### Unsupported Format

```
Error: Unsupported audio format
```

**Solution:** Convert to MP3, WAV, or other supported format

## Comparison: OpenAI vs AssemblyAI vs Speechmatics

| Feature | OpenAI Whisper | AssemblyAI | Speechmatics |
|---------|---------------|------------|--------------|
| Speaker diarization | gpt-4o-diarize only | Built-in | Built-in |
| Speaker labels | N/A | A, B, C... | S1, S2, S3... |
| Languages | 99+ | 99+ | 55+ |
| Max file size | 25 MB | 5 GB | Unlimited |
| Timestamps | Word & segment | Word | Word |
| Translation | Yes (to English) | No | No |
| Output formats | json, text, srt, vtt | json, text | json, text |

## Related Tools

* **stt_openai_OR_local_whisper_cli.py** - Interactive CLI with local Whisper support
* **stt_assemblyai.py** - AssemblyAI transcription tool
* **stt_speechmatics.py** - Speechmatics transcription tool
* **stt_video_using_openai.sh** - Video transcription wrapper

## License

Part of the CLIAI handy_scripts collection.
