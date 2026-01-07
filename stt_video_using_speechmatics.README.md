# Video Transcription using Speechmatics

Extract audio from video files and transcribe using Speechmatics with optional speaker diarisation.

## Features

* **Video to Audio Extraction**: Automatically extracts MP3 audio from video files using ffmpeg
* **Speaker Diarisation**: Identify and label multiple speakers (S1, S2, S3, etc.)
* **Enhanced Accuracy**: Uses enhanced operating point by default for better accuracy
* **Language Support**: 55+ languages
* **Idempotent**: Skip re-extraction and re-transcription if outputs already exist
* **Interactive Prompts**: Prompts for speaker count and language if not provided

## Prerequisites

```bash
# Required: ffmpeg for audio extraction
sudo pacman -S ffmpeg  # Arch Linux
# or
sudo apt install ffmpeg  # Debian/Ubuntu

# Required: stt_speechmatics.py in PATH
# Clone: https://github.com/CLIAI/handy_scripts

# Set your Speechmatics API key
export SPEECHMATICS_API_KEY="your_api_key_here"
```

Get your API key at: https://portal.speechmatics.com/

## Quick Start

```bash
# Basic transcription (prompts for speaker count and language)
./stt_video_using_speechmatics.sh video.mp4

# Specify max speakers
./stt_video_using_speechmatics.sh video.mp4 2

# Specify speakers and language
./stt_video_using_speechmatics.sh video.mp4 3 en
```

## Usage

```
Usage: stt_video_using_speechmatics.sh video_file [max_speakers [language_code]]

Arguments:
  video_file     Path to the video file to transcribe
  max_speakers   Maximum number of speakers (0=auto-detect, 1=no diarisation)
  language_code  Language code (default: en)
```

## Examples

### 1. Single Speaker Video

```bash
./stt_video_using_speechmatics.sh lecture.mp4 1 en
```

**Creates:**

* `lecture.mp4.mp3` - Extracted audio
* `lecture.mp4.mp3.speechmatics.json` - Full API response
* `lecture.mp4.mp3.txt` - Plain text transcript

### 2. Multi-Speaker Interview

```bash
./stt_video_using_speechmatics.sh interview.mp4 2 en
```

**Creates transcript with speaker labels:**

```
Speaker S1:	Welcome to the show.
Speaker S2:	Thanks for having me.
Speaker S1:	Let's talk about your latest project.
```

### 3. Auto-Detect Speaker Count

```bash
./stt_video_using_speechmatics.sh meeting.mp4 0 en
```

Enables diarisation but lets Speechmatics determine the number of speakers.

### 4. Interactive Mode

```bash
./stt_video_using_speechmatics.sh video.mp4
# Prompts:
# Max speakers [0] (0==any): 3
# Language code [en]: de
```

### 5. Non-English Languages

```bash
# German
./stt_video_using_speechmatics.sh video.mp4 2 de

# French
./stt_video_using_speechmatics.sh video.mp4 2 fr

# Japanese
./stt_video_using_speechmatics.sh video.mp4 2 ja
```

## Output Files

**Input:** `video.mp4`

**Output:**

* `video.mp4.mp3` - Extracted audio (128k, 44.1kHz)
* `video.mp4.mp3.speechmatics.json` - Full Speechmatics API response
* `video.mp4.mp3.txt` - Human-readable transcript

## Workflow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  video.mp4  │────▶│  video.mp3  │────▶│  video.mp3.txt   │
└─────────────┘     └─────────────┘     └──────────────────┘
                    (ffmpeg extract)    (Speechmatics STT)
```

1. **Check dependencies**: Verifies `stt_speechmatics.py` is in PATH
2. **Extract audio**: Uses ffmpeg to extract MP3 (skipped if exists)
3. **Transcribe**: Calls `stt_speechmatics.py` with enhanced mode
4. **Output**: Displays transcript location and content

## Idempotent Behavior

```bash
$ ./stt_video_using_speechmatics.sh video.mp4 2 en
# Extracts audio, transcribes

$ ./stt_video_using_speechmatics.sh video.mp4 2 en
# Skips extraction (MP3 exists), skips transcription (TXT exists)
File video.mp4.mp3 already exists.
SKIPPING: transcription of video.mp4.mp3 as video.mp4.mp3.txt already exists
```

**To force re-processing:** Delete existing `.mp3` and/or `.txt` files

## Speaker Count Options

| Value | Behavior |
|-------|----------|
| `0` | Diarisation enabled, auto-detect speaker count |
| `1` | No diarisation (single speaker) |
| `2+` | Diarisation with max speaker limit |

**Note:** Speechmatics uses `--max-speakers` (limit) vs AssemblyAI's `--expected-speakers` (hint).

## Enhanced Accuracy Mode

This script uses `--operating-point enhanced` by default, which provides:

* 10-22% Word Error Rate improvement over standard mode
* 7.88% WER (surpasses human-level accuracy of 8.14-10.5%)
* Better handling of technical terminology and proper nouns

To use standard mode (faster but less accurate), edit the script or call `stt_speechmatics.py` directly.

## Supported Video Formats

Any format supported by ffmpeg:

* MP4, MKV, AVI, MOV, WMV, FLV, WebM
* And many more

## Error Handling

### Missing stt_speechmatics.py

```
The script stt_speechmatics.py is required to run this program.
It is not currently in your PATH.
Please ensure that it is available.
One way to do this is by cloning the repository
https://github.com/CLIAI/handy_scripts
into a directory in your PATH.
```

**Solution:** Add the handy_scripts directory to your PATH

### Missing ffmpeg

```
ffmpeg: command not found
```

**Solution:** Install ffmpeg for your system

### Missing API Key

```
Error: SPEECHMATICS_API_KEY environment variable not set.
Get your API key at: https://portal.speechmatics.com/
```

**Solution:** `export SPEECHMATICS_API_KEY="your_key"`

## Comparison: Speechmatics vs AssemblyAI

| Feature | Speechmatics | AssemblyAI |
|---------|-------------|------------|
| Speaker labels | S1, S2, S3... | A, B, C... |
| Speaker parameter | `--max-speakers` (limit) | `--expected-speakers` (hint) |
| Languages | 55+ | 99+ |
| Default mode | Enhanced | Standard |
| Regions | EU, US, AU | EU, US |

## Related Tools

* **stt_speechmatics.py** - Underlying transcription tool
* **stt_speechmatics_speaker_mapper.py** - Map speaker labels (S1, S2) to actual names
* **stt_video_using_assemblyai.sh** - Alternative using AssemblyAI API

## License

Part of the CLIAI handy_scripts collection.
