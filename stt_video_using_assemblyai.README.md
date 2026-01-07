# Video Transcription using AssemblyAI

Extract audio from video files and transcribe using AssemblyAI with optional speaker diarisation.

## Features

* **Video to Audio Extraction**: Automatically extracts MP3 audio from video files using ffmpeg
* **Speaker Diarisation**: Identify and label multiple speakers (A, B, C, etc.)
* **Language Support**: 99+ languages including auto-detection
* **Idempotent**: Skip re-extraction and re-transcription if outputs already exist
* **Interactive Prompts**: Prompts for speaker count and language if not provided

## Prerequisites

```bash
# Required: ffmpeg for audio extraction
sudo pacman -S ffmpeg  # Arch Linux
# or
sudo apt install ffmpeg  # Debian/Ubuntu

# Required: stt_assemblyai.py in PATH
# Clone: https://github.com/CLIAI/handy_scripts

# Set your AssemblyAI API key
export ASSEMBLYAI_API_KEY="your_api_key_here"
```

Get your API key at: https://www.assemblyai.com/

## Quick Start

```bash
# Basic transcription (prompts for speaker count and language)
./stt_video_using_assemblyai.sh video.mp4

# Specify expected speakers
./stt_video_using_assemblyai.sh video.mp4 2

# Specify speakers and language
./stt_video_using_assemblyai.sh video.mp4 3 en
```

## Usage

```
Usage: stt_video_using_assemblyai.sh video_file [expected_speakers [language_code]]

Arguments:
  video_file         Path to the video file to transcribe
  expected_speakers  Number of speakers (0=auto-detect, 1=no diarisation)
  language_code      Language code (default: en)
```

## Examples

### 1. Single Speaker Video

```bash
./stt_video_using_assemblyai.sh lecture.mp4 1 en
```

**Creates:**

* `lecture.mp4.mp3` - Extracted audio
* `lecture.mp4.mp3.assemblyai.json` - Full API response
* `lecture.mp4.mp3.txt` - Plain text transcript

### 2. Multi-Speaker Interview

```bash
./stt_video_using_assemblyai.sh interview.mp4 2 en
```

**Creates transcript with speaker labels:**

```
Speaker A: Welcome to the show.
Speaker B: Thanks for having me.
Speaker A: Let's talk about your latest project.
```

### 3. Auto-Detect Speaker Count

```bash
./stt_video_using_assemblyai.sh meeting.mp4 0 en
```

Enables diarisation but lets AssemblyAI determine the number of speakers.

### 4. Interactive Mode

```bash
./stt_video_using_assemblyai.sh video.mp4
# Prompts:
# Expected speakers [0] (0==any): 3
# Language code [en]: de
```

## Output Files

**Input:** `video.mp4`

**Output:**

* `video.mp4.mp3` - Extracted audio (128k, 44.1kHz)
* `video.mp4.mp3.assemblyai.json` - Full AssemblyAI API response
* `video.mp4.mp3.txt` - Human-readable transcript

## Workflow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  video.mp4  │────▶│  video.mp3  │────▶│  video.mp3.txt   │
└─────────────┘     └─────────────┘     └──────────────────┘
                    (ffmpeg extract)    (AssemblyAI STT)
```

1. **Check dependencies**: Verifies `stt_assemblyai.py` is in PATH
2. **Extract audio**: Uses ffmpeg to extract MP3 (skipped if exists)
3. **Transcribe**: Calls `stt_assemblyai.py` with appropriate flags
4. **Output**: Displays transcript location and content

## Idempotent Behavior

```bash
$ ./stt_video_using_assemblyai.sh video.mp4 2 en
# Extracts audio, transcribes

$ ./stt_video_using_assemblyai.sh video.mp4 2 en
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
| `2+` | Diarisation with expected speaker hint |

## Supported Video Formats

Any format supported by ffmpeg:

* MP4, MKV, AVI, MOV, WMV, FLV, WebM
* And many more

## Error Handling

### Missing stt_assemblyai.py

```
The script stt_assemblyai.py is required to run this program.
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
Error: ASSEMBLYAI_API_KEY environment variable not set.
```

**Solution:** `export ASSEMBLYAI_API_KEY="your_key"`

## Related Tools

* **stt_assemblyai.py** - Underlying transcription tool
* **stt_assemblyai_speaker_mapper.py** - Map speaker labels (A, B) to actual names
* **stt_video_using_speechmatics.sh** - Alternative using Speechmatics API

## License

Part of the CLIAI handy_scripts collection.
