# Video Transcription using OpenAI Whisper

Extract audio from video files and transcribe using OpenAI Whisper API.

## Features

* **Video to Audio Extraction**: Automatically extracts MP3 audio from video files using ffmpeg
* **Language Support**: 99+ languages with auto-detection
* **Idempotent**: Skip re-extraction and re-transcription if outputs already exist
* **File Size Warning**: Alerts when audio exceeds OpenAI's 25MB limit

## Limitations

**No Speaker Diarization**: OpenAI's whisper-1 model does not support speaker diarization. For multi-speaker videos, use:

* `stt_video_using_assemblyai.sh` - Speaker labels: A, B, C...
* `stt_video_using_speechmatics.sh` - Speaker labels: S1, S2, S3...

## Prerequisites

```bash
# Required: ffmpeg for audio extraction
sudo pacman -S ffmpeg  # Arch Linux
# or
sudo apt install ffmpeg  # Debian/Ubuntu

# Required: stt_openai.py in PATH
# Clone: https://github.com/CLIAI/handy_scripts

# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"
```

Get your API key at: https://platform.openai.com/api-keys

## Quick Start

```bash
# Basic transcription (auto-detect language)
./stt_video_using_openai.sh video.mp4

# Specify language
./stt_video_using_openai.sh video.mp4 en

# German video
./stt_video_using_openai.sh video.mp4 de
```

## Usage

```
Usage: stt_video_using_openai.sh video_file [language_code]

Arguments:
  video_file     Path to the video file to transcribe
  language_code  Language code (default: auto)
```

## Examples

### 1. Auto-Detect Language

```bash
./stt_video_using_openai.sh lecture.mp4
```

**Creates:**

* `lecture.mp4.mp3` - Extracted audio
* `lecture.mp4.mp3.openai.json` - Full API response
* `lecture.mp4.mp3.txt` - Plain text transcript

### 2. Specify Language

```bash
./stt_video_using_openai.sh interview.mp4 en
```

### 3. Interactive Mode

```bash
./stt_video_using_openai.sh video.mp4
# Prompts:
# Language code [auto]: de
```

## Output Files

**Input:** `video.mp4`

**Output:**

* `video.mp4.mp3` - Extracted audio (128k, 44.1kHz)
* `video.mp4.mp3.openai.json` - Full OpenAI API response
* `video.mp4.mp3.txt` - Human-readable transcript

## Workflow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  video.mp4  │────▶│  video.mp3  │────▶│  video.mp3.txt   │
└─────────────┘     └─────────────┘     └──────────────────┘
                    (ffmpeg extract)    (OpenAI Whisper)
```

1. **Check dependencies**: Verifies `stt_openai.py` is in PATH
2. **Extract audio**: Uses ffmpeg to extract MP3 (skipped if exists)
3. **Check file size**: Warns if audio exceeds 25MB limit
4. **Transcribe**: Calls `stt_openai.py` with language option
5. **Output**: Displays transcript location and content

## File Size Limit

OpenAI Whisper API has a **25MB file size limit**.

For larger files:

```bash
# Option 1: Use lower bitrate extraction
ffmpeg -i video.mp4 -vn -ab 64k -ar 16000 -y video.mp3

# Option 2: Split into chunks
ffmpeg -i video.mp4 -f segment -segment_time 600 -vn -ab 128k chunk_%03d.mp3

# Option 3: Use AssemblyAI or Speechmatics (no size limit)
./stt_video_using_assemblyai.sh video.mp4
./stt_video_using_speechmatics.sh video.mp4
```

## Idempotent Behavior

```bash
$ ./stt_video_using_openai.sh video.mp4 en
# Extracts audio, transcribes

$ ./stt_video_using_openai.sh video.mp4 en
# Skips extraction (MP3 exists), skips transcription (TXT exists)
File video.mp4.mp3 already exists.
SKIPPING: transcription of video.mp4.mp3 as video.mp4.mp3.txt already exists
```

**To force re-processing:** Delete existing `.mp3` and/or `.txt` files

## Supported Video Formats

Any format supported by ffmpeg:

* MP4, MKV, AVI, MOV, WMV, FLV, WebM
* And many more

## Error Handling

### Missing stt_openai.py

```
The script stt_openai.py is required to run this program.
It is not currently in your PATH.
```

**Solution:** Add the handy_scripts directory to your PATH

### Missing ffmpeg

```
ffmpeg: command not found
```

**Solution:** Install ffmpeg for your system

### Missing API Key

```
Error: OPENAI_API_KEY environment variable not set.
```

**Solution:** `export OPENAI_API_KEY="your_key"`

### File Too Large

```
WARNING: Audio file is larger than 25MB (OpenAI limit).
Consider splitting the file or using AssemblyAI/Speechmatics instead.
```

**Solution:** Use lower bitrate, split file, or use alternative STT service

## Comparison: Video Transcription Tools

| Feature | OpenAI | AssemblyAI | Speechmatics |
|---------|--------|------------|--------------|
| Speaker diarization | No* | Yes | Yes |
| Max file size | 25 MB | 5 GB | Unlimited |
| Languages | 99+ | 99+ | 55+ |
| Translation | Yes | No | No |

*gpt-4o-transcribe-diarize model supports diarization but requires different API

## Related Tools

* **stt_openai.py** - Underlying transcription tool
* **stt_video_using_assemblyai.sh** - AssemblyAI video transcription (with diarization)
* **stt_video_using_speechmatics.sh** - Speechmatics video transcription (with diarization)

## License

Part of the CLIAI handy_scripts collection.
