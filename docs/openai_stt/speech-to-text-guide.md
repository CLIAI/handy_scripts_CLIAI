# OpenAI Speech-to-Text API Guide

## Overview

OpenAI provides two main speech-to-text endpoints: **transcriptions** and **translations**. The service uses the open-source Whisper model alongside newer GPT-4o-based options for enhanced quality.

## Available Models

| Model | Description | Output Formats |
|-------|-------------|----------------|
| `whisper-1` | Original open-source Whisper V2 model | json, text, srt, verbose_json, vtt |
| `gpt-4o-transcribe` | Higher quality transcription | json, text |
| `gpt-4o-mini-transcribe` | Lightweight alternative | json, text |
| `gpt-4o-transcribe-diarize` | Speaker-aware transcription with speaker identification | json, text, diarized_json |

## File Specifications

**Supported Formats:** MP3, MP4, MPEG, MPGA, M4A, WAV, WebM, FLAC, OGG

**Size Limitation:** File uploads are currently limited to 25 MB for direct submission. Larger files require splitting or compression.

## Endpoints

### Transcription Endpoint

```
POST https://api.openai.com/v1/audio/transcriptions
```

Transcribes audio into the input language.

### Translation Endpoint

```
POST https://api.openai.com/v1/audio/translations
```

Translates audio into English.

## Key Parameters

### Required Parameters

* **file**: The audio file object to transcribe
* **model**: ID of the model to use (e.g., `whisper-1`)

### Optional Parameters

* **language**: ISO 639-1 language code (e.g., `en`, `de`, `fr`)
* **prompt**: Text to guide transcription style (max 224 tokens for whisper-1)
* **response_format**: Output format (json, text, srt, verbose_json, vtt)
* **temperature**: Sampling temperature (0-1), lower = more deterministic
* **timestamp_granularities[]**: Enable word-level or segment-level timestamps (whisper-1 only)
* **chunking_strategy**: Controls audio chunking (required for diarize model on >30s audio)

## Language Support

Whisper supports 99+ languages including:

* **European:** English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Swedish, Norwegian, Danish, Finnish, Czech, Hungarian, Romanian, Greek, Bulgarian, Croatian, Ukrainian
* **Asian:** Japanese, Korean, Mandarin, Cantonese, Hindi, Thai, Vietnamese, Indonesian, Malay, Tamil, Tagalog
* **Other:** Arabic, Hebrew, Turkish, Swahili, Welsh

The model was trained on 98 languages, with official support for those exceeding 50% word error rate accuracy.

## Timestamps

The `timestamp_granularities[]` parameter enables:

* **word**: Word-level timestamps for precise editing
* **segment**: Segment-level timestamps for larger chunks

Only available with `whisper-1` model.

## Speaker Diarization

The `gpt-4o-transcribe-diarize` model can identify and label different speakers:

* Requires `chunking_strategy` for audio longer than 30 seconds
* Supports up to 4 reference clips (2-10 seconds) for known speaker identification
* Output format: `diarized_json`

## Prompting Best Practices

### Using the Prompt Parameter

Improve accuracy for:

* Technical terms and acronyms
* Proper nouns and brand names
* Preserving context across split audio segments

```python
# Example: Prompting for context
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    prompt="OpenAI, GPT-4, Whisper, LLM"
)
```

### Post-Processing with GPT-4

For larger vocabularies, post-process transcripts with GPT-4 to correct domain-specific terminology.

## Handling Large Files

For audio exceeding 25MB:

1. Use PyDub or similar tools to split audio
2. Avoid splitting mid-sentence
3. Use prompt parameter with previous segment's transcript to maintain context
4. Reassemble transcripts preserving order

## Example Usage

### Basic Transcription (Python)

```python
from openai import OpenAI
client = OpenAI()

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcript.text)
```

### With Timestamps

```python
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["word"]
)
```

### Translation to English

```python
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file
)
print(translation.text)
```

## Realtime Transcription

For streaming transcription, use the Realtime API with Voice Activity Detection (VAD):

* Supported models: `whisper-1`, `gpt-4o-transcribe-latest`, `gpt-4o-mini-transcribe`
* Configurable VAD parameters: threshold, prefix_padding_ms, silence_duration_ms

## Pricing

Whisper API pricing is based on audio duration. Check OpenAI's pricing page for current rates.

## Error Handling

Common issues:

* **File too large**: Split into chunks under 25MB
* **Unsupported format**: Convert to MP3, WAV, or other supported format
* **Rate limiting**: Implement exponential backoff
* **Authentication**: Ensure OPENAI_API_KEY is set correctly
