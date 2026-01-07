# OpenAI Audio API Reference

## Create Transcription

Transcribes audio into the input language.

```
POST https://api.openai.com/v1/audio/transcriptions
```

### Request Body (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Audio file (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm) |
| `model` | string | Yes | Model ID (`whisper-1`, `gpt-4o-transcribe`, etc.) |
| `language` | string | No | ISO 639-1 language code |
| `prompt` | string | No | Text to guide style (max 224 tokens for whisper-1) |
| `response_format` | string | No | Output format: json, text, srt, verbose_json, vtt |
| `temperature` | number | No | Sampling temperature (0-1) |
| `timestamp_granularities[]` | array | No | word, segment (whisper-1 only) |
| `chunking_strategy` | string/object | No | auto or server_vad config |

### Response (json format)

```json
{
  "text": "The transcribed text..."
}
```

### Response (verbose_json format)

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.470000267028809,
  "text": "The transcribed text...",
  "words": [
    {
      "word": "The",
      "start": 0.0,
      "end": 0.32
    }
  ],
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 8.47,
      "text": "The transcribed text...",
      "tokens": [464, 12808, ...],
      "temperature": 0.0,
      "avg_logprob": -0.27,
      "compression_ratio": 1.38,
      "no_speech_prob": 0.02
    }
  ]
}
```

## Create Translation

Translates audio into English.

```
POST https://api.openai.com/v1/audio/translations
```

### Request Body (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Audio file to translate |
| `model` | string | Yes | Model ID (only `whisper-1` supported) |
| `prompt` | string | No | Text to guide style |
| `response_format` | string | No | Output format |
| `temperature` | number | No | Sampling temperature (0-1) |

### Response

```json
{
  "text": "The translated English text..."
}
```

## cURL Examples

### Basic Transcription

```bash
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1"
```

### With Language and Prompt

```bash
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F language="en" \
  -F prompt="OpenAI, GPT-4, Whisper"
```

### With Timestamps

```bash
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F response_format="verbose_json" \
  -F "timestamp_granularities[]=word"
```

### Translation

```bash
curl https://api.openai.com/v1/audio/translations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/german-audio.mp3" \
  -F model="whisper-1"
```

## Python SDK Examples

### Basic Transcription

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

### With All Options

```python
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    language="en",
    prompt="Technical terms: API, SDK, REST",
    response_format="verbose_json",
    temperature=0,
    timestamp_granularities=["word", "segment"]
)
```

### Translation

```python
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 413 | Payload Too Large - File exceeds 25MB |
| 415 | Unsupported Media Type - Invalid file format |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

## Rate Limits

Rate limits vary by account tier. Check your account dashboard for current limits.

## File Size Limits

* Maximum file size: 25 MB
* For larger files, split using tools like PyDub or ffmpeg
