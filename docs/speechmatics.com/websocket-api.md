# Speechmatics WebSocket Real-Time Transcription API

## Connection Details

**Endpoint:** `wss://eu.rt.speechmatics.com/v2/`

For browser-based applications, temporary API keys must be passed as query parameters:

```
wss://eu.rt.speechmatics.com/v2?jwt=<temporary-key>
```

**Handshake Response:** Successful connections receive `101 Switching Protocols`, while failures return `400 Bad Request`, `401 Unauthorized`, or `405 Method Not Allowed`.

## Core Configuration

The `StartRecognition` message initiates sessions with these key options:

* **Language:** ISO language code (required)
* **Domain:** Optional specialization (finance, medical, etc.)
* **Audio Format:** Raw PCM (pcm_f32le, pcm_s16le, mulaw) or encoded files (wav, mp3, aac, ogg, flac, etc.)
* **Sample Rate:** Required for raw audio
* **Max Delay:** "0.7 to 4 seconds between word end and final transcript" delivery
* **Diarization:** Speaker, channel, or combined speaker-channel detection
* **Custom Vocabulary:** Additional words with pronunciation guides

## Message Flow

**Client sends:**

* `StartRecognition` — session initialization
* `AddAudio` — binary audio chunks
* `EndOfStream` — signals completion
* `SetRecognitionConfig` — runtime adjustments
* `GetSpeakers` — request speaker identifiers

**Server responds with:**

* `RecognitionStarted` — acknowledges session
* `AddPartialTranscript` — work-in-progress results (if enabled)
* `AddTranscript` — finalized transcription segments
* `EndOfUtterance` — silence-detected turn boundaries
* `EndOfTranscript` — stream completion
* `Error` or `Warning` — issues during processing

## Output Features

Results include:

* Per-word confidence scores (0.0–1.0)
* Timing metadata
* Speaker labels
* Entity recognition (dates, amounts)
* Optional translation to multiple target languages simultaneously

## Production Endpoints

**Realtime SaaS Regions:**

* EU1: eu.rt.speechmatics.com
* US1: us.rt.speechmatics.com

## Authentication

For WebSocket connections:

* Server-side: Include API key in Authorization header
* Client-side (browser): Use temporary JWT tokens as query parameters to avoid exposing long-lived API keys

Temporary tokens are generated via POST to `https://mp.speechmatics.com/v1/api_keys?type=rt` with TTL parameter (60-86400 seconds).
