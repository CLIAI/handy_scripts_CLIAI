# Speechmatics Realtime Transcription Quickstart

## Setup Steps

### 1. API Key Creation

"Create an API key in the portal here, which you'll use to securely access the API."

Navigate to Settings > API Keys in the Speechmatics Portal.

### 2. Library Installation

Two client options are available:

**JavaScript:**

```bash
npm install @speechmatics/real-time-client
```

**Python:**

```bash
pip install speechmatics-python
```

### 3. Configuration

Incorporate your API key into the code and establish connection settings with the WebSocket endpoint.

## Implementation Overview

The provided examples demonstrate connecting to an audio stream and processing transcription events. Both languages utilize similar patterns:

* Authenticate via JWT token
* Configure transcription parameters (language, operating point, max delay)
* Handle incoming message events

## Output Types

### Final Transcripts

Definitive results reflecting the best transcription, arriving incrementally with adjustable delay based on the `max_delay` setting for accuracy optimization.

### Partial Transcripts

Low-latency updates (typically under 500ms) that may revise as additional context becomes available. Enable these using `enable_partials` in transcription configuration for responsive user experiences.

Both output types can be combined to balance speed and accuracy requirements.

## Connection Details

**Endpoint:** `wss://eu.rt.speechmatics.com/v2/`

For browser-based applications, temporary API keys must be passed as query parameters:

```
wss://eu.rt.speechmatics.com/v2?jwt=<temporary-key>
```

## Key Configuration Parameters

* **Language** - ISO language code (required)
* **Operating Point** - "standard" or "enhanced" model
* **Max Delay** - Trade-off between latency and accuracy (0.7 to 4 seconds)
* **Enable Partials** - Whether to receive low-latency partial transcripts
* **Diarization** - Speaker identification options
* **Domain** - Specialization (e.g., finance, medical)

## Event Handling

Applications should handle these message types:

* `RecognitionStarted` - Session acknowledged
* `AddPartialTranscript` - Work-in-progress results
* `AddTranscript` - Finalized segments
* `EndOfUtterance` - Silence-detected boundaries
* `EndOfTranscript` - Stream completion
* `Error` or `Warning` - Processing issues

## Best Practices

* For voice applications, enable End of Utterance Detection
* Experiment with `max_delay` to find the right trade-off for your use case
* Use temporary keys for browser-based applications
* Handle both partial and final transcripts for optimal user experience
