# Speechmatics Python SDK

## Overview

Speechmatics offers modular Python SDK packages for different use cases. These packages replace the old `speechmatics-python` package, which has been deprecated.

**Deprecation Notice:** The `speechmatics-python` package has been deprecated. No new features will be added, with bug fixes and security patches only until 2025-12-31.

## Installation & Setup

The SDK offers modular packages for different use cases:

```bash
# Batch transcription
pip install speechmatics-batch

# Realtime streaming
pip install speechmatics-rt

# Voice agents
pip install speechmatics-voice

# Text-to-speech
pip install speechmatics-tts
```

Development setup requires cloning the repository and installing dependencies via `make install-dev`.

## New SDK Packages

Each client targets a specific Speechmatics API (e.g. real-time, batch transcription), making it easier to install only what you need and keep dependencies minimal.

The new SDK packages include:

* **speechmatics-rt** - A Python client for Speechmatics Real-Time API
* **speechmatics-batch** - An async Python client for Speechmatics Batch API
* **speechmatics-voice** - A Voice Agent Python client for Speechmatics Real-Time API
* **speechmatics-tts** - An async Python client for Speechmatics TTS API

**GitHub Repository:** https://github.com/speechmatics/speechmatics-python-sdk

## Legacy Package Documentation

The speechmatics python SDK and CLI is documented at: https://speechmatics.github.io/speechmatics-python/

## Core Capabilities

**Batch Transcription**: Upload audio files and receive transcripts with timestamps, speaker identification, and entity extraction. The async client handles job submission and polling for results.

**Realtime Streaming**: Process live audio with ultra-low latency (150ms p95), delivering both partial and final transcripts. Integrates with microphone input for immediate feedback.

**Voice Agents**: Combines real-time transcription with speaker diarization and automatic turn detection—ideal for conversational AI applications using adaptive configuration presets.

**Text-to-Speech**: Generate natural-sounding speech with multiple voice options and language support, available in streaming or batch modes.

## Key Advantages

The platform supports 55+ languages and achieves a 6.8% word error rate. Features include:

* Custom vocabularies (up to 1,000 words)
* Speaker diarization at no extra cost
* Realtime translation across 30+ languages
* Enterprise deployments benefit from SOC 2 Type II certification
* 99.9% uptime SLA
* Flexible hosting options including on-premises and air-gapped environments

## Getting Started

Users obtain API keys via the Speechmatics portal. Code examples demonstrate async/await patterns with event-driven handlers for processing transcription segments and turn boundaries—reflecting modern Python development standards.

## Real-Time Transcription with Python

### Installation

The Speechmatics Python library is available via pip:

```bash
pip3 install speechmatics-python
```

For microphone input, you'll also need PyAudio:

```bash
pip3 install pyaudio
```

Mac M1/M2 users should install PyAudio using Homebrew with specific compiler flags before using pip.

### Key Configuration Elements

The implementation requires several parameters:

* **API Key**: Your authentication credentials
* **Connection URL**: The WebSocket endpoint (e.g., `wss://eu.rt.speechmatics.com/v2/{language}`)
* **Device Index**: Microphone selection (defaults to system input)
* **Audio Settings**: PCM format specification with sample rate and chunk size

### Core Components

The code example demonstrates creating an `AudioProcessor` class that manages audio buffering asynchronously. PyAudio callbacks feed microphone data into this buffer while a `WebsocketClient` streams the audio to Speechmatics for transcription.

Event handlers capture both partial and final transcripts, enabling real-time feedback during speech processing.

### Configuration Options

Full list of parameters described in the official library documentation includes:

* Language selection
* Partial transcript enabling
* Processing modes ("enhanced" operating point)
* Maximum delay settings

### Enhanced Features

For voice applications, consider enabling End of Utterance Detection with `conversation_config` parameters to detect speech completion through silence monitoring.

## Documentation Links

* **Main SDK Documentation:** https://speechmatics.github.io/speechmatics-python/
* **API & Product Docs:** https://docs.speechmatics.com
* **SDK Introduction:** https://docs.speechmatics.com/sdk/
* **New SDK GitHub:** https://github.com/speechmatics/speechmatics-python-sdk
* **Legacy SDK GitHub:** https://github.com/speechmatics/speechmatics-python
* **PyPI Package:** https://pypi.org/project/speechmatics-python/
* **Python Microphone Guide:** https://docs.speechmatics.com/speech-to-text/realtime/guides/python-using-microphone
