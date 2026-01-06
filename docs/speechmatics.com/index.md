# Speechmatics Documentation Archive

This directory contains archived documentation from Speechmatics (speechmatics.com), a speech-to-text API platform.

**Archive Date:** 2026-01-06

**Main Documentation Portal:** https://docs.speechmatics.com/

## Archived Documentation

### Getting Started

* [Authentication](authentication.md) - API key creation, usage, and temporary keys
  * Source: https://docs.speechmatics.com/get-started/authentication
* [Realtime Quickstart](realtime-quickstart.md) - Quick setup guide for real-time transcription
  * Source: https://docs.speechmatics.com/speech-to-text/realtime/quickstart

### API Documentation

* [API Overview](api-overview.md) - Overview of REST and WebSocket APIs
  * Sources: https://docs.speechmatics.com/api-ref, https://docs.speechmatics.com/
* [WebSocket API](websocket-api.md) - Real-time transcription WebSocket API reference
  * Source: https://docs.speechmatics.com/api-ref/realtime-transcription-websocket
* [Batch API](batch-api.md) - REST API for batch transcription jobs
  * Sources: https://docs.speechmatics.com/jobsapi, https://docs.speechmatics.com/api-ref/batch/create-a-new-job

### Features

* [Speaker Diarization](speaker-diarization.md) - Speaker identification and separation
  * Sources: https://docs.speechmatics.com/features/diarization, https://docs.speechmatics.com/speech-to-text/batch/batch-diarization, https://docs.speechmatics.com/speech-to-text/realtime/realtime-diarization
* [Formatting Features](formatting-features.md) - Punctuation, capitalization, profanity handling
  * Sources: https://docs.speechmatics.com/speech-to-text/formatting, https://docs.speechmatics.com/features/punctuation-settings
* [Translation](translation.md) - Multi-language translation capabilities
  * Source: https://docs.speechmatics.com/speech-to-text/features/translation

### Languages and Models

* [Languages](languages.md) - Complete list of 55+ supported languages and models
  * Sources: https://docs.speechmatics.com/introduction/supported-languages, https://www.speechmatics.com/languages

### SDKs

* [Python SDK](python-sdk.md) - Python SDK documentation and usage
  * Sources: https://github.com/speechmatics/speechmatics-python-sdk, https://docs.speechmatics.com/sdk/, https://speechmatics.github.io/speechmatics-python/, https://docs.speechmatics.com/speech-to-text/realtime/guides/python-using-microphone

## Key Features Summary

* **Languages:** 55+ languages and dialects supported
* **APIs:** Both REST (batch) and WebSocket (real-time) available
* **Diarization:** Speaker, channel, and combined speaker-channel modes
* **Translation:** 30+ language pairs with real-time capability
* **Models:** Standard (faster) and Enhanced (more accurate) operating points
* **Python SDK:** Modular packages for batch, realtime, voice agents, and TTS
* **Accuracy:** 6.8% word error rate
* **Latency:** 150ms p95 for real-time transcription

## Additional Resources

* **GitHub Organization:** https://github.com/speechmatics
* **Speechmatics Academy:** https://github.com/speechmatics/speechmatics-academy
* **Main Website:** https://www.speechmatics.com/
* **Developer Portal:** https://www.speechmatics.com/developers

## File Structure

Each documentation page consists of two files:

* `{name}.md` - The documentation content in markdown format
* `{name}.meta.yaml` - Metadata including source URL, title, fetch date, and description

## Notes

* All documentation was fetched on 2026-01-06
* Source URLs are preserved in metadata files for reference and updates
* The Python SDK has transitioned from `speechmatics-python` (deprecated) to modular packages: `speechmatics-rt`, `speechmatics-batch`, `speechmatics-voice`, and `speechmatics-tts`
