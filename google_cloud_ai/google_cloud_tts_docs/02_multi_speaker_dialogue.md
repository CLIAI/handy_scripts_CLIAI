---
title: "Create dialogue with multi-speakers (Gemini-TTS)"
source_url: "https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers"
fetched_at: "2026-03-09"
api_version: "gemini"
tags: ["gemini-tts", "multi-speaker", "dialogue", "voices", "streaming", "prompting"]
---

# Gemini-TTS: Create dialogue with multi-speakers

## Overview

Gemini-TTS represents Google Cloud's latest text-to-speech technology. The service moves beyond traditional natural-sounding speech synthesis by providing granular control over generated audio using text-based prompts.

## Available Models

Three models are currently offered:

1. **gemini-2.5-flash-tts**: Optimized for low-latency, controllable single and multi-speaker audio generation suitable for everyday applications. Supports both unary and streaming synthesis.

2. **gemini-2.5-flash-lite-preview-tts**: A preview model focused on single-speaker synthesis with similar capabilities but currently limited to single-speaker scenarios.

3. **gemini-2.5-pro-tts**: Designed for high-control structured workflows including podcasts, audiobooks, and customer support applications.

All models support multiple output formats including LINEAR16, ALAW, MULAW, MP3, OGG_OPUS, and PCM.

## Key Capabilities

The platform enables users to control voice delivery through natural language prompts. Capabilities include managing style, accent, pace, tone, and even emotional expression within generated audio. Additional features support dynamic performance, enhanced pronunciation control, and natural conversation interactions with very low latency.

## Voice Options

Gemini-TTS provides 30 distinct prebuilt voices with specific characteristics. These include:

* **Female voices**: Achernar, Aoede, Callirrhoe, Despina, and others
* **Male voices**: Achird, Algenib, Charon, Enceladus, and others

Each is designed to provide varied tonal qualities for different use cases.

## Language Support

The service supports 24 generally available languages including English (multiple regions), Spanish, French, German, Japanese, Korean, and various Indian languages. An additional 76 languages are available in preview status, covering global linguistic diversity from Afrikaans to Vietnamese.

## Regional Availability

For Cloud Text-to-Speech API, service is available in global, US, EU, and Canadian regions. The Vertex AI API supports more granular regional distribution across multiple US and European locations, enabling data residency compliance.

## API Selection Guidance

* Choose **Cloud Text-to-Speech API** if you need specific audio encoding types or require streaming multiple text chunks.
* Choose **Vertex AI API** if you are already using Gemini-TTS in AI Studio or seeking temperature control for output randomness.

## Implementation Examples

### Cloud Text-to-Speech API

Single-speaker synthesis requires specifying a prompt, text content, voice name, and audio configuration. The API accepts text and prompt fields separately, with combined limits of 8,000 bytes.

Multi-speaker synthesis supports both freeform text input (using speaker aliases like "Sam: Hi Bob!") and structured input with explicitly marked speaker turns. This enables dialogue creation between distinct voices.

### Vertex AI API

This API combines prompt and text into a single `contents` field formatted as `"{prompt}: {text}"`. It supports unidirectional streaming where clients send one request and receive multiple responses. The platform allows temperature adjustment (0.0-2.0 range) for controlling output variability.

## Prompting Strategies

Effective synthesis relies on three coordinated components:

1. **Style prompts** establish emotional tone and context
2. **Text content** provides semantic meaning aligned with emotional intent
3. **Markup tags** inject specific localized actions or modifications

Markup tags operate in distinct modes:

* **Non-speech sounds**: `[sigh]`, `[uhm]`
* **Style modifiers**: `[sarcasm]`, `[whispering]`
* **Pacing controls**: `[short pause]`, `[long pause]`

## Advanced Features

**Media Studio** provides a web-based interface within Google Cloud Console for experimentation. Users can adjust encoding formats, sample rates, speaking speed, and volume gain before deployment.

**Safety filters** can be relaxed using the `relax_safety_filters` field, lowering thresholds for content that would otherwise be blocked.

## Technical Specifications

* Output audio duration is approximately 655 seconds maximum, with truncation occurring if input exceeds this limit.
* Default sampling rate is 24,000 Hz.
* The service maintains low latency for real-time conversational applications while supporting batch processing for longer-form content.
