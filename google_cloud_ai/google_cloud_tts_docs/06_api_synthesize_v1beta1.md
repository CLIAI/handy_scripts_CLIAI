---
title: "Method: text.synthesize (v1beta1)"
source_url: "https://cloud.google.com/text-to-speech/docs/reference/rest/v1beta1/text/synthesize"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/reference/rest/v1beta1/text/synthesize"
fetched_at: "2026-03-09"
api_version: "v1beta1"
tags: ["api-reference", "synthesize", "rest", "v1beta1", "tts"]
---

# Method: text.synthesize

Synthesizes speech synchronously: receive results after all text input has been processed.

## HTTP Request

```
POST https://texttospeech.googleapis.com/v1beta1/text:synthesize
```

The URL uses gRPC Transcoding syntax.

## Request Body

The request body contains data with the following structure:

| Field | Type | Description |
|-------|------|-------------|
| `input` | `SynthesisInput` | Required. The Synthesizer requires either plain text or SSML as input. |
| `voice` | `VoiceSelectionParams` | Required. The desired voice of the synthesized audio. |
| `audioConfig` | `AudioConfig` | Required. The configuration of the synthesized audio. |
| `enableTimePointing[]` | `TimepointType` | Whether and what timepoints are returned in the response. |
| `advancedVoiceOptions` | `AdvancedVoiceOptions` | Advanced voice options. |

## Response Body

If successful, the response body contains data with the following structure:

| Field | Type | Description |
|-------|------|-------------|
| `audioContent` | string (bytes format) | The audio data bytes encoded as specified in the request, including the header for encodings that are wrapped in containers (e.g. MP3, OGG_OPUS). |
| `timepoints[]` | `Timepoint` | A link between a position in the original request input and a corresponding time in the output audio. It's only supported via `<mark>` of SSML input. |
| `audioConfig` | `AudioConfig` | The audio metadata of `audioContent`. |

## Authorization Scopes

Requires the following OAuth scope:

* `https://www.googleapis.com/auth/cloud-platform`

## TimepointType

The type of timepoint information that is returned in the response.

| Enum Value | Description |
|------------|-------------|
| `TIMEPOINT_TYPE_UNSPECIFIED` | Not specified. No timepoint information will be returned. |
| `SSML_MARK` | Timepoint information of `<mark>` tags in SSML input will be returned. |

## AdvancedVoiceOptions

Used for advanced voice options.

| Field | Type | Description |
|-------|------|-------------|
| `lowLatencyJourneySynthesis` | boolean | Only for Journey voices. If false, the synthesis will be context aware and have higher latency. |

## Timepoint

This contains a mapping between a certain point in the input text and a corresponding time in the output audio.

| Field | Type | Description |
|-------|------|-------------|
| `markName` | string | Timepoint name as received from the client within `<mark>` tag. |
| `timeSeconds` | number | Time offset in seconds from the start of the synthesized audio. |
