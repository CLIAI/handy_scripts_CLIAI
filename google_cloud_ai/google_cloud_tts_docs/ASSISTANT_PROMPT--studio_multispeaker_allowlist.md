# Request: Enable Studio MultiSpeaker Voice on our GCP Project

## What we need

We need the **`en-US-Studio-MultiSpeaker`** voice enabled (allowlisted) on our
Google Cloud project for the **Cloud Text-to-Speech API**.

Currently we get this error when calling the API:

```
403 Multi-speaker voices are only available to allowlisted projects.
```

## Background

* The `en-US-Studio-MultiSpeaker` voice allows generating dialogue audio with
  up to 8 distinct speakers (designators R through Y) in a single API call.
* It was launched October 30, 2024 and is still classified as **Experimental**.
* There is **no self-service form, Console toggle, or API flag** to enable it.
* The only documented path is contacting Google Cloud Sales.

## Where to request

1. **Google Cloud Sales contact form**: https://cloud.google.com/contact
   - Mention: "Cloud Text-to-Speech API — request allowlist access for
     `en-US-Studio-MultiSpeaker` voice on project `[PROJECT_ID]`"
   - Include our use case (multi-speaker dialogue synthesis for [describe use])

2. **Google Issue Tracker** (secondary): https://issuetracker.google.com
   - File under Cloud Text-to-Speech component
   - Reference the 403 error and request allowlisting

## Community context

* Google Developer Forum thread confirms Sales-only path, no public form:
  https://discuss.google.dev/t/allowlist-for-studio-multispeaker-text-to-speech/193945
* GitHub issue #1681 (closed without granting access, just added warning banner):
  https://github.com/GoogleCloudPlatform/generative-ai/issues/1681
* No community members have publicly reported successfully getting allowlisted.

## Important context: Gemini TTS alternative exists

Google has since launched **Gemini TTS** (`gemini-2.5-flash-tts` and
`gemini-2.5-pro-tts`) which is **GA since September 2025** and requires **no
allowlist**. It supports multi-speaker dialogue with 30 named voices across
80+ languages.

We have **already verified Gemini TTS works** on our project — it produces
good multi-speaker dialogue audio. So if getting Studio MultiSpeaker
allowlisted proves difficult, we have a working fallback.

### Comparison

| Feature | Studio MultiSpeaker | Gemini TTS (Flash/Pro) |
|---------|---------------------|------------------------|
| Status | Experimental (allowlist) | GA (no restrictions) |
| Languages | en-US only | 21 GA + 65 preview |
| Max speakers | 8 per call | 2 per call |
| Voice control | SSML (deterministic) | Natural language prompts (stochastic) |
| Streaming | No | Yes |
| Pricing | ~$160/M characters | ~$0.50-1.00/M input tokens + $10-20/M output tokens |

### Why we still want Studio MultiSpeaker

* Supports **8 speakers** per call (vs 2 for Gemini TTS)
* **SSML deterministic control** (phoneme, prosody, emphasis)
* Existing script/pipeline built around the MultiSpeakerMarkup API
* Potentially useful for production workloads requiring reproducibility

## What to ask Sales specifically

> "We would like to request allowlist access for the `en-US-Studio-MultiSpeaker`
> voice in the Cloud Text-to-Speech API for project `[PROJECT_ID]`.
>
> We are aware this is an Experimental feature. Our use case is generating
> multi-speaker dialogue audio from text transcripts. We currently have a
> working pipeline using the `google-cloud-texttospeech` Python SDK that
> constructs `MultiSpeakerMarkup` with speaker turns, but receive a 403
> 'Multi-speaker voices are only available to allowlisted projects' error.
>
> Could you either:
> (a) add our project to the allowlist, or
> (b) direct us to the correct team/form for this request?
>
> We are also interested in understanding the timeline for this feature
> reaching GA status, or whether Google recommends migrating to Gemini TTS
> (`gemini-2.5-flash-tts`) for multi-speaker use cases instead."
