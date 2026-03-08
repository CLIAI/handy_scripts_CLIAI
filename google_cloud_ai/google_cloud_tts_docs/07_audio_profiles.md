---
title: "Use Device Profiles for Generated Audio"
source_url: "https://cloud.google.com/text-to-speech/docs/audio-profiles"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/audio-profiles"
fetched_at: "2026-03-09"
api_version: "v1"
tags: ["audio-profiles", "device-profiles", "effects", "audio-config", "tts"]
---

# Use Device Profiles for Generated Audio

You can optimize the synthetic speech produced by Cloud Text-to-Speech for playback on different types of hardware. For example, if your application's synthesized speech is primarily intended for playback on a small, wearable device, you can have Cloud Text-to-Speech generate audio that's optimized to that class of device.

## Available Audio Profiles

The following table lists the available audio profiles:

| Audio profile ID | Optimized for |
|---|---|
| `wearable-class-device` | Smart watches and other wearables, like Apple Watch, Wear OS watch |
| `handset-class-device` | Smartphones, like Google Pixel, Samsung Galaxy, Apple iPhone |
| `headphone-class-device` | Earbuds or headphones for audio playback, like Sennheiser headphones |
| `small-bluetooth-speaker-class-device` | Small home speakers, like Google Home Mini |
| `medium-bluetooth-speaker-class-device` | Smart home speakers, like Google Home |
| `large-home-entertainment-class-device` | Home entertainment systems or smart TVs, like Google Home Max, LG TV |
| `large-automotive-class-device` | Car speakers |
| `telephony-class-application` | Interactive Voice Response (IVR) systems |

## Using Audio Profiles

To apply an audio profile, set the `effectsProfileId` field in the `AudioConfig` object of your synthesis request. You can apply multiple device profiles to the same synthetic speech. The profiles are applied in the order specified.

**Important:** Avoid specifying the same profile more than once, as you can have undesirable results by applying the same profile multiple times.

The audio profile is optional. If you don't provide one, the resulting speech will not be modified.

**Note:** Each audio profile is optimized for a specific class of device. However, the make and model of the device used to tune the profile may not match your users' playback devices exactly. You may need to experiment with different profiles to find the optimal setting.

## Code Examples

### Python

```python
def synthesize_text_with_audio_profile():
    from google.cloud import texttospeech

    text = "hello"
    output = "output.mp3"
    effects_profile_id = "telephony-class-application"
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US")
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=[effects_profile_id],
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    with open(output, "wb") as out:
        out.write(response.audio_content)
```

### Go

The Go implementation uses the `texttospeech` package from `cloud.google.com/go/texttospeech/apiv1` and sets the `EffectsProfileId` field in the `AudioConfig` struct.

### Java

The Java implementation uses the `com.google.cloud.texttospeech.v1` package and calls `.addEffectsProfileId()` on the `AudioConfig.Builder`.

### Node.js

The Node.js implementation uses the `@google-cloud/text-to-speech` package and includes the `effectsProfileId` array in the `audioConfig` object of the request.
