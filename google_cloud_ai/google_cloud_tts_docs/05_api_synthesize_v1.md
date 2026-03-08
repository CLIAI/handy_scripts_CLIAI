---
title: "Method: text.synthesize (REST API v1)"
source_url: "https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize"
fetched_at: "2026-03-09"
api_version: "v1"
tags: ["api", "rest", "synthesize", "request", "response", "audio-config", "voice-selection"]
---

# Method: text.synthesize

Synthesizes speech synchronously: receive results after all text input has been processed.

## HTTP Request

```
POST https://texttospeech.googleapis.com/v1/text:synthesize
```

The URL uses [gRPC Transcoding](https://google.aip.dev/127) syntax.

## Request Body

The request body contains data with the following structure:

| Field | Type | Description |
| --- | --- | --- |
| `input` | [SynthesisInput](#synthesisinput) | Required. The Synthesizer requires either plain text or SSML as input. |
| `voice` | [VoiceSelectionParams](#voiceselectionparams) | Required. The desired voice of the synthesized audio. |
| `audioConfig` | [AudioConfig](#audioconfig) | Required. The configuration of the synthesized audio. |
| `advancedVoiceOptions` | [AdvancedVoiceOptions](#advancedvoiceoptions) | Optional. Advanced voice options. |

## Response Body

If successful, the response body contains data with the following structure:

| Field | Type | Description |
| --- | --- | --- |
| `audioContent` | string (bytes format) | The audio data bytes encoded as specified in the request, including the header for encodings that are wrapped in containers (e.g. MP3, OGG_OPUS). For LINEAR16 audio, the response includes a WAV header. Note: as with all bytes fields, protobuffers use a pure binary representation, whereas JSON representations use base64. A base64-encoded string. |

## Authorization

Requires the following OAuth scope:

* `https://www.googleapis.com/auth/cloud-platform`

## SynthesisInput

Contains text input to be synthesized. Either `text` or `ssml` must be supplied. Supplying both or neither returns `google.rpc.Code.INVALID_ARGUMENT`. The input size limit is 5000 bytes (for `text`) or 5000 bytes (for `ssml`, including tags).

| Field | Type | Description |
| --- | --- | --- |
| `text` | string | The raw text to be synthesized. May not be longer than 5000 bytes. |
| `ssml` | string | The SSML document to be synthesized. The SSML document must be valid and well-formed. May not be longer than 5000 bytes. |
| `customPronunciations` | [CustomPronunciations](#custompronunciations) | Optional. The pronunciation customizations to be applied to the input. If this is set, the input will be synthesized using the given pronunciation customizations. |

**Note:** `text` and `ssml` are a union field; only one should be set.

## VoiceSelectionParams

Description of which voice to use for a synthesis request.

| Field | Type | Description |
| --- | --- | --- |
| `languageCode` | string | Required. The language (and potentially also the region) of the voice expressed as a [BCP-47](https://www.rfc-editor.org/rfc/bcp/bcp47.txt) language tag, e.g. "en-US". This should not include a script tag (e.g. use "cmn-cn" rather than "cmn-Hant-cn"), because the script will be inferred from the input provided. The TTS service will use this parameter to help choose an appropriate voice. Note that the TTS service may choose a voice with a slightly different locale than the one selected; it may substitute a different region (e.g. using en-US rather than en-CA if there isn't a Canadian voice available), or even a different language, e.g. using "nb" (Norwegian Bokmal) instead of "no" (Norwegian). |
| `name` | string | The name of the voice. If both the name and the gender are not set, the service will choose a voice based on the other parameters such as languageCode. |
| `ssmlGender` | [SsmlVoiceGender](#ssmlvoicegender) | The preferred gender of the voice. If not set, the service will choose a voice based on the other parameters such as languageCode and name. Note that this is only a preference, not a requirement; if a voice of the appropriate gender is not available, the synthesizer should substitute a voice with a different gender rather than failing the request. |
| `customVoice` | [CustomVoiceParams](#customvoiceparams) | The configuration for a custom voice. If [CustomVoiceParams.model] is set, the service will choose the custom voice matching the specified configuration. |

## SsmlVoiceGender

Gender of the voice as described in [SSML voice element](https://www.w3.org/TR/speech-synthesis11/#edef_voice).

| Enum | Description |
| --- | --- |
| `SSML_VOICE_GENDER_UNSPECIFIED` | An unspecified gender. In VoiceSelectionParams, this means that the client doesn't care which gender the selected voice will have. |
| `MALE` | A male voice. |
| `FEMALE` | A female voice. |
| `NEUTRAL` | A gender-neutral voice. This voice is not yet supported. |

## AudioConfig

Description of audio data to be synthesized.

| Field | Type | Description |
| --- | --- | --- |
| `audioEncoding` | [AudioEncoding](#audioencoding) | Required. The format of the audio byte stream. |
| `speakingRate` | number | Optional. Input only. Speaking rate/speed, in the range [0.25, 4.0]. 1.0 is the normal native speed supported by the specific voice. 2.0 is twice as fast, and 0.5 is half as fast. If unset (0.0), defaults to the native 1.0 speed. Any other values < 0.25 or > 4.0 will return an error. This field is not supported by Chirp 3 HD voices. |
| `pitch` | number | Optional. Input only. Speaking pitch, in the range [-20.0, 20.0]. 20 means increase 20 semitones from the original pitch. -20 means decrease 20 semitones from the original pitch. This field is not supported by Chirp 3 HD voices. |
| `volumeGainDb` | number | Optional. Input only. Volume gain (in dB) of the normal native volume supported by the specific voice, in the range [-96.0, 16.0]. If unset, or set to a value of 0.0 (dB), will play at normal native signal amplitude. A value of -6.0 (dB) will play at approximately half the amplitude of the normal native signal amplitude. A value of +6.0 (dB) will play at approximately twice the amplitude of the normal native signal amplitude. Strongly recommend not to exceed +10 (dB) as there's usually no effective increase in loudness for any value greater than that. |
| `sampleRateHertz` | integer | Optional. The synthesis sample rate (in hertz) for this audio. When this is specified in SynthesizeSpeechRequest, if this is different from the voice's natural sample rate, then the synthesizer will honor this request by converting to the desired sample rate (which might result in worse audio quality), unless the specified sample rate is not supported for the encoding, in which case it will fail the request and return `google.rpc.Code.INVALID_ARGUMENT`. |
| `effectsProfileId` | string[] | Optional. Input only. An identifier which selects 'audio effects' profiles that are applied on (post synthesized) text to speech. Effects are applied on top of each other in the order they are given. See [audio profiles](https://cloud.google.com/text-to-speech/docs/audio-profiles) for current supported profile ids. |

## AudioEncoding

Configuration to set up audio encoder. The encoding determines the output audio format that we'd like.

| Enum | Description |
| --- | --- |
| `AUDIO_ENCODING_UNSPECIFIED` | Not specified. Will return result `google.rpc.Code.INVALID_ARGUMENT`. |
| `LINEAR16` | Uncompressed 16-bit signed little-endian samples (Linear PCM). Audio content returned as LINEAR16 also contains a WAV header. |
| `MP3` | MP3 audio at 32kbps. |
| `OGG_OPUS` | Opus encoded audio wrapped in an ogg container. The result will be a file which can be played natively on Android, and in browsers (at least Chrome and Firefox). The quality of the encoding is considerably higher than MP3 while using approximately the same bitrate. |
| `MULAW` | 8-bit samples that compand 14-bit audio samples using G.711 PCMU/mu-law. Audio content returned as MULAW also contains a WAV header. |
| `ALAW` | 8-bit samples that compand 14-bit audio samples using G.711 PCMA/A-law. Audio content returned as ALAW also contains a WAV header. Not supported by Chirp 3 HD voices. |

## CustomVoiceParams

Description of the custom voice to be synthesized.

| Field | Type | Description |
| --- | --- | --- |
| `model` | string | Required. The name of the AutoML model that synthesizes the custom voice. |
| `reportedUsage` | ReportedUsage | Optional. Deprecated. The usage of the synthesized audio to be reported. |

## ReportedUsage (Deprecated)

| Enum | Description |
| --- | --- |
| `REPORTED_USAGE_UNSPECIFIED` | Request with reported usage unspecified will be rejected. |
| `REALTIME` | For scenarios where the synthesized audio is not downloadable and can only be used in a real-time application (e.g., IVR). |
| `OFFLINE` | For scenarios where the synthesized audio is downloadable and can be reused (e.g., media creation). |

## AdvancedVoiceOptions

Used for advanced voice options.

| Field | Type | Description |
| --- | --- | --- |
| `lowLatencyJourneySynthesis` | boolean | Optional. Only for Journey voices. If false, the synthesis will be context aware and have higher latency. |

## CustomPronunciations

A collection of pronunciation customizations.

| Field | Type | Description |
| --- | --- | --- |
| `pronunciations` | [CustomPronunciationParams](#custompronunciationparams)[] | The pronunciation customizations applied to the input. |

## CustomPronunciationParams

Parameters for a custom pronunciation.

| Field | Type | Description |
| --- | --- | --- |
| `phrase` | string | The phrase to which the customization will be applied. The phrase can be multiple words (in the case of proper nouns etc), but should not span to a whole sentence. |
| `phoneticEncoding` | PhoneticEncoding | The phonetic encoding of the phrase. |
| `pronunciation` | string | The pronunciation of the phrase. This must be in the phonetic encoding specified above. |

## PhoneticEncoding

The phonetic encoding of the phrase.

| Enum | Description |
| --- | --- |
| `PHONETIC_ENCODING_UNSPECIFIED` | Not specified. |
| `PHONETIC_ENCODING_IPA` | IPA. See https://en.wikipedia.org/wiki/International_Phonetic_Alphabet |
| `PHONETIC_ENCODING_X_SAMPA` | X-SAMPA. See https://en.wikipedia.org/wiki/X-SAMPA |

## Example Request

```json
{
  "input": {
    "text": "Hello, world!"
  },
  "voice": {
    "languageCode": "en-US",
    "name": "en-US-Wavenet-D",
    "ssmlGender": "MALE"
  },
  "audioConfig": {
    "audioEncoding": "MP3"
  }
}
```

## Example with SSML

```json
{
  "input": {
    "ssml": "<speak>Hello <break time=\"500ms\"/> world!</speak>"
  },
  "voice": {
    "languageCode": "en-US",
    "name": "en-US-Neural2-F"
  },
  "audioConfig": {
    "audioEncoding": "OGG_OPUS",
    "speakingRate": 1.0,
    "pitch": 0.0,
    "volumeGainDb": 0.0
  }
}
```

## cURL Example

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello, world!"
    },
    "voice": {
      "languageCode": "en-US",
      "name": "en-US-Wavenet-D",
      "ssmlGender": "MALE"
    },
    "audioConfig": {
      "audioEncoding": "MP3"
    }
  }' \
  "https://texttospeech.googleapis.com/v1/text:synthesize"
```

The response will contain `audioContent` as a base64-encoded string. Decode it to get the audio file:

```bash
# Save response and decode
echo "$RESPONSE" | jq -r '.audioContent' | base64 --decode > output.mp3
```
