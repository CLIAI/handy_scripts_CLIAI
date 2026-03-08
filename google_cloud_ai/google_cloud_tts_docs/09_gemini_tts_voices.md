---
title: "Supported Voices and Languages"
source_url: "https://cloud.google.com/text-to-speech/docs/voices"
original_attempted_url: "https://cloud.google.com/text-to-speech/docs/tts-voices"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/voices"
fetched_at: "2026-03-09"
api_version: "v1"
tags: ["voices", "languages", "chirp3", "studio", "neural2", "wavenet", "standard", "tts"]
note: "Original URL /tts-voices returned 404; fetched from /voices instead"
---

# Supported Voices and Languages

Cloud Text-to-Speech generates audio with natural, human-like quality, which creates speech that sounds like a real person. To use these voices, specify a voice when sending a synthesis request.

## Voice Types Overview

| Voice Type | Intended For | Launch Stage | Controllability | Streaming |
|---|---|---|---|---|
| Chirp 3: HD voices | Conversational Agents | GA | - | Yes |
| Studio (Two speakers) | Media: Discussions and Interviews | Experimental | - | - |
| Studio (One speaker) | Media: Narration | GA | SSML | - |
| Neural2 | General purpose | GA | SSML | - |
| WaveNet | General purpose | GA | SSML | - |
| Standard | Cost efficient | GA | SSML | - |

## Chirp 3: HD Voices

These voices are driven by technology that captures nuances in human intonation, which make conversations more engaging. They come in 30 distinct styles across many languages, suitable for real-time and standard applications. They support advanced audio controls and low-latency real-time communication via text streaming.

**Important limitations:** Chirp 3: HD voices don't support SSML input, speaking rate, pitch-audio parameters, or A-Law audio encoding.

## Studio Voices

### Multispeaker Studio Voices

Studio multispeaker voices enable discussions and interviews with the multispeaker studio voices, which is based on the same technology behind Chirp 3: HD voices.

### Single Speaker Studio Voices

Studio voices designed for news reading support SSML, except for the following tags:

* `<mark>`
* `<emphasis>`
* `<prosody pitch>`
* `<lang>`

## Neural2 Voices

Neural2 voices are based on the same technology used to create a Custom Voice. They allow use of Custom Voice technology without training a custom voice and are available in global and single region endpoints.

## WaveNet Voices

WaveNet-generated voices are more warm and human-like than other synthetic voices. These models have been trained using raw audio samples of actual humans speaking, resulting in synthetic speech with more human-like emphasis and inflection on syllables, phonemes, and words.

## Standard Voices

Standard voices use parametric text-to-speech technology, which generates audio by passing outputs through signal processing algorithms known as vocoders.

## Supported Languages and Voices

The full list of supported languages includes (with voice types available in Standard, WaveNet, Neural2, Studio, and Chirp 3: HD categories):

* Arabic (ar-XA)
* Bengali (bn-IN)
* Bulgarian (bg-BG)
* Catalan (ca-ES)
* Chinese - Hong Kong (yue-HK)
* Chinese - Mandarin (cmn-CN, cmn-TW)
* Croatian (hr-HR)
* Czech (cs-CZ)
* Danish (da-DK)
* Dutch (Belgium) (nl-BE)
* Dutch (Netherlands) (nl-NL)
* English (Australia) (en-AU)
* English (India) (en-IN)
* English (UK) (en-GB)
* English (US) (en-US)
* Estonian (et-EE)
* Filipino (fil-PH)
* Finnish (fi-FI)
* French (Canada) (fr-CA)
* French (France) (fr-FR)
* Galician (gl-ES)
* German (de-DE)
* Greek (el-GR)
* Gujarati (gu-IN)
* Hebrew (he-IL)
* Hindi (hi-IN)
* Hungarian (hu-HU)
* Icelandic (is-IS)
* Indonesian (id-ID)
* Italian (it-IT)
* Japanese (ja-JP)
* Kannada (kn-IN)
* Korean (ko-KR)
* Latvian (lv-LV)
* Lithuanian (lt-LT)
* Malay (ms-MY)
* Malayalam (ml-IN)
* Marathi (mr-IN)
* Norwegian Bokmal (nb-NO)
* Polish (pl-PL)
* Portuguese (Brazil) (pt-BR)
* Portuguese (Portugal) (pt-PT)
* Punjabi (pa-IN)
* Romanian (ro-RO)
* Russian (ru-RU)
* Serbian (sr-RS)
* Slovak (sk-SK)
* Spanish (es-ES, es-US)
* Swedish (sv-SE)
* Tamil (ta-IN)
* Telugu (te-IN)
* Thai (th-TH)
* Turkish (tr-TR)
* Ukrainian (uk-UA)
* Vietnamese (vi-VN)

### Voice Naming Convention

Voice names follow the pattern: `{language-code}-{VoiceType}-{VoiceName}`

Examples:

* `en-US-Chirp3-HD-Achernar` - Chirp 3 HD voice
* `en-US-Studio-M` - Studio voice (Male)
* `en-US-Neural2-A` - Neural2 voice
* `en-US-Wavenet-A` - WaveNet voice
* `en-US-Standard-A` - Standard voice

### SSML Gender

Each voice is designated with an SSML gender (MALE or FEMALE) for selection purposes.

### Chirp 3: HD Voice Styles (30 voices)

Chirp 3: HD voices are available with named styles (e.g., Achernar, Aoede, Autonoe, Callirrhoe, Charon, Despina, Erinome, Fenrir, Galateo, Iapetus, Kore, Leda, Orus, Phoebe, Proteus, Rasalgethi, Sadachbia, Schedar, Sulafat, Umbriel, Vindemiatrix, Zubenelgenubi, and others).

These named voices are available across many of the supported languages listed above.
