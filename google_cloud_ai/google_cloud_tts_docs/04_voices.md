---
title: "Supported voices and languages"
source_url: "https://cloud.google.com/text-to-speech/docs/voices"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/voices"
fetched_at: "2026-03-09"
api_version: "v1"
tags: ["voices", "languages", "chirp3", "wavenet", "neural2", "studio", "standard"]
---

# Supported Voices and Languages

Google Cloud Text-to-Speech generates audio with natural, human-like quality, which creates speech that sounds like a real person.

> **Note:** The full voice listing table on the source page is very large (60+ languages, hundreds of voice entries). This document captures the structure and key information. For the complete per-voice table, refer to the [source page](https://cloud.google.com/text-to-speech/docs/voices) or use the [`voices.list` API endpoint](https://cloud.google.com/text-to-speech/docs/reference/rest/v1/voices/list).

## Voice Types Overview

The platform provides five primary voice categories:

### 1. Chirp 3: HD voices

Premium voices designed for conversational agents, available in 30 distinct styles with streaming support.

* Captures nuances in human intonation, which make conversations more engaging
* Supports advanced audio controls and low-latency real-time communication using text streaming
* **Does NOT support:** SSML input, speaking rate/pitch parameters, or A-Law audio encoding

### 2. Studio voices

Two options available:

* **Multispeaker voices** - for discussions and interviews
* **Single-speaker voices** - for narration

Studio voices support most SSML tags except for `<mark>`, `<emphasis>`, `<prosody pitch>`, and `<lang>`.

### 3. Neural2 voices

General-purpose voices based on custom voice technology. These are premium voices suitable for a wide range of applications.

### 4. WaveNet voices

Premium synthetic voices trained on raw audio samples of human speakers. These voices are potentially more warm and human-like than other synthetic voices due to training on actual human speech samples.

### 5. Standard voices

Cost-efficient options using parametric text-to-speech technology. Suitable for applications where cost is a primary concern.

## Supported Languages

The service supports 60+ languages including (but not limited to):

| Language | Code |
| --- | --- |
| Afrikaans (South Africa) | af-ZA |
| Arabic | ar-XA |
| Basque (Spain) | eu-ES |
| Bengali (India) | bn-IN |
| Bulgarian (Bulgaria) | bg-BG |
| Catalan (Spain) | ca-ES |
| Chinese (Mandarin, Hong Kong) | yue-HK |
| Chinese (Mandarin, Mainland) | cmn-CN |
| Chinese (Mandarin, Taiwan) | cmn-TW |
| Czech (Czech Republic) | cs-CZ |
| Danish (Denmark) | da-DK |
| Dutch (Belgium) | nl-BE |
| Dutch (Netherlands) | nl-NL |
| English (Australia) | en-AU |
| English (India) | en-IN |
| English (UK) | en-GB |
| English (US) | en-US |
| Filipino (Philippines) | fil-PH |
| Finnish (Finland) | fi-FI |
| French (Canada) | fr-CA |
| French (France) | fr-FR |
| Galician (Spain) | gl-ES |
| German (Germany) | de-DE |
| Greek (Greece) | el-GR |
| Gujarati (India) | gu-IN |
| Hebrew (Israel) | he-IL |
| Hindi (India) | hi-IN |
| Hungarian (Hungary) | hu-HU |
| Icelandic (Iceland) | is-IS |
| Indonesian (Indonesia) | id-ID |
| Italian (Italy) | it-IT |
| Japanese (Japan) | ja-JP |
| Kannada (India) | kn-IN |
| Korean (South Korea) | ko-KR |
| Latvian (Latvia) | lv-LV |
| Lithuanian (Lithuania) | lt-LT |
| Malay (Malaysia) | ms-MY |
| Malayalam (India) | ml-IN |
| Mandarin Chinese | cmn-CN |
| Marathi (India) | mr-IN |
| Norwegian (Norway) | nb-NO |
| Polish (Poland) | pl-PL |
| Portuguese (Brazil) | pt-BR |
| Portuguese (Portugal) | pt-PT |
| Punjabi (India) | pa-IN |
| Romanian (Romania) | ro-RO |
| Russian (Russia) | ru-RU |
| Serbian (Cyrillic) | sr-RS |
| Slovak (Slovakia) | sk-SK |
| Spanish (Spain) | es-ES |
| Spanish (US) | es-US |
| Swedish (Sweden) | sv-SE |
| Tamil (India) | ta-IN |
| Telugu (India) | te-IN |
| Thai (Thailand) | th-TH |
| Turkish (Turkey) | tr-TR |
| Ukrainian (Ukraine) | uk-UA |
| Vietnamese (Vietnam) | vi-VN |

## Voice Naming Convention

Voice names follow the pattern: `{language_code}-{VoiceType}-{Letter}`

Examples:

* `en-US-Standard-A` - US English, Standard voice, variant A
* `en-US-Wavenet-D` - US English, WaveNet voice, variant D
* `en-US-Neural2-F` - US English, Neural2 voice, variant F
* `en-US-Studio-M` - US English, Studio voice, variant M
* `en-US-Chirp3-HD-Achernar` - US English, Chirp 3 HD voice named Achernar

## Listing Voices Programmatically

Use the `voices.list` API endpoint to get the complete current listing:

```
GET https://texttospeech.googleapis.com/v1/voices
```

Or with a language filter:

```
GET https://texttospeech.googleapis.com/v1/voices?languageCode=en-US
```
