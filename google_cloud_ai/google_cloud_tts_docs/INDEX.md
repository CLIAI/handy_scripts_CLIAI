---
title: "Google Cloud TTS Documentation Archive - Index"
created_at: "2026-03-09"
purpose: "Local archive of Google Cloud Text-to-Speech API documentation for feature gap analysis"
---

# Google Cloud TTS Documentation Archive

Archived documentation from Google Cloud Text-to-Speech API, fetched 2026-03-09.

## Documents

| # | File | Title | API Version | Tags |
|---|------|-------|-------------|------|
| 1 | [01_overview.md](01_overview.md) | Cloud Text-to-Speech Documentation | general | overview, getting-started |
| 2 | [02_multi_speaker_dialogue.md](02_multi_speaker_dialogue.md) | Create Dialogue with Multi-Speakers | v1 | multi-speaker, dialogue |
| 3 | [03_ssml_reference.md](03_ssml_reference.md) | SSML Reference | v1 | ssml, phoneme, prosody |
| 4 | [04_voices.md](04_voices.md) | Supported Voices and Languages | v1 | voices, chirp3, studio, neural2 |
| 5 | [05_api_synthesize_v1.md](05_api_synthesize_v1.md) | REST API text.synthesize (v1) | v1 | api, AudioConfig, synthesize |
| 6 | [06_api_synthesize_v1beta1.md](06_api_synthesize_v1beta1.md) | REST API text.synthesize (v1beta1) | v1beta1 | api, beta, synthesize |
| 7 | [07_audio_profiles.md](07_audio_profiles.md) | Audio Device Profiles | v1 | audio-profiles, effects |
| 8 | [08_python_client_library.md](08_python_client_library.md) | Client Libraries | general | python, sdk, quickstart |
| 9 | [09_gemini_tts_voices.md](09_gemini_tts_voices.md) | Gemini TTS / Chirp3 Voices | v1 | gemini, chirp3, voices |

## Quick Reference

### Voice Types (by technology)

* **Chirp 3 HD** - Latest, 30 named styles, streaming support, no SSML
* **Studio MultiSpeaker** - Multi-speaker dialogues (experimental, en-US only)
* **Studio** - Broadcast quality, partial SSML
* **Neural2** - General purpose, full SSML
* **WaveNet** - Premium quality, full SSML
* **Standard** - Cost-efficient, full SSML

### AudioConfig Parameters

* `audioEncoding` - LINEAR16, MP3, OGG_OPUS, MULAW, ALAW
* `speakingRate` - 0.25 to 4.0 (default: 1.0)
* `pitch` - -20.0 to 20.0 semitones (default: 0.0)
* `volumeGainDb` - -96.0 to 16.0 dB (default: 0.0)
* `sampleRateHertz` - e.g., 16000, 24000, 48000
* `effectsProfileId` - Device-specific audio optimization
