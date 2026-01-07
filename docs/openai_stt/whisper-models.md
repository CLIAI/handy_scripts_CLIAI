# OpenAI Whisper Models

## Overview

Whisper is a general-purpose speech recognition model developed by OpenAI, capable of:

* Multilingual speech recognition across 100+ languages
* Speech translation into English
* Spoken language identification
* Voice activity detection

## Model Sizes

| Model | Parameters | VRAM | Relative Speed |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | ~10x faster |
| base | 74M | ~1GB | ~7x faster |
| small | 244M | ~2GB | ~4x faster |
| medium | 769M | ~5GB | ~2x faster |
| large | 1550M | ~10GB | baseline |
| turbo | 809M | ~6GB | ~8x faster |

## English-Only Variants

English-only models (`.en` suffix) exist for:

* tiny.en
* base.en
* small.en
* medium.en

These variants offer better performance for English-only use cases.

## API Models

### whisper-1

The production API model powered by Whisper V2:

* **Best for**: General-purpose transcription
* **Output formats**: json, text, srt, verbose_json, vtt
* **Features**: Timestamps, language detection, prompting

### GPT-4o Transcription Models

Newer models with enhanced quality:

| Model | Description |
|-------|-------------|
| `gpt-4o-transcribe` | Higher quality transcription |
| `gpt-4o-mini-transcribe` | Lightweight, cost-effective |
| `gpt-4o-transcribe-diarize` | Speaker identification |

**Output formats**: json, text only (diarized_json for diarize model)

## Performance Considerations

### Model Selection

* **Speed priority**: Use `tiny` or `turbo`
* **Accuracy priority**: Use `large` or `gpt-4o-transcribe`
* **English only**: Consider `.en` variants for medium and smaller models
* **Speaker diarization**: Use `gpt-4o-transcribe-diarize`

### Translation Limitation

The turbo model returns the original language even when `--task translate` is specified. For translation tasks, use medium or large models.

## Local vs API

### API (whisper-1)

* No hardware requirements
* 25MB file size limit
* Pay-per-use pricing
* Consistent performance

### Local Whisper

* Requires GPU with sufficient VRAM
* No file size limits
* One-time setup cost
* Performance depends on hardware

## Supported Languages

Whisper was trained on 98 languages. Major supported languages include:

### Tier 1 (Excellent Quality)

English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Japanese, Korean, Mandarin

### Tier 2 (Good Quality)

Arabic, Hindi, Turkish, Vietnamese, Thai, Indonesian, Czech, Greek, Hungarian, Romanian, Swedish, Danish, Norwegian, Finnish

### Tier 3 (Acceptable Quality)

Many additional languages with varying quality based on training data availability.

## Best Practices

1. **Choose appropriate model size** based on accuracy/speed requirements
2. **Use English-only models** when transcribing only English
3. **Specify language** when known to improve accuracy
4. **Use prompting** for technical terms and context
5. **Consider GPT-4o models** for highest quality requirements
