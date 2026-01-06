# Speechmatics Translation Feature

## Overview

Speechmatics enables audio translation into multiple languages through a single API call, supporting "over 30 languages" with both batch and real-time processing capabilities.

## Supported Language Pairs

Translation works bidirectionally between English and 33 target languages:

Bulgarian, Catalan, Mandarin, Czech, Danish, German, Greek, Spanish, Estonian, Finnish, French, Galician, Hindi, Croatian, Hungarian, Indonesian, Italian, Japanese, Korean, Lithuanian, Latvian, Malay, Dutch, Norwegian, Polish, Portuguese, Romanian, Russian, Slovakian, Slovenian, Swedish, Turkish, Ukrainian, and Vietnamese.

An additional pair exists: "Norwegian Bokmål to Nynorsk" (batch only).

## Configuration

Enable translation by adding this structure to your request:

```json
"translation_config": {
    "target_languages": ["es", "de"],
    "enable_partials": true
}
```

**Key constraint**: "You can configure up to five translation languages at a time."

## Output Formats

**Batch**: Returns a `translations` object containing arrays of translated segments per language code, including timing and speaker attribution.

**Real-time**: Streams `AddPartialTranslation` (lower latency) and `AddTranslation` (final, more accurate) messages per requested language.

## Best Practices

* Use enhanced operating point for higher accuracy
* Maintain default punctuation settings
* Account for processing time increases with additional languages
* Note: "Realtime sessions may have a 5-second delay when finalizing translations"

## Limitations

* Maximum 5 target languages per transcription
* Translations only in JSON format (not text/SRT)
* Reduced metadata compared to source language transcription

## Translation Unsupported Languages

Some languages are currently unsupported for translation:

Arabic, Bashkir, Belarusian, Welsh, Esperanto, Basque, Interlingua, Mongolian, Marathi, Tamil, Thai, Uyghur, Cantonese.
