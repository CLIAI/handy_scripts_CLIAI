# Speechmatics Speaker Diarization

## Overview

Speechmatics diarization "separates a transcript into distinct speakers or channels, so you can clearly see who said what." This feature applies to conversations, meetings, interviews, and multi-channel recordings.

## Three Diarization Modes

1. **Speaker Diarization** — Identifies each speaker by their voice characteristics, ideal for single-recording multi-speaker scenarios

2. **Channel Diarization** — Processes each audio channel separately, useful when speakers occupy distinct channels

3. **Channel & Speaker Diarization** — Combines both approaches (realtime only), handling multiple speakers across multiple channels

## Implementation Options

* **Realtime Diarization** — For live streaming audio in video conferencing and conversational AI applications
* **Batch Diarization** — For pre-recorded files including call recordings and podcasts

## Primary Use Cases

Applications include:

* Call centers
* Video conferences
* Medical consultations
* Media production

Where speaker identification supports training, compliance, quality assurance, and content analysis.

## Batch Speaker Diarization Configuration

Enable by setting `diarization` to `"speaker"` in the transcription config. Output includes speaker labels (S1, S2, etc.) on each word and punctuation object, with "UU" used for unidentified speakers.

### Customization Options

**Speaker Sensitivity** - Adjust detection precision via `speaker_sensitivity` (0-1 range, default 0.5). Higher values increase unique speaker detection.

**Prefer Current Speaker** - Set `prefer_current_speaker: true` to reduce false switches between similar-sounding speakers, though this may miss shorter speaker transitions.

**Max Speakers** - Prevent excessive speaker detection by setting `max_speakers` (minimum value: 2). By default, there is no limit on the number of speakers.

**Punctuation Integration** - The system uses sentence boundaries to refine speaker assignments for improved accuracy. Speaker diarization uses punctuation to improve accuracy. Small corrections are applied to speaker labels based on sentence boundaries. For example, if the system initially assigns 9 words in a sentence to S1 and 1 word to S2, the lone S2 word may be corrected to S1. This adjustment only works when punctuation is enabled.

## Batch Channel Diarization Configuration

Set `diarization` to `"channel"` and optionally add custom labels:

```json
"channel_diarization_labels": ["Agent", "Caller"]
```

Output includes a `channel` property identifying each speaker. Supports up to 100 separate input files.

## Realtime Speaker Diarization

Enable by setting `"diarization": "speaker"` in the transcription config. Results include speaker labels (S1, S2, etc.) or "UU" when speakers cannot be identified.

**Key Settings:**

* **Speaker Sensitivity** (0-1, default 0.5): Higher values increase likelihood of detecting more unique speakers
* **Prefer Current Speaker**: Reduces false switches between similar-sounding speakers
* **Max Speakers**: Prevents excessive speaker detection (minimum value: 2)

### Message Example - Start Recognition

```json
{
  "type": "transcription",
  "transcription_config": {
    "language": "en",
    "diarization": "speaker",
    "speaker_diarization_config": {
      "speaker_sensitivity": 0.6,
      "max_speakers": 10
    }
  }
}
```

## Realtime Channel Diarization Setup

Configure with `"diarization": "channel"` and provide `channel_diarization_labels` array. Send audio using `AddChannelAudio` messages with base64-encoded data.

**Limits:**

* SaaS: Maximum 2 channels
* On-prem: Depends on container configuration

### Message Example - Channel Audio

```json
{
  "message": "AddChannelAudio",
  "channel": "New_York",
  "data": "<base_64_encoded_data>"
}
```

## Speaker Labels

There are two types of labels:

* **S#** – S stands for speaker, and # is a sequential number identifying each speaker
* **UU** – Used when the speaker cannot be identified or diarization is not applied, for example, if background noise is transcribed as speech but no speaker can be determined

## Performance Impact

Diarization processing typically increases transcription time by 10-50%.

## Accuracy Notes

* Punctuation improves diarization accuracy through sentence boundary corrections
* Disabling punctuation via the `permitted_marks` setting in `punctuation_overrides` can reduce diarization accuracy
* Adjusting punctuation sensitivity can also affect how accurately speakers are identified
* The legacy Speaker Change Detection feature was deprecated in July 2024; use speaker diarization instead
