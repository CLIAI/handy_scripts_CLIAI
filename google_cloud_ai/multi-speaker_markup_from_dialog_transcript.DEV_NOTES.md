# DEV_NOTES: multi-speaker_markup_from_dialog_transcript.py

Development notes, gotchas, and architecture decisions.

## API Quality Constraints (Important)

The Google Cloud TTS API **does not provide bitrate, quality level, or codec
tuning controls**. The `AudioConfig` exposes exactly these parameters:

| Parameter | Python field | Our flag | Range |
|-----------|-------------|----------|-------|
| Encoding format | `audio_encoding` | `-e` | LINEAR16, MP3, OGG_OPUS, MULAW, ALAW |
| Speaking rate | `speaking_rate` | `--rate` | 0.25 – 4.0 |
| Pitch | `pitch` | `--pitch` | -20.0 – 20.0 semitones |
| Volume gain | `volume_gain_db` | `--volume` | -96.0 – 16.0 dB |
| Sample rate | `sample_rate_hertz` | `--sample-rate` | varies by encoding |
| Device profile | `effects_profile_id` | `--audio-profile` | 8 profiles |

That's the **complete set**. No `bitrate`, `quality`, or `codec_options` fields
exist. Verified against:

* REST API v1: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
* Local archive: `google_cloud_tts_docs/05_api_synthesize_v1.md` (AudioConfig section)

### MP3 is 32kbps fixed

From the API docs: `MP3` is described as "MP3 audio at 32kbps." There is no way
to request higher MP3 bitrate. For better quality lossy audio, use OGG_OPUS which
the docs describe as "considerably higher than MP3 while using approximately the
same bitrate."

### Chirp 3 HD voice limitations

Chirp 3: HD voices (the Gemini TTS voices we use) do NOT support:

* SSML input
* `speaking_rate` / `pitch` parameters
* A-Law audio encoding

This means `--rate`, `--pitch`, and `-e alaw` may be silently ignored or error
when used with `gemini-2.5-flash-tts` / `gemini-2.5-pro-tts`. This is documented
in `google_cloud_tts_docs/04_voices.md` and `09_gemini_tts_voices.md`.

**TODO:** Consider adding a warning when `--rate` or `--pitch` are used with
Gemini TTS models, since they likely have no effect.

## Architecture

### Evolution: Studio MultiSpeaker → Gemini TTS

The script was originally written for `en-US-Studio-MultiSpeaker` which uses
`MultiSpeakerMarkup` with speaker designators R-Y (up to 8 speakers). That API
is experimental and requires project allowlisting (403 error without it).

Pivoted to Gemini TTS (`gemini-2.5-flash-tts`) which:

* Is GA (no allowlist needed)
* Supports 21+ languages (vs English-only)
* Uses `MultiSpeakerVoiceConfig` with `MultispeakerPrebuiltVoice`
* Has 30 named voices (vs 8 anonymous designators)
* Uses natural language prompts instead of SSML

See `google_cloud_tts_docs/ASSISTANT_PROMPT--studio_multispeaker_allowlist.md`
for the allowlisting research.

### Freeform text format

Gemini TTS multi-speaker uses a freeform text format:

```
Speaker1: dialogue line
Speaker2: response line
```

The `speaker_alias` in `MultispeakerPrebuiltVoice` must match exactly the name
before the colon in the text. This is why `parse_input()` preserves speaker
names as-is (after stripping markdown).

### Byte limit and chunking

**IMPORTANT**: The limits differ by mode. Freeform multi-speaker is the most
restrictive — it is NOT 8000 bytes.

* **Freeform multi-speaker (our mode)**: 4000 bytes for dialogue text
* **Single-speaker Gemini TTS**: text ≤ 4000 + prompt ≤ 4000, combined ≤ 8000
* **Structured MultiSpeakerMarkup**: markup ≤ 4000 + prompt ≤ 4000, combined ≤ 8000
* **Vertex AI API**: 8000 bytes unified (contents field)
* **Effective limit**: We use 3500 bytes (`CHUNK_MAX_BYTES`) for safety margin
* **Freeform overhead**: Each turn adds `"Speaker: "` prefix + newline, so
  `freeform_bytes > input_bytes` (raw dialogue text)
* **Chunking**: Splits at turn boundaries. Each chunk gets the full prompt and
  all speaker voice configs (unused speakers in a chunk are harmless)
* **Audio concatenation**:
  - MP3: direct byte concatenation (frames are self-contained)
  - WAV/LINEAR16: strip 44-byte headers from chunks 2+, fix size fields
  - OGG_OPUS: best-effort concatenation (may have minor artifacts at boundaries)
  - MULAW/ALAW: raw concatenation

Source: https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers

### Global `args` pattern

Uses a module-level `global args` set in `main()`. This is intentional for
script simplicity — `emit()`, `emit_error()`, `log_verbose()` all need access
to `args.jsonl` and `args.verbose`. For a library, this would be refactored to
pass config explicitly.

## Dependencies

```
google-cloud-texttospeech>=2.31.0
```

Version 2.31.0+ is required for:

* `MultiSpeakerVoiceConfig`
* `MultispeakerPrebuiltVoice`
* `model_name` field in `VoiceSelectionParams`

## Testing

```bash
# Dry run (no API call) — verify parsing and config
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i examples/teacher_student_german_lesson.txt -n -v \
  --voices Orus,Aoede -o /dev/null -f

# Dry run with chunking estimate
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i large_dialogue.txt -n --chunk -o /dev/null -f

# JSONL dry run — verify event schema
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i examples/teacher_student_german_lesson.txt -n --jsonl -o /dev/null -f

# Actual generation (requires GCP credentials)
uv run multi-speaker_markup_from_dialog_transcript.py \
  -i examples/teacher_student_german_lesson.txt \
  -o examples/teacher_student_german_lesson.ogg -e ogg \
  --voices Orus,Aoede \
  -p "Patient, encouraging language teacher with an enthusiastic student"
```

## Potential Future Work

* **Rate/pitch warning**: Detect Chirp 3 HD models and warn about unsupported params
* **Per-speaker voice config**: JSON config file for per-speaker rate/pitch/voice
* **SSML in turns**: May require switching to non-Chirp voices
* **Streaming**: Gemini TTS supports streaming, not yet implemented
* **Temperature control**: Available via Vertex AI API, not Cloud TTS API
* **OGG chunking**: Proper Ogg page concatenation for artifact-free chunk merging
