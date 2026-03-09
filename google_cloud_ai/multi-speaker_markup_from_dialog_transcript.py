#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "google-cloud-texttospeech>=2.31.0",
# ]
# requires-python = ">=3.11"
# ///

# Generate multi-speaker dialogue audio using Gemini TTS
# https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
import os
import time
from google.cloud import texttospeech

# 30 Gemini TTS prebuilt voices - used for auto-assigning to speakers
GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

LLM_REFERENCE = r"""
# multi-speaker_markup_from_dialog_transcript.py — LLM/Agent Reference

## Purpose
Generate multi-speaker dialogue audio files from plain-text transcripts using
Google Cloud Gemini TTS. Designed for both interactive CLI use and programmatic
(--jsonl) pipeline integration.

## Prerequisites
- Google Cloud project with Text-to-Speech API enabled
- Application Default Credentials configured:
    gcloud auth application-default login
- Python >=3.11, or use `uv run` (dependencies auto-installed via PEP 723)

## Input Format
Plain text file with lines in `Speaker: dialogue text` format.
- Speaker names are extracted from the text before the first colon on each line.
- Markdown formatting on speaker names (**bold**, _italic_, etc.) is auto-stripped.
- Blank lines are ignored.
- Continuation lines (no colon) are appended to the previous speaker's turn.

### Example input file (examples/teacher_student_german_lesson.txt):
```
Teacher: Alright, let's start with a simple one. How do you say "good morning" in German?
Student: Umm, is it... Guten Morgen?
Teacher: Sehr gut! That's perfect. Now, how would you ask someone "How are you?"
Student: Wie geht es Ihnen?
Teacher: Wunderbar! Now try this one. How do you order a coffee in German?
Student: Einen Kaffee, bitte!
```

### Markdown-formatted input also works:
```
**Alex**: Hey, have you tried the new café?
**Sam**: Not yet, is it good?
```

## Output Formats

### Default behavior (3 files)
By default (no -e flag), the tool:
1. Requests lossless WAV (LINEAR16) from the API
2. Saves the .wav file
3. Transcodes locally via ffmpeg to:
   - .ogg (Opus VBR ~96kbps — excellent quality for speech)
   - .mp3 (LAME VBR V2 ~190kbps — high quality, widely compatible)

This produces 3 files with the same base name. The API's MP3 is only 32kbps,
so local transcoding from lossless WAV yields much better compressed output.

Requires ffmpeg installed. If missing, only .wav is produced (with warning).
Use --no-transcode to explicitly skip transcoding (WAV only).
Use -e to request a single specific format from the API directly.

### Human-readable output
Writes audio file(s) and prints status to stdout:
    Audio written to "output.wav"
      + transcoded: "output.ogg" (12345 bytes, ogg)
      + transcoded: "output.mp3" (23456 bytes, mp3)

Errors and warnings go to stderr.

### JSONL mode (--jsonl)
All output is machine-readable JSONL (one JSON object per line) on stdout.
Each line has an "event" field. Events emitted:

    {"event":"parsed","turns":6,"speakers":["Teacher","Student"],"input_bytes":335,"freeform_bytes":455,"prompt_bytes":42,"total_api_bytes":497,"api_limit_bytes":8000,"voice_map":{"Teacher":"Orus","Student":"Aoede"}}
    {"event":"generating","model":"gemini-2.5-flash-tts","language":"en-US","encoding":"LINEAR16","prompt":"..."}
    {"event":"completed","output_files":["out.wav","out.ogg","out.mp3"],"primary_file":"out.wav","audio_bytes":96288,"duration_seconds":1.23,"transcoded":[{"file":"out.ogg","format":"ogg","bytes":12345},{"file":"out.mp3","format":"mp3","bytes":23456}]}
    {"event":"dry_run","turns":6,"speakers":["Teacher","Student"],"input_bytes":335,"voice_map":{"Teacher":"Orus","Student":"Aoede"},"output_files":["out.wav","out.ogg","out.mp3"],"transcode":true}
    {"event":"skipped","output_file":"out.wav","reason":"already_exists"}
    {"event":"warning","message":"Estimated API payload is ~9000 bytes (text:8500 + prompt:500), Gemini TTS limit is ~8000 bytes. Request may fail. See: https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers"}
    {"event":"error","message":"2 speakers found but only 1 voices specified with --voices"}

Errors in JSONL mode are also printed as JSONL to stdout (not stderr) and
the process exits with code 1.

## Available Voices (30 Gemini TTS prebuilt)
Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe,
Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi,
Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird,
Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat

If --voices is omitted, speakers are auto-assigned in order from this list.

## Supported Languages (BCP-47 codes)
GA (21): en-US, en-GB, en-AU, en-IN, de-DE, fr-FR, fr-CA, es-ES, es-US,
  pt-BR, pt-PT, it-IT, nl-NL, ja-JP, ko-KR, cmn-CN, hi-IN, bn-IN, gu-IN,
  kn-IN, ml-IN, ta-IN, te-IN, ar-XA, id-ID, pl-PL, ru-RU, sv-SE, tr-TR,
  th-TH, vi-VN, and more.
The model handles mixed-language content within a single dialogue (e.g.
English teacher using German phrases inline).

## Audio Encodings and Quality
The API does NOT provide bitrate or quality-level controls. Quality is
determined entirely by the encoding format choice (-e flag):

| Encoding  | -e flag | Quality     | Details                                        |
|-----------|---------|-------------|------------------------------------------------|
| OGG_OPUS  | ogg     | Best lossy  | "Considerably higher than MP3 at same bitrate" |
| LINEAR16  | wav     | Lossless    | Uncompressed 16-bit PCM + WAV header           |
| MP3       | mp3     | Low (32kbps)| Fixed bitrate, no control. Default but NOT best|
| MULAW     | mulaw   | Telephony   | 8-bit G.711 mu-law                             |
| ALAW      | alaw    | Telephony   | 8-bit G.711 A-law. NOT supported by Chirp 3 HD |

DEFAULT BEHAVIOR: Requests lossless WAV from API, then transcodes locally via
ffmpeg to .ogg (Opus VBR ~96kbps) and .mp3 (LAME VBR V2 ~190kbps) — 3 files.
Use -e to request a single format from API directly (skips local transcoding).
Use --no-transcode to get just the WAV.

The complete AudioConfig parameter set (this is ALL the API exposes):
- audioEncoding — format (see table above)
- speakingRate — 0.25-4.0 (WARNING: ignored by Chirp 3 HD / Gemini voices)
- pitch — -20.0 to 20.0 semitones (WARNING: ignored by Chirp 3 HD / Gemini voices)
- volumeGainDb — -96.0 to 16.0 dB
- sampleRateHertz — default 24000; changing triggers resampling
- effectsProfileId — device profiles (8 available, see --audio-profile)

There are NO additional quality parameters (no bitrate, no codec options, no
quality level). Verified against REST API reference (see Sources below).

## Audio Device Profiles (--audio-profile)
Post-processing optimized for target playback device (repeatable flag):
- headphone-class-device — headphones, earbuds
- handset-class-device — smartphones
- small-bluetooth-speaker-class-device — portable BT speakers
- medium-bluetooth-speaker-class-device — smart home speakers (Google Home)
- large-home-entertainment-class-device — TVs, home entertainment
- large-automotive-class-device — car speakers
- telephony-class-application — IVR, phone systems
- wearable-class-device — smartwatches

## Models
- gemini-2.5-flash-tts (default) — fast, cost-efficient
- gemini-2.5-pro-tts — higher quality, better for podcasts/audiobooks

Note: Both use Chirp 3 HD voices which do NOT support SSML input,
speaking rate, pitch parameters, or A-Law encoding.

## Input Size Limits
- Standard Cloud TTS v1: 5000 bytes (text or SSML)
- Gemini TTS: 8000 bytes combined (text + prompt fields)
  Note: the freeform text includes "Speaker: " prefixes per turn, which add
  to the byte count beyond the raw dialogue text.
- Output audio: ~655 seconds max duration (truncated if exceeded)
- Source: https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers
  (see also: REST API ref at /rest/v1/text/synthesize for v1 limits)

## Chunked Generation (--chunk)
When input exceeds the ~8000 byte API limit, use --chunk to automatically split
the dialogue into multiple API calls and concatenate the audio output.
- Splits only at turn boundaries (never mid-sentence)
- Each chunk includes all speaker voice configs (for voice consistency)
- The prompt is sent with every chunk (for style consistency)
- MP3 output recommended for cleanest chunk concatenation
- WAV also supported (headers are properly merged)
- Dry run (-n) reports chunk count even without --chunk enabled
- JSONL events include chunk/total_chunks fields per API call

## Style Prompts (--prompt)
Natural language instructions controlling delivery style. Examples:
- "Casual conversation between friends"
- "News anchor reading headlines"
- "Patient teacher with enthusiastic student"
- "Dramatic audiobook narration"
Markup tags can be used in the dialogue text itself: [sigh], [whispering],
[short pause], [sarcasm], [excited], [extremely fast]

## Usage Examples

### Basic generation (produces .wav + .ogg + .mp3):
    uv run multi-speaker_markup_from_dialog_transcript.py -i dialogue.txt

### Custom voices + style prompt (3 files):
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i dialogue.txt --voices Charon,Kore \
      -p "Fun lighthearted banter between friends"

### Single format from API directly (no local transcoding):
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i dialogue.txt -e ogg -o output.ogg

### WAV only (no transcoding):
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i dialogue.txt --no-transcode

### German dialogue:
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i german_dialogue.txt -l de-DE --voices Charon,Aoede

### Programmatic JSONL pipeline:
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i dialogue.txt -o out.mp3 --jsonl --voices Orus,Aoede \
      | while IFS= read -r line; do
          event=$(echo "$line" | jq -r .event)
          case "$event" in
            completed) echo "Done: $(echo "$line" | jq -r .output_file)" ;;
            error)     echo "FAIL: $(echo "$line" | jq -r .message)" >&2 ;;
          esac
        done

### Long dialogue with chunking:
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i long_dialogue.txt --chunk --voices Charon,Kore \
      -p "Lively debate between colleagues"

### Dry run (no API call):
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i dialogue.txt -n --voices Charon,Kore

### Pipe from stdin:
    echo -e "Alice: Hello\nBob: Hi there" | \
      uv run multi-speaker_markup_from_dialog_transcript.py -i - -o greeting.mp3

## Exit Codes
- 0: success (or dry run)
- 1: error (file exists, invalid args, API error)
- 2: argparse error (missing required args)

## Official Documentation References
Use these to verify limits, parameters, and capabilities if they change:

- Multi-speaker dialogue guide (byte limits, voices, prompting):
  https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers
- REST API v1 reference (AudioConfig, AudioEncoding, all parameters):
  https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
- REST API v1beta1 reference:
  https://cloud.google.com/text-to-speech/docs/reference/rest/v1beta1/text/synthesize
- Audio device profiles:
  https://cloud.google.com/text-to-speech/docs/audio-profiles
- Supported voices listing:
  https://cloud.google.com/text-to-speech/docs/voices
- Gemini / Chirp 3 TTS voices:
  https://cloud.google.com/text-to-speech/docs/tts-voices
- SSML reference (not supported by Chirp 3 HD voices):
  https://cloud.google.com/text-to-speech/docs/ssml
- Quotas and limits:
  https://cloud.google.com/text-to-speech/quotas
- Python client library:
  https://cloud.google.com/text-to-speech/docs/libraries

Key facts to verify:
- MP3 is "MP3 audio at 32kbps" (AudioEncoding enum in REST API ref)
- OGG_OPUS: "considerably higher than MP3 while using approximately the same
  bitrate" (AudioEncoding enum in REST API ref)
- Gemini TTS combined limit: 8000 bytes text+prompt (multi-speaker guide)
- Chirp 3 HD voices: no SSML, no rate/pitch, no A-Law (voices page)
- AudioConfig has exactly 6 fields: audioEncoding, speakingRate, pitch,
  volumeGainDb, sampleRateHertz, effectsProfileId (REST API ref)
""".strip()


def emit(event_data):
    """Emit a JSONL event to stdout if --jsonl mode, else no-op."""
    if args.jsonl:
        print(json.dumps(event_data, ensure_ascii=False), flush=True)


def emit_error(message):
    """Emit error in appropriate format and exit."""
    if args.jsonl:
        print(json.dumps({"event": "error", "message": message}, ensure_ascii=False), flush=True)
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def emit_warning(message):
    """Emit warning in appropriate format."""
    if args.jsonl:
        print(json.dumps({"event": "warning", "message": message}, ensure_ascii=False), flush=True)
    else:
        print(f"WARNING: {message}", file=sys.stderr)


def log_verbose(message, *format_args):
    if args.verbose and not args.jsonl:
        print(message.format(*format_args), file=sys.stderr)


def parse_input(input_file):
    """Parse 'Speaker: text' dialogue format into (speaker_name, text) tuples."""
    speakers_seen = []  # ordered list of unique speaker names
    turns = []
    last_speaker = None

    for line in input_file:
        line = line.strip()
        if line:
            if ':' in line:
                speaker, text = line.split(':', 1)
                speaker = re.sub(r'[*_~`]', '', speaker).strip()
                text = text.strip()

                if speaker not in speakers_seen:
                    speakers_seen.append(speaker)
                    log_verbose("DEB:SPEAKER[{}] (#{})", speaker, len(speakers_seen))
            else:
                text = line
                speaker = last_speaker

            turns.append((speaker, text))
            last_speaker = speaker

            if args.verbose > 1 and not args.jsonl:
                print(f'APPEND:[{speaker}]:{text}', file=sys.stderr)

    return turns, speakers_seen


def build_voice_map(speakers_seen):
    """Map speaker names to Gemini voice IDs.

    Uses --voices if provided (comma-separated voice IDs matching speaker order),
    otherwise auto-assigns from GEMINI_VOICES pool.
    """
    if args.voices:
        voice_list = args.voices.split(',')
        if len(voice_list) < len(speakers_seen):
            emit_error(f"{len(speakers_seen)} speakers found but only {len(voice_list)} voices specified with --voices")
        voice_map = {}
        for i, speaker in enumerate(speakers_seen):
            voice_map[speaker] = voice_list[i].strip()
        return voice_map

    # Auto-assign: cycle through GEMINI_VOICES
    voice_map = {}
    for i, speaker in enumerate(speakers_seen):
        voice_map[speaker] = GEMINI_VOICES[i % len(GEMINI_VOICES)]
    return voice_map


def choose_audio_encoding():
    if args.encoding:
        return get_audio_encoding(args.encoding)
    # Default: WAV (lossless) — local transcoding to ogg+mp3 handles compression
    return texttospeech.AudioEncoding.LINEAR16


def list_voices():
    """List available voices for the specified language."""
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices(language_code=args.language)

    print("Gemini TTS prebuilt voices (for --voices flag):")
    for v in GEMINI_VOICES:
        print(f"  {v}")

    print(f"\nAll Cloud TTS voices for {args.language}:")
    for voice in response.voices:
        lang_codes = ", ".join(voice.language_codes)
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        print(f"  {voice.name}  ({lang_codes}, {gender})")


# Gemini TTS combined limit: 8000 bytes (text + prompt)
# Use 7500 as effective limit for safety margin
# Source: https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers
CHUNK_MAX_BYTES = 7500


def freeform_line_bytes(speaker, text):
    """Byte length of a single freeform line 'Speaker: text'."""
    return len(f"{speaker}: {text}".encode('utf-8'))


def chunk_turns(turns, prompt_bytes):
    """Split turns into chunks that each fit within the Gemini TTS byte limit.

    Each chunk is a list of (speaker, text) tuples. Splits only at turn
    boundaries — never mid-turn. Returns list of chunk lists.
    """
    max_text_bytes = CHUNK_MAX_BYTES - prompt_bytes
    if max_text_bytes <= 0:
        emit_error(f"Prompt alone is {prompt_bytes} bytes, exceeds chunk limit of {CHUNK_MAX_BYTES}")

    chunks = []
    current_chunk = []
    current_bytes = 0

    for speaker, text in turns:
        line_bytes = freeform_line_bytes(speaker, text)
        # +1 for newline separator between lines (except first in chunk)
        added_bytes = line_bytes + (1 if current_chunk else 0)

        if current_bytes + added_bytes > max_text_bytes and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_bytes = 0
            added_bytes = line_bytes  # first in new chunk, no newline prefix

        if line_bytes > max_text_bytes:
            emit_warning(f"Single turn by '{speaker}' is {line_bytes} bytes, exceeds chunk limit of {max_text_bytes}. "
                         f"It will be sent as its own chunk and may fail.")

        current_chunk.append((speaker, text))
        current_bytes += added_bytes

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def concatenate_audio(audio_parts, audio_encoding):
    """Concatenate audio chunks, handling format-specific requirements.

    MP3: frames are self-contained, direct concatenation works.
    WAV/LINEAR16: strip 44-byte WAV headers from chunks 2+, fix final header.
    Others: direct concatenation with warning about potential artifacts.
    """
    if len(audio_parts) == 1:
        return audio_parts[0]

    if audio_encoding == texttospeech.AudioEncoding.MP3:
        return b"".join(audio_parts)

    if audio_encoding == texttospeech.AudioEncoding.LINEAR16:
        # WAV format: 44-byte header + PCM data
        # Keep header from first chunk, strip from subsequent
        WAV_HEADER_SIZE = 44
        pcm_data = audio_parts[0]  # first chunk has complete WAV with header
        for part in audio_parts[1:]:
            if len(part) > WAV_HEADER_SIZE:
                pcm_data += part[WAV_HEADER_SIZE:]
            else:
                pcm_data += part
        # Fix WAV header: update ChunkSize (bytes 4-7) and Subchunk2Size (bytes 40-43)
        data_size = len(pcm_data) - WAV_HEADER_SIZE
        pcm_data = bytearray(pcm_data)
        struct.pack_into('<I', pcm_data, 4, data_size + 36)  # ChunkSize = DataSize + 36
        struct.pack_into('<I', pcm_data, 40, data_size)      # Subchunk2Size = DataSize
        return bytes(pcm_data)

    # OGG_OPUS, MULAW, ALAW: best-effort concatenation
    emit_warning(f"Chunked concatenation for this encoding may produce artifacts at chunk boundaries. "
                 f"Consider using MP3 or WAV with --chunk for cleanest results.")
    return b"".join(audio_parts)


def check_ffmpeg():
    """Check if ffmpeg is available, return path or None."""
    return shutil.which("ffmpeg")


def transcode_wav(wav_path, output_base):
    """Transcode WAV to high-quality OGG (Opus VBR) and MP3 (LAME VBR).

    Returns list of (format, path, size) for files created.
    """
    ffmpeg = check_ffmpeg()
    if not ffmpeg:
        emit_warning("ffmpeg not found — skipping local transcoding to .ogg and .mp3. "
                     "Install ffmpeg for automatic high-quality compression.")
        return []

    results = []
    transcode_specs = [
        # (extension, codec, quality args, description)
        (".ogg", "libopus", ["-b:a", "96k", "-vbr", "on"], "Opus VBR ~96kbps"),
        (".mp3", "libmp3lame", ["-q:a", "2"], "LAME VBR V2 ~190kbps"),
    ]

    for ext, codec, quality_args, description in transcode_specs:
        out_path = output_base + ext
        if not args.force and os.path.exists(out_path):
            log_verbose("DEB:Transcode: skipping {} (already exists)", out_path)
            emit_warning(f"Transcode target {out_path} already exists, skipping (use --force)")
            continue

        cmd = [ffmpeg, "-y", "-i", wav_path, "-c:a", codec] + quality_args + [out_path]
        log_verbose("DEB:Transcode: {} → {} ({})", wav_path, out_path, description)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                emit_warning(f"ffmpeg transcode to {ext} failed: {result.stderr.strip()[-200:]}")
                continue
            size = os.path.getsize(out_path)
            results.append((ext, out_path, size))
            log_verbose("DEB:Transcode: {} → {} bytes", out_path, size)
        except subprocess.TimeoutExpired:
            emit_warning(f"ffmpeg transcode to {ext} timed out after 120s")
        except OSError as e:
            emit_warning(f"ffmpeg transcode to {ext} failed: {e}")

    return results


def generate_audio(turns, speakers_seen, voice_map=None):
    """Generate multi-speaker audio using Gemini TTS."""
    client = texttospeech.TextToSpeechClient()

    if voice_map is None:
        voice_map = build_voice_map(speakers_seen)
    for speaker, voice_id in voice_map.items():
        log_verbose("DEB:VOICE_MAP[{}] = {}", speaker, voice_id)

    # Build freeform text: "Speaker: line\nSpeaker2: line\n..."
    freeform_lines = []
    for speaker, text in turns:
        freeform_lines.append(f"{speaker}: {text}")
    freeform_text = "\n".join(freeform_lines)

    input_kwargs = {"text": freeform_text}
    if args.prompt:
        input_kwargs["prompt"] = args.prompt
    synthesis_input = texttospeech.SynthesisInput(**input_kwargs)

    # Build multi-speaker voice config
    speaker_voice_configs = []
    for speaker, voice_id in voice_map.items():
        speaker_voice_configs.append(
            texttospeech.MultispeakerPrebuiltVoice(
                speaker_alias=speaker,
                speaker_id=voice_id,
            )
        )

    voice = texttospeech.VoiceSelectionParams(
        language_code=args.language,
        model_name=args.model,
        multi_speaker_voice_config=texttospeech.MultiSpeakerVoiceConfig(
            speaker_voice_configs=speaker_voice_configs,
        ),
    )

    audio_encoding = choose_audio_encoding()
    encoding_str = next(key for key, value in encoding_map.items() if value == audio_encoding)
    log_verbose("DEB:Chosen audio encoding: {}", encoding_str)

    audio_config_kwargs = {"audio_encoding": audio_encoding}
    if args.rate is not None:
        audio_config_kwargs["speaking_rate"] = args.rate
    if args.pitch is not None:
        audio_config_kwargs["pitch"] = args.pitch
    if args.volume is not None:
        audio_config_kwargs["volume_gain_db"] = args.volume
    if args.sample_rate is not None:
        audio_config_kwargs["sample_rate_hertz"] = args.sample_rate
    else:
        audio_config_kwargs["sample_rate_hertz"] = 24000
    if args.audio_profile:
        audio_config_kwargs["effects_profile_id"] = args.audio_profile

    audio_config = texttospeech.AudioConfig(**audio_config_kwargs)
    log_verbose("DEB:AudioConfig: {}", audio_config_kwargs)

    emit({"event": "generating", "model": args.model, "language": args.language,
          "encoding": encoding_str, "prompt": args.prompt})

    t0 = time.monotonic()
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    elapsed = time.monotonic() - t0
    log_verbose("DEB:API call took {:.2f}s", elapsed)

    return response.audio_content, elapsed


def generate_audio_chunked(turns, speakers_seen):
    """Generate audio in chunks when input exceeds API byte limit.

    Splits turns into chunks, makes separate API calls, and concatenates
    the resulting audio. The prompt and voice config are shared across all chunks.
    """
    prompt_bytes = len(args.prompt.encode('utf-8')) if args.prompt else 0
    chunks = chunk_turns(turns, prompt_bytes)
    voice_map = build_voice_map(speakers_seen)
    audio_encoding = choose_audio_encoding()
    encoding_str = next(key for key, value in encoding_map.items() if value == audio_encoding)

    log_verbose("DEB:Chunking: {} turns split into {} chunks (limit: {} bytes, prompt: {} bytes)",
                len(turns), len(chunks), CHUNK_MAX_BYTES, prompt_bytes)

    audio_parts = []
    total_elapsed = 0

    for i, chunk_turns_list in enumerate(chunks):
        chunk_text_bytes = sum(freeform_line_bytes(s, t) for s, t in chunk_turns_list) + max(len(chunk_turns_list) - 1, 0)
        log_verbose("DEB:Chunk {}/{}: {} turns, ~{} text bytes",
                    i + 1, len(chunks), len(chunk_turns_list), chunk_text_bytes)

        emit({"event": "generating", "model": args.model, "language": args.language,
              "encoding": encoding_str, "prompt": args.prompt,
              "chunk": i + 1, "total_chunks": len(chunks),
              "chunk_turns": len(chunk_turns_list), "chunk_bytes": chunk_text_bytes})

        audio_content, elapsed = generate_audio(chunk_turns_list, speakers_seen, voice_map=voice_map)
        audio_parts.append(audio_content)
        total_elapsed += elapsed

        log_verbose("DEB:Chunk {}/{}: {} audio bytes, {:.2f}s",
                    i + 1, len(chunks), len(audio_content), elapsed)

    combined = concatenate_audio(audio_parts, audio_encoding)
    log_verbose("DEB:Combined {} chunks: {} total audio bytes, {:.2f}s total",
                len(chunks), len(combined), total_elapsed)

    return combined, total_elapsed


encoding_map = {
    "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
    "WAV": texttospeech.AudioEncoding.LINEAR16,
    "MP3": texttospeech.AudioEncoding.MP3,
    "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
    "OGG": texttospeech.AudioEncoding.OGG_OPUS,
    "MULAW": texttospeech.AudioEncoding.MULAW,
    "ALAW": texttospeech.AudioEncoding.ALAW,
}


def get_audio_encoding(encoding_str):
    encoding = encoding_map.get(encoding_str.upper())
    if encoding is None:
        valid = ", ".join(sorted(set(encoding_map.keys())))
        emit_error(f"Unsupported audio encoding '{encoding_str}'. Valid: {valid}")
    return encoding


def main():
    global args
    parser = argparse.ArgumentParser(
        description="Generate multi-speaker dialogue audio using Gemini TTS. "
                    "By default requests lossless WAV from the API and locally transcodes "
                    "to high-quality .ogg (Opus VBR ~96kbps) and .mp3 (LAME VBR V2 ~190kbps), "
                    "producing 3 files. Use -e to request a single format from the API directly. "
                    "Use --help-llm for full reference with official doc links.",
        epilog="Example: %(prog)s -i dialogue.txt --voices Charon,Kore -p 'Friendly conversation'",
    )

    # I/O
    parser.add_argument("-i", "--input", default=None,
                        help="Input file with 'Speaker: text' format (use '-' for stdin)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path or base name. Default mode (no -e): this sets the base "
                             "name for .wav/.ogg/.mp3 (extension is stripped). With -e: exact output path.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite output files if they already exist")
    parser.add_argument("-e", "--encoding", default=None,
                        help="Request a SINGLE format from the API directly (disables local transcoding). "
                             "Options: mp3 (API 32kbps fixed), ogg (OGG_OPUS), wav (LINEAR16 lossless), "
                             "mulaw, alaw. Without -e, default behavior requests lossless WAV and locally "
                             "transcodes to high-quality .ogg + .mp3 via ffmpeg (3 files).")
    parser.add_argument("--no-transcode", action="store_true",
                        help="Get lossless WAV from API but skip local transcoding to .ogg/.mp3 "
                             "(produces only the .wav file). Useful if ffmpeg is unavailable.")

    # TTS config
    parser.add_argument("-l", "--language", default="en-US",
                        help="BCP-47 language code: en-US, de-DE, fr-FR, ja-JP, ko-KR, etc. (default: en-US)")
    parser.add_argument("-m", "--model", default="gemini-2.5-flash-tts",
                        help="Gemini TTS model: gemini-2.5-flash-tts (default), gemini-2.5-pro-tts")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Natural language style prompt, e.g. 'Casual conversation between friends'")
    parser.add_argument("--voices", default=None,
                        help="Comma-separated voice IDs per speaker in order of appearance. "
                             "E.g. 'Charon,Kore'. If omitted, auto-assigned from 30 prebuilt voices.")

    # Audio tuning
    parser.add_argument("--rate", type=float, default=None,
                        help="Speaking rate 0.25-4.0 (default: 1.0). "
                             "WARNING: may be ignored by Chirp 3 HD / Gemini TTS voices.")
    parser.add_argument("--pitch", type=float, default=None,
                        help="Pitch in semitones -20.0 to 20.0 (default: 0.0). "
                             "WARNING: may be ignored by Chirp 3 HD / Gemini TTS voices.")
    parser.add_argument("--volume", type=float, default=None,
                        help="Volume gain in dB -96.0 to 16.0 (default: 0.0)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="Sample rate in Hz (default: 24000). "
                             "Changing from native rate triggers resampling (may reduce quality).")
    parser.add_argument("--audio-profile", action="append", default=None,
                        help="Audio device profile (repeatable): headphone-class-device, "
                             "handset-class-device, small-bluetooth-speaker-class-device, "
                             "medium-bluetooth-speaker-class-device, "
                             "large-home-entertainment-class-device, "
                             "large-automotive-class-device, "
                             "telephony-class-application, wearable-class-device")

    # Modes
    parser.add_argument("--chunk", action="store_true",
                        help="Enable chunked generation for long dialogues that exceed the ~8000 byte "
                             "API limit. Splits at turn boundaries, makes multiple API calls, and "
                             "concatenates the audio. Without this flag, oversized input produces a warning.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Parse input and show plan without calling the API")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Verbose logging to stderr (-vv for trace)")
    parser.add_argument("--jsonl", action="store_true",
                        help="Machine-readable JSONL output to stdout (for pipelines/agents)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available Gemini and Cloud TTS voices, then exit")
    parser.add_argument("--help-llm", action="store_true",
                        help="Print extended reference documentation for LLM agents, then exit")

    args = parser.parse_args()

    if args.help_llm:
        print(LLM_REFERENCE)
        sys.exit(0)

    if args.list_voices:
        list_voices()
        sys.exit(0)

    if args.input is None:
        parser.error("the following arguments are required: -i/--input")

    # Determine if we're in single-format mode (-e) or multi-output mode (default)
    single_format = args.encoding is not None
    do_transcode = not single_format and not args.no_transcode

    if args.input == '-':
        input_file = sys.stdin
        if args.output:
            output_base = os.path.splitext(args.output)[0]
        else:
            output_base = "output"
    else:
        input_file = open(args.input, 'r')
        if args.output:
            output_base = os.path.splitext(args.output)[0]
        else:
            output_base = args.input

    if single_format:
        # Single format from API: use -e extension
        ext = args.encoding.lower()
        if ext in ("linear16", "ogg_opus"):
            ext = {"linear16": "wav", "ogg_opus": "ogg"}[ext]
        output_file = output_base + "." + ext
    else:
        # Default: WAV from API (+ local transcode to ogg/mp3)
        output_file = output_base + ".wav"

    # Check if primary output exists
    if not args.force and os.path.exists(output_file):
        if args.jsonl:
            emit({"event": "skipped", "output_file": output_file, "reason": "already_exists"})
        else:
            print(f"SKIPPING:ALREADY_EXISTS:{output_file} (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    log_verbose("Processing input from {}", args.input)
    try:
        turns, speakers_seen = parse_input(input_file)
    finally:
        if input_file is not sys.stdin:
            input_file.close()

    if not turns:
        emit_error("No dialogue turns parsed from input")

    input_bytes = sum(len(text.encode('utf-8')) for _, text in turns)
    voice_map = build_voice_map(speakers_seen)

    log_verbose("DEB:Parsed {} turns from {} speakers, ~{} bytes of text",
                len(turns), len(speakers_seen), input_bytes)
    log_verbose("DEB:Model: {}, Language: {}", args.model, args.language)
    if args.prompt:
        log_verbose("DEB:Prompt: {}", args.prompt)

    # Gemini TTS combined limit: 8000 bytes (text + prompt)
    # Source: https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers
    # Note: freeform_text includes "Speaker: " prefixes per turn, so actual API payload > input_bytes
    freeform_bytes = sum(len(f"{spk}: {txt}".encode('utf-8')) for spk, txt in turns) + len(turns) - 1  # newlines
    prompt_bytes = len(args.prompt.encode('utf-8')) if args.prompt else 0
    total_api_bytes = freeform_bytes + prompt_bytes
    needs_chunking = total_api_bytes > CHUNK_MAX_BYTES
    if needs_chunking and not args.chunk:
        emit_warning(f"Estimated API payload is ~{total_api_bytes} bytes (text:{freeform_bytes} + prompt:{prompt_bytes}), "
                     f"Gemini TTS limit is ~8000 bytes. Request may fail. "
                     f"Use --chunk to automatically split into multiple API calls. "
                     f"See: https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers")

    # Compute chunk plan for reporting (even if --chunk not set)
    chunk_count = len(chunk_turns(turns, prompt_bytes)) if needs_chunking else 1

    parsed_info = {
        "event": "parsed",
        "turns": len(turns),
        "speakers": speakers_seen,
        "input_bytes": input_bytes,
        "freeform_bytes": freeform_bytes,
        "prompt_bytes": prompt_bytes,
        "total_api_bytes": total_api_bytes,
        "api_limit_bytes": 8000,
        "chunks_needed": chunk_count,
        "chunking_enabled": args.chunk,
        "voice_map": voice_map,
    }
    emit(parsed_info)

    if args.dry_run:
        output_files = [output_file]
        if do_transcode:
            output_files += [output_base + ".ogg", output_base + ".mp3"]

        dry_info = {
            **parsed_info,
            "event": "dry_run",
            "model": args.model,
            "language": args.language,
            "prompt": args.prompt,
            "output_files": output_files,
            "transcode": do_transcode,
        }
        if args.jsonl:
            emit(dry_info)
        else:
            print(f"Dry run: {len(turns)} turns, {len(speakers_seen)} speakers, ~{total_api_bytes} API bytes")
            print(f"  Model: {args.model}  Language: {args.language}")
            for speaker, voice_id in voice_map.items():
                print(f"  {speaker} -> {voice_id}")
            if args.prompt:
                print(f"  Prompt: {args.prompt}")
            if chunk_count > 1:
                chunk_label = f"  Chunks: {chunk_count} API calls" + (" (--chunk enabled)" if args.chunk else " (needs --chunk flag)")
                print(chunk_label)
            if do_transcode:
                print(f"  Output (3 files): {output_base}.wav / .ogg / .mp3")
                if not check_ffmpeg():
                    print(f"  WARNING: ffmpeg not found — .ogg and .mp3 will be skipped")
            else:
                print(f"  Output: {output_file}")
        return

    use_chunking = args.chunk and needs_chunking
    if args.chunk and not needs_chunking:
        log_verbose("DEB:--chunk specified but input fits in single request ({} bytes <= {} limit), skipping chunking",
                    total_api_bytes, CHUNK_MAX_BYTES)

    if use_chunking:
        log_verbose("Generating audio in chunks (input ~{} bytes exceeds {} limit)", total_api_bytes, CHUNK_MAX_BYTES)
        audio_content, elapsed = generate_audio_chunked(turns, speakers_seen)
    else:
        log_verbose("Generating audio")
        audio_content, elapsed = generate_audio(turns, speakers_seen)

    log_verbose("Writing audio to {}", output_file)
    with open(output_file, "wb") as out:
        out.write(audio_content)

    output_files = [{"file": output_file, "format": "wav" if not single_format else args.encoding,
                     "bytes": len(audio_content), "source": "api"}]

    # Local transcoding (default mode: WAV → OGG + MP3)
    if do_transcode:
        log_verbose("Transcoding WAV to OGG and MP3 via ffmpeg")
        transcode_results = transcode_wav(output_file, output_base)
        for ext, path, size in transcode_results:
            output_files.append({"file": path, "format": ext.lstrip("."), "bytes": size, "source": "ffmpeg"})

    completed_info = {
        "event": "completed",
        "output_files": [f["file"] for f in output_files],
        "primary_file": output_file,
        "audio_bytes": len(audio_content),
        "duration_seconds": round(elapsed, 2),
    }
    if use_chunking:
        completed_info["chunks"] = chunk_count
    if do_transcode:
        completed_info["transcoded"] = [{"file": f["file"], "format": f["format"], "bytes": f["bytes"]}
                                        for f in output_files if f["source"] == "ffmpeg"]

    if args.jsonl:
        emit(completed_info)
    else:
        chunk_note = f" ({chunk_count} chunks concatenated)" if use_chunking else ""
        print(f'Audio written to "{output_file}"{chunk_note}')
        for f in output_files:
            if f["source"] == "ffmpeg":
                print(f'  + transcoded: "{f["file"]}" ({f["bytes"]} bytes, {f["format"]})')


if __name__ == "__main__":
    main()
