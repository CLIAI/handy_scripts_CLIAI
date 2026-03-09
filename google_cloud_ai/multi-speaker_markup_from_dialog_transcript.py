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

### Default (human-readable)
Writes audio file and prints status to stdout:
    Audio content written to file "output.mp3"

Errors and warnings go to stderr.

### JSONL mode (--jsonl)
All output is machine-readable JSONL (one JSON object per line) on stdout.
Each line has an "event" field. Events emitted:

    {"event":"parsed","turns":6,"speakers":["Teacher","Student"],"input_bytes":335,"voice_map":{"Teacher":"Orus","Student":"Aoede"}}
    {"event":"generating","model":"gemini-2.5-flash-tts","language":"en-US","encoding":"MP3","prompt":"..."}
    {"event":"completed","output_file":"out.mp3","audio_bytes":96288,"duration_seconds":1.23}
    {"event":"dry_run","turns":6,"speakers":["Teacher","Student"],"input_bytes":335,"voice_map":{"Teacher":"Orus","Student":"Aoede"},"output_file":"out.mp3"}
    {"event":"skipped","output_file":"out.mp3","reason":"already_exists"}
    {"event":"warning","message":"Input text is ~9000 bytes, Gemini TTS limit is ~8000 bytes"}
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

## Audio Encodings
MP3 (default), WAV/LINEAR16, OGG/OGG_OPUS, MULAW, ALAW

## Models
- gemini-2.5-flash-tts (default) — fast, cost-efficient
- gemini-2.5-pro-tts — higher quality, better for podcasts/audiobooks

## Style Prompts (--prompt)
Natural language instructions controlling delivery style. Examples:
- "Casual conversation between friends"
- "News anchor reading headlines"
- "Patient teacher with enthusiastic student"
- "Dramatic audiobook narration"
Markup tags can be used in the dialogue text itself: [sigh], [whispering],
[short pause], [sarcasm], [excited], [extremely fast]

## Usage Examples

### Basic generation:
    uv run multi-speaker_markup_from_dialog_transcript.py -i dialogue.txt

### Custom voices + style prompt:
    uv run multi-speaker_markup_from_dialog_transcript.py \
      -i dialogue.txt --voices Charon,Kore \
      -p "Fun lighthearted banter between friends"

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
    return get_audio_encoding(args.encoding) if args.encoding else texttospeech.AudioEncoding.MP3


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


def generate_audio(turns, speakers_seen):
    """Generate multi-speaker audio using Gemini TTS."""
    client = texttospeech.TextToSpeechClient()

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
        description="Generate multi-speaker dialogue audio using Gemini TTS.",
        epilog="Example: %(prog)s -i dialogue.txt --voices Charon,Kore -p 'Friendly conversation'",
    )

    # I/O
    parser.add_argument("-i", "--input", default=None,
                        help="Input file with 'Speaker: text' format (use '-' for stdin)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output audio file path (default: input filename + .mp3)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite output file if it already exists")
    parser.add_argument("-e", "--encoding", default=None,
                        help="Audio encoding: mp3 (default), wav, ogg, mulaw, alaw")

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
                        help="Speaking rate 0.25-4.0 (default: 1.0)")
    parser.add_argument("--pitch", type=float, default=None,
                        help="Pitch in semitones -20.0 to 20.0 (default: 0.0)")
    parser.add_argument("--volume", type=float, default=None,
                        help="Volume gain in dB -96.0 to 16.0 (default: 0.0)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="Sample rate in Hz (default: 24000)")
    parser.add_argument("--audio-profile", action="append", default=None,
                        help="Audio device profile (repeatable): headphone-class-device, "
                             "handset-class-device, telephony-class-application, etc.")

    # Modes
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

    if args.input == '-':
        input_file = sys.stdin
        output_file = args.output if args.output else "output.mp3"
    else:
        input_file = open(args.input, 'r')
        if args.output:
            output_file = args.output
            if not args.encoding:
                args.encoding = os.path.splitext(output_file)[1][1:]
        else:
            ext = args.encoding if args.encoding else "mp3"
            output_file = args.input + "." + ext

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

    if input_bytes > 8000:
        emit_warning(f"Input text is ~{input_bytes} bytes, Gemini TTS limit is ~8000 bytes (prompt+text). Request may fail.")

    parsed_info = {
        "event": "parsed",
        "turns": len(turns),
        "speakers": speakers_seen,
        "input_bytes": input_bytes,
        "voice_map": voice_map,
    }
    emit(parsed_info)

    if args.dry_run:
        dry_info = {
            **parsed_info,
            "event": "dry_run",
            "model": args.model,
            "language": args.language,
            "prompt": args.prompt,
            "output_file": output_file,
        }
        if args.jsonl:
            emit(dry_info)
        else:
            print(f"Dry run: {len(turns)} turns, {len(speakers_seen)} speakers, ~{input_bytes} bytes")
            print(f"  Model: {args.model}  Language: {args.language}")
            for speaker, voice_id in voice_map.items():
                print(f"  {speaker} -> {voice_id}")
            if args.prompt:
                print(f"  Prompt: {args.prompt}")
            print(f"  Output would be: {output_file}")
        return

    log_verbose("Generating audio")
    audio_content, elapsed = generate_audio(turns, speakers_seen)

    log_verbose("Writing audio to {}", output_file)
    with open(output_file, "wb") as out:
        out.write(audio_content)

    if args.jsonl:
        emit({
            "event": "completed",
            "output_file": output_file,
            "audio_bytes": len(audio_content),
            "duration_seconds": round(elapsed, 2),
        })
    else:
        print(f'Audio content written to file "{output_file}"')


if __name__ == "__main__":
    main()
