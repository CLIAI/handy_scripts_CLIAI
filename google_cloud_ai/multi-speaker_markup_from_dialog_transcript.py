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
import re
import sys
import os
from google.cloud import texttospeech

# 30 Gemini TTS prebuilt voices - used for auto-assigning to speakers
GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]


def log_verbose(message, *format_args):
    if args.verbose:
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
                    if args.verbose:
                        print(f'DEB:SPEAKER[{speaker}] (#{len(speakers_seen)})', file=sys.stderr)
            else:
                text = line
                speaker = last_speaker

            turns.append((speaker, text))
            last_speaker = speaker

            if args.verbose > 1:
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
            print(f"ERROR: {len(speakers_seen)} speakers found but only {len(voice_list)} voices specified with --voices",
                  file=sys.stderr)
            sys.exit(1)
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

    print(f"Gemini TTS prebuilt voices (for --voices flag):")
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

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content


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
        print(f"ERROR: Unsupported audio encoding '{encoding_str}'. Valid: {valid}", file=sys.stderr)
        sys.exit(1)
    return encoding


def main():
    global args
    parser = argparse.ArgumentParser(
        description="Generate multi-speaker dialogue audio using Gemini TTS.",
        epilog="Example: %(prog)s -i dialogue.txt --voices Charon,Kore --prompt 'Friendly conversation'",
    )
    parser.add_argument("-i", "--input", default=None, help="Input file with 'Speaker: text' format (use '-' for stdin)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Enable verbose logging (-vv for more)")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Parse input and show plan without generating audio")
    parser.add_argument("-e", "--encoding", default=None, help="Audio encoding (wav, mp3, ogg, mulaw, alaw)")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: input filename + extension)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output file if it already exists")
    parser.add_argument("-l", "--language", default="en-US",
                        help="BCP-47 language code, e.g. en-US, de-DE, fr-FR, ja-JP, ko-KR (default: en-US)")
    parser.add_argument("-m", "--model", default="gemini-2.5-flash-tts",
                        help="Gemini TTS model (default: gemini-2.5-flash-tts). Options: "
                             "gemini-2.5-flash-tts, gemini-2.5-pro-tts")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Natural language style prompt for delivery, e.g. "
                             "'Casual conversation between friends' or 'News anchor reading headlines'")
    parser.add_argument("--voices", default=None,
                        help="Comma-separated Gemini voice IDs for speakers in order of appearance. "
                             "E.g. 'Charon,Kore' assigns first speaker to Charon, second to Kore. "
                             "Available: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, etc. "
                             "Use --list-voices to see all. If omitted, auto-assigned.")
    parser.add_argument("--rate", type=float, default=None, help="Speaking rate 0.25-4.0 (default: 1.0)")
    parser.add_argument("--pitch", type=float, default=None, help="Pitch in semitones -20.0 to 20.0 (default: 0.0)")
    parser.add_argument("--volume", type=float, default=None, help="Volume gain in dB -96.0 to 16.0 (default: 0.0)")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate in Hz (default: 24000)")
    parser.add_argument("--audio-profile", action="append", default=None,
                        help="Audio device profile (repeatable): headphone-class-device, "
                             "handset-class-device, telephony-class-application, etc.")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    args = parser.parse_args()

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
        print(f"SKIPPING:ALREADY_EXISTS:{output_file} (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    log_verbose("Processing input from {}", args.input)
    try:
        turns, speakers_seen = parse_input(input_file)
    finally:
        if input_file is not sys.stdin:
            input_file.close()

    input_bytes = sum(len(text.encode('utf-8')) for _, text in turns)
    voice_map = build_voice_map(speakers_seen)

    log_verbose("DEB:Parsed {} turns from {} speakers, ~{} bytes of text", len(turns), len(speakers_seen), input_bytes)
    log_verbose("DEB:Model: {}, Language: {}", args.model, args.language)
    if args.prompt:
        log_verbose("DEB:Prompt: {}", args.prompt)

    if input_bytes > 8000:
        print(f"WARNING: Input text is ~{input_bytes} bytes, Gemini TTS limit is ~8000 bytes (prompt+text). Request may fail.", file=sys.stderr)

    if not args.dry_run:
        log_verbose("Generating audio")
        audio_content = generate_audio(turns, speakers_seen)

        log_verbose("Writing audio to {}", output_file)
        with open(output_file, "wb") as out:
            out.write(audio_content)

        print(f'Audio content written to file "{output_file}"')
    else:
        print(f"Dry run: {len(turns)} turns, {len(speakers_seen)} speakers, ~{input_bytes} bytes")
        print(f"  Model: {args.model}  Language: {args.language}")
        for speaker, voice_id in voice_map.items():
            print(f"  {speaker} -> {voice_id}")
        if args.prompt:
            print(f"  Prompt: {args.prompt}")
        print(f"  Output would be: {output_file}")


if __name__ == "__main__":
    main()
