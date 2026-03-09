#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "google-cloud-texttospeech>=2.14.0",
# ]
# requires-python = ">=3.11"
# ///

# Generate dialogue with multiple speakers
# https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers

import argparse
import re
import sys
import os
from google.cloud import texttospeech

def log_verbose(message, *format_args):
    if args.verbose:
        print(message.format(*format_args), file=sys.stderr)

def parse_input(input_file):
    speakers = {}
    turns = []
    last_speaker = None

    for line in input_file:
        line = line.strip()
        if line:
            if ':' in line:
                speaker, text = line.split(':', 1)
                speaker = re.sub(r'[*_~`]', '', speaker).strip()
                text = text.strip()

                if speaker not in speakers:
                    if len(speakers) >= len(args.speakers):
                        raise ValueError("More than {} speakers detected in the input.".format(len(args.speakers)))
                    speakers[speaker] = args.speakers[len(speakers)]
                    if args.verbose:
                        print('DEB:SPEAKER[{}]="{}"'.format(speaker, speakers[speaker]), file=sys.stderr)
            else:
                text = line
                speaker = last_speaker

            turn = texttospeech.MultiSpeakerMarkup.Turn()
            turn.text = text
            turn.speaker = speakers[speaker]
            turns.append(turn)
            last_speaker = speaker

            if args.verbose > 1:
                print('APPEND:[{} as "{}"]:{}'.format(speaker, speakers[speaker], text), file=sys.stderr)

    return turns

def choose_audio_encoding():
    return get_audio_encoding(args.encoding) if args.encoding else texttospeech.AudioEncoding.MP3

def list_voices():
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices(language_code=args.language)
    multi_speaker = []
    other = []
    for voice in response.voices:
        lang_codes = ", ".join(voice.language_codes)
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        entry = f"  {voice.name}  ({lang_codes}, {gender})"
        if "MultiSpeaker" in voice.name:
            multi_speaker.append(entry)
        else:
            other.append(entry)
    if multi_speaker:
        print("Multi-Speaker voices:")
        print("\n".join(multi_speaker))
    if other:
        print(f"\nOther voices for {args.language}:" if args.language else "\nOther voices:")
        print("\n".join(other))

def generate_audio(turns):
    client = texttospeech.TextToSpeechClient()

    multi_speaker_markup = texttospeech.MultiSpeakerMarkup()
    multi_speaker_markup.turns.extend(turns)

    synthesis_input = texttospeech.SynthesisInput(multi_speaker_markup=multi_speaker_markup)

    voice = texttospeech.VoiceSelectionParams(
        language_code=args.language, name=args.voice_name
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
    # "FLAC": texttospeech.AudioEncoding.FLAC,  # not yet supported: https://github.com/googleapis/google-cloud-python/issues/13239
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
    parser = argparse.ArgumentParser(description="Generate multi-speaker audio from input text.")
    parser.add_argument("-i", "--input", default=None, help="Input file (use '-' for stdin)")
    parser.add_argument("-s", "--speakers", default="R,S,T,U,V,W,X,Y", help="Speaker mapping (comma-separated, up to 8) [R,S,T,U,V,W,X,Y]")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Enable verbose logging")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Run the script without generating or writing audio")
    parser.add_argument("-e", "--encoding", default=None, help="Audio encoding (wav, mp3, ogg, mulaw, alaw)")
    parser.add_argument("-o", "--output", default=None, help="Output file (default is input filename with appropriate extension)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output file if it already exists")
    parser.add_argument("-l", "--language", default="en-US", help="Language code (default: en-US)")
    parser.add_argument("--voice-name", default=None, help="Voice name (default: {language}-Studio-MultiSpeaker)")
    parser.add_argument("--rate", type=float, default=None, help="Speaking rate 0.25-4.0 (default: 1.0)")
    parser.add_argument("--pitch", type=float, default=None, help="Pitch in semitones -20.0 to 20.0 (default: 0.0)")
    parser.add_argument("--volume", type=float, default=None, help="Volume gain in dB -96.0 to 16.0 (default: 0.0, recommend max +10)")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate in Hz (e.g. 16000, 24000, 48000)")
    parser.add_argument("--audio-profile", action="append", default=None,
                        help="Audio device profile (can be repeated). Options: "
                             "headphone-class-device, handset-class-device, "
                             "small-bluetooth-speaker-class-device, medium-bluetooth-speaker-class-device, "
                             "large-home-entertainment-class-device, large-automotive-class-device, "
                             "telephony-class-application, wearable-class-device")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    args = parser.parse_args()

    if args.voice_name is None:
        args.voice_name = f"{args.language}-Studio-MultiSpeaker"

    if args.list_voices:
        list_voices()
        sys.exit(0)

    if args.input is None:
        parser.error("the following arguments are required: -i/--input")

    args.speakers = args.speakers.split(',')

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

    log_verbose(f"Processing input from {args.input}")
    try:
        turns = parse_input(input_file)
    finally:
        if input_file is not sys.stdin:
            input_file.close()

    input_bytes = sum(len(t.text.encode('utf-8')) for t in turns)
    log_verbose("DEB:Parsed {} turns from {} speakers, ~{} bytes of text", len(turns), len(set(t.speaker for t in turns)), input_bytes)

    if input_bytes > 5000:
        print(f"WARNING: Input text is ~{input_bytes} bytes, API limit is 5000 bytes. Request may fail.", file=sys.stderr)

    if not args.dry_run:
        log_verbose("Generating audio")
        audio_content = generate_audio(turns)

        log_verbose(f"Writing audio to {output_file}")
        with open(output_file, "wb") as out:
            out.write(audio_content)

        print(f'Audio content written to file "{output_file}"')
    else:
        audio_encoding = choose_audio_encoding()
        encoding_str = next(key for key, value in encoding_map.items() if value == audio_encoding)
        log_verbose("DEB:Chosen audio encoding: {}", encoding_str)
        print(f"Dry run: {len(turns)} turns, ~{input_bytes} bytes. Output would be: {output_file}")

if __name__ == "__main__":
    main()
