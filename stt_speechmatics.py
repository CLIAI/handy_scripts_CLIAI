#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "requests>=2.31",
# ]
# requires-python = ">=3.11"
# ///
"""
Speechmatics Speech-to-Text (STT) Transcription Tool

Transcribe audio files using the Speechmatics API with support for speaker
diarisation, multiple languages, and batch processing.

Usage:
    ./stt_speechmatics.py audio.mp3
    ./stt_speechmatics.py -d audio.mp3  # With speaker diarisation
    ./stt_speechmatics.py -l en -d -e 3 audio.mp3  # English, 3 speakers

Environment:
    SPEECHMATICS_API_KEY - Your Speechmatics API key (required)
"""

import os
import requests
import time
import argparse
import sys
import json

# ----------------------------------------------------------------------
# Simple verbosity-aware logger.
#   -v          → INFO messages
#   -vvvvv      → DEBUG messages
# All logs are written to STDERR and are suppressed by --quiet
# ----------------------------------------------------------------------
def _should_log(args, level_threshold):
    # Return True when the current verbosity meets the threshold
    return getattr(args, "verbose", 0) >= level_threshold and not getattr(args, "quiet", False)

def log_error(args, message):
    # Errors are always shown
    print(f"ERROR: {message}", file=sys.stderr)

def log_warning(args, message):
    if _should_log(args, 0):
        print(f"WARNING: {message}", file=sys.stderr)

def log_info(args, message):
    if _should_log(args, 1):
        print(f"INFO: {message}", file=sys.stderr)

def log_debug(args, message):
    if _should_log(args, 5):
        print(f"DEBUG: {message}", file=sys.stderr)

# ----------------------------------------------------------------------
# Filename helpers for long filenames
# ----------------------------------------------------------------------

KNOWN_EXTENSIONS = {
    "mp3", "mp4", "wav", "flac", "m4a", "ogg",
    "json", "txt", "srt", "vtt", "md",
    "speechmatics"
}


def _split_known_suffix(filename, known_extensions=KNOWN_EXTENSIONS):
    """
    Return (basename, chained_suffix_with_dot)
    Example:
        >>> _split_known_suffix("foo.bar.mp4.speechmatics.json")
        ('foo.bar', '.mp4.speechmatics.json')
    """
    parts = filename.split(".")
    if len(parts) == 1:
        return filename, ""

    suffix_parts = []
    for part in reversed(parts[1:]):  # skip first chunk
        if part.lower() in known_extensions:
            suffix_parts.insert(0, part)
        else:
            break

    if suffix_parts:
        suffix = "." + ".".join(suffix_parts)
        base_parts_count = len(parts) - len(suffix_parts)
        basename = ".".join(parts[:base_parts_count])
        return basename, suffix
    else:
        basename, ext = os.path.splitext(filename)
        return basename, ext


def make_safe_filename(path, max_component_length=255):
    """
    Ensure the final component of *path* is <= max_component_length bytes.
    If it is longer, truncate the basename until it fits.
    """
    dir_name, file_name = os.path.split(path)
    if len(file_name.encode()) <= max_component_length:
        return path

    base, suffix = _split_known_suffix(file_name)
    allowed = max(1, max_component_length - len(suffix.encode()))
    truncated_base = base.encode()[:allowed].decode(errors="ignore")
    safe_name = truncated_base + suffix
    return os.path.join(dir_name, safe_name)


# ----------------------------------------------------------------------
# Region configuration
# ----------------------------------------------------------------------

REGIONS = {
    'eu': 'https://eu1.asr.api.speechmatics.com/v2',
    'eu1': 'https://eu1.asr.api.speechmatics.com/v2',
    'us': 'https://us1.asr.api.speechmatics.com/v2',
    'us1': 'https://us1.asr.api.speechmatics.com/v2',
    'au': 'https://au1.asr.api.speechmatics.com/v2',
    'au1': 'https://au1.asr.api.speechmatics.com/v2',
}


# ----------------------------------------------------------------------
# Speaker identification integration
# ----------------------------------------------------------------------

def load_enrolled_speakers(tag: str, context: str = "default", args=None):
    """
    Load enrolled speakers from speaker_detection database.

    Args:
        tag: Tag to filter speakers
        context: Name context to use for labels
        args: For logging

    Returns:
        List of speaker configs for Speechmatics API, or empty list
    """
    try:
        # Import from speaker_detection module
        import sys
        from pathlib import Path

        # Add current directory to path if needed
        script_dir = Path(__file__).parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        # Try to import speaker_detection functions
        try:
            # Read speaker database directly (avoid circular import)
            import os
            db_dir = Path(os.environ.get(
                "SPEAKERS_EMBEDDINGS_DIR",
                os.path.expanduser("~/.config/speakers_embeddings")
            )) / "db"

            if not db_dir.exists():
                log_info(args, f"Speaker database not found: {db_dir}")
                return []

            speakers = []
            for path in db_dir.glob("*.json"):
                try:
                    with open(path) as f:
                        speakers.append(json.load(f))
                except (json.JSONDecodeError, IOError):
                    continue

            # Filter by tag
            tag_set = set(t.strip() for t in tag.split(","))
            filtered = [
                s for s in speakers
                if tag_set & set(s.get("tags", []))
            ]

            if not filtered:
                log_info(args, f"No speakers found with tag(s): {tag}")
                return []

            # Build speaker configs
            speakers_config = []
            for speaker in filtered:
                embs = speaker.get("embeddings", {}).get("speechmatics", [])
                if not embs:
                    continue

                # Collect all identifiers
                identifiers = []
                for emb in embs:
                    if emb.get("external_id"):
                        identifiers.append(emb["external_id"])
                    identifiers.extend(emb.get("all_identifiers", []))

                if identifiers:
                    # Get name for the specified context
                    name = speaker.get("names", {}).get(
                        context,
                        speaker.get("names", {}).get("default", speaker["id"])
                    )
                    speakers_config.append({
                        "label": name,
                        "speaker_identifiers": list(set(identifiers))[:50],  # API max
                    })

            log_info(args, f"Loaded {len(speakers_config)} enrolled speakers for identification")
            return speakers_config

        except Exception as e:
            log_warning(args, f"Error loading speakers: {e}")
            return []

    except Exception as e:
        log_warning(args, f"Speaker identification not available: {e}")
        return []


# ----------------------------------------------------------------------
# API functions
# ----------------------------------------------------------------------

def create_job(api_token, audio_input, args):
    """
    Submit a transcription job to Speechmatics.

    Args:
        api_token: API key
        audio_input: Path to audio file or URL
        args: Parsed arguments

    Returns:
        Job ID string
    """
    url = f"{args.base_url}/jobs"
    headers = {
        'Authorization': f'Bearer {api_token}'
    }

    # Build transcription config
    transcription_config = {
        "language": args.language,
    }

    # Add operating point (standard or enhanced)
    if args.operating_point:
        transcription_config["operating_point"] = args.operating_point

    # Add diarisation config
    if args.diarisation:
        transcription_config["diarization"] = "speaker"

        # Add speaker diarisation config if options specified
        speaker_config = {}
        if args.max_speakers > 0:
            speaker_config["max_speakers"] = args.max_speakers
        if args.speaker_sensitivity is not None:
            speaker_config["speaker_sensitivity"] = args.speaker_sensitivity

        # Add enrolled speakers for identification
        speakers_tag = getattr(args, 'speakers_tag', None)
        if speakers_tag:
            speakers_context = getattr(args, 'speakers_context', 'default')
            enrolled = load_enrolled_speakers(speakers_tag, speakers_context, args)
            if enrolled:
                speaker_config["speakers"] = enrolled

        if speaker_config:
            transcription_config["speaker_diarization_config"] = speaker_config

    config = {
        "type": "transcription",
        "transcription_config": transcription_config,
    }

    log_debug(args, f"Job config: {json.dumps(config, indent=2)}")

    response = None
    try:
        # Check if input is URL or file
        if audio_input.startswith('http://') or audio_input.startswith('https://'):
            # URL-based submission
            config["fetch_data"] = {"url": audio_input}
            response = requests.post(url, headers=headers, json=config)
        else:
            # File-based submission (multipart form)
            with open(audio_input, 'rb') as f:
                files = {
                    'data_file': (os.path.basename(audio_input), f),
                }
                data = {
                    'config': json.dumps(config)
                }
                response = requests.post(url, headers=headers, files=files, data=data)

        response.raise_for_status()
        result = response.json()
        job_id = result['id']
        log_info(args, f"Job created: {job_id}")
        log_debug(args, f"Job response: {json.dumps(result, indent=2)}")
        return job_id

    except Exception as e:
        log_error(args, f"Error creating job: {e}")
        if response:
            log_error(args, f"REST RESPONSE: {response.text}")
        raise


def wait_for_job(api_token, job_id, args, poll_interval=5, max_wait=3600):
    """
    Poll job status until complete or error.

    Args:
        api_token: API key
        job_id: Job ID to monitor
        args: Parsed arguments
        poll_interval: Seconds between polls
        max_wait: Maximum wait time in seconds

    Returns:
        Final job status dict
    """
    url = f"{args.base_url}/jobs/{job_id}"
    headers = {
        'Authorization': f'Bearer {api_token}'
    }

    start_time = time.time()
    response = None

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            job = response.json()['job']
            status = job.get('status')

            log_info(args, f"Job status: {status}")
            log_debug(args, f"Job details: {json.dumps(job, indent=2)}")

            if status == 'done':
                return job
            elif status == 'rejected':
                error_msg = job.get('errors', [{}])[0].get('message', 'Unknown error')
                raise Exception(f"Job rejected: {error_msg}")
            elif status == 'running':
                time.sleep(poll_interval)
            else:
                # Unexpected status
                log_warning(args, f"Unexpected status: {status}")
                time.sleep(poll_interval)

    except Exception as e:
        log_error(args, f"Error waiting for job: {e}")
        if response:
            log_error(args, f"REST RESPONSE: {response.text}")
        raise


def get_transcript(api_token, job_id, args, format='json-v2'):
    """
    Retrieve transcript for completed job.

    Args:
        api_token: API key
        job_id: Completed job ID
        args: Parsed arguments
        format: Output format (json-v2, txt, srt)

    Returns:
        Transcript content (dict for json-v2, string for txt/srt)
    """
    url = f"{args.base_url}/jobs/{job_id}/transcript"
    headers = {
        'Authorization': f'Bearer {api_token}'
    }
    params = {'format': format}

    response = None
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        if format == 'json-v2':
            return response.json()
        else:
            return response.text

    except Exception as e:
        log_error(args, f"Error getting transcript: {e}")
        if response:
            log_error(args, f"REST RESPONSE: {response.text}")
        raise


# ----------------------------------------------------------------------
# META message helper
# ----------------------------------------------------------------------

def get_meta_message(args):
    """
    Get META warning message for STT transcripts.

    Returns empty string if disabled via flag or environment variable.
    Returns custom message if STT_META_MESSAGE env var is set.
    Returns default message otherwise.
    """
    # Check command-line flag
    if getattr(args, 'no_meta_message', False) or getattr(args, 'disable_meta_message', False):
        return ""

    # Check environment variable for disabling
    if os.environ.get('STT_META_MESSAGE_DISABLE', '').lower() in ('1', 'true', 'yes'):
        return ""

    # Check for custom message
    custom_message = os.environ.get('STT_META_MESSAGE', '').strip()
    if custom_message:
        return f"---\nmeta: {custom_message}\n---\n"

    # Default META message
    default_message = (
        "THIS IS AN AUTOMATED SPEECH-TO-TEXT (STT) TRANSCRIPT AND MAY CONTAIN TRANSCRIPTION ERRORS. "
        "This transcript was generated by automated speech recognition technology and should be treated "
        "as a rough transcription for reference purposes. Common types of errors include: incorrect word "
        "recognition (especially homophones, proper nouns, technical terminology, or words in noisy audio "
        "conditions), missing or incorrect punctuation, speaker misidentification in multi-speaker scenarios, "
        "and timing inaccuracies. For best comprehension and to mentally correct potential errors, please consider: "
        "the broader conversational context, relevant domain knowledge, technical background of the subject matter, "
        "and any supplementary information about the speakers or topic. This transcript is intended to convey "
        "the general content and flow of the conversation rather than serving as a verbatim, word-perfect record. "
        "When critical accuracy is required, please verify important details against the original audio source."
    )

    return f"---\nmeta: {default_message}\n---\n"


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

def format_transcript_txt(transcript_json, args):
    """
    Format JSON transcript as human-readable text.

    Speechmatics format:
    - results array contains words/punctuation
    - speaker field contains S1, S2, etc. (or UU for unknown)

    Args:
        transcript_json: JSON transcript from API
        args: Parsed arguments

    Returns:
        Formatted text string
    """
    results = transcript_json.get('results', [])

    if not results:
        return ""

    lines = []
    current_speaker = None
    current_text = []

    for item in results:
        item_type = item.get('type')

        if item_type == 'word':
            # Speaker can be at top level or inside alternatives
            speaker = item.get('speaker')
            # Get content from alternatives
            content = ''
            alternatives = item.get('alternatives', [])
            if alternatives:
                content = alternatives[0].get('content', '')
                # Also check for speaker in alternatives (used with speaker identification)
                if not speaker:
                    speaker = alternatives[0].get('speaker')
            speaker = speaker or 'UU'

            if args.diarisation:
                if speaker != current_speaker:
                    # Flush current text
                    if current_text and current_speaker:
                        lines.append(f"Speaker {current_speaker}:\t{' '.join(current_text)}")
                    current_speaker = speaker
                    current_text = [content] if content else []
                else:
                    if content:
                        current_text.append(content)
            else:
                if content:
                    current_text.append(content)

        elif item_type == 'punctuation':
            # Get content from alternatives
            content = ''
            if item.get('alternatives'):
                content = item['alternatives'][0].get('content', '')
            if content and current_text:
                # Append punctuation to last word (no space)
                current_text[-1] = current_text[-1] + content

    # Flush remaining text
    if current_text:
        if args.diarisation and current_speaker:
            lines.append(f"Speaker {current_speaker}:\t{' '.join(current_text)}")
        else:
            lines.append(' '.join(current_text))

    return '\n'.join(lines) + '\n' if lines else ""


def write_str(args, output, string, mode='w'):
    if output != '-':
        with open(output, mode) as f:
            f.write(string)
    if output == '-' or not args.quiet:
        print(string, end='')


def write_transcript_to_file(args, output, transcript_json, audio_input):
    """Write transcript to output files."""
    import copy
    args_force_quiet = copy.deepcopy(args)
    args_force_quiet.quiet = True
    json_path = make_safe_filename(audio_input + '.speechmatics.json')

    # Add META note to JSON if enabled
    meta_message_text = get_meta_message(args).replace("---\nmeta: ", "").replace("\n---\n", "").strip()
    if meta_message_text:
        transcript_with_meta = {
            "_meta_note": meta_message_text,
            **transcript_json
        }
        write_str(args_force_quiet, json_path, json.dumps(transcript_with_meta, indent=2))
    else:
        write_str(args_force_quiet, json_path, json.dumps(transcript_json, indent=2))

    if not args.quiet:
        log_info(args, f"Server response written to {json_path}")

    # Prepend META message to TXT output if enabled
    meta_message = get_meta_message(args)
    if meta_message:
        write_str(args, output, meta_message)

    # Format and write text transcript
    txt_content = format_transcript_txt(transcript_json, args)
    write_str(args, output, txt_content, 'a' if meta_message else 'w')

    if output != '-' and not args.quiet:
        log_info(args, f"Output written to {output}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def stt_speechmatics_main(args, api_token):
    audio_input = args.audio_input

    try:
        log_info(args, "Processing audio input...")

        # Determine the output file
        if args.output == '-':
            potential_output = audio_input + '.txt'
            output = potential_output if os.path.exists(potential_output) else '-'
        else:
            output = args.output if args.output is not None else audio_input + '.txt'
            output = make_safe_filename(output)
        log_info(args, f"output filename: {output}")

        # Check if output file exists before making the transcript
        if os.path.exists(output):
            if not args.quiet and args.verbose:
                sys.stderr.write(f'SKIPPING: transcription of {audio_input} as {output} already exists\n')
            if (not args.quiet) or args.output == '-':
                with open(output, 'r') as f:
                    print(f.read())
            sys.exit(0)

        # Create and submit job
        log_info(args, "Submitting transcription job...")
        job_id = create_job(api_token, audio_input, args)

        # Wait for completion
        log_info(args, "Waiting for job to complete...")
        job = wait_for_job(api_token, job_id, args)

        # Get transcript
        log_info(args, "Retrieving transcript...")
        transcript_json = get_transcript(api_token, job_id, args)

        # Write output
        log_info(args, "Writing output files...")
        write_transcript_to_file(args, output, transcript_json, audio_input)
        log_info(args, "Done.")

    except Exception as e:
        log_error(args, f'Error: {e}')
        sys.exit(1)


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description='Transcribe audio file using Speechmatics API.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3
  %(prog)s -d audio.mp3                    # With speaker diarisation
  %(prog)s -d --max-speakers 3 audio.mp3   # Limit to 3 speakers
  %(prog)s -l de audio.mp3                 # German transcription
  %(prog)s -R us -d audio.mp3              # US region with diarisation
  %(prog)s -d --speakers-tag podcast audio.mp3  # Use enrolled speakers

Environment:
  SPEECHMATICS_API_KEY       Your Speechmatics API key (required)
  SPEAKERS_EMBEDDINGS_DIR    Speaker database location (for --speakers-tag)
  STT_META_MESSAGE_DISABLE=1 Disable META warning message
  STT_META_MESSAGE="..."     Custom META message
"""
    )
    parser.add_argument('audio_input', type=str,
                        help='Path to audio file or URL to transcribe')
    parser.add_argument('-d', '--diarisation', action='store_true',
                        help='Enable speaker diarisation (S1, S2, S3, etc.)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path. Default: {audio}.txt. Use "-" for stdout only.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress all status messages')
    parser.add_argument('--max-speakers', type=int, default=0,
                        help='Maximum number of speakers for diarisation (minimum: 2, default: unlimited)')
    parser.add_argument('--speaker-sensitivity', type=float, default=None,
                        help='Speaker detection sensitivity (0-1, default: 0.5). Higher = more speakers.')
    parser.add_argument('-l', '--language', type=str, default='en',
                        help='Language code (ISO 639-1). Default: en. Examples: de, fr, es, ja, zh')
    parser.add_argument('-R', '--region', choices=['eu', 'eu1', 'us', 'us1', 'au', 'au1'],
                        default='eu',
                        help='API region: eu (EU1), us (US1), au (AU1). Default: eu')
    parser.add_argument('--operating-point', choices=['standard', 'enhanced'],
                        default=None,
                        help='Model accuracy: standard (faster) or enhanced (more accurate)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity. -v for INFO, -vvvvv for DEBUG.')
    parser.add_argument('--no-meta-message', '--disable-meta-message', action='store_true',
                        dest='no_meta_message',
                        help='Disable META warning message about transcription errors')
    parser.add_argument('--speakers-tag', metavar='TAG',
                        help='Use enrolled speakers from speaker_detection with this tag')
    parser.add_argument('--speakers-context', metavar='CTX', default='default',
                        help='Name context to use for speaker labels (default: default)')
    return parser


if __name__ == "__main__":
    try:
        api_token = os.environ["SPEECHMATICS_API_KEY"]
    except KeyError:
        print("Error: SPEECHMATICS_API_KEY environment variable not set.")
        print("Get your API key at: https://portal.speechmatics.com/")
        sys.exit(1)

    parser = make_arg_parser()
    args = parser.parse_args()

    # Set base URL based on region
    args.base_url = REGIONS.get(args.region, REGIONS['eu'])

    stt_speechmatics_main(args, api_token)
