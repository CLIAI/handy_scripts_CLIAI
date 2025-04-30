#!/usr/bin/env python
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


def upload_file(api_token, audio_input, args):
    if audio_input.startswith('http://') or audio_input.startswith('https://'):
        return audio_input
    url = f"{args.base_url}/v2/upload"
    headers = {
        'authorization': api_token,
        'content-type': 'application/octet-stream'
    }
    response = None  # ensure defined for exception handling
    try:
        with open(audio_input, 'rb') as f:
            response = requests.post(url, headers=headers, data=f)
        response.raise_for_status()
        upload_url = response.json()['upload_url']
        log_info(args, f"File uploaded. URL: {upload_url}")
        log_debug(args, f"Upload response JSON: {response.text}")
        return upload_url
    except Exception as e:
        log_error(args, f"Error in upload_file: {e}")
        if response:
            log_error(args, f"REST RESPONSE: {response.text}")
        raise

def create_transcript(api_token, audio_url, speaker_labels, args):
    url = f"{args.base_url}/v2/transcript"
    headers = {
        "authorization": api_token,
        "content-type": "application/json"
    }
    response = None  # ensure defined for exception handling
    data = {
        "audio_url": audio_url,
        "speaker_labels": speaker_labels,
    }
    if args.language != 'auto':
        data["language_code"] = args.language
    if args.expected_speakers != -1:
        data["speakers_expected"] = args.expected_speakers

    log_debug(args, f"Transcript request payload: {json.dumps(data)}")
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        transcript_id = response.json()['id']
        log_info(args, f"Transcript ID: {transcript_id}")
        
        polling_endpoint = f"{args.base_url}/v2/transcript/{transcript_id}"
        while True:
            response = requests.get(polling_endpoint, headers=headers)
            response.raise_for_status()
            transcription_result = response.json()
            status = transcription_result['status']
            log_info(args, f"Current status: {status}")
            log_debug(args, f"Full status JSON: {json.dumps(transcription_result)}")
            if status == "completed":
                return transcription_result
            elif status == "error":
                raise Exception(f"Transcription failed: {transcription_result['error']}")
            elif status in ["queued", "processing"]:
                time.sleep(5)
            else:
                raise Exception(f"Unknown status: {status}")
    except Exception as e:
        log_error(args, f"Error in create_transcript: {e}")
        if response:
            log_error(args, f"REST RESPONSE: {response.text}")
        raise

def write_str(args, output, string, mode='w'):
    if output != '-':
        with open(output, mode) as f:
            f.write(string)
    if output == '-' or not args.quiet:
        print(string)

def write_transcript_to_file(args, output, transcript, audio_input):
    import copy
    args_force_quiet = copy.deepcopy(args)
    args_force_quiet.quiet = True
    write_str(args_force_quiet, audio_input + '.assemblyai.json', json.dumps(transcript))
    if not args.quiet:
        log_info(args, f"Server response written to {audio_input + '.assemblyai.json'}")
    
    if args.diarisation:
        for utterance in transcript['utterances']:
            chunk_str = f"Speaker {utterance['speaker']}:" + utterance['text'] + '\n'
            write_str(args, output, chunk_str, 'a')
    else:
        write_str(args, output, transcript['text'] + '\n')

    if output != '-' and not args.quiet:
        log_info(args, f"Output written to {output}")

def stt_assemblyai_main(args, api_token):
    audio_input = args.audio_input
    speaker_labels = args.diarisation

    try:
        log_info(args, "Processing audio input...")

        # Determine the output file
        if args.output == '-':
            potential_output = audio_input + '.txt'
            output = potential_output if os.path.exists(potential_output) else '-'
        else:
            output = args.output if args.output is not None else audio_input + '.txt'
        log_info(args, f"output filename: {output}")
        
        # Check if output file exists before making the transcript
        if os.path.exists(output):
            if not args.quiet and args.verbose:
                sys.stderr.write(f'SKIPPING: transcription of {audio_input} as {output} already exists\n')
            if (not args.quiet) or args.output == '-':
                with open(output, 'r') as f:
                    print(f.read())
            sys.exit(0)
        
        # Create the transcript
        log_info(args, "Uploading audio file...")
        upload_url = upload_file(api_token, audio_input, args)
        log_info(args, "Creating transcript...")
        transcript = create_transcript(api_token, upload_url, speaker_labels, args)
        
        # Write the transcript to the output file
        log_info(args, "Transcript created. Writing output...")
        write_transcript_to_file(args, output, transcript, audio_input)
        log_info(args, "Done.")
    except Exception as e:
        log_error(args, f'Error: {e}')
        sys.exit(1)

def make_arg_parser():
    parser = argparse.ArgumentParser(description='Transcribe audio file using AssemblyAI API.')
    parser.add_argument('audio_input', type=str, help='The path to the audio file or URL to transcribe.')
    parser.add_argument('-d', '--diarisation', action='store_true', help='Enable speaker diarisation. This will label each speaker in the transcription.')
    parser.add_argument('-o', '--output', type=str, default=None, help='The path to the output file to store the result. If not provided, the result will be saved to a file with the same name as the input file but with a .txt extension. If "-" is provided, the result will be printed only to standard output and no files saved.')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress all status messages to standard output. If an output file is specified, the result will still be saved to that file (or standard output if `-` is specified).')
    parser.add_argument('-e', '--expected-speakers', type=int, default=-1, help='The expected number of speakers for diarisation. This helps improve the accuracy of speaker labelling.')
    parser.add_argument('-l', '--language', type=str, default='auto', help='The dominant language in the audio file. Example codes: en, en_au, en_uk, en_us, es, fr, de, it, pt, nl, hi, ja, zh, fi, ko, pl, ru. Default is "auto" for automatic language detection.')
    parser.add_argument('-R', '--region', choices=['eu','us'], default='eu', help='Select region for API endpoints: "eu" or "us". Defaults to EU')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity. Use -v for INFO, -vv for more detail, up to -vvvvv for DEBUG output.')
    return parser

if __name__ == "__main__":
    try:
        api_token = os.environ["ASSEMBLYAI_API_KEY"]
    except KeyError:
        print("Error: ASSEMBLYAI_API_KEY environment variable not set.")
        sys.exit(1)
    parser = make_arg_parser()
    args = parser.parse_args()
    # Set base URL based on region selection
    args.base_url = 'https://api.eu.assemblyai.com' if args.region == 'eu' else 'https://api.assemblyai.com'
    stt_assemblyai_main(args, api_token)
