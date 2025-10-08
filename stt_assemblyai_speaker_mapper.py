#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "instructor>=1.0.0",
#   "pydantic>=2.0.0",
#   "openai>=1.0.0",
# ]
# requires-python = ">=3.11"
# ///
"""
AssemblyAI Speaker Name Mapper

Post-processing tool to replace speaker labels (A, B, C) with actual names
in AssemblyAI transcription JSON files. Uses recursive traversal to handle
any JSON structure, making it future-proof and format-agnostic.

Usage:
    # Detect speakers
    ./stt_assemblyai_speaker_mapper.py --detect audio.assemblyai.json

    # Map via inline comma-separated names
    ./stt_assemblyai_speaker_mapper.py -m "Alice,Bob" audio.assemblyai.json

    # Map via file (auto-detects format)
    ./stt_assemblyai_speaker_mapper.py -M speakers.txt audio.assemblyai.json

    # Interactive mapping
    ./stt_assemblyai_speaker_mapper.py --interactive audio.assemblyai.json
"""

import argparse
import sys
import json
import os
import re
from typing import Dict, List, Optional, Union

# Optional LLM detection support
try:
    import instructor
    from pydantic import BaseModel, Field
    from openai import OpenAI
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    instructor = None
    BaseModel = None
    Field = None
    OpenAI = None

# ----------------------------------------------------------------------
# Verbosity-aware logger (matches stt_assemblyai.py pattern)
# ----------------------------------------------------------------------
def _should_log(args, level_threshold):
    return getattr(args, "verbose", 0) >= level_threshold and not getattr(args, "quiet", False)

def log_error(args, message):
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
# Core JSON Traversal Functions
# ----------------------------------------------------------------------

def detect_speakers_in_json(json_obj, found=None):
    """
    Recursively find all unique values associated with "speaker" keys.

    Args:
        json_obj: JSON object (dict/list/primitive)
        found: Set to accumulate speaker labels (internal use)

    Returns:
        Set of speaker labels (e.g., {'A', 'B', 'C'})
    """
    if found is None:
        found = set()

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == "speaker" and isinstance(value, str):
                found.add(value)
            else:
                detect_speakers_in_json(value, found)
    elif isinstance(json_obj, list):
        for item in json_obj:
            detect_speakers_in_json(item, found)

    return found


def replace_speakers_recursive(obj, speaker_map):
    """
    Recursively traverse JSON and replace all "speaker" key values.

    Args:
        obj: JSON object (dict/list/primitive)
        speaker_map: Dict mapping speaker labels (e.g., {'A': 'Alice Anderson'})

    Returns:
        Modified copy with speaker replacements applied
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == "speaker" and isinstance(value, str):
                # Found a speaker field - apply mapping if available
                result[key] = speaker_map.get(value, value)
            else:
                # Recurse into nested structures
                result[key] = replace_speakers_recursive(value, speaker_map)
        return result

    elif isinstance(obj, list):
        # Recurse into list items
        return [replace_speakers_recursive(item, speaker_map) for item in obj]

    else:
        # Primitive value - return as-is
        return obj


def find_transcript_segments(json_obj):
    """
    Find lists of transcript segments in JSON.

    Heuristic: Look for lists of dicts containing 'speaker' and 'text' keys.
    Tries common paths first ('utterances'), then searches recursively.

    Args:
        json_obj: JSON object to search

    Returns:
        List of segment dicts with 'speaker' and 'text' keys
    """
    # Fast path: check common AssemblyAI location
    if isinstance(json_obj, dict) and 'utterances' in json_obj:
        return json_obj['utterances']

    # Recursive search
    segments = []

    if isinstance(json_obj, dict):
        for value in json_obj.values():
            segments.extend(find_transcript_segments(value))

    elif isinstance(json_obj, list):
        # Check if this list looks like transcript segments
        if json_obj and isinstance(json_obj[0], dict):
            if 'speaker' in json_obj[0] and 'text' in json_obj[0]:
                return json_obj

        # Otherwise recurse into list items
        for item in json_obj:
            segments.extend(find_transcript_segments(item))

    return segments


# ----------------------------------------------------------------------
# LLM-Assisted Speaker Detection (Optional)
# ----------------------------------------------------------------------

if INSTRUCTOR_AVAILABLE:
    class SpeakerMapping(BaseModel):
        """Individual speaker label to name mapping."""
        speaker_label: str = Field(
            description="Speaker label from transcript (e.g., 'A', 'B', 'SPEAKER_00')"
        )
        speaker_name: str = Field(
            description="Identified name or role for this speaker"
        )

    class SpeakerDetection(BaseModel):
        """Pydantic model for LLM speaker detection response."""
        speakers: List[SpeakerMapping] = Field(
            description='List of speaker mappings. Must include one mapping for EACH detected speaker label.'
        )
        confidence: str = Field(
            description="Confidence level: low, medium, or high",
            default="medium"
        )
        reasoning: str = Field(
            description="Brief explanation of how speakers were identified",
            default=""
        )


def extract_transcript_sample(json_obj: dict, max_utterances: int = 20) -> str:
    """
    Extract strategic sample of transcript for LLM analysis.

    Strategy:
    1. Include first few utterances (introductions often here)
    2. Include utterances with potential name mentions (proper nouns)
    3. Include utterances from each speaker
    4. Limit total to avoid token limits

    Args:
        json_obj: Full AssemblyAI JSON
        max_utterances: Maximum utterances to include

    Returns:
        Formatted transcript sample string
    """
    utterances = json_obj.get('utterances', [])

    if not utterances:
        return ""

    # Strategy 1: First N utterances (catch introductions)
    first_n = min(10, len(utterances))
    sample_utterances = utterances[:first_n]

    # Strategy 2: Add utterances with potential names (proper nouns)
    if len(utterances) > first_n:
        for utt in utterances[first_n:]:
            text = utt.get('text', '')
            # Simple heuristic: contains capitalized words (potential names)
            if has_proper_nouns(text) and len(sample_utterances) < max_utterances:
                sample_utterances.append(utt)

    # Strategy 3: Ensure all speakers represented
    represented_speakers = {u.get('speaker') for u in sample_utterances}
    all_speakers = {u.get('speaker') for u in utterances}
    missing_speakers = all_speakers - represented_speakers

    if missing_speakers and len(sample_utterances) < max_utterances:
        for utt in utterances:
            if utt.get('speaker') in missing_speakers:
                sample_utterances.append(utt)
                missing_speakers.remove(utt.get('speaker'))
                if not missing_speakers or len(sample_utterances) >= max_utterances:
                    break

    # Format as readable transcript
    lines = []
    for utt in sample_utterances:
        speaker = utt.get('speaker', 'Unknown')
        text = utt.get('text', '')
        lines.append(f"Speaker {speaker}: {text}")

    return '\n'.join(lines)


def has_proper_nouns(text: str) -> bool:
    """
    Check if text contains capitalized words (potential names).

    Args:
        text: Input text to check

    Returns:
        True if proper nouns detected
    """
    # Match capitalized words that aren't sentence starts
    pattern = r'(?<![.!?]\s)(?<!\A)\b[A-Z][a-z]+'
    return bool(re.search(pattern, text))


def detect_speakers_llm(
    provider_model: str,
    transcript_sample: str,
    detected_labels: List[str],
    endpoint: Optional[str] = None,
    args=None
):
    """
    Detect speaker names using LLM via Instructor.

    Args:
        provider_model: Provider and model string (e.g., "openai/gpt-4o-mini")
        transcript_sample: Sample of transcript text
        detected_labels: List of detected speaker labels (e.g., ['A', 'B'])
        endpoint: Optional custom endpoint URL
        args: Arguments namespace (for logging)

    Returns:
        SpeakerDetection object with suggested speaker names

    Raises:
        RuntimeError: If Instructor is not available
        Exception: If LLM detection fails
    """
    if not INSTRUCTOR_AVAILABLE:
        raise RuntimeError(
            "Instructor library not available. Install with: pip install instructor openai"
        )

    log_debug(args, f"LLM provider: {provider_model}")
    log_debug(args, f"Detected labels: {detected_labels}")

    # Build prompt
    prompt = f"""Analyze this conversation transcript and identify the speakers.

DETECTED SPEAKERS: {', '.join(detected_labels)}

Your task is to create a mapping of each detected speaker label to their actual name or professional role.

Look for:
- Direct name mentions (e.g., "Hi Alice", "Thanks Bob")
- Introductions ("I'm...", "My name is...")
- Self-references using third person ("Alice is happy", "Bob appreciates")
- Professional roles if names aren't mentioned (Host, Guest, Expert, Interviewer)

TRANSCRIPT SAMPLE:
{transcript_sample}

You must provide a mapping for EACH detected speaker label ({', '.join(detected_labels)}) to their identified name or role.
Use "Unknown" only if you cannot determine identity with reasonable confidence.

Example output format:
- If detected speakers are ["A", "B"], you should return: {{"A": "Alice Anderson", "B": "Bob Martinez"}}
"""

    try:
        # Create Instructor client
        if endpoint:
            # Custom endpoint (e.g., remote Ollama)
            log_info(args, f"Using custom endpoint: {endpoint}")
            base_client = OpenAI(base_url=endpoint, api_key="none")
            client = instructor.from_openai(
                base_client,
                mode=instructor.Mode.JSON
            )
            model = provider_model.split("/")[1] if "/" in provider_model else provider_model
        else:
            # Standard provider
            log_info(args, f"Using provider: {provider_model}")
            client = instructor.from_provider(provider_model, mode=instructor.Mode.TOOLS)
            # Extract model name from provider_model (e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini")
            model = provider_model.split("/")[1] if "/" in provider_model else provider_model

        # Call LLM with structured output
        log_debug(args, "Calling LLM...")
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=SpeakerDetection,
            max_retries=3  # Automatic retry on validation errors
        )

        log_debug(args, f"LLM response - Confidence: {result.confidence}")
        log_debug(args, f"LLM response - Speakers: {result.speakers}")
        log_debug(args, f"LLM response - Reasoning: {result.reasoning}")

        # Convert List[SpeakerMapping] back to Dict[str, str] for compatibility
        result.speakers = {mapping.speaker_label: mapping.speaker_name for mapping in result.speakers}

        return result

    except Exception as e:
        log_error(args, f"LLM detection failed: {e}")
        raise


def handle_llm_detection(args, json_data, detected_speakers):
    """
    Handle LLM-assisted speaker detection.

    Args:
        args: Parsed arguments
        json_data: Full transcript JSON
        detected_speakers: Set of detected speaker labels

    Returns:
        Speaker mapping dict
    """
    # Determine provider spec
    provider_spec = args.llm_detect or args.llm_interactive or args.llm_detect_fallback

    if not provider_spec:
        # Default provider
        provider_spec = "openai/gpt-4o-mini"
        log_info(args, f"No provider specified, using default: {provider_spec}")

    try:
        # Extract transcript sample
        transcript_sample = extract_transcript_sample(
            json_data,
            max_utterances=args.llm_sample_size
        )

        if not transcript_sample:
            raise ValueError("No transcript utterances found for LLM analysis")

        log_debug(args, f"Transcript sample ({len(transcript_sample)} chars)")
        log_debug(args, f"Sample preview: {transcript_sample[:200]}...")

        # Call LLM
        log_info(args, "Analyzing transcript with LLM...")
        detection_result = detect_speakers_llm(
            provider_spec,
            transcript_sample,
            list(detected_speakers),
            endpoint=args.llm_endpoint,
            args=args
        )

        log_info(args, f"LLM confidence: {detection_result.confidence}")
        if detection_result.reasoning:
            log_info(args, f"LLM reasoning: {detection_result.reasoning}")

        # Interactive mode: show suggestions as defaults
        if args.llm_interactive:
            return prompt_interactive_with_suggestions(
                detected_speakers,
                detection_result.speakers,
                args
            )

        # Auto mode: use LLM suggestions directly
        else:
            speaker_map = detection_result.speakers

            # Warn about unknown speakers
            for speaker, name in speaker_map.items():
                if name.lower() == "unknown":
                    log_warning(args, f"LLM could not identify speaker {speaker}")

            return speaker_map

    except Exception as e:
        log_error(args, f"LLM detection failed: {e}")

        # Fallback mode: continue with manual
        if args.llm_detect_fallback:
            log_warning(args, "Falling back to manual interactive mode")
            return prompt_interactive_mapping(detected_speakers, args)
        else:
            raise


def prompt_interactive_with_suggestions(
    detected_speakers: set,
    ai_suggestions: dict,
    args
) -> dict:
    """
    Interactive prompts with AI suggestions as defaults.

    Args:
        detected_speakers: Set of speaker labels
        ai_suggestions: Dict of AI-suggested names
        args: Arguments namespace

    Returns:
        Final speaker mapping dict
    """
    print("\n=== Speaker Mapping (AI-Assisted) ===", file=sys.stderr)

    speaker_map = {}

    for speaker in sorted(detected_speakers):
        # Get AI suggestion
        suggestion = ai_suggestions.get(speaker, "Unknown")

        # Prompt with suggestion as default
        prompt_text = f"Speaker {speaker} [{suggestion}]: "
        user_input = input(prompt_text).strip()

        if user_input:
            # User override
            speaker_map[speaker] = user_input
            log_debug(args, f"User override: {speaker} → {user_input}")
        else:
            # Accept AI suggestion
            if suggestion != "Unknown":
                speaker_map[speaker] = suggestion
                log_debug(args, f"Accepted AI suggestion: {speaker} → {suggestion}")
            else:
                log_warning(args, f"No name provided for {speaker}, keeping original")

    return speaker_map


# ----------------------------------------------------------------------
# Speaker Mapping Parsers
# ----------------------------------------------------------------------

def parse_speaker_map_inline(names_str, detected_speakers):
    """
    Parse comma-separated speaker names.

    Args:
        names_str: Comma-separated names (e.g., "Alice,Bob,Charlie")
        detected_speakers: Set of detected speaker labels

    Returns:
        Dict mapping sorted speakers to names
    """
    names = [n.strip() for n in names_str.split(',') if n.strip()]
    speakers_sorted = sorted(detected_speakers)

    speaker_map = {}
    for i, speaker in enumerate(speakers_sorted):
        if i < len(names):
            speaker_map[speaker] = names[i]

    return speaker_map


def parse_speaker_map_file(filepath, detected_speakers):
    """
    Parse speaker mapping file with auto-format detection.

    Supports 4 formats:
    1. Sequential: "Alice\\nBob" → A→Alice, B→Bob (sorted)
    2. Key:value: "A: Alice\\nB: Bob"
    3. Full labels: "Speaker A: Alice\\nSpeaker B: Bob"
    4. Mixed: Combination of formats 2 and 3

    Args:
        filepath: Path to mapping file
        detected_speakers: Set of detected speaker labels (used for sequential format)

    Returns:
        Dict mapping speaker labels to names
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not lines:
        return {}

    # Detect format: does first line contain ':'?
    if ':' in lines[0]:
        # Key:value format (Formats 2, 3, 4)
        speaker_map = {}
        for line in lines:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            speaker_map[key] = value
        return speaker_map

    else:
        # Sequential format (Format 1)
        speakers_sorted = sorted(detected_speakers)
        speaker_map = {}
        for i, name in enumerate(lines):
            if i < len(speakers_sorted):
                speaker_map[speakers_sorted[i]] = name.strip()
        return speaker_map


def prompt_interactive_mapping(detected_speakers, args):
    """
    Interactively prompt user for speaker names.

    Args:
        detected_speakers: Set of detected speaker labels
        args: Argument namespace (for logging)

    Returns:
        Dict mapping speaker labels to user-provided names
    """
    print("\n=== Detected Speakers ===", file=sys.stderr)
    speaker_map = {}

    for speaker in sorted(detected_speakers):
        prompt_text = f"Name for '{speaker}' (press Enter to keep): "
        name = input(prompt_text).strip()
        if name:
            speaker_map[speaker] = name
            log_debug(args, f"Mapped: {speaker} → {name}")

    return speaker_map


# ----------------------------------------------------------------------
# Output Generation
# ----------------------------------------------------------------------

def generate_txt_from_json(json_obj):
    """
    Generate formatted transcript TXT from JSON.

    Format: "{speaker}:\\t{text}\\n" (tab after colon)

    Args:
        json_obj: JSON object (potentially with mapped speaker names)

    Returns:
        Formatted transcript string
    """
    segments = find_transcript_segments(json_obj)

    if not segments:
        return ""

    lines = []
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '')
        lines.append(f"{speaker}:\t{text}")

    return '\n'.join(lines) + '\n'


def generate_output_path(input_path, extension=''):
    """
    Generate output path by inserting '.mapped' before final extension.

    Examples:
        audio.mp3.assemblyai.json → audio.mp3.assemblyai.mapped
        file.json → file.mapped

    Args:
        input_path: Input file path
        extension: Extension to add (e.g., '.json', '.txt')

    Returns:
        Output path with '.mapped' inserted
    """
    # Split into base and extension
    if input_path.endswith('.json'):
        base = input_path[:-5]  # Remove .json
        return f"{base}.mapped{extension}"
    else:
        # Generic fallback
        base, ext = os.path.splitext(input_path)
        return f"{base}.mapped{extension}"


# ----------------------------------------------------------------------
# Validation and Logging
# ----------------------------------------------------------------------

def validate_and_log_mapping(speaker_map, detected_speakers, args):
    """
    Validate speaker mapping and log coverage information.

    Args:
        speaker_map: Dict of speaker label → name mappings
        detected_speakers: Set of detected speaker labels
        args: Argument namespace (for logging)
    """
    mapped = set(speaker_map.keys())
    detected = set(detected_speakers)

    unmapped = detected - mapped
    extra = mapped - detected

    if unmapped:
        log_warning(args, f"Unmapped speakers (keeping original): {', '.join(sorted(unmapped))}")

    if extra:
        log_warning(args, f"Extra mappings for non-existent speakers: {', '.join(sorted(extra))}")

    log_info(args, f"Detected {len(detected)} speaker(s): {', '.join(sorted(detected))}")

    if speaker_map:
        log_info(args, "Applied mappings:")
        for speaker in sorted(speaker_map.keys()):
            log_info(args, f"  {speaker} → {speaker_map[speaker]}")


# ----------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------

def write_json(filepath, data, args):
    """Write JSON data to file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    log_info(args, f"Wrote JSON: {filepath}")


def write_txt(filepath, content, args):
    """Write text content to file."""
    with open(filepath, 'w') as f:
        f.write(content)
    log_info(args, f"Wrote TXT: {filepath}")


# ----------------------------------------------------------------------
# Argument Parsing
# ----------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replace speaker labels with names in AssemblyAI JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect speakers in JSON
  %(prog)s --detect audio.assemblyai.json

  # LLM-assisted speaker detection (auto)
  %(prog)s --llm-detect openai/gpt-4o-mini audio.assemblyai.json
  %(prog)s --llm-detect ollama/llama3.2 audio.assemblyai.json
  %(prog)s --llm-detect anthropic/claude-3-5-haiku audio.assemblyai.json

  # LLM-assisted interactive (AI suggestions as defaults)
  %(prog)s --llm-interactive openai/gpt-4o-mini audio.assemblyai.json

  # Map via comma-separated names (sorted order)
  %(prog)s -m "Alice,Bob,Charlie" audio.assemblyai.json

  # Map via file (auto-detects format)
  %(prog)s -M speakers.txt audio.assemblyai.json

  # Interactive mapping (manual)
  %(prog)s --interactive audio.assemblyai.json

  # Verbose output with force overwrite
  %(prog)s -vv -f -m "Host,Guest" interview.json

Mapping File Formats:
  Sequential:    Alice\\nBob\\nCharlie
  Key:value:     A: Alice\\nB: Bob
  Full labels:   Speaker A: Alice\\nSpeaker B: Bob

LLM Providers (requires: pip install instructor openai):
  openai/MODEL        - OpenAI (gpt-4o-mini, gpt-4o, etc.)
  anthropic/MODEL     - Anthropic Claude (claude-3-5-sonnet, etc.)
  google/MODEL        - Google Gemini (gemini-2.0-flash-exp, etc.)
  groq/MODEL          - Groq (llama-3.1-70b-versatile, etc.)
  ollama/MODEL        - Local Ollama (llama3.2, mistral, etc.)
  Custom endpoint:    --llm-detect ollama/llama3.2 --llm-endpoint http://server:11434
        """
    )

    # Positional argument
    parser.add_argument(
        'input_json',
        help='Path to AssemblyAI JSON file (e.g., audio.assemblyai.json)'
    )

    # Mapping sources (mutually exclusive)
    mapping_group = parser.add_mutually_exclusive_group()
    mapping_group.add_argument(
        '-m', '--speaker-map',
        help='Comma-separated speaker names (e.g., "Alice,Bob,Charlie")'
    )
    mapping_group.add_argument(
        '-M', '--speaker-map-file',
        help='Path to file with speaker mappings (auto-detects format)'
    )
    mapping_group.add_argument(
        '--interactive',
        action='store_true',
        help='Interactively prompt for speaker names'
    )
    mapping_group.add_argument(
        '--llm-detect',
        metavar='PROVIDER/MODEL',
        help='Automatically detect speaker names using LLM '
             '(e.g., "openai/gpt-4o-mini", "ollama/llama3.2", "anthropic/claude-3-5-haiku")'
    )
    mapping_group.add_argument(
        '--llm-interactive',
        metavar='PROVIDER/MODEL',
        help='Interactive mode with AI-suggested speaker names as defaults'
    )
    mapping_group.add_argument(
        '--llm-detect-fallback',
        metavar='PROVIDER/MODEL',
        help='Try LLM detection, fall back to manual interactive if it fails'
    )

    # LLM configuration (optional)
    parser.add_argument(
        '--llm-endpoint',
        metavar='URL',
        help='Custom LLM endpoint URL (for remote Ollama or custom servers)'
    )
    parser.add_argument(
        '--llm-sample-size',
        type=int,
        default=20,
        metavar='N',
        help='Number of utterances to send to LLM for analysis (default: 20)'
    )

    # Output control
    parser.add_argument(
        '-o', '--output',
        help='Output base name (default: auto-generate with .mapped)'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--txt-only',
        action='store_true',
        help='Generate only .txt file (skip .json)'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Generate only .json file (skip .txt)'
    )
    parser.add_argument(
        '--detect',
        action='store_true',
        help='Only show detected speakers and exit (no processing)'
    )

    # Logging
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (-v=INFO, -vvvvv=DEBUG)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all non-error output'
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    """Main execution flow."""
    args = parse_args()

    # Load input JSON
    try:
        with open(args.input_json, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        log_error(args, f"File not found: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_error(args, f"Invalid JSON: {e}")
        sys.exit(1)

    log_debug(args, f"Loaded JSON from: {args.input_json}")

    # Detect speakers
    detected_speakers = detect_speakers_in_json(json_data)

    if not detected_speakers:
        log_error(args, "No speakers detected in JSON (no 'speaker' keys found)")
        sys.exit(1)

    log_debug(args, f"Detected speakers: {detected_speakers}")

    # Detect-only mode
    if args.detect:
        print(f"Detected speakers: {', '.join(sorted(detected_speakers))}")
        return

    # Build speaker map
    if args.llm_detect or args.llm_interactive or args.llm_detect_fallback:
        # LLM-assisted detection
        speaker_map = handle_llm_detection(args, json_data, detected_speakers)
    elif args.speaker_map:
        speaker_map = parse_speaker_map_inline(args.speaker_map, detected_speakers)
    elif args.speaker_map_file:
        try:
            speaker_map = parse_speaker_map_file(args.speaker_map_file, detected_speakers)
        except FileNotFoundError:
            log_error(args, f"Mapping file not found: {args.speaker_map_file}")
            sys.exit(1)
    elif args.interactive:
        speaker_map = prompt_interactive_mapping(detected_speakers, args)
    else:
        log_error(args, "No mapping source provided (use -m, -M, --interactive, or --llm-detect)")
        sys.exit(1)

    if not speaker_map:
        log_warning(args, "Empty speaker mapping - no changes will be made")

    # Validate and log mapping
    validate_and_log_mapping(speaker_map, detected_speakers, args)

    # Apply mapping
    log_debug(args, "Applying speaker mapping to JSON...")
    mapped_json = replace_speakers_recursive(json_data, speaker_map)

    # Determine output paths
    if args.output:
        output_base = args.output
    else:
        output_base = generate_output_path(args.input_json, extension='')

    json_output = f"{output_base}.json"
    txt_output = f"{output_base}.txt"

    # Check for existing files
    if not args.force:
        existing = []
        if not args.txt_only and os.path.exists(json_output):
            existing.append(json_output)
        if not args.json_only and os.path.exists(txt_output):
            existing.append(txt_output)

        if existing:
            log_error(args, f"Output file(s) already exist: {', '.join(existing)}")
            log_error(args, "Use -f/--force to overwrite")
            sys.exit(1)

    # Write outputs
    if not args.txt_only:
        write_json(json_output, mapped_json, args)

    if not args.json_only:
        txt_content = generate_txt_from_json(mapped_json)
        if txt_content:
            write_txt(txt_output, txt_content, args)
        else:
            log_warning(args, "No transcript segments found in JSON - TXT file not created")

    # Summary
    if not args.quiet:
        outputs = []
        if not args.txt_only:
            outputs.append(json_output)
        if not args.json_only and txt_content:
            outputs.append(txt_output)
        print(f"Created: {', '.join(outputs)}")


if __name__ == "__main__":
    main()
