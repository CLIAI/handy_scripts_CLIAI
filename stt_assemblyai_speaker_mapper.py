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

Features:
- LLM-assisted speaker detection with multiple provider support
- Interactive mapping with AI suggestions as defaults
- Audio preview: hear samples of each speaker during mapping
- Verification mode: review and correct existing mappings with audio
- Speaker audio extraction: save speaker segments to separate files

Usage:
    # Detect speakers
    ./stt_assemblyai_speaker_mapper.py --detect audio.assemblyai.json

    # LLM-assisted interactive (AI suggestions + audio preview)
    ./stt_assemblyai_speaker_mapper.py --llm-interactive gpt-4o-mini audio.assemblyai.json

    # Preview audio samples for a speaker
    ./stt_assemblyai_speaker_mapper.py --preview-speaker A audio.assemblyai.json

    # Verify/review existing mappings with audio
    ./stt_assemblyai_speaker_mapper.py --verify audio.assemblyai.mapped.json

    # Extract speaker audio to file
    ./stt_assemblyai_speaker_mapper.py --extract-speaker A -o speaker_a.mp3 audio.assemblyai.json

    # Map via inline comma-separated names
    ./stt_assemblyai_speaker_mapper.py -m "Alice,Bob" audio.assemblyai.json

    # Interactive mapping (manual)
    ./stt_assemblyai_speaker_mapper.py --interactive audio.assemblyai.json

Requirements for audio features:
    - ffmpeg (for audio extraction)
    - mpv, ffplay, or mplayer (for playback with seeking)
"""

import argparse
import sys
import json
import os
import re
import shutil
import tempfile
import subprocess
from typing import Dict, List, Optional, Union, Tuple

# Optional LLM detection support
try:
    import instructor
    from pydantic import BaseModel, Field, ConfigDict
    from openai import OpenAI
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    instructor = None
    BaseModel = None
    Field = None
    ConfigDict = None
    OpenAI = None

# ----------------------------------------------------------------------
# Model Shortcuts - Map common names to full provider/model strings
# ----------------------------------------------------------------------
MODEL_SHORTCUTS = {
    # OpenAI models (best structured output support)
    '4o-mini': 'openai/gpt-4o-mini',
    'gpt-4o-mini': 'openai/gpt-4o-mini',
    '4o': 'openai/gpt-4o',
    'gpt-4o': 'openai/gpt-4o',
    '4.1': 'openai/gpt-4.1',
    'gpt-4.1': 'openai/gpt-4.1',
    '4.1-mini': 'openai/gpt-4.1-mini',
    'gpt-4.1-mini': 'openai/gpt-4.1-mini',
    '4.1-nano': 'openai/gpt-4.1-nano',
    'gpt-4.1-nano': 'openai/gpt-4.1-nano',
    'o1': 'openai/o1',
    'o3-mini': 'openai/o3-mini',

    # Anthropic Claude (best accuracy for speaker detection)
    'sonnet': 'anthropic/claude-sonnet-4-5',
    'claude-sonnet': 'anthropic/claude-sonnet-4-5',
    'sonnet-4-5': 'anthropic/claude-sonnet-4-5',
    'claude-sonnet-4-5': 'anthropic/claude-sonnet-4-5',
    'opus': 'anthropic/claude-opus-4-1',
    'claude-opus': 'anthropic/claude-opus-4-1',
    'opus-4-1': 'anthropic/claude-opus-4-1',
    'claude-opus-4-1': 'anthropic/claude-opus-4-1',
    'haiku': 'anthropic/claude-3-5-haiku',
    'claude-haiku': 'anthropic/claude-3-5-haiku',
    'haiku-3-5': 'anthropic/claude-3-5-haiku',
    'claude-3-5-haiku': 'anthropic/claude-3-5-haiku',
    'sonnet-3-7': 'anthropic/claude-3-7-sonnet',
    'claude-3-7-sonnet': 'anthropic/claude-3-7-sonnet',

    # Google Gemini (cost leader)
    'gemini': 'google/gemini-2.5-flash',
    'gemini-flash': 'google/gemini-2.5-flash',
    'gemini-2.5': 'google/gemini-2.5-flash',
    'gemini-2.5-flash': 'google/gemini-2.5-flash',
    'gemini-2.0': 'google/gemini-2.0-flash',
    'gemini-2.0-flash': 'google/gemini-2.0-flash',
    'gemini-pro': 'google/gemini-2.0-pro-experimental',
    'gemini-2.0-pro': 'google/gemini-2.0-pro-experimental',

    # Groq (ultra-fast inference)
    'llama': 'groq/llama-3.3-70b-versatile',
    'llama3.3': 'groq/llama-3.3-70b-versatile',
    'llama-3.3': 'groq/llama-3.3-70b-versatile',
    'llama3.2': 'groq/llama-3.2-3b-preview',
    'llama-3.2': 'groq/llama-3.2-3b-preview',
    'llama3.1': 'groq/llama-3.1-8b-instant',
    'llama-3.1': 'groq/llama-3.1-8b-instant',

    # DeepSeek (cost effective with caching)
    'deepseek': 'deepseek/deepseek-v3.2-exp',
    'deepseek-v3': 'deepseek/deepseek-v3.2-exp',
    'deepseek-v3.2': 'deepseek/deepseek-v3.2-exp',
    'deepseek-r1': 'deepseek/deepseek-r1',

    # Mistral
    'mistral': 'mistral/mistral-large-latest',
    'mistral-large': 'mistral/mistral-large-latest',
    'mistral-medium': 'mistral/mistral-medium-3',
    'mistral-small': 'mistral/mistral-small',
    'codestral': 'mistral/codestral',

    # Ollama (local deployment) - Small CPU-optimized models
    'ollama': 'ollama/llama3.2',
    'ollama-llama': 'ollama/llama3.2',
    'ollama-mistral': 'ollama/mistral',

    # SmolLM2 series (best small models for CPU, 16GB RAM friendly)
    'smollm2': 'ollama/smollm2:1.7b',
    'smollm2:1.7b': 'ollama/smollm2:1.7b',
    'smollm2-1.7b': 'ollama/smollm2:1.7b',
    'smollm2:360m': 'ollama/smollm2:360m',
    'smollm2-360m': 'ollama/smollm2:360m',
    'smollm2:135m': 'ollama/smollm2:135m',
    'smollm2-135m': 'ollama/smollm2:135m',

    # Qwen2.5 small variants (excellent small coders)
    'qwen2.5:0.5b': 'ollama/qwen2.5:0.5b',
    'qwen2.5-0.5b': 'ollama/qwen2.5:0.5b',
    'qwen2.5:1.5b': 'ollama/qwen2.5:1.5b',
    'qwen2.5-1.5b': 'ollama/qwen2.5:1.5b',
    'qwen2.5:3b': 'ollama/qwen2.5:3b',
    'qwen2.5-3b': 'ollama/qwen2.5:3b',
    'qwen2.5-coder:0.5b': 'ollama/qwen2.5-coder:0.5b',
    'qwen2.5-coder-0.5b': 'ollama/qwen2.5-coder:0.5b',
    'qwen2.5-coder:1.5b': 'ollama/qwen2.5-coder:1.5b',
    'qwen2.5-coder-1.5b': 'ollama/qwen2.5-coder:1.5b',
    'qwen2.5-coder:3b': 'ollama/qwen2.5-coder:3b',
    'qwen2.5-coder-3b': 'ollama/qwen2.5-coder:3b',

    # Llama 3.2 small variants (fast on CPU)
    'llama3.2:1b': 'ollama/llama3.2:1b',
    'llama3.2-1b': 'ollama/llama3.2:1b',
    'llama3.2:3b': 'ollama/llama3.2:3b',
    'llama3.2-3b': 'ollama/llama3.2:3b',

    # Phi models (punches above weight)
    'phi3': 'ollama/phi3:mini',
    'phi3:mini': 'ollama/phi3:mini',
    'phi3-mini': 'ollama/phi3:mini',
    'phi3:3.8b': 'ollama/phi3:3.8b',
    'phi4': 'ollama/phi4:14b',

    # DeepSeek Coder (coding specialist)
    'deepseek-coder': 'ollama/deepseek-coder:1.3b',
    'deepseek-coder:1.3b': 'ollama/deepseek-coder:1.3b',
    'deepseek-coder-1.3b': 'ollama/deepseek-coder:1.3b',

    # Other popular small models
    'tinyllama': 'ollama/tinyllama:1.1b',
    'tinyllama:1.1b': 'ollama/tinyllama:1.1b',
    'stablelm-zephyr': 'ollama/stablelm-zephyr:3b',
    'stablelm-zephyr:3b': 'ollama/stablelm-zephyr:3b',
    'orca-mini': 'ollama/orca-mini:3b',
    'orca-mini:3b': 'ollama/orca-mini:3b',
}


def resolve_model_shortcut(model_string: str) -> str:
    """
    Resolve model shortcut to full provider/model string.

    Args:
        model_string: Model name or shortcut (e.g., "4o-mini", "sonnet", "openai/gpt-4o")

    Returns:
        Full provider/model string (e.g., "openai/gpt-4o-mini")

    Examples:
        >>> resolve_model_shortcut("4o-mini")
        'openai/gpt-4o-mini'
        >>> resolve_model_shortcut("sonnet")
        'anthropic/claude-sonnet-4-5'
        >>> resolve_model_shortcut("openai/gpt-4o")
        'openai/gpt-4o'
    """
    # If already in provider/model format, return as-is
    if '/' in model_string:
        return model_string

    # Case-insensitive lookup
    model_lower = model_string.lower()
    return MODEL_SHORTCUTS.get(model_lower, model_string)


# ----------------------------------------------------------------------
# META Message Helper (for transcript warnings)
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
# About File Helper
# ----------------------------------------------------------------------

def get_about_file_path(input_json: str) -> str:
    """
    Generate about file path from input JSON path.

    Examples:
        audio.mp3.assemblyai.json → audio.mp3.about.md

    Args:
        input_json: Path to input JSON file

    Returns:
        Path to about file
    """
    if input_json.endswith('.assemblyai.json'):
        base_audio = input_json[:-len('.assemblyai.json')]
    else:
        base_audio = input_json
    return f"{base_audio}.about.md"


def get_about_file_content(input_json: str) -> Optional[str]:
    """
    Load .about.md file content if it exists.

    About files provide context about the audio (speaker names, roles, topics)
    to help improve LLM speaker detection accuracy.

    Args:
        input_json: Path to input JSON file

    Returns:
        Content of about file, or None if file doesn't exist
    """
    about_path = get_about_file_path(input_json)

    if os.path.exists(about_path):
        try:
            with open(about_path, 'r') as f:
                return f.read().strip()
        except Exception:
            return None
    return None


# Directory context filename (searched in parent directories)
DIRECTORY_CONTEXT_FILENAME = "SPEAKER.CONTEXT.md"


def find_directory_context_file(input_json: str) -> Optional[str]:
    """
    Find SPEAKER.CONTEXT.md in same directory or parent directories.

    Searches both original path and resolved path (via realpath).
    Similar to how .gitignore or .editorconfig files work.

    Args:
        input_json: Path to input JSON file

    Returns:
        Path to found context file, or None if not found
    """
    # Get base audio path
    if input_json.endswith('.assemblyai.json'):
        base_audio = input_json[:-len('.assemblyai.json')]
    else:
        base_audio = input_json

    original_dir = os.path.dirname(os.path.abspath(base_audio)) or '.'
    resolved_dir = os.path.dirname(os.path.realpath(base_audio)) or '.'

    # Walk up both paths, collect unique directories
    dirs_to_check = []
    for start_dir in [original_dir, resolved_dir]:
        current = start_dir
        while current:
            if current not in dirs_to_check:
                dirs_to_check.append(current)
            parent = os.path.dirname(current)
            if parent == current:  # Reached root
                break
            current = parent

    # Return first found
    for dir_path in dirs_to_check:
        context_path = os.path.join(dir_path, DIRECTORY_CONTEXT_FILENAME)
        if os.path.exists(context_path):
            return context_path
    return None


def get_directory_context_content(input_json: str) -> tuple[Optional[str], Optional[str]]:
    """
    Load directory context file content if it exists.

    Args:
        input_json: Path to input JSON file

    Returns:
        Tuple of (content, path) or (None, None) if not found
    """
    context_path = find_directory_context_file(input_json)
    if context_path:
        try:
            with open(context_path, 'r') as f:
                return f.read().strip(), context_path
        except Exception:
            return None, None
    return None, None


# ----------------------------------------------------------------------
# Audio Preview Functions
# ----------------------------------------------------------------------

def find_audio_player() -> Tuple[Optional[str], List[str]]:
    """
    Find available audio player with seeking support.

    Checks for players in order of preference:
    1. mpv - Best choice, excellent seeking, terminal-friendly
    2. ffplay - Good fallback, comes with ffmpeg
    3. mplayer - Older but capable

    Returns:
        Tuple of (player_name, base_command_args) or (None, []) if none found
    """
    players = [
        ('mpv', ['mpv', '--no-video', '--term-osd-bar']),
        ('ffplay', ['ffplay', '-nodisp', '-autoexit']),
        ('mplayer', ['mplayer', '-vo', 'null']),
    ]

    for name, cmd in players:
        if shutil.which(cmd[0]):
            return name, cmd

    return None, []


def find_ffmpeg() -> Optional[str]:
    """Check if ffmpeg is available."""
    return shutil.which('ffmpeg')


def get_audio_file_path(input_json: str) -> str:
    """
    Derive audio file path from JSON path.

    Examples:
        audio.mp3.assemblyai.json → audio.mp3
        audio.mp3.assemblyai.mapped.json → audio.mp3

    Args:
        input_json: Path to JSON file

    Returns:
        Path to original audio file
    """
    path = input_json
    # Remove known suffixes
    for suffix in ['.assemblyai.mapped.json', '.assemblyai.json', '.mapped.json']:
        if path.endswith(suffix):
            return path[:-len(suffix)]
    # Fallback: just remove .json
    if path.endswith('.json'):
        return path[:-5]
    return path


def get_speaker_utterances(
    json_data: dict,
    speaker_label: str
) -> List[dict]:
    """
    Get all utterances for a specific speaker with timing info.

    Args:
        json_data: Full AssemblyAI JSON data
        speaker_label: Speaker label to filter (e.g., 'A', 'B')

    Returns:
        List of utterance dicts with 'start', 'end', 'text' keys
    """
    utterances = json_data.get('utterances', [])
    return [u for u in utterances if u.get('speaker') == speaker_label]


def format_duration(ms: int) -> str:
    """Format milliseconds as human-readable duration."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def extract_speaker_audio(
    audio_file: str,
    utterances: List[dict],
    output_file: str,
    max_samples: int = 10,
    max_duration_per_sample: float = 8.0,
    silence_gap: float = 0.3,
    args=None
) -> Tuple[bool, str]:
    """
    Extract and concatenate speaker audio samples using ffmpeg.

    Uses ffmpeg's filter_complex for efficient single-pass extraction.
    Adds short silence between samples for clarity.

    Args:
        audio_file: Path to source audio file
        utterances: List of utterance dicts with 'start' and 'end' (in ms)
        output_file: Path to output concatenated audio
        max_samples: Maximum number of samples to extract
        max_duration_per_sample: Max duration per sample in seconds
        silence_gap: Silence duration between samples in seconds
        args: Arguments namespace for logging

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not utterances:
        return False, "No utterances found for speaker"

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg not found. Install with: sudo pacman -S ffmpeg"

    if not os.path.exists(audio_file):
        return False, f"Audio file not found: {audio_file}"

    # Select samples (first N, capped by max_duration_per_sample)
    selected = []
    for utt in utterances[:max_samples]:
        start_ms = utt.get('start', 0)
        end_ms = utt.get('end', 0)
        duration_s = (end_ms - start_ms) / 1000

        # Cap duration
        if duration_s > max_duration_per_sample:
            end_ms = start_ms + int(max_duration_per_sample * 1000)

        selected.append({
            'start': start_ms / 1000,  # Convert to seconds
            'end': end_ms / 1000,
            'text': utt.get('text', '')[:50]  # Preview text
        })

    if not selected:
        return False, "No samples selected"

    # Build ffmpeg filter_complex
    # Format: [0]atrim=start=X:end=Y,asetpts=PTS-STARTPTS[aN];...
    # Then: [a1][silence][a2][silence]...concat
    filter_parts = []
    concat_inputs = []

    for i, sample in enumerate(selected):
        label = f"a{i}"
        filter_parts.append(
            f"[0]atrim=start={sample['start']:.3f}:end={sample['end']:.3f},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
        concat_inputs.append(f"[{label}]")

        # Add silence between samples (except after last)
        if i < len(selected) - 1 and silence_gap > 0:
            silence_label = f"s{i}"
            # Generate silence using anullsrc
            filter_parts.append(
                f"anullsrc=r=44100:cl=stereo,atrim=0:{silence_gap:.2f}[{silence_label}]"
            )
            concat_inputs.append(f"[{silence_label}]")

    # Concat all segments
    n_segments = len(concat_inputs)
    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={n_segments}:v=0:a=1[out]"
    )

    filter_complex = ';'.join(filter_parts)

    # Build ffmpeg command
    cmd = [
        ffmpeg,
        '-i', audio_file,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-y',  # Overwrite output
        '-loglevel', 'error',
        output_file
    ]

    try:
        log_debug(args, f"Running ffmpeg with {len(selected)} samples")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, f"ffmpeg failed: {error_msg}"

        total_duration = sum(s['end'] - s['start'] for s in selected)
        return True, f"Extracted {len(selected)} samples ({format_duration(int(total_duration * 1000))})"

    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"
    except Exception as e:
        return False, f"ffmpeg error: {e}"


def play_audio_file(
    filepath: str,
    player_name: str = None,
    player_cmd: List[str] = None,
    args=None
) -> bool:
    """
    Play audio file with seeking-capable terminal player.

    Args:
        filepath: Path to audio file to play
        player_name: Name of player (for display)
        player_cmd: Command and args to run player
        args: Arguments namespace for logging

    Returns:
        True if playback completed, False on error
    """
    if player_cmd is None:
        player_name, player_cmd = find_audio_player()

    if not player_cmd:
        log_error(args, "No audio player found. Install mpv: sudo pacman -S mpv")
        return False

    # Build full command
    cmd = player_cmd + [filepath]

    # Show controls hint
    if player_name == 'mpv':
        hint = "mpv: ←/→ seek 5s, ↑/↓ seek 1m, space=pause, q=quit"
    elif player_name == 'ffplay':
        hint = "ffplay: ←/→ seek 10s, space=pause, q=quit"
    else:
        hint = f"{player_name}: use arrow keys to seek, q=quit"

    print(f"→ Playing ({hint})", file=sys.stderr)

    try:
        # Run player, letting it take over terminal
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("", file=sys.stderr)  # Clean line after Ctrl+C
        return True  # User interrupted, not an error
    except Exception as e:
        log_error(args, f"Playback failed: {e}")
        return False


def preview_speaker_audio(
    audio_file: str,
    json_data: dict,
    speaker_label: str,
    speaker_name: str = None,
    max_samples: int = 10,
    args=None
) -> bool:
    """
    High-level function to preview audio samples for a speaker.

    Extracts samples to temp file, plays them, then cleans up.

    Args:
        audio_file: Path to source audio file
        json_data: Full AssemblyAI JSON data
        speaker_label: Speaker label (e.g., 'A', 'B')
        speaker_name: Display name for speaker (optional)
        max_samples: Maximum samples to extract
        args: Arguments namespace

    Returns:
        True if preview completed successfully
    """
    display_name = speaker_name or f"Speaker {speaker_label}"

    # Get utterances for this speaker
    utterances = get_speaker_utterances(json_data, speaker_label)

    if not utterances:
        print(f"No utterances found for {display_name}", file=sys.stderr)
        return False

    # Calculate stats
    total_duration = sum(u.get('end', 0) - u.get('start', 0) for u in utterances)

    print(f"\nExtracting samples for {display_name}...", file=sys.stderr)
    print(f"  Found {len(utterances)} utterances ({format_duration(total_duration)})", file=sys.stderr)

    # Find audio player first
    player_name, player_cmd = find_audio_player()
    if not player_cmd:
        print("ERROR: No audio player found. Install mpv: sudo pacman -S mpv", file=sys.stderr)
        return False

    # Create temp file for extracted audio
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Extract audio samples
        success, message = extract_speaker_audio(
            audio_file,
            utterances,
            tmp_path,
            max_samples=max_samples,
            args=args
        )

        if not success:
            print(f"ERROR: {message}", file=sys.stderr)
            return False

        print(f"  {message}", file=sys.stderr)

        # Play the extracted audio
        return play_audio_file(tmp_path, player_name, player_cmd, args)

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
        context: str = Field(
            description="Brief contextual information about this speaker: topics discussed, role in conversation, keywords, adjectives, or identifying characteristics to help identify them even if name is uncertain",
            default=""
        )

    class SpeakerDetection(BaseModel):
        """Pydantic model for LLM speaker detection response."""
        model_config = ConfigDict(extra='allow')

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
    args=None,
    input_json: Optional[str] = None
):
    """
    Detect speaker names using LLM via Instructor.

    Args:
        provider_model: Provider and model string or shortcut (e.g., "4o-mini", "sonnet", "openai/gpt-4o-mini")
        transcript_sample: Sample of transcript text
        detected_labels: List of detected speaker labels (e.g., ['A', 'B'])
        endpoint: Optional custom endpoint URL
        args: Arguments namespace (for logging)
        input_json: Path to input JSON file (for loading .about.md context)

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

    # Resolve model shortcuts to full provider/model strings
    original_model = provider_model
    provider_model = resolve_model_shortcut(provider_model)

    if original_model != provider_model:
        log_debug(args, f"Resolved shortcut '{original_model}' → '{provider_model}'")

    log_debug(args, f"LLM provider: {provider_model}")
    log_debug(args, f"Detected labels: {detected_labels}")

    # Build prompt
    prompt = f"""Analyze this conversation transcript and identify the speakers.

DETECTED SPEAKERS: {', '.join(detected_labels)}

Your task is to create a mapping of each detected speaker label to their actual name or professional role, along with contextual information to help identify them.

CRITICAL WARNING - Avoid Address Confusion:
When someone says a name in their utterance, they are usually ADDRESSING that person, NOT identifying themselves.
- "Alice, what do you think?" → Speaker is NOT Alice, they are talking TO Alice
- "Bob, I agree with you" → Speaker is NOT Bob, they are responding TO Bob
- "Thanks John for joining us" → Speaker is NOT John, they are welcoming John
Pay careful attention to WHO is being addressed vs WHO is speaking. The name mentioned is typically the listener, not the speaker.

Look for:
- Direct name mentions - but remember: the name mentioned is usually the ADDRESSEE, not the speaker
- Introductions ("I'm...", "My name is...") - these DO identify the speaker
- Self-references using third person ("Alice is happy", "Bob appreciates")
- Professional roles if names aren't mentioned (Host, Guest, Expert, Interviewer)
- Topics they discussed (AI, research, product features, etc.)
- Their role in the conversation (asking questions, explaining, presenting, etc.)
- Keywords, adjectives, or characteristics that identify them
"""

    # Initialize context variables
    dir_context = None
    about_content = None

    # Add directory context if available (STT-IN-BATCH.CONTEXT.md)
    # This is general context for all files in the directory tree
    if input_json:
        dir_context, dir_context_path = get_directory_context_content(input_json)
        if dir_context:
            if not getattr(args, 'quiet', False):
                print(f"Using directory context from: {dir_context_path}", file=sys.stderr)
            log_debug(args, f"Directory context ({len(dir_context)} chars)")
            prompt += f"""
DIRECTORY CONTEXT (applies to all files in this project):
{dir_context}

"""

    # Add file-specific about context if available
    # This is specific to this particular audio file
    if input_json:
        about_content = get_about_file_content(input_json)
        if about_content:
            about_path = get_about_file_path(input_json)
            # Always inform user (not just in verbose mode) since this affects results
            if not getattr(args, 'quiet', False):
                print(f"Using file context from: {about_path}", file=sys.stderr)
            log_debug(args, f"About file content ({len(about_content)} chars)")
            prompt += f"""
FILE-SPECIFIC CONTEXT (for this audio file):
{about_content}

"""

    # Add instruction if any context was provided
    if dir_context or about_content:
        prompt += """Use the above context to help identify speakers. The context may contain speaker names, roles, topics discussed, or other identifying information.
"""

    prompt += f"""
TRANSCRIPT SAMPLE:
{transcript_sample}

You must provide a mapping for EACH detected speaker label ({', '.join(detected_labels)}) including:
1. speaker_label: The label (A, B, C, etc.)
2. speaker_name: Their identified name or role (use "Unknown" if uncertain)
3. context: Brief contextual info about this speaker - topics discussed, role, keywords, or identifying characteristics (even if name is uncertain)

Example output format for speakers ["A", "B"]:
{{
  "speakers": [
    {{"speaker_label": "A", "speaker_name": "Alice Anderson", "context": "Host, asked questions about AI ethics and neural networks"}},
    {{"speaker_label": "B", "speaker_name": "Unknown", "context": "Guest expert, discussed research background and transformer architectures"}}
  ],
  "confidence": "medium",
  "reasoning": "Speaker A identified by introduction, Speaker B's name not mentioned"
}}

The context field should help identify the speaker even if the name is wrong or unknown.
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

        # Extract speaker contexts before converting to dict
        speaker_contexts = {mapping.speaker_label: mapping.context for mapping in result.speakers}

        # Convert List[SpeakerMapping] back to Dict[str, str] for compatibility
        result.speakers = {mapping.speaker_label: mapping.speaker_name for mapping in result.speakers}

        # Store contexts as a separate attribute for interactive mode
        result.speaker_contexts = speaker_contexts

        return result

    except Exception as e:
        log_error(args, f"LLM detection failed: {e}")
        raise


def handle_llm_detection(args, json_data, detected_speakers):
    """
    Handle LLM-assisted speaker detection.

    Supports auto-loading from cached suggestions file for faster workflow.

    Args:
        args: Parsed arguments
        json_data: Full transcript JSON
        detected_speakers: Set of detected speaker labels

    Returns:
        Speaker mapping dict
    """
    # Check for cached suggestions file (unless --force)
    suggestions_path = get_suggestions_file_path(args.input_json)

    if args.llm_interactive and os.path.exists(suggestions_path) and not args.force:
        # Auto-load cached suggestions
        try:
            log_info(args, "Found cached suggestions file, loading...")
            _, ai_suggestions, metadata = load_suggestions_from_file(suggestions_path, args)

            log_info(args, "Using cached LLM suggestions (use --force to regenerate)")

            # Go directly to interactive review
            # Note: metadata might not have contexts for older cached files
            speaker_contexts = metadata.get('speaker_contexts', {})
            return prompt_interactive_with_suggestions(
                detected_speakers,
                ai_suggestions,
                speaker_contexts,
                args.input_json,
                args,
                json_data=json_data
            )
        except (FileNotFoundError, ValueError) as e:
            log_warning(args, f"Failed to load suggestions file: {e}")
            log_warning(args, "Falling back to LLM generation")
            # Continue to LLM generation below

    # Determine provider spec
    provider_spec = args.llm_detect or args.llm_interactive or args.llm_detect_fallback or args.generate_suggestions_only

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
            args=args,
            input_json=args.input_json
        )

        log_info(args, f"LLM confidence: {detection_result.confidence}")
        if detection_result.reasoning:
            log_info(args, f"LLM reasoning: {detection_result.reasoning}")

        # Save suggestions to file for future use
        # (unless running in --generate-suggestions-only mode, which saves in main())
        if not args.generate_suggestions_only:
            try:
                save_suggestions_to_file(
                    suggestions_path,
                    detected_speakers,
                    detection_result.speakers,
                    detection_result,
                    provider_spec,
                    args.input_json,
                    args
                )
            except Exception as e:
                log_warning(args, f"Failed to save suggestions file: {e}")
                # Non-fatal, continue with workflow

        # Interactive mode: show suggestions as defaults
        if args.llm_interactive:
            # Get contexts from detection result
            speaker_contexts = getattr(detection_result, 'speaker_contexts', {})
            return prompt_interactive_with_suggestions(
                detected_speakers,
                detection_result.speakers,
                speaker_contexts,
                args.input_json,
                args,
                json_data=json_data
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


def expand_command_placeholders(command: str, input_json: str) -> str:
    """
    Expand placeholders in command with actual file paths.

    Args:
        command: Command string with placeholders like {audio}, {text}, etc.
        input_json: Path to input JSON file (e.g., audio.mp3.assemblyai.json)

    Returns:
        Command with placeholders replaced by quoted file paths

    Supported placeholders:
        {audio} {a}           - Original audio file (audio.mp3)
        {text} {transcript} {t} {txt} - Base transcript (audio.mp3.txt)
        {json} {j}            - Base JSON (audio.mp3.assemblyai.json)
        {mapped-json} {mj}    - Mapped JSON (audio.mp3.assemblyai.mapped.json)
        {mapped-text} {mapped-txt} {mt} - Mapped text (audio.mp3.assemblyai.mapped.txt)
        {suggestions} {suggestions-json} {sj} - Suggestions (audio.mp3.assemblyai.mapping-suggestions.json)
    """
    import shlex
    import os

    # Derive audio file path from JSON path
    # audio.mp3.assemblyai.json → audio.mp3
    if input_json.endswith('.assemblyai.json'):
        base_audio = input_json[:-len('.assemblyai.json')]
    else:
        # Fallback: use input_json as base
        base_audio = input_json

    # Build file paths
    files = {
        # Audio
        '{audio}': base_audio,
        '{a}': base_audio,

        # Text transcript
        '{text}': f'{base_audio}.txt',
        '{transcript}': f'{base_audio}.txt',
        '{t}': f'{base_audio}.txt',
        '{txt}': f'{base_audio}.txt',

        # Base JSON
        '{json}': input_json,
        '{j}': input_json,

        # Mapped outputs
        '{mapped-json}': f'{base_audio}.assemblyai.mapped.json',
        '{mj}': f'{base_audio}.assemblyai.mapped.json',
        '{mapped-text}': f'{base_audio}.assemblyai.mapped.txt',
        '{mapped-txt}': f'{base_audio}.assemblyai.mapped.txt',
        '{mt}': f'{base_audio}.assemblyai.mapped.txt',

        # Suggestions
        '{suggestions}': f'{base_audio}.assemblyai.mapping-suggestions.json',
        '{suggestions-json}': f'{base_audio}.assemblyai.mapping-suggestions.json',
        '{sj}': f'{base_audio}.assemblyai.mapping-suggestions.json',

        # About file
        '{about}': f'{base_audio}.about.md',
        '{ab}': f'{base_audio}.about.md',
    }

    # Replace placeholders with quoted paths
    result = command
    for placeholder, filepath in files.items():
        if placeholder in result:
            # Quote the filepath to handle spaces
            quoted = shlex.quote(filepath)
            result = result.replace(placeholder, quoted)

    return result


def execute_command(command: str, input_json: str, args) -> bool:
    """
    Execute a shell command with placeholder expansion.

    Args:
        command: Command to execute (may contain placeholders)
        input_json: Path to input JSON file
        args: Arguments namespace

    Returns:
        True if command executed successfully, False otherwise
    """
    import subprocess
    import termios
    import os

    # Expand placeholders
    expanded = expand_command_placeholders(command, input_json)

    # Show what we're executing
    print(f"\n→ Executing: {expanded}", file=sys.stderr)

    # Save terminal state BEFORE launching subprocess
    fd = sys.stdin.fileno()
    old_attrs = None
    if os.isatty(fd):
        try:
            old_attrs = termios.tcgetattr(fd)
        except Exception:
            pass  # Not a terminal or can't get attributes

    try:
        # Execute command with shell, letting it inherit terminal directly
        # NO stdin/stdout/stderr redirection - subprocess gets full terminal control
        result = subprocess.run(
            expanded,
            shell=True,
            check=False
        )

        if result.returncode != 0:
            # Check if killed by signal (negative return codes)
            if result.returncode < 0:
                signal_num = -result.returncode
                if signal_num == 2:  # SIGINT
                    log_debug(args, "Command interrupted by user (Ctrl+C)")
                else:
                    log_warning(args, f"Command killed by signal {signal_num}")
            else:
                log_warning(args, f"Command exited with code {result.returncode}")
            return False

        return True

    except Exception as e:
        log_error(args, f"Command execution failed: {e}")
        return False

    finally:
        # Clear stdin buffer to remove any pending input
        if sys.stdin.isatty():
            try:
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except Exception:
                pass

        # Restore terminal state
        if old_attrs:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass


def show_command_help():
    """Show available commands and placeholders."""
    help_text = """
=== Interactive Commands ===

Special commands:
  skip              - Abort mapping (can rerun later)
  help              - Show this help message
  play              - Play entire audio file (alias for: !play {audio})
  speak             - Preview audio samples for CURRENT speaker being prompted
  speak A           - Preview audio samples for speaker A (or any label)
  about             - Edit about file with context (opens $EDITOR)
  !<command>        - Execute shell command with placeholders

Placeholders (use in ! commands):
  {audio} {a}       - Original audio file
  {text} {t}        - Base transcript (.txt)
  {json} {j}        - Base JSON (.assemblyai.json)
  {mapped-text} {mt} - Mapped transcript (output)
  {mapped-json} {mj} - Mapped JSON (output)
  {suggestions} {sj} - Speaker suggestions (.mapping-suggestions.json)
  {about} {ab}      - About file with context (.about.md)

Examples:
  speak               - Hear samples of current speaker
  speak B             - Hear samples of speaker B
  !play {audio}       - Play the entire audio file
  !less {text}        - View transcript
  !jq .speakers {json} - Inspect speakers in JSON
  !head -50 {text}    - Show first 50 lines
  !grep "Alice" {text} - Search for keyword
  about               - Edit about file to add speaker context

About File (.about.md):
  Create {audiofile}.about.md with speaker names, roles, and context.
  This context is passed to the LLM for improved speaker detection.

Press Enter to accept AI suggestion, or type a name to override.
"""
    print(help_text, file=sys.stderr)


def prompt_interactive_with_suggestions(
    detected_speakers: set,
    ai_suggestions: dict,
    speaker_contexts: dict,
    input_json: str,
    args,
    json_data: dict = None
) -> dict:
    """
    Interactive prompts with AI suggestions as defaults.

    Args:
        detected_speakers: Set of speaker labels
        ai_suggestions: Dict of AI-suggested names
        speaker_contexts: Dict of speaker labels to context information
        input_json: Path to input JSON file (for command execution)
        args: Arguments namespace
        json_data: Full JSON data (for audio preview feature)

    Returns:
        Final speaker mapping dict, or None if user chooses to skip
    """
    # Derive audio file path for speak command
    audio_file = get_audio_file_path(input_json)

    # First, show ALL AI-detected mappings upfront for context
    print("\n=== AI-Detected Speaker Mappings ===", file=sys.stderr)
    for speaker in sorted(detected_speakers):
        suggestion = ai_suggestions.get(speaker, "Unknown")
        context = speaker_contexts.get(speaker, "")
        if context:
            print(f"{speaker} => {suggestion} # {context}", file=sys.stderr)
        else:
            print(f"{speaker} => {suggestion}", file=sys.stderr)

    # Then prompt for confirmation/override
    print("\n=== Review and Confirm ===", file=sys.stderr)
    print("  Enter=accept | name=override | skip=abort | speak=hear speaker | help=commands", file=sys.stderr)
    print("  !cmd: run shell commands with {a}udio {t}ext {j}son {mt}apped-text placeholders", file=sys.stderr)
    print("", file=sys.stderr)

    speaker_map = {}

    for speaker in sorted(detected_speakers):
        # Get AI suggestion
        suggestion = ai_suggestions.get(speaker, "Unknown")

        # Prompt with format: "A => [Alice Anderson]: "
        prompt_text = f"{speaker} => [{suggestion}]: "

        while True:  # Loop to allow commands without consuming the prompt
            try:
                user_input = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or Ctrl+D - treat as skip request
                print("\n\nInterrupted - skipping speaker mapping.", file=sys.stderr)
                print("You can rerun this command later to map speakers.", file=sys.stderr)
                return None

            # Check for special commands
            if user_input.lower() == 'skip':
                print("\nSkipping speaker mapping - no files will be created.", file=sys.stderr)
                print("You can rerun this command later to map speakers.", file=sys.stderr)
                return None

            elif user_input.lower() == 'help':
                show_command_help()
                continue  # Re-prompt for same speaker

            elif user_input.lower() == 'play':
                # Built-in alias for !play {audio}
                try:
                    execute_command('play {audio}', input_json, args)
                except KeyboardInterrupt:
                    # User pressed Ctrl+C during command execution - just continue prompting
                    print("", file=sys.stderr)
                continue  # Re-prompt for same speaker

            elif user_input.lower().startswith('speak'):
                # Preview audio samples for a speaker
                if json_data is None:
                    print("ERROR: Audio preview not available (no JSON data)", file=sys.stderr)
                    continue

                # Parse speak command: "speak" or "speak A"
                parts = user_input.split(None, 1)
                if len(parts) > 1:
                    # Specific speaker requested
                    target_speaker = parts[1].strip()
                    if target_speaker not in detected_speakers:
                        print(f"Unknown speaker: {target_speaker}", file=sys.stderr)
                        print(f"Available: {', '.join(sorted(detected_speakers))}", file=sys.stderr)
                        continue
                else:
                    # Default to current speaker being prompted
                    target_speaker = speaker

                # Get suggested name for display
                target_name = ai_suggestions.get(target_speaker, f"Speaker {target_speaker}")

                try:
                    preview_speaker_audio(
                        audio_file,
                        json_data,
                        target_speaker,
                        speaker_name=target_name,
                        args=args
                    )
                except KeyboardInterrupt:
                    print("", file=sys.stderr)
                print("", file=sys.stderr)
                continue  # Re-prompt for same speaker

            elif user_input.lower() == 'about':
                # Open about file in editor for adding speaker context
                about_path = get_about_file_path(input_json)
                editor = os.environ.get('EDITOR', os.environ.get('VISUAL', 'nano'))
                try:
                    import subprocess
                    print(f"\n→ Opening {about_path} in {editor}...", file=sys.stderr)
                    subprocess.run([editor, about_path], check=False)
                    if os.path.exists(about_path):
                        print(f"✓ About file saved: {about_path}", file=sys.stderr)
                    print("", file=sys.stderr)
                except Exception as e:
                    log_error(args, f"Failed to open editor: {e}")
                continue  # Re-prompt for same speaker

            elif user_input.startswith('!'):
                # Execute arbitrary command
                command = user_input[1:].strip()  # Remove '!' prefix
                if command:
                    try:
                        execute_command(command, input_json, args)
                    except KeyboardInterrupt:
                        # User pressed Ctrl+C during command execution - just continue prompting
                        print("", file=sys.stderr)
                else:
                    log_warning(args, "Empty command after '!'")
                continue  # Re-prompt for same speaker

            else:
                # Normal speaker name input
                break

        # Process speaker name (user_input is now a name or empty)
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

def generate_txt_from_json(json_obj, args=None):
    """
    Generate formatted transcript TXT from JSON.

    Format: "{speaker}:\\t{text}\\n" (tab after colon)

    Args:
        json_obj: JSON object (potentially with mapped speaker names)
        args: Optional arguments namespace (for META message control)

    Returns:
        Formatted transcript string
    """
    segments = find_transcript_segments(json_obj)

    if not segments:
        return ""

    lines = []

    # Prepend META message if enabled and args provided
    if args:
        meta_message = get_meta_message(args)
        if meta_message:
            lines.append(meta_message.rstrip('\n'))

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
# Suggestions File I/O
# ----------------------------------------------------------------------

def get_suggestions_file_path(input_json_path):
    """
    Generate suggestions file path from input JSON path.

    Examples:
        audio.mp3.assemblyai.json → audio.mp3.assemblyai.mapping-suggestions.json

    Args:
        input_json_path: Path to input JSON file

    Returns:
        Path to suggestions file
    """
    if input_json_path.endswith('.json'):
        base = input_json_path[:-5]  # Remove .json
        return f"{base}.mapping-suggestions.json"
    else:
        return f"{input_json_path}.mapping-suggestions.json"


def save_suggestions_to_file(
    suggestions_path: str,
    detected_speakers: set,
    speaker_suggestions: dict,
    detection_result,
    model: str,
    input_file: str,
    args
):
    """
    Save LLM speaker suggestions to JSON file for later use.

    Args:
        suggestions_path: Path to save suggestions file
        detected_speakers: Set of detected speaker labels
        speaker_suggestions: Dict of speaker label → name mappings
        detection_result: SpeakerDetection result from LLM
        model: Model string used for detection
        input_file: Original input JSON path
        args: Arguments namespace
    """
    from datetime import datetime, timezone

    suggestions_data = {
        "detected_speakers": sorted(detected_speakers),
        "suggestions": speaker_suggestions,
        "speaker_contexts": getattr(detection_result, 'speaker_contexts', {}),
        "confidence": getattr(detection_result, 'confidence', 'unknown'),
        "reasoning": getattr(detection_result, 'reasoning', ''),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_file": input_file
    }

    with open(suggestions_path, 'w') as f:
        json.dump(suggestions_data, f, indent=2)

    log_info(args, f"Saved suggestions to: {suggestions_path}")


def load_suggestions_from_file(suggestions_path: str, args):
    """
    Load speaker suggestions from JSON file.

    Args:
        suggestions_path: Path to suggestions file
        args: Arguments namespace

    Returns:
        Tuple of (detected_speakers_list, speaker_suggestions_dict, metadata_dict)

    Raises:
        FileNotFoundError: If suggestions file doesn't exist
        ValueError: If suggestions file is invalid
    """
    try:
        with open(suggestions_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Suggestions file not found: {suggestions_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in suggestions file: {e}")

    # Validate required fields
    required_fields = ['detected_speakers', 'suggestions']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Suggestions file missing required field: {field}")

    detected_speakers = data['detected_speakers']
    suggestions = data['suggestions']

    # Metadata
    metadata = {
        'confidence': data.get('confidence', 'unknown'),
        'reasoning': data.get('reasoning', ''),
        'timestamp': data.get('timestamp', ''),
        'model': data.get('model', 'unknown'),
        'input_file': data.get('input_file', ''),
        'speaker_contexts': data.get('speaker_contexts', {})
    }

    log_info(args, f"Loaded suggestions from: {suggestions_path}")
    log_info(args, f"  Model: {metadata['model']}")
    log_info(args, f"  Confidence: {metadata['confidence']}")
    if metadata['reasoning']:
        log_info(args, f"  Reasoning: {metadata['reasoning']}")

    return detected_speakers, suggestions, metadata


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
    """Write JSON data to file with optional META note."""
    # Add META note to JSON if enabled
    meta_message_text = get_meta_message(args).replace("---\nmeta: ", "").replace("\n---\n", "").strip()
    if meta_message_text:
        data_with_meta = {
            "_meta_note": meta_message_text,
            **data
        }
        with open(filepath, 'w') as f:
            json.dump(data_with_meta, f, indent=2)
    else:
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

Model Shortcuts (can use these instead of full provider/model):
  OpenAI:      4o-mini, gpt-4o-mini → openai/gpt-4o-mini (recommended for cost)
               4o, gpt-4o → openai/gpt-4o
               4.1, gpt-4.1 → openai/gpt-4.1
               o1 → openai/o1 (reasoning model)
  Anthropic:   sonnet, claude-sonnet → anthropic/claude-sonnet-4-5 (best accuracy)
               opus, claude-opus → anthropic/claude-opus-4-1
               haiku, claude-haiku → anthropic/claude-3-5-haiku
  Google:      gemini, gemini-flash → google/gemini-2.5-flash (cost leader)
               gemini-pro → google/gemini-2.0-pro-experimental
  Groq:        llama, llama3.3 → groq/llama-3.3-70b-versatile (ultra-fast)
               llama3.2 → groq/llama-3.2-3b-preview
  DeepSeek:    deepseek → deepseek/deepseek-v3.2-exp (cost effective)

Ollama Small Models (CPU-optimized for 16GB+ RAM):
  Best Overall: smollm2, smollm2:1.7b → ollama/smollm2:1.7b (~1.5GB, 20-30 t/s)
  Ultra-Fast:   smollm2:360m → ollama/smollm2:360m (~0.5GB, 40-80 t/s)
                smollm2:135m → ollama/smollm2:135m (~0.3GB, 50-100+ t/s)
  Best Coder:   qwen2.5-coder:1.5b → ollama/qwen2.5-coder:1.5b (~1.2GB)
                qwen2.5-coder:3b → ollama/qwen2.5-coder:3b (~2.5GB)
  Fast Chat:    llama3.2:1b → ollama/llama3.2:1b (~0.9GB, 200-300 t/s)
                llama3.2:3b → ollama/llama3.2:3b (~2.3GB, 15-25 t/s)
  Specialized:  phi3, phi3:mini → ollama/phi3:mini (~2.4GB, punches above weight)
                deepseek-coder:1.3b → ollama/deepseek-coder:1.3b (~1GB, coding)
                tinyllama:1.1b → ollama/tinyllama:1.1b (~0.9GB, IoT-capable)

Shortcut Examples:
  %(prog)s --llm-detect 4o-mini audio.assemblyai.json
  %(prog)s --llm-detect sonnet audio.assemblyai.json
  %(prog)s --llm-detect gemini audio.assemblyai.json
  %(prog)s --llm-detect smollm2:1.7b audio.assemblyai.json       # Best small model
  %(prog)s --llm-detect qwen2.5-coder:1.5b audio.assemblyai.json # Best small coder
  %(prog)s --llm-detect llama3.2:1b audio.assemblyai.json        # Ultra-fast local

Audio Preview Features (requires ffmpeg + mpv/ffplay/mplayer):
  Interactive 'speak' command:
    During --llm-interactive mapping, type 'speak' to hear the current speaker
    or 'speak A' to hear any speaker. Uses arrow keys to seek in playback.

  Preview a speaker:
    %(prog)s --preview-speaker A audio.assemblyai.json
    %(prog)s --preview-speaker B --max-samples 5 audio.assemblyai.json

  Extract speaker audio to file:
    %(prog)s --extract-speaker A audio.assemblyai.json
    %(prog)s --extract-speaker A -o alice_samples.mp3 audio.assemblyai.json

  Verify/review existing mappings:
    %(prog)s --verify audio.assemblyai.mapped.json
    Plays each speaker's audio and prompts to confirm or change names.
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
    mapping_group.add_argument(
        '--generate-suggestions-only',
        metavar='PROVIDER/MODEL',
        help='Generate speaker name suggestions using LLM and save to file, then exit '
             '(no mapping applied). Enables batch pre-computation for later interactive review.'
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
        help='Overwrite existing output files and force regeneration of cached suggestions'
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
    parser.add_argument(
        '--stdout-only',
        action='store_true',
        help='Output speaker mapping as JSON to stdout instead of files (for benchmarking)'
    )

    # Audio preview features
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify/review existing speaker mappings with audio preview. '
             'Plays audio samples for each speaker and allows name corrections.'
    )
    parser.add_argument(
        '--preview-speaker',
        metavar='LABEL',
        help='Preview audio samples for a specific speaker label (e.g., A, B) and exit'
    )
    parser.add_argument(
        '--extract-speaker',
        metavar='LABEL',
        help='Extract audio samples for a speaker to a file (use with -o for output path)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10,
        metavar='N',
        help='Maximum number of audio samples to extract/preview (default: 10)'
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

    # META message control
    parser.add_argument(
        '--no-meta-message', '--disable-meta-message',
        action='store_true',
        dest='no_meta_message',
        help='Disable the META warning message about potential transcription errors (can also set STT_META_MESSAGE_DISABLE=1)'
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Verify Mode
# ----------------------------------------------------------------------

def run_verify_mode(args, json_data: dict, current_mappings: dict = None):
    """
    Interactive verification of speaker mappings with audio preview.

    For each speaker, plays audio samples and asks user to confirm or change the name.

    Args:
        args: Parsed arguments
        json_data: Full AssemblyAI JSON data
        current_mappings: Existing speaker name mappings (from mapped JSON)

    Returns:
        Updated speaker mapping dict, or None if user aborts
    """
    audio_file = get_audio_file_path(args.input_json)

    if not os.path.exists(audio_file):
        log_error(args, f"Audio file not found: {audio_file}")
        return None

    # Get all speakers from JSON
    detected_speakers = detect_speakers_in_json(json_data)

    if not detected_speakers:
        log_error(args, "No speakers found in JSON")
        return None

    # If no current mappings provided, speakers are their own names
    if current_mappings is None:
        current_mappings = {s: s for s in detected_speakers}

    # Build reverse mapping (name -> original label) for display
    # This handles mapped JSON where speakers are already names
    original_labels = {}
    for speaker in detected_speakers:
        # Check if this speaker value is a mapped name or an original label
        if speaker in current_mappings.values():
            # It's a mapped name, find original label
            for label, name in current_mappings.items():
                if name == speaker:
                    original_labels[speaker] = label
                    break
        else:
            original_labels[speaker] = speaker

    print("\n=== Speaker Verification Mode ===", file=sys.stderr)
    print("Review each speaker with audio samples. Confirm or change names.", file=sys.stderr)
    print("Commands: [Enter]=confirm, [name]=change, [r]=replay, [s]=skip speaker, [q]=quit", file=sys.stderr)
    print("", file=sys.stderr)

    updated_map = {}
    speakers_list = sorted(detected_speakers)

    for i, speaker in enumerate(speakers_list, 1):
        original_label = original_labels.get(speaker, speaker)
        current_name = speaker  # In mapped JSON, speaker IS the name

        print(f"\n[{i}/{len(speakers_list)}] {current_name}", file=sys.stderr)
        if original_label != current_name:
            print(f"  (originally: {original_label})", file=sys.stderr)

        # Play audio samples
        try:
            preview_speaker_audio(
                audio_file,
                json_data,
                speaker,  # Use current speaker value (might be name or label)
                speaker_name=current_name,
                max_samples=args.max_samples,
                args=args
            )
        except KeyboardInterrupt:
            print("", file=sys.stderr)

        # Prompt for confirmation/change
        while True:
            try:
                prompt = f"  Confirm [{current_name}]: "
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nAborted.", file=sys.stderr)
                return None

            if user_input.lower() == 'q':
                print("\nQuitting verification.", file=sys.stderr)
                return None

            elif user_input.lower() == 's':
                # Skip this speaker, keep current name
                updated_map[original_label] = current_name
                log_info(args, f"Skipped: {original_label} → {current_name}")
                break

            elif user_input.lower() == 'r':
                # Replay audio
                try:
                    preview_speaker_audio(
                        audio_file,
                        json_data,
                        speaker,
                        speaker_name=current_name,
                        max_samples=args.max_samples,
                        args=args
                    )
                except KeyboardInterrupt:
                    print("", file=sys.stderr)
                continue

            elif user_input == '':
                # Confirm current name
                updated_map[original_label] = current_name
                print(f"  ✓ Confirmed: {current_name}", file=sys.stderr)
                break

            else:
                # User provided new name
                updated_map[original_label] = user_input
                print(f"  ✓ Changed: {current_name} → {user_input}", file=sys.stderr)
                break

    print("\n=== Verification Complete ===", file=sys.stderr)
    print("Updated mappings:", file=sys.stderr)
    for label, name in sorted(updated_map.items()):
        print(f"  {label} → {name}", file=sys.stderr)

    return updated_map


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

    # Preview speaker audio mode
    if args.preview_speaker:
        speaker_label = args.preview_speaker
        if speaker_label not in detected_speakers:
            log_error(args, f"Unknown speaker: {speaker_label}")
            print(f"Available speakers: {', '.join(sorted(detected_speakers))}", file=sys.stderr)
            sys.exit(1)

        audio_file = get_audio_file_path(args.input_json)
        if not os.path.exists(audio_file):
            log_error(args, f"Audio file not found: {audio_file}")
            sys.exit(1)

        success = preview_speaker_audio(
            audio_file,
            json_data,
            speaker_label,
            max_samples=args.max_samples,
            args=args
        )
        sys.exit(0 if success else 1)

    # Extract speaker audio mode
    if args.extract_speaker:
        speaker_label = args.extract_speaker
        if speaker_label not in detected_speakers:
            log_error(args, f"Unknown speaker: {speaker_label}")
            print(f"Available speakers: {', '.join(sorted(detected_speakers))}", file=sys.stderr)
            sys.exit(1)

        audio_file = get_audio_file_path(args.input_json)
        if not os.path.exists(audio_file):
            log_error(args, f"Audio file not found: {audio_file}")
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base = get_audio_file_path(args.input_json)
            output_path = f"{base}.speaker_{speaker_label}.mp3"

        # Check if output exists
        if os.path.exists(output_path) and not args.force:
            log_error(args, f"Output file exists: {output_path}")
            log_error(args, "Use -f/--force to overwrite")
            sys.exit(1)

        # Get utterances and extract
        utterances = get_speaker_utterances(json_data, speaker_label)
        if not utterances:
            log_error(args, f"No utterances found for speaker {speaker_label}")
            sys.exit(1)

        print(f"Extracting audio for speaker {speaker_label}...", file=sys.stderr)
        success, message = extract_speaker_audio(
            audio_file,
            utterances,
            output_path,
            max_samples=args.max_samples,
            args=args
        )

        if success:
            print(f"✓ {message}", file=sys.stderr)
            print(f"Saved to: {output_path}", file=sys.stderr)
            sys.exit(0)
        else:
            log_error(args, message)
            sys.exit(1)

    # Verify mode - review existing speaker mappings with audio preview
    if args.verify:
        updated_map = run_verify_mode(args, json_data)
        if updated_map:
            # Ask if user wants to save changes
            try:
                save_prompt = input("\nSave updated mappings? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nNot saving.", file=sys.stderr)
                return

            if save_prompt == 'y':
                # Apply mappings and save
                mapped_json = replace_speakers_recursive(json_data, updated_map)
                output_base = generate_output_path(args.input_json, extension='')
                json_output = f"{output_base}.json"
                txt_output = f"{output_base}.txt"

                # Check for existing files
                if not args.force:
                    existing = []
                    if os.path.exists(json_output):
                        existing.append(json_output)
                    if os.path.exists(txt_output):
                        existing.append(txt_output)
                    if existing:
                        log_error(args, f"Output file(s) already exist: {', '.join(existing)}")
                        log_error(args, "Use -f/--force to overwrite")
                        return

                write_json(json_output, mapped_json, args)
                txt_content = generate_txt_from_json(mapped_json, args)
                if txt_content:
                    write_txt(txt_output, txt_content, args)

                print(f"✓ Saved: {json_output}, {txt_output}", file=sys.stderr)
            else:
                print("Changes not saved.", file=sys.stderr)
        return

    # Generate-suggestions-only mode
    if args.generate_suggestions_only:
        suggestions_path = get_suggestions_file_path(args.input_json)

        # Check if suggestions already exist (unless --force)
        if os.path.exists(suggestions_path) and not args.force:
            log_info(args, f"Suggestions file already exists: {suggestions_path}")
            log_info(args, "Use --force to regenerate")
            return

        log_info(args, "Generating speaker name suggestions using LLM...")

        # Call LLM to generate suggestions
        try:
            # Extract transcript sample
            transcript_sample = extract_transcript_sample(
                json_data,
                max_utterances=args.llm_sample_size
            )

            if not transcript_sample:
                log_error(args, "No transcript utterances found for LLM analysis")
                sys.exit(1)

            # Call LLM
            provider_spec = args.generate_suggestions_only
            detection_result = detect_speakers_llm(
                provider_spec,
                transcript_sample,
                list(detected_speakers),
                endpoint=args.llm_endpoint,
                args=args,
                input_json=args.input_json
            )

            # Save suggestions to file
            save_suggestions_to_file(
                suggestions_path,
                detected_speakers,
                detection_result.speakers,
                detection_result,
                provider_spec,
                args.input_json,
                args
            )

            log_info(args, f"✓ Suggestions saved to: {suggestions_path}")
            log_info(args, f"Run with --llm-interactive to review and apply mappings")
            return

        except Exception as e:
            log_error(args, f"Failed to generate suggestions: {e}")
            sys.exit(1)

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

    # Check if user skipped mapping (returns None)
    if speaker_map is None:
        log_info(args, "Mapping skipped by user - exiting without creating files")
        return

    if not speaker_map:
        log_warning(args, "Empty speaker mapping - no changes will be made")

    # Validate and log mapping
    validate_and_log_mapping(speaker_map, detected_speakers, args)

    # Apply mapping
    log_debug(args, "Applying speaker mapping to JSON...")
    mapped_json = replace_speakers_recursive(json_data, speaker_map)

    # Stdout-only mode (for benchmarking)
    if args.stdout_only:
        output = {
            "mappings": speaker_map,
            "detected_speakers": sorted(detected_speakers),
        }
        print(json.dumps(output, indent=2))
        return

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
        txt_content = generate_txt_from_json(mapped_json, args)
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
