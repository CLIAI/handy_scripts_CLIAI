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
        return f"META:\t{custom_message}\n"

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

    return f"META:\t{default_message}\n"


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
    args=None
):
    """
    Detect speaker names using LLM via Instructor.

    Args:
        provider_model: Provider and model string or shortcut (e.g., "4o-mini", "sonnet", "openai/gpt-4o-mini")
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

Look for:
- Direct name mentions (e.g., "Hi Alice", "Thanks Bob")
- Introductions ("I'm...", "My name is...")
- Self-references using third person ("Alice is happy", "Bob appreciates")
- Professional roles if names aren't mentioned (Host, Guest, Expert, Interviewer)
- Topics they discussed (AI, research, product features, etc.)
- Their role in the conversation (asking questions, explaining, presenting, etc.)
- Keywords, adjectives, or characteristics that identify them

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
                args
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
            args=args
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
    speaker_contexts: dict,
    args
) -> dict:
    """
    Interactive prompts with AI suggestions as defaults.

    Args:
        detected_speakers: Set of speaker labels
        ai_suggestions: Dict of AI-suggested names
        speaker_contexts: Dict of speaker labels to context information
        args: Arguments namespace

    Returns:
        Final speaker mapping dict, or None if user chooses to skip
    """
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
    print("  - Press Enter to accept suggestion", file=sys.stderr)
    print("  - Type name to override", file=sys.stderr)
    print("  - Type 'skip' to abort mapping (can rerun later)", file=sys.stderr)
    print("", file=sys.stderr)

    speaker_map = {}

    for speaker in sorted(detected_speakers):
        # Get AI suggestion
        suggestion = ai_suggestions.get(speaker, "Unknown")

        # Prompt with format: "A => [Alice Anderson]: "
        prompt_text = f"{speaker} => [{suggestion}]: "
        user_input = input(prompt_text).strip()

        # Check for skip command
        if user_input.lower() == 'skip':
            print("\nSkipping speaker mapping - no files will be created.", file=sys.stderr)
            print("You can rerun this command later to map speakers.", file=sys.stderr)
            return None

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
    meta_message_text = get_meta_message(args).replace("META:\t", "").strip()
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
                args=args
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
