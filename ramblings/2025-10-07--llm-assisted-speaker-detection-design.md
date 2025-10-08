# LLM-Assisted Speaker Detection Design for stt_assemblyai_speaker_mapper.py

**Design Date:** 2025-10-07
**Purpose:** Add LLM-powered automatic speaker name detection/suggestion to speaker mapper tool
**Status:** Design complete - implementation pending
**Related:** `ramblings/2025-10-07--llm-cli-tools-research-for-python-integration.md`

---

## Executive Summary

This document designs an LLM-assisted feature for automatically detecting and suggesting speaker names from AssemblyAI transcripts. The feature integrates with multiple LLM backends (mods, ollama, Instructor) and works in both automatic and interactive modes.

### Quick Overview

**Feature Goal:** Analyze transcript content to suggest speaker identities

```bash
# Automatic detection with LLM
./stt_assemblyai_speaker_mapper.py --llm-detect mods:gpt-4o-mini audio.json

# Interactive with AI suggestions
./stt_assemblyai_speaker_mapper.py --llm-interactive mods:gpt-4o-mini audio.json
# Prompts: Speaker A [Alice Anderson]: _
```

**Key Design Decisions:**

1. **Use Instructor Library** for structured outputs (10x better than manual JSON parsing)
2. **Support Multiple Backends** via modular abstraction (mods, ollama, direct API)
3. **Pydantic Models** for type-safe LLM responses
4. **Testable Architecture** with mock backends for CI/CD
5. **Graceful Degradation** - falls back to manual entry if LLM fails

---

## 1. User Experience Design

### 1.1 Non-Interactive Mode (Auto-Detection)

**Goal:** Automatically detect speaker names from transcript, apply mapping without user input

```bash
# Basic usage
./stt_assemblyai_speaker_mapper.py --llm-detect audio.assemblyai.json
# Uses default backend (mods) and model (gpt-4o-mini)

# Specify backend and model
./stt_assemblyai_speaker_mapper.py --llm-detect mods:gpt-4o-mini audio.json
./stt_assemblyai_speaker_mapper.py --llm-detect ollama:llama3 audio.json
./stt_assemblyai_speaker_mapper.py --llm-detect instructor:anthropic/claude-sonnet-4-5 audio.json

# With verbose output
./stt_assemblyai_speaker_mapper.py -v --llm-detect mods:gpt-4o-mini audio.json
# INFO: Sending transcript to LLM for speaker detection...
# INFO: LLM identified 2 speakers: Alice Anderson, Bob Smith
# INFO: Applying mapping: A→Alice Anderson, B→Bob Smith
```

**Flow:**

1. Load transcript JSON
2. Detect speakers (A, B, C, ...)
3. Send sample utterances to LLM with prompt
4. LLM returns suggested names
5. Apply mapping automatically
6. Generate output files

### 1.2 Interactive Mode with AI Suggestions

**Goal:** Show AI-suggested names as defaults, user can accept (Enter) or override (type name)

```bash
# Interactive with AI assistance
./stt_assemblyai_speaker_mapper.py --llm-interactive mods:gpt-4o-mini audio.json
```

**Interaction Flow:**

```
INFO: Analyzing transcript with LLM...
INFO: LLM suggested speaker names based on context

=== Speaker Mapping (AI-Assisted) ===
Speaker A [Alice Anderson]: _              # User presses Enter → Accept "Alice Anderson"
Speaker B [Bob Smith]: Robert              # User types → Override to "Robert"
Speaker C [Unknown]: Charlie Chaplin       # AI unsure, user provides name
```

**Behavior:**

* **Default shown in brackets** `[Alice Anderson]`
* **Press Enter** → Accept AI suggestion
* **Type name** → Override with custom name
* **Empty AI suggestion** → `[Unknown]` shown, requires user input
* **Confidence scoring** (future): Show `[Alice Anderson 95%]` vs `[Unknown 40%]`

### 1.3 Fallback Detection Mode

**Goal:** Try LLM detection, fall back to manual if it fails

```bash
./stt_assemblyai_speaker_mapper.py --llm-detect-fallback mods:gpt-4o-mini audio.json
```

**Flow:**

1. Attempt LLM detection
2. If fails (API error, invalid response, etc.) → Fall back to `--interactive` mode
3. Log warning, continue with manual entry

---

## 2. LLM Backend Architecture

### 2.1 Modular Backend Design

**Strategy Pattern** for swappable LLM backends:

```
┌─────────────────────────────────────────┐
│    stt_assemblyai_speaker_mapper.py     │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │   LLMDetector                     │ │
│  │   (High-level API)                │ │
│  └────────────┬──────────────────────┘ │
│               │                         │
│  ┌────────────┴──────────────────────┐ │
│  │   LLMBackend (Protocol)           │ │
│  │   - detect_speakers()             │ │
│  │   - supports_structured_output()  │ │
│  └────────────┬──────────────────────┘ │
│               │                         │
│    ┌──────────┴──────────┬────────────┐│
│    ▼                     ▼            ▼││
│┌────────┐  ┌──────────┐  ┌───────────┐││
││ Mods   │  │ Ollama   │  │Instructor │││
││Backend │  │ Backend  │  │ Backend   │││
│└────────┘  └──────────┘  └───────────┘││
└─────────────────────────────────────────┘
```

### 2.2 Backend Selection Syntax

**Format:** `backend:model` or `backend:provider/model`

**Examples:**

* `mods:gpt-4o-mini` → Mods CLI with GPT-4o-mini
* `mods:claude-sonnet-4-5` → Mods CLI with Claude
* `ollama:llama3` → Ollama with Llama 3
* `ollama:mistral:7b` → Ollama with Mistral 7B
* `instructor:openai/gpt-4o-mini` → Instructor with OpenAI
* `instructor:anthropic/claude-sonnet-4-5` → Instructor with Claude
* `instructor:ollama/llama3` → Instructor with local Ollama

**Default:** `mods:gpt-4o-mini` (best balance of cost, speed, quality)

### 2.3 Backend Classes

#### Protocol Definition

```python
from typing import Protocol, List, Dict
from pydantic import BaseModel

class SpeakerDetection(BaseModel):
    """Pydantic model for LLM speaker detection response"""
    speakers: Dict[str, str]  # {" A": "Alice Anderson", "B": "Bob Smith"}
    confidence: str = "medium"  # low, medium, high
    reasoning: str = ""  # Why these names were chosen

class LLMBackend(Protocol):
    """Protocol for LLM backends"""

    def detect_speakers(
        self,
        transcript_text: str,
        detected_labels: List[str]
    ) -> SpeakerDetection:
        """Detect speaker names from transcript"""
        ...
```

#### ModsBackend (CLI-based)

```python
import subprocess
import json
from typing import List

class ModsBackend:
    """Mods CLI backend for LLM processing"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def detect_speakers(
        self,
        transcript_text: str,
        detected_labels: List[str]
    ) -> SpeakerDetection:
        """Use mods to detect speaker names"""

        prompt = self._build_prompt(transcript_text, detected_labels)

        # Call mods CLI
        result = subprocess.run(
            ["mods", "-m", self.model, "-f", "json", prompt],
            input=transcript_text,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"Mods failed: {result.stderr}")

        # Parse JSON response
        data = json.loads(result.stdout)
        return SpeakerDetection(**data)

    def _build_prompt(self, text: str, labels: List[str]) -> str:
        return f"""
Analyze this transcript and identify the speakers.

Detected speaker labels: {', '.join(labels)}

Based on the conversation content, suggest likely names or roles for each speaker.
Respond with JSON matching this structure:
{{
    "speakers": {{"A": "suggested name", "B": "suggested name"}},
    "confidence": "low|medium|high",
    "reasoning": "brief explanation"
}}
"""
```

#### InstructorBackend (Structured)

```python
import instructor
from anthropic import Anthropic
from openai import OpenAI

class InstructorBackend:
    """Instructor library backend with structured outputs"""

    def __init__(self, provider_model: str = "openai/gpt-4o-mini"):
        """
        Args:
            provider_model: Format "provider/model"
                Examples: "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"
        """
        self.provider_model = provider_model
        self.client = self._init_client()

    def _init_client(self):
        """Initialize Instructor client for provider"""
        provider, model = self.provider_model.split("/", 1)

        if provider == "openai":
            return instructor.from_openai(OpenAI())
        elif provider == "anthropic":
            return instructor.from_anthropic(Anthropic())
        elif provider == "ollama":
            # Ollama uses OpenAI-compatible API
            return instructor.from_openai(
                OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                ),
                mode=instructor.Mode.JSON
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def detect_speakers(
        self,
        transcript_text: str,
        detected_labels: List[str]
    ) -> SpeakerDetection:
        """Use Instructor for structured speaker detection"""

        prompt = f"""
Analyze this transcript and identify the speakers.

Detected speaker labels: {', '.join(detected_labels)}

Based on the conversation content, context clues, and speaking patterns,
suggest the most likely names or roles for each speaker.
"""

        # Instructor automatically enforces Pydantic model
        response = self.client.chat.completions.create(
            model=self._extract_model(),
            response_model=SpeakerDetection,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing conversations and identifying speakers."},
                {"role": "user", "content": f"{prompt}\n\nTranscript:\n{transcript_text}"}
            ],
            max_tokens=1024
        )

        return response

    def _extract_model(self) -> str:
        """Extract model name from provider_model string"""
        return self.provider_model.split("/", 1)[1]
```

#### OllamaBackend (Local)

```python
class OllamaBackend:
    """Ollama backend for local/offline processing"""

    def __init__(self, model: str = "llama3"):
        self.model = model

    def detect_speakers(
        self,
        transcript_text: str,
        detected_labels: List[str]
    ) -> SpeakerDetection:
        """Use Ollama for local speaker detection"""

        prompt = self._build_prompt(transcript_text, detected_labels)

        # Call ollama CLI
        result = subprocess.run(
            ["ollama", "run", self.model, prompt],
            input=transcript_text,
            capture_output=True,
            text=True,
            timeout=60  # Local models may be slower
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama failed: {result.stderr}")

        # Ollama doesn't guarantee JSON, so parse manually
        try:
            data = json.loads(result.stdout)
            return SpeakerDetection(**data)
        except json.JSONDecodeError:
            # Fallback: extract names manually from text response
            return self._parse_text_response(result.stdout, detected_labels)

    def _parse_text_response(self, text: str, labels: List[str]) -> SpeakerDetection:
        """Parse non-JSON response (fallback)"""
        # Simple heuristic parsing
        speakers = {}
        for label in labels:
            # Look for patterns like "Speaker A: Alice" or "A is Alice"
            import re
            patterns = [
                rf"{label}[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
                rf"Speaker {label}[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)*)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    speakers[label] = match.group(1)
                    break

        return SpeakerDetection(
            speakers=speakers,
            confidence="low",
            reasoning="Parsed from unstructured response"
        )
```

### 2.4 Backend Factory

```python
def get_llm_backend(backend_spec: str) -> LLMBackend:
    """
    Factory to create LLM backend from specification string.

    Args:
        backend_spec: Format "backend:model" or "backend:provider/model"
            Examples:
                "mods:gpt-4o-mini"
                "ollama:llama3"
                "instructor:openai/gpt-4o-mini"
                "instructor:anthropic/claude-sonnet-4-5"

    Returns:
        LLMBackend instance
    """
    if ":" not in backend_spec:
        raise ValueError(f"Invalid backend spec: {backend_spec}. Expected format: backend:model")

    backend_type, model_spec = backend_spec.split(":", 1)

    if backend_type == "mods":
        return ModsBackend(model=model_spec)

    elif backend_type == "ollama":
        return OllamaBackend(model=model_spec)

    elif backend_type == "instructor":
        return InstructorBackend(provider_model=model_spec)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
```

---

## 3. Prompt Engineering

### 3.1 Speaker Detection Prompt Template

**Goal:** Maximize accuracy of speaker name detection from transcript context

```python
SPEAKER_DETECTION_PROMPT = """
You are analyzing a conversation transcript with speaker diarisation.

TASK: Identify the most likely names or roles for each speaker based on:
- Context clues in the conversation
- Names mentioned in the dialogue
- Speaking patterns and relationships
- Topic expertise signals

DETECTED SPEAKERS: {speaker_labels}

TRANSCRIPT EXCERPT:
{transcript_sample}

INSTRUCTIONS:
1. Analyze the conversation carefully
2. Look for direct name mentions (e.g., "Hi Alice", "Thanks Bob")
3. Infer roles if names aren't mentioned (e.g., "Host", "Guest", "Interviewer")
4. Assign confidence: high (names mentioned), medium (strong context clues), low (guessing)

RESPONSE FORMAT:
Provide a JSON object with this structure:
{{
    "speakers": {{
        "A": "Full Name or Role",
        "B": "Full Name or Role",
        ...
    }},
    "confidence": "low" | "medium" | "high",
    "reasoning": "Brief explanation of how you identified each speaker"
}}

IMPORTANT:
- Use "Unknown" if you cannot determine identity with reasonable confidence
- Prefer specific names over generic roles
- Include first and last names if mentioned
- Be conservative with confidence ratings
"""
```

### 3.2 Transcript Sampling Strategy

**Challenge:** Transcripts may be very long (exceeds token limits)

**Solution:** Send strategic samples

```python
def extract_transcript_sample(transcript_json: dict, max_utterances: int = 20) -> str:
    """
    Extract strategic sample of transcript for LLM analysis.

    Strategy:
    1. Include first few utterances (introductions often here)
    2. Include utterances with potential name mentions
    3. Include utterances from each speaker
    4. Limit total to avoid token limits

    Args:
        transcript_json: Full AssemblyAI JSON
        max_utterances: Maximum utterances to include

    Returns:
        Formatted transcript sample string
    """
    utterances = transcript_json.get('utterances', [])

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
    """Check if text contains capitalized words (potential names)"""
    import re
    # Match capitalized words that aren't sentence starts
    pattern = r'(?<![.!?]\s)(?<!\A)\b[A-Z][a-z]+'
    return bool(re.search(pattern, text))
```

---

## 4. Integration with Existing Code

### 4.1 New CLI Arguments

Add to `parse_args()`:

```python
# LLM-assisted detection group
llm_group = parser.add_argument_group('LLM-Assisted Detection')

llm_group.add_argument(
    '--llm-detect',
    metavar='BACKEND:MODEL',
    help='Automatically detect speaker names using LLM. '
         'Format: backend:model (e.g., "mods:gpt-4o-mini", "ollama:llama3")'
)

llm_group.add_argument(
    '--llm-interactive',
    metavar='BACKEND:MODEL',
    help='Interactive mode with AI-suggested speaker names as defaults'
)

llm_group.add_argument(
    '--llm-detect-fallback',
    metavar='BACKEND:MODEL',
    help='Try LLM detection, fall back to manual if it fails'
)

llm_group.add_argument(
    '--llm-sample-size',
    type=int,
    default=20,
    help='Number of utterances to send to LLM for analysis (default: 20)'
)
```

### 4.2 Modified Main Flow

```python
def main():
    args = parse_args()

    # ... (existing code: load JSON, detect speakers) ...

    # NEW: LLM-assisted detection
    if args.llm_detect or args.llm_interactive or args.llm_detect_fallback:
        speaker_map = handle_llm_detection(args, json_data, detected_speakers)

    # EXISTING: Manual mapping methods
    elif args.speaker_map:
        speaker_map = parse_speaker_map_inline(args.speaker_map, detected_speakers)
    elif args.speaker_map_file:
        speaker_map = parse_speaker_map_file(args.speaker_map_file, detected_speakers)
    elif args.interactive:
        speaker_map = prompt_interactive_mapping(detected_speakers, args)
    else:
        log_error(args, "No mapping source provided")
        sys.exit(1)

    # ... (rest of existing code) ...
```

### 4.3 LLM Detection Handler

```python
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
    # Determine backend spec
    backend_spec = args.llm_detect or args.llm_interactive or args.llm_detect_fallback

    if not backend_spec:
        # Default backend
        backend_spec = "mods:gpt-4o-mini"

    try:
        # Create backend
        backend = get_llm_backend(backend_spec)
        log_info(args, f"Using LLM backend: {backend_spec}")

        # Extract transcript sample
        transcript_sample = extract_transcript_sample(
            json_data,
            max_utterances=args.llm_sample_size
        )

        log_debug(args, f"Transcript sample ({len(transcript_sample)} chars):")
        log_debug(args, transcript_sample[:500] + "...")

        # Call LLM
        log_info(args, "Analyzing transcript with LLM...")
        detection_result = backend.detect_speakers(
            transcript_sample,
            list(detected_speakers)
        )

        log_info(args, f"LLM confidence: {detection_result.confidence}")
        log_debug(args, f"LLM reasoning: {detection_result.reasoning}")

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
```

### 4.4 Interactive with Suggestions

```python
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
```

---

## 5. Testing Strategy

### 5.1 Mock Backend for Testing

```python
class MockLLMBackend:
    """Mock backend for testing without real API calls"""

    def __init__(self, mock_response: SpeakerDetection = None):
        self.mock_response = mock_response or SpeakerDetection(
            speakers={"A": "Alice", "B": "Bob"},
            confidence="high",
            reasoning="Mock response for testing"
        )
        self.call_count = 0
        self.last_transcript = None
        self.last_labels = None

    def detect_speakers(
        self,
        transcript_text: str,
        detected_labels: List[str]
    ) -> SpeakerDetection:
        self.call_count += 1
        self.last_transcript = transcript_text
        self.last_labels = detected_labels
        return self.mock_response
```

### 5.2 Unit Tests

```python
# test_llm_detection.py

import unittest
from unittest.mock import Mock, patch
import json

class TestLLMDetection(unittest.TestCase):

    def test_mock_backend(self):
        """Test mock backend returns expected response"""
        mock_backend = MockLLMBackend()
        result = mock_backend.detect_speakers("transcript", ["A", "B"])

        self.assertEqual(result.speakers, {"A": "Alice", "B": "Bob"})
        self.assertEqual(result.confidence, "high")
        self.assertEqual(mock_backend.call_count, 1)

    def test_backend_factory_mods(self):
        """Test backend factory creates correct backend"""
        backend = get_llm_backend("mods:gpt-4o-mini")
        self.assertIsInstance(backend, ModsBackend)
        self.assertEqual(backend.model, "gpt-4o-mini")

    def test_backend_factory_ollama(self):
        """Test Ollama backend creation"""
        backend = get_llm_backend("ollama:llama3")
        self.assertIsInstance(backend, OllamaBackend)
        self.assertEqual(backend.model, "llama3")

    def test_backend_factory_instructor(self):
        """Test Instructor backend creation"""
        backend = get_llm_backend("instructor:openai/gpt-4o-mini")
        self.assertIsInstance(backend, InstructorBackend)
        self.assertEqual(backend.provider_model, "openai/gpt-4o-mini")

    def test_transcript_sampling(self):
        """Test transcript sample extraction"""
        transcript_json = {
            "utterances": [
                {"speaker": "A", "text": "Hello, I'm Alice"},
                {"speaker": "B", "text": "Hi Alice, I'm Bob"},
                {"speaker": "A", "text": "Nice to meet you"},
            ]
        }

        sample = extract_transcript_sample(transcript_json, max_utterances=10)

        self.assertIn("Speaker A: Hello, I'm Alice", sample)
        self.assertIn("Speaker B: Hi Alice, I'm Bob", sample)

    def test_proper_noun_detection(self):
        """Test proper noun detection heuristic"""
        self.assertTrue(has_proper_nouns("Hello Alice, how are you?"))
        self.assertTrue(has_proper_nouns("My name is Bob Smith"))
        self.assertFalse(has_proper_nouns("this is all lowercase"))
        self.assertFalse(has_proper_nouns("This is a sentence."))  # Sentence start doesn't count

    @patch('subprocess.run')
    def test_mods_backend_success(self, mock_run):
        """Test ModsBackend with successful response"""
        # Mock subprocess.run response
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                "speakers": {"A": "Alice", "B": "Bob"},
                "confidence": "high",
                "reasoning": "Names mentioned in conversation"
            })
        )

        backend = ModsBackend(model="gpt-4o-mini")
        result = backend.detect_speakers("transcript", ["A", "B"])

        self.assertEqual(result.speakers, {"A": "Alice", "B": "Bob"})
        self.assertEqual(result.confidence, "high")

        # Verify subprocess.run was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertIn("mods", call_args[0][0])
        self.assertIn("-m", call_args[0][0])
        self.assertIn("gpt-4o-mini", call_args[0][0])

    @patch('subprocess.run')
    def test_mods_backend_failure(self, mock_run):
        """Test ModsBackend handles errors gracefully"""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="API error"
        )

        backend = ModsBackend()
        with self.assertRaises(RuntimeError):
            backend.detect_speakers("transcript", ["A", "B"])
```

### 5.3 Integration Tests with Real Backends

```bash
# Manual integration testing script
# tests/test_llm_backends_integration.sh

#!/bin/bash
set -e

echo "=== Testing LLM Backends Integration ==="

# Test 1: Mods backend (requires OPENAI_API_KEY)
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Testing mods backend..."
    python3 -m pytest tests/integration/test_mods_backend.py -v
else
    echo "SKIP: mods backend (no OPENAI_API_KEY)"
fi

# Test 2: Ollama backend (requires ollama running)
if command -v ollama &> /dev/null && ollama list | grep -q llama3; then
    echo "Testing ollama backend..."
    python3 -m pytest tests/integration/test_ollama_backend.py -v
else
    echo "SKIP: ollama backend (ollama not installed or llama3 not pulled)"
fi

# Test 3: Instructor backend
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Testing instructor backend..."
    python3 -m pytest tests/integration/test_instructor_backend.py -v
else
    echo "SKIP: instructor backend (no OPENAI_API_KEY)"
fi

echo "=== Integration tests complete ==="
```

### 5.4 Test Command-Line Interface

```python
# test_cli_llm.py

import unittest
import subprocess
import json
import tempfile

class TestLLMCLI(unittest.TestCase):
    """Test LLM features via CLI"""

    def setUp(self):
        """Create sample test JSON"""
        self.test_json = {
            "utterances": [
                {"speaker": "A", "text": "Hi, I'm Alice"},
                {"speaker": "B", "text": "Hello Alice, I'm Bob"}
            ]
        }

        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        json.dump(self.test_json, self.temp_file)
        self.temp_file.close()

    def test_llm_detect_flag_format(self):
        """Test --llm-detect flag format validation"""
        # Valid format
        result = subprocess.run(
            ["./stt_assemblyai_speaker_mapper.py", "--llm-detect", "mods:gpt-4o-mini", "--detect", self.temp_file.name],
            capture_output=True,
            text=True
        )
        # Should not error on flag parsing
        self.assertNotIn("Invalid backend spec", result.stderr)

    def test_llm_detect_invalid_backend(self):
        """Test error handling for invalid backend"""
        result = subprocess.run(
            ["./stt_assemblyai_speaker_mapper.py", "--llm-detect", "invalid:model", self.temp_file.name],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown backend", result.stderr)
```

### 5.5 Test Data Sets

Create curated test transcripts for validation:

```python
# tests/fixtures/test_transcripts.py

TRANSCRIPT_WITH_NAMES = {
    "utterances": [
        {"speaker": "A", "text": "Hello, my name is Alice Anderson"},
        {"speaker": "B", "text": "Nice to meet you Alice, I'm Bob"},
        {"speaker": "A", "text": "Great to meet you too, Bob"},
    ]
}

TRANSCRIPT_WITHOUT_NAMES = {
    "utterances": [
        {"speaker": "A", "text": "Welcome to the show"},
        {"speaker": "B", "text": "Thanks for having me"},
        {"speaker": "A", "text": "Let's get started"},
    ]
}

TRANSCRIPT_ROLES = {
    "utterances": [
        {"speaker": "A", "text": "Today we're interviewing our guest"},
        {"speaker": "B", "text": "I'm excited to share my story"},
        {"speaker": "A", "text": "Our listeners will love it"},
    ]
}

# Expected LLM outputs for validation
EXPECTED_DETECTION_WITH_NAMES = {
    "speakers": {"A": "Alice Anderson", "B": "Bob"},
    "confidence": "high"
}

EXPECTED_DETECTION_ROLES = {
    "speakers": {"A": "Host", "B": "Guest"},
    "confidence": "medium"
}
```

---

## 6. Error Handling & Edge Cases

### 6.1 Error Scenarios

1. **LLM API unavailable**
   * Mods: API key missing/invalid
   * Ollama: Service not running
   * Instructor: Network error
   * **Handling:** Catch exception, log error, optionally fall back

2. **LLM returns invalid JSON**
   * **Handling:** Retry with clearer prompt, or fall back to manual

3. **LLM returns "Unknown" for all speakers**
   * **Handling:** Warn user, optionally fall back to manual

4. **Transcript too long (token limit exceeded)**
   * **Handling:** Use sampling strategy (already designed)

5. **LLM hallucination (invents names)**
   * **Handling:** Confidence scores, user review in interactive mode

### 6.2 Retry Logic

```python
def detect_with_retry(
    backend: LLMBackend,
    transcript: str,
    labels: List[str],
    max_retries: int = 3
) -> SpeakerDetection:
    """Detect speakers with retry logic"""

    for attempt in range(max_retries):
        try:
            result = backend.detect_speakers(transcript, labels)

            # Validate response
            if not result.speakers:
                raise ValueError("Empty speaker mapping returned")

            return result

        except Exception as e:
            if attempt == max_retries - 1:
                raise

            log_warning(args, f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## 7. Documentation Updates

### 7.1 README.md Additions

Add to `stt_assemblyai_speaker_mapper.README.md`:

```markdown
## LLM-Assisted Speaker Detection (Optional)

### Prerequisites

Install optional LLM backends:

bash
# For mods (multi-provider CLI)
brew install charmbracelet/tap/mods
export OPENAI_API_KEY="sk-..."

# For ollama (local models)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# For Instructor (Python library)
pip install "instructor[anthropic,openai]"


### Usage

#### Automatic Detection

bash
# Use LLM to automatically detect speaker names
./stt_assemblyai_speaker_mapper.py --llm-detect mods:gpt-4o-mini audio.json

# Use local Ollama model (offline, privacy-focused)
./stt_assemblyai_speaker_mapper.py --llm-detect ollama:llama3 audio.json

# Use Instructor with Claude (highest quality)
./stt_assemblyai_speaker_mapper.py --llm-detect instructor:anthropic/claude-sonnet-4-5 audio.json


#### Interactive with AI Suggestions

bash
# AI suggests names, you confirm or override
./stt_assemblyai_speaker_mapper.py --llm-interactive mods:gpt-4o-mini audio.json


Prompts will show:

Speaker A [Alice Anderson]: _    # Press Enter to accept
Speaker B [Unknown]: Bob         # Type to override


#### Fallback Mode

bash
# Try LLM, fall back to manual if it fails
./stt_assemblyai_speaker_mapper.py --llm-detect-fallback mods:gpt-4o-mini audio.json


### Supported Backends

| Backend | Format | Requirements | Cost | Offline |
|---------|--------|-------------|------|---------|
| `mods` | `mods:model` | API key | $$ | ❌ |
| `ollama` | `ollama:model` | Ollama installed | Free | ✅ |
| `instructor` | `instructor:provider/model` | pip install instructor | $$ | ⚠️ |

### Cost Estimation

- **mods:gpt-4o-mini**: ~$0.001-0.01 per transcript
- **ollama:llama3**: Free (local compute)
- **instructor:anthropic/claude-sonnet-4-5**: ~$0.01-0.05 per transcript

### Accuracy

LLM detection works best when:
- ✅ Speaker names are mentioned in conversation
- ✅ Speakers introduce themselves
- ✅ Distinct speaking patterns/topics
- ⚠️ May struggle with generic conversations
- ⚠️ Requires user review for critical applications
```

---

## 8. Implementation Phases

### Phase 1: MVP (Minimal Viable Product)

**Goal:** Get basic LLM detection working

**Tasks:**

1. Create `SpeakerDetection` Pydantic model
2. Implement `ModsBackend` class
3. Implement `get_llm_backend()` factory
4. Add `--llm-detect` CLI argument
5. Implement `extract_transcript_sample()`
6. Integrate into main flow
7. Basic error handling

**Testing:** Manual testing with mods + gpt-4o-mini

**Time Estimate:** 4-6 hours

### Phase 2: Structured Outputs

**Goal:** Add Instructor library for robust parsing

**Tasks:**

1. `pip install instructor`
2. Implement `InstructorBackend` class
3. Update factory to support instructor backend
4. Add retry logic
5. Unit tests for Instructor integration

**Testing:** Automated tests with mock responses

**Time Estimate:** 3-4 hours

### Phase 3: Interactive Mode

**Goal:** Add AI-assisted interactive prompts

**Tasks:**

1. Implement `prompt_interactive_with_suggestions()`
2. Add `--llm-interactive` CLI argument
3. Format prompts with suggestions as defaults
4. Handle user overrides
5. Unit tests for interactive flow

**Testing:** Manual testing + mocked input tests

**Time Estimate:** 2-3 hours

### Phase 4: Ollama Backend

**Goal:** Support local/offline models

**Tasks:**

1. Implement `OllamaBackend` class
2. Implement text response parser (fallback)
3. Add Ollama to documentation
4. Integration tests

**Testing:** Test with actual Ollama installation

**Time Estimate:** 3-4 hours

### Phase 5: Production Hardening

**Goal:** Handle edge cases, improve reliability

**Tasks:**

1. Comprehensive error handling
2. Retry logic with exponential backoff
3. Fallback mode (`--llm-detect-fallback`)
4. Logging improvements
5. Cost estimation warnings
6. Performance optimization

**Testing:** Integration tests, edge case validation

**Time Estimate:** 4-5 hours

### Phase 6: Documentation & Polish

**Goal:** Complete documentation and examples

**Tasks:**

1. Update README.md
2. Add usage examples
3. Create rambling doc with lessons learned
4. Performance benchmarks
5. Troubleshooting guide

**Testing:** Documentation review

**Time Estimate:** 2-3 hours

**Total Estimated Time:** 18-25 hours

---

## 9. Configuration File Support (Future)

### 9.1 Config File Format

```yaml
# llm_config.yml

llm:
  default_backend: "mods:gpt-4o-mini"

  # Backend-specific settings
  backends:
    mods:
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"

    ollama:
      model: "llama3"
      base_url: "http://localhost:11434"

    instructor:
      provider: "openai/gpt-4o-mini"
      max_retries: 3

  # Detection settings
  detection:
    sample_size: 20
    min_confidence: "medium"
    fallback_to_manual: true

  # Prompt templates
  prompts:
    speaker_detection: |
      Analyze this transcript and identify speakers...
      {custom_prompt}
```

### 9.2 Config Loading

```python
def load_llm_config(config_path: str = None) -> dict:
    """Load LLM configuration from YAML file"""
    import yaml

    if config_path is None:
        # Try default locations
        candidates = [
            "llm_config.yml",
            "~/.config/stt_assemblyai_speaker_mapper/llm.yml",
            "/etc/stt_speaker_mapper/llm.yml"
        ]
        for path in candidates:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                config_path = expanded
                break

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Return defaults
    return {
        "llm": {
            "default_backend": "mods:gpt-4o-mini",
            "detection": {
                "sample_size": 20,
                "min_confidence": "medium"
            }
        }
    }
```

---

## 10. Future Enhancements

### 10.1 Confidence-Based Suggestions

Show confidence scores in interactive mode:

```
Speaker A [Alice Anderson 95%]: _
Speaker B [Unknown 30%]: _
```

### 10.2 Multi-Language Support

Detect transcript language and adjust prompts:

```python
def detect_language(transcript_text: str) -> str:
    """Detect primary language in transcript"""
    # Use langdetect or similar
    from langdetect import detect
    return detect(transcript_text)

# Adjust prompt based on language
if language == "es":
    prompt = SPANISH_DETECTION_PROMPT
```

### 10.3 Speaker Relationship Detection

Detect relationships (host/guest, interviewer/interviewee):

```python
class SpeakerRelationship(BaseModel):
    speaker_a: str
    speaker_b: str
    relationship: str  # "host/guest", "colleagues", "friends", etc.
```

### 10.4 Cost Tracking

Track API costs per run:

```python
class CostTracker:
    PRICING = {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "claude-sonnet-4-5": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000}
    }

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
```

### 10.5 Caching for Repeated Runs

Cache LLM responses to avoid repeated API calls:

```python
import hashlib
import json

def cache_llm_response(transcript_hash: str, response: SpeakerDetection):
    """Cache LLM response for transcript"""
    cache_dir = os.path.expanduser("~/.cache/stt_speaker_mapper")
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{transcript_hash}.json")
    with open(cache_file, 'w') as f:
        json.dump(response.dict(), f)

def get_cached_response(transcript_hash: str) -> SpeakerDetection | None:
    """Retrieve cached LLM response"""
    cache_file = os.path.expanduser(f"~/.cache/stt_speaker_mapper/{transcript_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
            return SpeakerDetection(**data)
    return None

def compute_transcript_hash(transcript_text: str) -> str:
    """Compute hash of transcript for cache key"""
    return hashlib.sha256(transcript_text.encode()).hexdigest()[:16]
```

---

## 11. Summary & Recommendations

### Key Design Decisions

1. **Use Instructor Library** - Superior to manual JSON parsing (10x time savings)
2. **Support Multiple Backends** - Flexibility for different use cases (cloud vs local)
3. **Pydantic for Validation** - Type-safe, automatic retries
4. **Modular Architecture** - Easy to extend with new backends
5. **Graceful Degradation** - Falls back to manual if LLM fails

### Recommended Implementation Order

1. **Start with Phase 1 (MVP)** - Get basic mods integration working
2. **Add Phase 2 (Instructor)** - Improve robustness
3. **Add Phase 3 (Interactive)** - Better UX
4. **Phases 4-6 as needed** - Based on user feedback

### Testing Strategy

* **Unit tests** with mock backends (no API calls in CI/CD)
* **Integration tests** with real backends (manual/optional)
* **CLI tests** to validate argument parsing
* **Fixtures** with known-good transcripts for validation

### Documentation

* Update README with LLM features
* Add troubleshooting guide
* Include cost/accuracy information
* Provide examples for each backend

---

## Appendix A: Example Prompts

### A.1 High-Quality Detection Prompt

```
You are analyzing a conversation transcript with speaker diarisation.

OBJECTIVE: Identify the most likely full names or professional roles for each speaker.

ANALYSIS APPROACH:
1. Scan for direct name mentions (e.g., "Hi Alice", "Thanks, Bob")
2. Look for introductions ("I'm...", "My name is...")
3. Identify contextual clues (titles, expertise, relationship dynamics)
4. Infer professional roles if names aren't mentioned (Host, Guest, Expert, Interviewer)

DETECTED SPEAKER LABELS: {labels}

TRANSCRIPT SAMPLE:
{transcript}

RESPONSE REQUIREMENTS:
- Use full names when mentioned (e.g., "Alice Anderson", not "Alice")
- Use professional roles for unnamed speakers (e.g., "Host", "Guest", "Expert")
- Mark uncertain identifications as "Unknown"
- Provide confidence rating: high (names explicit), medium (strong context), low (weak inference)
- Include brief reasoning for each identification

OUTPUT FORMAT (JSON):
{
  "speakers": {
    "A": "Full Name or Role",
    "B": "Full Name or Role"
  },
  "confidence": "high" | "medium" | "low",
  "reasoning": "Brief explanation of identification method and key evidence"
}
```

### A.2 Role-Based Detection Prompt

```
Analyze this conversation to identify speaker roles and relationships.

TRANSCRIPT:
{transcript}

SPEAKERS: {labels}

Identify:
1. Professional roles (host, guest, interviewer, expert, caller, etc.)
2. Relationship dynamics (formal/informal, authority/peer)
3. Subject matter expertise signals

Output JSON:
{
  "speakers": {
    "A": "Role description",
    "B": "Role description"
  },
  "relationships": "Brief description of speaker dynamics",
  "confidence": "high|medium|low",
  "reasoning": "Key evidence for role assignments"
}
```

---

## Appendix B: Backend Comparison Matrix

| Criterion | Mods | Ollama | Instructor |
|-----------|------|--------|-----------|
| **Setup Complexity** | Low (brew install) | Medium (install + pull models) | Low (pip install) |
| **Cost** | API fees ($0.001-0.05/call) | Free (local compute) | API fees |
| **Speed** | Fast (cloud) | Medium-Slow (local) | Fast (cloud) |
| **Accuracy** | High (GPT-4o, Claude) | Medium (Llama3, Mistral) | Highest (with retries) |
| **Offline Support** | ❌ No | ✅ Yes | ⚠️ With Ollama backend |
| **Privacy** | ❌ Data sent to cloud | ✅ Local processing | ⚠️ Depends on provider |
| **Structured Output** | ⚠️ Prompt-based | ⚠️ Inconsistent | ✅ Validated Pydantic |
| **Error Handling** | Manual | Manual | ✅ Automatic retries |
| **Type Safety** | ❌ None | ❌ None | ✅ Full Pydantic |
| **Best For** | Quick prototyping | Privacy-critical | Production robustness |

---

## Appendix C: Dependency Installation

```bash
# Core dependencies (required)
pip install pydantic

# Optional: Instructor library (recommended)
pip install instructor
pip install "instructor[anthropic]"  # For Claude
pip install "instructor[openai]"     # For OpenAI

# Optional: Mods CLI (default backend)
brew install charmbracelet/tap/mods
# Or: winget install charmbracelet.mods (Windows)
# Or: yay -S mods (Arch Linux)

# Optional: Ollama (local backend)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama pull mistral

# Environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-07
**Author:** Design based on research by AI Research Agent
**Status:** Ready for implementation approval
**Next Steps:** User approval → Begin Phase 1 (MVP) implementation
