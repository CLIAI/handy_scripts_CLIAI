# Speechmatics Speaker Name Mapper

Post-processing tool to replace speaker labels (S1, S2, S3) with actual speaker names in Speechmatics transcription JSON files.

## Purpose

When `stt_speechmatics.py` generates transcriptions with speaker diarisation (`-d` flag), it produces speaker labels like "Speaker S1", "Speaker S2", etc. This tool allows you to replace those generic labels with actual names after you've reviewed the transcript and identified who each speaker is.

## Key Features

* **Format-agnostic**: Uses recursive JSON traversal to find and replace ALL `"speaker"` keys
* **Future-proof**: Works even if Speechmatics changes their JSON format
* **Multiple input methods**: Comma-separated, file-based, interactive prompts, or **LLM-assisted detection**
* **Non-destructive**: Creates new `.mapped.json` and `.mapped.txt` files, preserves originals
* **Idempotent**: Can remap the same transcript multiple times with different mappings
* **LLM-powered** (optional): Automatically detect speaker names from conversation context using AI
* **Audio preview**: Hear samples of each speaker during interactive mapping

## Speaker Labels

Speechmatics uses different speaker label format than AssemblyAI:

* **S1, S2, S3, ...** — Sequential speaker numbers
* **UU** — Unidentified/unknown speaker (when speaker cannot be determined)

## LLM-Assisted Speaker Detection (Optional)

Use AI to automatically identify speaker names from transcript context.

### Prerequisites

```bash
# Core dependencies
pip install instructor pydantic

# Provider-specific (install as needed)
pip install openai          # For OpenAI
pip install anthropic       # For Anthropic/Claude
pip install google-generativeai  # For Google Gemini

# For local Ollama (no API key needed)
ollama pull llama3.2
```

### Quick Start

```bash
# Automatic detection with OpenAI
./stt_speechmatics_speaker_mapper.py --llm-detect openai/gpt-4o-mini audio.speechmatics.json

# Local/offline with Ollama (free, no API key)
./stt_speechmatics_speaker_mapper.py --llm-detect ollama/llama3.2 audio.speechmatics.json

# Interactive mode with AI suggestions
./stt_speechmatics_speaker_mapper.py --llm-interactive anthropic/claude-3-5-haiku audio.speechmatics.json
```

### Supported LLM Providers

| Provider | Format | Example | Requirements |
|----------|--------|---------|--------------|
| **OpenAI** | `openai/MODEL` | `openai/gpt-4o-mini` | API key |
| **Anthropic** | `anthropic/MODEL` | `anthropic/claude-3-5-haiku` | API key |
| **Google** | `google/MODEL` | `google/gemini-2.0-flash` | API key |
| **Groq** | `groq/MODEL` | `groq/llama-3.3-70b-versatile` | API key |
| **Ollama** | `ollama/MODEL` | `ollama/llama3.2` | Local (no API key) |

### Model Shortcuts

```bash
# OpenAI
./stt_speechmatics_speaker_mapper.py --llm-detect 4o-mini audio.speechmatics.json

# Anthropic
./stt_speechmatics_speaker_mapper.py --llm-detect sonnet audio.speechmatics.json

# Google
./stt_speechmatics_speaker_mapper.py --llm-detect gemini audio.speechmatics.json

# Local
./stt_speechmatics_speaker_mapper.py --llm-detect smollm2:1.7b audio.speechmatics.json
```

### LLM Detection Modes

#### 1. Automatic Detection

AI analyzes transcript and applies best-guess speaker names:

```bash
./stt_speechmatics_speaker_mapper.py --llm-detect openai/gpt-4o-mini audio.speechmatics.json
```

**Output:**

```
INFO: Analyzing transcript with LLM...
INFO: LLM confidence: high
INFO: Detected 2 speaker(s): S1, S2
INFO: Applied mappings:
INFO:   S1 → Alice Anderson
INFO:   S2 → Bob Smith
Created: audio.speechmatics.mapped.json, audio.speechmatics.mapped.txt
```

#### 2. Interactive with AI Suggestions

AI suggests names, you confirm or override:

```bash
./stt_speechmatics_speaker_mapper.py --llm-interactive openai/gpt-4o-mini audio.speechmatics.json
```

**Interaction:**

```
=== AI-Detected Speaker Mappings ===
S1 => Alice Anderson # Host, asked questions
S2 => Bob Smith # Guest expert
UU => Unknown

=== Review and Confirm ===
  Enter=accept | name=override | skip=abort | speak=hear speaker | help=commands

S1 => [Alice Anderson]: _               ← Press Enter to accept
S2 => [Bob Smith]: Robert               ← Type to override
UU => [Unknown]: _                       ← Press Enter to keep
```

## Audio Preview Features

Preview speaker audio samples during interactive mapping.

### Requirements

* **ffmpeg** - For audio extraction
* **mpv** (preferred), **ffplay**, or **mplayer** - For playback

### Usage

```bash
# Preview a speaker
./stt_speechmatics_speaker_mapper.py --preview-speaker S1 audio.speechmatics.json

# Extract speaker audio to file
./stt_speechmatics_speaker_mapper.py --extract-speaker S1 -o speaker_s1.mp3 audio.speechmatics.json
```

During interactive mode:

```
S1 => [Alice Anderson]: speak        ← Hear samples for S1
S1 => [Alice Anderson]: speak S2     ← Hear samples for S2
S1 => [Alice Anderson]: play         ← Play entire audio file
```

## Workflow Integration

### LLM-Powered (Automated)

```bash
# Step 1: Transcribe audio with speaker diarisation
./stt_speechmatics.py -d audio.mp3
# Creates: audio.mp3.speechmatics.json, audio.mp3.txt

# Step 2: Let AI identify speakers
./stt_speechmatics_speaker_mapper.py --llm-detect openai/gpt-4o-mini audio.mp3.speechmatics.json
# Creates: audio.mp3.speechmatics.mapped.json, audio.mp3.speechmatics.mapped.txt

# Step 3: Review results
cat audio.mp3.speechmatics.mapped.txt
# Output:
# Alice Anderson:	Hello there
# Bob Smith:	Hi, how are you?
```

### Manual Mapping (Traditional)

```bash
# Step 1: Transcribe
./stt_speechmatics.py -d audio.mp3

# Step 2: Review transcript to identify speakers
cat audio.mp3.txt
# Output:
# Speaker S1:	Hello there
# Speaker S2:	Hi, how are you?

# Step 3: Detect speakers
./stt_speechmatics_speaker_mapper.py --detect audio.mp3.speechmatics.json
# Output: Detected speakers: S1, S2

# Step 4: Apply mapping
./stt_speechmatics_speaker_mapper.py -m "Alice Anderson,Bob Martinez" audio.mp3.speechmatics.json

# Step 5: Review
cat audio.mp3.speechmatics.mapped.txt
```

## Usage Examples

### 1. Detect Speakers (Dry-run)

```bash
./stt_speechmatics_speaker_mapper.py --detect audio.speechmatics.json
```

**Output:**

```
Detected speakers: S1, S2, S3, UU
```

### 2. Inline Comma-Separated Mapping

```bash
./stt_speechmatics_speaker_mapper.py -m "Alice,Bob,Charlie" audio.speechmatics.json
```

Maps speakers in **sorted order**:

* S1 → Alice
* S2 → Bob
* S3 → Charlie
* UU → (unmapped)

### 3. File-Based Mapping

#### Sequential Format

**File:** `speakers.txt`

```
Alice Anderson
Bob Martinez
Charlie Chaplin
```

**Usage:**

```bash
./stt_speechmatics_speaker_mapper.py -M speakers.txt audio.speechmatics.json
```

#### Key:Value Format

**File:** `speakers.txt`

```
S1: Alice Anderson
S2: Bob Martinez
S3: Charlie Chaplin
```

#### Full Speaker Labels

**File:** `speakers.txt`

```
Speaker S1: Alice Anderson
Speaker S2: Bob Martinez
```

### 4. Interactive Mapping

```bash
./stt_speechmatics_speaker_mapper.py --interactive audio.speechmatics.json
```

**Interaction:**

```
=== Detected Speakers ===
Name for 'S1' (press Enter to keep): Alice Anderson
Name for 'S2' (press Enter to keep): Bob Martinez
Name for 'UU' (press Enter to keep):
```

## Context Files for Speaker Detection

Improve LLM detection accuracy with context files.

### Directory Context (SPEAKER.CONTEXT.md)

Applies to all audio files in a directory tree:

```markdown
# Project Context

This project contains recordings from Company X.

Common speakers:
* Greg Williams - CEO, leads most meetings
* Alice Chen - CTO, discusses technical topics
```

### File-Specific Context (.about.md)

Create `{audiofile}.about.md`:

```markdown
## Meeting Context

This is a product planning meeting between:
* Alice Chen - Product Manager, leads the discussion
* Bob Smith - Engineering Lead, discusses technical feasibility
```

## Command-Line Options

### Positional Arguments

* `input_json` - Path to Speechmatics JSON file (e.g., `audio.speechmatics.json`)

### Mapping Sources (Mutually Exclusive)

* `-m, --speaker-map STR` - Comma-separated speaker names
* `-M, --speaker-map-file PATH` - File with speaker mappings
* `--interactive` - Interactively prompt for speaker names
* `--llm-detect PROVIDER/MODEL` - Automatic LLM detection
* `--llm-interactive PROVIDER/MODEL` - Interactive with AI suggestions
* `--llm-detect-fallback PROVIDER/MODEL` - LLM with manual fallback

### LLM Configuration

* `--llm-endpoint URL` - Custom endpoint (for remote Ollama)
* `--llm-sample-size N` - Utterances to analyze (default: 20)

### Output Control

* `-o, --output BASE` - Output base name
* `-f, --force` - Overwrite existing output files
* `--txt-only` - Generate only .txt file
* `--json-only` - Generate only .json file
* `--detect` - Only show detected speakers and exit
* `--stdout-only` - Output mapping as JSON to stdout

### Audio Preview

* `--preview-speaker LABEL` - Preview audio for a speaker (e.g., S1)
* `--extract-speaker LABEL` - Extract speaker audio to file
* `--max-samples N` - Maximum samples to extract (default: 10)

### Logging

* `-v, --verbose` - Increase verbosity (-v=INFO, -vvvvv=DEBUG)
* `-q, --quiet` - Suppress all non-error output

### META Message Control

* `--no-meta-message` - Disable META warning message

## Output Files

### Default Naming

* **Input:** `audio.mp3.speechmatics.json`
* **JSON output:** `audio.mp3.speechmatics.mapped.json`
* **TXT output:** `audio.mp3.speechmatics.mapped.txt`

### TXT Format

```
Alice Anderson:	Hello there
Bob Martinez:	Hi, how are you?
Alice Anderson:	I'm doing well
```

**Note:** Tab character (`\t`) after colon for easy parsing/alignment

### JSON Format

All `"speaker"` key values are replaced throughout the JSON structure:

```json
{
  "results": [
    {
      "type": "word",
      "alternatives": [{"content": "Hello", "confidence": 0.98}],
      "start_time": 0.5,
      "end_time": 0.9,
      "speaker": "Alice Anderson"
    }
  ]
}
```

## Validation & Warnings

### Unmapped Speakers

```bash
WARNING: Unmapped speakers (keeping original): UU
```

### Extra Mappings

```bash
WARNING: Extra mappings for non-existent speakers: S5
```

## Error Handling

### File Not Found

```bash
ERROR: File not found: audio.speechmatics.json
```

### Invalid JSON

```bash
ERROR: Invalid JSON: Expecting value: line 1 column 1 (char 0)
```

### No Speakers Detected

```bash
ERROR: No speakers detected in JSON (no 'speaker' keys found)
```

This means diarisation was not enabled. Run `stt_speechmatics.py` with `-d` flag.

## Comparison: Speechmatics vs AssemblyAI

| Feature | Speechmatics | AssemblyAI |
|---------|-------------|------------|
| Speaker labels | S1, S2, S3... | A, B, C... |
| Unknown speaker | UU | - |
| JSON suffix | `.speechmatics.json` | `.assemblyai.json` |
| Mapped suffix | `.speechmatics.mapped.json` | `.assemblyai.mapped.json` |

## Related Tools

* `stt_speechmatics.py` - Main transcription tool (creates the JSON files)
* `stt_assemblyai_speaker_mapper.py` - Similar tool for AssemblyAI transcripts
* `stt_assemblyai.py` - Alternative transcription using AssemblyAI

## License

Part of the CLIAI handy_scripts collection.
