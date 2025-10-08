# AssemblyAI Speaker Name Mapper

Post-processing tool to replace speaker labels (A, B, C) with actual speaker names in AssemblyAI transcription JSON files.

## Purpose

When `stt_assemblyai.py` generates transcriptions with speaker diarisation (`-d` flag), it produces speaker labels like "Speaker A", "Speaker B", etc. This tool allows you to replace those generic labels with actual names after you've reviewed the transcript and identified who each speaker is.

## Key Features

* **Format-agnostic**: Uses recursive JSON traversal to find and replace ALL `"speaker"` keys, regardless of JSON structure
* **Future-proof**: Works even if AssemblyAI changes their JSON format
* **Multiple input methods**: Comma-separated, file-based (4 formats), interactive prompts, or **LLM-assisted detection**
* **Non-destructive**: Creates new `.mapped.json` and `.mapped.txt` files, preserves originals
* **Idempotent**: Can remap the same transcript multiple times with different mappings
* **LLM-powered** (optional): Automatically detect speaker names from conversation context using AI

## LLM-Assisted Speaker Detection (Optional)

**NEW:** Use AI to automatically identify speaker names from transcript context!

### Prerequisites

Install Instructor library for LLM integration:

```bash
# Core dependencies
pip install instructor pydantic

# Provider-specific (install as needed)
pip install openai          # For OpenAI
pip install anthropic       # For Anthropic/Claude
pip install google-generativeai  # For Google Gemini

# For local Ollama (no API key needed)
# Install Ollama from ollama.com
ollama pull llama3.2
```

### Quick Start

```bash
# Automatic detection with OpenAI
./stt_assemblyai_speaker_mapper.py --llm-detect openai/gpt-4o-mini audio.json

# Local/offline with Ollama (free, no API key)
./stt_assemblyai_speaker_mapper.py --llm-detect ollama/llama3.2 audio.json

# Interactive mode with AI suggestions
./stt_assemblyai_speaker_mapper.py --llm-interactive anthropic/claude-3-5-haiku audio.json
```

### Supported LLM Providers

| Provider | Format | Example | Requirements |
|----------|--------|---------|--------------|
| **OpenAI** | `openai/MODEL` | `openai/gpt-4o-mini` | API key |
| **Anthropic** | `anthropic/MODEL` | `anthropic/claude-3-5-haiku` | API key |
| **Google** | `google/MODEL` | `google/gemini-2.0-flash-exp` | API key |
| **Groq** | `groq/MODEL` | `groq/llama-3.1-70b-versatile` | API key (ultra-fast) |
| **Ollama** | `ollama/MODEL` | `ollama/llama3.2` | Local (no API key) |
| **100+ more** | via LiteLLM | `litellm/...` | Varies |

**Set API keys:**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
```

### LLM Detection Modes

#### 1. Automatic Detection

AI analyzes transcript and applies best-guess speaker names:

```bash
./stt_assemblyai_speaker_mapper.py --llm-detect openai/gpt-4o-mini audio.json
```

**Output:**

```
INFO: Analyzing transcript with LLM...
INFO: LLM confidence: high
INFO: LLM reasoning: Names explicitly mentioned in conversation
INFO: Detected 2 speaker(s): A, B
INFO: Applied mappings:
INFO:   A → Alice Anderson
INFO:   B → Bob Smith
Created: audio.assemblyai.mapped.json, audio.mapped.txt
```

#### 2. Interactive with AI Suggestions

AI suggests names, you confirm or override:

```bash
./stt_assemblyai_speaker_mapper.py --llm-interactive openai/gpt-4o-mini audio.json
```

**Interaction:**

```
=== Speaker Mapping (AI-Assisted) ===
Speaker A [Alice Anderson]: _          ← Press Enter to accept
Speaker B [Bob Smith]: Robert          ← Type to override
Speaker C [Unknown]: Charlie Chaplin   ← AI unsure, provide name
```

#### 3. Fallback Mode

Try AI, fall back to manual if it fails:

```bash
./stt_assemblyai_speaker_mapper.py --llm-detect-fallback ollama/llama3.2 audio.json
```

If LLM fails (API error, timeout, etc.), automatically switches to manual interactive mode.

### Advanced LLM Options

#### Custom Endpoint (Remote Ollama)

```bash
./stt_assemblyai_speaker_mapper.py \
  --llm-detect ollama/llama3.2 \
  --llm-endpoint http://gpu-server:11434 \
  audio.json
```

#### Sample Size Control

```bash
# Send more utterances for better context (default: 20)
./stt_assemblyai_speaker_mapper.py \
  --llm-detect openai/gpt-4o-mini \
  --llm-sample-size 30 \
  audio.json
```

#### Verbose LLM Output

```bash
./stt_assemblyai_speaker_mapper.py -vv --llm-detect openai/gpt-4o-mini audio.json
```

Shows detailed LLM reasoning and confidence scores.

### Cost & Performance

| Provider | Speed | Cost/transcript | Quality | Offline |
|----------|-------|-----------------|---------|---------|
| Groq | ⚡⚡⚡ | ~$0.001 | ⭐⭐⭐⭐ | No |
| OpenAI gpt-4o-mini | ⚡⚡ | ~$0.005 | ⭐⭐⭐⭐⭐ | No |
| Anthropic Haiku | ⚡⚡⭐ | ~$0.002 | ⭐⭐⭐⭐ | No |
| Anthropic Sonnet | ⚡⚡ | ~$0.020 | ⭐⭐⭐⭐⭐ | No |
| Ollama (local) | ⚡ | Free | ⭐⭐⭐ | **Yes** |

**Recommended:** Start with `groq/llama-3.1-70b-versatile` (fast + cheap) or `ollama/llama3.2` (free + offline).

### How It Works

The LLM analyzes a strategic sample of the transcript looking for:

* **Direct name mentions**: "Hi Alice", "Thanks Bob"
* **Introductions**: "I'm...", "My name is..."
* **Context clues**: Professional roles, relationships, topics
* **Speaking patterns**: Formality, expertise signals

It returns structured suggestions with confidence levels:

* **High**: Names explicitly mentioned
* **Medium**: Strong contextual clues
* **Low**: Weak inference (often returns "Unknown")

## Workflow Integration

### Option 1: LLM-Powered (Automated)

**NEW:** Let AI identify speakers automatically!

```bash
# Step 1: Transcribe audio with speaker diarisation
./stt_assemblyai.py -d audio.mp3
# Creates: audio.mp3.assemblyai.json, audio.mp3.txt

# Step 2: Let AI identify speakers (single command!)
./stt_assemblyai_speaker_mapper.py --llm-detect openai/gpt-4o-mini audio.mp3.assemblyai.json
# Creates: audio.mp3.assemblyai.mapped.json, audio.mp3.mapped.txt

# Step 3: Review results
cat audio.mp3.mapped.txt
# Output:
# Alice Anderson:	Hello there
# Bob Smith:	Hi, how are you?
# Alice Anderson:	I'm doing well
```

**Even better - Interactive with AI suggestions:**

```bash
# Step 2 alternative: AI suggests, you confirm/override
./stt_assemblyai_speaker_mapper.py --llm-interactive openai/gpt-4o-mini audio.mp3.assemblyai.json
# Prompts:
# Speaker A [Alice Anderson]: ← Press Enter to accept
# Speaker B [Bob Smith]: ← Press Enter to accept
```

### Option 2: Manual Mapping (Traditional)

```bash
# Step 1: Transcribe audio with speaker diarisation
./stt_assemblyai.py -d audio.mp3
# Creates: audio.mp3.assemblyai.json, audio.mp3.txt

# Step 2: Review transcript to identify speakers
cat audio.mp3.txt
# Output:
# Speaker A: Hello there
# Speaker B: Hi, how are you?
# Speaker A: I'm doing well

# Step 3: Detect speakers in JSON (optional)
./stt_assemblyai_speaker_mapper.py --detect audio.mp3.assemblyai.json
# Output: Detected speakers: A, B

# Step 4: Apply speaker name mapping manually
./stt_assemblyai_speaker_mapper.py -m "Alice Anderson,Beat Barrinson" audio.mp3.assemblyai.json
# Creates: audio.mp3.assemblyai.mapped.json, audio.mp3.mapped.txt

# Step 5: Review mapped transcript
cat audio.mp3.mapped.txt
# Output:
# Alice Anderson:	Hello there
# Beat Barrinson:	Hi, how are you?
# Alice Anderson:	I'm doing well
```

## Usage Examples

### 1. Detect Speakers (Dry-run)

```bash
./stt_assemblyai_speaker_mapper.py --detect audio.assemblyai.json
```

**Output:**

```
Detected speakers: A, B, C
```

### 2. Inline Comma-Separated Mapping

```bash
./stt_assemblyai_speaker_mapper.py -m "Alice Anderson,Beat Barrinson,Charlie Chaplin" audio.assemblyai.json
```

Maps speakers in **sorted order**:

* A → Alice Anderson
* B → Beat Barrinson
* C → Charlie Chaplin

### 3. File-Based Mapping (Auto-Detects Format)

#### Format 1: Sequential (Simple)

**File:** `speakers.txt`

```
Alice Anderson
Beat Barrinson
Charlie Chaplin
```

**Usage:**

```bash
./stt_assemblyai_speaker_mapper.py -M speakers.txt audio.assemblyai.json
```

**Mapping:** Sorted speakers → Sequential names

* A → Alice Anderson
* B → Beat Barrinson
* C → Charlie Chaplin

#### Format 2: Explicit Key:Value

**File:** `speakers.txt`

```
A: Alice Anderson
B: Beat Barrinson
C: Charlie Chaplin
```

**Usage:**

```bash
./stt_assemblyai_speaker_mapper.py -M speakers.txt audio.assemblyai.json
```

**Mapping:** Direct key-to-value

#### Format 3: Full Speaker Labels

**File:** `speakers.txt`

```
Speaker A: Alice Anderson
Speaker B: Beat Barrinson
```

**Usage:**

```bash
./stt_assemblyai_speaker_mapper.py -M speakers.txt audio.assemblyai.json
```

**Mapping:** Full label as key

#### Format 4: Mixed (Flexible)

**File:** `speakers.txt`

```
A: Alice Anderson
Speaker B: Beat Barrinson
C: Charlie Chaplin
```

**Usage:**

```bash
./stt_assemblyai_speaker_mapper.py -M speakers.txt audio.assemblyai.json
```

**Mapping:** Handles both formats in the same file

### 4. Interactive Mapping

```bash
./stt_assemblyai_speaker_mapper.py --interactive audio.assemblyai.json
```

**Interaction:**

```
=== Detected Speakers ===
Name for 'A' (press Enter to keep): Alice Anderson
Name for 'B' (press Enter to keep): Beat Barrinson
Name for 'C' (press Enter to keep):
INFO: Detected 3 speaker(s): A, B, C
INFO: Applied mappings:
INFO:   A → Alice Anderson
INFO:   B → Beat Barrinson
INFO: Wrote JSON: audio.assemblyai.mapped.json
INFO: Wrote TXT: audio.mapped.txt
Created: audio.assemblyai.mapped.json, audio.mapped.txt
```

### 5. Advanced Options

#### Verbose Output + Force Overwrite

```bash
./stt_assemblyai_speaker_mapper.py -vv -f -m "Host,Guest" interview.json
```

#### Custom Output Path

```bash
./stt_assemblyai_speaker_mapper.py -o final_transcript -m "Alice,Bob" audio.json
# Creates: final_transcript.json, final_transcript.txt
```

#### Generate Only TXT (Quick Preview)

```bash
./stt_assemblyai_speaker_mapper.py --txt-only -m "Alice,Bob" audio.json
# Creates only: audio.mapped.txt
```

#### Generate Only JSON

```bash
./stt_assemblyai_speaker_mapper.py --json-only -m "Alice,Bob" audio.json
# Creates only: audio.assemblyai.mapped.json
```

## Command-Line Options

### Positional Arguments

* `input_json` - Path to AssemblyAI JSON file (e.g., `audio.assemblyai.json`)

### Mapping Sources (Mutually Exclusive)

* `-m, --speaker-map STR` - Comma-separated speaker names (e.g., `"Alice,Bob,Charlie"`)
* `-M, --speaker-map-file PATH` - File with speaker mappings (auto-detects format)
* `--interactive` - Interactively prompt for speaker names

### Output Control

* `-o, --output BASE` - Output base name (default: auto-generate with `.mapped`)
* `-f, --force` - Overwrite existing output files
* `--txt-only` - Generate only .txt file (skip .json)
* `--json-only` - Generate only .json file (skip .txt)
* `--detect` - Only show detected speakers and exit (no processing)

### Logging

* `-v, --verbose` - Increase verbosity (count-based: `-v` = INFO, `-vvvvv` = DEBUG)
* `-q, --quiet` - Suppress all non-error output

## Output Files

### Default Naming

* **Input:** `audio.mp3.assemblyai.json`
* **JSON output:** `audio.mp3.assemblyai.mapped.json` (full JSON with speaker fields replaced)
* **TXT output:** `audio.mp3.mapped.txt` (formatted transcript with tab after speaker name)

### TXT Format

```
Alice Anderson:	Hello there
Beat Barrinson:	Hi, how are you?
Alice Anderson:	I'm doing well
```

**Note:** Tab character (`\t`) after colon for easy parsing/alignment

### JSON Format

All `"speaker"` key values are replaced throughout the JSON structure:

```json
{
  "utterances": [
    {
      "speaker": "Alice Anderson",
      "text": "Hello there",
      "confidence": 0.95,
      "start": 100,
      "end": 1500,
      "words": [
        {
          "text": "Hello",
          "start": 100,
          "end": 500,
          "confidence": 0.98,
          "speaker": "Alice Anderson"
        }
      ]
    }
  ]
}
```

## How It Works: Recursive Traversal

The tool uses **recursive JSON traversal** to find and replace speaker values, making it robust against JSON structure changes:

```python
def replace_speakers_recursive(obj, speaker_map):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "speaker" and isinstance(value, str):
                # Replace speaker value
                obj[key] = speaker_map.get(value, value)
            else:
                # Recurse into nested structures
                replace_speakers_recursive(value, speaker_map)
    elif isinstance(obj, list):
        for item in obj:
            replace_speakers_recursive(item, speaker_map)
```

**Benefits:**

* Works with ANY JSON structure containing `"speaker"` keys
* Future-proof: handles AssemblyAI API changes
* Comprehensive: catches speaker references in unexpected locations
* Portable: could work with other STT providers' JSON formats

## Validation & Warnings

The tool validates your mapping and provides helpful warnings:

### Unmapped Speakers

```bash
WARNING: Unmapped speakers (keeping original): C
```

You provided mapping for A and B, but speaker C exists in the transcript. C will remain as "Speaker C".

### Extra Mappings

```bash
WARNING: Extra mappings for non-existent speakers: D
```

You provided a mapping for speaker D, but no speaker D exists in the JSON.

### Empty Mapping

```bash
WARNING: Empty speaker mapping - no changes will be made
```

No valid mappings were found (e.g., empty file or all speakers skipped in interactive mode).

## Error Handling

### File Not Found

```bash
ERROR: File not found: audio.assemblyai.json
```

### Invalid JSON

```bash
ERROR: Invalid JSON: Expecting value: line 1 column 1 (char 0)
```

### No Speakers Detected

```bash
ERROR: No speakers detected in JSON (no 'speaker' keys found)
```

This means the JSON doesn't contain speaker diarisation data. Run `stt_assemblyai.py` with `-d` flag to enable diarisation.

### No Mapping Source

```bash
ERROR: No mapping source provided (use -m, -M, or --interactive)
```

### Output Files Exist

```bash
ERROR: Output file(s) already exist: audio.mapped.txt, audio.assemblyai.mapped.json
ERROR: Use -f/--force to overwrite
```

## Edge Cases & Tips

### Partial Mapping

You can map only some speakers:

```bash
# Only map speaker A, keep B and C as-is
./stt_assemblyai_speaker_mapper.py -m "Alice" audio.json
```

### Comment Lines in Mapping Files

Mapping files support comment lines (lines starting with `#`):

```
# Project interview speakers
A: Alice Anderson
B: Beat Barrinson
# C was not identified yet
```

### Remapping Multiple Times

The tool is idempotent - you can remap the same file multiple times:

```bash
# First attempt (wrong names)
./stt_assemblyai_speaker_mapper.py -m "John,Jane" audio.json

# Correct attempt
./stt_assemblyai_speaker_mapper.py -f -m "Alice,Bob" audio.json
```

### Integration with Wrapper Scripts

Works seamlessly with `stt_video_using_assemblyai.sh`:

```bash
# Extract and transcribe
./stt_video_using_assemblyai.sh -d video.mp4

# Review transcript
cat video.mp4.txt

# Map speakers
./stt_assemblyai_speaker_mapper.py -m "Host,Guest" video.mp4.assemblyai.json
```

## Troubleshooting

### Problem: TXT file not created

**Cause:** No transcript segments found in JSON

**Solution:** Check that JSON contains diarisation data with `--detect` flag

### Problem: Wrong speaker order

**Cause:** Speakers are mapped in **sorted** order (A, B, C)

**Solution:** Use explicit key:value format in mapping file:

```
B: Bob (first speaker chronologically)
A: Alice (second speaker chronologically)
```

### Problem: JSON structure changed

**Cause:** AssemblyAI updated their API response format

**Solution:** The recursive traversal should handle this automatically. If not, file an issue with sample JSON.

## Development & Testing

### Run Unit Tests

```bash
python3 test_stt_assemblyai_speaker_mapper.py
```

### Test with Sample Data

```bash
# Create sample JSON
cat > sample.json << 'EOF'
{
  "utterances": [
    {"speaker": "A", "text": "Hello"},
    {"speaker": "B", "text": "Hi there"}
  ]
}
EOF

# Test detection
./stt_assemblyai_speaker_mapper.py --detect sample.json

# Test mapping
./stt_assemblyai_speaker_mapper.py -m "Alice,Bob" sample.json
```

## Related Tools

* `stt_assemblyai.py` - Main transcription tool (creates the JSON files this tool processes)
* `stt_video_using_assemblyai.sh` - Wrapper script for video transcription
* `google_cloud_ai/multi-speaker_markup_from_dialog_transcript.py` - Similar tool for Google Cloud AI TTS

## License

Part of the CLIAI handy_scripts collection.
