# speaker-llm - LLM-based Speaker Name Detection

Detect speaker names from conversation transcripts using Large Language Model analysis. Part of the speaker-* tool ecosystem.

## Overview

`speaker-llm` uses LLM providers (Claude, GPT, Ollama) to analyze transcript text and identify speaker names through conversational patterns like direct address, self-introduction, and third-person mentions.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Transcript    │───>│   speaker-llm   │───>│   Detections    │
│   (S1, S2...)   │    │   + LLM API     │    │   (names,conf)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Provider Order:  │
                    │  1. Anthropic     │
                    │  2. OpenAI        │
                    │  3. Ollama        │
                    └───────────────────┘
```

## Installation

No installation required. The script uses `uv run` for dependency management.

```bash
# Run directly - dependencies are auto-installed
./speaker-llm analyze transcript.json

# Requirements:
# - Python 3.11+
# - One of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or Ollama server
```

## Quick Start

```bash
# 1. Check available providers
./speaker-llm providers

# 2. Full analysis with evidence
./speaker-llm analyze meeting.json

# 3. Quick mode - just names
./speaker-llm detect-names meeting.json

# 4. Use specific provider
./speaker-llm analyze meeting.json --provider openai

# 5. Use specific model
./speaker-llm analyze meeting.json --model claude-3-5-sonnet-20241022
```

## Commands

### `analyze` - Full Speaker Name Analysis

Performs comprehensive analysis with confidence scores and evidence.

```bash
./speaker-llm analyze <transcript> [OPTIONS]

Options:
  -p, --provider PROVIDER    LLM provider: anthropic, openai, ollama
  -m, --model MODEL          Model to use (provider default if not specified)
  -c, --context NAME         Context name for the conversation
  -f, --format FORMAT        Output format: text, json (default: json)
  --no-cache                 Bypass response cache
  -v, --verbose              Show evidence quotes
  -q, --quiet                Suppress status messages
```

Examples:

```bash
# Basic analysis (auto-select provider)
./speaker-llm analyze meeting.speechmatics.json

# Use Claude with specific model
./speaker-llm analyze meeting.json --provider anthropic --model claude-3-opus-20240229

# Human-readable output with evidence
./speaker-llm analyze meeting.json --format text --verbose

# Skip cache for fresh analysis
./speaker-llm analyze meeting.json --no-cache
```

Output (JSON):

```json
{
  "detections": [
    {
      "speaker_label": "S1",
      "detected_name": "Alice",
      "confidence": 0.85,
      "evidence": [
        "Hi everyone, this is Alice from the product team",
        "Thanks Alice, great point"
      ]
    },
    {
      "speaker_label": "S2",
      "detected_name": "Bob",
      "confidence": 0.70,
      "evidence": [
        "Bob, what do you think about this approach?"
      ]
    }
  ],
  "model": "claude-3-haiku-20240307",
  "provider": "anthropic",
  "processed_at": "2026-01-17T14:00:00Z"
}
```

### `detect-names` - Quick Name Detection

Simplified mode that returns just the speaker-to-name mapping.

```bash
./speaker-llm detect-names <transcript> [OPTIONS]

Options:
  -p, --provider PROVIDER    LLM provider
  -m, --model MODEL          Model to use
  -f, --format FORMAT        Output format: text, json (default: json)
  --no-cache                 Bypass response cache
```

Examples:

```bash
# Quick detection
./speaker-llm detect-names meeting.json

# Text output for scripting
./speaker-llm detect-names meeting.json --format text
# Output:
# S1: Alice
# S2: Bob
```

Output (JSON):

```json
{
  "names": {
    "S1": "Alice",
    "S2": "Bob",
    "S3": null
  },
  "model": "gpt-4o-mini",
  "provider": "openai",
  "processed_at": "2026-01-17T14:00:00Z"
}
```

### `providers` - Show Available Providers

Display status of all LLM providers.

```bash
./speaker-llm providers
```

Output:

```
LLM Providers:
========================================

anthropic:
  Status: available
  Env var: ANTHROPIC_API_KEY
  Default model: claude-3-haiku-20240307

openai:
  Status: not configured
  Env var: OPENAI_API_KEY
  Default model: gpt-4o-mini

ollama:
  Status: available
  Env var: OLLAMA_HOST
  Default model: llama3.2
  Host: http://localhost:11434

========================================
Auto-selected provider: anthropic
```

### `clear-cache` - Clear Response Cache

Remove cached LLM responses.

```bash
./speaker-llm clear-cache [--force]

Options:
  -f, --force    Skip confirmation prompt
```

## LLM Providers

Provider priority order (first available is used):

| Priority | Provider | Env Variable | Default Model |
|----------|----------|--------------|---------------|
| 1 | Anthropic | `ANTHROPIC_API_KEY` | claude-3-haiku-20240307 |
| 2 | OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
| 3 | Ollama | `OLLAMA_HOST` | llama3.2 |

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
./speaker-llm analyze transcript.json
```

Recommended models:

* `claude-3-haiku-20240307` - Fast, cost-effective (default)
* `claude-3-5-sonnet-20241022` - Better accuracy
* `claude-3-opus-20240229` - Best quality

### OpenAI GPT

```bash
export OPENAI_API_KEY="sk-..."
./speaker-llm analyze transcript.json --provider openai
```

Recommended models:

* `gpt-4o-mini` - Fast, cost-effective (default)
* `gpt-4o` - Better accuracy
* `gpt-4-turbo` - High quality

### Ollama (Local)

```bash
# Start Ollama server (if not running)
ollama serve

# Use default host
./speaker-llm analyze transcript.json --provider ollama

# Or specify custom host
export OLLAMA_HOST="http://192.168.1.100:11434"
./speaker-llm analyze transcript.json --provider ollama
```

Recommended models:

* `llama3.2` - Good balance (default)
* `llama3.2:70b` - Better accuracy
* `mistral` - Fast alternative

## Detection Patterns

The LLM is prompted to look for these name revelation patterns:

| Pattern | Example |
|---------|---------|
| Direct address | "Alice, can you explain..." |
| Self-reference | "I'm Bob and I think..." |
| Third-person mention | "As Carol mentioned earlier..." |
| Introduction | "Hi, this is Dave from sales" |
| Role-based | "Our host John will..." |
| Conversation flow | "[S1] says something" then "[S2]: Thanks, Alice" |

## Caching

Responses are cached based on transcript content hash to avoid redundant API calls.

```
~/.cache/speaker-llm/
└── abc123def456.json    # Cached response
```

Cache behavior:

* Cache is keyed by transcript content hash (first 16 chars of SHA256)
* Separate cache entries for `analyze` and `detect-names` modes
* Use `--no-cache` to bypass cache
* Use `clear-cache` command to remove all cached responses

Custom cache directory:

```bash
export SPEAKER_LLM_CACHE_DIR="/path/to/cache"
```

## Integration with speaker-assign

`speaker-llm` is designed to work with `speaker-assign` for multi-signal speaker identification:

```bash
# speaker-assign calls speaker-llm internally when --use-llm is specified
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --use-llm

# speaker-llm output format is compatible with speaker-assign signal input
```

The `detections` array format matches what `speaker-assign` expects:

```json
{
  "detections": [
    {
      "speaker_label": "S1",
      "detected_name": "alice",
      "confidence": 0.85,
      "evidence": ["..."]
    }
  ]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `SPEAKER_LLM_CACHE_DIR` | `~/.cache/speaker-llm` | Response cache directory |

## Transcript Formats

Supported transcript formats:

### AssemblyAI Format

```json
{
  "utterances": [
    {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello, I'm Alice"},
    {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi Alice, Bob here"}
  ]
}
```

### Speechmatics Format

```json
{
  "results": [
    {"start_time": 1.0, "end_time": 2.0, "speaker": "S1", "alternatives": [{"content": "Hello"}]},
    {"start_time": 2.5, "end_time": 3.5, "speaker": "S2", "alternatives": [{"content": "Hi there"}]}
  ]
}
```

## Related Tools

* [`speaker-assign`](./speaker-assign.README.md) - Multi-signal speaker name assignment
* [`speaker-catalog`](./speaker-catalog.README.md) - Recording inventory management
* [`speaker-review`](./speaker-review.README.md) - Manual review and correction
* [`speaker_detection`](./speaker_detection.README.md) - Voice embedding-based identification
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics STT with diarization
* [`stt_assemblyai.py`](./stt_assemblyai.README.md) - AssemblyAI transcription
