# speaker-assign - Multi-Signal Speaker Name Assignment

Combine embedding matches, LLM analysis, and context hints to assign speaker names to transcript labels (S1, S2, A, B, etc.). Part of the speaker-* tool ecosystem.

## Overview

`speaker-assign` provides intelligent speaker name assignment by combining multiple signal sources with weighted scoring. It maps anonymous transcript speaker labels to known speaker identities using:

* Embedding-based voice matching via `speaker_detection identify`
* LLM conversation analysis for name detection
* Context hints from expected speaker lists
* Cross-backend agreement when multiple transcripts exist

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Transcript    │    │   speaker-      │    │   Speaker       │
│   (S1, S2...)   │───>│   assign        │<───│   Embeddings    │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Weighted Signals   │
                    │  ├── embeddings 0.4 │
                    │  ├── llm        0.3 │
                    │  ├── context    0.2 │
                    │  └── agreement  0.1 │
                    └──────────┬──────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │  $SPEAKERS_EMBEDDINGS│
                    │  _DIR/assignments/   │
                    │  └── abc123.yaml     │
                    └──────────────────────┘
```

## Installation

No installation required. The script uses `uv run` for dependency management.

```bash
# Run directly - dependencies are auto-installed
./speaker-assign assign meeting.mp3 --transcript meeting.json

# Requirements:
# - Python 3.11+
# - b3sum (optional, falls back to SHA256)
# - speaker_detection (for embedding matches)
# - speaker-llm (optional, for LLM analysis)
```

## Quick Start

```bash
# 1. Assign speakers using embedding matches
./speaker-assign assign meeting.mp3 \
    --transcript meeting.speechmatics.json \
    --use-embeddings

# 2. Assign with LLM analysis and context
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --use-llm \
    --context team-standup

# 3. Dry run to preview assignments
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --dry-run

# 4. Show current assignments
./speaker-assign show meeting.mp3

# 5. Clear assignments
./speaker-assign clear meeting.mp3
```

## Commands

### `assign` - Assign speaker names to transcript labels

```bash
./speaker-assign assign <audio> --transcript FILE [OPTIONS]

Options:
  -t, --transcript FILE       Path to transcript JSON file (required)
  -e, --use-embeddings        Use speaker_detection identify for voice matching
  --min-trust LEVEL           Minimum embedding trust level: high, medium, low (default: low)
  -l, --use-llm               Use LLM conversation analysis for name detection
  -c, --context NAME          Speaker context for name resolution
  --expected-speakers IDS     Comma-separated expected speaker IDs
  --tags TAGS                 Filter speakers by tags (for embedding lookup)
  --threshold FLOAT           Minimum confidence for assignment (default: 0.3)
  -o, --output FILE           Output file for assignment results
  -f, --format FORMAT         Output format: text (default), json
  -n, --dry-run               Show assignments without saving
  -v, --verbose               Verbose output showing signal collection
  -q, --quiet                 Suppress non-essential output
```

Examples:

```bash
# Basic assignment with embeddings
./speaker-assign assign meeting.mp3 \
    --transcript meeting.speechmatics.json \
    --use-embeddings

# Full signal collection with context
./speaker-assign assign interview.wav \
    --transcript interview.json \
    --use-embeddings \
    --use-llm \
    --context podcast-ep42 \
    --expected-speakers alice,bob,guest

# High-trust embeddings only
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --min-trust high

# Verbose dry run
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --verbose \
    --dry-run

# JSON output for scripting
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --format json

# Custom confidence threshold
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --threshold 0.5

# Filter by speaker tags
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --tags team-alpha,engineering
```

### `show` - Display current assignments

```bash
./speaker-assign show <audio>

Options:
  -f, --format FORMAT    Output format: text (default), json, yaml
```

The `<audio>` argument can be:

* Full path to audio file
* b3sum prefix (at least 6 characters)

Examples:

```bash
# Show assignments by path
./speaker-assign show meeting.mp3

# Show by b3sum prefix
./speaker-assign show abc123

# JSON output
./speaker-assign show meeting.mp3 --format json

# YAML output
./speaker-assign show meeting.mp3 --format yaml
```

Output includes:

* Recording identifier
* Context and method used
* Assignment timestamp
* Threshold and trust settings
* Speaker mappings with confidence and scores
* Contributing signals
* Alternative candidates

### `clear` - Remove assignments

```bash
./speaker-assign clear <audio> [--force]

Options:
  -f, --force    Skip confirmation prompt
  -q, --quiet    Suppress output
```

Examples:

```bash
# Clear with confirmation
./speaker-assign clear meeting.mp3

# Clear by b3sum prefix
./speaker-assign clear abc123

# Force clear (no confirmation)
./speaker-assign clear meeting.mp3 --force
```

## Signal Types

Speaker assignment combines multiple signal sources using weighted scoring:

| Signal Type | Weight | Description |
|-------------|--------|-------------|
| `embedding_match` | 0.4 | Voice embedding similarity from speaker_detection identify |
| `llm_name_detection` | 0.3 | Names detected via LLM analysis of conversation |
| `context_expected` | 0.2 | Speakers in the expected_speakers list |
| `cross_backend_agreement` | 0.1 | Agreement between multiple STT backend transcripts |

### Embedding Match (0.4 base weight)

The most reliable signal. Calls `speaker_detection identify` to compare voice characteristics against enrolled speaker profiles. Weight is further adjusted by trust level (see Trust Multipliers below).

### LLM Name Detection (0.3 weight)

Analyzes transcript content for speaker name mentions. Looks for:

* Self-introductions ("Hi, I'm Alice")
* Name references ("Bob, what do you think?")
* Contextual name usage

Requires `speaker-llm` tool and `--use-llm` flag.

### Context Expected (0.2 weight)

Speakers listed in the expected_speakers list receive a base signal. This helps when you know who participated but need to match labels to identities.

### Cross Backend Agreement (0.1 weight)

When multiple STT backends produce transcripts, agreement between them on speaker count and patterns provides additional confidence.

## Trust Multipliers

Embedding match weights are multiplied by trust level:

| Trust Level | Multiplier | Effective Weight |
|-------------|------------|------------------|
| `high` | 1.0 | 0.40 |
| `medium` | 0.7 | 0.28 |
| `low` | 0.4 | 0.16 |
| `invalidated` | 0.0 | 0.00 |
| `unknown` | 0.5 | 0.20 |

Trust levels come from speaker profile embeddings:

* **high** - Recent, verified enrollment with good audio quality
* **medium** - Valid enrollment but older or lower quality
* **low** - Weak match or degraded samples
* **invalidated** - Explicitly marked as unreliable

Use `--min-trust` to set minimum trust level for embedding matches:

```bash
# Only use high-trust embeddings
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings \
    --min-trust high
```

## Confidence Levels

Combined weighted scores map to confidence levels:

| Level | Score Threshold | Meaning |
|-------|-----------------|---------|
| `high` | >= 0.7 | Strong confidence, multiple agreeing signals |
| `medium` | >= 0.4 | Moderate confidence, some signal agreement |
| `low` | >= 0.2 | Weak confidence, limited signal support |
| `unassigned` | < 0.2 or below threshold | Insufficient evidence |

The `--threshold` option sets the minimum score for assignment (default: 0.3). Speakers scoring below threshold are marked "unassigned" but candidates are still listed.

## Assignment Schema

Assignments are stored as YAML files in `$SPEAKERS_EMBEDDINGS_DIR/assignments/`:

```yaml
schema_version: 1

recording_b3sum: abc123def456789...
transcript_path: /path/to/meeting.speechmatics.json
assigned_at: 2026-01-17T10:00:00Z
method: speaker-assign-v1.0.0
context: team-standup
min_trust: low
threshold: 0.3

mappings:
  S1:
    speaker_id: alice
    confidence: high
    score: 0.72
    signals:
      - type: embedding_match
        score: 0.85
        trust_level: high
        backend: pyannote
      - type: context_expected
        score: 0.5
        context: team-standup
        reason: in expected_speakers list
    candidates:
      - speaker_id: carol
        score: 0.31
      - speaker_id: bob
        score: 0.28

  S2:
    speaker_id: bob
    confidence: medium
    score: 0.54
    signals:
      - type: embedding_match
        score: 0.67
        trust_level: medium
        backend: pyannote
      - type: llm_name_detection
        score: 0.6
        detected_name: Bob
        evidence:
          - "Thanks Bob, that's helpful"

  S3:
    speaker_id: null
    confidence: unassigned
    score: 0.18
    signals: []
    candidates:
      - speaker_id: guest-john
        score: 0.18
      - speaker_id: carol
        score: 0.12
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Root storage directory |

## Storage

Assignment files are stored alongside other speaker data:

```
$SPEAKERS_EMBEDDINGS_DIR/
├── assignments/                # Speaker label assignments (speaker-assign)
│   ├── abc123def456.yaml       # Named by recording b3sum
│   └── xyz789...yaml
├── catalog/                    # Recording catalog (speaker-catalog)
├── db/                         # Speaker profiles (speaker_detection)
├── embeddings/                 # Embedding vectors (speaker_detection)
└── samples/                    # Voice samples (speaker_samples)
```

## Integration with Other Tools

`speaker-assign` integrates with the speaker-* ecosystem:

```bash
# 1. Add recording to catalog with expected speakers
./speaker-catalog add meeting.mp3 --context team-standup
./speaker-catalog set-context meeting.mp3 --expected-speakers alice,bob,carol

# 2. Transcribe with STT
./stt_speechmatics.py meeting.mp3 -o meeting.speechmatics.json

# 3. Register transcript in catalog
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.speechmatics.json

# 4. Assign speaker names (uses expected_speakers from catalog)
./speaker-assign assign meeting.mp3 \
    --transcript meeting.speechmatics.json \
    --use-embeddings \
    --use-llm

# 5. Review assignments
./speaker-assign show meeting.mp3

# 6. Manual review and corrections
./speaker-review meeting.mp3

# 7. Check status
./speaker-catalog status meeting.mp3
# Output: assigned
```

### Integration with speaker-catalog

When a recording is in the catalog, `speaker-assign` automatically:

* Retrieves `context` name from catalog entry
* Retrieves `expected_speakers` list for context signals
* Saves assignments in the standard location

### Integration with speaker-review

After automated assignment:

```bash
# Review assignments interactively
./speaker-review meeting.mp3

# Shows assignments with confidence indicators
# Allows manual corrections
# Updates review status in catalog
```

## Related Tools

* [`speaker-catalog`](./speaker-catalog.README.md) - Recording inventory and processing state management
* [`speaker-review`](./speaker-review.README.md) - Manual review and correction of assignments
* [`speaker_detection`](./speaker_detection.README.md) - Speaker profile and embedding management
* [`speaker_samples`](./speaker_samples.README.md) - Voice sample extraction with provenance
* [`speaker_segments`](./speaker_segments.README.md) - Extract segment timestamps from transcripts
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics STT with speaker diarization
* [`stt_assemblyai.py`](./stt_assemblyai.README.md) - AssemblyAI transcription
