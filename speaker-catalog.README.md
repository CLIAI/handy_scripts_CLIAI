# speaker-catalog - Recording Inventory and Processing State Management

Track recordings, their processing state, transcripts, and review progress. Part of the speaker-* tool ecosystem.

## Overview

`speaker-catalog` provides centralized management for audio recordings in the speaker identification pipeline. It tracks:

* Recording inventory with content-addressable b3sum identifiers
* Processing state progression (unprocessed -> transcribed -> assigned -> reviewed -> complete)
* Context information including expected speakers and tags
* Transcript registrations from multiple STT backends

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Files   │    │   speaker-      │    │  STT Backends   │
│  (recordings)   │───>│   catalog       │<───│  (transcripts)  │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                                v
                    ┌──────────────────────┐
                    │  $SPEAKERS_EMBEDDINGS│
                    │  _DIR/catalog/       │
                    │  ├── abc123.yaml     │
                    │  ├── def456.yaml     │
                    │  └── ...             │
                    └──────────────────────┘
```

## Installation

No installation required. The script uses `uv run` for dependency management.

```bash
# Run directly - dependencies are auto-installed
./speaker-catalog list

# Requirements:
# - Python 3.11+
# - ffprobe (for duration detection, usually bundled with ffmpeg)
# - b3sum (optional, falls back to SHA256)
# - jq (for query command)
```

## Quick Start

```bash
# 1. Add a recording to the catalog
./speaker-catalog add meeting.mp3 --context team-standup --tags work,weekly

# 2. List all recordings
./speaker-catalog list

# 3. Register a transcript from STT
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.speechmatics.json

# 4. Check status of a recording
./speaker-catalog status meeting.mp3

# 5. Show detailed info
./speaker-catalog show meeting.mp3
```

## Commands

### `add` - Add recording to catalog

```bash
./speaker-catalog add <audio> [--context NAME] [--tags TAG,...]

Options:
  -c, --context     Context name (e.g., 'team-standup', 'podcast-ep42')
  -t, --tags        Comma-separated tags for filtering
  -f, --force       Overwrite if already exists
  -q, --quiet       Suppress output
```

Examples:

```bash
# Basic add
./speaker-catalog add meeting.mp3

# Add with context and tags
./speaker-catalog add interview.wav --context podcast --tags guest-john,episode-15

# Force update existing entry
./speaker-catalog add meeting.mp3 --context updated-context --force
```

### `list` - List recordings in catalog

```bash
./speaker-catalog list [--status STATUS] [--context NAME] [--needs-review]

Options:
  -s, --status        Filter by status (unprocessed, transcribed, assigned, reviewed, complete)
  -c, --context       Filter by context name
  -r, --needs-review  Show only recordings needing review
  -f, --format        Output format: table (default), json, ids, paths
  --limit             Maximum results to show
  --offset            Skip first N results
```

Examples:

```bash
# List all recordings
./speaker-catalog list

# Filter by status
./speaker-catalog list --status transcribed

# Filter by context
./speaker-catalog list --context podcast

# Show recordings needing review
./speaker-catalog list --needs-review

# JSON output for scripting
./speaker-catalog list --format json

# Get just paths for piping
./speaker-catalog list --format paths | xargs -I {} ls -la {}

# Pagination
./speaker-catalog list --limit 10
./speaker-catalog list --limit 10 --offset 20
```

### `show` - Show detailed recording info

```bash
./speaker-catalog show <audio>

Options:
  -f, --format    Output format: text (default), json, yaml
```

The `<audio>` argument can be:

* Full path to audio file
* b3sum prefix (at least 6 characters)

Examples:

```bash
# Show by path
./speaker-catalog show meeting.mp3

# Show by b3sum prefix
./speaker-catalog show abc123

# JSON output
./speaker-catalog show meeting.mp3 --format json

# YAML output
./speaker-catalog show meeting.mp3 --format yaml
```

### `status` - Quick status check

```bash
./speaker-catalog status <audio>

Options:
  -f, --format    Output format: text (default), json
```

Returns one of: `unprocessed`, `transcribed`, `assigned`, `reviewed`, `complete`

```bash
# Get status
./speaker-catalog status meeting.mp3
# Output: transcribed

# JSON output for scripting
./speaker-catalog status meeting.mp3 --format json
# Output: {"status": "transcribed"}
```

### `register-transcript` - Register transcript from STT backend

```bash
./speaker-catalog register-transcript <audio> --backend NAME --transcript FILE

Options:
  -b, --backend       STT backend name (required): speechmatics, assemblyai, etc.
  -t, --transcript    Path to transcript JSON file (required)
  --version           Backend version string
  --tool-version      Version of tool that created transcript
  -f, --force         Replace existing transcript for this backend
  -q, --quiet         Suppress output
```

Examples:

```bash
# Register Speechmatics transcript
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.speechmatics.json

# Register AssemblyAI transcript
./speaker-catalog register-transcript meeting.mp3 \
    --backend assemblyai \
    --transcript meeting.assemblyai.json

# With version info
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.speechmatics.json \
    --version "speechmatics-v2" \
    --tool-version "stt_speechmatics.py-1.2.0"

# Replace existing transcript
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting-v2.speechmatics.json \
    --force
```

### `set-context` - Update context and expected speakers

```bash
./speaker-catalog set-context <audio> [--context NAME] [--expected-speakers ID,...]

Options:
  -c, --context            Context name
  -e, --expected-speakers  Comma-separated expected speaker IDs
  -t, --tags               Comma-separated tags to add
  --remove-tags            Comma-separated tags to remove
  -q, --quiet              Suppress output
```

Examples:

```bash
# Set context name
./speaker-catalog set-context meeting.mp3 --context team-standup

# Set expected speakers (helps with assignment verification)
./speaker-catalog set-context meeting.mp3 --expected-speakers alice,bob,carol

# Add tags
./speaker-catalog set-context meeting.mp3 --tags important,q4-review

# Remove tags
./speaker-catalog set-context meeting.mp3 --remove-tags draft,temp

# Combined update
./speaker-catalog set-context meeting.mp3 \
    --context weekly-sync \
    --expected-speakers alice,bob \
    --tags recurring
```

### `remove` - Remove from catalog

```bash
./speaker-catalog remove <audio> [--force]

Options:
  -f, --force    Skip confirmation prompt
  -q, --quiet    Suppress output
```

Examples:

```bash
# Remove with confirmation
./speaker-catalog remove meeting.mp3

# Remove by b3sum prefix
./speaker-catalog remove abc123

# Force remove (no confirmation)
./speaker-catalog remove meeting.mp3 --force
```

### `query` - Query with jq expressions

```bash
./speaker-catalog query '<jq-expression>'
```

Query all catalog entries as a JSON array using jq expressions.

Examples:

```bash
# List all b3sum identifiers
./speaker-catalog query '.[].recording.b3sum'

# Get recordings with specific status
./speaker-catalog query '.[] | select(.status == "transcribed")'

# Count recordings by status
./speaker-catalog query 'group_by(.status) | map({status: .[0].status, count: length})'

# Find recordings with specific context
./speaker-catalog query '.[] | select(.context.name == "podcast") | .recording.path'

# Get recordings with more than 2 speakers detected
./speaker-catalog query '.[] | select(.transcriptions[0].speakers_detected > 2)'
```

## Status Values

Recordings progress through these status values:

| Status | Description |
|--------|-------------|
| `unprocessed` | Recording added but no transcript registered |
| `transcribed` | At least one transcript registered |
| `assigned` | Speaker labels assigned (assignments file exists) |
| `reviewed` | Partial review completed |
| `complete` | Fully reviewed and verified |

Status is automatically computed based on:

* Presence of transcripts in the catalog entry
* Existence of assignments file in `$SPEAKERS_EMBEDDINGS_DIR/assignments/`
* Review status field in the entry

## Storage

Catalog entries are stored as YAML files in `$SPEAKERS_EMBEDDINGS_DIR/catalog/`:

```
$SPEAKERS_EMBEDDINGS_DIR/
├── catalog/                    # Recording catalog (speaker-catalog)
│   ├── abc123def456.yaml       # Named by b3sum
│   └── xyz789...yaml
├── assignments/                # Speaker label assignments
│   └── abc123def456.yaml
├── db/                         # Speaker profiles (speaker_detection)
├── embeddings/                 # Embedding vectors (speaker_detection)
└── samples/                    # Voice samples (speaker_samples)
```

## Catalog Entry Schema

Each catalog entry is a YAML file with this structure:

```yaml
schema_version: 1

recording:
  path: /absolute/path/to/meeting.mp3
  b3sum: abc123def456789...       # First 32 chars of Blake3 hash
  duration_sec: 1847.5            # Duration in seconds (from ffprobe)
  discovered_at: 2026-01-17T10:00:00Z

context:
  name: team-standup              # Human-readable context identifier
  expected_speakers:              # Speaker IDs expected in this recording
    - alice
    - bob
    - carol
  tags:                           # Arbitrary tags for filtering
    - work
    - weekly
    - q1-2026

transcriptions:                   # Registered transcripts (multiple backends)
  - backend: speechmatics
    version: speechmatics-v2
    transcript_path: /path/to/meeting.speechmatics.json
    processed_at: 2026-01-17T10:30:00Z
    tool_version: stt_speechmatics.py-1.2.0
    speakers_detected: 3
  - backend: assemblyai
    version: assemblyai-v1
    transcript_path: /path/to/meeting.assemblyai.json
    processed_at: 2026-01-17T11:00:00Z
    tool_version: stt_assemblyai.py-1.0.0
    speakers_detected: 3

review:
  status: partial                 # none | partial | complete
  reviewed_at: 2026-01-17T14:00:00Z
  notes: "Need to verify S3 assignment"

status: assigned                  # Computed: unprocessed | transcribed | assigned | reviewed | complete
updated_at: 2026-01-17T14:00:00Z
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Root storage directory |

## Integration with Other Tools

`speaker-catalog` integrates with the speaker-* ecosystem:

```bash
# 1. Add recording to catalog
./speaker-catalog add meeting.mp3 --context team-standup

# 2. Transcribe with STT
./stt_speechmatics.py meeting.mp3 -o meeting.speechmatics.json

# 3. Register transcript
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.speechmatics.json

# 4. Set expected speakers
./speaker-catalog set-context meeting.mp3 \
    --expected-speakers alice,bob,carol

# 5. Extract speaker segments
./speaker_segments meeting.speechmatics.json --speaker S1

# 6. Extract voice samples
./speaker_samples extract meeting.mp3 \
    -t meeting.speechmatics.json \
    -l S1 -s alice

# 7. Enroll speaker
./speaker_detection enroll alice meeting.mp3 \
    --from-transcript meeting.speechmatics.json \
    --speaker-label S1

# 8. Check catalog status
./speaker-catalog status meeting.mp3
# Output: assigned
```

## Related Tools

* [`speaker_detection`](./speaker_detection.README.md) - Speaker profile and embedding management
* [`speaker_samples`](./speaker_samples.README.md) - Voice sample extraction with provenance
* [`speaker_segments`](./speaker_segments.README.md) - Extract segment timestamps from transcripts
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics STT with speaker diarization
* [`stt_assemblyai.py`](./stt_assemblyai.README.md) - AssemblyAI transcription
