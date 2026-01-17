# speaker-process - Batch Recording Processing Orchestrator

Orchestrate batch processing of recordings through the speaker-* pipeline: transcription, speaker assignment, and catalog management. Part of the speaker-* tool ecosystem.

## Overview

`speaker-process` automates the end-to-end workflow for processing audio recordings:

1. Add recording to catalog (if not exists)
2. Transcribe with each configured STT backend
3. Register transcripts in catalog
4. Run speaker assignment
5. Update catalog status

It supports both immediate processing and queue-based batch processing with parallel execution.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Files   │    │   speaker-      │    │  STT Backends   │
│  (recordings)   │───>│   process       │───>│  (speechmatics) │
└─────────────────┘    └────────┬────────┘    │  (assemblyai)   │
                                │              └─────────────────┘
                                v
                    ┌──────────────────────┐
                    │  Pipeline Steps:     │
                    │  1. catalog add      │
                    │  2. transcribe       │
                    │  3. register         │
                    │  4. speaker-assign   │
                    └──────────┬───────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │  $SPEAKERS_EMBEDDINGS│
                    │  _DIR/               │
                    │  ├── catalog/        │
                    │  ├── assignments/    │
                    │  └── process_queue/  │
                    └──────────────────────┘
```

## Installation

No installation required. The script uses `uv run` for dependency management.

```bash
# Run directly - dependencies are auto-installed
./speaker-process process meeting.mp3

# Requirements:
# - Python 3.11+
# - STT tools: stt_speechmatics.py, stt_assemblyai.py
# - speaker-catalog (for catalog management)
# - speaker-assign (for speaker assignment)
# - b3sum (optional, falls back to SHA256)
```

## Quick Start

```bash
# 1. Process a single recording
./speaker-process process meeting.mp3

# 2. Process all recordings in a directory
./speaker-process process ./recordings/ --recursive

# 3. Queue recordings for later processing
./speaker-process queue ./recordings/ --context team-standup

# 4. Run queued processing with parallel jobs
./speaker-process run --parallel 4

# 5. Check queue status
./speaker-process status
```

## Commands

### `process` - Process recording(s) immediately

```bash
./speaker-process process <audio|directory> [OPTIONS]

Options:
  -b, --backend BACKENDS    Comma-separated backends (default: speechmatics,assemblyai)
  -c, --context NAME        Context name for new recordings
  -p, --parallel N          Number of parallel jobs (default: 4)
  -o, --output-dir DIR      Output directory for transcripts
  -r, --recursive           Recursively scan directories
  -s, --skip-existing       Skip already processed recordings
  -n, --dry-run             Show what would run without executing
  -q, --quiet               Suppress non-essential output
```

Examples:

```bash
# Process a single recording
./speaker-process process meeting.mp3

# Process all recordings in a directory
./speaker-process process ./recordings/

# Recursive processing with context
./speaker-process process ./project/ --recursive --context project-alpha

# Use only Speechmatics backend
./speaker-process process meeting.mp3 --backend speechmatics

# Dry run to preview processing
./speaker-process process ./recordings/ --dry-run

# Custom output directory
./speaker-process process meeting.mp3 --output-dir ./transcripts/

# Parallel processing with 8 workers
./speaker-process process ./large_dataset/ --parallel 8 --recursive
```

### `queue` - Add recording(s) to processing queue

```bash
./speaker-process queue <audio|directory> [OPTIONS]

Options:
  -b, --backend BACKENDS    Comma-separated backends (default: speechmatics,assemblyai)
  -c, --context NAME        Context name for new recordings
  -r, --recursive           Recursively scan directories
  -q, --quiet               Suppress non-essential output
```

Examples:

```bash
# Queue a single recording
./speaker-process queue interview.mp3

# Queue a directory
./speaker-process queue ./unprocessed/

# Queue with context and specific backends
./speaker-process queue ./meetings/ \
    --context weekly-standup \
    --backend speechmatics,assemblyai \
    --recursive

# Queue silently (for scripts)
./speaker-process queue ./files/ --quiet
```

### `status` - Show processing queue status

```bash
./speaker-process status [OPTIONS]

Options:
  -f, --format FORMAT    Output format: text (default), json
  -v, --verbose          Show detailed item info
```

Examples:

```bash
# Show queue summary
./speaker-process status

# Verbose output with item details
./speaker-process status --verbose

# JSON output for scripting
./speaker-process status --format json
```

Output includes:

* Total items in queue
* Count by status (pending, processing, completed, failed, skipped)
* Individual item details (with `--verbose`)

### `run` - Run processing on queued items

```bash
./speaker-process run [OPTIONS]

Options:
  -l, --limit N          Maximum number of items to process
  -p, --parallel N       Number of parallel jobs (default: 4)
  -o, --output-dir DIR   Output directory for transcripts
  -s, --skip-existing    Skip already processed recordings
  -n, --dry-run          Show what would run without executing
  -q, --quiet            Suppress non-essential output
```

Examples:

```bash
# Process all queued items
./speaker-process run

# Process with 8 parallel workers
./speaker-process run --parallel 8

# Process only next 10 items
./speaker-process run --limit 10

# Dry run to see what would be processed
./speaker-process run --dry-run

# Custom output directory
./speaker-process run --output-dir ./output/
```

### `clear-queue` - Clear the processing queue

```bash
./speaker-process clear-queue [OPTIONS]

Options:
  -s, --status STATUS    Only clear items with this status
  -f, --force            Skip confirmation
  -q, --quiet            Suppress non-essential output
```

Examples:

```bash
# Clear all items (with confirmation)
./speaker-process clear-queue

# Clear only failed items
./speaker-process clear-queue --status failed

# Clear completed items without confirmation
./speaker-process clear-queue --status completed --force
```

## Processing Workflow

When processing a recording, `speaker-process` executes these steps:

### Step 1: Catalog Registration

```
audio.mp3 ──> speaker-catalog add ──> catalog/{b3sum}.yaml
```

Adds the recording to the catalog if not already present. Uses `speaker-catalog add` with optional context.

### Step 2: Transcription

```
audio.mp3 ──> stt_speechmatics.py ──> audio.speechmatics.json
audio.mp3 ──> stt_assemblyai.py ──> audio.assemblyai.json
```

Runs each configured STT backend. Transcripts are saved alongside the audio file or in the specified output directory.

### Step 3: Transcript Registration

```
audio.speechmatics.json ──> speaker-catalog register-transcript
```

Registers each transcript in the catalog, linking it to the recording.

### Step 4: Speaker Assignment

```
audio.mp3 + transcript ──> speaker-assign ──> assignments/{b3sum}.yaml
```

Runs speaker assignment using the first successful transcript. Uses embedding matching and context hints to map speaker labels to known identities.

## Queue Storage

Queue data is stored in `$SPEAKERS_EMBEDDINGS_DIR/process_queue.yaml`:

```yaml
schema_version: 1
updated_at: 2026-01-17T10:00:00Z

items:
  - audio_path: /path/to/meeting.mp3
    b3sum: abc123def456...
    status: pending
    context: team-standup
    backends:
      - speechmatics
      - assemblyai
    queued_at: 2026-01-17T09:00:00Z
    started_at: null
    completed_at: null
    error: null
    results: {}

  - audio_path: /path/to/interview.mp3
    b3sum: xyz789...
    status: completed
    context: podcast
    backends:
      - speechmatics
    queued_at: 2026-01-17T08:00:00Z
    started_at: 2026-01-17T09:30:00Z
    completed_at: 2026-01-17T09:35:00Z
    error: null
    results:
      transcripts:
        speechmatics: /path/to/interview.speechmatics.json
```

### Queue Item Statuses

| Status | Description |
|--------|-------------|
| `pending` | Queued, waiting to be processed |
| `processing` | Currently being processed |
| `completed` | Successfully processed |
| `failed` | Processing failed (see error field) |
| `skipped` | Skipped (already processed or excluded) |

## Supported STT Backends

| Backend | Tool | Notes |
|---------|------|-------|
| `speechmatics` | `stt_speechmatics.py` | High-quality diarization |
| `assemblyai` | `stt_assemblyai.py` | Good speaker detection |
| `openai` | `stt_openai.py` | Fast, no diarization |
| `whisper` | `stt_openai_OR_local_whisper_cli.py` | Local or API |

Default backends: `speechmatics,assemblyai`

## Supported Audio Formats

* `.wav` - Waveform Audio
* `.mp3` - MPEG Audio
* `.flac` - Free Lossless Audio Codec
* `.m4a` - MPEG-4 Audio
* `.ogg` - Ogg Vorbis
* `.opus` - Opus Audio
* `.aac` - Advanced Audio Coding
* `.wma` - Windows Media Audio

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Root storage directory |

## Storage

Processing artifacts are stored alongside other speaker data:

```
$SPEAKERS_EMBEDDINGS_DIR/
├── process_queue.yaml         # Processing queue (speaker-process)
├── catalog/                   # Recording catalog (speaker-catalog)
│   └── {b3sum}.yaml
├── assignments/               # Speaker assignments (speaker-assign)
│   └── {b3sum}.yaml
├── db/                        # Speaker profiles (speaker_detection)
├── embeddings/                # Embedding vectors (speaker_detection)
└── samples/                   # Voice samples (speaker_samples)
```

## Integration with Other Tools

`speaker-process` orchestrates the full speaker-* pipeline:

```bash
# Typical workflow orchestrated by speaker-process:

# 1. Manual: Add recording to catalog (or let speaker-process do it)
./speaker-catalog add meeting.mp3 --context team-standup

# 2. speaker-process handles:
#    - Transcription with STT backends
#    - Transcript registration in catalog
#    - Speaker assignment

./speaker-process process meeting.mp3 --context team-standup

# 3. Manual: Review assignments
./speaker-review meeting.mp3

# 4. Check status
./speaker-catalog status meeting.mp3
# Output: assigned
```

### Batch Processing Example

```bash
# Queue all recordings from a project directory
./speaker-process queue ./project_recordings/ \
    --context project-alpha \
    --recursive

# Check queue status
./speaker-process status

# Run processing overnight
nohup ./speaker-process run --parallel 8 > process.log 2>&1 &

# Check progress
./speaker-process status --verbose
```

### CI/CD Integration

```bash
# Process new recordings in a pipeline
./speaker-process process $INPUT_FILE \
    --backend speechmatics \
    --output-dir $OUTPUT_DIR \
    --quiet

# Check exit code for success/failure
if [ $? -eq 0 ]; then
    echo "Processing succeeded"
else
    echo "Processing failed"
fi
```

## Related Tools

* [`speaker-catalog`](./speaker-catalog.README.md) - Recording inventory and processing state management
* [`speaker-assign`](./speaker-assign.README.md) - Multi-signal speaker name assignment
* [`speaker-review`](./speaker-review.README.md) - Manual review and correction of assignments
* [`speaker_detection`](./speaker_detection.README.md) - Speaker profile and embedding management
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics STT with speaker diarization
* [`stt_assemblyai.py`](./stt_assemblyai.README.md) - AssemblyAI transcription
