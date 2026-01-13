# speaker_samples - Voice Sample Extraction Tool

Extract voice samples from audio files using transcript timing information. Store samples with full provenance metadata for speaker enrollment pipelines.

## Overview

`speaker_samples` bridges the gap between STT transcription and speaker enrollment. It extracts audio segments for specific speakers from transcripts, stores them with complete provenance metadata, and enables downstream enrollment workflows.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio File    │    │   Transcript    │    │    speaker_     │
│  (meeting.mp3)  │───▶│  (*.json)       │───▶│    samples      │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
                                           ┌──────────────────────┐
                                           │  $SPEAKERS_EMBEDDINGS│
                                           │  _DIR/samples/       │
                                           │  ├── alice/          │
                                           │  │   ├── sample-001  │
                                           │  │   └── *.meta.yaml │
                                           │  └── bob/            │
                                           └──────────────────────┘
```

## Installation

```bash
# No installation needed - just ensure ffmpeg is available
which ffmpeg  # Required for audio extraction

# Optional: install PyYAML for YAML metadata output
pip install pyyaml
```

## Quick Start

```bash
# 1. Inspect transcript to find speaker labels
./speaker_samples speakers meeting.speechmatics.json
# Output: Format: speechmatics
#         Speakers: S1, S2, S3

# 2. Preview segments for a speaker
./speaker_samples segments -t meeting.speechmatics.json -l S1 | head

# 3. Extract samples
./speaker_samples extract meeting.mp3 \
    -t meeting.speechmatics.json \
    -l S1 \
    -s alice

# 4. List extracted samples
./speaker_samples list alice

# 5. View sample metadata
./speaker_samples info alice sample-001
```

## Commands

### `extract` - Extract samples from audio

```bash
./speaker_samples extract <audio> -t <transcript> -l <label> -s <speaker_id>

Options:
  -t, --transcript    Transcript JSON file (required)
  -l, --speaker-label Speaker label in transcript (required)
  -s, --speaker-id    Target speaker ID for storage (required)
  --format           Output format: mp3, wav (default: mp3)
  --min-duration     Minimum segment duration in seconds (default: 0.5)
  --max-gap          Max gap to merge adjacent segments (default: 1.0s)
  --max-segments     Maximum number of segments to extract
  --max-duration     Maximum total duration to extract
  -n, --dry-run      Show what would be extracted
  -v, --verbose      Verbose output
```

Example:

```bash
# Extract up to 30 seconds of Alice's speech
./speaker_samples extract meeting.mp3 \
    -t meeting.speechmatics.json \
    -l S1 \
    -s alice \
    --max-duration 30 \
    -v
```

### `segments` - Output segment times as JSONL

Outputs segment timing as JSONL for piping to other tools.

```bash
./speaker_samples segments -t <transcript> -l <label> [--audio file] [--speaker-id id]

# Output format (one JSON object per line):
{"speaker_id": "alice", "audio": "meeting.mp3", "start": 10.5, "end": 25.3, "text": "..."}
```

Example - pipe to enrollment:

```bash
./speaker_samples segments -t meeting.json -l S1 -s alice | \
    while read line; do
        start=$(echo "$line" | jq -r .start)
        end=$(echo "$line" | jq -r .end)
        echo "Segment: $start - $end"
    done
```

### `list` - List stored samples

```bash
# List all speakers with sample counts
./speaker_samples list

# List samples for specific speaker
./speaker_samples list alice

# JSON output
./speaker_samples list --format json
./speaker_samples list alice --format json
```

### `info` - Show sample metadata

```bash
./speaker_samples info <speaker_id> <sample_id>

# JSON format
./speaker_samples info alice sample-001 --format json
```

### `remove` - Remove samples

```bash
# Remove specific sample
./speaker_samples remove alice sample-001

# Remove all samples for speaker
./speaker_samples remove alice --all

# Remove samples from specific source audio
./speaker_samples remove alice --source "meeting.mp3"

# Force (skip confirmation)
./speaker_samples remove alice --all -f
```

### `speakers` - List speakers in transcript

```bash
./speaker_samples speakers <transcript>

# Output:
# Format: speechmatics
# Speakers: S1, S2, S3
```

## Storage Structure

Samples are stored under `$SPEAKERS_EMBEDDINGS_DIR/samples/`:

```
$SPEAKERS_EMBEDDINGS_DIR/
├── db/                     # Speaker profiles (speaker_detection)
├── embeddings/             # Embedding vectors (speaker_detection)
└── samples/                # Voice samples (speaker_samples)
    ├── alice/
    │   ├── sample-001.mp3
    │   ├── sample-001.meta.yaml
    │   ├── sample-002.mp3
    │   └── sample-002.meta.yaml
    └── bob/
        └── ...
```

## Metadata Format

Each sample has a `.meta.yaml` sidecar file with full provenance:

```yaml
version: 1
sample_id: sample-001
source:
  audio_file: /path/to/meeting.mp3
  audio_hash: sha256:abc123...
  transcript_file: /path/to/meeting.speechmatics.json
segment:
  speaker_label: S1
  start_sec: 10.5
  end_sec: 25.3
  duration_sec: 14.8
  text: "transcribed text..."
extraction:
  tool: speaker_samples
  tool_version: 1.0.0
  extracted_at: 2026-01-12T10:30:00Z
```

## Supported Transcript Formats

### Speechmatics

```json
{
  "results": [
    {
      "type": "word",
      "start_time": 0.04,
      "end_time": 0.32,
      "alternatives": [{"content": "Hello", "speaker": "S1"}]
    }
  ]
}
```

### AssemblyAI

```json
{
  "utterances": [
    {
      "speaker": "A",
      "start": 40,
      "end": 320,
      "text": "Hello"
    }
  ]
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SPEAKERS_EMBEDDINGS_DIR` | Storage location | `~/.config/speakers_embeddings` |

## Integration with speaker_detection

`speaker_samples` and `speaker_detection` share the same `$SPEAKERS_EMBEDDINGS_DIR`:

```bash
# 1. Create speaker profile
./speaker_detection add alice --name "Alice Smith"

# 2. Extract voice samples
./speaker_samples extract meeting.mp3 -t meeting.json -l S1 -s alice

# 3. Enroll using samples (future: --from-samples flag)
./speaker_detection enroll alice meeting.mp3 --from-transcript meeting.json --speaker-label S1

# 4. List samples and embeddings
./speaker_samples list alice
./speaker_detection embeddings alice
```

## Typical Workflow

```bash
# Step 1: Transcribe audio with speaker diarization
./stt_speechmatics.py meeting.mp3 -o meeting.speechmatics.json

# Step 2: Identify speakers in transcript (manual or automated)
# - Listen to audio
# - Map S1="Alice", S2="Bob"

# Step 3: Extract samples for enrollment
./speaker_samples extract meeting.mp3 -t meeting.json -l S1 -s alice
./speaker_samples extract meeting.mp3 -t meeting.json -l S2 -s bob

# Step 4: Create profiles and enroll
./speaker_detection add alice --name "Alice Smith" --tag work
./speaker_detection enroll alice meeting.mp3 --from-transcript meeting.json --speaker-label S1

# Step 5: Future transcriptions auto-identify speakers
./stt_speechmatics.py new_meeting.mp3 --speakers-tag work
```

## Related Tools

* [`speaker_detection`](./speaker_detection.README.md) - Speaker profile and embedding management
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics STT with speaker identification
* [`stt_assemblyai_speaker_mapper.py`](./stt_assemblyai_speaker_mapper.README.md) - AssemblyAI speaker mapping

## See Also

* [Speaker Identity System Architecture](./ramblings/2026-01-12--speaker-identity-system.md) - Central design document
