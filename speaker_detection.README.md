# speaker_detection

Universal CLI for managing speaker embeddings across multiple backends with jq-queryable JSON storage.

## Overview

`speaker_detection` provides a unified interface for:

* Managing speaker profiles with metadata, tags, and context-specific names
* Enrolling speakers using audio samples
* Identifying and verifying speakers in audio files
* Exporting speaker data for STT integration

## Installation

No installation required. The script uses `uv run` for dependency management.

Requirements:

* Python 3.11+
* ffmpeg (for audio segment extraction)
* jq (for query command)

## Quick Start

```bash
# Add a speaker
./speaker_detection add john-smith --name "John Smith" --tag team-alpha

# Enroll speaker from audio
./speaker_detection enroll john-smith audio/john_sample.mp3

# Or enroll from transcript segments
./speaker_detection enroll john-smith audio/meeting.mp3 \
    --from-transcript meeting.speechmatics.json \
    --speaker-label S1

# Identify speaker in new audio
./speaker_detection identify audio/unknown_speaker.mp3

# List all speakers
./speaker_detection list

# Export for STT integration
./speaker_detection export --tags team-alpha --format speechmatics
```

## Commands

### Speaker Management

```bash
# Add speaker with context-specific names
./speaker_detection add alice --name "Alice Anderson" \
    --name-context podcast="Dr. Alice Anderson" \
    --name-context internal="Alice A." \
    --nickname "AA" \
    --tag podcast-hosts \
    --metadata organization=TechCorp

# List speakers
./speaker_detection list                    # Table format
./speaker_detection list --format json      # JSON output
./speaker_detection list --format ids       # IDs only
./speaker_detection list --tags podcast     # Filter by tag
./speaker_detection list --limit 10         # Pagination: first 10
./speaker_detection list --offset 5 --limit 5  # Skip 5, show next 5

# Show speaker details
./speaker_detection show alice
./speaker_detection show alice --format yaml

# Update speaker
./speaker_detection update alice --name "Alice B. Anderson"
./speaker_detection update alice --tag team-beta
./speaker_detection update alice --remove-tag old-tag

# Manage tags
./speaker_detection tag alice --add new-tag
./speaker_detection tag alice --remove old-tag

# Delete speaker
./speaker_detection delete alice
./speaker_detection delete alice --force
```

### Embedding Management

```bash
# Enroll from full audio file (best for short samples)
./speaker_detection enroll alice audio/alice_intro.mp3

# Enroll from specific time segments
./speaker_detection enroll alice audio/meeting.mp3 --segments 10.5:45.2,78:120

# Enroll using segments from a transcript
./speaker_detection enroll alice audio/meeting.mp3 \
    --from-transcript meeting.speechmatics.json \
    --speaker-label S1

# List embeddings for a speaker
./speaker_detection embeddings alice
./speaker_detection embeddings alice --backend speechmatics
./speaker_detection embeddings alice --show-trust  # Show trust level and sample counts

# Remove an embedding
./speaker_detection remove-embedding alice emb-abc12345

# Check embedding validity (based on sample review states)
./speaker_detection check-validity              # All speakers
./speaker_detection check-validity alice        # Specific speaker
./speaker_detection check-validity -v           # Verbose (show all embeddings)

# Validate schema of profiles and embeddings
./speaker_detection validate                    # All speakers
./speaker_detection validate alice              # Specific speaker
./speaker_detection validate --strict           # Non-zero exit on warnings
```

### Detection/Identification

```bash
# Identify speaker in audio (matches against all enrolled speakers)
./speaker_detection identify audio/unknown.mp3

# Filter candidates by tag
./speaker_detection identify audio/unknown.mp3 --tags podcast-hosts

# Verify specific speaker
./speaker_detection verify alice audio/sample.mp3
```

### Export & Query

```bash
# Export all speakers as JSON
./speaker_detection export

# Export specific tags in Speechmatics format
./speaker_detection export --tags team-alpha --format speechmatics

# Use specific name context for export
./speaker_detection export --context podcast

# Query with jq expressions
./speaker_detection query '.[].id'
./speaker_detection query '.[] | select(.tags | contains(["podcast"]))'
```

## STT Integration

### With stt_speechmatics.py

Use enrolled speakers directly during transcription:

```bash
# Transcribe with speaker identification
./stt_speechmatics.py -d --speakers-tag podcast-hosts audio.mp3

# Use specific name context
./stt_speechmatics.py -d --speakers-tag team --speakers-context internal audio.mp3
```

### Export Pipeline

```bash
# Export and pipe to STT
./speaker_detection export --tags team --format speechmatics | \
    ./stt_speechmatics.py -d --speakers-file - audio.mp3
```

## Storage

Speaker profiles are stored as JSON files in `$SPEAKERS_EMBEDDINGS_DIR/db/` (default: `~/.config/speakers_embeddings/db/`).

### Speaker Profile Schema

```json
{
  "id": "john-smith",
  "version": 1,
  "names": {
    "default": "John Smith",
    "podcast": "Dr. John Smith",
    "internal": "John S."
  },
  "nicknames": ["Johnny", "JS"],
  "description": "Software engineer",
  "metadata": {"organization": "TechCorp", "role": "CTO"},
  "tags": ["team-alpha", "podcast-hosts"],
  "embeddings": {
    "speechmatics": [{
      "id": "emb-abc12345",
      "external_id": "<encrypted_identifier>",
      "source_audio": "/path/to/file.mp3",
      "source_segments": [{"start": 0, "end": 30}],
      "model_version": "speechmatics-v2",
      "created_at": "2026-01-07T10:00:00Z"
    }]
  },
  "created_at": "2026-01-07T09:00:00Z",
  "updated_at": "2026-01-07T10:00:00Z"
}
```

## Backends

### Speechmatics (default)

API-based speaker identification using encrypted identifiers.

Requirements:

* `SPEECHMATICS_API_KEY` environment variable

### Future Backends

* **PyAnnote**: Local PyAnnote 3.1 embeddings (requires torch)
* **SpeechBrain**: Local SpeechBrain ECAPA-TDNN (requires torch)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Database location |
| `SPEAKER_DETECTION_BACKEND` | `speechmatics` | Default backend |
| `SPEAKER_BACKENDS_CONFIG` | `speaker_detection_backends/backends.yaml` | Backend registry config |
| `SPEECHMATICS_API_KEY` | - | Speechmatics API key |
| `SPEAKER_DETECTION_DEBUG` | - | Enable debug logging |

## Best Practices

### Enrollment

* Use 5-30 seconds of clear speech per enrollment
* Enroll multiple samples from different recordings for robustness
* Extract segments from transcripts to ensure single-speaker audio
* Avoid noisy or overlapping speech in enrollment samples

### Tags & Contexts

* Use tags for filtering (e.g., `team-alpha`, `podcast-hosts`)
* Use contexts for name variants (e.g., `podcast` for formal names)
* Keep tag names lowercase with hyphens

## Evaluation

Run speaker detection benchmarks:

```bash
cd evals/speaker_detection

# Generate test audio
make all

# Run benchmarks
./benchmark.py

# Run specific test
./benchmark.py --tests 001

# Dry run (check audio files exist)
./benchmark.py --dry-run
```

## Shell Completions

Tab-completion scripts are available for bash, zsh, and fish:

```bash
# Bash - add to ~/.bashrc
source completions/bash/speaker_detection.bash

# Zsh - add to fpath in ~/.zshrc
fpath=(completions/zsh $fpath)
autoload -Uz compinit && compinit

# Fish - symlink to completions directory
ln -s completions/fish/speaker_detection.fish ~/.config/fish/completions/
```

## Related Tools

* [`speaker_samples`](./speaker_samples.README.md) - Voice sample extraction with provenance
* [`speaker_segments`](./speaker_segments.README.md) - Extract segment timestamps from transcripts
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speech-to-text with speaker diarization
* [`stt_assemblyai.py`](./stt_assemblyai.README.md) - AssemblyAI transcription
* [`stt_speechmatics_speaker_mapper.py`](./stt_speechmatics_speaker_mapper.README.md) - Map speaker labels to names
