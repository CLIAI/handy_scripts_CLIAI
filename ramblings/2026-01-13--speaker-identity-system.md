# Speaker Identity System Architecture

This document describes the unified speaker identity ecosystem in `handy_scripts_CLIAI` - a collection of tools for speech-to-text transcription with speaker identification capabilities.

## System Overview

```mermaid
graph TB
    subgraph "Input Audio"
        A[Audio File<br/>meeting.mp3]
    end

    subgraph "STT Providers"
        B1[stt_speechmatics.py]
        B2[stt_assemblyai_speaker_mapper.py]
        B3[stt_openai.py]
    end

    subgraph "Transcript Output"
        C1[Speechmatics JSON<br/>results + speakers]
        C2[AssemblyAI JSON<br/>utterances A/B/C]
    end

    subgraph "Speaker Identity Tools"
        D1[speaker_samples<br/>Sample Extraction]
        D2[speaker_detection<br/>Profile Management]
    end

    subgraph "Unified Storage"
        E[(SPEAKERS_EMBEDDINGS_DIR)]
        E1[db/*.json<br/>Speaker Profiles]
        E2[embeddings/*/<br/>Embedding Vectors]
        E3[samples/*/<br/>Voice Samples]
    end

    A --> B1
    A --> B2
    A --> B3

    B1 --> C1
    B2 --> C2

    C1 --> D1
    C2 --> D1
    D1 --> E3

    E3 --> D2
    D2 --> E1
    D2 --> E2

    E1 --> B1
    E2 --> B1

    style E fill:#e1f5fe
    style D1 fill:#fff3e0
    style D2 fill:#fff3e0
```

## Data Flow

```mermaid
flowchart LR
    subgraph "1. Transcription"
        Audio[Audio] --> STT[STT Tool]
        STT --> Transcript[Transcript JSON]
    end

    subgraph "2. Sample Extraction"
        Transcript --> Samples[speaker_samples]
        Audio --> Samples
        Samples --> |YAML metadata| Storage1[samples/person/]
    end

    subgraph "3. Speaker Enrollment"
        Storage1 --> Detection[speaker_detection]
        Detection --> |API call| Provider[Backend API]
        Provider --> |embedding| Storage2[embeddings/]
        Detection --> |profile| Storage3[db/*.json]
    end

    subgraph "4. Identification"
        Storage3 --> STT2[STT Tool]
        Storage2 --> STT2
        Audio2[New Audio] --> STT2
        STT2 --> Named[Named Transcript]
    end
```

## Component Responsibilities

### Speech-to-Text Tools

| Tool | Provider | Diarization | Speaker ID | Output Format |
|------|----------|-------------|------------|---------------|
| `stt_speechmatics.py` | Speechmatics | S1, S2... | Yes | JSON (results) |
| `stt_assemblyai_speaker_mapper.py` | AssemblyAI | A, B, C... | LLM-based | JSON (utterances) |
| `stt_openai.py` | OpenAI | No | No | Text |

### Speaker Identity Tools

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `speaker_samples` | Voice sample extraction | Audio + Transcript | MP3 + YAML metadata |
| `speaker_detection` | Profile & embedding mgmt | Audio/Samples | JSON profiles, API embeddings |

## Storage Layout

```
$SPEAKERS_EMBEDDINGS_DIR/           # Default: ~/.config/speakers_embeddings
├── config.json                     # Global settings
├── db/                             # Speaker profiles
│   ├── alice.json                  # {id, names, tags, embeddings}
│   ├── bob.json
│   └── ...
├── embeddings/                     # Embedding vectors (local backends)
│   ├── alice/
│   │   └── emb-abc123.npy
│   └── ...
└── samples/                        # Voice samples (speaker_samples)
    ├── alice/
    │   ├── sample-001.mp3
    │   ├── sample-001.meta.yaml    # Provenance metadata
    │   ├── sample-002.mp3
    │   └── sample-002.meta.yaml
    └── bob/
        └── ...
```

## Speaker Profile Schema

```json
{
  "id": "alice",
  "version": 1,
  "names": {
    "default": "Alice Smith",
    "family": "Mom",
    "business": "Dr. Alice Smith"
  },
  "nicknames": ["Ali"],
  "tags": ["family", "primary"],
  "description": "Mother, software engineer",
  "metadata": {
    "email": "alice@example.com"
  },
  "embeddings": {
    "speechmatics": [
      {
        "id": "emb-abc123",
        "external_id": "spk_xyz...",
        "source_audio": "/path/to/audio.mp3",
        "created_at": "2026-01-12T..."
      }
    ]
  },
  "created_at": "2026-01-12T...",
  "updated_at": "2026-01-12T..."
}
```

## Sample Metadata Schema

```yaml
version: 2
sample_id: sample-001
b3sum: abc123def456...                 # Blake3 hash of THIS sample audio

source:
  audio_file: /path/to/meeting.mp3
  audio_b3sum: xyz789...               # Blake3 of source recording
  transcript_file: /path/to/meeting.speechmatics.json

segment:
  speaker_label: S1
  start_sec: 10.5
  end_sec: 25.3
  duration_sec: 14.8
  text: "transcribed speech content..."

extraction:
  tool: speaker_samples
  tool_version: 1.1.0
  extracted_at: 2026-01-12T10:30:00Z

review:
  status: pending                      # pending | reviewed | rejected
  reviewed_at: null
  notes: null
```

## Typical Workflows

### Initial Speaker Enrollment

```mermaid
sequenceDiagram
    participant User
    participant stt as stt_speechmatics.py
    participant samples as speaker_samples
    participant detect as speaker_detection
    participant api as Speechmatics API

    User->>stt: Transcribe meeting.mp3
    stt->>api: Batch transcription
    api-->>stt: JSON with S1, S2 diarization
    stt-->>User: meeting.speechmatics.json

    User->>samples: Extract S1 as "alice"
    samples-->>User: samples/alice/sample-001.mp3

    User->>detect: add alice --name "Alice"
    User->>detect: enroll alice meeting.mp3 -t meeting.json -l S1
    detect->>api: Create speaker identifier
    api-->>detect: Encrypted ID
    detect-->>User: Enrolled embedding emb-xyz
```

### Speaker Identification in New Recording

```mermaid
sequenceDiagram
    participant User
    participant stt as stt_speechmatics.py
    participant detect as speaker_detection
    participant api as Speechmatics API

    User->>detect: export --tags family
    detect-->>stt: Speaker identifiers

    User->>stt: Transcribe new_call.mp3 --speakers-tag family
    stt->>api: Transcription + speaker IDs
    api-->>stt: JSON with "Alice", "Bob" labels
    stt-->>User: Named transcript
```

## Backend Architecture

```mermaid
graph TB
    subgraph "speaker_detection CLI"
        CLI[CLI Parser]
        CMDs[Command Handlers]
    end

    subgraph "Backends"
        BASE[EmbeddingBackend<br/>Abstract Base]
        SM[SpeechmaticsBackend]
        PA[PyAnnoteBackend<br/>future]
        SB[SpeechBrainBackend<br/>future]
    end

    subgraph "External APIs"
        SMAPI[Speechmatics API]
    end

    subgraph "Local Models"
        LOCAL[Local Embedding<br/>future]
    end

    CLI --> CMDs
    CMDs --> BASE
    BASE --> SM
    BASE --> PA
    BASE --> SB
    SM --> SMAPI
    PA --> LOCAL
    SB --> LOCAL
```

## Transcript Format Comparison

### Speechmatics (results array, seconds)

```json
{
  "results": [
    {
      "type": "word",
      "start_time": 0.04,
      "end_time": 0.32,
      "speaker": "S1",
      "alternatives": [{"content": "Hello", "speaker": "Alice"}]
    }
  ],
  "speakers": [
    {
      "label": "S1",
      "speaker_identifiers": ["spk_..."]
    }
  ]
}
```

### AssemblyAI (utterances array, milliseconds)

```json
{
  "utterances": [
    {
      "speaker": "A",
      "start": 40,
      "end": 320,
      "text": "Hello, how are you?"
    }
  ]
}
```

## Design Principles

1. **UNIX Philosophy**: Each tool does one thing well. Composable via pipes and files.

2. **Unified Storage**: Single `$SPEAKERS_EMBEDDINGS_DIR` namespace for all speaker data.

3. **Provenance Tracking**: Full metadata for reproducibility (audio hash, timestamps, source files).

4. **Backend Agnostic**: Abstract interface supports multiple embedding providers.

5. **Context-Aware Names**: Same person can have different display names per context (family vs business).

6. **jq-Compatible**: JSON storage enables ad-hoc queries via standard tooling.

## Tool Quick Reference

```bash
# Speaker profile management
./speaker_detection add <id> --name "Name" [--tag tag]
./speaker_detection list [--tags tag] [--format json]
./speaker_detection show <id>
./speaker_detection update <id> [--name "New Name"]
./speaker_detection delete <id> [--force]

# Speaker enrollment
./speaker_detection enroll <id> <audio> [--from-transcript file.json --speaker-label S1]
./speaker_detection embeddings <id>
./speaker_detection identify <audio> [--tags tags]
./speaker_detection verify <id> <audio>

# Sample extraction
./speaker_samples extract <audio> -t transcript.json -l S1 -s speaker_id
./speaker_samples segments -t transcript.json -l S1  # JSONL output
./speaker_samples list [speaker_id]
./speaker_samples info <speaker_id> <sample_id>
./speaker_samples remove <speaker_id> [--all]

# STT with speaker identification
./stt_speechmatics.py <audio> --speakers-tag <tag> [--speaker-id id1,id2]

# Review workflow
./speaker_samples review <speaker_id> <sample_id> --approve|--reject [--notes "..."]
./speaker_samples list <speaker_id> --show-review --status pending|reviewed|rejected

# Trust level verification
./speaker_detection embeddings <id> --show-trust
./speaker_detection check-validity [speaker_id]
```

## Review State and Trust Levels

Diarization produces many samples from recordings. Users may not review all samples immediately but want to use them for embeddings. The review/trust system tracks sample quality.

### Sample Review States

```mermaid
stateDiagram-v2
    [*] --> pending: Sample extracted (b3sum computed)
    pending --> reviewed: Approve
    pending --> rejected: Reject
    reviewed --> rejected: Error found
    rejected --> reviewed: Mistake corrected
```

Each sample has a review status stored in its `.meta.yaml`:

```yaml
review:
  status: pending | reviewed | rejected
  reviewed_at: 2026-01-15T10:30:00Z
  notes: "confirmed correct speaker"
```

### Content-Addressable Tracking with Blake3

Samples are identified by their blake3 hash (`b3sum`). This enables:

* **Content verification** - detect if audio was modified
* **Provenance tracking** - link embeddings to exact source samples
* **Trust computation** - aggregate sample states into embedding confidence

### Trust Level Hierarchy

Embeddings store which samples (by b3sum) were used during enrollment:

```json
{
  "id": "emb-abc12345",
  "samples": {
    "reviewed": ["abc123...", "def456..."],
    "unreviewed": ["789xyz..."],
    "rejected": []
  },
  "trust_level": "medium"
}
```

Trust levels computed from sample lists:

| Level | Criteria | Use Case |
|-------|----------|----------|
| **HIGH** | All samples reviewed, none rejected | Critical identification |
| **MEDIUM** | Mix of reviewed + unreviewed, none rejected | General use |
| **LOW** | All samples unreviewed | Exploration only |
| **INVALIDATED** | Any sample rejected | Needs re-enrollment |

### Invalidation Detection

When samples are later rejected, embeddings become invalid:

```bash
# Check all embeddings against current sample states
./speaker_detection check-validity

# Sample output:
# INVALIDATED: alice/speechmatics/emb-abc12345
#   Newly rejected samples: abc123...
#
# Checked 5 embeddings across 2 speakers
#   1 INVALIDATED (re-enrollment needed)
```

This enables continuous quality improvement: review samples as time permits, then re-enroll speakers whose embeddings are now suspect.

## See Also

* [speaker_detection.README.md](../speaker_detection.README.md)
* [speaker_samples.README.md](../speaker_samples.README.md)
* [stt_speechmatics.README.md](../stt_speechmatics.README.md)
* [evals/speaker_detection/](../evals/speaker_detection/) - Test suite
