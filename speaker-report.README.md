# speaker-report - Quality Metrics and Recommendations

Quality metrics and recommendations for the speaker detection system. Part of the speaker-* tool ecosystem.

## Overview

`speaker-report` provides system health monitoring and actionable recommendations for your speaker identification pipeline. It analyzes:

* Recording processing progress and status distribution
* Speaker enrollment quality and trust levels
* Assignment confidence scores
* Context-based coverage metrics
* Stale recordings that need attention

```
                    ┌─────────────────────────────────────┐
                    │         speaker-report              │
                    │    Quality Metrics Dashboard        │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┐
        │              │           │           │              │
        v              v           v           v              v
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   catalog/   │ │  db/     │ │assignments│ │embeddings│ │ samples/ │
│  (entries)   │ │(profiles)│ │(mappings) │ │(vectors) │ │ (audio)  │
└──────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Installation

No installation required. The script uses `uv run` for dependency management.

```bash
# Run directly - dependencies are auto-installed
./speaker-report status

# Requirements:
# - Python 3.11+
# - PyYAML (auto-installed via uv)
```

## Quick Start

```bash
# 1. Check overall system status (default command)
./speaker-report

# 2. View coverage by context
./speaker-report coverage

# 3. Find low-confidence assignments
./speaker-report confidence --below 70

# 4. Find stale recordings
./speaker-report stale --days 14

# 5. Review speaker enrollment
./speaker-report speakers

# 6. Get JSON output for scripting
./speaker-report status --format json
```

## Commands

### `status` - Overall System Status (Default)

```bash
./speaker-report status [--confidence-threshold PCT] [--stale-days N]

Options:
  --confidence-threshold  Percentage threshold for low confidence (default: 70)
  --stale-days           Days threshold for stale recordings (default: 30)
  -f, --format           Output format: text (default), json
```

Example output:

```
Speaker Detection System Status
================================
Recordings:     42 total
  - Processed:  38 (90%)
  - Reviewed:   25 (60%)
  - Pending:    4

Speakers:       12 enrolled
  - High trust: 8
  - Medium:     3
  - Low:        1

Contexts:       3 defined
  - team-standup: 15 recordings, 12 reviewed
  - podcast:      20 recordings, 10 reviewed
  - interview:    7 recordings, 3 reviewed

Recommendations:
  - 4 recordings have low-confidence assignments
  - 2 speakers need more reviewed samples
```

### `coverage` - Review Coverage by Context

```bash
./speaker-report coverage [--context NAME]

Options:
  -c, --context    Filter by specific context name
  -f, --format     Output format: text (default), json
```

Shows processing status distribution per context:

```bash
# All contexts
./speaker-report coverage

# Specific context
./speaker-report coverage --context podcast
```

Example output:

```
Coverage by Context
====================

Context: team-standup
  Total:       15
  Unprocessed: 0
  Transcribed: 2
  Assigned:    1
  Reviewed:    8
  Complete:    4
  Coverage:    80%

Context: podcast
  Total:       20
  Unprocessed: 2
  Transcribed: 5
  Assigned:    3
  Reviewed:    7
  Complete:    3
  Coverage:    50%
```

### `confidence` - Low Confidence Assignments

```bash
./speaker-report confidence [--below PCT]

Options:
  -b, --below     Confidence threshold percentage (default: 70)
  -f, --format    Output format: text (default), json
```

Lists recordings with speaker assignments below the confidence threshold:

```bash
# Default 70% threshold
./speaker-report confidence

# More strict threshold
./speaker-report confidence --below 80

# JSON for scripting
./speaker-report confidence --below 60 --format json
```

Example output:

```
Recordings Below 70% Confidence
========================================

Found 4 recording(s):

  meeting-2026-01-15.mp3
    B3SUM: abc123def456789...
    Context: team-standup
    - S2 -> bob (low)
    - S4 -> (unassigned) (unassigned)

  podcast-ep42.wav
    B3SUM: xyz789abc123456...
    Context: podcast
    - S3 -> (unassigned) (unassigned)
```

### `stale` - Recordings with Old Processing

```bash
./speaker-report stale [--days N]

Options:
  -d, --days      Days threshold (default: 30)
  -f, --format    Output format: text (default), json
```

Identifies recordings that have not been updated recently:

```bash
# Default 30 days
./speaker-report stale

# More aggressive - 14 days
./speaker-report stale --days 14

# Check for very old recordings
./speaker-report stale --days 90
```

Example output:

```
Recordings Not Updated in 30+ Days
========================================

Found 3 recording(s):

  old-meeting.mp3
    Status: assigned
    Last updated: 45 days ago
    Context: team-standup

  legacy-audio.wav
    Status: transcribed
    Last updated: 60 days ago
    Context: archive
```

### `speakers` - Speaker Enrollment Summary

```bash
./speaker-report speakers

Options:
  -f, --format    Output format: text (default), json
```

Shows speaker enrollment quality and sample counts:

```bash
./speaker-report speakers
```

Example output:

```
Speaker Enrollment Summary
==========================

Total speakers: 12

ID                   Name                 Trust      Samples  Reviewed
----------------------------------------------------------------------
alice                Alice Smith          high       15       12
bob                  Bob Jones            high       10       8
carol                Carol Davis          medium     5        3
dave                 Dave Wilson          low        2        1

By trust level:
  - high: 8
  - medium: 3
  - low: 1

Speakers needing more reviewed samples (2):
  - carol (3 reviewed)
  - dave (1 reviewed)
```

## Output Formats

All commands support `--format json` for machine-readable output:

```bash
# JSON output for scripting
./speaker-report status --format json

# Parse with jq
./speaker-report status --format json | jq '.recommendations'

# Extract specific metrics
./speaker-report speakers --format json | jq '.speakers[] | select(.trust_level == "low")'
```

## Recommendations Engine

The status command generates actionable recommendations based on:

| Condition | Recommendation |
|-----------|---------------|
| Low-confidence assignments | "N recording(s) have low-confidence assignments" |
| Speakers with few samples | "Speaker(s) X, Y need more reviewed samples" |
| Pending recordings | "N recording(s) pending transcription" |
| Stale recordings | "N recording(s) have not been updated recently" |
| Unreviewed contexts | "Context(s) 'X' have no reviewed recordings" |

## Integration with Other Tools

`speaker-report` reads data from the speaker-* ecosystem:

```bash
# Pipeline workflow
# 1. Add recordings
./speaker-catalog add *.mp3 --context team-standup

# 2. Process transcripts
for f in *.mp3; do
    ./stt_speechmatics.py "$f" -o "${f%.mp3}.json"
    ./speaker-catalog register-transcript "$f" --backend speechmatics --transcript "${f%.mp3}.json"
done

# 3. Assign speakers
./speaker-assign batch --context team-standup

# 4. Review assignments
./speaker-review --context team-standup

# 5. Check system health
./speaker-report status

# 6. Find issues
./speaker-report confidence --below 80
./speaker-report stale --days 7
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Root storage directory |

## Storage Layout

`speaker-report` reads from these directories:

```
$SPEAKERS_EMBEDDINGS_DIR/
├── catalog/          # Recording catalog entries (speaker-catalog)
│   └── {b3sum}.yaml
├── assignments/      # Speaker label mappings (speaker-assign)
│   └── {b3sum}.yaml
├── db/               # Speaker profiles (speaker_detection)
│   └── {speaker_id}.yaml
├── embeddings/       # Voice embeddings (speaker_detection)
│   └── {speaker_id}/
│       └── *.npy
└── samples/          # Voice samples (speaker_samples)
    └── {speaker_id}/
        └── *.wav
```

## Related Tools

* [`speaker-catalog`](./speaker-catalog.README.md) - Recording inventory management
* [`speaker-assign`](./speaker-assign.README.md) - Speaker label assignment
* [`speaker-review`](./speaker-review.README.md) - Interactive review TUI
* [`speaker_detection`](./speaker_detection.README.md) - Speaker profiles and embeddings
* [`speaker_samples`](./speaker_samples.README.md) - Voice sample extraction

## Common Workflows

### Weekly Health Check

```bash
#!/bin/bash
# weekly-speaker-health.sh

echo "=== Speaker Detection Health Report ==="
echo ""

# Overall status
./speaker-report status

echo ""
echo "=== Action Items ==="

# Low confidence (needs review)
low_conf=$(./speaker-report confidence --format json | jq '.count')
if [ "$low_conf" -gt 0 ]; then
    echo "- Review $low_conf low-confidence assignments"
fi

# Stale recordings
stale=$(./speaker-report stale --days 14 --format json | jq '.count')
if [ "$stale" -gt 0 ]; then
    echo "- Process $stale stale recordings"
fi

# Pending transcription
pending=$(./speaker-report status --format json | jq '.recordings.pending')
if [ "$pending" -gt 0 ]; then
    echo "- Transcribe $pending pending recordings"
fi
```

### Context-Focused Review

```bash
# Check specific context coverage
./speaker-report coverage --context podcast

# If coverage is low, start review session
./speaker-review --context podcast
```

### Pre-Enrollment Quality Check

```bash
# Before enrolling new speakers, check existing quality
./speaker-report speakers --format json | jq '
  .speakers
  | map(select(.trust_level != "high"))
  | map(.speaker_id)
'
```
