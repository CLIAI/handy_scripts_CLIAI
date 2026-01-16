# speaker_segments - Extract Speaker Segment Timestamps

Standalone CLI tool for extracting speaker segment timestamps from transcript JSON files. Supports AssemblyAI and Speechmatics formats with multiple output options.

## Overview

`speaker_segments` provides a lightweight way to extract timing information for specific speakers from transcripts without the full sample extraction workflow of `speaker_samples`.

```
┌─────────────────┐         ┌─────────────────┐
│   Transcript    │  ──▶    │ speaker_segments│
│  (*.json)       │         └────────┬────────┘
└─────────────────┘                  │
                                     ▼
                     ┌───────────────────────────────┐
                     │  JSON:   [{"start": 10.5, "end": 25.3}, ...]
                     │  Tuples: [(10.5, 25.3), ...]
                     │  CSV:    start,end\n10.5,25.3\n...
                     └───────────────────────────────┘
```

## Installation

No installation required. Uses `uv run` for dependency management.

## Quick Start

```bash
# List available speakers in a transcript
./speaker_segments transcript.json dummy --list-speakers

# Extract segments for speaker S1 as JSON
./speaker_segments transcript.json S1

# Output as Python tuples (for scripting)
./speaker_segments transcript.json S1 --format tuples

# Output as CSV
./speaker_segments transcript.json S1 --format csv

# Merge segments with small gaps
./speaker_segments transcript.json S1 --merge-gap 2.0
```

## Usage

```
speaker_segments <transcript_file> <speaker_label> [options]

Positional arguments:
  transcript_file     Path to transcript JSON file (AssemblyAI or Speechmatics)
  speaker_label       Speaker label to extract (e.g., 'S1', 'Alice', 'A')

Options:
  -f, --format        Output format: json (default), tuples, csv
  --merge-gap N       Merge segments with gaps smaller than N seconds (default: 0)
  --list-speakers     List available speakers in the transcript and exit
  -h, --help          Show help message
```

## Output Formats

### JSON (default)

```json
[
  {"start": 10.5, "end": 25.3},
  {"start": 45.0, "end": 60.2}
]
```

### Tuples

```python
[(10.5, 25.3), (45.0, 60.2)]
```

### CSV

```csv
start,end
10.5,25.3
45.0,60.2
```

## Merge Gap

The `--merge-gap` option combines consecutive segments when the gap between them is smaller than the specified threshold:

```bash
# Without merge: 3 segments with 0.5s and 5s gaps
./speaker_segments transcript.json S1
# [{"start": 1.0, "end": 3.0}, {"start": 3.5, "end": 5.0}, {"start": 10.0, "end": 12.0}]

# With merge-gap=1.0: First two segments merged (0.5s < 1.0s)
./speaker_segments transcript.json S1 --merge-gap 1.0
# [{"start": 1.0, "end": 5.0}, {"start": 10.0, "end": 12.0}]
```

## Supported Transcript Formats

### AssemblyAI

```json
{
  "utterances": [
    {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello"},
    {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi"}
  ]
}
```

Note: AssemblyAI uses milliseconds; they are automatically converted to seconds.

### Speechmatics

```json
{
  "results": [
    {
      "type": "word",
      "start_time": 0.5,
      "end_time": 1.0,
      "alternatives": [{"content": "Hello", "speaker": "S1"}]
    }
  ]
}
```

## Integration Examples

### Pipe to ffmpeg for audio extraction

```bash
./speaker_segments meeting.json S1 --format csv | tail -n +2 | \
    while IFS=, read start end; do
        ffmpeg -i meeting.mp3 -ss "$start" -to "$end" -c copy "segment_${start}.mp3"
    done
```

### Use with speaker_detection enroll

```bash
# Get segments and format for --segments flag
SEGS=$(./speaker_segments meeting.json S1 --format tuples | \
    python3 -c "import sys; segs=eval(sys.stdin.read()); print(','.join(f'{s}:{e}' for s,e in segs))")

./speaker_detection enroll alice meeting.mp3 --segments "$SEGS"
```

### Generate segment list for review

```bash
./speaker_segments meeting.json S1 --format json | \
    jq -r '.[] | "[\(.start)s - \(.end)s] duration: \(.end - .start | floor)s"'
```

## Related Tools

* [`speaker_samples`](./speaker_samples.README.md) - Full sample extraction with metadata storage
* [`speaker_detection`](./speaker_detection.README.md) - Speaker profile and embedding management
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics transcription

## See Also

* Shell completions available in `completions/{bash,zsh,fish}/`
