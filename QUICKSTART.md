# Quick Start: Speaker Identification in 5 Minutes

This guide walks you through a complete speaker identification workflow that you can copy-paste and run.

## Prerequisites

```bash
# Required tools
python3 --version   # 3.10+
ffmpeg -version     # Audio processing
jq --version        # JSON querying (optional)

# Required Python packages
pip install pyyaml requests

# API key for Speechmatics (enrollment/identification)
export SPEECHMATICS_API_KEY="your-key-here"
```

## Complete End-to-End Example

### Step 1: Transcribe Audio with Speaker Diarization

```bash
# Transcribe a meeting recording
./stt_speechmatics.py meeting.mp3 -o meeting.speechmatics.json

# Check what speakers were detected
./speaker_samples speakers meeting.speechmatics.json
# Output:
# Format: speechmatics
# Speakers: S1, S2, S3
```

### Step 2: Create Speaker Profiles

```bash
# Create profiles for the speakers you want to identify
./speaker_detection add alice --name "Alice Anderson"
./speaker_detection add bob --name "Bob Smith"

# Verify they were created
./speaker_detection list
# Output:
# alice   Alice Anderson
# bob     Bob Smith
```

### Step 3: Extract Voice Samples

```bash
# Extract Alice's voice samples from the transcript
./speaker_samples extract meeting.mp3 \
    -t meeting.speechmatics.json \
    -l S1 \
    -s alice

# Extract Bob's voice samples
./speaker_samples extract meeting.mp3 \
    -t meeting.speechmatics.json \
    -l S2 \
    -s bob

# Review extracted samples
./speaker_samples list alice --show-review
# Output:
# sample-001  5.2s  pending  "Hello everyone, let's get started..."
# sample-002  3.8s  pending  "I think we should focus on..."
```

### Step 4: Review and Approve Samples (Optional but Recommended)

```bash
# Listen to samples and approve good ones
./speaker_samples review alice sample-001 --approve
./speaker_samples review alice sample-002 --approve

# Reject samples that are wrong (background noise, wrong speaker)
./speaker_samples review alice sample-003 --reject --notes "Background music"
```

### Step 5: Enroll Speakers with Embeddings

```bash
# Enroll Alice using her audio samples
./speaker_detection enroll alice meeting.mp3 \
    --from-transcript meeting.speechmatics.json \
    --speaker-label S1

# Enroll Bob
./speaker_detection enroll bob meeting.mp3 \
    --from-transcript meeting.speechmatics.json \
    --speaker-label S2

# Verify embeddings were created
./speaker_detection embeddings alice --show-trust
# Output:
# emb-abc123  2026-01-16  [high] (2r/0u/0x)
```

### Step 6: Identify Speakers in New Audio

```bash
# Transcribe a new recording
./stt_speechmatics.py new_meeting.mp3 -o new_meeting.json

# Identify who is speaking
./speaker_detection identify new_meeting.mp3
# Output:
# S1: alice (confidence: 0.92)
# S2: bob (confidence: 0.87)
# S3: unknown
```

## Where Is My Data?

All speaker data is stored in `$SPEAKERS_EMBEDDINGS_DIR` (default: `~/.config/speakers_embeddings/`):

```bash
# View storage location
echo $SPEAKERS_EMBEDDINGS_DIR

# List speaker profiles
ls ~/.config/speakers_embeddings/db/
# alice.json  bob.json

# List voice samples
ls ~/.config/speakers_embeddings/samples/
# alice/  bob/

# View a profile
cat ~/.config/speakers_embeddings/db/alice.json | jq .
```

## Common Operations

### Add Context-Specific Names

```bash
# Alice goes by "Al" on podcasts
./speaker_detection update alice --name-context podcast="Al"

# Export with context
./speaker_detection export --context podcast
```

### Organize with Tags

```bash
# Tag speakers by team
./speaker_detection tag alice --add team-alpha
./speaker_detection tag bob --add team-beta

# List only team-alpha members
./speaker_detection list --tags team-alpha
```

### Query with jq

```bash
# Find speakers with embeddings
./speaker_detection query '.[] | select(.embeddings | length > 0) | .id'

# Export as JSON
./speaker_detection export --format json > speakers.json
```

### Check Embedding Validity

```bash
# After reviewing samples, check if embeddings need re-enrollment
./speaker_detection check-validity

# If samples were rejected, you'll see:
# INVALIDATED: alice - emb-abc123 (1 rejected sample)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Data storage location |
| `SPEAKER_DETECTION_BACKEND` | `speechmatics` | Embedding backend |
| `SPEECHMATICS_API_KEY` | (none) | Required for enrollment |

## Testing Your Setup

```bash
# Use a different directory for testing
export SPEAKERS_EMBEDDINGS_DIR=/tmp/test_speakers

# Run your commands...
./speaker_detection add test --name "Test Speaker"
./speaker_detection list

# Clean up when done
rm -rf /tmp/test_speakers
```

## Next Steps

* **[speaker_detection.README.md](speaker_detection.README.md)** - Full CLI reference
* **[speaker_samples.README.md](speaker_samples.README.md)** - Sample extraction details
* **[CONTRIBUTING.md](CONTRIBUTING.md)** - Coding guidelines
* **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common errors and fixes
