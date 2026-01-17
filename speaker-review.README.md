# speaker-review - Interactive TUI for Reviewing Speaker Assignments

Review diarization assignments, approve/reject samples, and grow the speaker embeddings database through an interactive terminal interface. Part of the speaker-* tool ecosystem.

## Overview

`speaker-review` provides an interactive TUI for manually reviewing automated speaker assignments. It allows you to verify, correct, or reject speaker identifications made by `speaker-assign`, and automatically extract approved samples to grow the speaker embeddings database.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Assignments   │    │   speaker-      │    │   Speaker       │
│   (abc123.yaml) │───>│   review        │───>│   Samples       │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Interactive TUI    │
                    │  ├── approve → extract
                    │  ├── reject  → flag
                    │  ├── skip    → defer
                    │  └── play    → verify
                    └──────────┬──────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │  ~/.cache/speaker-   │
                    │  review/session.yaml │
                    └──────────────────────┘
```

## Installation

No installation required. The script uses `uv run` for dependency management.

```bash
# Run directly - dependencies are auto-installed
./speaker-review meeting.mp3

# Dependencies installed automatically:
# - pyyaml>=6.0
# - rich>=13.0

# Requirements:
# - Python 3.11+
# - b3sum (optional, falls back to SHA256)
# - mpv or ffplay (for audio playback)
# - speaker_samples (optional, for sample extraction)
# - speaker_detection (optional, for profile editing)
```

## Quick Start

```bash
# 1. Review a specific recording
./speaker-review meeting.mp3

# 2. Continue a previous session
./speaker-review --continue

# 3. Review with simple prompt mode (for non-TTY environments)
./speaker-review meeting.mp3 --simple

# 4. Check current session status
./speaker-review status

# 5. Clear saved session
./speaker-review clear
```

## Commands

### `review` - Start or continue a review session (default)

```bash
./speaker-review [review] <audio> [OPTIONS]
./speaker-review --continue

Options:
  <audio>              Path to audio file or b3sum prefix
  -c, --continue       Continue previous saved session
  --context NAME       Filter recordings by context
  --speaker ID         Review samples for specific speaker
  -s, --simple         Use simple prompt mode (no raw terminal input)
```

The `<audio>` argument can be:

* Full path to audio file
* b3sum prefix (at least 6 characters)

Examples:

```bash
# Review specific recording
./speaker-review meeting.mp3

# Review by b3sum prefix
./speaker-review abc123

# Continue interrupted session
./speaker-review --continue

# Filter by context
./speaker-review --context team-standup

# Simple mode for scripts or non-TTY
./speaker-review meeting.mp3 --simple
```

### `status` - Show current session status

```bash
./speaker-review status
```

Output includes:

* Recording identifier (b3sum)
* Audio file path
* Context name
* Current position in review
* Review progress (approved/rejected/skipped counts)

Example output:

```
Active Session
Recording: abc123de...
Audio: meeting.mp3
Context: team-standup
Progress: 5/12
Reviewed: 4/12
  Approved: 2, Rejected: 1, Skipped: 1
```

### `clear` - Clear saved session

```bash
./speaker-review clear
```

Removes the saved session file, allowing you to start fresh.

## Keybindings

During interactive review, use these single-key commands:

| Key | Action | Description |
|-----|--------|-------------|
| `a` | Approve | Mark assignment correct, extract voice sample |
| `r` | Reject | Mark assignment as incorrect |
| `s` | Skip | Defer decision for later review |
| `e` | Edit | Edit speaker profile (names, tags) |
| `p` | Play | Play the audio segment |
| `n` | Next | Go to next segment |
| `N` | Previous | Go to previous segment |
| `c` | Context | Set or confirm context name |
| `?` | Help | Show help panel |
| `q` | Quit | Save progress and exit |

### Action Details

**Approve (`a`):**

* Marks the segment as correctly assigned
* Prompts for speaker ID if not already assigned
* Automatically calls `speaker_samples extract` to add the segment to the speaker's sample collection
* Advances to the next segment

**Reject (`r`):**

* Marks the assignment as incorrect
* Optionally prompts for rejection notes
* Advances to the next segment

**Skip (`s`):**

* Defers decision without marking approve/reject
* Useful for uncertain cases to revisit later
* Advances to the next segment

**Play (`p`):**

* Plays the audio segment using mpv or ffplay
* Helps verify the speaker assignment by listening

**Edit (`e`):**

* Opens speaker profile editing
* Requires `speaker_detection` tool
* Allows changing display name and other profile fields

## Session Persistence

Review sessions are automatically saved to:

```
~/.cache/speaker-review/session.yaml
```

The session file stores:

* Recording identifier (b3sum)
* Audio and transcript paths
* Context name
* Current segment index
* All review decisions with timestamps
* Session start and update times

Session persistence allows you to:

* Interrupt a review and continue later with `--continue`
* Resume exactly where you left off
* Keep all decisions even after quitting

The cache directory respects `XDG_CACHE_HOME` if set.

## Audio Playback

`speaker-review` supports audio playback for verifying speaker assignments. It tries players in order:

1. **mpv** (preferred) - Uses `--start` and `--end` flags for precise segment playback
2. **ffplay** - Fallback using `-ss` and `-t` flags

If neither player is available, playback is disabled with a warning message.

Install a player:

```bash
# Arch Linux
sudo pacman -S mpv

# Ubuntu/Debian
sudo apt install mpv

# macOS
brew install mpv
```

## Integration with speaker-assign

`speaker-review` reads assignment files created by `speaker-assign`:

```bash
# 1. Run automated assignment
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings

# 2. Review the assignments
./speaker-review meeting.mp3

# 3. Approved segments are extracted to speaker samples
```

Assignment files are read from:

```
$SPEAKERS_EMBEDDINGS_DIR/assignments/{b3sum}.yaml
```

## Integration with speaker_samples

When you approve a segment, `speaker-review` automatically calls:

```bash
speaker_samples extract <audio> \
    -t <transcript> \
    -l <speaker_label> \
    -s <speaker_id> \
    --max-segments 1 \
    -q
```

This extracts the approved segment and adds it to the speaker's sample collection with full provenance metadata. This grows the embeddings database over time through human-verified samples.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEAKERS_EMBEDDINGS_DIR` | `~/.config/speakers_embeddings` | Root storage directory |
| `XDG_CACHE_HOME` | `~/.cache` | Cache directory base for session files |

## Storage

Session files and related data:

```
~/.cache/
└── speaker-review/
    └── session.yaml          # Current review session state

$SPEAKERS_EMBEDDINGS_DIR/
├── assignments/              # Speaker assignments (read by speaker-review)
│   └── abc123def456.yaml
├── catalog/                  # Recording catalog (for context lookup)
│   └── abc123def456.yaml
├── db/                       # Speaker profiles (for editing)
│   └── alice.yaml
└── samples/                  # Voice samples (written on approve)
    └── alice/
        └── sample-001.mp3
```

## Workflow Example

Complete workflow from recording to reviewed samples:

```bash
# 1. Add recording to catalog
./speaker-catalog add meeting.mp3 --context team-standup

# 2. Transcribe with speaker diarization
./stt_speechmatics.py meeting.mp3 -o meeting.json

# 3. Register transcript
./speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.json

# 4. Run automated assignment
./speaker-assign assign meeting.mp3 \
    --transcript meeting.json \
    --use-embeddings

# 5. Review assignments interactively
./speaker-review meeting.mp3

# During review:
# - Press 'p' to play each segment
# - Press 'a' to approve correct assignments (extracts samples)
# - Press 'r' to reject incorrect assignments
# - Press 's' to skip uncertain cases
# - Press 'q' to quit (progress is saved)

# 6. Continue later if needed
./speaker-review --continue

# 7. Check final status
./speaker-review status
```

## Related Tools

* [`speaker-assign`](./speaker-assign.README.md) - Multi-signal speaker name assignment
* [`speaker-catalog`](./speaker-catalog.README.md) - Recording inventory and processing state management
* [`speaker_detection`](./speaker_detection.README.md) - Speaker profile and embedding management
* [`speaker_samples`](./speaker_samples.README.md) - Voice sample extraction with provenance
* [`speaker_segments`](./speaker_segments.README.md) - Extract segment timestamps from transcripts
* [`stt_speechmatics.py`](./stt_speechmatics.README.md) - Speechmatics STT with speaker diarization
* [`stt_assemblyai.py`](./stt_assemblyai.README.md) - AssemblyAI transcription
