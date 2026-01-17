# Speaker Diarization System - Future Work & Architecture

## Problem Statement

Managing speaker diarization across recordings with:

* Mixed processing states (unprocessed, processed with old/new tooling)
* Multiple transcription backends (AssemblyAI, Speechmatics)
* Incremental review workflows (partial reviews, revisits)
* Growing speaker embedding database with confidence levels
* Need for provenance tracking and intelligent re-processing

## Core Principles

### UNIX Philosophy

```
1. Each tool does ONE thing well
2. Tools work together via standard interfaces (JSON, YAML, pipes)
3. Text streams as universal interface
4. Small, composable, testable units
```

### Tool Ecosystem Pattern

Following `git <command>` / `cargo <command>` convention:

```
speaker <command> [args]     →  dispatches to speaker-<command>

Examples:
  speaker detect list        →  speaker-detect list
  speaker samples extract    →  speaker-samples extract
  speaker catalog add        →  speaker-catalog add
  speaker review start       →  speaker-review start
```

This allows:

* Independent development/testing of each subcommand
* Shell completion at subcommand level
* Clear separation of concerns
* Easy to add new commands

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RECORDING CATALOG                               │
│  speaker-catalog: inventory of recordings + processing state            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRANSCRIPTION LAYER                               │
│  stt_speechmatics, stt_assemblyai: raw transcription + diarization      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SPEAKER ASSIGNMENT LAYER                            │
│  speaker-assign: combine signals → confident speaker names              │
│                                                                          │
│  Signals:                                                                │
│    • Embedding matches (speaker-detect identify)                        │
│    • LLM conversation analysis ("Alice, what do you think?")            │
│    • Context rules (known meeting participants)                         │
│    • Cross-backend agreement                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         REVIEW LAYER                                     │
│  speaker-review: interactive sample review → feedback loop              │
│                                                                          │
│    approve → extract sample → add to embeddings database                │
│    reject  → mark incorrect → invalidate dependent embeddings           │
│    skip    → defer decision                                             │
│    edit    → fix speaker profile (names, context)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       QUALITY & REPORTING                                │
│  speaker-report: coverage, confidence, re-processing recommendations    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tool Inventory

### Existing Tools (maintain as-is)

| Tool | Scope | Status |
|------|-------|--------|
| `speaker_detection` | Profile/embedding CRUD, identify, verify | ✅ Stable |
| `speaker_samples` | Sample extraction, review status | ✅ Stable |
| `speaker_segments` | Timestamp extraction from transcripts | ✅ Stable |
| `stt_speechmatics` | Speechmatics transcription | ✅ Stable |
| `stt_assemblyai` | AssemblyAI transcription | ✅ Stable |

### New Tools (proposed)

| Tool | Scope | Priority |
|------|-------|----------|
| `speaker-catalog` | Recording inventory & processing state | P1 |
| `speaker-assign` | Multi-signal speaker name assignment | P1 |
| `speaker-review` | Interactive TUI for sample review | P1 |
| `speaker-process` | Batch processing orchestration | P2 |
| `speaker-report` | Quality metrics & recommendations | P2 |
| `speaker-llm` | LLM-based conversation analysis | P2 |
| `speaker` | Umbrella dispatcher (optional) | P3 |

---

## Data Models

### Recording Catalog Entry

```yaml
# $SPEAKERS_EMBEDDINGS_DIR/catalog/{recording_b3sum}.yaml

schema_version: 1

recording:
  path: /archive/meetings/2026-01-15-standup.mp3
  b3sum: "abc123def456..."
  duration_sec: 1847.3
  discovered_at: "2026-01-15T10:00:00Z"

context:
  name: "team-standup"                    # Context for name resolution
  expected_speakers: [alice, bob, carol]  # Hints for assignment
  project: "acme-project"
  tags: [recurring, internal]

transcriptions:
  - backend: speechmatics
    version: "speechmatics-v2.1"
    transcript_path: "./2026-01-15-standup.speechmatics.json"
    processed_at: "2026-01-15T10:30:00Z"
    tool_version: "stt_speechmatics-1.2.0"
    config:
      language: en
      diarization: true
      speakers_tag: team-standup
    results:
      speakers_detected: 3
      word_count: 4521

  - backend: assemblyai
    version: "assemblyai-2024-q4"
    transcript_path: "./2026-01-15-standup.assemblyai.json"
    processed_at: "2026-01-15T11:00:00Z"
    tool_version: "stt_assemblyai-1.1.0"
    results:
      speakers_detected: 3
      word_count: 4498

speaker_assignment:
  version: 1
  assigned_at: "2026-01-15T12:00:00Z"
  method: "speaker-assign-v1.0"
  mappings:
    S1:
      speaker_id: alice
      confidence: high
      signals:
        - type: embedding_match
          score: 0.89
          embedding_id: emb-abc123
        - type: llm_name_detection
          evidence: "Bob said 'Alice, can you...'"
        - type: context_expected

    S2:
      speaker_id: bob
      confidence: medium
      signals:
        - type: embedding_match
          score: 0.72

    S3:
      speaker_id: null
      confidence: unassigned
      candidates:
        - speaker_id: carol
          score: 0.45
        - speaker_id: dave
          score: 0.41

review:
  status: partial           # none | partial | complete
  coverage_pct: 40
  sessions:
    - at: "2026-01-16T09:00:00Z"
      reviewer: "gw"
      samples_presented: 10
      approved: 7
      rejected: 2
      skipped: 1
      notes: "S3 sounds like Carol but low confidence"
```

### Confidence Taxonomy

```yaml
# Embedding Trust Levels (existing)
embedding_trust:
  high:        # All source samples manually reviewed/approved
  medium:      # Some samples reviewed, some auto-extracted
  low:         # All samples auto-extracted, no review
  invalidated: # Any source sample rejected

# Speaker Assignment Confidence (new)
assignment_confidence:
  confirmed:        # Manually verified by reviewer
  high:             # Multiple independent signals agree (embedding + LLM + context)
  medium:           # Single strong signal or partial agreement
  low:              # Heuristic only, needs review
  unassigned:       # No match found

# Recording Review Status (new)
review_status:
  none:      # No samples reviewed
  partial:   # Some samples reviewed (coverage_pct < 100)
  complete:  # All speaker segments have reviewed samples
```

---

## User Workflows

### Workflow 1: Fresh Recording Processing

```bash
# 1. Add recording to catalog
speaker-catalog add meeting.mp3 --context team-standup

# 2. Process with transcription backends
stt_speechmatics meeting.mp3 -d --speakers-tag team-standup
stt_assemblyai meeting.mp3 --speaker-labels

# 3. Register transcripts in catalog
speaker-catalog register-transcript meeting.mp3 \
    --backend speechmatics \
    --transcript meeting.speechmatics.json

# 4. Run speaker assignment (combines all signals)
speaker-assign meeting.mp3 \
    --transcript meeting.speechmatics.json \
    --use-embeddings \
    --use-llm \
    --context team-standup

# 5. Review samples interactively
speaker-review meeting.mp3

# 6. Check status
speaker-catalog status meeting.mp3
```

### Workflow 2: Batch Processing

```bash
# Process all unprocessed recordings
speaker-catalog list --status unprocessed | \
    xargs -I {} speaker-process {} --backend speechmatics,assemblyai

# Or with parallel processing
speaker-catalog list --status unprocessed --format paths | \
    parallel -j4 speaker-process {} --backend speechmatics
```

### Workflow 3: Incremental Review Session

```bash
# Start review session (picks up where left off)
speaker-review --continue

# Or review specific recording
speaker-review meeting.mp3

# Or review all recordings needing attention
speaker-catalog list --needs-review | xargs speaker-review
```

### Workflow 4: Re-processing After Tool Upgrade

```bash
# Find recordings processed with old version
speaker-catalog list --tool-version "stt_speechmatics<1.2.0"

# Re-process (preserves review state)
speaker-catalog list --tool-version "stt_speechmatics<1.2.0" | \
    xargs speaker-process --force-reprocess

# Compare old vs new
speaker-report diff meeting.mp3 --old-version 1.1.0 --new-version 1.2.0
```

### Workflow 5: Growing Embeddings Database

```bash
# Start with high-confidence embeddings only
speaker-assign meeting.mp3 --min-trust high

# After more reviews, include medium
speaker-assign meeting.mp3 --min-trust medium

# Compare results
speaker-report compare-assignments meeting.mp3 \
    --run1 "high-only" \
    --run2 "high+medium"
```

---

## Tool Specifications

### speaker-catalog

**Purpose**: Inventory of recordings with processing/review state.

```bash
# Commands
speaker-catalog add <audio> [--context NAME] [--tags TAG,...]
speaker-catalog list [--status STATUS] [--context NAME] [--needs-review]
speaker-catalog show <audio>
speaker-catalog status <audio>
speaker-catalog register-transcript <audio> --backend NAME --transcript FILE
speaker-catalog set-context <audio> --context NAME [--expected-speakers ID,...]
speaker-catalog remove <audio> [--force]

# Status values: unprocessed, transcribed, assigned, reviewed, complete
```

**Storage**: `$SPEAKERS_EMBEDDINGS_DIR/catalog/{b3sum}.yaml`

### speaker-assign

**Purpose**: Combine multiple signals to assign speaker names.

```bash
# Commands
speaker-assign <audio> --transcript FILE [OPTIONS]

Options:
  --use-embeddings          Use speaker_detection identify
  --min-trust LEVEL         Minimum embedding trust (high|medium|low)
  --use-llm                 Use LLM conversation analysis
  --llm-model MODEL         LLM model for analysis
  --context NAME            Speaker context for name resolution
  --threshold FLOAT         Minimum confidence for assignment
  --output FILE             Output assignment results
  --dry-run                 Show assignments without saving

# Output signals used
speaker-assign show <audio>  # Show current assignments with signal details
```

**Algorithm**:

```
1. Load transcript, extract speaker segments
2. For each speaker label (S1, S2, ...):
   a. Run embedding identification (if --use-embeddings)
   b. Run LLM name detection (if --use-llm)
   c. Check context expected speakers
   d. Combine signals with confidence weights
   e. Assign if confidence > threshold, else mark unassigned
3. Detect conflicts (same person assigned to multiple labels)
4. Save to catalog
```

### speaker-review

**Purpose**: Interactive TUI for reviewing speaker samples.

```bash
# Commands
speaker-review [<audio>]            # Review specific or next-in-queue
speaker-review --continue           # Continue previous session
speaker-review --context NAME       # Review all in context

# During review (TUI keybindings)
a = approve (extract sample, add to database)
r = reject (mark incorrect)
s = skip (defer, no decision)
e = edit speaker (open profile editor)
c = set/confirm context
n = next sample
p = previous sample
q = quit (save progress)
? = help
```

**Features**:

* Audio playback of segment
* Show transcript text
* Show assignment confidence and signals
* Show speaker profile (names, existing samples)
* Quick context switching
* Progress indicator

### speaker-llm

**Purpose**: LLM-based speaker name detection from conversation.

```bash
speaker-llm analyze <transcript> [--model MODEL] [--context NAME]

# Output: JSON with detected names and evidence
{
  "detections": [
    {
      "speaker_label": "S1",
      "detected_name": "Alice",
      "confidence": 0.85,
      "evidence": [
        {"type": "direct_address", "text": "Alice, what do you think?", "speaker": "S2"},
        {"type": "self_reference", "text": "This is Alice speaking", "speaker": "S1"}
      ]
    }
  ]
}
```

### speaker-report

**Purpose**: Quality metrics and re-processing recommendations.

```bash
speaker-report status                    # Overall system status
speaker-report coverage [--context NAME] # Review coverage by context
speaker-report confidence [--below PCT]  # Recordings below confidence threshold
speaker-report stale [--days N]          # Recordings needing re-processing
speaker-report compare-backends <audio>  # Compare Speechmatics vs AssemblyAI
speaker-report diff <audio> --v1 X --v2 Y # Compare processing versions
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (per tool)
   * `test_{tool}.py` - Command parsing, core logic
   * Mock external services (STT APIs, LLM)

2. **Integration Tests**
   * Full workflow with real audio
   * Multi-tool pipelines

3. **Regression Tests**
   * Golden file comparisons
   * Assignment stability across versions

### Test Datasets

```
evals/speaker_diarization/
├── audio/
│   ├── two-speakers-clear.wav       # Easy case
│   ├── three-speakers-overlap.wav   # Challenging
│   ├── single-speaker-long.wav      # Edge case
│   └── noisy-conference.wav         # Difficult
├── transcripts/
│   ├── *.speechmatics.json
│   └── *.assemblyai.json
├── expected/
│   ├── assignments/                 # Expected speaker assignments
│   └── samples/                     # Expected extracted samples
└── scenarios/
    ├── fresh_processing.py          # New recording workflow
    ├── incremental_review.py        # Partial review workflow
    ├── reprocessing.py              # Version upgrade workflow
    └── embedding_growth.py          # Learning from reviews
```

### Test Scenarios

```python
# Scenario: Fresh Processing
def test_fresh_processing():
    """New recording → transcribe → assign → review → embeddings grow"""
    # 1. Add to catalog
    # 2. Transcribe with both backends
    # 3. Run assignment (no embeddings yet)
    # 4. Review samples (approve some)
    # 5. Verify embeddings created
    # 6. Re-run assignment (should improve)

# Scenario: Incremental Review
def test_incremental_review():
    """Partial review → continue → complete"""
    # 1. Start review, approve 3 samples, quit
    # 2. Verify progress saved
    # 3. Continue review, approve remaining
    # 4. Verify status = complete

# Scenario: Re-processing Preservation
def test_reprocess_preserves_reviews():
    """Upgrade tool → reprocess → reviews preserved"""
    # 1. Process with v1.0
    # 2. Review some samples
    # 3. Upgrade to v1.1, reprocess
    # 4. Verify reviews still valid
    # 5. Verify only changed segments need re-review

# Scenario: Confidence Levels
def test_confidence_levels():
    """High-only vs all embeddings comparison"""
    # 1. Create mix of high/medium/low trust embeddings
    # 2. Assign with --min-trust high
    # 3. Assign with --min-trust low
    # 4. Compare results
    # 5. Verify high-trust is subset with higher accuracy
```

---

## Implementation Phases

### Phase 1: Foundation (P1)

* [ ] `speaker-catalog` - Recording inventory
* [ ] Catalog data model and storage
* [ ] Integration with existing tools
* [ ] Basic `speaker-report status`

### Phase 2: Assignment (P1)

* [ ] `speaker-assign` - Multi-signal assignment
* [ ] Embedding signal integration
* [ ] Context-based assignment
* [ ] Confidence scoring

### Phase 3: Review (P1)

* [ ] `speaker-review` - TUI interface
* [ ] Audio playback integration
* [ ] Review state persistence
* [ ] Feedback to embeddings

### Phase 4: LLM Integration (P2)

* [ ] `speaker-llm` - Conversation analysis
* [ ] Name detection prompts
* [ ] Integration with speaker-assign
* [ ] Caching for efficiency

### Phase 5: Batch & Reporting (P2)

* [ ] `speaker-process` - Batch orchestration
* [ ] Full `speaker-report` suite
* [ ] Re-processing recommendations
* [ ] Quality dashboards

### Phase 6: Umbrella Command (P3)

* [ ] `speaker` dispatcher
* [ ] Unified shell completions
* [ ] Plugin architecture for extensions

---

## Open Questions

1. **Storage format**: YAML vs JSON for catalog entries?
   * YAML: human-readable, comments
   * JSON: faster parsing, wider tool support

2. **Review TUI framework**: textual vs rich vs custom?
   * textual: full TUI, complex
   * rich: simpler, good for prompts
   * Custom: minimal dependencies

3. **LLM integration**: local vs API?
   * Local (ollama): privacy, no cost, slower
   * API (claude/gpt): faster, cost, privacy concerns

4. **Audio playback in review**: how?
   * System player (mpv, ffplay)
   * Python library (pygame, sounddevice)
   * Web UI alternative

5. **Multi-user review**: needed?
   * Single user: simpler
   * Multi-user: reviewer attribution, conflict resolution

---

## Related Documents

* [`speaker_detection.README.md`](./speaker_detection.README.md) - Profile/embedding management
* [`speaker_samples.README.md`](./speaker_samples.README.md) - Sample extraction
* [`speaker_segments.README.md`](./speaker_segments.README.md) - Timestamp extraction
* [`CONTRIBUTING.md`](./CONTRIBUTING.md) - Environment variables, development setup
* [`ramblings/2026-01-12--speaker-identity-system.md`](./ramblings/2026-01-12--speaker-identity-system.md) - Original architecture

---

## Appendix: Signal Combination Algorithm

```python
def combine_signals(signals: list[Signal], context: Context) -> Assignment:
    """
    Combine multiple signals into a single speaker assignment.

    Signal types and weights:
      - embedding_match: 0.4 (scaled by embedding trust)
      - llm_name_detection: 0.3
      - context_expected: 0.2
      - cross_backend_agreement: 0.1

    Trust multipliers for embedding_match:
      - high: 1.0
      - medium: 0.7
      - low: 0.4
    """
    scores = defaultdict(float)
    evidence = defaultdict(list)

    for signal in signals:
        if signal.type == "embedding_match":
            trust_mult = {"high": 1.0, "medium": 0.7, "low": 0.4}[signal.trust]
            weight = 0.4 * trust_mult
            scores[signal.speaker_id] += weight * signal.score
            evidence[signal.speaker_id].append(signal)

        elif signal.type == "llm_name_detection":
            weight = 0.3
            scores[signal.speaker_id] += weight * signal.confidence
            evidence[signal.speaker_id].append(signal)

        elif signal.type == "context_expected":
            weight = 0.2
            if signal.speaker_id in context.expected_speakers:
                scores[signal.speaker_id] += weight
                evidence[signal.speaker_id].append(signal)

        elif signal.type == "cross_backend_agreement":
            weight = 0.1
            scores[signal.speaker_id] += weight
            evidence[signal.speaker_id].append(signal)

    if not scores:
        return Assignment(speaker_id=None, confidence="unassigned", signals=[])

    best_id = max(scores, key=scores.get)
    best_score = scores[best_id]

    confidence = (
        "high" if best_score > 0.7 else
        "medium" if best_score > 0.4 else
        "low"
    )

    return Assignment(
        speaker_id=best_id,
        confidence=confidence,
        score=best_score,
        signals=evidence[best_id]
    )
```

---

*Last updated: 2026-01-17*
