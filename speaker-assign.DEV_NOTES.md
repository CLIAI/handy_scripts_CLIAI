# speaker-assign Developer Notes

Developer documentation for the `speaker-assign` tool - a multi-signal speaker name assignment system.

## 1. Architecture Overview

### Signal Combination Algorithm

The `speaker-assign` tool combines multiple independent signals to assign speaker identities (e.g., `john-smith`) to transcript labels (e.g., `S1`, `S2`, `A`, `B`). The algorithm follows a weighted voting approach:

```
                         ┌─────────────────────┐
                         │   Transcript File   │
                         │  (AssemblyAI/SM)    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Extract Speaker    │
                         │  Labels (S1, S2...) │
                         └──────────┬──────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Embedding     │      │   LLM Name      │      │   Context       │
│   Match Signal  │      │   Detection     │      │   Expected      │
│   (40% weight)  │      │   (30% weight)  │      │   (20% weight)  │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         │   ┌────────────────────┴────────────────────┐   │
         │   │                                         │   │
         │   ▼                                         ▼   │
         │  ┌─────────────────────────────────────────────┐│
         │  │        Cross-Backend Agreement             ││
         │  │              (10% weight)                  ││
         │  └─────────────────────────────────────────────┘│
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Weighted Scoring   │
                       │  + Trust Multiplier │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   Confidence Level  │
                       │   + Best Candidate  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Assignment Result  │
                       │  (saved to YAML)    │
                       └─────────────────────┘
```

### Scoring Formula

For each candidate speaker ID, the final score is computed as:

```
score(speaker_id) = sum over all signals s:
    weight(s.type) * trust_multiplier(s) * s.score
```

Where:

* `weight(s.type)` is the signal type weight from `SIGNAL_WEIGHTS`
* `trust_multiplier(s)` is applied only to embedding signals based on trust level
* `s.score` is the raw confidence from the signal source (0.0 to 1.0)

The candidate with the highest final score becomes the assignment, subject to the threshold check.

## 2. Signal Weights and Rationale

```python
SIGNAL_WEIGHTS = {
    "embedding_match": 0.4,        # Voiceprint - strongest biometric signal
    "llm_name_detection": 0.3,     # Content analysis - finds names in conversation
    "context_expected": 0.2,       # Prior knowledge - expected participants
    "cross_backend_agreement": 0.1 # Corroboration bonus - multiple backends agree
}
```

### Weight Rationale

**Embedding Match (40%)**

* Voiceprint embeddings are the strongest biometric signal
* Direct voice comparison is more reliable than textual inference
* However, not given 100% weight because embeddings can be wrong (poor quality samples, voice changes over time, similar-sounding speakers)

**LLM Name Detection (30%)**

* Conversational context often reveals identities through greetings, name mentions, or role references
* Example: "Thanks for joining us, Dr. Smith" clearly identifies a speaker
* Powerful but prone to hallucination; needs corroboration from other signals

**Context Expected (20%)**

* Prior knowledge about who should be in the recording (e.g., a weekly standup has known team members)
* Useful as a tie-breaker when multiple candidates have similar scores
* Lower weight because it is circumstantial, not evidence-based

**Cross-Backend Agreement (10%)**

* When multiple embedding backends (Speechmatics, PyAnnote, SpeechBrain) agree on a match, confidence increases
* Small bonus weight because it is a meta-signal (derived from other signals)
* Currently a placeholder for future multi-backend implementation

## 3. Trust Multipliers and Why

```python
TRUST_MULTIPLIERS = {
    "high": 1.0,         # Verified samples, full weight
    "medium": 0.7,       # Partially verified, reduced confidence
    "low": 0.4,          # Unverified, minimal confidence
    "invalidated": 0.0,  # Explicitly marked bad, ignore
    "unknown": 0.5,      # No trust info, neutral stance
}
```

### Trust Level Rationale

Trust levels come from the `speaker_samples` tool where operators review and validate sample quality.

**High Trust (1.0x)**

* Sample has been manually verified as belonging to the correct speaker
* Used for high-stakes applications where accuracy is critical
* Full contribution to the weighted score

**Medium Trust (0.7x)**

* Sample partially verified or auto-enrolled with good quality indicators
* Reasonable confidence but not absolute certainty
* 30% penalty reflects residual uncertainty

**Low Trust (0.4x)**

* Unverified sample, possibly auto-enrolled without review
* Kept in the scoring pool but with significant penalty
* Useful when no better evidence exists

**Invalidated (0.0x)**

* Operator explicitly marked the sample as incorrect or corrupted
* Completely excluded from scoring
* Prevents known-bad data from polluting assignments

**Unknown (0.5x)**

* No trust metadata available (legacy embeddings, external sources)
* Neutral treatment - neither trusted nor distrusted
* Encourages migration to trust-tracked samples

### Effective Weight Calculation

The trust multiplier applies to the base signal weight:

```
effective_weight = base_weight * trust_multiplier

# Example: embedding match with medium trust
effective = 0.4 * 0.7 = 0.28
```

This means a medium-trust embedding match has less influence than a high-trust embedding match, encouraging operators to verify their samples.

## 4. Graceful Degradation

The tool is designed to work even when dependent tools are unavailable.

### When speaker_detection Is Unavailable

```python
def collect_embedding_signals(...) -> list[Signal]:
    try:
        result = subprocess.run(
            ["speaker_detection", "identify", str(audio_path)],
            ...
        )
        # Process results...
    except FileNotFoundError:
        pass  # Gracefully return empty signal list
    return signals
```

Behavior:

* No embedding signals are collected
* Tool continues with LLM and context signals only
* Assignment still possible if other signals provide enough confidence
* User sees reduced accuracy but tool remains functional

### When speaker-llm Is Unavailable

```python
def collect_llm_signals(...) -> list[Signal]:
    try:
        result = subprocess.run(
            ["speaker-llm", "analyze", str(transcript_path)],
            ...
        )
        # Process results...
    except FileNotFoundError:
        pass  # Gracefully return empty signal list
    return signals
```

Behavior:

* No LLM name detection signals are collected
* Embedding and context signals still contribute
* Common scenario in environments without LLM access

### When No Signals Are Available

```python
if not scores:
    return Assignment(
        speaker_label=speaker_label,
        speaker_id=None,
        confidence="unassigned",
        score=0.0,
        signals=[],
        candidates=[],
    )
```

Behavior:

* Speaker label is marked as "unassigned"
* No false assignments are made
* Operator must manually assign or provide more context

### Degradation Hierarchy

```
Full capability:     Embeddings + LLM + Context → High accuracy
Embeddings only:     Embeddings + Context      → Good accuracy
LLM only:            LLM + Context             → Moderate accuracy
Context only:        Context                   → Low accuracy (tie-breaker level)
Nothing:             Unassigned                → Manual intervention required
```

## 5. Integration Points with Other Tools

### Upstream Dependencies

**speaker_detection** (`identify` command)

* Provides embedding match signals
* Called as subprocess: `speaker_detection identify <audio> --format json`
* Returns JSON array of matches with speaker_id, score, trust_level, backend

**speaker-llm** (`analyze` command) - (planned/optional)

* Provides LLM name detection signals
* Called as subprocess: `speaker-llm analyze <transcript> [--context NAME]`
* Returns JSON with detections array containing speaker_label, detected_name, confidence, evidence

**speaker-catalog** (storage layer)

* Provides catalog entries with context information
* Direct file access to `catalog/{b3sum}.yaml`
* Extracts context.name and context.expected_speakers

### Downstream Integration

**stt_speechmatics_speaker_mapper.py**

* Consumes assignment results to replace S1/S2 with actual names
* Reads from `assignments/{b3sum}.yaml`

**speaker-catalog** (status progression)

* Assignment creation triggers status change: `transcribed` -> `assigned`
* Catalog queries can filter by assignment status

### Data Flow

```
┌────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Audio File     │────▶│ speaker-catalog  │────▶│ Catalog Entry     │
│ (source)       │     │ add              │     │ (context info)    │
└────────────────┘     └──────────────────┘     └───────────────────┘
        │                                                │
        ▼                                                │
┌────────────────┐     ┌──────────────────┐              │
│ Transcription  │────▶│ Transcript JSON  │              │
│ (STT service)  │     │ (speakers: S1..) │              │
└────────────────┘     └────────┬─────────┘              │
                                │                        │
                                ▼                        ▼
                       ┌──────────────────────────────────┐
                       │         speaker-assign           │
                       │  (combines all signals)          │
                       └────────────────┬─────────────────┘
                                        │
                                        ▼
                       ┌────────────────────────────────┐
                       │  assignments/{b3sum}.yaml      │
                       │  S1 -> john-smith              │
                       │  S2 -> jane-doe                │
                       └────────────────────────────────┘
```

### Shared Storage

All tools share the `$SPEAKERS_EMBEDDINGS_DIR` root:

```
$SPEAKERS_EMBEDDINGS_DIR/
├── catalog/           # speaker-catalog entries
│   └── {b3sum}.yaml
├── assignments/       # speaker-assign output ← THIS TOOL
│   └── {b3sum}.yaml
├── db/                # speaker_detection profiles
│   └── {speaker_id}/
└── samples/           # speaker_samples extracted audio
    └── {speaker_id}/
```

## 6. Testing Approach

### Unit Testing Strategy

**Signal Collection Functions**

```python
def test_collect_context_signals():
    """Context signals should be created for each expected speaker."""
    signals = collect_context_signals(
        speaker_label="S1",
        context_name="team-standup",
        expected_speakers=["alice", "bob", "charlie"]
    )
    assert len(signals) == 3
    assert all(s.type == "context_expected" for s in signals)
    assert {s.speaker_id for s in signals} == {"alice", "bob", "charlie"}
```

**Signal Combination Logic**

```python
def test_combine_signals_selects_highest_score():
    """Combination should select speaker with highest weighted score."""
    signals = [
        Signal(type="embedding_match", speaker_id="alice", score=0.9,
               evidence={"trust_level": "high"}),
        Signal(type="embedding_match", speaker_id="bob", score=0.7,
               evidence={"trust_level": "high"}),
    ]
    assignment = combine_signals("S1", signals, threshold=0.3)
    assert assignment.speaker_id == "alice"
```

**Trust Multiplier Application**

```python
def test_trust_multiplier_reduces_low_trust():
    """Low trust embeddings should have reduced weight."""
    signals = [
        Signal(type="embedding_match", speaker_id="alice", score=0.8,
               evidence={"trust_level": "low"}),    # 0.4 * 0.4 * 0.8 = 0.128
        Signal(type="embedding_match", speaker_id="bob", score=0.5,
               evidence={"trust_level": "high"}),   # 0.4 * 1.0 * 0.5 = 0.200
    ]
    assignment = combine_signals("S1", signals, threshold=0.1)
    assert assignment.speaker_id == "bob"  # Lower raw score but higher trust
```

### Integration Testing

**Subprocess Invocation**

```python
def test_assign_command_integration(self):
    """Full assignment flow from audio + transcript."""
    result = subprocess.run(
        ["./speaker-assign", "assign", audio_path,
         "--transcript", transcript_path,
         "--dry-run", "--format", "json"],
        capture_output=True,
        text=True
    )
    self.assertEqual(result.returncode, 0)
    output = json.loads(result.stdout)
    self.assertIn("mappings", output)
```

**Graceful Degradation**

```python
def test_works_without_speaker_detection(self):
    """Tool should still work when speaker_detection is unavailable."""
    # Temporarily rename speaker_detection or use PATH manipulation
    with modified_path(exclude="speaker_detection"):
        result = subprocess.run(
            ["./speaker-assign", "assign", audio_path,
             "--transcript", transcript_path,
             "--expected-speakers", "alice,bob",
             "--dry-run"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
```

### Test Categories

* **Unit tests**: Signal collection, combination logic, trust multipliers
* **Integration tests**: Full command flows with real/mock dependencies
* **Edge cases**: Empty transcripts, no speakers, all signals unavailable
* **Format tests**: AssemblyAI vs Speechmatics transcript parsing

### Isolated Test Environment

```python
class TestSpeakerAssign(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["SPEAKERS_EMBEDDINGS_DIR"] = self.temp_dir

        # Create mock catalog entry with context
        catalog_dir = Path(self.temp_dir) / "catalog"
        catalog_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
```

## 7. Future Enhancements

### Batch Assignment

Process multiple recordings in a single invocation:

```bash
speaker-assign batch --dir /recordings --context team-standup
speaker-assign batch --catalog-query "status=transcribed"
```

Implementation considerations:

* Parallel processing with `concurrent.futures`
* Progress reporting for long batches
* Summary report: assigned, unassigned, errors
* Atomic writes to prevent partial state

### Learning from Feedback

Improve assignment accuracy by incorporating correction feedback:

```bash
# Record a correction
speaker-assign correct <audio> --label S1 --to john-smith --reason "operator review"

# View correction history
speaker-assign corrections <audio>

# Apply corrections to retrain embeddings
speaker-assign apply-corrections --speaker john-smith
```

Implementation considerations:

* Store corrections in `corrections/{b3sum}.yaml`
* Track correction reason and timestamp
* Optional: feed corrections back to embedding enrollment
* Build correction statistics for quality monitoring

### Confidence Calibration

Auto-tune confidence thresholds based on historical accuracy:

```bash
speaker-assign calibrate --corpus /verified-recordings
```

Implementation considerations:

* Compare assignments against verified ground truth
* Compute precision/recall at different thresholds
* Suggest optimal threshold for target accuracy
* Per-context calibration (some contexts are harder than others)

### Multi-Backend Embedding Support

Full implementation of cross-backend agreement signal:

```python
# Future: collect signals from multiple backends
embedding_signals_sm = collect_embedding_signals(backend="speechmatics", ...)
embedding_signals_pa = collect_embedding_signals(backend="pyannote", ...)

# Check for agreement
if (embedding_signals_sm[0].speaker_id == embedding_signals_pa[0].speaker_id):
    signals.append(Signal(
        type="cross_backend_agreement",
        speaker_id=embedding_signals_sm[0].speaker_id,
        score=1.0,
    ))
```

### Interactive Assignment Mode

TUI for manual review and correction:

```bash
speaker-assign interactive <audio> --transcript FILE
```

Features:

* Show top candidates with confidence scores
* Play audio segments for verification
* Accept/reject/override assignments
* Mark unassigned speakers for manual entry

### Export Formats

Support additional output formats for integration:

```bash
speaker-assign show <audio> --format srt   # SubRip with speaker names
speaker-assign show <audio> --format vtt   # WebVTT with speaker cues
speaker-assign show <audio> --format csv   # Spreadsheet-friendly
```

---

*Part of the speaker-* tool ecosystem for managing speaker identification workflows.*
