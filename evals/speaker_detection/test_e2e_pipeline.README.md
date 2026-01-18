# Test Collection: e2e

End-to-end pipeline integration tests for the speaker-* tool ecosystem.

## Purpose

These tests validate the complete speaker diarization workflow from audio input to final speaker identification. They exercise the full pipeline integration between tools.

## Test File

`evals/speaker_detection/test_e2e_pipeline.py`

## Running

```bash
./run_speaker_diarization_tests.sh e2e
```

Or directly:

```bash
python evals/speaker_detection/test_e2e_pipeline.py
```

## Test Count

17 tests (2 main scenarios with multiple assertions)

## Tests Included

| Test | Description |
|------|-------------|
| `test_e2e_pipeline` | Complete single-recording pipeline: catalog -> transcribe -> assign -> review |
| `test_multi_recording_pipeline` | Multiple recordings with shared speakers, context propagation |

## Pipeline Flow Tested

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Audio     │───>│   catalog   │───>│   assign    │───>│   review    │
│   Input     │    │   (add)     │    │   (map)     │    │   (verify)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │
                         v                   v
                   ┌───────────┐       ┌───────────┐
                   │ Transcript│       │ Profiles  │
                   │  (mock)   │       │  (mock)   │
                   └───────────┘       └───────────┘
```

## What Gets Validated

* **Catalog operations**: Recording addition, status tracking
* **Assignment flow**: Label mapping, confidence scoring
* **Data persistence**: Assignments stored and retrievable
* **Context propagation**: Expected speakers flow through pipeline
* **Multi-recording**: Shared speaker profiles across recordings

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` with synthetic audio and mock transcripts.

## Docker Support

E2E tests can run in Docker for reproducible CI/CD:

```bash
./run_speaker_diarization_tests.sh docker
```

## Related Documentation

* Test runner: `run_speaker_diarization_tests.sh.README.md`
* General testing: `evals/TESTING.md`
