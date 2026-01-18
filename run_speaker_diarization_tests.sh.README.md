# Speaker-* Tool Test Runner

Unified test runner for the speaker diarization tool ecosystem.

## Overview

`run_speaker_diarization_tests.sh` provides a single entry point for running all tests in the speaker-* tool suite. It supports running individual test collections, combined collections, and Docker-isolated testing.

## Usage

```bash
./run_speaker_diarization_tests.sh [COLLECTION]
./run_speaker_diarization_tests.sh --doc [COLLECTION]
./run_speaker_diarization_tests.sh --doc-path [COLLECTION]
```

## Collections

| Collection | Tests | Description |
|------------|-------|-------------|
| `all` | 198 | All tests (default) |
| `unit` | 181 | All unit tests (fast, no API) |
| `e2e` | 17 | End-to-end pipeline integration |
| `catalog` | 23 | speaker-catalog tests |
| `assign` | 24 | speaker-assign tests |
| `review` | 18 | speaker-review tests |
| `llm` | 23 | speaker-llm tests |
| `process` | 22 | speaker-process tests |
| `report` | 26 | speaker-report tests |
| `segments` | 8 | speaker_segments tests |
| `profiles` | 10 | audio profiles tests |
| `legacy` | 27 | Original speaker_detection/samples tests |
| `docker` | - | Run all tests in Docker container |

## Examples

```bash
# Run all tests
./run_speaker_diarization_tests.sh

# Run quick unit tests
./run_speaker_diarization_tests.sh unit

# Run only speaker-catalog tests
./run_speaker_diarization_tests.sh catalog

# Run in Docker for isolated environment
./run_speaker_diarization_tests.sh docker

# Show available collections
./run_speaker_diarization_tests.sh list
```

## Documentation Commands

```bash
# View documentation for a collection
./run_speaker_diarization_tests.sh --doc catalog

# Get path to documentation file
./run_speaker_diarization_tests.sh --doc-path catalog

# View this documentation
./run_speaker_diarization_tests.sh --doc all
```

## Output

The runner displays:

* Test execution progress with colored status
* Pass/fail counts per collection
* Summary with totals

```
========================================
Speaker-* Tool Test Runner
========================================

Running: speaker-catalog
  PASS: speaker-catalog (23 tests)

Running: speaker-assign
  PASS: speaker-assign (24 tests)

...

========================================
Results: 181 passed, 0 failed, 0 skipped
========================================
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | One or more tests failed |
| 2 | Test skipped (dependencies missing) |

## Environment

Tests automatically use isolated `SPEAKERS_EMBEDDINGS_DIR` in `/tmp` to avoid affecting user data.

## Docker Testing

For fully isolated, reproducible testing:

```bash
./run_speaker_diarization_tests.sh docker
```

This builds a Docker image with all dependencies and runs the complete test suite.

## Related Documentation

* General testing methodology: `evals/TESTING.md`
* Individual collections: `evals/speaker_detection/test_*.README.md`

## Tool Documentation

Each speaker-* tool has its own README:

* `speaker-catalog.README.md`
* `speaker-assign.README.md`
* `speaker-review.README.md`
* `speaker-llm.README.md`
* `speaker-process.README.md`
* `speaker-report.README.md`
