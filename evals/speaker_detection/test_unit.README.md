# Test Collection: unit

Combined unit tests for all speaker-* tools. This is a meta-collection that runs all fast, isolated unit tests without external API dependencies.

## Purpose

Run all unit tests quickly to validate core functionality. No API keys required, no external services called.

## Running

```bash
./run_speaker_diarization_tests.sh unit
```

## Test Count

181 tests total

## Collections Included

The `unit` collection runs all of the following:

| Collection | Tests | Description |
|------------|-------|-------------|
| `catalog` | 23 | Recording inventory management |
| `assign` | 24 | Speaker label assignment |
| `review` | 18 | Interactive review workflow |
| `llm` | 23 | LLM-based name detection (mocked) |
| `process` | 22 | Batch processing queue |
| `report` | 26 | Pipeline status reporting |
| `segments` | 8 | Transcript segment extraction |
| `profiles` | 10 | Audio format profiles |
| `legacy` | 27 | Original CLI and samples tools |

## Characteristics

* **Fast**: All tests complete in seconds
* **Isolated**: Each test uses temporary directories
* **No API calls**: LLM and STT backends are mocked
* **Deterministic**: No external dependencies affect results

## Running Individual Collections

To run a specific subset:

```bash
./run_speaker_diarization_tests.sh catalog   # Just catalog tests
./run_speaker_diarization_tests.sh assign    # Just assign tests
# etc.
```

## Related Documentation

* Individual collection docs: `test_speaker_*.README.md`
* Test runner: `run_speaker_diarization_tests.sh.README.md`
* General testing: `evals/TESTING.md`
