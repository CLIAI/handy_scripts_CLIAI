# Test Collection: report

Tests for `speaker-report` - Pipeline Status and Health Reporting.

## Tool Under Test

`speaker-report` provides status reports on the speaker identification pipeline:

* Overall pipeline status and recommendations
* Coverage analysis by context
* Confidence level tracking
* Stale assignment detection
* Speaker sample status

## Test File

`evals/speaker_detection/test_speaker_report.py`

## Running

```bash
./run_speaker_diarization_tests.sh report
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_report.py
```

## Test Count

26 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_status_empty` | Status on empty pipeline |
| `test_status_with_data` | Status with populated data |
| `test_status_json_format` | JSON format for status |
| `test_status_default_command` | Default command is status |
| `test_status_recommendations` | Recommendations in status output |
| `test_coverage_empty` | Coverage on empty catalog |
| `test_coverage_by_context` | Coverage breakdown by context |
| `test_coverage_filter_context` | Filter coverage to specific context |
| `test_coverage_json_format` | JSON format for coverage |
| `test_confidence_empty` | Confidence report on empty data |
| `test_confidence_finds_low` | Detect low-confidence assignments |
| `test_confidence_threshold` | Custom confidence threshold |
| `test_confidence_json_format` | JSON format for confidence |
| `test_stale_empty` | Stale detection on empty data |
| `test_stale_finds_old` | Find stale assignments |
| `test_stale_custom_days` | Custom staleness threshold |
| `test_stale_ignores_complete` | Complete items not marked stale |
| `test_stale_json_format` | JSON format for stale |
| `test_speakers_empty` | Speakers report on empty profiles |
| `test_speakers_with_data` | Speakers report with profiles |
| `test_speakers_needing_samples` | Identify speakers needing samples |
| `test_speakers_json_format` | JSON format for speakers |
| `test_version_flag` | Version output |
| `test_invalid_format` | Error for invalid format option |
| `test_missing_directories` | Handle missing data directories |
| `test_malformed_yaml` | Handle malformed YAML files |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` to avoid affecting user data.

## Related Documentation

* Tool README: `speaker-report.README.md`
* General testing: `evals/TESTING.md`
