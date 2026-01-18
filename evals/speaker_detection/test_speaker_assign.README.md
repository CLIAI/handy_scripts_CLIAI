# Test Collection: assign

Tests for `speaker-assign` - Automated Speaker Label Assignment.

## Tool Under Test

`speaker-assign` maps generic diarization labels (S1, S2, ...) to known speaker profiles:

* Signal combination from embeddings, voice characteristics, and context
* Threshold-based confidence filtering
* Assignment persistence and management
* Support for expected speaker constraints

## Test File

`evals/speaker_detection/test_speaker_assign.py`

## Running

```bash
./run_speaker_diarization_tests.sh assign
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_assign.py
```

## Test Count

24 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_assign_basic` | Basic two-speaker assignment |
| `test_assign_three_speakers` | Assignment with three speakers |
| `test_assign_with_expected_speakers` | Assignment constrained to expected speakers |
| `test_assign_with_cli_expected_speakers` | Expected speakers via CLI flag |
| `test_assign_dry_run` | Preview mode without persisting |
| `test_assign_dry_run_json` | JSON output in dry-run mode |
| `test_assign_saves_to_assignments_dir` | Verify persistent storage |
| `test_assign_output_file` | Custom output file path |
| `test_show_assignments` | Display existing assignments |
| `test_show_assignments_json` | JSON format for show command |
| `test_show_nonexistent` | Error handling for missing assignments |
| `test_show_by_b3sum_prefix` | Lookup by partial b3sum |
| `test_clear_assignments` | Remove assignments |
| `test_clear_nonexistent` | Clear on missing assignments |
| `test_assign_json_output` | JSON output format validation |
| `test_assign_threshold` | Confidence threshold filtering |
| `test_assign_min_trust` | Minimum trust level requirement |
| `test_signal_combination` | Multiple signal combination |
| `test_assign_missing_audio` | Error handling for missing audio |
| `test_assign_missing_transcript` | Error handling for missing transcript |
| `test_assign_empty_transcript` | Handling of empty transcripts |
| `test_assign_verbose` | Verbose output mode |
| `test_assign_quiet` | Quiet/minimal output mode |
| `test_assign_speechmatics_format` | Speechmatics transcript format support |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` to avoid affecting user data.

## Related Documentation

* Tool README: `speaker-assign.README.md`
* General testing: `evals/TESTING.md`
