# Test Collection: segments

Tests for `speaker_segments` module - Transcript Segment Extraction.

## Module Under Test

`speaker_segments` extracts and formats speaker segments from transcripts:

* Multiple output formats (JSON, tuples, CSV)
* Segment merging based on gap thresholds
* Multi-format transcript support (Speechmatics, AssemblyAI)
* Speaker listing and filtering

## Test File

`evals/speaker_detection/test_speaker_segments.py`

## Running

```bash
./run_speaker_diarization_tests.sh segments
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_segments.py
```

## Test Count

8 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_json_output_format` | JSON segment output format |
| `test_tuples_output_format` | Tuple-based output format |
| `test_csv_output_format` | CSV segment output |
| `test_merge_gap_functionality` | Merge adjacent segments within gap threshold |
| `test_speechmatics_format` | Speechmatics transcript parsing |
| `test_list_speakers` | List unique speakers in transcript |
| `test_speaker_not_found` | Error handling for unknown speaker |
| `test_file_not_found` | Error handling for missing file |

## Environment

Tests use isolated temporary directories.

## Related Documentation

* General testing: `evals/TESTING.md`
