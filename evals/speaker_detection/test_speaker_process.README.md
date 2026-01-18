# Test Collection: process

Tests for `speaker-process` - Batch Processing Queue Management.

## Tool Under Test

`speaker-process` manages batch processing of audio files through the pipeline:

* Queue-based processing with priority support
* Dry-run mode for validation
* Progress tracking and status reporting
* Concurrent queue access handling

## Test File

`evals/speaker_detection/test_speaker_process.py`

## Running

```bash
./run_speaker_diarization_tests.sh process
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_process.py
```

## Test Count

22 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_queue_single_file` | Add single file to queue |
| `test_queue_directory` | Add directory of files to queue |
| `test_queue_with_context` | Add with context metadata |
| `test_queue_duplicate` | Duplicate detection in queue |
| `test_status_empty_queue` | Status on empty queue |
| `test_status_with_items` | Status with queued items |
| `test_status_json_format` | JSON format for status |
| `test_status_verbose` | Verbose status output |
| `test_clear_queue_force` | Force clear entire queue |
| `test_clear_queue_by_status` | Clear by item status |
| `test_process_dry_run` | Dry-run processing |
| `test_process_nonexistent_file` | Error for missing files |
| `test_process_non_audio_file` | Reject non-audio files |
| `test_process_empty_directory` | Handle empty directories |
| `test_run_empty_queue` | Run on empty queue |
| `test_run_dry_run` | Run with dry-run flag |
| `test_run_with_limit` | Limit items processed |
| `test_audio_extensions` | Supported audio extension detection |
| `test_recursive_directory` | Recursive directory scanning |
| `test_process_with_mock_tools` | Integration with mock speaker tools |
| `test_special_characters_in_path` | Handle paths with special characters |
| `test_concurrent_queue_access` | Concurrent queue operations |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` to avoid affecting user data.

## Related Documentation

* Tool README: `speaker-process.README.md`
* General testing: `evals/TESTING.md`
