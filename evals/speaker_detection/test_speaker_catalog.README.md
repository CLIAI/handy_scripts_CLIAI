# Test Collection: catalog

Tests for `speaker-catalog` - Recording Inventory and Processing State Management.

## Tool Under Test

`speaker-catalog` provides centralized management for audio recordings:

* Recording inventory with content-addressable b3sum identifiers
* Processing state progression (unprocessed -> transcribed -> assigned -> reviewed -> complete)
* Context information including expected speakers and tags
* Transcript registrations from multiple STT backends

## Test File

`evals/speaker_detection/test_speaker_catalog.py`

## Running

```bash
./run_speaker_diarization_tests.sh catalog
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_catalog.py
```

## Test Count

23 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_add_recording` | Add a recording to catalog with b3sum identification |
| `test_add_with_context` | Add recording with context tags and expected speakers |
| `test_add_duplicate_fails` | Verify duplicate detection prevents double-adding |
| `test_list_empty` | List command works on empty catalog |
| `test_list_with_entries` | List shows all catalog entries |
| `test_list_filter_by_status` | Filter listings by processing status |
| `test_list_filter_by_context` | Filter listings by context |
| `test_show_recording` | Show full details for a recording |
| `test_show_nonexistent` | Graceful error for missing recording |
| `test_status_unprocessed` | Status reports unprocessed state correctly |
| `test_status_after_transcript` | Status updates after transcript registration |
| `test_register_transcript` | Register transcript from STT backend |
| `test_register_transcript_multiple_backends` | Multiple backends per recording |
| `test_set_context` | Update recording context |
| `test_set_context_tags` | Update recording tags |
| `test_remove_recording` | Remove recording from catalog |
| `test_remove_by_b3sum_prefix` | Remove by partial b3sum match |
| `test_query_jq` | JQ-style queries on catalog |
| `test_query_complex_expression` | Complex JQ expressions |
| `test_add_nonexistent_file` | Error handling for missing files |
| `test_register_transcript_not_in_catalog` | Error for transcript without catalog entry |
| `test_status_not_in_catalog` | Error for status on uncataloged file |
| `test_b3sum_prefix_lookup` | Lookup recordings by b3sum prefix |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` to avoid affecting user data.

## Related Documentation

* Tool README: `speaker-catalog.README.md`
* General testing: `evals/TESTING.md`
