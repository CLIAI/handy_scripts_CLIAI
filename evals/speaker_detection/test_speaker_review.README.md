# Test Collection: review

Tests for `speaker-review` - Interactive Assignment Review Workflow.

## Tool Under Test

`speaker-review` provides interactive review of speaker assignments:

* Session-based review workflow with persistence
* Review by audio file or context
* Approve, reject, or correct assignments
* Track review progress across sessions

## Test File

`evals/speaker_detection/test_speaker_review.py`

## Running

```bash
./run_speaker_diarization_tests.sh review
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_review.py
```

## Test Count

18 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_status_no_session` | Status when no review session exists |
| `test_status_command` | Status command output format |
| `test_clear_no_session` | Clear on empty session |
| `test_clear_command` | Clear existing session |
| `test_review_no_assignments` | Review with no pending assignments |
| `test_review_specific_audio_no_assignments` | Review specific file with no assignments |
| `test_session_persistence` | Session state preserved across invocations |
| `test_session_continue_no_session` | Continue without existing session |
| `test_help_output` | Help text completeness |
| `test_help_shows_subcommands` | All subcommands documented |
| `test_version_output` | Version command |
| `test_review_subcommand_explicit` | Explicit review subcommand |
| `test_review_context_option` | Filter by context |
| `test_review_simple_mode_option` | Simplified review mode |
| `test_review_finds_assignments` | Discover pending assignments |
| `test_review_by_b3sum_prefix` | Review by partial b3sum |
| `test_review_nonexistent_audio` | Error for missing audio |
| `test_invalid_subcommand` | Error handling for unknown commands |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` to avoid affecting user data.

## Related Documentation

* Tool README: `speaker-review.README.md`
* General testing: `evals/TESTING.md`
