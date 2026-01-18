# Test Collection: legacy

Tests for original `speaker_detection` CLI and `speaker_samples` tools.

## Tools Under Test

### speaker_detection

Core speaker profile management:

* Add, list, show, update, delete speaker profiles
* Tagging and context-aware naming
* Enrollment from audio samples
* Embedding management and trust tracking

### speaker_samples

Sample extraction and review workflow:

* Extract speaker samples from transcribed audio
* Review workflow (approve/reject samples)
* Trust level computation
* B3sum integrity verification

## Test Files

* `evals/speaker_detection/test_cli.py` - speaker_detection CLI tests
* `evals/speaker_detection/test_samples_and_trust.py` - speaker_samples and trust tests

## Running

```bash
./run_speaker_diarization_tests.sh legacy
```

Or directly:

```bash
python evals/speaker_detection/test_cli.py
python evals/speaker_detection/test_samples_and_trust.py
```

## Test Count

27 tests total (16 + 11)

## Tests Included

### test_cli.py (16 tests)

| Test | Description |
|------|-------------|
| `test_add_speaker` | Add new speaker profile |
| `test_list_speakers` | List all speaker profiles |
| `test_show_speaker` | Show speaker profile details |
| `test_update_speaker` | Update speaker metadata |
| `test_delete_speaker` | Delete speaker profile |
| `test_tag_command` | Tag management |
| `test_export` | Export speaker data |
| `test_query` | JQ-style queries on profiles |
| `test_name_context` | Context-aware naming |
| `test_error_handling` | Error conditions |
| `test_enroll_dry_run` | Enrollment preview |
| `test_enroll_from_stdin_dry_run` | Enroll from stdin |
| `test_enroll_from_transcript_dry_run` | Enroll from transcript |
| `test_identify_error_handling` | Identification errors |
| `test_verify_error_handling` | Verification errors |
| `test_embeddings_command` | Embedding management |

### test_samples_and_trust.py (11 tests)

| Test | Description |
|------|-------------|
| `test_samples_speakers_cmd` | Speaker listing in samples |
| `test_samples_extract_with_b3sum` | Extract with integrity hash |
| `test_samples_review_approve` | Approve sample |
| `test_samples_review_reject` | Reject sample |
| `test_samples_list_with_review` | List with review status |
| `test_trust_level_computation` | Trust level calculation |
| `test_b3sum_computation` | B3sum hash computation |
| `test_check_validity_no_embeddings` | Validity with no embeddings |
| `test_check_validity_with_mock_embedding` | Validity with embeddings |
| `test_check_validity_detects_invalidation` | Detect invalid embeddings |
| `test_embeddings_show_trust` | Show trust in embeddings |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` to avoid affecting user data.

## Related Documentation

* General testing: `evals/TESTING.md`
