# Test Collection: llm

Tests for `speaker-llm` - LLM-Based Speaker Name Detection.

## Tool Under Test

`speaker-llm` uses large language models to extract speaker names from transcripts:

* Multi-provider support (OpenAI, Anthropic, Google)
* Speaker name detection from conversation context
* Confidence scoring and validation
* Response caching for cost efficiency

## Test File

`evals/speaker_detection/test_speaker_llm.py`

## Running

```bash
./run_speaker_diarization_tests.sh llm
```

Or directly:

```bash
python evals/speaker_detection/test_speaker_llm.py
```

## Test Count

23 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_providers_command` | List available LLM providers |
| `test_no_provider_available` | Graceful handling when no API keys configured |
| `test_extract_conversation_assemblyai` | Extract conversation from AssemblyAI format |
| `test_missing_transcript` | Error handling for missing transcript file |
| `test_invalid_json_transcript` | Error handling for malformed JSON |
| `test_version_command` | Version output |
| `test_help_command` | Help text |
| `test_analyze_help` | Analyze subcommand documentation |
| `test_detect_names_help` | Detect-names subcommand documentation |
| `test_cache_directory_creation` | Cache directory initialization |
| `test_clear_cache_empty` | Clear empty cache |
| `test_clear_cache_with_files` | Clear populated cache |
| `test_unknown_provider` | Error for unsupported provider |
| `test_provider_not_configured` | Error when provider API key missing |
| `test_parse_llm_response_valid_json` | Parse valid JSON responses |
| `test_parse_llm_response_markdown_codeblock` | Parse markdown-wrapped JSON |
| `test_analyze_output_schema` | Validate analyze output schema |
| `test_detection_schema` | Validate detection output schema |
| `test_default_models_defined` | Default models configured for each provider |
| `test_env_vars_defined` | Environment variables documented |
| `test_detection_patterns_documented` | Detection patterns explained |
| `test_assemblyai_format_support` | AssemblyAI transcript format |
| `test_speechmatics_format_support` | Speechmatics transcript format |

## Environment

Tests use isolated `SPEAKERS_EMBEDDINGS_DIR` and mock LLM calls to avoid API costs.

## Related Documentation

* Tool README: `speaker-llm.README.md`
* General testing: `evals/TESTING.md`
