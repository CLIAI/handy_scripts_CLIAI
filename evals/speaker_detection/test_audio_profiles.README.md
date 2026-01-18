# Test Collection: profiles

Tests for `audio_profiles` module - Audio Format Profiles for STT Backends.

## Module Under Test

`audio_profiles` manages audio encoding profiles for different STT backends:

* Predefined profiles for common backends (Speechmatics, AssemblyAI, Whisper)
* Custom profile registration
* FFmpeg argument generation
* Bit depth and sample rate configuration

## Test File

`evals/speaker_detection/test_audio_profiles.py`

## Running

```bash
./run_speaker_diarization_tests.sh profiles
```

Or directly:

```bash
python evals/speaker_detection/test_audio_profiles.py
```

## Test Count

10 tests

## Tests Included

| Test | Description |
|------|-------------|
| `test_get_profile_known_backends` | Retrieve profiles for known backends |
| `test_get_profile_unknown_backend` | Default profile for unknown backends |
| `test_register_profile` | Register custom audio profile |
| `test_format_ffmpeg_args_basic` | Basic FFmpeg argument generation |
| `test_format_ffmpeg_args_order` | FFmpeg argument ordering |
| `test_format_ffmpeg_args_bit_depth` | Bit depth specification |
| `test_format_ffmpeg_args_different_formats` | Various output formats (wav, mp3, ogg) |
| `test_backend_default_audio_profile` | Backend default profile selection |
| `test_backend_custom_audio_profile` | Override backend default profile |
| `test_backend_named_audio_profile` | Use named profile for backend |

## Environment

Tests use isolated temporary directories.

## Related Documentation

* General testing: `evals/TESTING.md`
