#!/usr/bin/env python3
"""
Unit tests for audio profiles system.

Tests:
1. get_profile() returns correct profiles
2. format_ffmpeg_args() generates correct arguments
3. backend.audio_profile property works correctly

Usage:
    ./test_audio_profiles.py              # Run all tests
    ./test_audio_profiles.py -v           # Verbose output
    python -m pytest test_audio_profiles.py  # Using pytest
"""

import sys
from pathlib import Path

# Add repo root to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from speaker_detection_backends.audio_profiles import (
    AudioProfile,
    PROFILES,
    get_profile,
    format_ffmpeg_args,
    register_profile,
)
from speaker_detection_backends.base import EmbeddingBackend


class TestResult:
    """Simple test result container."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None


# ==============================================================================
# Test: get_profile()
# ==============================================================================


def test_get_profile_known_backends() -> TestResult:
    """Test get_profile() returns correct profiles for known backends."""
    result = TestResult("get_profile_known_backends")

    # Test speechmatics profile
    profile = get_profile("speechmatics")
    if profile.sample_rate != 16000:
        result.error = f"speechmatics sample_rate: expected 16000, got {profile.sample_rate}"
        return result
    if profile.channels != 1:
        result.error = f"speechmatics channels: expected 1, got {profile.channels}"
        return result
    if profile.format != "wav":
        result.error = f"speechmatics format: expected 'wav', got '{profile.format}'"
        return result

    # Test pyannote profile
    profile = get_profile("pyannote")
    if profile.sample_rate != 16000:
        result.error = f"pyannote sample_rate: expected 16000, got {profile.sample_rate}"
        return result
    if profile.channels != 1:
        result.error = f"pyannote channels: expected 1, got {profile.channels}"
        return result

    # Test default profile
    profile = get_profile("default")
    if profile.sample_rate != 16000:
        result.error = f"default sample_rate: expected 16000, got {profile.sample_rate}"
        return result

    result.passed = True
    return result


def test_get_profile_unknown_backend() -> TestResult:
    """Test get_profile() returns default for unknown backends."""
    result = TestResult("get_profile_unknown_backend")

    profile = get_profile("nonexistent_backend")
    default_profile = get_profile("default")

    if profile.sample_rate != default_profile.sample_rate:
        result.error = f"Unknown backend should return default profile"
        return result
    if profile.channels != default_profile.channels:
        result.error = f"Unknown backend should return default profile"
        return result
    if profile.format != default_profile.format:
        result.error = f"Unknown backend should return default profile"
        return result

    result.passed = True
    return result


def test_register_profile() -> TestResult:
    """Test registering a custom profile."""
    result = TestResult("register_profile")

    custom_profile = AudioProfile(
        sample_rate=44100,
        channels=2,
        format="mp3",
        bit_depth=24,
        max_duration_sec=300.0,
    )
    register_profile("custom_test", custom_profile)

    retrieved = get_profile("custom_test")
    if retrieved.sample_rate != 44100:
        result.error = f"Custom profile sample_rate: expected 44100, got {retrieved.sample_rate}"
        return result
    if retrieved.channels != 2:
        result.error = f"Custom profile channels: expected 2, got {retrieved.channels}"
        return result
    if retrieved.max_duration_sec != 300.0:
        result.error = f"Custom profile max_duration_sec: expected 300.0, got {retrieved.max_duration_sec}"
        return result

    # Clean up
    del PROFILES["custom_test"]

    result.passed = True
    return result


# ==============================================================================
# Test: format_ffmpeg_args()
# ==============================================================================


def test_format_ffmpeg_args_basic() -> TestResult:
    """Test format_ffmpeg_args() generates correct basic arguments."""
    result = TestResult("format_ffmpeg_args_basic")

    profile = AudioProfile(sample_rate=16000, channels=1, format="wav")
    args = format_ffmpeg_args(profile)

    # Check required arguments are present
    if "-ar" not in args:
        result.error = "Missing -ar argument"
        return result
    if "16000" not in args:
        result.error = "Missing sample rate value"
        return result
    if "-ac" not in args:
        result.error = "Missing -ac argument"
        return result
    if "1" not in args:
        result.error = "Missing channel count value"
        return result
    if "-f" not in args:
        result.error = "Missing -f argument"
        return result
    if "wav" not in args:
        result.error = "Missing format value"
        return result

    result.passed = True
    return result


def test_format_ffmpeg_args_order() -> TestResult:
    """Test format_ffmpeg_args() arguments are in correct order."""
    result = TestResult("format_ffmpeg_args_order")

    profile = AudioProfile(sample_rate=16000, channels=1, format="wav")
    args = format_ffmpeg_args(profile)

    # Check argument order: -ar value -ac value -f value
    ar_idx = args.index("-ar")
    ac_idx = args.index("-ac")
    f_idx = args.index("-f")

    if args[ar_idx + 1] != "16000":
        result.error = f"Sample rate value should follow -ar, got {args[ar_idx + 1]}"
        return result
    if args[ac_idx + 1] != "1":
        result.error = f"Channel count should follow -ac, got {args[ac_idx + 1]}"
        return result
    if args[f_idx + 1] != "wav":
        result.error = f"Format should follow -f, got {args[f_idx + 1]}"
        return result

    result.passed = True
    return result


def test_format_ffmpeg_args_bit_depth() -> TestResult:
    """Test format_ffmpeg_args() handles bit depth correctly for wav format."""
    result = TestResult("format_ffmpeg_args_bit_depth")

    # Test 16-bit
    profile = AudioProfile(sample_rate=16000, channels=1, format="wav", bit_depth=16)
    args = format_ffmpeg_args(profile)
    if "-acodec" not in args or "pcm_s16le" not in args:
        result.error = "16-bit wav should use pcm_s16le codec"
        return result

    # Test 24-bit
    profile = AudioProfile(sample_rate=16000, channels=1, format="wav", bit_depth=24)
    args = format_ffmpeg_args(profile)
    if "-acodec" not in args or "pcm_s24le" not in args:
        result.error = "24-bit wav should use pcm_s24le codec"
        return result

    # Test 32-bit
    profile = AudioProfile(sample_rate=16000, channels=1, format="wav", bit_depth=32)
    args = format_ffmpeg_args(profile)
    if "-acodec" not in args or "pcm_s32le" not in args:
        result.error = "32-bit wav should use pcm_s32le codec"
        return result

    result.passed = True
    return result


def test_format_ffmpeg_args_different_formats() -> TestResult:
    """Test format_ffmpeg_args() with different audio formats."""
    result = TestResult("format_ffmpeg_args_different_formats")

    # Test mp3 format (should not add wav-specific codec)
    profile = AudioProfile(sample_rate=44100, channels=2, format="mp3", bit_depth=16)
    args = format_ffmpeg_args(profile)

    if "-ar" not in args or "44100" not in args:
        result.error = "MP3 profile missing sample rate"
        return result
    if "-ac" not in args or "2" not in args:
        result.error = "MP3 profile missing channel count"
        return result
    if "-f" not in args or "mp3" not in args:
        result.error = "MP3 profile missing format"
        return result

    # mp3 format should NOT have pcm codec
    if "pcm_s16le" in args:
        result.error = "MP3 format should not use PCM codec"
        return result

    result.passed = True
    return result


# ==============================================================================
# Test: backend.audio_profile property
# ==============================================================================


class MockBackend(EmbeddingBackend):
    """Mock backend for testing audio_profile property."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def requires_api_key(self) -> bool:
        return False

    def enroll_speaker(self, audio_path, segments=None):
        return {"external_id": "test", "model_version": "mock-v1"}

    def identify_speaker(self, audio_path, candidates, threshold=0.354):
        return []


class MockBackendWithCustomProfile(EmbeddingBackend):
    """Mock backend with custom audio profile."""

    @property
    def name(self) -> str:
        return "mock_custom"

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def audio_profile(self):
        """Return a custom AudioProfile directly."""
        return AudioProfile(
            sample_rate=48000,
            channels=2,
            format="flac",
            bit_depth=24,
            max_duration_sec=600.0,
        )

    def enroll_speaker(self, audio_path, segments=None):
        return {"external_id": "test", "model_version": "mock_custom-v1"}

    def identify_speaker(self, audio_path, candidates, threshold=0.354):
        return []


def test_backend_default_audio_profile() -> TestResult:
    """Test that default backend returns 'default' audio profile."""
    result = TestResult("backend_default_audio_profile")

    backend = MockBackend()

    # Check audio_profile property returns "default"
    if backend.audio_profile != "default":
        result.error = f"Expected 'default', got '{backend.audio_profile}'"
        return result

    # Check get_audio_profile() resolves to AudioProfile
    profile = backend.get_audio_profile()
    if not isinstance(profile, AudioProfile):
        result.error = f"get_audio_profile() should return AudioProfile, got {type(profile)}"
        return result

    if profile.sample_rate != 16000:
        result.error = f"Default profile sample_rate: expected 16000, got {profile.sample_rate}"
        return result

    result.passed = True
    return result


def test_backend_custom_audio_profile() -> TestResult:
    """Test backend with custom AudioProfile."""
    result = TestResult("backend_custom_audio_profile")

    backend = MockBackendWithCustomProfile()

    # Check audio_profile returns AudioProfile directly
    profile = backend.audio_profile
    if not isinstance(profile, AudioProfile):
        result.error = f"Expected AudioProfile, got {type(profile)}"
        return result

    if profile.sample_rate != 48000:
        result.error = f"Custom profile sample_rate: expected 48000, got {profile.sample_rate}"
        return result
    if profile.channels != 2:
        result.error = f"Custom profile channels: expected 2, got {profile.channels}"
        return result
    if profile.format != "flac":
        result.error = f"Custom profile format: expected 'flac', got '{profile.format}'"
        return result

    # Check get_audio_profile() returns the same profile
    resolved = backend.get_audio_profile()
    if resolved.sample_rate != 48000:
        result.error = "get_audio_profile() should return custom profile"
        return result

    result.passed = True
    return result


def test_backend_named_audio_profile() -> TestResult:
    """Test backend that returns profile name string."""
    result = TestResult("backend_named_audio_profile")

    # Create a backend that returns "speechmatics" profile name
    class SpeechmaticsStyleBackend(MockBackend):
        @property
        def audio_profile(self):
            return "speechmatics"

    backend = SpeechmaticsStyleBackend()

    if backend.audio_profile != "speechmatics":
        result.error = f"Expected 'speechmatics', got '{backend.audio_profile}'"
        return result

    profile = backend.get_audio_profile()
    speechmatics_profile = get_profile("speechmatics")

    if profile.sample_rate != speechmatics_profile.sample_rate:
        result.error = "Profile should match speechmatics profile"
        return result

    result.passed = True
    return result


# ==============================================================================
# Main test runner
# ==============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audio profiles unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tests = [
        # get_profile() tests
        test_get_profile_known_backends,
        test_get_profile_unknown_backend,
        test_register_profile,
        # format_ffmpeg_args() tests
        test_format_ffmpeg_args_basic,
        test_format_ffmpeg_args_order,
        test_format_ffmpeg_args_bit_depth,
        test_format_ffmpeg_args_different_formats,
        # backend.audio_profile tests
        test_backend_default_audio_profile,
        test_backend_custom_audio_profile,
        test_backend_named_audio_profile,
    ]

    print("Audio Profiles Unit Tests")
    print("=" * 40)

    passed = 0
    failed = 0
    results = []

    for test_func in tests:
        try:
            result = test_func()
            results.append(result)

            if result.passed:
                print(f"  PASS: {result.name}")
                passed += 1
            else:
                print(f"  FAIL: {result.name}")
                if args.verbose and result.error:
                    print(f"        Error: {result.error}")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {test_func.__name__}")
            if args.verbose:
                print(f"        Exception: {e}")
            failed += 1

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
