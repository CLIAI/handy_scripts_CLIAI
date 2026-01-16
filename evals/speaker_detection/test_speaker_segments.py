#!/usr/bin/env python3
"""
Unit tests for speaker_segments CLI tool.

Tests segment extraction from mock transcripts with various formats.

Usage:
    ./test_speaker_segments.py              # Run all tests
    ./test_speaker_segments.py -v           # Verbose output
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
SPEAKER_SEGMENTS = REPO_ROOT / "speaker_segments"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None


def run_cmd(args: list, stdin_input: str = None) -> tuple:
    """Run speaker_segments command, return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(SPEAKER_SEGMENTS)] + args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=stdin_input,
    )
    return result.returncode, result.stdout, result.stderr


def test_json_output_format() -> TestResult:
    """Test JSON output format with mock AssemblyAI transcript."""
    result = TestResult("json_output_format")

    # Create mock AssemblyAI transcript
    transcript_data = {
        "utterances": [
            {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello"},
            {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi there"},
            {"speaker": "A", "start": 11000, "end": 15000, "text": "How are you?"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        rc, stdout, stderr = run_cmd([transcript_file, "A", "--format", "json"])

        if rc != 0:
            result.error = f"Command failed with rc={rc}: {stderr}"
            return result

        # Parse JSON output
        try:
            segments = json.loads(stdout)
        except json.JSONDecodeError as e:
            result.error = f"Failed to parse JSON output: {e}\nOutput: {stdout}"
            return result

        # Verify structure
        if not isinstance(segments, list):
            result.error = f"Expected list, got {type(segments)}"
            return result

        if len(segments) != 2:
            result.error = f"Expected 2 segments for speaker A, got {len(segments)}"
            return result

        # Verify segment structure
        for seg in segments:
            if "start" not in seg or "end" not in seg:
                result.error = f"Segment missing start/end keys: {seg}"
                return result

        # Verify values (AssemblyAI uses milliseconds, converted to seconds)
        expected = [{"start": 1.0, "end": 5.0}, {"start": 11.0, "end": 15.0}]
        if segments != expected:
            result.error = f"Segment values mismatch.\nExpected: {expected}\nGot: {segments}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_tuples_output_format() -> TestResult:
    """Test tuples output format."""
    result = TestResult("tuples_output_format")

    transcript_data = {
        "utterances": [
            {"speaker": "A", "start": 2000, "end": 4000, "text": "Test"},
            {"speaker": "A", "start": 8000, "end": 12000, "text": "More"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        rc, stdout, stderr = run_cmd([transcript_file, "A", "--format", "tuples"])

        if rc != 0:
            result.error = f"Command failed: {stderr}"
            return result

        # Should be valid Python tuple syntax
        expected = "[(2.0, 4.0), (8.0, 12.0)]"
        if stdout.strip() != expected:
            result.error = f"Tuples format mismatch.\nExpected: {expected}\nGot: {stdout.strip()}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_csv_output_format() -> TestResult:
    """Test CSV output format."""
    result = TestResult("csv_output_format")

    transcript_data = {
        "utterances": [
            {"speaker": "A", "start": 1500, "end": 3500, "text": "Test"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        rc, stdout, stderr = run_cmd([transcript_file, "A", "--format", "csv"])

        if rc != 0:
            result.error = f"Command failed: {stderr}"
            return result

        lines = stdout.strip().split("\n")
        if lines[0] != "start,end":
            result.error = f"CSV header mismatch: {lines[0]}"
            return result

        if lines[1] != "1.5,3.5":
            result.error = f"CSV data mismatch: {lines[1]}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_merge_gap_functionality() -> TestResult:
    """Test merge-gap functionality combines close segments."""
    result = TestResult("merge_gap_functionality")

    # Create transcript with segments that have small gaps
    transcript_data = {
        "utterances": [
            {"speaker": "A", "start": 1000, "end": 3000, "text": "First"},
            {"speaker": "A", "start": 3500, "end": 5000, "text": "Second"},  # 0.5s gap
            {"speaker": "A", "start": 10000, "end": 12000, "text": "Third"},  # 5s gap (not merged)
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        # Without merge-gap: should get 3 segments
        rc, stdout, stderr = run_cmd([transcript_file, "A", "--format", "json"])
        if rc != 0:
            result.error = f"Command failed: {stderr}"
            return result

        segments_no_merge = json.loads(stdout)
        if len(segments_no_merge) != 3:
            result.error = f"Expected 3 segments without merge, got {len(segments_no_merge)}"
            return result

        # With merge-gap=1.0: should merge first two (0.5s gap < 1.0s)
        rc, stdout, stderr = run_cmd([transcript_file, "A", "--format", "json", "--merge-gap", "1.0"])
        if rc != 0:
            result.error = f"Command with merge-gap failed: {stderr}"
            return result

        segments_merged = json.loads(stdout)
        if len(segments_merged) != 2:
            result.error = f"Expected 2 segments with merge-gap=1.0, got {len(segments_merged)}"
            return result

        # First merged segment should span from 1.0 to 5.0
        if segments_merged[0]["start"] != 1.0 or segments_merged[0]["end"] != 5.0:
            result.error = f"Merged segment incorrect: {segments_merged[0]}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_speechmatics_format() -> TestResult:
    """Test with Speechmatics transcript format."""
    result = TestResult("speechmatics_format")

    # Create mock Speechmatics transcript
    transcript_data = {
        "results": [
            {
                "type": "word",
                "start_time": 0.5,
                "end_time": 1.0,
                "alternatives": [{"content": "Hello", "speaker": "S1"}],
            },
            {
                "type": "word",
                "start_time": 1.2,
                "end_time": 1.8,
                "alternatives": [{"content": "world", "speaker": "S1"}],
            },
            {
                "type": "word",
                "start_time": 2.0,
                "end_time": 2.5,
                "alternatives": [{"content": "Hi", "speaker": "S2"}],
            },
            {
                "type": "word",
                "start_time": 3.0,
                "end_time": 3.5,
                "alternatives": [{"content": "there", "speaker": "S1"}],
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        rc, stdout, stderr = run_cmd([transcript_file, "S1", "--format", "json"])

        if rc != 0:
            result.error = f"Command failed: {stderr}"
            return result

        segments = json.loads(stdout)

        # S1 speaks at 0.5-1.8 (merged), then at 3.0-3.5
        if len(segments) != 2:
            result.error = f"Expected 2 segments for S1, got {len(segments)}: {segments}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_list_speakers() -> TestResult:
    """Test --list-speakers flag."""
    result = TestResult("list_speakers")

    transcript_data = {
        "utterances": [
            {"speaker": "Alice", "start": 1000, "end": 5000, "text": "Hello"},
            {"speaker": "Bob", "start": 6000, "end": 10000, "text": "Hi"},
            {"speaker": "Alice", "start": 11000, "end": 15000, "text": "Bye"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        # Use a dummy speaker label since --list-speakers exits early
        rc, stdout, stderr = run_cmd([transcript_file, "dummy", "--list-speakers"])

        if rc != 0:
            result.error = f"Command failed: {stderr}"
            return result

        if "Alice" not in stdout or "Bob" not in stdout:
            result.error = f"Missing speakers in output: {stdout}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_speaker_not_found() -> TestResult:
    """Test error handling for non-existent speaker."""
    result = TestResult("speaker_not_found")

    transcript_data = {
        "utterances": [
            {"speaker": "Alice", "start": 1000, "end": 5000, "text": "Hello"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transcript_data, f)
        transcript_file = f.name

    try:
        rc, stdout, stderr = run_cmd([transcript_file, "NonExistent"])

        if rc == 0:
            result.error = "Expected non-zero exit code for missing speaker"
            return result

        if "not found" not in stderr.lower():
            result.error = f"Expected 'not found' in error message: {stderr}"
            return result

        result.passed = True

    finally:
        os.unlink(transcript_file)

    return result


def test_file_not_found() -> TestResult:
    """Test error handling for missing file."""
    result = TestResult("file_not_found")

    rc, stdout, stderr = run_cmd(["/nonexistent/file.json", "A"])

    if rc == 0:
        result.error = "Expected non-zero exit code for missing file"
        return result

    if "not found" not in stderr.lower():
        result.error = f"Expected 'not found' in error message: {stderr}"
        return result

    result.passed = True
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="speaker_segments CLI unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tests = [
        test_json_output_format,
        test_tuples_output_format,
        test_csv_output_format,
        test_merge_gap_functionality,
        test_speechmatics_format,
        test_list_speakers,
        test_speaker_not_found,
        test_file_not_found,
    ]

    print("speaker_segments CLI Unit Tests")
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
            print(f"  FAIL: {test_func.__name__} (exception)")
            if args.verbose:
                print(f"        Exception: {e}")
            failed += 1

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
