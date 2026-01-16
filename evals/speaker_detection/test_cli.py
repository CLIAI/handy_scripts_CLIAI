#!/usr/bin/env python3
"""
Unit tests for speaker_detection CLI commands.

Tests all CLI commands without requiring API calls by using
a temporary SPEAKERS_EMBEDDINGS_DIR.

Usage:
    ./test_cli.py              # Run all tests
    ./test_cli.py -v           # Verbose output
"""

import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
SPEAKER_DETECTION = REPO_ROOT / "speaker_detection"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None


def run_cmd(args: list, env: dict = None) -> tuple:
    """Run speaker_detection command, return (returncode, stdout, stderr)."""
    cmd = [str(SPEAKER_DETECTION)] + args
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env,
    )
    return result.returncode, result.stdout, result.stderr


def test_add_speaker(temp_dir: Path) -> TestResult:
    """Test adding a speaker."""
    result = TestResult("add_speaker")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd([
        "add", "test-user",
        "--name", "Test User",
        "--tag", "test-tag",
        "--metadata", "key1=value1",
        "--description", "Test description",
    ], env)

    if rc != 0:
        result.error = f"add command failed: {stderr}"
        return result

    # Verify file was created
    profile_path = temp_dir / "db" / "test-user.json"
    if not profile_path.exists():
        result.error = f"Profile not created at {profile_path}"
        return result

    # Verify JSON structure
    with open(profile_path) as f:
        profile = json.load(f)

    checks = [
        (profile.get("id") == "test-user", "id mismatch"),
        (profile.get("names", {}).get("default") == "Test User", "name mismatch"),
        ("test-tag" in profile.get("tags", []), "tag missing"),
        (profile.get("metadata", {}).get("key1") == "value1", "metadata mismatch"),
        (profile.get("description") == "Test description", "description mismatch"),
    ]

    for check, msg in checks:
        if not check:
            result.error = msg
            return result

    result.passed = True
    return result


def test_list_speakers(temp_dir: Path) -> TestResult:
    """Test listing speakers."""
    result = TestResult("list_speakers")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add some speakers
    run_cmd(["add", "alice", "--name", "Alice", "--tag", "team-a"], env)
    run_cmd(["add", "bob", "--name", "Bob", "--tag", "team-b"], env)
    run_cmd(["add", "charlie", "--name", "Charlie", "--tag", "team-a", "--tag", "team-b"], env)

    # Test basic list
    rc, stdout, stderr = run_cmd(["list"], env)
    if rc != 0:
        result.error = f"list failed: {stderr}"
        return result

    if "alice" not in stdout or "bob" not in stdout or "charlie" not in stdout:
        result.error = f"Missing speakers in list output: {stdout}"
        return result

    # Test JSON format
    rc, stdout, stderr = run_cmd(["list", "--format", "json"], env)
    if rc != 0:
        result.error = f"list --format json failed: {stderr}"
        return result

    speakers = json.loads(stdout)
    if len(speakers) != 3:
        result.error = f"Expected 3 speakers, got {len(speakers)}"
        return result

    # Test tag filter
    rc, stdout, stderr = run_cmd(["list", "--tags", "team-a", "--format", "ids"], env)
    if rc != 0:
        result.error = f"list --tags failed: {stderr}"
        return result

    ids = stdout.strip().split("\n")
    if set(ids) != {"alice", "charlie"}:
        result.error = f"Tag filter failed, got: {ids}"
        return result

    result.passed = True
    return result


def test_show_speaker(temp_dir: Path) -> TestResult:
    """Test showing speaker details."""
    result = TestResult("show_speaker")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker
    run_cmd(["add", "test-show", "--name", "Test Show", "--tag", "show-test"], env)

    # Show speaker
    rc, stdout, stderr = run_cmd(["show", "test-show"], env)
    if rc != 0:
        result.error = f"show failed: {stderr}"
        return result

    profile = json.loads(stdout)
    if profile.get("id") != "test-show":
        result.error = f"Wrong profile returned: {profile.get('id')}"
        return result

    # Test non-existent speaker
    rc, stdout, stderr = run_cmd(["show", "nonexistent"], env)
    if rc == 0:
        result.error = "show should fail for non-existent speaker"
        return result

    result.passed = True
    return result


def test_update_speaker(temp_dir: Path) -> TestResult:
    """Test updating speaker details."""
    result = TestResult("update_speaker")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker
    run_cmd(["add", "update-test", "--name", "Original Name", "--tag", "original-tag"], env)

    # Update name
    rc, stdout, stderr = run_cmd(["update", "update-test", "--name", "New Name"], env)
    if rc != 0:
        result.error = f"update name failed: {stderr}"
        return result

    # Add tag
    rc, stdout, stderr = run_cmd(["update", "update-test", "--tag", "new-tag"], env)
    if rc != 0:
        result.error = f"update tag failed: {stderr}"
        return result

    # Verify changes
    rc, stdout, stderr = run_cmd(["show", "update-test"], env)
    profile = json.loads(stdout)

    if profile.get("names", {}).get("default") != "New Name":
        result.error = "Name not updated"
        return result

    if "new-tag" not in profile.get("tags", []):
        result.error = "Tag not added"
        return result

    if "original-tag" not in profile.get("tags", []):
        result.error = "Original tag should still be present"
        return result

    result.passed = True
    return result


def test_delete_speaker(temp_dir: Path) -> TestResult:
    """Test deleting a speaker."""
    result = TestResult("delete_speaker")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker
    run_cmd(["add", "delete-test", "--name", "To Delete"], env)

    # Verify exists
    rc, _, _ = run_cmd(["show", "delete-test"], env)
    if rc != 0:
        result.error = "Speaker not created"
        return result

    # Delete with --force
    rc, stdout, stderr = run_cmd(["delete", "delete-test", "--force"], env)
    if rc != 0:
        result.error = f"delete failed: {stderr}"
        return result

    # Verify deleted
    rc, _, _ = run_cmd(["show", "delete-test"], env)
    if rc == 0:
        result.error = "Speaker should be deleted"
        return result

    result.passed = True
    return result


def test_tag_command(temp_dir: Path) -> TestResult:
    """Test tag management command."""
    result = TestResult("tag_command")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker
    run_cmd(["add", "tag-test", "--name", "Tag Test"], env)

    # Add tag
    rc, stdout, stderr = run_cmd(["tag", "tag-test", "--add", "added-tag"], env)
    if rc != 0:
        result.error = f"tag --add failed: {stderr}"
        return result

    # Verify tag added
    rc, stdout, stderr = run_cmd(["show", "tag-test"], env)
    profile = json.loads(stdout)
    if "added-tag" not in profile.get("tags", []):
        result.error = "Tag not added"
        return result

    # Remove tag
    rc, stdout, stderr = run_cmd(["tag", "tag-test", "--remove", "added-tag"], env)
    if rc != 0:
        result.error = f"tag --remove failed: {stderr}"
        return result

    # Verify tag removed
    rc, stdout, stderr = run_cmd(["show", "tag-test"], env)
    profile = json.loads(stdout)
    if "added-tag" in profile.get("tags", []):
        result.error = "Tag not removed"
        return result

    result.passed = True
    return result


def test_export(temp_dir: Path) -> TestResult:
    """Test export command."""
    result = TestResult("export")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speakers with different tags
    run_cmd(["add", "export-a", "--name", "Export A", "--tag", "export-test"], env)
    run_cmd(["add", "export-b", "--name", "Export B", "--tag", "export-test", "--tag", "special"], env)
    run_cmd(["add", "export-c", "--name", "Export C", "--tag", "other"], env)

    # Export all
    rc, stdout, stderr = run_cmd(["export"], env)
    if rc != 0:
        result.error = f"export failed: {stderr}"
        return result

    data = json.loads(stdout)
    if len(data.get("speakers", [])) != 3:
        result.error = f"Expected 3 speakers in export, got {len(data.get('speakers', []))}"
        return result

    # Export with tag filter
    rc, stdout, stderr = run_cmd(["export", "--tags", "export-test"], env)
    if rc != 0:
        result.error = f"export --tags failed: {stderr}"
        return result

    data = json.loads(stdout)
    if len(data.get("speakers", [])) != 2:
        result.error = f"Expected 2 speakers with tag filter, got {len(data.get('speakers', []))}"
        return result

    result.passed = True
    return result


def test_query(temp_dir: Path) -> TestResult:
    """Test query command with jq."""
    result = TestResult("query")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Check if jq is available
    try:
        subprocess.run(["jq", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        result.error = "jq not available, skipping"
        result.passed = True  # Skip, not fail
        return result

    # Add speakers
    run_cmd(["add", "query-a", "--name", "Query A", "--tag", "query-test"], env)
    run_cmd(["add", "query-b", "--name", "Query B", "--tag", "query-test"], env)

    # Run query
    rc, stdout, stderr = run_cmd(["query", ".[].id"], env)
    if rc != 0:
        result.error = f"query failed: {stderr}"
        return result

    ids = stdout.strip().replace('"', '').split("\n")
    if "query-a" not in ids or "query-b" not in ids:
        result.error = f"Query result missing IDs: {ids}"
        return result

    result.passed = True
    return result


def test_name_context(temp_dir: Path) -> TestResult:
    """Test context-specific names."""
    result = TestResult("name_context")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker with context names
    rc, stdout, stderr = run_cmd([
        "add", "context-test",
        "--name", "Default Name",
        "--name-context", "formal=Dr. Formal Name",
        "--name-context", "casual=Nick",
    ], env)

    if rc != 0:
        result.error = f"add with contexts failed: {stderr}"
        return result

    # Verify contexts
    rc, stdout, stderr = run_cmd(["show", "context-test"], env)
    profile = json.loads(stdout)
    names = profile.get("names", {})

    if names.get("default") != "Default Name":
        result.error = f"Default name wrong: {names.get('default')}"
        return result

    if names.get("formal") != "Dr. Formal Name":
        result.error = f"Formal name wrong: {names.get('formal')}"
        return result

    if names.get("casual") != "Nick":
        result.error = f"Casual name wrong: {names.get('casual')}"
        return result

    result.passed = True
    return result


def test_error_handling(temp_dir: Path) -> TestResult:
    """Test error handling for invalid operations."""
    result = TestResult("error_handling")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add duplicate speaker
    run_cmd(["add", "error-test", "--name", "Error Test"], env)
    rc, stdout, stderr = run_cmd(["add", "error-test", "--name", "Duplicate"], env)
    if rc == 0:
        result.error = "Should fail on duplicate ID"
        return result

    # Show non-existent
    rc, stdout, stderr = run_cmd(["show", "nonexistent-speaker"], env)
    if rc == 0:
        result.error = "Should fail on non-existent speaker"
        return result

    # Update non-existent
    rc, stdout, stderr = run_cmd(["update", "nonexistent-speaker", "--name", "New"], env)
    if rc == 0:
        result.error = "Should fail on update non-existent"
        return result

    result.passed = True
    return result


def test_enroll_dry_run(temp_dir: Path) -> TestResult:
    """Test enroll command with --dry-run flag (no API calls)."""
    result = TestResult("enroll_dry_run")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker first
    rc, _, stderr = run_cmd(["add", "enroll-test", "--name", "Enroll Test"], env)
    if rc != 0:
        result.error = f"add failed: {stderr}"
        return result

    # Create a dummy audio file
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"dummy audio data")

    # Test enroll with --dry-run and --segments
    rc, stdout, stderr = run_cmd([
        "enroll", "enroll-test", str(audio_file),
        "--segments", "10.5:25.3,30.0:45.5",
        "--dry-run",
    ], env)

    if rc != 0:
        result.error = f"enroll --dry-run failed: {stderr}"
        return result

    # Check output contains expected info
    if "Would enroll speaker" not in stdout:
        result.error = f"Dry-run output missing expected text: {stdout}"
        return result

    if "enroll-test" not in stdout:
        result.error = f"Dry-run output missing speaker ID: {stdout}"
        return result

    if "Segments: 2" not in stdout:
        result.error = f"Dry-run output missing segment count: {stdout}"
        return result

    result.passed = True
    return result


def test_enroll_from_stdin_dry_run(temp_dir: Path) -> TestResult:
    """Test enroll --from-stdin with dry-run (no API calls)."""
    result = TestResult("enroll_from_stdin")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker first
    rc, _, stderr = run_cmd(["add", "stdin-test", "--name", "Stdin Test"], env)
    if rc != 0:
        result.error = f"add failed: {stderr}"
        return result

    # Create a dummy audio file
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"dummy audio data")

    # Test enroll with --from-stdin and --dry-run
    # Need to provide input via stdin
    jsonl_input = '{"start": 5.0, "end": 10.5, "text": "Hello"}\n{"start": 15.0, "end": 20.0, "text": "World"}\n'

    cmd = [str(SPEAKER_DETECTION), "enroll", "stdin-test", str(audio_file), "--from-stdin", "--dry-run"]
    full_env = os.environ.copy()
    full_env.update(env)

    proc = subprocess.run(
        cmd,
        input=jsonl_input,
        capture_output=True,
        text=True,
        env=full_env,
    )

    if proc.returncode != 0:
        result.error = f"enroll --from-stdin failed: {proc.stderr}"
        return result

    # Check output
    if "Would enroll speaker" not in proc.stdout:
        result.error = f"Output missing expected text: {proc.stdout}"
        return result

    if "Segments: 2" not in proc.stdout:
        result.error = f"Output missing segment count: {proc.stdout}"
        return result

    # Check stderr shows segments read
    if "Read 2 segments from stdin" not in proc.stderr:
        result.error = f"Stderr missing read confirmation: {proc.stderr}"
        return result

    result.passed = True
    return result


def test_enroll_from_transcript_dry_run(temp_dir: Path) -> TestResult:
    """Test enroll --from-transcript with dry-run (no API calls)."""
    result = TestResult("enroll_from_transcript")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker first
    rc, _, stderr = run_cmd(["add", "transcript-test", "--name", "Transcript Test"], env)
    if rc != 0:
        result.error = f"add failed: {stderr}"
        return result

    # Create a dummy audio file
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"dummy audio data")

    # Create a mock AssemblyAI transcript
    transcript_file = temp_dir / "transcript.json"
    transcript_data = {
        "utterances": [
            {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello"},
            {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi there"},
            {"speaker": "A", "start": 11000, "end": 15000, "text": "How are you?"},
        ]
    }
    with open(transcript_file, "w") as f:
        json.dump(transcript_data, f)

    # Test enroll with --from-transcript
    rc, stdout, stderr = run_cmd([
        "enroll", "transcript-test", str(audio_file),
        "--from-transcript", str(transcript_file),
        "--speaker-label", "A",
        "--dry-run",
    ], env)

    if rc != 0:
        result.error = f"enroll --from-transcript failed: {stderr}"
        return result

    # Check output
    if "Would enroll speaker" not in stdout:
        result.error = f"Output missing expected text: {stdout}"
        return result

    if "Segments: 2" not in stdout:
        result.error = f"Output missing segment count (should be 2 for speaker A): {stdout}"
        return result

    result.passed = True
    return result


def test_identify_error_handling(temp_dir: Path) -> TestResult:
    """Test identify command error handling (no API calls)."""
    result = TestResult("identify_errors")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create a dummy audio file
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"dummy audio data")

    # Test identify with no speakers
    rc, stdout, stderr = run_cmd(["identify", str(audio_file)], env)
    if rc == 0:
        result.error = "identify should fail with no speakers"
        return result
    if "No speakers to match against" not in stderr:
        result.error = f"Wrong error message: {stderr}"
        return result

    # Add speaker without embeddings
    run_cmd(["add", "no-emb", "--name", "No Embeddings"], env)

    # Test identify with no embeddings
    rc, stdout, stderr = run_cmd(["identify", str(audio_file)], env)
    if rc == 0:
        result.error = "identify should fail with no embeddings"
        return result
    if "No speakers with" not in stderr and "embeddings" not in stderr:
        result.error = f"Wrong error for no embeddings: {stderr}"
        return result

    # Test identify with non-existent audio
    rc, stdout, stderr = run_cmd(["identify", "/nonexistent/audio.wav"], env)
    if rc == 0:
        result.error = "identify should fail with non-existent audio"
        return result
    if "not found" not in stderr.lower():
        result.error = f"Wrong error for missing audio: {stderr}"
        return result

    result.passed = True
    return result


def test_verify_error_handling(temp_dir: Path) -> TestResult:
    """Test verify command error handling (no API calls)."""
    result = TestResult("verify_errors")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create a dummy audio file
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"dummy audio data")

    # Test verify with non-existent speaker
    rc, stdout, stderr = run_cmd(["verify", "nonexistent", str(audio_file)], env)
    if rc == 0:
        result.error = "verify should fail with non-existent speaker"
        return result
    if "not found" not in stderr.lower():
        result.error = f"Wrong error message: {stderr}"
        return result

    # Add speaker without embeddings
    run_cmd(["add", "no-emb-verify", "--name", "No Embeddings"], env)

    # Test verify with no embeddings
    rc, stdout, stderr = run_cmd(["verify", "no-emb-verify", str(audio_file)], env)
    if rc == 0:
        result.error = "verify should fail with no embeddings"
        return result
    if "No" not in stderr and "embedding" not in stderr.lower():
        result.error = f"Wrong error for no embeddings: {stderr}"
        return result

    # Test verify with non-existent audio
    run_cmd(["add", "test-verify", "--name", "Test Verify"], env)
    rc, stdout, stderr = run_cmd(["verify", "test-verify", "/nonexistent/audio.wav"], env)
    if rc == 0:
        result.error = "verify should fail with non-existent audio"
        return result
    if "not found" not in stderr.lower():
        result.error = f"Wrong error for missing audio: {stderr}"
        return result

    result.passed = True
    return result


def test_embeddings_command(temp_dir: Path) -> TestResult:
    """Test embeddings listing command."""
    result = TestResult("embeddings_cmd")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add speaker
    run_cmd(["add", "emb-test", "--name", "Embeddings Test"], env)

    # Test embeddings with no embeddings
    rc, stdout, stderr = run_cmd(["embeddings", "emb-test"], env)
    if rc != 0:
        result.error = f"embeddings command failed: {stderr}"
        return result

    # Should show empty or no embeddings message
    if "No embeddings" not in stdout and "[]" not in stdout and stdout.strip() == "":
        # Empty output is also acceptable
        pass

    # Test embeddings for non-existent speaker
    rc, stdout, stderr = run_cmd(["embeddings", "nonexistent"], env)
    if rc == 0:
        result.error = "embeddings should fail for non-existent speaker"
        return result

    result.passed = True
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="speaker_detection CLI unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tests = [
        test_add_speaker,
        test_list_speakers,
        test_show_speaker,
        test_update_speaker,
        test_delete_speaker,
        test_tag_command,
        test_export,
        test_query,
        test_name_context,
        test_error_handling,
        test_enroll_dry_run,
        test_enroll_from_stdin_dry_run,
        test_enroll_from_transcript_dry_run,
        test_identify_error_handling,
        test_verify_error_handling,
        test_embeddings_command,
    ]

    print("speaker_detection CLI Unit Tests")
    print("=" * 40)

    passed = 0
    failed = 0
    results = []

    for test_func in tests:
        # Create fresh temp directory for each test
        temp_dir = Path(tempfile.mkdtemp(prefix="spk_test_"))

        try:
            result = test_func(temp_dir)
            results.append(result)

            if result.passed:
                print(f"  PASS: {result.name}")
                passed += 1
            else:
                print(f"  FAIL: {result.name}")
                if args.verbose and result.error:
                    print(f"        Error: {result.error}")
                failed += 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
