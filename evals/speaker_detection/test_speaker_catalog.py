#!/usr/bin/env python3
"""
Unit tests for speaker-catalog CLI tool.

Tests all CLI commands for recording inventory and processing state management.

Usage:
    ./test_speaker_catalog.py              # Run all tests
    ./test_speaker_catalog.py -v           # Verbose output
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
SPEAKER_CATALOG = REPO_ROOT / "speaker-catalog"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.skipped = False


def run_cmd(args: list, env: dict = None, stdin_input: str = None) -> tuple:
    """Run speaker-catalog command, return (returncode, stdout, stderr)."""
    cmd = [str(SPEAKER_CATALOG)] + args
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env,
        input=stdin_input,
    )
    return result.returncode, result.stdout, result.stderr


def create_test_audio(temp_dir: Path, filename: str = "test_audio.wav", duration: float = 1.0, unique_id: str = None) -> Path:
    """Create a test audio file with unique content.

    Args:
        temp_dir: Directory to create the file in
        filename: Name of the audio file
        duration: Duration in seconds
        unique_id: Optional unique identifier to embed in the file for unique b3sum
    """
    audio_path = temp_dir / filename

    # Generate unique identifier if not provided
    if unique_id is None:
        import uuid
        unique_id = str(uuid.uuid4())

    # Create a minimal WAV file with unique content
    # WAV header for specified duration at 44100 Hz, 16-bit mono
    import struct
    import hashlib

    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    file_size = 36 + data_size

    # Create deterministic but unique audio data based on unique_id
    # This ensures different files have different b3sum hashes
    hash_seed = hashlib.sha256(unique_id.encode()).digest()

    with open(audio_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # audio format (PCM)
        f.write(struct.pack("<H", 1))   # num channels
        f.write(struct.pack("<I", sample_rate))  # sample rate
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))

        # Write unique audio data based on hash_seed
        # Repeat the hash seed to fill the audio data, creating unique content
        seed_extended = (hash_seed * ((data_size // len(hash_seed)) + 1))[:data_size]
        f.write(seed_extended)

    return audio_path


def create_mock_transcript(temp_dir: Path, filename: str = "transcript.json") -> Path:
    """Create a mock AssemblyAI-style transcript."""
    transcript_path = temp_dir / filename
    transcript_data = {
        "utterances": [
            {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello everyone"},
            {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi there"},
            {"speaker": "A", "start": 11000, "end": 15000, "text": "How are you?"},
            {"speaker": "B", "start": 16000, "end": 20000, "text": "I'm doing well"},
        ]
    }
    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=2)
    return transcript_path


# =============================================================================
# Add Command Tests
# =============================================================================

def test_add_recording(temp_dir: Path) -> TestResult:
    """Test adding an audio file to the catalog."""
    result = TestResult("add_recording")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create test audio
    audio_path = create_test_audio(temp_dir)

    rc, stdout, stderr = run_cmd(["add", str(audio_path)], env)

    if rc != 0:
        result.error = f"add command failed: {stderr}"
        return result

    if "Added:" not in stdout:
        result.error = f"Missing 'Added:' in output: {stdout}"
        return result

    if "b3sum:" not in stdout:
        result.error = f"Missing 'b3sum:' in output: {stdout}"
        return result

    # Verify catalog entry was created
    catalog_dir = temp_dir / "catalog"
    entries = list(catalog_dir.glob("*.yaml"))
    if len(entries) != 1:
        result.error = f"Expected 1 catalog entry, got {len(entries)}"
        return result

    result.passed = True
    return result


def test_add_with_context(temp_dir: Path) -> TestResult:
    """Test adding with --context and --tags options."""
    result = TestResult("add_with_context")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)

    rc, stdout, stderr = run_cmd([
        "add", str(audio_path),
        "--context", "team-meeting",
        "--tags", "meeting,weekly,team-a",
    ], env)

    if rc != 0:
        result.error = f"add command failed: {stderr}"
        return result

    if "context: team-meeting" not in stdout:
        result.error = f"Context not shown in output: {stdout}"
        return result

    # Verify catalog entry has context
    catalog_dir = temp_dir / "catalog"
    entries = list(catalog_dir.glob("*.yaml"))
    if len(entries) != 1:
        result.error = f"Expected 1 catalog entry, got {len(entries)}"
        return result

    # Load and check entry
    import yaml
    with open(entries[0]) as f:
        entry = yaml.safe_load(f)

    ctx = entry.get("context", {})
    if ctx.get("name") != "team-meeting":
        result.error = f"Context name mismatch: {ctx.get('name')}"
        return result

    tags = ctx.get("tags", [])
    if set(tags) != {"meeting", "weekly", "team-a"}:
        result.error = f"Tags mismatch: {tags}"
        return result

    result.passed = True
    return result


def test_add_duplicate_fails(temp_dir: Path) -> TestResult:
    """Test that adding duplicate recording fails without --force."""
    result = TestResult("add_duplicate_fails")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)

    # Add first time
    rc, _, _ = run_cmd(["add", str(audio_path)], env)
    if rc != 0:
        result.error = "First add should succeed"
        return result

    # Add second time (should fail)
    rc, stdout, stderr = run_cmd(["add", str(audio_path)], env)
    if rc == 0:
        result.error = "Duplicate add should fail without --force"
        return result

    if "already in catalog" not in stderr:
        result.error = f"Expected 'already in catalog' error: {stderr}"
        return result

    result.passed = True
    return result


# =============================================================================
# List Command Tests
# =============================================================================

def test_list_empty(temp_dir: Path) -> TestResult:
    """Test listing when catalog is empty."""
    result = TestResult("list_empty")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["list"], env)

    if rc != 0:
        result.error = f"list command failed: {stderr}"
        return result

    if "No recordings in catalog" not in stdout:
        result.error = f"Expected 'No recordings' message: {stdout}"
        return result

    result.passed = True
    return result


def test_list_with_entries(temp_dir: Path) -> TestResult:
    """Test listing after adding recordings."""
    result = TestResult("list_with_entries")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add multiple recordings
    audio1 = create_test_audio(temp_dir, "audio1.wav")
    audio2 = create_test_audio(temp_dir, "audio2.wav", duration=2.0)
    audio3 = create_test_audio(temp_dir, "audio3.wav", duration=3.0)

    run_cmd(["add", str(audio1), "--context", "ctx-a"], env)
    run_cmd(["add", str(audio2), "--context", "ctx-b"], env)
    run_cmd(["add", str(audio3), "--context", "ctx-a"], env)

    # List all
    rc, stdout, stderr = run_cmd(["list"], env)

    if rc != 0:
        result.error = f"list command failed: {stderr}"
        return result

    if "Total: 3 recording(s)" not in stdout:
        result.error = f"Expected 3 recordings: {stdout}"
        return result

    # Test JSON format
    rc, stdout, _ = run_cmd(["list", "--format", "json"], env)
    if rc != 0:
        result.error = "list --format json failed"
        return result

    entries = json.loads(stdout)
    if len(entries) != 3:
        result.error = f"JSON output has {len(entries)} entries, expected 3"
        return result

    result.passed = True
    return result


def test_list_filter_by_status(temp_dir: Path) -> TestResult:
    """Test filtering list by --status."""
    result = TestResult("list_filter_by_status")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add recordings (all will be "unprocessed" initially)
    audio1 = create_test_audio(temp_dir, "audio1.wav")
    audio2 = create_test_audio(temp_dir, "audio2.wav")

    run_cmd(["add", str(audio1)], env)
    run_cmd(["add", str(audio2)], env)

    # Register transcript for audio1 to change its status
    transcript = create_mock_transcript(temp_dir)
    run_cmd([
        "register-transcript", str(audio1),
        "--backend", "assemblyai",
        "--transcript", str(transcript),
    ], env)

    # Filter by unprocessed
    rc, stdout, _ = run_cmd(["list", "--status", "unprocessed", "--format", "json"], env)
    if rc != 0:
        result.error = "list --status unprocessed failed"
        return result

    entries = json.loads(stdout)
    if len(entries) != 1:
        result.error = f"Expected 1 unprocessed, got {len(entries)}"
        return result

    # Filter by transcribed
    rc, stdout, _ = run_cmd(["list", "--status", "transcribed", "--format", "json"], env)
    if rc != 0:
        result.error = "list --status transcribed failed"
        return result

    entries = json.loads(stdout)
    if len(entries) != 1:
        result.error = f"Expected 1 transcribed, got {len(entries)}"
        return result

    result.passed = True
    return result


def test_list_filter_by_context(temp_dir: Path) -> TestResult:
    """Test filtering list by --context."""
    result = TestResult("list_filter_by_context")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Add recordings with different contexts
    audio1 = create_test_audio(temp_dir, "audio1.wav")
    audio2 = create_test_audio(temp_dir, "audio2.wav")
    audio3 = create_test_audio(temp_dir, "audio3.wav")

    run_cmd(["add", str(audio1), "--context", "team-standup"], env)
    run_cmd(["add", str(audio2), "--context", "interview"], env)
    run_cmd(["add", str(audio3), "--context", "team-standup"], env)

    # Filter by context
    rc, stdout, _ = run_cmd(["list", "--context", "team-standup", "--format", "json"], env)

    if rc != 0:
        result.error = "list --context failed"
        return result

    entries = json.loads(stdout)
    if len(entries) != 2:
        result.error = f"Expected 2 team-standup recordings, got {len(entries)}"
        return result

    # Verify all have correct context
    for e in entries:
        if e["context"] != "team-standup":
            result.error = f"Wrong context: {e['context']}"
            return result

    result.passed = True
    return result


# =============================================================================
# Show Command Tests
# =============================================================================

def test_show_recording(temp_dir: Path) -> TestResult:
    """Test showing details of a recording."""
    result = TestResult("show_recording")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir, duration=5.0)
    run_cmd(["add", str(audio_path), "--context", "demo", "--tags", "test,example"], env)

    rc, stdout, stderr = run_cmd(["show", str(audio_path)], env)

    if rc != 0:
        result.error = f"show command failed: {stderr}"
        return result

    # Check key information is displayed
    checks = [
        ("Recording:" in stdout, "Missing 'Recording:' header"),
        ("B3SUM:" in stdout, "Missing B3SUM"),
        ("Duration:" in stdout, "Missing Duration"),
        ("Status:" in stdout, "Missing Status"),
        ("Context:" in stdout, "Missing Context"),
        ("demo" in stdout, "Missing context name 'demo'"),
    ]

    for check, msg in checks:
        if not check:
            result.error = msg
            return result

    # Test JSON format
    rc, stdout, _ = run_cmd(["show", str(audio_path), "--format", "json"], env)
    if rc != 0:
        result.error = "show --format json failed"
        return result

    entry = json.loads(stdout)
    if entry.get("recording", {}).get("b3sum") is None:
        result.error = "JSON output missing b3sum"
        return result

    result.passed = True
    return result


def test_show_nonexistent(temp_dir: Path) -> TestResult:
    """Test showing a non-existent recording fails."""
    result = TestResult("show_nonexistent")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Use a file path that exists in the temp_dir but is not in catalog
    # to avoid glob issues with absolute paths starting with /
    audio_path = create_test_audio(temp_dir, "not_in_catalog.wav")

    rc, stdout, stderr = run_cmd(["show", str(audio_path)], env)

    if rc == 0:
        result.error = "show should fail for recording not in catalog"
        return result

    if "not in catalog" not in stderr.lower():
        result.error = f"Expected 'not in catalog' error: {stderr}"
        return result

    result.passed = True
    return result


# =============================================================================
# Status Command Tests
# =============================================================================

def test_status_unprocessed(temp_dir: Path) -> TestResult:
    """Test status of a new (unprocessed) recording."""
    result = TestResult("status_unprocessed")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    rc, stdout, stderr = run_cmd(["status", str(audio_path)], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    if stdout.strip() != "unprocessed":
        result.error = f"Expected 'unprocessed', got: {stdout.strip()}"
        return result

    result.passed = True
    return result


def test_status_after_transcript(temp_dir: Path) -> TestResult:
    """Test status changes to 'transcribed' after registering transcript."""
    result = TestResult("status_after_transcript")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    # Register transcript
    transcript = create_mock_transcript(temp_dir)
    run_cmd([
        "register-transcript", str(audio_path),
        "--backend", "assemblyai",
        "--transcript", str(transcript),
    ], env)

    rc, stdout, stderr = run_cmd(["status", str(audio_path)], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    if stdout.strip() != "transcribed":
        result.error = f"Expected 'transcribed', got: {stdout.strip()}"
        return result

    # Test JSON format
    rc, stdout, _ = run_cmd(["status", str(audio_path), "--format", "json"], env)
    status_data = json.loads(stdout)
    if status_data.get("status") != "transcribed":
        result.error = f"JSON status mismatch: {status_data}"
        return result

    result.passed = True
    return result


# =============================================================================
# Register-Transcript Command Tests
# =============================================================================

def test_register_transcript(temp_dir: Path) -> TestResult:
    """Test registering a transcript for a recording."""
    result = TestResult("register_transcript")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    transcript = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "register-transcript", str(audio_path),
        "--backend", "assemblyai",
        "--transcript", str(transcript),
    ], env)

    if rc != 0:
        result.error = f"register-transcript failed: {stderr}"
        return result

    if "Registered transcript:" not in stdout:
        result.error = f"Missing confirmation: {stdout}"
        return result

    if "Backend: assemblyai" not in stdout:
        result.error = f"Missing backend info: {stdout}"
        return result

    if "Speakers detected: 2" not in stdout:
        result.error = f"Missing speaker count: {stdout}"
        return result

    # Verify in show output
    rc, stdout, _ = run_cmd(["show", str(audio_path), "--format", "json"], env)
    entry = json.loads(stdout)

    transcriptions = entry.get("transcriptions", [])
    if len(transcriptions) != 1:
        result.error = f"Expected 1 transcription, got {len(transcriptions)}"
        return result

    if transcriptions[0].get("backend") != "assemblyai":
        result.error = f"Backend mismatch: {transcriptions[0]}"
        return result

    result.passed = True
    return result


def test_register_transcript_multiple_backends(temp_dir: Path) -> TestResult:
    """Test registering transcripts from multiple backends."""
    result = TestResult("register_transcript_multiple_backends")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    # Create transcripts
    transcript1 = create_mock_transcript(temp_dir, "transcript1.json")
    transcript2 = create_mock_transcript(temp_dir, "transcript2.json")

    # Register two different backends
    run_cmd([
        "register-transcript", str(audio_path),
        "--backend", "assemblyai",
        "--transcript", str(transcript1),
    ], env)

    run_cmd([
        "register-transcript", str(audio_path),
        "--backend", "speechmatics",
        "--transcript", str(transcript2),
    ], env)

    # Verify both are registered
    rc, stdout, _ = run_cmd(["show", str(audio_path), "--format", "json"], env)
    entry = json.loads(stdout)

    transcriptions = entry.get("transcriptions", [])
    if len(transcriptions) != 2:
        result.error = f"Expected 2 transcriptions, got {len(transcriptions)}"
        return result

    backends = {t["backend"] for t in transcriptions}
    if backends != {"assemblyai", "speechmatics"}:
        result.error = f"Backend mismatch: {backends}"
        return result

    result.passed = True
    return result


# =============================================================================
# Set-Context Command Tests
# =============================================================================

def test_set_context(temp_dir: Path) -> TestResult:
    """Test updating context of a recording."""
    result = TestResult("set_context")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    rc, stdout, stderr = run_cmd([
        "set-context", str(audio_path),
        "--context", "team-retrospective",
        "--expected-speakers", "alice,bob,charlie",
    ], env)

    if rc != 0:
        result.error = f"set-context failed: {stderr}"
        return result

    if "Updated context" not in stdout:
        result.error = f"Missing confirmation: {stdout}"
        return result

    # Verify context was set
    rc, stdout, _ = run_cmd(["show", str(audio_path), "--format", "json"], env)
    entry = json.loads(stdout)

    ctx = entry.get("context", {})
    if ctx.get("name") != "team-retrospective":
        result.error = f"Context name not set: {ctx}"
        return result

    expected = ctx.get("expected_speakers", [])
    if set(expected) != {"alice", "bob", "charlie"}:
        result.error = f"Expected speakers mismatch: {expected}"
        return result

    result.passed = True
    return result


def test_set_context_tags(temp_dir: Path) -> TestResult:
    """Test adding and removing tags via set-context."""
    result = TestResult("set_context_tags")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path), "--tags", "original"], env)

    # Add more tags
    run_cmd([
        "set-context", str(audio_path),
        "--tags", "new-tag1,new-tag2",
    ], env)

    rc, stdout, _ = run_cmd(["show", str(audio_path), "--format", "json"], env)
    entry = json.loads(stdout)
    tags = entry.get("context", {}).get("tags", [])

    if "original" not in tags or "new-tag1" not in tags:
        result.error = f"Tags not added correctly: {tags}"
        return result

    # Remove a tag
    run_cmd([
        "set-context", str(audio_path),
        "--remove-tags", "original",
    ], env)

    rc, stdout, _ = run_cmd(["show", str(audio_path), "--format", "json"], env)
    entry = json.loads(stdout)
    tags = entry.get("context", {}).get("tags", [])

    if "original" in tags:
        result.error = f"Tag 'original' should have been removed: {tags}"
        return result

    if "new-tag1" not in tags:
        result.error = f"Tag 'new-tag1' should still exist: {tags}"
        return result

    result.passed = True
    return result


# =============================================================================
# Remove Command Tests
# =============================================================================

def test_remove_recording(temp_dir: Path) -> TestResult:
    """Test removing a recording from the catalog."""
    result = TestResult("remove_recording")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    # Verify exists
    rc, _, _ = run_cmd(["show", str(audio_path)], env)
    if rc != 0:
        result.error = "Recording not added"
        return result

    # Remove with --force (skip confirmation)
    rc, stdout, stderr = run_cmd(["remove", str(audio_path), "--force"], env)

    if rc != 0:
        result.error = f"remove failed: {stderr}"
        return result

    if "Removed:" not in stdout:
        result.error = f"Missing removal confirmation: {stdout}"
        return result

    # Verify removed
    rc, _, _ = run_cmd(["show", str(audio_path)], env)
    if rc == 0:
        result.error = "Recording should be removed"
        return result

    result.passed = True
    return result


def test_remove_by_b3sum_prefix(temp_dir: Path) -> TestResult:
    """Test removing a recording by b3sum prefix."""
    result = TestResult("remove_by_b3sum_prefix")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path)], env)

    # Get the b3sum from list
    rc, stdout, _ = run_cmd(["list", "--format", "json"], env)
    entries = json.loads(stdout)
    b3sum = entries[0]["b3sum"]
    prefix = b3sum[:8]  # Use first 8 characters

    # Remove by prefix
    rc, stdout, stderr = run_cmd(["remove", prefix, "--force"], env)

    if rc != 0:
        result.error = f"remove by prefix failed: {stderr}"
        return result

    # Verify removed
    rc, stdout, _ = run_cmd(["list", "--format", "json"], env)
    entries = json.loads(stdout)
    if len(entries) != 0:
        result.error = "Recording should be removed"
        return result

    result.passed = True
    return result


# =============================================================================
# Query Command Tests
# =============================================================================

def test_query_jq(temp_dir: Path) -> TestResult:
    """Test query command with jq expression."""
    result = TestResult("query_jq")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Check if jq is available
    try:
        subprocess.run(["jq", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        result.error = "jq not available, skipping"
        result.passed = True  # Skip, not fail
        result.skipped = True
        return result

    # Add recordings with different contexts
    audio1 = create_test_audio(temp_dir, "audio1.wav")
    audio2 = create_test_audio(temp_dir, "audio2.wav")

    run_cmd(["add", str(audio1), "--context", "ctx-1"], env)
    run_cmd(["add", str(audio2), "--context", "ctx-2"], env)

    # Query for b3sums
    rc, stdout, stderr = run_cmd(["query", ".[].recording.b3sum"], env)

    if rc != 0:
        result.error = f"query failed: {stderr}"
        return result

    # Should have 2 b3sum values
    lines = [l for l in stdout.strip().split("\n") if l.strip()]
    if len(lines) != 2:
        result.error = f"Expected 2 b3sums in output, got {len(lines)}: {stdout}"
        return result

    # Query with filter
    rc, stdout, _ = run_cmd(["query", '.[] | select(.context.name == "ctx-1") | .recording.b3sum'], env)

    if rc != 0:
        result.error = "query with filter failed"
        return result

    lines = [l for l in stdout.strip().split("\n") if l.strip()]
    if len(lines) != 1:
        result.error = f"Filter should return 1 result, got {len(lines)}"
        return result

    result.passed = True
    return result


def test_query_complex_expression(temp_dir: Path) -> TestResult:
    """Test query with more complex jq expressions."""
    result = TestResult("query_complex_expression")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Check if jq is available
    try:
        subprocess.run(["jq", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        result.skipped = True
        result.passed = True
        return result

    # Add recordings
    audio1 = create_test_audio(temp_dir, "audio1.wav")
    audio2 = create_test_audio(temp_dir, "audio2.wav")
    audio3 = create_test_audio(temp_dir, "audio3.wav")

    run_cmd(["add", str(audio1), "--context", "meeting", "--tags", "important"], env)
    run_cmd(["add", str(audio2), "--context", "interview"], env)
    run_cmd(["add", str(audio3), "--context", "meeting", "--tags", "weekly"], env)

    # Count entries by context
    rc, stdout, _ = run_cmd(["query", "group_by(.context.name) | map({context: .[0].context.name, count: length})"], env)

    if rc != 0:
        result.error = "Complex query failed"
        return result

    data = json.loads(stdout)
    # Should have 2 groups: meeting (2) and interview (1)
    if len(data) != 2:
        result.error = f"Expected 2 groups, got {len(data)}"
        return result

    result.passed = True
    return result


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_add_nonexistent_file(temp_dir: Path) -> TestResult:
    """Test adding a non-existent file fails."""
    result = TestResult("add_nonexistent_file")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["add", "/nonexistent/path/audio.wav"], env)

    if rc == 0:
        result.error = "add should fail for non-existent file"
        return result

    if "not found" not in stderr:
        result.error = f"Expected 'not found' error: {stderr}"
        return result

    result.passed = True
    return result


def test_register_transcript_not_in_catalog(temp_dir: Path) -> TestResult:
    """Test registering transcript for recording not in catalog fails."""
    result = TestResult("register_transcript_not_in_catalog")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript = create_mock_transcript(temp_dir)

    # Don't add to catalog, try to register transcript
    rc, stdout, stderr = run_cmd([
        "register-transcript", str(audio_path),
        "--backend", "test",
        "--transcript", str(transcript),
    ], env)

    if rc == 0:
        result.error = "Should fail for recording not in catalog"
        return result

    if "not in catalog" not in stderr.lower():
        result.error = f"Expected catalog error: {stderr}"
        return result

    result.passed = True
    return result


def test_status_not_in_catalog(temp_dir: Path) -> TestResult:
    """Test status for recording not in catalog fails."""
    result = TestResult("status_not_in_catalog")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["status", "/nonexistent/audio.wav"], env)

    if rc == 0:
        result.error = "status should fail for non-existent recording"
        return result

    result.passed = True
    return result


def test_b3sum_prefix_lookup(temp_dir: Path) -> TestResult:
    """Test that b3sum prefix can be used to reference recordings."""
    result = TestResult("b3sum_prefix_lookup")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    run_cmd(["add", str(audio_path), "--context", "test-context"], env)

    # Get the b3sum
    rc, stdout, _ = run_cmd(["list", "--format", "json"], env)
    entries = json.loads(stdout)
    b3sum = entries[0]["b3sum"]

    # Show by b3sum prefix
    prefix = b3sum[:8]
    rc, stdout, stderr = run_cmd(["show", prefix], env)

    if rc != 0:
        result.error = f"show by b3sum prefix failed: {stderr}"
        return result

    if "test-context" not in stdout:
        result.error = "Context not found in show output"
        return result

    # Status by b3sum prefix
    rc, stdout, _ = run_cmd(["status", prefix], env)
    if rc != 0:
        result.error = "status by b3sum prefix failed"
        return result

    result.passed = True
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="speaker-catalog CLI unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check for pyyaml
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required for tests. Install with: pip install pyyaml")
        return 2

    tests = [
        # Add command tests
        test_add_recording,
        test_add_with_context,
        test_add_duplicate_fails,

        # List command tests
        test_list_empty,
        test_list_with_entries,
        test_list_filter_by_status,
        test_list_filter_by_context,

        # Show command tests
        test_show_recording,
        test_show_nonexistent,

        # Status command tests
        test_status_unprocessed,
        test_status_after_transcript,

        # Register-transcript command tests
        test_register_transcript,
        test_register_transcript_multiple_backends,

        # Set-context command tests
        test_set_context,
        test_set_context_tags,

        # Remove command tests
        test_remove_recording,
        test_remove_by_b3sum_prefix,

        # Query command tests
        test_query_jq,
        test_query_complex_expression,

        # Error handling tests
        test_add_nonexistent_file,
        test_register_transcript_not_in_catalog,
        test_status_not_in_catalog,
        test_b3sum_prefix_lookup,
    ]

    print("speaker-catalog CLI Unit Tests")
    print("=" * 40)

    passed = 0
    failed = 0
    skipped = 0
    results = []

    for test_func in tests:
        # Create fresh temp directory for each test
        temp_dir = Path(tempfile.mkdtemp(prefix="catalog_test_"))

        try:
            result = test_func(temp_dir)
            results.append(result)

            if result.skipped:
                print(f"  SKIP: {result.name}")
                skipped += 1
            elif result.passed:
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
                import traceback
                print(f"        {e}")
                traceback.print_exc()
            failed += 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
