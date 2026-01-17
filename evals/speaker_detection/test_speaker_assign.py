#!/usr/bin/env python3
"""
Unit tests for speaker-assign CLI tool.

Tests multi-signal speaker name assignment for transcript labels.

Usage:
    ./test_speaker_assign.py              # Run all tests
    ./test_speaker_assign.py -v           # Verbose output
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
SPEAKER_ASSIGN = REPO_ROOT / "speaker-assign"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.skipped = False


def run_cmd(args: list, env: dict = None, stdin_input: str = None) -> tuple:
    """Run speaker-assign command, return (returncode, stdout, stderr)."""
    cmd = [str(SPEAKER_ASSIGN)] + args
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


def create_mock_transcript(temp_dir: Path, filename: str = "transcript.json", num_speakers: int = 2) -> Path:
    """Create a mock AssemblyAI-style transcript with multiple speakers."""
    transcript_path = temp_dir / filename

    if num_speakers == 2:
        transcript_data = {
            "utterances": [
                {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello everyone, this is Alice speaking"},
                {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi Alice, Bob here"},
                {"speaker": "A", "start": 11000, "end": 15000, "text": "How is the project going?"},
                {"speaker": "B", "start": 16000, "end": 20000, "text": "Making good progress, thanks for asking"},
                {"speaker": "A", "start": 21000, "end": 25000, "text": "Great, let me know if you need any help"},
            ]
        }
    elif num_speakers == 3:
        transcript_data = {
            "utterances": [
                {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello everyone, I'm Alice"},
                {"speaker": "B", "start": 6000, "end": 10000, "text": "Hi, Bob here"},
                {"speaker": "C", "start": 11000, "end": 15000, "text": "And I'm Carol"},
                {"speaker": "A", "start": 16000, "end": 20000, "text": "Let's start the meeting"},
                {"speaker": "B", "start": 21000, "end": 25000, "text": "Sounds good"},
                {"speaker": "C", "start": 26000, "end": 30000, "text": "I have some updates"},
            ]
        }
    else:
        transcript_data = {
            "utterances": [
                {"speaker": "A", "start": 1000, "end": 5000, "text": "Hello there"},
            ]
        }

    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=2)
    return transcript_path


def create_mock_transcript_speechmatics(temp_dir: Path, filename: str = "transcript_sm.json") -> Path:
    """Create a mock Speechmatics-style transcript."""
    transcript_path = temp_dir / filename
    transcript_data = {
        "results": [
            {"start_time": 1.0, "end_time": 2.0, "speaker": "S1", "alternatives": [{"content": "Hello", "speaker": "S1"}]},
            {"start_time": 2.5, "end_time": 3.5, "speaker": "S2", "alternatives": [{"content": "Hi there", "speaker": "S2"}]},
            {"start_time": 4.0, "end_time": 5.0, "speaker": "S1", "alternatives": [{"content": "How are you", "speaker": "S1"}]},
        ]
    }
    with open(transcript_path, "w") as f:
        json.dump(transcript_data, f, indent=2)
    return transcript_path


def setup_catalog_entry(temp_dir: Path, audio_path: Path, context_name: str = None, expected_speakers: list = None) -> str:
    """Create a catalog entry for an audio file and return the b3sum."""
    import hashlib

    # Compute b3sum (or sha256 fallback)
    try:
        result = subprocess.run(
            ["b3sum", "--no-names", str(audio_path)],
            capture_output=True,
            text=True,
            check=True
        )
        b3sum = result.stdout.strip()[:32]
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha256 = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        b3sum = sha256.hexdigest()[:32]

    # Create catalog directory
    catalog_dir = temp_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # Create catalog entry
    import yaml
    entry = {
        "recording": {
            "b3sum": b3sum,
            "original_path": str(audio_path),
        },
        "context": {
            "name": context_name,
            "expected_speakers": expected_speakers or [],
        },
    }

    catalog_path = catalog_dir / f"{b3sum}.yaml"
    with open(catalog_path, "w") as f:
        yaml.dump(entry, f, default_flow_style=False)

    return b3sum


# =============================================================================
# Basic Assignment Tests
# =============================================================================

def test_assign_basic(temp_dir: Path) -> TestResult:
    """Test basic assignment with transcript only (no signals)."""
    result = TestResult("assign_basic")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    if rc != 0:
        result.error = f"assign command failed: {stderr}"
        return result

    if "Found 2 speakers" not in stdout:
        result.error = f"Expected 'Found 2 speakers' in output: {stdout}"
        return result

    if "Assignments saved" not in stdout:
        result.error = f"Missing 'Assignments saved' in output: {stdout}"
        return result

    # Verify file was saved
    assignments_dir = temp_dir / "assignments"
    entries = list(assignments_dir.glob("*.yaml"))
    if len(entries) != 1:
        result.error = f"Expected 1 assignment file, got {len(entries)}"
        return result

    result.passed = True
    return result


def test_assign_three_speakers(temp_dir: Path) -> TestResult:
    """Test assignment with 3 speakers."""
    result = TestResult("assign_three_speakers")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir, num_speakers=3)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    if rc != 0:
        result.error = f"assign command failed: {stderr}"
        return result

    if "Found 3 speakers" not in stdout:
        result.error = f"Expected 'Found 3 speakers' in output: {stdout}"
        return result

    result.passed = True
    return result


def parse_json_output(stdout: str) -> dict:
    """Parse JSON from stdout, handling potential text before JSON."""
    # Find where JSON starts (first '{' or '[')
    json_start = -1
    for i, c in enumerate(stdout):
        if c in '{[':
            json_start = i
            break
    if json_start >= 0:
        return json.loads(stdout[json_start:])
    return json.loads(stdout)


def test_assign_with_expected_speakers(temp_dir: Path) -> TestResult:
    """Test assignment using expected speakers from context."""
    result = TestResult("assign_with_expected_speakers")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # Create catalog entry with expected speakers
    b3sum = setup_catalog_entry(
        temp_dir, audio_path,
        context_name="team-meeting",
        expected_speakers=["alice", "bob"]
    )

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--format", "json",
    ], env)

    if rc != 0:
        result.error = f"assign command failed: {stderr}"
        return result

    # Parse JSON output
    try:
        data = parse_json_output(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON output: {stdout}"
        return result

    if data.get("context") != "team-meeting":
        result.error = f"Expected context 'team-meeting', got: {data.get('context')}"
        return result

    result.passed = True
    return result


def test_assign_with_cli_expected_speakers(temp_dir: Path) -> TestResult:
    """Test assignment with --expected-speakers CLI option."""
    result = TestResult("assign_with_cli_expected_speakers")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--expected-speakers", "alice,bob,carol",
        "--format", "json",
    ], env)

    if rc != 0:
        result.error = f"assign command failed: {stderr}"
        return result

    # Parse JSON output
    try:
        data = parse_json_output(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON output: {stdout}"
        return result

    # Check that mappings exist
    mappings = data.get("mappings", {})
    if "A" not in mappings or "B" not in mappings:
        result.error = f"Missing speaker mappings: {mappings.keys()}"
        return result

    result.passed = True
    return result


# =============================================================================
# Dry Run Tests
# =============================================================================

def test_assign_dry_run(temp_dir: Path) -> TestResult:
    """Test that dry run doesn't save assignments."""
    result = TestResult("assign_dry_run")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--dry-run",
    ], env)

    if rc != 0:
        result.error = f"assign --dry-run failed: {stderr}"
        return result

    if "DRY RUN" not in stdout:
        result.error = f"Expected 'DRY RUN' in output: {stdout}"
        return result

    # Verify no file was saved
    assignments_dir = temp_dir / "assignments"
    if assignments_dir.exists():
        entries = list(assignments_dir.glob("*.yaml"))
        if len(entries) > 0:
            result.error = f"Dry run should not create files, found {len(entries)}"
            return result

    result.passed = True
    return result


def test_assign_dry_run_json(temp_dir: Path) -> TestResult:
    """Test dry run with JSON format output."""
    result = TestResult("assign_dry_run_json")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--dry-run",
        "--format", "json",
    ], env)

    if rc != 0:
        result.error = f"assign --dry-run --format json failed: {stderr}"
        return result

    # Should still output valid JSON
    try:
        # Find JSON in output (may have DRY RUN message before)
        json_start = stdout.find("{")
        if json_start >= 0:
            data = json.loads(stdout[json_start:])
        else:
            data = json.loads(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON in dry-run output: {stdout}"
        return result

    if "mappings" not in data:
        result.error = f"Missing 'mappings' in output: {data.keys()}"
        return result

    result.passed = True
    return result


# =============================================================================
# Save and Persistence Tests
# =============================================================================

def test_assign_saves_to_assignments_dir(temp_dir: Path) -> TestResult:
    """Test that assignments are saved to YAML file."""
    result = TestResult("assign_saves_to_assignments_dir")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    if rc != 0:
        result.error = f"assign command failed: {stderr}"
        return result

    # Verify file exists
    assignments_dir = temp_dir / "assignments"
    entries = list(assignments_dir.glob("*.yaml"))
    if len(entries) != 1:
        result.error = f"Expected 1 assignment file, got {len(entries)}"
        return result

    # Load and verify content
    import yaml
    with open(entries[0]) as f:
        data = yaml.safe_load(f)

    if "schema_version" not in data:
        result.error = f"Missing schema_version in saved file"
        return result

    if "recording_b3sum" not in data:
        result.error = f"Missing recording_b3sum in saved file"
        return result

    if "mappings" not in data:
        result.error = f"Missing mappings in saved file"
        return result

    result.passed = True
    return result


def test_assign_output_file(temp_dir: Path) -> TestResult:
    """Test --output option saves to specified file."""
    result = TestResult("assign_output_file")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)
    output_path = temp_dir / "custom_output.yaml"

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--output", str(output_path),
    ], env)

    if rc != 0:
        result.error = f"assign with --output failed: {stderr}"
        return result

    if not output_path.exists():
        result.error = f"Output file not created: {output_path}"
        return result

    # Verify content
    import yaml
    with open(output_path) as f:
        data = yaml.safe_load(f)

    if "mappings" not in data:
        result.error = f"Missing mappings in output file"
        return result

    result.passed = True
    return result


# =============================================================================
# Show Command Tests
# =============================================================================

def test_show_assignments(temp_dir: Path) -> TestResult:
    """Test show command displays assignments."""
    result = TestResult("show_assignments")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # First, create assignments
    run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    # Then show
    rc, stdout, stderr = run_cmd(["show", str(audio_path)], env)

    if rc != 0:
        result.error = f"show command failed: {stderr}"
        return result

    if "Assignments for:" not in stdout:
        result.error = f"Missing 'Assignments for:' in output: {stdout}"
        return result

    if "Mappings:" not in stdout:
        result.error = f"Missing 'Mappings:' in output: {stdout}"
        return result

    result.passed = True
    return result


def test_show_assignments_json(temp_dir: Path) -> TestResult:
    """Test show command with JSON format."""
    result = TestResult("show_assignments_json")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # First, create assignments
    run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    # Then show as JSON
    rc, stdout, stderr = run_cmd(["show", str(audio_path), "--format", "json"], env)

    if rc != 0:
        result.error = f"show --format json failed: {stderr}"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON output: {stdout}"
        return result

    if "mappings" not in data:
        result.error = f"Missing 'mappings' in JSON: {data.keys()}"
        return result

    result.passed = True
    return result


def test_show_nonexistent(temp_dir: Path) -> TestResult:
    """Test show for recording without assignments."""
    result = TestResult("show_nonexistent")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)

    rc, stdout, stderr = run_cmd(["show", str(audio_path)], env)

    if rc == 0:
        result.error = "show should fail for recording without assignments"
        return result

    if "No assignments found" not in stderr:
        result.error = f"Expected 'No assignments found' error: {stderr}"
        return result

    result.passed = True
    return result


def test_show_by_b3sum_prefix(temp_dir: Path) -> TestResult:
    """Test show command with b3sum prefix."""
    result = TestResult("show_by_b3sum_prefix")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # Create catalog entry to get b3sum
    b3sum = setup_catalog_entry(temp_dir, audio_path)

    # Create assignments
    run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    # Show by b3sum prefix
    prefix = b3sum[:8]
    rc, stdout, stderr = run_cmd(["show", prefix], env)

    if rc != 0:
        result.error = f"show by b3sum prefix failed: {stderr}"
        return result

    if "Assignments for:" not in stdout:
        result.error = f"Missing 'Assignments for:' in output: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Clear Command Tests
# =============================================================================

def test_clear_assignments(temp_dir: Path) -> TestResult:
    """Test clear removes assignment file."""
    result = TestResult("clear_assignments")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # First, create assignments
    run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    # Verify file exists
    assignments_dir = temp_dir / "assignments"
    entries_before = list(assignments_dir.glob("*.yaml"))
    if len(entries_before) != 1:
        result.error = f"Assignment not created before clear"
        return result

    # Clear with --force
    rc, stdout, stderr = run_cmd(["clear", str(audio_path), "--force"], env)

    if rc != 0:
        result.error = f"clear command failed: {stderr}"
        return result

    if "Cleared assignments" not in stdout:
        result.error = f"Missing 'Cleared assignments' in output: {stdout}"
        return result

    # Verify file removed
    entries_after = list(assignments_dir.glob("*.yaml"))
    if len(entries_after) != 0:
        result.error = f"Assignment file should be removed, found {len(entries_after)}"
        return result

    result.passed = True
    return result


def test_clear_nonexistent(temp_dir: Path) -> TestResult:
    """Test clear for recording without assignments."""
    result = TestResult("clear_nonexistent")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)

    rc, stdout, stderr = run_cmd(["clear", str(audio_path), "--force"], env)

    # Should succeed gracefully (nothing to clear)
    if rc != 0:
        result.error = f"clear should succeed gracefully: {stderr}"
        return result

    result.passed = True
    return result


# =============================================================================
# JSON Output Tests
# =============================================================================

def test_assign_json_output(temp_dir: Path) -> TestResult:
    """Test JSON format output structure."""
    result = TestResult("assign_json_output")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--format", "json",
    ], env)

    if rc != 0:
        result.error = f"assign --format json failed: {stderr}"
        return result

    try:
        data = parse_json_output(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON output: {stdout}"
        return result

    # Verify required fields
    required_fields = ["schema_version", "recording_b3sum", "transcript_path", "assigned_at", "method", "mappings"]
    for field in required_fields:
        if field not in data:
            result.error = f"Missing required field: {field}"
            return result

    # Verify mappings structure
    mappings = data.get("mappings", {})
    for label, info in mappings.items():
        if "confidence" not in info:
            result.error = f"Missing 'confidence' in mapping for {label}"
            return result
        if "score" not in info:
            result.error = f"Missing 'score' in mapping for {label}"
            return result

    result.passed = True
    return result


# =============================================================================
# Threshold Tests
# =============================================================================

def test_assign_threshold(temp_dir: Path) -> TestResult:
    """Test different threshold values affect assignments."""
    result = TestResult("assign_threshold")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # Test with high threshold
    rc, stdout1, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--threshold", "0.9",
        "--format", "json",
        "--dry-run",
    ], env)

    if rc != 0:
        result.error = f"assign with high threshold failed: {stderr}"
        return result

    # Test with low threshold
    rc, stdout2, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--threshold", "0.1",
        "--format", "json",
        "--dry-run",
    ], env)

    if rc != 0:
        result.error = f"assign with low threshold failed: {stderr}"
        return result

    # Parse outputs
    try:
        json_start1 = stdout1.find("{")
        data1 = json.loads(stdout1[json_start1:]) if json_start1 >= 0 else json.loads(stdout1)
        json_start2 = stdout2.find("{")
        data2 = json.loads(stdout2[json_start2:]) if json_start2 >= 0 else json.loads(stdout2)
    except json.JSONDecodeError as e:
        result.error = f"Failed to parse JSON output: {e}"
        return result

    # Verify threshold is recorded
    if data1.get("threshold") != 0.9:
        result.error = f"Threshold 0.9 not recorded: {data1.get('threshold')}"
        return result

    if data2.get("threshold") != 0.1:
        result.error = f"Threshold 0.1 not recorded: {data2.get('threshold')}"
        return result

    result.passed = True
    return result


def test_assign_min_trust(temp_dir: Path) -> TestResult:
    """Test min-trust parameter is recorded."""
    result = TestResult("assign_min_trust")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--min-trust", "high",
        "--format", "json",
    ], env)

    if rc != 0:
        result.error = f"assign with --min-trust failed: {stderr}"
        return result

    try:
        data = parse_json_output(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON output: {stdout}"
        return result

    if data.get("min_trust") != "high":
        result.error = f"min_trust not recorded: {data.get('min_trust')}"
        return result

    result.passed = True
    return result


# =============================================================================
# Signal Combination Tests
# =============================================================================

def test_signal_combination(temp_dir: Path) -> TestResult:
    """Test that signal weighting works with expected speakers."""
    result = TestResult("signal_combination")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # Create catalog entry with expected speakers
    setup_catalog_entry(
        temp_dir, audio_path,
        context_name="test-meeting",
        expected_speakers=["alice", "bob"]
    )

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--format", "json",
    ], env)

    if rc != 0:
        result.error = f"assign with context signals failed: {stderr}"
        return result

    try:
        data = parse_json_output(stdout)
    except json.JSONDecodeError:
        result.error = f"Invalid JSON output: {stdout}"
        return result

    # Verify context is used
    if data.get("context") != "test-meeting":
        result.error = f"Context not used: {data.get('context')}"
        return result

    # Verify mappings have signal info
    mappings = data.get("mappings", {})
    for label, info in mappings.items():
        # With expected speakers, we should have candidates or signals
        # Even if unassigned, structure should be there
        if "score" not in info:
            result.error = f"Missing score for speaker {label}"
            return result

    result.passed = True
    return result


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_assign_missing_audio(temp_dir: Path) -> TestResult:
    """Test error when audio file doesn't exist."""
    result = TestResult("assign_missing_audio")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    transcript_path = create_mock_transcript(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", "/nonexistent/audio.wav",
        "--transcript", str(transcript_path),
    ], env)

    if rc == 0:
        result.error = "assign should fail for missing audio"
        return result

    if "not found" not in stderr.lower():
        result.error = f"Expected 'not found' error: {stderr}"
        return result

    result.passed = True
    return result


def test_assign_missing_transcript(temp_dir: Path) -> TestResult:
    """Test error when transcript file doesn't exist."""
    result = TestResult("assign_missing_transcript")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", "/nonexistent/transcript.json",
    ], env)

    if rc == 0:
        result.error = "assign should fail for missing transcript"
        return result

    if "not found" not in stderr.lower():
        result.error = f"Expected 'not found' error: {stderr}"
        return result

    result.passed = True
    return result


def test_assign_empty_transcript(temp_dir: Path) -> TestResult:
    """Test error when transcript has no speakers."""
    result = TestResult("assign_empty_transcript")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)

    # Create empty transcript
    transcript_path = temp_dir / "empty_transcript.json"
    with open(transcript_path, "w") as f:
        json.dump({"utterances": []}, f)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    if rc == 0:
        result.error = "assign should fail for empty transcript"
        return result

    if "No speakers found" not in stderr:
        result.error = f"Expected 'No speakers found' error: {stderr}"
        return result

    result.passed = True
    return result


# =============================================================================
# Verbose and Quiet Mode Tests
# =============================================================================

def test_assign_verbose(temp_dir: Path) -> TestResult:
    """Test verbose output includes extra information."""
    result = TestResult("assign_verbose")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # -v goes before the subcommand
    rc, stdout, stderr = run_cmd([
        "-v",
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--dry-run",
    ], env)

    if rc != 0:
        result.error = f"assign -v failed: {stderr}"
        return result

    if "Processing speaker" not in stdout:
        result.error = f"Verbose mode should show 'Processing speaker': {stdout}"
        return result

    result.passed = True
    return result


def test_assign_quiet(temp_dir: Path) -> TestResult:
    """Test quiet mode suppresses non-essential output."""
    result = TestResult("assign_quiet")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript(temp_dir)

    # -q goes before the subcommand
    rc, stdout, stderr = run_cmd([
        "-q",
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    if rc != 0:
        result.error = f"assign -q failed: {stderr}"
        return result

    # Quiet mode should have minimal output
    if "Found 2 speakers" in stdout:
        result.error = f"Quiet mode should suppress speaker count: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Speechmatics Format Test
# =============================================================================

def test_assign_speechmatics_format(temp_dir: Path) -> TestResult:
    """Test assignment with Speechmatics transcript format."""
    result = TestResult("assign_speechmatics_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    audio_path = create_test_audio(temp_dir)
    transcript_path = create_mock_transcript_speechmatics(temp_dir)

    rc, stdout, stderr = run_cmd([
        "assign", str(audio_path),
        "--transcript", str(transcript_path),
    ], env)

    if rc != 0:
        result.error = f"assign with Speechmatics format failed: {stderr}"
        return result

    if "Found 2 speakers" not in stdout:
        result.error = f"Expected 'Found 2 speakers' in output: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="speaker-assign CLI unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check for pyyaml
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required for tests. Install with: pip install pyyaml")
        return 2

    tests = [
        # Basic assignment tests
        test_assign_basic,
        test_assign_three_speakers,
        test_assign_with_expected_speakers,
        test_assign_with_cli_expected_speakers,

        # Dry run tests
        test_assign_dry_run,
        test_assign_dry_run_json,

        # Save and persistence tests
        test_assign_saves_to_assignments_dir,
        test_assign_output_file,

        # Show command tests
        test_show_assignments,
        test_show_assignments_json,
        test_show_nonexistent,
        test_show_by_b3sum_prefix,

        # Clear command tests
        test_clear_assignments,
        test_clear_nonexistent,

        # JSON output tests
        test_assign_json_output,

        # Threshold tests
        test_assign_threshold,
        test_assign_min_trust,

        # Signal combination tests
        test_signal_combination,

        # Error handling tests
        test_assign_missing_audio,
        test_assign_missing_transcript,
        test_assign_empty_transcript,

        # Verbose and quiet mode tests
        test_assign_verbose,
        test_assign_quiet,

        # Format tests
        test_assign_speechmatics_format,
    ]

    print("speaker-assign CLI Unit Tests")
    print("=" * 40)

    passed = 0
    failed = 0
    skipped = 0
    results = []

    for test_func in tests:
        # Create fresh temp directory for each test
        temp_dir = Path(tempfile.mkdtemp(prefix="assign_test_"))

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
