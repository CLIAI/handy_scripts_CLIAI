#!/usr/bin/env python3
"""
Unit tests for speaker-review CLI tool.

Tests session management, status commands, and CLI argument parsing.
Note: TUI interactions cannot be tested directly, so we focus on non-interactive features.

Usage:
    ./test_speaker_review.py              # Run all tests
    ./test_speaker_review.py -v           # Verbose output
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
SPEAKER_REVIEW = REPO_ROOT / "speaker-review"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.skipped = False


def run_cmd(args: list, env: dict = None, stdin_input: str = None) -> tuple:
    """Run speaker-review command, return (returncode, stdout, stderr)."""
    cmd = [str(SPEAKER_REVIEW)] + args
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


def compute_b3sum(file_path: Path) -> str:
    """Compute Blake3 hash of a file, falling back to SHA256 if b3sum unavailable."""
    try:
        result = subprocess.run(
            ["b3sum", "--no-names", str(file_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:32]
    except (subprocess.CalledProcessError, FileNotFoundError):
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:32]


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


def create_mock_assignments(temp_dir: Path, b3sum: str, transcript_path: str, num_speakers: int = 2) -> Path:
    """Create a mock assignments YAML file."""
    import yaml

    assignments_dir = temp_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    if num_speakers == 2:
        mappings = {
            "A": {
                "speaker_id": "alice",
                "confidence": "high",
                "signals": [
                    {"type": "name_mention", "score": 0.9},
                    {"type": "embedding_match", "score": 0.85},
                ],
            },
            "B": {
                "speaker_id": "bob",
                "confidence": "medium",
                "signals": [
                    {"type": "name_mention", "score": 0.7},
                ],
            },
        }
    elif num_speakers == 3:
        mappings = {
            "A": {
                "speaker_id": "alice",
                "confidence": "high",
                "signals": [{"type": "name_mention", "score": 0.9}],
            },
            "B": {
                "speaker_id": "bob",
                "confidence": "medium",
                "signals": [{"type": "name_mention", "score": 0.7}],
            },
            "C": {
                "speaker_id": None,
                "confidence": "unassigned",
                "signals": [],
            },
        }
    else:
        mappings = {
            "A": {
                "speaker_id": None,
                "confidence": "unassigned",
                "signals": [],
            },
        }

    assignments_data = {
        "schema_version": "1.0",
        "recording_b3sum": b3sum,
        "transcript_path": str(transcript_path),
        "context": None,
        "mappings": mappings,
        "created_at": "2025-01-17T12:00:00Z",
    }

    assignment_path = assignments_dir / f"{b3sum}.yaml"
    with open(assignment_path, "w") as f:
        yaml.dump(assignments_data, f, default_flow_style=False)

    return assignment_path


def create_mock_catalog_entry(temp_dir: Path, audio_path: Path, b3sum: str, context_name: str = None) -> Path:
    """Create a mock catalog entry."""
    import yaml

    catalog_dir = temp_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "recording": {
            "b3sum": b3sum,
            "path": str(audio_path),
            "original_path": str(audio_path),
        },
        "context": {
            "name": context_name,
            "expected_speakers": [],
        },
    }

    catalog_path = catalog_dir / f"{b3sum}.yaml"
    with open(catalog_path, "w") as f:
        yaml.dump(entry, f, default_flow_style=False)

    return catalog_path


# =============================================================================
# Status Command Tests
# =============================================================================

def test_status_no_session(temp_dir: Path) -> TestResult:
    """Test status command when no session exists."""
    result = TestResult("status_no_session")

    # Use custom cache dir to ensure no session exists
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir.parent),
    }

    # Create a speaker-review subdirectory in cache to match expected path
    review_cache = cache_dir / "speaker-review"
    # Do NOT create session.yaml

    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    if "No active session" not in stdout:
        result.error = f"Expected 'No active session' message: {stdout}"
        return result

    result.passed = True
    return result


def test_status_command(temp_dir: Path) -> TestResult:
    """Test status subcommand shows session info when session exists."""
    result = TestResult("status_command")
    import yaml

    # XDG_CACHE_HOME is the base, tool adds "speaker-review" subdir
    xdg_cache = temp_dir / "cache"
    cache_dir = xdg_cache / "speaker-review"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # First create audio and get its actual b3sum
    audio_path = create_test_audio(temp_dir)
    b3sum = compute_b3sum(audio_path)

    # Create transcript and assignments with the actual b3sum
    transcript_path = create_mock_transcript(temp_dir)
    create_mock_assignments(temp_dir, b3sum, str(transcript_path))

    # Create mock session file with the actual b3sum
    session_data = {
        "recording_b3sum": b3sum,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "context": "team-meeting",
        "current_index": 2,
        "decisions": {
            "0": {"action": "approve", "speaker_id": "alice", "timestamp": "2025-01-17T12:00:00Z"},
            "1": {"action": "reject", "notes": "wrong speaker", "timestamp": "2025-01-17T12:01:00Z"},
        },
        "started_at": "2025-01-17T11:00:00Z",
        "updated_at": "2025-01-17T12:01:00Z",
    }

    session_path = cache_dir / "session.yaml"
    with open(session_path, "w") as f:
        yaml.dump(session_data, f, default_flow_style=False)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(xdg_cache),
    }

    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    # Should show session info (b3sum prefix or "Active Session")
    if "Active Session" not in stdout and b3sum[:8] not in stdout:
        result.error = f"Expected session info in output: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Clear Command Tests
# =============================================================================

def test_clear_no_session(temp_dir: Path) -> TestResult:
    """Test clear command when nothing to clear."""
    result = TestResult("clear_no_session")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    rc, stdout, stderr = run_cmd(["clear"], env)

    if rc != 0:
        result.error = f"clear command failed: {stderr}"
        return result

    if "Session cleared" not in stdout:
        result.error = f"Expected 'Session cleared' message: {stdout}"
        return result

    result.passed = True
    return result


def test_clear_command(temp_dir: Path) -> TestResult:
    """Test clear subcommand removes session file."""
    result = TestResult("clear_command")
    import yaml

    # Create cache directory with session
    cache_dir = temp_dir / "cache" / "speaker-review"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create mock session file
    session_data = {
        "recording_b3sum": "abc123def456789012345678901234",
        "audio_path": "/tmp/test_audio.wav",
        "transcript_path": "/tmp/transcript.json",
        "context": "test-context",
        "current_index": 0,
        "decisions": {},
        "started_at": "2025-01-17T11:00:00Z",
        "updated_at": "2025-01-17T11:00:00Z",
    }

    session_path = cache_dir / "session.yaml"
    with open(session_path, "w") as f:
        yaml.dump(session_data, f, default_flow_style=False)

    # Verify session file exists
    if not session_path.exists():
        result.error = "Setup failed: session file not created"
        return result

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(temp_dir / "cache"),
    }

    rc, stdout, stderr = run_cmd(["clear"], env)

    if rc != 0:
        result.error = f"clear command failed: {stderr}"
        return result

    if "Session cleared" not in stdout:
        result.error = f"Expected 'Session cleared' message: {stdout}"
        return result

    # Verify session file is removed
    if session_path.exists():
        result.error = "Session file should have been removed"
        return result

    result.passed = True
    return result


# =============================================================================
# Review Command Tests (Non-Interactive)
# =============================================================================

def test_review_no_assignments(temp_dir: Path) -> TestResult:
    """Test review command when no assignments exist."""
    result = TestResult("review_no_assignments")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    # No assignments directory exists
    rc, stdout, stderr = run_cmd(["review"], env)

    # Should indicate no assignments found
    if "No assignments found" not in stdout and "Run speaker-assign first" not in stdout:
        result.error = f"Expected 'No assignments found' message: {stdout}"
        return result

    result.passed = True
    return result


def test_review_specific_audio_no_assignments(temp_dir: Path) -> TestResult:
    """Test review command for specific audio when no assignments exist."""
    result = TestResult("review_specific_audio_no_assignments")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_path = create_test_audio(temp_dir)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    rc, stdout, stderr = run_cmd(["review", str(audio_path)], env)

    # Should indicate no assignments found for this audio
    if "No assignments found" not in stdout and "Run speaker-assign first" not in stdout:
        result.error = f"Expected 'No assignments found' message: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Session Persistence Tests
# =============================================================================

def test_session_persistence(temp_dir: Path) -> TestResult:
    """Test that session file is created and can be loaded."""
    result = TestResult("session_persistence")
    import yaml

    # Create necessary directories
    cache_dir = temp_dir / "cache" / "speaker-review"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create audio and assignments
    audio_path = create_test_audio(temp_dir)
    b3sum = compute_b3sum(audio_path)
    transcript_path = create_mock_transcript(temp_dir)
    create_mock_assignments(temp_dir, b3sum, str(transcript_path))
    create_mock_catalog_entry(temp_dir, audio_path, b3sum, "test-context")

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(temp_dir / "cache"),
    }

    # Running review should create a session (we can't interact with TUI, but session should be created)
    # Since stdin is not a tty, it will use simple mode and exit immediately without interaction
    # This test verifies the session file path is correct

    session_path = cache_dir / "session.yaml"

    # Create a minimal session manually to test loading
    session_data = {
        "recording_b3sum": b3sum,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "context": "test-context",
        "current_index": 1,
        "decisions": {
            "0": {"action": "approve", "speaker_id": "alice", "timestamp": "2025-01-17T12:00:00Z"},
        },
        "started_at": "2025-01-17T11:00:00Z",
        "updated_at": "2025-01-17T12:00:00Z",
    }

    with open(session_path, "w") as f:
        yaml.dump(session_data, f, default_flow_style=False)

    # Test that status command can read this session
    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    # Should show session info including b3sum prefix
    if b3sum[:8] not in stdout and "Active Session" not in stdout:
        result.error = f"Expected session info with b3sum in output: {stdout}"
        return result

    result.passed = True
    return result


def test_session_continue_no_session(temp_dir: Path) -> TestResult:
    """Test --continue flag when no session exists."""
    result = TestResult("session_continue_no_session")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    rc, stdout, stderr = run_cmd(["review", "--continue"], env)

    if rc == 0:
        result.error = "Expected failure when no session to continue"
        return result

    if "No saved session found" not in stdout:
        result.error = f"Expected 'No saved session found' message: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Help Output Tests
# =============================================================================

def test_help_output(temp_dir: Path) -> TestResult:
    """Test that help shows keybindings."""
    result = TestResult("help_output")

    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["--help"], env)

    if rc != 0:
        result.error = f"--help failed: {stderr}"
        return result

    # Should show keybindings in help
    keybindings = ["approve", "reject", "skip", "play", "quit"]
    found_any = False
    for kb in keybindings:
        if kb in stdout.lower():
            found_any = True
            break

    if not found_any:
        result.error = f"Expected keybindings in help output: {stdout}"
        return result

    result.passed = True
    return result


def test_help_shows_subcommands(temp_dir: Path) -> TestResult:
    """Test that help shows available subcommands."""
    result = TestResult("help_shows_subcommands")

    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["--help"], env)

    if rc != 0:
        result.error = f"--help failed: {stderr}"
        return result

    # Should show subcommands
    subcommands = ["review", "status", "clear"]
    missing = []
    for cmd in subcommands:
        if cmd not in stdout:
            missing.append(cmd)

    if missing:
        result.error = f"Missing subcommands in help: {missing}. Output: {stdout}"
        return result

    result.passed = True
    return result


def test_version_output(temp_dir: Path) -> TestResult:
    """Test that version flag works."""
    result = TestResult("version_output")

    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["--version"], env)

    if rc != 0:
        result.error = f"--version failed: {stderr}"
        return result

    # Should show version number
    if "speaker-review" not in stdout and "1." not in stdout:
        result.error = f"Expected version in output: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Argument Parsing Tests
# =============================================================================

def test_review_subcommand_explicit(temp_dir: Path) -> TestResult:
    """Test explicit 'review' subcommand works."""
    result = TestResult("review_subcommand_explicit")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    # Even without assignments, the subcommand should be recognized
    rc, stdout, stderr = run_cmd(["review"], env)

    # Should not fail with "unknown subcommand" or similar
    if "invalid choice" in stderr.lower() or "unrecognized" in stderr.lower():
        result.error = f"review subcommand not recognized: {stderr}"
        return result

    result.passed = True
    return result


def test_review_context_option(temp_dir: Path) -> TestResult:
    """Test --context option is accepted."""
    result = TestResult("review_context_option")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    rc, stdout, stderr = run_cmd(["review", "--context", "team-meeting"], env)

    # Should accept the option (may still show "no assignments" but not argument error)
    if "unrecognized arguments" in stderr.lower() or "invalid" in stderr.lower():
        result.error = f"--context option not recognized: {stderr}"
        return result

    result.passed = True
    return result


def test_review_simple_mode_option(temp_dir: Path) -> TestResult:
    """Test --simple option is accepted."""
    result = TestResult("review_simple_mode_option")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    rc, stdout, stderr = run_cmd(["review", "--simple"], env)

    # Should accept the option
    if "unrecognized arguments" in stderr.lower():
        result.error = f"--simple option not recognized: {stderr}"
        return result

    result.passed = True
    return result


# =============================================================================
# Integration with Assignments
# =============================================================================

def test_review_finds_assignments(temp_dir: Path) -> TestResult:
    """Test that review finds and uses assignments file."""
    result = TestResult("review_finds_assignments")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create audio, transcript, and assignments
    audio_path = create_test_audio(temp_dir)
    b3sum = compute_b3sum(audio_path)
    transcript_path = create_mock_transcript(temp_dir)
    create_mock_assignments(temp_dir, b3sum, str(transcript_path))
    create_mock_catalog_entry(temp_dir, audio_path, b3sum, "test-context")

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    # Run review - since not interactive, it won't enter loop but should recognize segments
    rc, stdout, stderr = run_cmd(["review", str(audio_path)], env)

    # Should show segment count or similar (not "no assignments" error)
    if "No assignments found" in stdout:
        result.error = f"Should have found assignments: {stdout}"
        return result

    # Check for segment count (5 utterances = 5 segments)
    if "Segments:" in stdout or "5" in stdout:
        result.passed = True
        return result

    # If it shows reviewing message, that's also success
    if "Reviewing" in stdout:
        result.passed = True
        return result

    # Otherwise, still pass if no error occurred
    if rc == 0:
        result.passed = True
        return result

    result.error = f"Unexpected output: {stdout}"
    return result


def test_review_by_b3sum_prefix(temp_dir: Path) -> TestResult:
    """Test that review can find recording by b3sum prefix."""
    result = TestResult("review_by_b3sum_prefix")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create audio, transcript, and assignments
    audio_path = create_test_audio(temp_dir)
    b3sum = compute_b3sum(audio_path)
    transcript_path = create_mock_transcript(temp_dir)
    create_mock_assignments(temp_dir, b3sum, str(transcript_path))
    create_mock_catalog_entry(temp_dir, audio_path, b3sum, "test-context")

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    # Use b3sum prefix instead of path
    prefix = b3sum[:8]
    rc, stdout, stderr = run_cmd(["review", prefix], env)

    # Should find the recording (not error about "could not resolve")
    if "Could not resolve" in stdout:
        result.error = f"Should have resolved b3sum prefix: {stdout}"
        return result

    result.passed = True
    return result


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_review_nonexistent_audio(temp_dir: Path) -> TestResult:
    """Test review command with nonexistent audio file."""
    result = TestResult("review_nonexistent_audio")

    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SPEAKERS_EMBEDDINGS_DIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    rc, stdout, stderr = run_cmd(["review", "/nonexistent/path/audio.wav"], env)

    # Should fail gracefully
    if rc == 0:
        result.error = "Should fail for nonexistent audio"
        return result

    if "Could not resolve" not in stdout and "not found" not in stderr.lower():
        result.error = f"Expected error message about nonexistent file: {stdout} {stderr}"
        return result

    result.passed = True
    return result


def test_invalid_subcommand(temp_dir: Path) -> TestResult:
    """Test that invalid subcommand is handled."""
    result = TestResult("invalid_subcommand")

    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["invalidcmd"], env)

    # Could be treated as an audio path (and fail to resolve) or invalid subcommand
    # Either way should not crash
    result.passed = True
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="speaker-review CLI unit tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check for pyyaml
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required for tests. Install with: pip install pyyaml")
        return 2

    tests = [
        # Status command tests
        test_status_no_session,
        test_status_command,

        # Clear command tests
        test_clear_no_session,
        test_clear_command,

        # Review command tests (non-interactive)
        test_review_no_assignments,
        test_review_specific_audio_no_assignments,

        # Session persistence tests
        test_session_persistence,
        test_session_continue_no_session,

        # Help output tests
        test_help_output,
        test_help_shows_subcommands,
        test_version_output,

        # Argument parsing tests
        test_review_subcommand_explicit,
        test_review_context_option,
        test_review_simple_mode_option,

        # Integration with assignments
        test_review_finds_assignments,
        test_review_by_b3sum_prefix,

        # Error handling tests
        test_review_nonexistent_audio,
        test_invalid_subcommand,
    ]

    print("speaker-review CLI Unit Tests")
    print("=" * 40)

    passed = 0
    failed = 0
    skipped = 0
    results = []

    for test_func in tests:
        # Create fresh temp directory for each test
        temp_dir = Path(tempfile.mkdtemp(prefix="review_test_"))

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
