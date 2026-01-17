#!/usr/bin/env python3
"""
Unit tests for speaker-report CLI tool.

Tests all CLI commands for quality metrics and recommendations.

Usage:
    ./test_speaker_report.py              # Run all tests
    ./test_speaker_report.py -v           # Verbose output
"""

import json
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
SPEAKER_REPORT = REPO_ROOT / "speaker-report"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.skipped = False


def run_cmd(args: list, env: dict = None) -> tuple:
    """Run speaker-report command, return (returncode, stdout, stderr)."""
    cmd = [str(SPEAKER_REPORT)] + args
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


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def past_iso(days: int) -> str:
    """Return ISO datetime string for N days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def create_catalog_entry(
    temp_dir: Path,
    b3sum: str,
    context_name: str = None,
    transcriptions: list = None,
    review_status: str = "none",
    updated_at: str = None,
) -> Path:
    """Create a catalog entry YAML file."""
    import yaml

    catalog_dir = temp_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "schema_version": 1,
        "recording": {
            "path": f"/test/audio/{b3sum}.wav",
            "b3sum": b3sum,
            "duration_sec": 60.0,
            "discovered_at": "2026-01-01T00:00:00Z",
        },
        "context": {
            "name": context_name,
            "expected_speakers": [],
            "tags": [],
        },
        "transcriptions": transcriptions or [],
        "review": {
            "status": review_status,
        },
        "updated_at": updated_at or utc_now_iso(),
    }

    entry_path = catalog_dir / f"{b3sum}.yaml"
    with open(entry_path, "w") as f:
        yaml.dump(entry, f)

    return entry_path


def create_speaker_profile(
    temp_dir: Path,
    speaker_id: str,
    trust_level: str = "unverified",
    display_name: str = None,
    sample_count: int = 0,
    reviewed_samples: int = 0,
) -> Path:
    """Create a speaker profile YAML file."""
    import yaml

    db_dir = temp_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "speaker_id": speaker_id,
        "display_name": display_name,
        "trust_level": trust_level,
        "enrollment_count": sample_count,
        "samples": [{"reviewed": True}] * reviewed_samples,
        "updated_at": utc_now_iso(),
    }

    profile_path = db_dir / f"{speaker_id}.yaml"
    with open(profile_path, "w") as f:
        yaml.dump(profile, f)

    # Create samples directory with dummy files
    if sample_count > 0:
        samples_dir = temp_dir / "samples" / speaker_id
        samples_dir.mkdir(parents=True, exist_ok=True)
        for i in range(sample_count):
            (samples_dir / f"sample_{i}.wav").touch()

    return profile_path


def create_assignment(
    temp_dir: Path,
    b3sum: str,
    mappings: dict,
) -> Path:
    """Create an assignment YAML file."""
    import yaml

    assignments_dir = temp_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    assignment = {
        "recording_b3sum": b3sum,
        "mappings": mappings,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }

    assign_path = assignments_dir / f"{b3sum}.yaml"
    with open(assign_path, "w") as f:
        yaml.dump(assignment, f)

    return assign_path


# =============================================================================
# Status Command Tests
# =============================================================================

def test_status_empty(temp_dir: Path) -> TestResult:
    """Test status command with empty catalog."""
    result = TestResult("status_empty")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create empty directories
    (temp_dir / "catalog").mkdir(parents=True, exist_ok=True)
    (temp_dir / "db").mkdir(parents=True, exist_ok=True)
    (temp_dir / "assignments").mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    if "Speaker Detection System Status" not in stdout:
        result.error = f"Missing header in output: {stdout}"
        return result

    if "Recordings:     0 total" not in stdout:
        result.error = f"Expected 0 recordings: {stdout}"
        return result

    result.passed = True
    return result


def test_status_with_data(temp_dir: Path) -> TestResult:
    """Test status command with catalog data."""
    result = TestResult("status_with_data")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create test data
    create_catalog_entry(temp_dir, "abc123", context_name="team-standup")
    create_catalog_entry(
        temp_dir, "def456", context_name="podcast",
        transcriptions=[{"backend": "speechmatics"}]
    )
    create_catalog_entry(
        temp_dir, "ghi789", context_name="team-standup",
        transcriptions=[{"backend": "assemblyai"}],
        review_status="complete"
    )

    create_speaker_profile(temp_dir, "alice", trust_level="high", sample_count=5, reviewed_samples=3)
    create_speaker_profile(temp_dir, "bob", trust_level="medium", sample_count=3, reviewed_samples=2)

    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    if "Recordings:     3 total" not in stdout:
        result.error = f"Expected 3 recordings: {stdout}"
        return result

    if "Speakers:       2 enrolled" not in stdout:
        result.error = f"Expected 2 speakers: {stdout}"
        return result

    if "team-standup" not in stdout:
        result.error = f"Missing context in output: {stdout}"
        return result

    result.passed = True
    return result


def test_status_json_format(temp_dir: Path) -> TestResult:
    """Test status command with JSON output."""
    result = TestResult("status_json_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create minimal data
    create_catalog_entry(temp_dir, "abc123")
    create_speaker_profile(temp_dir, "alice", trust_level="high")

    rc, stdout, stderr = run_cmd(["--format", "json", "status"], env)

    if rc != 0:
        result.error = f"status --format json failed: {stderr}"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON output: {e}"
        return result

    # Check structure
    if "recordings" not in data:
        result.error = "Missing 'recordings' in JSON output"
        return result

    if "speakers" not in data:
        result.error = "Missing 'speakers' in JSON output"
        return result

    if "recommendations" not in data:
        result.error = "Missing 'recommendations' in JSON output"
        return result

    result.passed = True
    return result


def test_status_default_command(temp_dir: Path) -> TestResult:
    """Test that status is the default command when no subcommand given."""
    result = TestResult("status_default_command")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create empty directories
    (temp_dir / "catalog").mkdir(parents=True, exist_ok=True)
    (temp_dir / "db").mkdir(parents=True, exist_ok=True)
    (temp_dir / "assignments").mkdir(parents=True, exist_ok=True)

    # Run without subcommand
    rc, stdout, stderr = run_cmd([], env)

    if rc != 0:
        result.error = f"default command failed: {stderr}"
        return result

    if "Speaker Detection System Status" not in stdout:
        result.error = "Default command should be status"
        return result

    result.passed = True
    return result


def test_status_recommendations(temp_dir: Path) -> TestResult:
    """Test that status generates appropriate recommendations."""
    result = TestResult("status_recommendations")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create recordings with issues
    # 1. Pending recording (no transcript)
    create_catalog_entry(temp_dir, "pending1", context_name="test")

    # 2. Low confidence assignment
    create_catalog_entry(
        temp_dir, "lowconf1", context_name="test",
        transcriptions=[{"backend": "test"}]
    )
    create_assignment(temp_dir, "lowconf1", {
        "S1": {"speaker_id": "alice", "confidence": "low"}
    })

    # 3. Speaker needing samples
    create_speaker_profile(temp_dir, "newbie", trust_level="low", sample_count=1, reviewed_samples=1)

    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"status command failed: {stderr}"
        return result

    if "Recommendations:" not in stdout:
        result.error = "Missing Recommendations section"
        return result

    # Should have recommendation about pending
    if "pending" not in stdout.lower():
        result.error = "Missing recommendation about pending recordings"
        return result

    result.passed = True
    return result


# =============================================================================
# Coverage Command Tests
# =============================================================================

def test_coverage_empty(temp_dir: Path) -> TestResult:
    """Test coverage command with empty catalog."""
    result = TestResult("coverage_empty")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    (temp_dir / "catalog").mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = run_cmd(["coverage"], env)

    if rc != 0:
        result.error = f"coverage command failed: {stderr}"
        return result

    if "Coverage by Context" not in stdout:
        result.error = f"Missing header: {stdout}"
        return result

    result.passed = True
    return result


def test_coverage_by_context(temp_dir: Path) -> TestResult:
    """Test coverage command shows context breakdown."""
    result = TestResult("coverage_by_context")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create entries in different contexts
    create_catalog_entry(temp_dir, "abc1", context_name="ctx-a")
    create_catalog_entry(temp_dir, "abc2", context_name="ctx-a",
                        transcriptions=[{"backend": "test"}])
    create_catalog_entry(temp_dir, "abc3", context_name="ctx-b")
    create_catalog_entry(temp_dir, "abc4", context_name="ctx-b",
                        transcriptions=[{"backend": "test"}],
                        review_status="complete")

    rc, stdout, stderr = run_cmd(["coverage"], env)

    if rc != 0:
        result.error = f"coverage command failed: {stderr}"
        return result

    if "ctx-a" not in stdout or "ctx-b" not in stdout:
        result.error = f"Missing context names: {stdout}"
        return result

    if "Unprocessed:" not in stdout:
        result.error = "Missing status breakdown"
        return result

    result.passed = True
    return result


def test_coverage_filter_context(temp_dir: Path) -> TestResult:
    """Test coverage command with --context filter."""
    result = TestResult("coverage_filter_context")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_catalog_entry(temp_dir, "abc1", context_name="ctx-a")
    create_catalog_entry(temp_dir, "abc2", context_name="ctx-b")

    rc, stdout, stderr = run_cmd(["coverage", "--context", "ctx-a"], env)

    if rc != 0:
        result.error = f"coverage --context failed: {stderr}"
        return result

    if "ctx-a" not in stdout:
        result.error = "Filtered context not shown"
        return result

    if "ctx-b" in stdout:
        result.error = "Unfiltered context should not be shown"
        return result

    result.passed = True
    return result


def test_coverage_json_format(temp_dir: Path) -> TestResult:
    """Test coverage command with JSON output."""
    result = TestResult("coverage_json_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_catalog_entry(temp_dir, "abc1", context_name="test-ctx")

    rc, stdout, stderr = run_cmd(["--format", "json", "coverage"], env)

    if rc != 0:
        result.error = f"coverage --format json failed: {stderr}"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
        return result

    if "test-ctx" not in data:
        result.error = "Context not in JSON output"
        return result

    result.passed = True
    return result


# =============================================================================
# Confidence Command Tests
# =============================================================================

def test_confidence_empty(temp_dir: Path) -> TestResult:
    """Test confidence command with no assignments."""
    result = TestResult("confidence_empty")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    (temp_dir / "catalog").mkdir(parents=True, exist_ok=True)
    (temp_dir / "assignments").mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = run_cmd(["confidence"], env)

    if rc != 0:
        result.error = f"confidence command failed: {stderr}"
        return result

    if "No recordings below threshold" not in stdout:
        result.error = f"Expected empty message: {stdout}"
        return result

    result.passed = True
    return result


def test_confidence_finds_low(temp_dir: Path) -> TestResult:
    """Test confidence command finds low-confidence assignments."""
    result = TestResult("confidence_finds_low")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create recording with low confidence assignment
    create_catalog_entry(
        temp_dir, "low1", context_name="test",
        transcriptions=[{"backend": "test"}]
    )
    create_assignment(temp_dir, "low1", {
        "S1": {"speaker_id": "alice", "confidence": "high"},
        "S2": {"speaker_id": "bob", "confidence": "low"},
    })

    # Create recording with high confidence (should not appear)
    create_catalog_entry(
        temp_dir, "high1",
        transcriptions=[{"backend": "test"}]
    )
    create_assignment(temp_dir, "high1", {
        "S1": {"speaker_id": "alice", "confidence": "high"},
    })

    rc, stdout, stderr = run_cmd(["confidence"], env)

    if rc != 0:
        result.error = f"confidence command failed: {stderr}"
        return result

    if "Found 1 recording" not in stdout:
        result.error = f"Should find 1 low-confidence recording: {stdout}"
        return result

    if "bob" not in stdout or "low" not in stdout.lower():
        result.error = "Low confidence assignment details missing"
        return result

    result.passed = True
    return result


def test_confidence_threshold(temp_dir: Path) -> TestResult:
    """Test confidence command with custom threshold."""
    result = TestResult("confidence_threshold")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_catalog_entry(
        temp_dir, "med1",
        transcriptions=[{"backend": "test"}]
    )
    create_assignment(temp_dir, "med1", {
        "S1": {"speaker_id": "alice", "confidence": "medium"},  # 70%
    })

    # Default threshold (70) should not catch medium
    rc, stdout, _ = run_cmd(["confidence", "--below", "70"], env)
    if "No recordings" not in stdout:
        result.error = "Medium confidence should not be below 70"
        return result

    # Higher threshold (80) should catch medium
    rc, stdout, _ = run_cmd(["confidence", "--below", "80"], env)
    if "Found 1 recording" not in stdout:
        result.error = "Medium confidence should be below 80"
        return result

    result.passed = True
    return result


def test_confidence_json_format(temp_dir: Path) -> TestResult:
    """Test confidence command with JSON output."""
    result = TestResult("confidence_json_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_catalog_entry(
        temp_dir, "test1",
        transcriptions=[{"backend": "test"}]
    )
    create_assignment(temp_dir, "test1", {
        "S1": {"speaker_id": "alice", "confidence": "low"},
    })

    rc, stdout, stderr = run_cmd(["--format", "json", "confidence"], env)

    if rc != 0:
        result.error = f"confidence --format json failed: {stderr}"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
        return result

    if "threshold" not in data or "count" not in data:
        result.error = "Missing expected fields in JSON"
        return result

    if data["count"] != 1:
        result.error = f"Expected count=1, got {data['count']}"
        return result

    result.passed = True
    return result


# =============================================================================
# Stale Command Tests
# =============================================================================

def test_stale_empty(temp_dir: Path) -> TestResult:
    """Test stale command with no stale recordings."""
    result = TestResult("stale_empty")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create recent recording
    create_catalog_entry(temp_dir, "recent1", updated_at=utc_now_iso())

    rc, stdout, stderr = run_cmd(["stale"], env)

    if rc != 0:
        result.error = f"stale command failed: {stderr}"
        return result

    if "No stale recordings" not in stdout:
        result.error = f"Should find no stale recordings: {stdout}"
        return result

    result.passed = True
    return result


def test_stale_finds_old(temp_dir: Path) -> TestResult:
    """Test stale command finds old recordings."""
    result = TestResult("stale_finds_old")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create old recording (45 days ago)
    create_catalog_entry(
        temp_dir, "old1", context_name="test",
        transcriptions=[{"backend": "test"}],
        updated_at=past_iso(45)
    )

    # Create recent recording
    create_catalog_entry(temp_dir, "recent1", updated_at=utc_now_iso())

    rc, stdout, stderr = run_cmd(["stale"], env)

    if rc != 0:
        result.error = f"stale command failed: {stderr}"
        return result

    if "Found 1 recording" not in stdout:
        result.error = f"Should find 1 stale recording: {stdout}"
        return result

    if "45 days ago" not in stdout:
        result.error = "Age should be shown"
        return result

    result.passed = True
    return result


def test_stale_custom_days(temp_dir: Path) -> TestResult:
    """Test stale command with custom days threshold."""
    result = TestResult("stale_custom_days")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create recording 10 days old
    create_catalog_entry(
        temp_dir, "old1",
        transcriptions=[{"backend": "test"}],
        updated_at=past_iso(10)
    )

    # Default 30 days - should not find it
    rc, stdout, _ = run_cmd(["stale", "--days", "30"], env)
    if "No stale recordings" not in stdout:
        result.error = "10-day-old recording should not be stale at 30-day threshold"
        return result

    # 7 days threshold - should find it
    rc, stdout, _ = run_cmd(["stale", "--days", "7"], env)
    if "Found 1 recording" not in stdout:
        result.error = "10-day-old recording should be stale at 7-day threshold"
        return result

    result.passed = True
    return result


def test_stale_ignores_complete(temp_dir: Path) -> TestResult:
    """Test stale command ignores complete recordings."""
    result = TestResult("stale_ignores_complete")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create old but complete recording
    # Note: status="complete" requires both transcriptions AND an assignment file
    create_catalog_entry(
        temp_dir, "complete1",
        transcriptions=[{"backend": "test"}],
        review_status="complete",
        updated_at=past_iso(60)
    )
    # Create assignment file to make it truly "complete"
    create_assignment(temp_dir, "complete1", {
        "S1": {"speaker_id": "alice", "confidence": "high"}
    })

    rc, stdout, stderr = run_cmd(["stale", "--days", "30"], env)

    if rc != 0:
        result.error = f"stale command failed: {stderr}"
        return result

    if "No stale recordings" not in stdout:
        result.error = "Complete recordings should be ignored"
        return result

    result.passed = True
    return result


def test_stale_json_format(temp_dir: Path) -> TestResult:
    """Test stale command with JSON output."""
    result = TestResult("stale_json_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_catalog_entry(
        temp_dir, "old1",
        transcriptions=[{"backend": "test"}],
        updated_at=past_iso(45)
    )

    rc, stdout, stderr = run_cmd(["--format", "json", "stale"], env)

    if rc != 0:
        result.error = f"stale --format json failed: {stderr}"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
        return result

    if "threshold_days" not in data or "count" not in data:
        result.error = "Missing expected fields"
        return result

    result.passed = True
    return result


# =============================================================================
# Speakers Command Tests
# =============================================================================

def test_speakers_empty(temp_dir: Path) -> TestResult:
    """Test speakers command with no enrolled speakers."""
    result = TestResult("speakers_empty")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    (temp_dir / "db").mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = run_cmd(["speakers"], env)

    if rc != 0:
        result.error = f"speakers command failed: {stderr}"
        return result

    if "Total speakers: 0" not in stdout:
        result.error = f"Should show 0 speakers: {stdout}"
        return result

    result.passed = True
    return result


def test_speakers_with_data(temp_dir: Path) -> TestResult:
    """Test speakers command with enrolled speakers."""
    result = TestResult("speakers_with_data")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_speaker_profile(temp_dir, "alice", trust_level="high",
                          display_name="Alice Smith", sample_count=10, reviewed_samples=8)
    create_speaker_profile(temp_dir, "bob", trust_level="medium",
                          display_name="Bob Jones", sample_count=5, reviewed_samples=3)
    create_speaker_profile(temp_dir, "carol", trust_level="low",
                          sample_count=2, reviewed_samples=1)

    rc, stdout, stderr = run_cmd(["speakers"], env)

    if rc != 0:
        result.error = f"speakers command failed: {stderr}"
        return result

    if "Total speakers: 3" not in stdout:
        result.error = f"Should show 3 speakers: {stdout}"
        return result

    if "alice" not in stdout or "bob" not in stdout:
        result.error = "Speaker IDs should be listed"
        return result

    if "By trust level:" not in stdout:
        result.error = "Trust level summary missing"
        return result

    result.passed = True
    return result


def test_speakers_needing_samples(temp_dir: Path) -> TestResult:
    """Test speakers command shows speakers needing samples."""
    result = TestResult("speakers_needing_samples")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Speaker with enough samples
    create_speaker_profile(temp_dir, "alice", trust_level="high",
                          sample_count=10, reviewed_samples=8)

    # Speaker needing samples (< 3 reviewed)
    create_speaker_profile(temp_dir, "newbie", trust_level="low",
                          sample_count=1, reviewed_samples=1)

    rc, stdout, stderr = run_cmd(["speakers"], env)

    if rc != 0:
        result.error = f"speakers command failed: {stderr}"
        return result

    if "needing more reviewed samples" not in stdout:
        result.error = "Should show speakers needing samples"
        return result

    if "newbie" not in stdout:
        result.error = "Should list newbie as needing samples"
        return result

    result.passed = True
    return result


def test_speakers_json_format(temp_dir: Path) -> TestResult:
    """Test speakers command with JSON output."""
    result = TestResult("speakers_json_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    create_speaker_profile(temp_dir, "alice", trust_level="high",
                          display_name="Alice", sample_count=5)

    rc, stdout, stderr = run_cmd(["--format", "json", "speakers"], env)

    if rc != 0:
        result.error = f"speakers --format json failed: {stderr}"
        return result

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
        return result

    if "total" not in data or "speakers" not in data:
        result.error = "Missing expected fields"
        return result

    if len(data["speakers"]) != 1:
        result.error = f"Expected 1 speaker, got {len(data['speakers'])}"
        return result

    speaker = data["speakers"][0]
    if speaker.get("speaker_id") != "alice":
        result.error = f"Wrong speaker ID: {speaker.get('speaker_id')}"
        return result

    result.passed = True
    return result


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_version_flag(temp_dir: Path) -> TestResult:
    """Test --version flag."""
    result = TestResult("version_flag")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    rc, stdout, stderr = run_cmd(["--version"], env)

    # argparse uses returncode 0 for --version
    if "speaker-report" not in stdout and "speaker-report" not in stderr:
        result.error = f"Version info missing: stdout={stdout}, stderr={stderr}"
        return result

    result.passed = True
    return result


def test_invalid_format(temp_dir: Path) -> TestResult:
    """Test invalid format option."""
    result = TestResult("invalid_format")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    (temp_dir / "catalog").mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = run_cmd(["status", "--format", "invalid"], env)

    if rc == 0:
        result.error = "Should fail with invalid format"
        return result

    result.passed = True
    return result


def test_missing_directories(temp_dir: Path) -> TestResult:
    """Test handling when directories don't exist yet."""
    result = TestResult("missing_directories")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Don't create any directories - tool should handle gracefully
    rc, stdout, stderr = run_cmd(["status"], env)

    if rc != 0:
        result.error = f"Should handle missing directories: {stderr}"
        return result

    if "0 total" not in stdout:
        result.error = "Should show 0 items for missing directories"
        return result

    result.passed = True
    return result


def test_malformed_yaml(temp_dir: Path) -> TestResult:
    """Test handling of malformed YAML files."""
    result = TestResult("malformed_yaml")
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create a malformed YAML file
    catalog_dir = temp_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    bad_file = catalog_dir / "bad123.yaml"
    with open(bad_file, "w") as f:
        f.write("this: is: not: valid: yaml: [\n")

    # Create a good file too
    create_catalog_entry(temp_dir, "good456")

    rc, stdout, stderr = run_cmd(["status"], env)

    # Should still work, with warning
    if rc != 0:
        result.error = f"Should handle malformed YAML gracefully: {stderr}"
        return result

    # Should process the good file
    if "1 total" not in stdout:
        result.error = "Should still count valid entries"
        return result

    result.passed = True
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="speaker-report CLI unit tests")
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
        test_status_empty,
        test_status_with_data,
        test_status_json_format,
        test_status_default_command,
        test_status_recommendations,

        # Coverage command tests
        test_coverage_empty,
        test_coverage_by_context,
        test_coverage_filter_context,
        test_coverage_json_format,

        # Confidence command tests
        test_confidence_empty,
        test_confidence_finds_low,
        test_confidence_threshold,
        test_confidence_json_format,

        # Stale command tests
        test_stale_empty,
        test_stale_finds_old,
        test_stale_custom_days,
        test_stale_ignores_complete,
        test_stale_json_format,

        # Speakers command tests
        test_speakers_empty,
        test_speakers_with_data,
        test_speakers_needing_samples,
        test_speakers_json_format,

        # Edge cases
        test_version_flag,
        test_invalid_format,
        test_missing_directories,
        test_malformed_yaml,
    ]

    print("speaker-report CLI Unit Tests")
    print("=" * 40)

    passed = 0
    failed = 0
    skipped = 0
    results = []

    for test_func in tests:
        # Create fresh temp directory for each test
        temp_dir = Path(tempfile.mkdtemp(prefix="report_test_"))

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
                print(f"        Exception: {e}")
            failed += 1

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary
    print("-" * 40)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total:   {len(tests)} tests")

    # Detailed failures in verbose mode
    if args.verbose and failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"  - {r.name}: {r.error}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
