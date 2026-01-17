#!/usr/bin/env python3
"""
End-to-end integration test for the speaker-* tool pipeline.

Tests the complete workflow:
1. speaker-catalog add (add recording to catalog)
2. speaker-catalog register-transcript (register mock transcript)
3. speaker-assign (assign speaker names)
4. speaker-review status (check review status)
5. speaker-report status (check system status)

This test uses synthetic audio and mock transcripts to verify
the tools work together without requiring API keys.

Usage:
    python test_e2e_pipeline.py [-v]

Docker:
    docker build -f evals/Dockerfile.test -t speaker-tools-test .
    docker run --rm speaker-tools-test python evals/speaker_detection/test_e2e_pipeline.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

SPEAKER_CATALOG = REPO_ROOT / "speaker-catalog"
SPEAKER_ASSIGN = REPO_ROOT / "speaker-assign"
SPEAKER_REVIEW = REPO_ROOT / "speaker-review"
SPEAKER_REPORT = REPO_ROOT / "speaker-report"
SPEAKER_DETECTION = REPO_ROOT / "speaker_detection"

VERBOSE = "-v" in sys.argv


# =============================================================================
# Utilities
# =============================================================================

def log(msg: str) -> None:
    """Print message if verbose."""
    if VERBOSE:
        print(f"  [DEBUG] {msg}")


def run_cmd(cmd: list[str], env: dict = None, check: bool = True) -> tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    log(f"Running: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        env=full_env,
    )

    if result.returncode != 0 and check:
        log(f"STDERR: {result.stderr}")

    return result.returncode, result.stdout, result.stderr


def create_test_audio(path: Path, duration_sec: float = 2.0, unique_id: int = 0) -> None:
    """Create a WAV file with unique content using sine tone."""
    # Use different frequencies to ensure unique b3sum
    # Each unique_id gets a different base frequency
    freq = 440 + (unique_id * 100)  # 440Hz, 540Hz, 640Hz, etc.
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency={freq}:duration={duration_sec}",
        "-ar", "16000", "-ac", "1",
        str(path)
    ], capture_output=True, check=True)


def create_mock_transcript(path: Path, speakers: list[str]) -> None:
    """Create a mock AssemblyAI-style transcript."""
    utterances = []
    time_offset = 0

    texts = [
        "Hello everyone, this is Alice speaking.",
        "Hi Alice, Bob here. How are you today?",
        "I'm doing great Bob, thanks for asking.",
        "Let me introduce Carol who just joined us.",
        "Hello, this is Carol. Nice to meet everyone.",
    ]

    for i, (speaker, text) in enumerate(zip(speakers * 3, texts * 2)):
        if i >= len(texts):
            break
        utterances.append({
            "speaker": speaker,
            "start": time_offset * 1000,  # milliseconds
            "end": (time_offset + 3) * 1000,
            "text": text,
        })
        time_offset += 4

    transcript = {
        "utterances": utterances,
        "audio_duration": time_offset * 1000,
    }

    with open(path, "w") as f:
        json.dump(transcript, f, indent=2)


# =============================================================================
# Test Cases
# =============================================================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None


def test_e2e_pipeline(temp_dir: Path) -> list[TestResult]:
    """Run the complete end-to-end pipeline test."""
    results = []
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create test files
    audio_path = temp_dir / "test_meeting.wav"
    transcript_path = temp_dir / "test_meeting.assemblyai.json"

    # === Setup ===
    result = TestResult("setup_test_files")
    try:
        create_test_audio(audio_path, duration_sec=5.0, unique_id=42)
        create_mock_transcript(transcript_path, ["A", "B", "C"])
        result.passed = True
    except Exception as e:
        result.error = f"Failed to create test files: {e}"
    results.append(result)

    if not result.passed:
        return results  # Can't continue without test files

    # === Step 1: speaker-catalog add ===
    result = TestResult("catalog_add")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "add", str(audio_path),
        "--context", "test-meeting",
        "--tags", "e2e,test",
    ], env)

    if rc == 0:
        # Verify catalog entry was created
        catalog_dir = temp_dir / "catalog"
        if catalog_dir.exists() and list(catalog_dir.glob("*.yaml")):
            result.passed = True
            log(f"Catalog entry created in {catalog_dir}")
        else:
            result.error = "Catalog entry not created"
    else:
        result.error = f"speaker-catalog add failed: {stderr}"
    results.append(result)

    # === Step 2: speaker-catalog register-transcript ===
    result = TestResult("catalog_register_transcript")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "register-transcript", str(audio_path),
        "--backend", "assemblyai",
        "--transcript", str(transcript_path),
    ], env)

    if rc == 0:
        result.passed = True
    else:
        result.error = f"speaker-catalog register-transcript failed: {stderr}"
    results.append(result)

    # === Step 3: speaker-catalog status ===
    result = TestResult("catalog_status_transcribed")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "status", str(audio_path),
    ], env)

    if rc == 0 and "transcribed" in stdout.lower():
        result.passed = True
    else:
        result.error = f"Expected status 'transcribed', got: {stdout}"
    results.append(result)

    # === Step 4: speaker-assign ===
    result = TestResult("assign_speakers")
    rc, stdout, stderr = run_cmd([
        SPEAKER_ASSIGN, "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--context", "test-meeting",
        "--expected-speakers", "alice,bob,carol",
    ], env)

    if rc == 0:
        # Verify assignments were created
        assignments_dir = temp_dir / "assignments"
        if assignments_dir.exists() and list(assignments_dir.glob("*.yaml")):
            result.passed = True
            log(f"Assignments created in {assignments_dir}")
        else:
            result.error = "Assignments not created"
    else:
        result.error = f"speaker-assign failed: {stderr}"
    results.append(result)

    # === Step 5: speaker-assign show ===
    result = TestResult("assign_show")
    rc, stdout, stderr = run_cmd([
        SPEAKER_ASSIGN, "show", str(audio_path),
        "--format", "json",
    ], env)

    if rc == 0:
        try:
            data = json.loads(stdout)
            if "mappings" in data and len(data["mappings"]) > 0:
                result.passed = True
                log(f"Assignments: {list(data['mappings'].keys())}")
            else:
                result.error = "No mappings in assignment output"
        except json.JSONDecodeError as e:
            result.error = f"Invalid JSON: {e}"
    else:
        result.error = f"speaker-assign show failed: {stderr}"
    results.append(result)

    # === Step 6: speaker-catalog status (should be assigned) ===
    result = TestResult("catalog_status_assigned")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "status", str(audio_path),
    ], env)

    if rc == 0 and "assigned" in stdout.lower():
        result.passed = True
    else:
        result.error = f"Expected status 'assigned', got: {stdout}"
    results.append(result)

    # === Step 7: speaker-review status ===
    result = TestResult("review_status")
    rc, stdout, stderr = run_cmd([
        SPEAKER_REVIEW, "status",
    ], env)

    # Should succeed (may or may not have active session)
    if rc == 0:
        result.passed = True
    else:
        result.error = f"speaker-review status failed: {stderr}"
    results.append(result)

    # === Step 8: speaker-report status ===
    result = TestResult("report_status")
    rc, stdout, stderr = run_cmd([
        SPEAKER_REPORT, "status",
    ], env)

    if rc == 0:
        # Check output contains expected sections
        if "Recording" in stdout or "recording" in stdout.lower():
            result.passed = True
        else:
            result.error = f"Unexpected report output: {stdout[:200]}"
    else:
        result.error = f"speaker-report status failed: {stderr}"
    results.append(result)

    # === Step 9: speaker-report coverage ===
    result = TestResult("report_coverage")
    rc, stdout, stderr = run_cmd([
        SPEAKER_REPORT, "coverage",
    ], env)

    if rc == 0:
        result.passed = True
    else:
        result.error = f"speaker-report coverage failed: {stderr}"
    results.append(result)

    # === Step 10: speaker-catalog list ===
    result = TestResult("catalog_list")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "list", "--format", "json",
    ], env)

    if rc == 0:
        try:
            data = json.loads(stdout)
            if isinstance(data, list) and len(data) == 1:
                result.passed = True
            else:
                result.error = f"Expected 1 recording, got: {len(data) if isinstance(data, list) else 'not a list'}"
        except json.JSONDecodeError:
            result.error = f"Invalid JSON: {stdout[:100]}"
    else:
        result.error = f"speaker-catalog list failed: {stderr}"
    results.append(result)

    # === Step 11: Cleanup test - speaker-assign clear ===
    result = TestResult("assign_clear")
    rc, stdout, stderr = run_cmd([
        SPEAKER_ASSIGN, "clear", str(audio_path), "--force",
    ], env)

    if rc == 0:
        # Verify assignments were removed
        assignments_dir = temp_dir / "assignments"
        if not list(assignments_dir.glob("*.yaml")):
            result.passed = True
        else:
            result.error = "Assignments not cleared"
    else:
        result.error = f"speaker-assign clear failed: {stderr}"
    results.append(result)

    # === Step 12: speaker-catalog remove ===
    result = TestResult("catalog_remove")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "remove", str(audio_path), "--force",
    ], env)

    if rc == 0:
        # Verify catalog entry was removed
        catalog_dir = temp_dir / "catalog"
        if not list(catalog_dir.glob("*.yaml")):
            result.passed = True
        else:
            result.error = "Catalog entry not removed"
    else:
        result.error = f"speaker-catalog remove failed: {stderr}"
    results.append(result)

    return results


def test_multi_recording_pipeline(temp_dir: Path) -> list[TestResult]:
    """Test pipeline with multiple recordings."""
    results = []
    env = {"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)}

    # Create multiple test recordings
    recordings = []
    for i in range(3):
        audio_path = temp_dir / f"meeting_{i}.wav"
        transcript_path = temp_dir / f"meeting_{i}.assemblyai.json"

        create_test_audio(audio_path, duration_sec=3.0, unique_id=100 + i)
        create_mock_transcript(transcript_path, ["A", "B"])
        recordings.append((audio_path, transcript_path))

    # === Add all recordings ===
    result = TestResult("multi_catalog_add")
    all_added = True
    for audio_path, _ in recordings:
        rc, _, stderr = run_cmd([
            SPEAKER_CATALOG, "add", str(audio_path),
            "--context", "batch-test",
        ], env)
        if rc != 0:
            all_added = False
            result.error = f"Failed to add {audio_path.name}: {stderr}"
            break

    if all_added:
        result.passed = True
    results.append(result)

    # === List and verify count ===
    result = TestResult("multi_catalog_list")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "list", "--format", "json",
    ], env)

    if rc == 0:
        try:
            data = json.loads(stdout)
            if len(data) == 3:
                result.passed = True
            else:
                result.error = f"Expected 3 recordings, got {len(data)}"
        except json.JSONDecodeError:
            result.error = "Invalid JSON"
    else:
        result.error = f"List failed: {stderr}"
    results.append(result)

    # === Filter by context ===
    result = TestResult("multi_filter_context")
    rc, stdout, stderr = run_cmd([
        SPEAKER_CATALOG, "list",
        "--context", "batch-test",
        "--format", "json",
    ], env)

    if rc == 0:
        try:
            data = json.loads(stdout)
            if len(data) == 3:
                result.passed = True
            else:
                result.error = f"Expected 3 in context, got {len(data)}"
        except json.JSONDecodeError:
            result.error = "Invalid JSON"
    else:
        result.error = f"Filter failed: {stderr}"
    results.append(result)

    # === Report status shows all ===
    result = TestResult("multi_report_status")
    rc, stdout, stderr = run_cmd([
        SPEAKER_REPORT, "status",
    ], env)

    if rc == 0:
        result.passed = True
    else:
        result.error = f"Report failed: {stderr}"
    results.append(result)

    return results


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print("=" * 60)
    print("Speaker-* Pipeline End-to-End Integration Tests")
    print("=" * 60)

    # Check required tools exist
    required_tools = [
        (SPEAKER_CATALOG, "speaker-catalog"),
        (SPEAKER_ASSIGN, "speaker-assign"),
        (SPEAKER_REVIEW, "speaker-review"),
        (SPEAKER_REPORT, "speaker-report"),
    ]

    missing = []
    for path, name in required_tools:
        if not path.exists():
            missing.append(name)

    if missing:
        print(f"\nERROR: Missing required tools: {', '.join(missing)}")
        print("Run from repository root directory.")
        return 1

    # Check ffmpeg for audio generation
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: ffmpeg required for test audio generation")
        return 1

    all_results = []

    # Run single recording pipeline test
    print("\n--- Single Recording Pipeline ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="e2e_single_"))
    try:
        results = test_e2e_pipeline(temp_dir)
        all_results.extend(results)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Run multi-recording pipeline test
    print("\n--- Multi-Recording Pipeline ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="e2e_multi_"))
    try:
        results = test_multi_recording_pipeline(temp_dir)
        all_results.extend(results)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("-" * 60)

    passed = 0
    failed = 0

    for r in all_results:
        if r.passed:
            print(f"  PASS: {r.name}")
            passed += 1
        else:
            print(f"  FAIL: {r.name}")
            if r.error:
                print(f"        {r.error}")
            failed += 1

    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
