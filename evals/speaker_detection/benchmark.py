#!/usr/bin/env python3
"""
Speaker Detection Benchmark

Tests speaker enrollment and identification accuracy using espeak-ng
generated test audio files.

Usage:
    ./benchmark.py                      # Run all tests
    ./benchmark.py --tests 001          # Run specific test
    ./benchmark.py --backend speechmatics  # Use specific backend
    ./benchmark.py --dry-run            # Show what would be done
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add repo root to path for imports
sys.path.insert(0, str(REPO_ROOT))


def log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, file=sys.stderr)


def load_test_cases(test_filter: str = None) -> list:
    """Load test case definitions from samples/*.test.json"""
    samples_dir = SCRIPT_DIR / "samples"
    cases = []

    for path in sorted(samples_dir.glob("*.test.json")):
        with open(path) as f:
            case = json.load(f)
            case["_path"] = str(path)

            # Filter by test ID if specified
            if test_filter:
                filters = [f.strip() for f in test_filter.split(",")]
                if not any(case["id"].startswith(f) for f in filters):
                    continue

            cases.append(case)

    return cases


def check_audio_files(case: dict) -> list:
    """Check if required audio files exist, return list of missing files."""
    missing = []

    for speaker_id, speaker_info in case.get("speakers", {}).items():
        audio_path = SCRIPT_DIR / speaker_info["enrollment_audio"]
        if not audio_path.exists():
            missing.append(str(audio_path))

    test_audio = SCRIPT_DIR / case["test_audio"]
    if not test_audio.exists():
        missing.append(str(test_audio))

    return missing


def run_test(case: dict, backend: str, temp_dir: Path, verbose: bool = False) -> dict:
    """
    Run a single test case.

    Returns dict with:
        - passed: bool
        - enrolled: list of speaker IDs enrolled
        - identified: list of speaker IDs identified
        - expected: list of expected speaker IDs
        - error: str if failed
    """
    result = {
        "test_id": case["id"],
        "passed": False,
        "enrolled": [],
        "identified": [],
        "expected": case["expected_speakers"],
        "error": None,
    }

    try:
        # Import speaker_detection functions
        from speaker_detection_backends import get_backend

        # Set temp database location
        db_dir = temp_dir / "db"
        emb_dir = temp_dir / "embeddings"
        db_dir.mkdir(parents=True, exist_ok=True)
        emb_dir.mkdir(parents=True, exist_ok=True)

        os.environ["SPEAKERS_EMBEDDINGS_DIR"] = str(temp_dir)

        # Load backend
        backend_instance = get_backend(backend)

        # Phase 1: Enroll each speaker
        log(f"  Enrolling speakers...", verbose)
        enrolled_profiles = []

        for speaker_id, speaker_info in case.get("speakers", {}).items():
            audio_path = SCRIPT_DIR / speaker_info["enrollment_audio"]

            # Create speaker profile
            profile = {
                "id": speaker_id,
                "version": 1,
                "names": {"default": speaker_id.title()},
                "nicknames": [],
                "description": f"Test speaker {speaker_id}",
                "metadata": {},
                "tags": ["test"],
                "embeddings": {},
            }

            # Enroll speaker
            try:
                enrollment = backend_instance.enroll_speaker(audio_path)
                profile["embeddings"][backend] = [{
                    "id": f"emb-{speaker_id}",
                    "external_id": enrollment.get("external_id"),
                    "all_identifiers": enrollment.get("all_identifiers", []),
                    "model_version": enrollment.get("model_version", f"{backend}-v2"),
                    "source_audio": str(audio_path),
                }]
                result["enrolled"].append(speaker_id)
                enrolled_profiles.append(profile)

                # Save profile to temp DB
                with open(db_dir / f"{speaker_id}.json", "w") as f:
                    json.dump(profile, f, indent=2)

                log(f"    Enrolled: {speaker_id}", verbose)
            except Exception as e:
                log(f"    Failed to enroll {speaker_id}: {e}", verbose)
                result["error"] = f"Enrollment failed for {speaker_id}: {e}"
                return result

        # Phase 2: Identify speakers in test audio
        log(f"  Identifying speakers in test audio...", verbose)
        test_audio = SCRIPT_DIR / case["test_audio"]

        try:
            matches = backend_instance.identify_speaker(
                test_audio,
                enrolled_profiles,
                threshold=0.354,
            )

            result["identified"] = [m["speaker_id"] for m in matches]
            log(f"    Identified: {result['identified']}", verbose)

        except Exception as e:
            result["error"] = f"Identification failed: {e}"
            return result

        # Phase 3: Score results
        expected_set = set(case["expected_speakers"])
        identified_set = set(result["identified"])

        # All expected speakers should be identified
        result["passed"] = expected_set == identified_set

        if not result["passed"]:
            missing = expected_set - identified_set
            extra = identified_set - expected_set
            if missing:
                result["error"] = f"Missing speakers: {missing}"
            if extra:
                result["error"] = (result.get("error", "") + f" Extra speakers: {extra}").strip()

    except Exception as e:
        result["error"] = f"Test error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Speaker Detection Benchmark")
    parser.add_argument("--tests", "-t", help="Filter tests by ID prefix (comma-separated)")
    parser.add_argument("--backend", "-b", default="speechmatics",
                        help="Embedding backend (default: speechmatics)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be done without running")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary directory after test")
    args = parser.parse_args()

    # Load test cases
    cases = load_test_cases(args.tests)

    if not cases:
        print("No test cases found.", file=sys.stderr)
        return 1

    print(f"Speaker Detection Benchmark")
    print(f"Backend: {args.backend}")
    print(f"Tests: {len(cases)}")
    print()

    if args.dry_run:
        for case in cases:
            missing = check_audio_files(case)
            status = "READY" if not missing else f"MISSING: {missing}"
            print(f"  {case['id']}: {status}")
        return 0

    # Check for missing audio files
    all_missing = []
    for case in cases:
        missing = check_audio_files(case)
        all_missing.extend(missing)

    if all_missing:
        print("Missing audio files. Run 'make' in evals/speaker_detection/ first:")
        for m in all_missing:
            print(f"  {m}")
        return 2

    # Run tests
    passed = 0
    failed = 0
    results = []

    for case in cases:
        print(f"Test: {case['id']} - {case.get('description', '')}")

        # Create temp directory for this test
        temp_dir = Path(tempfile.mkdtemp(prefix=f"spk_eval_{case['id']}_"))

        try:
            result = run_test(case, args.backend, temp_dir, args.verbose)
            results.append(result)

            if result["passed"]:
                print(f"  PASS")
                passed += 1
            else:
                print(f"  FAIL: {result.get('error', 'Unknown error')}")
                failed += 1

        finally:
            if not args.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

        print()

    # Summary
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
