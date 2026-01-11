#!/usr/bin/env bash
#
# Speaker Detection Test Runner
#
# Runs both unit tests (no API) and integration tests (with API).
#
# Usage:
#   ./test_all.sh           # Run all tests
#   ./test_all.sh --unit    # Unit tests only
#   ./test_all.sh --api     # API tests only
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

run_total=0
run_passed=0
run_failed=0
run_skipped=0

run_test() {
  local name="$1"
  shift
  run_total=$((run_total + 1))
  if "$@"; then
    echo "PASS: ${name}"
    run_passed=$((run_passed + 1))
  else
    local rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "SKIP: ${name}"
      run_skipped=$((run_skipped + 1))
    else
      echo "FAIL: ${name}"
      run_failed=$((run_failed + 1))
    fi
  fi
}

# Parse arguments
RUN_UNIT=true
RUN_API=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)
      RUN_API=false
      shift
      ;;
    --api)
      RUN_UNIT=false
      shift
      ;;
    *)
      echo "Usage: $0 [--unit|--api]"
      exit 1
      ;;
  esac
done

echo "Speaker Detection Tests"
echo "======================="
echo

# Unit tests (no API required)
if $RUN_UNIT; then
  echo "=== Unit Tests (No API) ==="
  run_test "cli_unit_tests" "${SCRIPT_DIR}/test_cli.py"
  echo
fi

# API tests (requires SPEECHMATICS_API_KEY)
if $RUN_API; then
  echo "=== API Integration Tests ==="
  if [ -z "${SPEECHMATICS_API_KEY:-}" ]; then
    echo "SKIP: SPEECHMATICS_API_KEY not set"
    run_skipped=$((run_skipped + 1))
    run_total=$((run_total + 1))
  else
    # Check if audio files exist
    if [ ! -f "${SCRIPT_DIR}/audio/enroll_alice.wav" ]; then
      echo "Generating test audio files..."
      make -C "${SCRIPT_DIR}" all
    fi
    run_test "speechmatics_benchmark" "${SCRIPT_DIR}/benchmark.py"
  fi
  echo
fi

echo "======================="
echo "Results: total=${run_total} passed=${run_passed} failed=${run_failed} skipped=${run_skipped}"

if [ "$run_failed" -gt 0 ]; then
  exit 1
fi

if [ "$run_skipped" -gt 0 ] && [ "$run_passed" -eq 0 ]; then
  exit 2
fi

exit 0
