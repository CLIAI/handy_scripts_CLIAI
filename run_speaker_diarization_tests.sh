#!/usr/bin/env bash
#
# Unified Test Runner for Speaker-* Tool Ecosystem
#
# Usage:
#   ./run_speaker_diarization_tests.sh                    # Run all tests
#   ./run_speaker_diarization_tests.sh unit               # Unit tests only (fast, no API)
#   ./run_speaker_diarization_tests.sh e2e                # End-to-end pipeline tests
#   ./run_speaker_diarization_tests.sh catalog            # speaker-catalog tests only
#   ./run_speaker_diarization_tests.sh assign             # speaker-assign tests only
#   ./run_speaker_diarization_tests.sh review             # speaker-review tests only
#   ./run_speaker_diarization_tests.sh llm                # speaker-llm tests only
#   ./run_speaker_diarization_tests.sh process            # speaker-process tests only
#   ./run_speaker_diarization_tests.sh report             # speaker-report tests only
#   ./run_speaker_diarization_tests.sh legacy             # Original speaker_detection/samples tests
#   ./run_speaker_diarization_tests.sh docker             # Run all tests in Docker container
#   ./run_speaker_diarization_tests.sh list               # List available test collections
#   ./run_speaker_diarization_tests.sh --doc [COLLECTION] # Show documentation for collection
#   ./run_speaker_diarization_tests.sh --doc-path [COLLECTION]  # Show path to doc file
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVALS_DIR="${SCRIPT_DIR}/evals/speaker_detection"

# Documentation paths for each collection
declare -A DOC_PATHS=(
  [catalog]="${EVALS_DIR}/test_speaker_catalog.README.md"
  [assign]="${EVALS_DIR}/test_speaker_assign.README.md"
  [review]="${EVALS_DIR}/test_speaker_review.README.md"
  [llm]="${EVALS_DIR}/test_speaker_llm.README.md"
  [process]="${EVALS_DIR}/test_speaker_process.README.md"
  [report]="${EVALS_DIR}/test_speaker_report.README.md"
  [segments]="${EVALS_DIR}/test_speaker_segments.README.md"
  [profiles]="${EVALS_DIR}/test_audio_profiles.README.md"
  [legacy]="${EVALS_DIR}/test_legacy.README.md"
  [e2e]="${EVALS_DIR}/test_e2e_pipeline.README.md"
  [unit]="${EVALS_DIR}/test_unit.README.md"
  [all]="${SCRIPT_DIR}/run_speaker_diarization_tests.sh.README.md"
)

# Handle --doc: output documentation for a collection
if [ "${1:-}" = "--doc" ]; then
  collection="${2:-all}"
  doc_path="${DOC_PATHS[$collection]:-}"
  if [ -z "$doc_path" ] || [ ! -f "$doc_path" ]; then
    echo "No documentation for collection: $collection" >&2
    echo "Available collections: ${!DOC_PATHS[*]}" >&2
    exit 1
  fi
  cat "$doc_path"
  exit 0
fi

# Handle --doc-path: output path to documentation file
if [ "${1:-}" = "--doc-path" ]; then
  collection="${2:-all}"
  doc_path="${DOC_PATHS[$collection]:-}"
  if [ -z "$doc_path" ]; then
    echo "Unknown collection: $collection" >&2
    echo "Available collections: ${!DOC_PATHS[*]}" >&2
    exit 1
  fi
  echo "$doc_path"
  exit 0
fi

# Colors (if terminal supports it)
if [ -t 1 ]; then
  GREEN='\033[0;32m'
  RED='\033[0;31m'
  YELLOW='\033[0;33m'
  BLUE='\033[0;34m'
  NC='\033[0m' # No Color
else
  GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

# Counters
total=0
passed=0
failed=0
skipped=0

run_test() {
  local name="$1"
  local cmd="$2"
  total=$((total + 1))

  echo -e "${BLUE}Running:${NC} $name"

  if eval "$cmd" > /tmp/test_output_$$.txt 2>&1; then
    # Extract pass/fail counts from output
    local test_passed=$(grep -oP 'Results?: \K\d+(?= passed)' /tmp/test_output_$$.txt 2>/dev/null | tail -1 || echo "1")
    local test_failed=$(grep -oP '\d+(?= failed)' /tmp/test_output_$$.txt 2>/dev/null | tail -1 || echo "0")

    if [ "$test_failed" = "0" ]; then
      echo -e "  ${GREEN}PASS${NC}: $name (${test_passed} tests)"
      passed=$((passed + 1))
    else
      echo -e "  ${RED}FAIL${NC}: $name (${test_failed} failures)"
      cat /tmp/test_output_$$.txt | grep -E "(FAIL|Error|error)" | head -5
      failed=$((failed + 1))
    fi
  else
    local rc=$?
    if [ "$rc" -eq 2 ]; then
      echo -e "  ${YELLOW}SKIP${NC}: $name"
      skipped=$((skipped + 1))
    else
      echo -e "  ${RED}FAIL${NC}: $name (exit code: $rc)"
      cat /tmp/test_output_$$.txt | tail -10
      failed=$((failed + 1))
    fi
  fi
  rm -f /tmp/test_output_$$.txt
}

run_collection() {
  local collection="$1"

  case "$collection" in
    catalog)
      run_test "speaker-catalog" "python ${EVALS_DIR}/test_speaker_catalog.py"
      ;;
    assign)
      run_test "speaker-assign" "python ${EVALS_DIR}/test_speaker_assign.py"
      ;;
    review)
      run_test "speaker-review" "python ${EVALS_DIR}/test_speaker_review.py"
      ;;
    llm)
      run_test "speaker-llm" "python ${EVALS_DIR}/test_speaker_llm.py"
      ;;
    process)
      run_test "speaker-process" "python ${EVALS_DIR}/test_speaker_process.py"
      ;;
    report)
      run_test "speaker-report" "python ${EVALS_DIR}/test_speaker_report.py"
      ;;
    segments)
      run_test "speaker-segments" "python ${EVALS_DIR}/test_speaker_segments.py"
      ;;
    profiles)
      run_test "audio-profiles" "python ${EVALS_DIR}/test_audio_profiles.py"
      ;;
    legacy)
      run_test "speaker_detection CLI" "python ${EVALS_DIR}/test_cli.py"
      run_test "speaker_samples & trust" "python ${EVALS_DIR}/test_samples_and_trust.py"
      ;;
    e2e)
      run_test "E2E pipeline" "python ${EVALS_DIR}/test_e2e_pipeline.py"
      ;;
    unit)
      # All unit tests (no API required)
      run_collection catalog
      run_collection assign
      run_collection review
      run_collection llm
      run_collection process
      run_collection report
      run_collection segments
      run_collection profiles
      run_collection legacy
      ;;
    all)
      run_collection unit
      run_collection e2e
      ;;
    docker)
      echo -e "${BLUE}Building Docker image...${NC}"
      docker build -f evals/Dockerfile.test -t speaker-tools-test .
      echo -e "${BLUE}Running tests in Docker...${NC}"
      docker run --rm speaker-tools-test
      echo -e "${GREEN}Docker tests completed${NC}"
      exit 0
      ;;
    *)
      echo "Unknown collection: $collection"
      echo "Use './run_speaker_diarization_tests.sh list' to see available collections"
      exit 1
      ;;
  esac
}

print_help() {
  cat << 'EOF'
Speaker-* Tool Test Runner
==========================

Usage: ./run_speaker_diarization_tests.sh [COLLECTION]
       ./run_speaker_diarization_tests.sh --doc [COLLECTION]
       ./run_speaker_diarization_tests.sh --doc-path [COLLECTION]

Collections:
  all       Run all tests (default)
  unit      All unit tests (fast, no API required)
  e2e       End-to-end pipeline integration tests

  catalog   speaker-catalog tests (23 tests)
  assign    speaker-assign tests (24 tests)
  review    speaker-review tests (18 tests)
  llm       speaker-llm tests (23 tests)
  process   speaker-process tests (22 tests)
  report    speaker-report tests (26 tests)
  segments  speaker_segments tests (8 tests)
  profiles  audio profiles tests (10 tests)
  legacy    Original speaker_detection/samples tests (27 tests)

  docker    Build and run all tests in Docker container
  list      Show this help

Documentation:
  --doc [COLLECTION]       Show documentation for a collection (default: all)
  --doc-path [COLLECTION]  Show path to documentation file

Examples:
  ./run_speaker_diarization_tests.sh              # Run all tests
  ./run_speaker_diarization_tests.sh unit         # Quick unit tests
  ./run_speaker_diarization_tests.sh catalog      # Test only speaker-catalog
  ./run_speaker_diarization_tests.sh docker       # Run in isolated Docker environment
  ./run_speaker_diarization_tests.sh --doc catalog  # View catalog test docs

Test counts:
  Unit tests:  181 total
  E2E tests:    17 total
  Total:       198 tests
EOF
}

# Main
echo "========================================"
echo "Speaker-* Tool Test Runner"
echo "========================================"
echo

if [ $# -eq 0 ]; then
  collection="all"
else
  collection="$1"
fi

if [ "$collection" = "list" ] || [ "$collection" = "help" ] || [ "$collection" = "-h" ] || [ "$collection" = "--help" ]; then
  print_help
  exit 0
fi

run_collection "$collection"

echo
echo "========================================"
echo -e "Results: ${GREEN}${passed} passed${NC}, ${RED}${failed} failed${NC}, ${YELLOW}${skipped} skipped${NC}"
echo "========================================"

if [ "$failed" -gt 0 ]; then
  exit 1
fi
exit 0
