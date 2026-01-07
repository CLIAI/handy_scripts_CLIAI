#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

run_total=0
run_passed=0
run_failed=0
run_skipped=0

run_eval() {
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

run_eval "stt_assemblyai" "${SCRIPT_DIR}/run_stt_assemblyai.py"
run_eval "stt_speechmatics" "${SCRIPT_DIR}/run_stt_speechmatics.py"
run_eval "stt_openai" "${SCRIPT_DIR}/run_stt_openai.py"

echo "All evals: total=${run_total} passed=${run_passed} failed=${run_failed} skipped=${run_skipped}"

if [ "$run_failed" -gt 0 ]; then
  exit 1
fi

if [ "$run_skipped" -gt 0 ]; then
  exit 2
fi

exit 0
