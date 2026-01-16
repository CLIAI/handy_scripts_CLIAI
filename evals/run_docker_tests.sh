#!/usr/bin/env bash
#
# Run all tests in Docker container for reproducibility
#
# Usage:
#   ./run_docker_tests.sh              # Build and run tests
#   ./run_docker_tests.sh --no-cache   # Force rebuild
#   ./run_docker_tests.sh --shell      # Interactive shell in container
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="speaker-tools-test"

# Parse arguments
BUILD_ARGS=""
RUN_MODE="test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache)
            BUILD_ARGS="--no-cache"
            shift
            ;;
        --shell)
            RUN_MODE="shell"
            shift
            ;;
        --help)
            echo "Usage: $0 [--no-cache] [--shell]"
            echo ""
            echo "Options:"
            echo "  --no-cache  Force rebuild without cache"
            echo "  --shell     Start interactive shell instead of tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

echo "=== Building Docker test image ==="
docker build $BUILD_ARGS -f evals/Dockerfile.test -t "$IMAGE_NAME" .

echo ""
echo "=== Running tests in Docker ==="

if [ "$RUN_MODE" = "shell" ]; then
    echo "(Interactive shell - run ./evals/speaker_detection/test_all.sh to test)"
    docker run -it --rm \
        -e SPEAKERS_EMBEDDINGS_DIR=/tmp/test_speakers \
        "$IMAGE_NAME" \
        bash
else
    # Run tests
    docker run --rm \
        -e SPEAKERS_EMBEDDINGS_DIR=/tmp/test_speakers \
        "$IMAGE_NAME"

    echo ""
    echo "=== All Docker tests completed successfully ==="
fi
