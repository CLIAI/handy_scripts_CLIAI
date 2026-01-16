# Contributing to handy_scripts_CLIAI

This document establishes coding guidelines and design principles for tools in this repository.

## Core Philosophy: KISS UNIX

Tools in this repository follow the UNIX philosophy:

1. **Do one thing well** - Each tool has a focused purpose
2. **Compose via pipes and files** - Tools read/write JSON, YAML, plain text
3. **Text streams are universal interface** - Prefer text over binary formats
4. **Small is beautiful** - Favor simple implementations over clever ones
5. **Fail fast, fail loud** - Clear error messages, non-zero exit codes

## Python CLI Tool Conventions

### File Naming and Execution

```
✓ speaker_detection     (no extension, executable)
✓ speaker_samples       (no extension, executable)
✗ speaker_detection.py  (avoid .py for CLI tools)
```

**Why no `.py` extension?**

* Cleaner invocation: `./speaker_detection` vs `python3 speaker_detection.py`
* Signals "this is a command" not "this is a library"
* Matches UNIX convention (`ls`, `grep`, `git`)

**Required setup:**

```bash
chmod +x tool_name           # Make executable
```

**Required shebang:**

```python
#!/usr/bin/env python3
```

### Dual-Use Pattern: CLI + Library

Tools should work both as standalone commands AND as importable modules:

```python
#!/usr/bin/env python3
"""
tool_name - Brief description

Usage:
    tool_name command [options]

Environment:
    SOME_VAR - Description (default: value)
"""

import argparse
import sys

# ----------------------------------------------------------------------
# Core Functions (importable)
# ----------------------------------------------------------------------

def do_something(input_path, options=None):
    """
    Process input and return result.

    Can be called directly when imported as library.
    """
    # Implementation
    return result


def another_function(data):
    """Another reusable function."""
    pass


# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------

def cmd_action(args):
    """CLI handler for 'action' command."""
    result = do_something(args.input, args.options)
    print(json.dumps(result, indent=2))
    return 0


# ----------------------------------------------------------------------
# Main (CLI entry point)
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tool description")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ... parser setup ...

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

**Benefits:**

* Test core logic without subprocess spawning
* Compose tools programmatically
* Reuse functions across tools

**Import example:**

```python
from speaker_detection import load_speaker, compute_b3sum
from speaker_samples import extract_segments
```

### Section Organization

Use comment banners to organize code:

```python
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Core Functions (importable)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
```

## Storage Conventions

### Shared Data Directory

Tools that share data use a common directory specified by environment variable:

```
$SPEAKERS_EMBEDDINGS_DIR/           # Default: ~/.config/speakers_embeddings
├── config.json                     # Global settings (optional)
├── db/                             # Structured data (JSON per entity)
│   ├── alice.json
│   └── bob.json
├── embeddings/                     # Binary/opaque data by entity
│   ├── alice/
│   └── bob/
└── samples/                        # Extracted artifacts by entity
    ├── alice/
    │   ├── sample-001.mp3
    │   └── sample-001.meta.yaml    # Sidecar metadata
    └── bob/
```

**Principles:**

* **One JSON file per entity** - Easy to inspect, `jq`-queryable, git-diffable
* **Sidecar metadata** - `file.ext` + `file.meta.yaml` for provenance
* **Content-addressable where possible** - Use hashes (b3sum/sha256) as identifiers

### JSON Storage Schema

```python
SCHEMA_VERSION = 1  # Bump on breaking changes

def create_entity(id: str, ...) -> dict:
    """Create entity with standard fields."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": id,
        "version": SCHEMA_VERSION,
        # ... entity-specific fields ...
        "created_at": now,
        "updated_at": now,
    }
```

### YAML Sidecar Metadata

For artifacts (audio, images), store provenance in `.meta.yaml`:

```yaml
version: 2
artifact_id: sample-001
b3sum: abc123...              # Content hash for verification

source:
  file: /path/to/source.mp3
  b3sum: xyz789...            # Source content hash

extraction:
  tool: tool_name
  tool_version: 1.0.0
  extracted_at: 2026-01-16T10:30:00Z
```

## Output Conventions

### Exit Codes

```python
return 0   # Success
return 1   # Error (general)
return 2   # Usage error (bad arguments)
```

### Output Streams

```python
# Results go to stdout (for piping)
print(json.dumps(result, indent=2))

# Status/progress goes to stderr (for humans)
print("Processing...", file=sys.stderr)

# Errors go to stderr
print(f"Error: {message}", file=sys.stderr)
```

### Output Formats

Support multiple formats via `--format`:

```python
if args.format == "json":
    print(json.dumps(data, indent=2))
elif args.format == "table":
    print(format_table(data))
elif args.format == "ids":
    print("\n".join(item["id"] for item in data))
```

## Environment Variables

### CRITICAL: Always Check Env Vars First

**Every script MUST check environment variables before using defaults.** This enables test isolation:

```python
# CORRECT - env var first, then default
DEFAULT_DIR = os.path.expanduser("~/.config/tool_name")

def get_data_dir() -> Path:
    return Path(os.environ.get("TOOL_NAME_DIR", DEFAULT_DIR))

# INCORRECT - hardcoded (breaks test isolation!)
def get_data_dir() -> Path:
    return Path.home() / ".config" / "tool_name"  # BAD!
```

**Why this matters:**

* Tests set `SPEAKERS_EMBEDDINGS_DIR=/tmp/test_$$` for isolation
* CI/CD can point to ephemeral directories
* Multiple test runs don't interfere

### Naming Convention

```
TOOL_NAME_SETTING        # Tool-specific
SPEAKERS_EMBEDDINGS_DIR  # Shared across speaker tools
SPEAKER_DETECTION_DEBUG  # Debug flags
```

### Standard Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SPEAKERS_EMBEDDINGS_DIR` | All speaker data | `~/.config/speakers_embeddings` |
| `SPEAKER_DETECTION_BACKEND` | Default backend | `speechmatics` |
| `SPEECHMATICS_API_KEY` | API access | (none) |

## Error Handling

### User-Facing Errors

```python
if not path.exists():
    print(f"Error: File not found: {path}", file=sys.stderr)
    return 1
```

### Validation

```python
def validate_id(id: str) -> bool:
    """IDs: lowercase alphanumeric with hyphens/underscores."""
    import re
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", id))
```

## Testing

**See [evals/TESTING.md](evals/TESTING.md) for comprehensive testing documentation.**

### Key Principles

1. **Reproducible audio** - Use espeak-ng for synthetic voices
2. **Isolated directories** - Always use env vars, never hardcode paths
3. **Docker-first** - All tests runnable in `evals/Dockerfile.test`

### Test Directory Structure

```
evals/
├── TESTING.md            # Comprehensive testing docs
├── Dockerfile.test       # Reproducible test environment
└── tool_name/
    ├── Makefile          # Audio generation
    ├── audio/            # Generated audio (gitignored)
    ├── samples/          # Reference transcripts
    ├── test_cli.py       # CLI integration tests
    ├── benchmark.py      # Performance benchmarks
    └── test_all.sh       # Run all tests
```

### Running Tests

```bash
# Local (requires espeak-ng, ffmpeg)
cd evals/speaker_detection
make all           # Generate test audio
./test_all.sh      # Run tests

# Docker (reproducible)
docker build -f evals/Dockerfile.test -t speaker-tools-test .
docker run --rm speaker-tools-test
```

### Import for Testing

```python
# In test_cli.py - set env BEFORE imports!
import os
import tempfile
TEST_DIR = tempfile.mkdtemp(prefix="test_")
os.environ["SPEAKERS_EMBEDDINGS_DIR"] = TEST_DIR

# Now import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from speaker_detection import load_speaker, compute_b3sum
```

## Documentation

### Docstring at Top

Every tool has a module docstring with:

* One-line description
* Usage examples
* Environment variables

### README Per Tool

```
tool_name.README.md
```

### Architecture in Ramblings

```
ramblings/YYYY-MM-DD--topic-name.md
```

Use Mermaid diagrams for visual documentation.

## Hash Functions

Prefer blake3 (`b3sum`) for content hashing:

```python
def compute_b3sum(file_path: Path) -> str:
    """Compute blake3 hash, fallback to sha256."""
    try:
        result = subprocess.run(
            ["b3sum", "--no-names", str(file_path)],
            capture_output=True, check=True, text=True,
        )
        return result.stdout.strip()[:32]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to sha256
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:32]
```

**Why blake3?**

* Fast (SIMD-optimized)
* Content-addressable like git
* `b3sum` CLI widely available

## Composability Examples

### Pipe JSON Between Tools

```bash
./speaker_samples segments -t transcript.json -l S1 | \
    jq -r '.start, .end' | \
    xargs -n2 ./process_segment
```

### Use as Library

```python
from speaker_detection import load_speaker, get_samples_by_source_audio
from speaker_samples import extract_segment

profile = load_speaker("alice")
samples = get_samples_by_source_audio("alice", audio_b3sum)
```

### Chain Commands

```bash
# Extract → Review → Enroll
./speaker_samples extract audio.mp3 -t transcript.json -l S1 -s alice
./speaker_samples review alice sample-001 --approve
./speaker_detection enroll alice audio.mp3 -t transcript.json -l S1
```

## Checklist for New Tools

- [ ] No `.py` extension, has shebang `#!/usr/bin/env python3`
- [ ] `chmod +x` applied
- [ ] Module docstring with usage and env vars
- [ ] Core functions separate from CLI handlers
- [ ] Works as importable library
- [ ] Uses shared storage conventions if applicable
- [ ] Outputs JSON to stdout, status to stderr
- [ ] Returns proper exit codes
- [ ] Has `{tool_name}.README.md`
- [ ] Tests in `evals/{tool_name}/`
