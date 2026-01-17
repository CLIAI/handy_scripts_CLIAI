# speaker-catalog Developer Notes

Developer documentation for the `speaker-catalog` tool - a recording inventory and processing state management system.

## 1. Architecture Decisions

### Why YAML for Storage

YAML was chosen as the primary storage format for catalog entries:

* **Human-readable**: Operators can inspect and understand catalog state without tooling
* **Human-editable**: Manual corrections or bulk edits are straightforward with any text editor
* **Git-friendly**: Diffs are meaningful, merge conflicts are resolvable
* **Comment support**: Unlike JSON, YAML allows inline documentation for complex entries
* **PyYAML fallback**: If PyYAML is unavailable, the tool gracefully degrades to JSON format

The storage format is abstracted through `load_yaml()` and `save_yaml()` functions, enabling future format changes without touching command logic.

### Why b3sum as Canonical ID

Blake3 hash (b3sum) serves as the content-addressable identifier for recordings:

* **Content-addressable**: Same audio file always maps to same ID, regardless of filename or path
* **Fast**: Blake3 is significantly faster than SHA256/MD5, important for large audio files
* **Collision-resistant**: 128-bit truncation (first 32 hex chars) provides ample uniqueness
* **Prefix-addressable**: Commands accept b3sum prefixes (6+ chars) for convenience
* **Rename-resilient**: Moving or renaming files does not break catalog references

The tool automatically falls back to SHA256 if `b3sum` command is not available:

```python
def compute_b3sum(file_path: Path) -> str:
    try:
        result = subprocess.run(["b3sum", "--no-names", str(file_path)], ...)
        return result.stdout.strip()[:32]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to SHA256
        import hashlib
        sha256 = hashlib.sha256()
        # ... chunked reading ...
        return sha256.hexdigest()[:32]
```

### Status Computation Logic

Status is **derived from state**, not stored directly. This is a deliberate design choice:

* **Single source of truth**: No risk of status field drifting out of sync with actual state
* **Automatic progression**: Status updates automatically as transcripts/assignments are added
* **Query-time computation**: `compute_status()` evaluates current state on each access

Status progression:

```
unprocessed -> transcribed -> assigned -> reviewed -> complete
```

The `compute_status()` function checks:

1. Are there any transcripts? No -> `unprocessed`
2. Do assignments exist for this b3sum? No -> `transcribed`
3. Is review complete? Yes -> `complete`
4. Is review partial? Yes -> `reviewed`
5. Otherwise -> `assigned`

## 2. Implementation Notes

### Uses `uv run` for Dependencies

The script uses inline script metadata (PEP 723) for dependency management:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
```

This enables zero-setup execution - `uv` automatically creates an isolated environment with PyYAML installed on first run.

### argparse Subcommand Pattern

Commands are organized using argparse subparsers:

```python
subparsers = parser.add_subparsers(dest="command", required=True)

add_parser = subparsers.add_parser("add", help="Add a recording to the catalog")
add_parser.add_argument("audio", help="Path to audio file")
add_parser.set_defaults(func=cmd_add)
```

Each command has a dedicated `cmd_*` function that receives the parsed `argparse.Namespace`.

### Fallback Strategies

The tool implements graceful degradation for optional dependencies:

**b3sum -> SHA256**:

```python
try:
    result = subprocess.run(["b3sum", ...])
except (subprocess.CalledProcessError, FileNotFoundError):
    # Fallback to hashlib.sha256
```

**PyYAML -> JSON**:

```python
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def save_yaml(path, data):
    if YAML_AVAILABLE:
        yaml.dump(data, f, ...)
    else:
        json.dump(data, f, ...)
```

### Audio Duration Detection

Uses `ffprobe` for duration extraction, fails gracefully if unavailable:

```python
def get_audio_duration(file_path: Path) -> Optional[float]:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ], ...)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None
```

## 3. Storage Layout

All data is stored under `$SPEAKERS_EMBEDDINGS_DIR` (default: `~/.config/speakers_embeddings/`):

```
$SPEAKERS_EMBEDDINGS_DIR/
├── catalog/
│   ├── {b3sum1}.yaml      # Catalog entry for recording 1
│   ├── {b3sum2}.yaml      # Catalog entry for recording 2
│   └── ...
├── assignments/
│   ├── {b3sum1}.yaml      # Speaker assignments for recording 1
│   └── ...
└── profiles/              # (managed by other speaker-* tools)
    └── ...
```

### Catalog Entry Schema (v1)

```yaml
schema_version: 1
recording:
  path: /absolute/path/to/audio.wav
  b3sum: abc123def456789012345678901234ab
  duration_sec: 123.45
  discovered_at: "2025-01-15T10:30:00Z"
context:
  name: team-standup
  expected_speakers:
    - speaker_id_1
    - speaker_id_2
  tags:
    - important
    - q4-review
transcriptions:
  - backend: speechmatics
    version: speechmatics-v1
    transcript_path: /path/to/transcript.json
    processed_at: "2025-01-15T11:00:00Z"
    tool_version: "1.0.0"
    speakers_detected: 3
status: transcribed  # computed, not stored
updated_at: "2025-01-15T11:00:00Z"
```

## 4. Schema Evolution Strategy

### Version Field

Every catalog entry includes `schema_version: 1` for future migrations.

### Migration Rules

* **Minor versions**: Add fields, never remove
* **New optional fields**: Code must handle missing fields with defaults
* **Breaking changes**: Bump major version, provide migration script

### Example Migration Pattern

```python
def migrate_v1_to_v2(entry: dict) -> dict:
    """Migrate schema v1 to v2."""
    if entry.get("schema_version", 1) >= 2:
        return entry

    # Add new v2 field with default
    entry["new_field"] = entry.get("new_field", "default_value")
    entry["schema_version"] = 2
    return entry
```

### Backward Compatibility

Code should always check for field existence:

```python
# Good: handles missing field
duration = entry.get("recording", {}).get("duration_sec")

# Bad: assumes field exists
duration = entry["recording"]["duration_sec"]
```

## 5. Testing Approach

### Subprocess-based Testing

Tests invoke the tool as a subprocess, matching real-world usage:

```python
def test_add_command(self):
    result = subprocess.run(
        ["./speaker-catalog", "add", audio_path],
        capture_output=True,
        text=True
    )
    self.assertEqual(result.returncode, 0)
```

### Temp Directory Isolation

Each test run should use isolated temporary directories:

```python
import tempfile
import os

class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["SPEAKERS_EMBEDDINGS_DIR"] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
```

### Test Categories

* **Unit tests**: Individual functions (compute_status, compute_b3sum)
* **Integration tests**: Full command flows (add -> register-transcript -> show)
* **Edge cases**: Missing files, corrupt YAML, concurrent access

### Example Test Structure

```python
class TestAddCommand(unittest.TestCase):
    """Test the 'add' subcommand."""

    def test_add_new_recording(self):
        """Adding a new recording creates catalog entry."""
        pass

    def test_add_duplicate_fails(self):
        """Adding duplicate without --force fails."""
        pass

    def test_add_with_force_overwrites(self):
        """Adding with --force updates existing entry."""
        pass

    def test_add_nonexistent_file_fails(self):
        """Adding non-existent file returns error."""
        pass
```

## 6. Future Enhancements

### Bulk Import from Directory

Scan a directory and add all audio files:

```bash
speaker-catalog import-dir /path/to/recordings --context "project-x" --recursive
```

Implementation notes:

* Use `pathlib.Path.rglob()` for recursive scanning
* Filter by audio extensions (.wav, .mp3, .flac, .m4a, .ogg)
* Skip already-cataloged files (b3sum match)
* Report summary: added, skipped, errors

### Duplicate Detection

Find recordings with identical content:

```bash
speaker-catalog find-duplicates
speaker-catalog dedupe --keep-first
```

Implementation notes:

* Group entries by b3sum (should not happen with current design)
* Check for same-content different-path scenarios
* Option to remove duplicates, keeping specified copy

### Integration with speaker-process

Seamless workflow integration:

```bash
# Process all unprocessed recordings
speaker-catalog list --status unprocessed --format paths | \
  xargs -I{} speaker-process {}

# Auto-register transcripts
speaker-process audio.wav --register-catalog
```

Implementation notes:

* Add `--register-catalog` flag to speaker-process
* speaker-process calls `speaker-catalog register-transcript` on completion
* Status automatically progresses from unprocessed to transcribed

### Additional Enhancement Ideas

* **Export/import**: JSON/YAML dump of entire catalog for backup/migration
* **Statistics**: `speaker-catalog stats` showing processing summary
* **Webhooks**: Notify external systems on status changes
* **Locking**: File locking for concurrent access safety
* **Archive**: Move completed recordings to archive storage

---

*Part of the speaker-* tool ecosystem for managing speaker identification workflows.*
