# speaker-process Developer Notes

Developer documentation for the `speaker-process` tool - a batch recording processing orchestrator.

## 1. Architecture Decisions

### Why Orchestrator Pattern

`speaker-process` follows the orchestrator pattern rather than a monolithic approach:

* **Single Responsibility**: Each tool (speaker-catalog, stt_*, speaker-assign) does one thing well
* **Composability**: Users can run individual tools when needed
* **Testability**: Each component can be tested independently
* **Flexibility**: New STT backends or processing steps can be added without modifying core logic
* **Error Isolation**: Failure in one step doesn't corrupt other components

The orchestrator coordinates these tools while maintaining their independence:

```
speaker-process
    ├── speaker-catalog add
    ├── stt_speechmatics.py / stt_assemblyai.py
    ├── speaker-catalog register-transcript
    └── speaker-assign
```

### Why Queue-Based Processing

The queue mechanism provides several benefits:

* **Resumability**: Processing can be interrupted and resumed
* **Parallel Execution**: Multiple recordings can be processed concurrently
* **Status Tracking**: Failed/completed items are tracked for debugging
* **Decoupling**: Queueing and processing can happen at different times
* **Batch Operations**: Large directories can be queued incrementally

The queue is persisted as YAML for human inspection and manual manipulation:

```yaml
items:
  - audio_path: /path/to/file.mp3
    b3sum: abc123...
    status: pending
    backends: [speechmatics, assemblyai]
```

### Why Thread-Based Parallelism

Python's `concurrent.futures.ThreadPoolExecutor` is used for parallel processing:

* **I/O Bound**: STT operations are primarily I/O bound (API calls)
* **Simple**: ThreadPoolExecutor provides a clean interface
* **GIL Tolerance**: External subprocess calls release the GIL
* **Resource Efficient**: Threads share memory, suitable for file operations

For CPU-bound processing, a `ProcessPoolExecutor` alternative could be added.

## 2. Implementation Notes

### Tool Discovery

Tools are discovered in two locations:

1. System PATH (using `shutil.which`)
2. Relative to the script (same directory)

```python
def find_tool(name: str) -> Optional[Path]:
    """Find a tool in PATH or relative to this script."""
    from shutil import which
    path_tool = which(name)
    if path_tool:
        return Path(path_tool)

    script_dir = Path(__file__).parent.resolve()
    relative_tool = script_dir / name
    if relative_tool.exists() and os.access(relative_tool, os.X_OK):
        return relative_tool

    return None
```

This allows the tool to work both when installed system-wide and when running from the repository.

### Backend Command Mapping

STT backends are mapped to their command-line tools:

```python
def get_stt_command(backend: str) -> Optional[str]:
    commands = {
        "speechmatics": "stt_speechmatics.py",
        "assemblyai": "stt_assemblyai.py",
        "openai": "stt_openai.py",
        "whisper": "stt_openai_OR_local_whisper_cli.py",
    }
    return commands.get(backend)
```

New backends can be added by extending this mapping.

### Thread-Safe Queue Operations

The queue uses a threading lock to prevent race conditions during parallel processing:

```python
class ProcessingQueue:
    def __init__(self):
        self._lock = threading.Lock()

    def update_status(self, b3sum: str, status: str, ...):
        with self._lock:
            if b3sum in self.items:
                item = self.items[b3sum]
                item.status = status
                self._save()
```

### Error Handling Strategy

Processing failures are handled gracefully:

* **Individual Step Failures**: Logged but don't stop processing
* **Transcript Failures**: Other backends continue
* **Assignment Failures**: Recording is still marked with transcripts
* **Queue Persistence**: Status is updated even on failures

```python
def process_single(...) -> ProcessResult:
    result = ProcessResult(success=True)

    # Step 1: Catalog
    if ensure_in_catalog(...):
        result.steps_completed.append("catalog_add")
    else:
        result.steps_failed.append("catalog_add")

    # Continue with other steps...
    result.success = len(result.transcripts) > 0
    return result
```

## 3. Queue Schema

### Queue File Structure

```yaml
schema_version: 1
updated_at: "2026-01-17T10:00:00Z"

items:
  - audio_path: /absolute/path/to/audio.mp3
    b3sum: abc123def456789012345678901234ab
    status: pending              # pending | processing | completed | failed | skipped
    context: team-standup        # optional context name
    backends:                    # STT backends to use
      - speechmatics
      - assemblyai
    queued_at: "2026-01-17T09:00:00Z"
    started_at: null             # set when processing starts
    completed_at: null           # set when processing ends
    error: null                  # error message if failed
    results:                     # processing results
      transcripts:
        speechmatics: /path/to/audio.speechmatics.json
        assemblyai: /path/to/audio.assemblyai.json
```

### Status Transitions

```
pending ──> processing ──> completed
                      └──> failed
                      └──> skipped
```

* `pending`: Initial state when queued
* `processing`: Being actively processed
* `completed`: All steps succeeded (at least one transcript)
* `failed`: Processing failed (no transcripts or critical error)
* `skipped`: Explicitly skipped (e.g., already processed)

## 4. Processing Pipeline Details

### Step 1: Catalog Registration

```python
def ensure_in_catalog(audio_path, context, dry_run, quiet):
    # Check if already in catalog
    result = subprocess.run(
        ["speaker-catalog", "status", str(audio_path)],
        capture_output=True
    )
    if result.returncode == 0:
        return True  # Already in catalog

    # Add to catalog
    cmd = ["speaker-catalog", "add", str(audio_path), "--quiet"]
    if context:
        cmd.extend(["--context", context])
    return subprocess.run(cmd).returncode == 0
```

### Step 2: Transcription

Each backend is invoked independently:

```python
def transcribe_with_backend(audio_path, backend, output_dir, ...):
    stt_tool = find_tool(get_stt_command(backend))
    output_path = output_dir / f"{audio_path.stem}.{backend}.json"

    if output_path.exists():
        return output_path  # Already transcribed

    cmd = [str(stt_tool), str(audio_path), "-o", str(output_path)]
    result = subprocess.run(cmd, capture_output=True)
    return output_path if result.returncode == 0 else None
```

### Step 3: Transcript Registration

```python
def register_transcript_in_catalog(audio_path, transcript_path, backend, ...):
    cmd = [
        "speaker-catalog", "register-transcript", str(audio_path),
        "--backend", backend,
        "--transcript", str(transcript_path),
        "--quiet",
    ]
    return subprocess.run(cmd).returncode == 0
```

### Step 4: Speaker Assignment

Uses the first successful transcript:

```python
def run_speaker_assign(audio_path, transcript_path, context, ...):
    cmd = [
        "speaker-assign", "assign", str(audio_path),
        "--transcript", str(transcript_path),
        "--use-embeddings",
    ]
    if context:
        cmd.extend(["--context", context])
    return subprocess.run(cmd).returncode == 0
```

## 5. Testing Approach

### Unit Test Structure

Tests should cover:

* **Queue Operations**: add, get_pending, update_status, clear
* **Tool Discovery**: find_tool with various configurations
* **Audio Detection**: is_audio_file, find_audio_files
* **Processing Logic**: process_single with mocked tools

### Integration Test Structure

```python
def test_full_pipeline(temp_dir):
    """Test complete processing pipeline."""
    # Create test audio
    audio_path = create_test_audio(temp_dir)

    # Create mock STT tools
    create_mock_stt_tool(temp_dir, "speechmatics")

    # Run processing
    result = subprocess.run([
        "speaker-process", "process", str(audio_path),
        "--backend", "speechmatics"
    ], env={"SPEAKERS_EMBEDDINGS_DIR": str(temp_dir)})

    # Verify results
    assert result.returncode == 0
    assert (temp_dir / "catalog").exists()
```

### Mock Tool Creation

For testing without real STT services:

```python
def create_mock_stt_tool(temp_dir, backend):
    """Create a mock STT tool that produces valid output."""
    tool_path = temp_dir / f"stt_{backend}.py"
    tool_path.write_text('''#!/usr/bin/env python3
import json
import sys
output = {"utterances": [{"speaker": "A", "text": "Hello"}]}
with open(sys.argv[3], "w") as f:
    json.dump(output, f)
''')
    tool_path.chmod(0o755)
```

## 6. Future Enhancements

### Progress Reporting

Add real-time progress reporting:

```bash
speaker-process run --progress
# Processing: 5/20 (25%)
# Current: meeting_2026_01_15.mp3
# Elapsed: 00:15:30
# ETA: 00:45:00
```

Implementation notes:

* Use `tqdm` or custom progress bar
* Track bytes/files processed
* Estimate completion time from historical data

### Webhook Notifications

Notify external systems on completion:

```bash
speaker-process run --webhook "https://api.example.com/processing-complete"
```

Implementation notes:

* POST JSON payload with processing results
* Support authentication headers
* Retry on failure with exponential backoff

### Priority Queue

Support priority ordering:

```bash
speaker-process queue urgent.mp3 --priority high
speaker-process queue batch/ --priority low
```

Implementation notes:

* Add priority field to queue items
* Sort by priority, then by queued_at
* Allow priority modification

### Backend Plugins

Dynamic backend loading:

```python
# In ~/.config/speakers_embeddings/backends.yaml
backends:
  custom_stt:
    command: /usr/local/bin/custom-stt
    args: ["--format", "json", "--output", "{output}"]
    output_pattern: "{stem}.custom.json"
```

### Retry Mechanism

Automatic retry for transient failures:

```bash
speaker-process run --retry 3 --retry-delay 60
```

Implementation notes:

* Distinguish transient vs permanent failures
* Exponential backoff between retries
* Track retry count in queue item

### Distributed Processing

Support distributed worker nodes:

```bash
# Controller
speaker-process controller --port 8765

# Workers
speaker-process worker --controller localhost:8765
```

Implementation notes:

* Use Redis or RabbitMQ for queue distribution
* Worker heartbeat monitoring
* Load balancing across workers

### Resource Limits

Control resource usage:

```bash
speaker-process run --max-memory 8G --max-cpu 4
```

Implementation notes:

* Monitor subprocess resource usage
* Pause/throttle when limits approached
* Report resource statistics

---

*Part of the speaker-* tool ecosystem for managing speaker identification workflows.*
