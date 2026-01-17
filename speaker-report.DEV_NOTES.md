# speaker-report Developer Notes

Developer documentation for the `speaker-report` tool - a quality metrics and recommendations system for the speaker detection pipeline.

## 1. Architecture Decisions

### Read-Only Design

`speaker-report` is intentionally read-only and does not modify any data:

* **Safe to run anytime**: No risk of corrupting state or losing data
* **Idempotent**: Running multiple times produces same results
* **Parallelizable**: Can run alongside other tools without conflicts
* **Audit-friendly**: Pure observation, no side effects

This differs from other speaker-* tools that modify state (catalog, assign, review).

### Aggregation Pattern

The tool follows a simple three-phase pattern:

1. **Load**: Read all relevant YAML files from multiple directories
2. **Compute**: Aggregate, filter, and derive metrics
3. **Format**: Produce human-readable or JSON output

```python
def cmd_status(args):
    # Phase 1: Load
    entries = load_all_catalog_entries()
    profiles = load_all_speaker_profiles()
    assignments = load_all_assignments()

    # Phase 2: Compute
    stats = compute_system_stats(entries, profiles, assignments)

    # Phase 3: Format
    print(format_status_report(stats, args.format))
```

### Recommendation Engine

Recommendations are generated from computed statistics, not hardcoded rules. This makes the system extensible:

```python
def generate_recommendations(stats, entries, profiles, assignments):
    recommendations = []

    if stats.low_confidence_count > 0:
        recommendations.append(
            f"{stats.low_confidence_count} recording(s) have low-confidence assignments"
        )

    # Add more rules as needed without changing the core logic
    return recommendations
```

Rules can be easily added or modified without touching the main command logic.

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

This enables zero-setup execution with automatic environment management.

### Data Classes for Type Safety

Data is structured using `@dataclass` for clarity and type hints:

```python
@dataclass
class CatalogEntry:
    b3sum: str
    path: str
    duration_sec: Optional[float]
    status: str
    context_name: Optional[str]
    # ...

@dataclass
class SystemStats:
    total_recordings: int = 0
    processed: int = 0
    # ...
    recommendations: list[str] = field(default_factory=list)
```

This provides:

* Clear documentation of data shape
* IDE autocompletion
* Runtime type checking (with optional tools)
* Default value handling

### Confidence Level Mapping

Confidence is stored as strings (high/medium/low) in YAML but converted to percentages for threshold comparisons:

```python
confidence_map = {"high": 90, "medium": 70, "low": 40, "unassigned": 0}
confidence = confidence_map.get(confidence_str, 0)

if confidence < threshold:
    # Flag as low confidence
```

This mapping is configurable and can be adjusted based on operational needs.

### Default Command Handling

When no subcommand is specified, `status` is used as default:

```python
args = parser.parse_args()

if args.command is None:
    args.command = "status"
    args.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    args.stale_days = 30
    args.func = cmd_status

return args.func(args)
```

This provides a convenient default behavior while maintaining explicit subcommand structure.

## 3. Data Loading Strategy

### Load-All Pattern

Each load function reads all files from its directory:

```python
def load_all_catalog_entries() -> list[CatalogEntry]:
    catalog_dir = get_catalog_dir()
    entries = []

    for entry_file in sorted(catalog_dir.glob("*.yaml")):
        try:
            data = load_yaml(entry_file)
            # Transform to CatalogEntry
            entries.append(CatalogEntry(...))
        except Exception as e:
            print(f"Warning: Failed to load {entry_file}: {e}", file=sys.stderr)

    return entries
```

Key design choices:

* **Sorted iteration**: Ensures consistent ordering across runs
* **Error tolerance**: Bad files are warned but don't stop processing
* **Full load**: All data is loaded into memory for cross-referencing

### Cross-Reference via b3sum

The b3sum serves as the join key between catalog entries and assignments:

```python
# Map b3sum to entry for efficient lookup
entry_map = {e.b3sum: e for e in entries}

# Cross-reference in confidence report
for assignment in assignments:
    entry = entry_map.get(assignment.b3sum)
    # Access entry metadata for assignment
```

### Status Computation

Status is derived at load time, not stored:

```python
def compute_status(entry: dict, b3sum: str) -> str:
    transcriptions = entry.get("transcriptions", [])
    if not transcriptions:
        return "unprocessed"

    assignments_path = get_assignments_dir() / f"{b3sum}.yaml"
    if not assignments_path.exists():
        return "transcribed"

    review = entry.get("review", {})
    review_status = review.get("status", "none")

    if review_status == "complete":
        return "complete"
    elif review_status == "partial":
        return "reviewed"
    else:
        return "assigned"
```

This ensures status always reflects actual state of the filesystem.

## 4. Output Formatting

### Dual Format Support

Every command supports both text and JSON output:

```python
def format_status_report(stats: SystemStats, format_type: str) -> str:
    if format_type == "json":
        return json.dumps({
            "recordings": {...},
            "speakers": {...},
            # Full structured data
        }, indent=2)

    # Text format with human-readable presentation
    lines = []
    lines.append("Speaker Detection System Status")
    lines.append("=" * 32)
    # ...
    return "\n".join(lines)
```

### JSON Structure

JSON output follows a consistent pattern:

* Top-level object with descriptive keys
* Arrays for lists of items
* Nested objects for grouped metrics
* All original data preserved (no lossy transformation)

```json
{
  "recordings": {
    "total": 42,
    "processed": 38,
    "reviewed": 25,
    "pending": 4
  },
  "speakers": {
    "total": 12,
    "high_trust": 8,
    "medium_trust": 3,
    "low_trust": 1
  },
  "recommendations": [
    "4 recordings have low-confidence assignments"
  ]
}
```

### Text Alignment

Text output uses consistent column alignment:

```python
# Fixed-width columns
lines.append(f"{'ID':<20} {'Name':<20} {'Trust':<10} {'Samples':<8}")

# Percentage formatting
processed_pct = (stats.processed / total * 100) if total > 0 else 0
lines.append(f"  - Processed:  {stats.processed} ({processed_pct:.0f}%)")
```

## 5. Testing Approach

### Subprocess-based Testing

Tests invoke the tool as a subprocess, matching real-world usage:

```python
def test_status_command(self):
    result = subprocess.run(
        ["./speaker-report", "status"],
        capture_output=True,
        text=True,
        env={"SPEAKERS_EMBEDDINGS_DIR": str(self.temp_dir)}
    )
    self.assertEqual(result.returncode, 0)
    self.assertIn("Speaker Detection System Status", result.stdout)
```

### Temp Directory Isolation

Each test uses isolated temporary directories:

```python
class TestSpeakerReport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.env = {"SPEAKERS_EMBEDDINGS_DIR": self.temp_dir}

        # Create required directories
        (Path(self.temp_dir) / "catalog").mkdir()
        (Path(self.temp_dir) / "db").mkdir()
        (Path(self.temp_dir) / "assignments").mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
```

### Test Data Fixtures

Helper functions create test data:

```python
def create_test_catalog_entry(temp_dir: Path, b3sum: str, status: str = "assigned"):
    entry_path = temp_dir / "catalog" / f"{b3sum}.yaml"
    entry = {
        "schema_version": 1,
        "recording": {
            "path": f"/test/{b3sum}.wav",
            "b3sum": b3sum,
            "duration_sec": 60.0,
            "discovered_at": "2026-01-01T00:00:00Z",
        },
        "context": {"name": "test-context"},
        "transcriptions": [{"backend": "test"}],
        "updated_at": "2026-01-15T00:00:00Z",
    }
    with open(entry_path, "w") as f:
        yaml.dump(entry, f)
    return entry_path
```

### Test Categories

* **Unit tests**: Individual functions (compute_status, days_since)
* **Integration tests**: Full command flows (status with data)
* **Edge cases**: Empty directories, malformed YAML, missing files
* **JSON output**: Verify JSON structure and parsability

## 6. Future Enhancements

### Trend Analysis

Track metrics over time:

```bash
# Record current metrics
speaker-report status --format json >> metrics-history.jsonl

# Analyze trends
speaker-report trends --days 30
```

Implementation notes:

* Store timestamped snapshots in `$SPEAKERS_EMBEDDINGS_DIR/metrics/`
* Compute deltas between snapshots
* Generate trend graphs (ASCII or external tool integration)

### Custom Rules Engine

User-defined recommendation rules:

```yaml
# rules.yaml
rules:
  - name: "High-priority context"
    condition: "context == 'production' and coverage < 90"
    severity: "warning"
    message: "Production context below 90% coverage"

  - name: "Stale speakers"
    condition: "days_since_update > 60 and trust_level != 'high'"
    severity: "info"
    message: "Speaker {speaker_id} not updated in {days} days"
```

### Export/Import

Snapshot and restore metrics:

```bash
# Export all metrics
speaker-report export --output metrics-2026-01.json

# Compare with previous snapshot
speaker-report diff metrics-2026-01.json
```

### Integration Hooks

Webhook notifications on threshold breaches:

```bash
speaker-report watch --webhook https://example.com/alerts \
    --when "low_confidence_count > 10"
```

### Dashboard Mode

Continuous display with auto-refresh:

```bash
# Watch mode, refresh every 30 seconds
speaker-report status --watch --interval 30
```

Implementation would use:

* Terminal clear/redraw
* Signal handling for clean exit
* Optional curses for richer display

---

*Part of the speaker-* tool ecosystem for managing speaker identification workflows.*
