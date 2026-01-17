# speaker-review Developer Notes

Developer documentation for the `speaker-review` interactive TUI tool for reviewing speaker diarization assignments.

## Overview

`speaker-review` provides an interactive terminal interface for reviewing and validating speaker assignments from diarization results. Reviewers can approve, reject, or skip segments while building up the speaker embeddings database through verified samples.

## TUI Architecture

### Rich Library Foundation

The TUI is built on the [rich](https://rich.readthedocs.io/) library, providing:

* `Console` - Main rendering engine for styled terminal output
* `Panel` - Bordered containers for segment display
* `Table` - Structured summary output
* `Text` - Styled text composition with color/formatting
* `Prompt` - User input collection

Rich was chosen for its simplicity and zero-configuration terminal support. It degrades gracefully and requires no curses/ncurses knowledge.

### Two Rendering Modes

The tool implements two distinct input modes to handle different terminal environments:

#### 1. Raw Terminal Mode (`review_loop`)

```
Lines 683-743
```

* Uses `tty.setraw()` and `termios` for single-character input
* Immediate key response without Enter key
* Screen clearing between renders for clean UI
* Full-featured interactive experience

#### 2. Simple Prompt Mode (`review_loop_simple`)

```
Lines 745-786
```

* Uses `rich.prompt.Prompt.ask()` for line-based input
* Works in non-TTY environments (pipes, CI, etc.)
* No raw terminal manipulation
* Fallback when raw mode fails or `--simple` flag is used

**Mode Selection Logic (lines 887-894):**

```python
if args.simple or not sys.stdin.isatty():
    review_loop_simple(session, console)
else:
    try:
        review_loop(session, console)
    except Exception:
        review_loop_simple(session, console)
```

## Session State Management

### State Location

Session state persists to `~/.cache/speaker-review/session.yaml` (respects `XDG_CACHE_HOME`).

### ReviewSession Dataclass

```python
@dataclass
class ReviewSession:
    recording_b3sum: str       # Recording identifier
    audio_path: str            # Path to audio file
    transcript_path: str       # Path to transcript JSON
    context: Optional[str]     # Context name (e.g., "meeting-2024")
    segments: list[Segment]    # Segments to review
    current_index: int = 0     # Current position
    decisions: dict[int, ReviewDecision] = field(default_factory=dict)
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
```

### State Persistence Strategy

* **Save on every action**: Each approve/reject/skip/context change triggers `save_session()`
* **Incremental progress**: Decisions stored by segment index, allowing resume at any point
* **Segment regeneration**: On resume, segments are rebuilt from assignments file (not stored in session)
* **Timestamp tracking**: Both `started_at` and `updated_at` for audit trail

### Session Lifecycle

1. **New session**: Created from audio file/b3sum or by scanning assignments directory
2. **Active session**: Loaded with `--continue` flag
3. **Clear session**: Explicit `clear` command removes state file

## Audio Playback Strategy

### Player Priority

The tool attempts audio playback with a fallback chain:

```
mpv (primary) -> ffplay (fallback) -> error message
```

### mpv (Preferred)

```python
subprocess.run([
    "mpv",
    "--no-video",       # Audio-only
    "--really-quiet",   # Suppress output
    f"--start={start}", # Seek to segment start
    f"--end={end}",     # Stop at segment end
    audio_path,
])
```

**Why mpv first:**

* Excellent codec support
* Precise seeking
* Minimal output/noise
* Common on Linux systems

### ffplay (Fallback)

```python
subprocess.run([
    "ffplay",
    "-nodisp",          # No video window
    "-autoexit",        # Exit when done
    "-ss", str(start),  # Seek position
    "-t", str(end - start),  # Duration
    audio_path,
], capture_output=True)  # Suppress stderr noise
```

**Why ffplay as fallback:**

* Ships with ffmpeg (very common)
* Supports same formats
* `-autoexit` ensures clean return

### Playback Blocking

Playback is synchronous (blocking). After playback completes, user sees "Press Enter to continue..." to acknowledge before returning to main loop.

## Integration Points

### speaker_samples extract

On approval (line 576-598), the tool calls:

```bash
speaker_samples extract <audio> \
    -t <transcript> \
    -l <speaker_label> \
    -s <speaker_id> \
    --max-segments 1 \
    -q
```

This extracts a verified audio sample and adds it to the embeddings database. The integration:

* Only triggers when valid timing exists (`start > 0` and `end > start`)
* Only triggers when audio file exists
* Gracefully handles missing `speaker_samples` command
* Reports success/failure to user

### speaker_detection update

On edit action (line 641-668), the tool calls:

```bash
speaker_detection update <speaker_id> --name <new_name>
```

This updates speaker profile metadata. Used for:

* Correcting display names
* Adding/updating speaker information
* Gracefully handles missing `speaker_detection` command

### Assignment Files

The tool reads from `~/.config/speakers_embeddings/assignments/<b3sum>.yaml`:

```yaml
mappings:
  A:
    speaker_id: "john-doe"
    confidence: "high"
    signals:
      - type: embedding_match
        score: 0.92
  B:
    speaker_id: null
    confidence: "unassigned"
```

### Catalog Files

Reads from `~/.config/speakers_embeddings/catalog/<b3sum>.yaml` for:

* Audio file path
* Context name
* Recording metadata

## Testing Challenges

### Why TUI Testing is Difficult

1. **Raw terminal mode**: `tty.setraw()` doesn't work in test environments
2. **Input simulation**: Cannot easily mock `getch()` character-by-character input
3. **Screen state**: `console.clear()` and rendering are side-effect heavy
4. **Blocking playback**: Audio playback blocks the event loop

### Current Testing Approach

The tool is largely untested via automated tests. Manual testing is the primary verification method.

### Potential Testing Strategies

1. **Mock Console**: Inject a mock `Console` object that captures output
2. **Test simple mode**: The `review_loop_simple` function is more testable since it uses `Prompt.ask()`
3. **Test actions in isolation**: Each `action_*` function can be tested independently
4. **Integration tests with fixtures**: Pre-built session/assignment YAML files

Example unit test approach for actions:

```python
def test_action_skip():
    session = ReviewSession(
        recording_b3sum="abc123",
        audio_path="/tmp/test.wav",
        transcript_path="/tmp/test.json",
        context=None,
        segments=[Segment(index=0, ...)],
    )
    mock_console = MagicMock()

    action_skip(session, mock_console)

    assert 0 in session.decisions
    assert session.decisions[0].action == "skip"
```

### Test Coverage Gaps

* No tests for segment loading from different transcript formats (AssemblyAI, Speechmatics)
* No tests for b3sum resolution
* No tests for session persistence round-trip

## Future: Textual Upgrade

### Why Textual?

[Textual](https://textual.textualize.io/) is a modern TUI framework (also by the Rich author) that provides:

* **True widget system**: Buttons, inputs, tables, trees as first-class widgets
* **Reactive data binding**: Automatic UI updates when data changes
* **CSS-like styling**: Separate style from logic
* **Async-first**: Built on asyncio for responsive UIs
* **Testing framework**: Built-in testing utilities with `pilot` for simulating input

### Migration Path

1. **Phase 1**: Keep current rich implementation, add Textual as optional dependency
2. **Phase 2**: Build Textual app alongside, feature-flag between them
3. **Phase 3**: Deprecate rich-only mode once Textual version is stable

### Textual Architecture Vision

```python
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button
from textual.containers import Container

class SegmentViewer(Static):
    """Widget to display current segment."""
    def compose(self) -> ComposeResult:
        yield Static(id="speaker-info")
        yield Static(id="text-content")
        yield Static(id="signals")

class SpeakerReviewApp(App):
    BINDINGS = [
        ("a", "approve", "Approve"),
        ("r", "reject", "Reject"),
        ("s", "skip", "Skip"),
        ("p", "play", "Play"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield SegmentViewer()
        yield Footer()

    def action_approve(self):
        # Handle approval with proper async
        pass
```

### Benefits of Textual Upgrade

* **Better testing**: `app.run_test()` and `pilot.press()` for automated testing
* **Keyboard handling**: No manual `tty.setraw()` management
* **Layout system**: Flexible container-based layouts
* **Scrolling**: Native scrollable widgets for long content
* **Mouse support**: Click-based interactions if desired
* **Hot reloading**: CSS changes apply without restart during development

### Estimated Effort

* Initial port: 2-3 days
* Feature parity: 1 week
* Testing coverage: 1-2 days additional

## File Structure

```
speaker-review                    # Main executable (uv script)
speaker-review.DEV_NOTES.md       # This file

~/.cache/speaker-review/
  session.yaml                    # Persistent session state

~/.config/speakers_embeddings/
  assignments/<b3sum>.yaml        # Input: speaker assignments
  catalog/<b3sum>.yaml            # Input: recording metadata
  db/<speaker_id>/                # Output: speaker profiles
```

## Dependencies

```python
# Required
rich>=13.0      # TUI rendering

# Optional but recommended
pyyaml>=6.0     # YAML support (falls back to JSON)

# System (for playback)
mpv             # Preferred audio player
ffplay          # Fallback (part of ffmpeg)

# System (for integration)
speaker_samples     # Sample extraction
speaker_detection   # Profile updates
b3sum              # Blake3 hashing (falls back to SHA256)
```
