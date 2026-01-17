# speaker-llm Developer Notes

Developer documentation for the `speaker-llm` tool - LLM-based speaker name detection from transcripts.

## 1. Architecture Overview

### Component Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         speaker-llm                                 │
├─────────────────┬──────────────────┬──────────────────────────────┤
│  Transcript     │   LLM Provider   │   Output                      │
│  Parser         │   Abstraction    │   Generator                   │
│                 │                  │                               │
│  - AssemblyAI   │  - Anthropic     │  - JSON detections            │
│  - Speechmatics │  - OpenAI        │  - Quick names map            │
│                 │  - Ollama        │  - Human-readable             │
└────────┬────────┴────────┬─────────┴──────────────┬───────────────┘
         │                 │                        │
         v                 v                        v
    ┌─────────┐     ┌─────────────┐          ┌─────────────┐
    │Transcript│     │  Response   │          │  speaker-   │
    │  .json   │     │   Cache     │          │  assign     │
    └─────────┘     │ ~/.cache/   │          │  (consumer) │
                    │ speaker-llm │          └─────────────┘
                    └─────────────┘
```

### Data Flow

```
┌──────────────┐     ┌────────────────┐     ┌────────────────┐
│  Transcript  │────>│  Extract       │────>│  Build         │
│  JSON File   │     │  Conversation  │     │  LLM Prompt    │
└──────────────┘     └────────────────┘     └───────┬────────┘
                                                    │
                     ┌────────────────┐             │
                     │  Check Cache   │<────────────┘
                     │  (hash-based)  │
                     └───────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Cache Hit?                   │
              │                              │
         Yes  │                              │  No
              v                              v
    ┌──────────────┐              ┌──────────────────┐
    │ Return       │              │ Call LLM Provider│
    │ Cached       │              │ (Anthropic/      │
    │ Response     │              │  OpenAI/Ollama)  │
    └──────────────┘              └────────┬─────────┘
                                           │
                                           v
                                  ┌──────────────────┐
                                  │ Parse LLM        │
                                  │ Response JSON    │
                                  └────────┬─────────┘
                                           │
                                           v
                                  ┌──────────────────┐
                                  │ Save to Cache    │
                                  │ Return Result    │
                                  └──────────────────┘
```

## 2. LLM Provider Abstraction

### Provider Interface

```python
class LLMProvider:
    """Base class for LLM providers."""
    name: str = "base"

    def is_available(self) -> bool:
        """Check if provider is configured and accessible."""
        raise NotImplementedError

    def generate(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Generate response from LLM."""
        raise NotImplementedError
```

### Provider Selection Logic

```python
def get_available_provider() -> Optional[LLMProvider]:
    """Priority order: Anthropic > OpenAI > Ollama"""
    providers = [
        AnthropicProvider(),  # Check ANTHROPIC_API_KEY
        OpenAIProvider(),     # Check OPENAI_API_KEY
        OllamaProvider(),     # Check OLLAMA_HOST or localhost
    ]

    for provider in providers:
        if provider.is_available():
            return provider
    return None
```

### Design Rationale

**Priority Order:**

1. **Anthropic first** - Best quality for conversation analysis, Claude excels at nuanced text understanding
2. **OpenAI second** - Widely available, good fallback
3. **Ollama last** - Local/private option, no API costs, but may have lower accuracy

**Availability Checks:**

* Anthropic/OpenAI: Check for environment variable presence
* Ollama: Actually ping the server (may be running without explicit config)

## 3. Prompt Engineering

### Analysis Prompt Structure

```
┌─────────────────────────────────────────────────┐
│  1. Task Description                            │
│     "Analyze this conversation transcript..."   │
├─────────────────────────────────────────────────┤
│  2. Speaker Labels to Identify                  │
│     "SPEAKERS TO IDENTIFY: S1, S2, S3"          │
├─────────────────────────────────────────────────┤
│  3. Detection Patterns                          │
│     - Direct address patterns                   │
│     - Self-reference patterns                   │
│     - Third-person mentions                     │
│     - Introduction patterns                     │
├─────────────────────────────────────────────────┤
│  4. Conversation Text                           │
│     "[S1]: Hello, I'm Alice..."                 │
│     "[S2]: Hi Alice, Bob here..."               │
├─────────────────────────────────────────────────┤
│  5. Output Format Instructions                  │
│     JSON schema with examples                   │
└─────────────────────────────────────────────────┘
```

### Why These Patterns?

| Pattern | Reliability | Notes |
|---------|-------------|-------|
| Self-introduction | High | "I'm Alice" is very reliable |
| Direct address | Medium-High | "Thanks, Bob" often reveals names |
| Third-person | Medium | "As Carol said" may be ambiguous |
| Role-based | Medium | "The host John" requires context |
| Conversation flow | Low-Medium | Inferred from reply patterns |

### Confidence Score Guidelines

Prompted to LLM:

* **0.9-1.0**: Explicit self-introduction or multiple confirmations
* **0.7-0.9**: Clear direct address or third-person reference
* **0.5-0.7**: Single mention or inferred from context
* **0.3-0.5**: Weak evidence, possibly ambiguous
* **<0.3**: Speculative, should be marked as unconfirmed

## 4. Caching Strategy

### Cache Key Generation

```python
def compute_transcript_hash(transcript_text: str) -> str:
    """SHA256 hash truncated to 16 chars."""
    return hashlib.sha256(transcript_text.encode()).hexdigest()[:16]
```

### Cache Separation

* `analyze` mode: `{hash}.json`
* `detect-names` mode: `{hash}_quick.json`

This prevents quick mode from returning full analysis results and vice versa.

### Cache Invalidation Scenarios

| Scenario | Cached? | Reason |
|----------|---------|--------|
| Same transcript, same mode | Yes | Identical analysis |
| Same transcript, different mode | No | Different prompts |
| Same transcript, different provider | Yes | Same content analysis |
| Modified transcript | No | Hash changes |

### Why Cache by Content Hash?

* **Not file path**: Same content in different locations should hit cache
* **Not modification time**: Re-saved files with same content should hit cache
* **Content-based**: Ensures semantic equivalence

## 5. Error Handling

### Graceful Degradation

```python
try:
    response = provider.generate(prompt, model)
    parsed = parse_llm_response(response.content)
except Exception as e:
    return {
        "detections": [],
        "model": model,
        "processed_at": utc_now_iso(),
        "error": str(e),
    }
```

### Error Categories

| Error Type | Handling | User Message |
|------------|----------|--------------|
| No provider available | Exit with instructions | "Set ANTHROPIC_API_KEY or..." |
| API key invalid | Return error in result | "Authentication failed" |
| API rate limit | Return error (no retry) | "Rate limit exceeded" |
| JSON parse failure | Return empty detections | "Failed to parse LLM response" |
| Timeout (Ollama) | Return error | "Request timed out" |

### Why No Automatic Retry?

* API costs: Retrying failed requests adds cost
* User control: Let user decide to retry with `--no-cache`
* Transparency: Error is visible, not hidden by retry

## 6. Integration with speaker-assign

### Expected Input Format (from speaker-assign)

`speaker-assign` calls `speaker-llm` as a subprocess:

```bash
speaker-llm analyze transcript.json --format json
```

### Required Output Format

```json
{
  "detections": [
    {
      "speaker_label": "S1",
      "detected_name": "alice",
      "confidence": 0.85,
      "evidence": ["Quote 1", "Quote 2"]
    }
  ]
}
```

### Signal Integration in speaker-assign

```python
def collect_llm_signals(speaker_label, transcript_path):
    """From speaker-assign: calls speaker-llm."""
    result = subprocess.run(
        ["speaker-llm", "analyze", str(transcript_path)],
        capture_output=True,
        text=True,
    )
    analysis = json.loads(result.stdout)

    for detection in analysis.get("detections", []):
        if detection.get("speaker_label") == speaker_label:
            signals.append(Signal(
                type="llm_name_detection",
                speaker_id=detection.get("detected_name").lower().replace(" ", "-"),
                score=detection.get("confidence", 0.5),
                evidence=detection.get("evidence", []),
            ))
```

## 7. Testing Approach

### Unit Tests with Mocked Providers

```python
class MockProvider(LLMProvider):
    """Test provider that returns predefined responses."""
    name = "mock"

    def __init__(self, response: str):
        self.response = response

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, model: str = None) -> LLMResponse:
        return LLMResponse(
            content=self.response,
            model="mock-model",
            provider="mock",
        )
```

### Test Categories

1. **Transcript Parsing**

   * AssemblyAI format extraction
   * Speechmatics format extraction
   * Empty transcript handling

2. **Provider Selection**

   * Priority order verification
   * Fallback when primary unavailable
   * Explicit provider override

3. **Response Parsing**

   * Valid JSON extraction
   * Markdown code block handling
   * Malformed response handling

4. **Caching**

   * Cache hit verification
   * Cache miss verification
   * Cache bypass with --no-cache

5. **Integration**

   * Output format compatibility with speaker-assign
   * CLI argument handling

### Mock Response Examples

```python
MOCK_ANALYSIS_RESPONSE = '''
{
    "detections": [
        {
            "speaker_label": "S1",
            "detected_name": "Alice",
            "confidence": 0.85,
            "evidence": ["Hi everyone, this is Alice speaking"]
        }
    ],
    "notes": "Clear self-introduction detected"
}
'''

MOCK_QUICK_RESPONSE = '''
{
    "names": {
        "S1": "Alice",
        "S2": "Bob"
    }
}
'''
```

## 8. Future Enhancements

### Multi-Turn Conversation Analysis

Currently processes entire transcript at once. Future version could:

* Analyze conversation in chunks for long transcripts
* Track name mentions across segments
* Build progressive confidence as more evidence accumulates

### Speaker Profile Integration

```bash
# Future: cross-reference with known speakers
speaker-llm analyze transcript.json --profiles-dir ~/.speakers/
```

Would allow:

* Matching detected names to known speaker profiles
* Suggesting corrections based on profile metadata
* Higher confidence when names match enrolled speakers

### Confidence Calibration

Track historical accuracy to calibrate confidence scores:

```python
# Future: adjust confidence based on provider accuracy
calibrated_confidence = raw_confidence * provider_accuracy_factor
```

### Multi-Provider Consensus

Use multiple providers and combine results:

```bash
# Future: consensus mode
speaker-llm analyze transcript.json --consensus
```

Would call multiple providers and:

* Increase confidence when providers agree
* Flag disagreements for manual review
* Reduce single-provider bias

### Streaming Support

For real-time transcription scenarios:

```bash
# Future: streaming mode
speaker-llm stream --input-pipe
```

Would process utterances as they arrive and update detections incrementally.

## 9. Performance Considerations

### API Costs

| Provider | Model | Approx Cost per 1K tokens |
|----------|-------|---------------------------|
| Anthropic | claude-3-haiku | $0.00025 input, $0.00125 output |
| Anthropic | claude-3-sonnet | $0.003 input, $0.015 output |
| OpenAI | gpt-4o-mini | $0.00015 input, $0.0006 output |
| OpenAI | gpt-4o | $0.005 input, $0.015 output |
| Ollama | any | Free (local compute) |

### Optimization Strategies

1. **Use caching aggressively** - Default enabled
2. **Use quick mode for speed** - `detect-names` uses simpler prompt
3. **Use haiku/mini models** - Cheaper, often sufficient
4. **Local Ollama for bulk** - No API costs

### Latency Expectations

| Provider | Model | Typical Latency |
|----------|-------|-----------------|
| Anthropic | haiku | 1-3 seconds |
| Anthropic | sonnet | 3-8 seconds |
| OpenAI | gpt-4o-mini | 1-3 seconds |
| Ollama | llama3.2 | 5-15 seconds (depends on hardware) |

---

*Part of the speaker-* tool ecosystem for managing speaker identification workflows.*
