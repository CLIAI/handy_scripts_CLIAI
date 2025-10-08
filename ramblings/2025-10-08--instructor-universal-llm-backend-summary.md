# Instructor as Universal LLM Backend - Summary

**Date:** 2025-10-08
**Purpose:** Executive summary of Instructor's universal LLM compatibility
**Related:** `2025-10-07--llm-cli-tools-research-for-python-integration.md`

---

## TL;DR - YES! ✅

**Instructor can talk to ANY LLM backend via:**

1. **15+ official providers** (direct support)
2. **100+ providers via LiteLLM** (single integration)
3. **200+ models via OpenRouter** (single API)
4. **ANY OpenAI-compatible endpoint** (custom backends)

**This means:** We can simplify the architecture to **Instructor-only** and let users choose backend via configuration!

---

## Simplified Architecture

### Before (Original Design)
```
stt_assemblyai_speaker_mapper.py
    ├── ModsBackend (subprocess)
    ├── OllamaBackend (subprocess)
    └── InstructorBackend (API)
```

### After (Unified Design)
```
stt_assemblyai_speaker_mapper.py
    └── Instructor (universal)
            ├── Ollama (local)
            ├── OpenAI (cloud)
            ├── Anthropic (cloud)
            ├── Google Gemini (cloud)
            ├── Groq (cloud - fast)
            ├── LiteLLM (100+ providers)
            ├── OpenRouter (200+ models)
            └── Custom endpoints
```

**Benefits:**
* ✅ Single API surface
* ✅ Type-safe Pydantic models everywhere
* ✅ Automatic validation + retries
* ✅ No subprocess complexity
* ✅ Unified error handling

---

## Provider Syntax

### Format
```bash
--llm-detect PROVIDER/MODEL
```

### Examples

**Cloud providers:**
```bash
--llm-detect openai/gpt-4o-mini
--llm-detect anthropic/claude-3-5-sonnet-20241022
--llm-detect google/gemini-2.0-flash-exp
--llm-detect groq/llama-3.1-70b-versatile
--llm-detect mistral/mistral-large-latest
```

**Local Ollama:**
```bash
--llm-detect ollama/llama3.2
--llm-detect ollama/mistral:7b
--llm-detect ollama/qwen2.5
```

**Remote Ollama:**
```bash
--llm-detect ollama/llama3.2 --llm-endpoint http://server:11434
```

**Via LiteLLM (100+ providers):**
```bash
--llm-detect litellm/gpt-3.5-turbo
--llm-detect litellm/claude-3-opus-20240229
--llm-detect litellm/gemini/gemini-pro
--llm-detect litellm/bedrock/anthropic.claude-v2
--llm-detect litellm/azure/gpt-4
--llm-detect litellm/ollama/llama2
```

**Via OpenRouter (200+ models):**
```bash
--llm-detect openrouter/google/gemini-2.0-flash-lite-001
--llm-detect openrouter/anthropic/claude-3-opus
--llm-detect openrouter/meta-llama/llama-3-70b-instruct
```

**Custom OpenAI-compatible endpoint:**
```bash
--llm-detect custom/model-name --llm-endpoint http://localhost:8080/v1
```

---

## Code Simplification

### Unified Implementation

```python
import instructor
from pydantic import BaseModel, Field

class SpeakerDetection(BaseModel):
    speakers: dict[str, str]  # {"A": "Alice Anderson", "B": "Bob Smith"}
    confidence: str = Field(description="low, medium, or high")
    reasoning: str = Field(description="Brief explanation")

# Single unified function for ALL backends
def detect_speakers_llm(
    provider_model: str,  # e.g., "openai/gpt-4o-mini"
    transcript_text: str,
    detected_labels: list[str],
    endpoint: str | None = None  # For custom endpoints
) -> SpeakerDetection:
    """Detect speakers using any LLM via Instructor"""

    # Create client (Instructor handles provider routing)
    if endpoint:
        # Custom endpoint
        from openai import OpenAI
        client = instructor.from_openai(
            OpenAI(base_url=endpoint, api_key="none"),
            mode=instructor.Mode.JSON
        )
        model = provider_model.split("/")[1]
    else:
        # Standard provider
        client = instructor.from_provider(provider_model)
        model = None  # Instructor auto-extracts from provider_model

    # Build prompt
    prompt = f"""Analyze this transcript and identify speakers.

Detected speaker labels: {', '.join(detected_labels)}

Based on the conversation, suggest names or roles for each speaker.

Transcript:
{transcript_text}
"""

    # Call LLM with structured output
    result = client.chat.completions.create(
        model=model,  # None if using from_provider (auto-handled)
        messages=[{"role": "user", "content": prompt}],
        response_model=SpeakerDetection,
        max_retries=3  # Automatic retry on validation errors
    )

    return result
```

That's it! **No backend-specific code needed.**

---

## Configuration File (Optional)

```yaml
# llm_config.yml
llm:
  # Default backend
  default: "openai/gpt-4o-mini"

  # Provider API keys (or use environment variables)
  api_keys:
    OPENAI_API_KEY: "sk-..."
    ANTHROPIC_API_KEY: "sk-ant-..."
    GROQ_API_KEY: "gsk_..."

  # Fallback chain
  fallback_chain:
    - "groq/llama-3.1-70b-versatile"  # Fast + cheap
    - "openai/gpt-4o-mini"            # Reliable
    - "anthropic/claude-3-5-haiku"    # High quality
    - "ollama/llama3.2"                # Free local fallback

  # Detection settings
  detection:
    max_retries: 3
    sample_size: 20
    timeout: 30
```

---

## CLI Usage Examples

### Basic (Default Provider)
```bash
./stt_assemblyai_speaker_mapper.py --llm-detect audio.json
# Uses default: openai/gpt-4o-mini
```

### Specify Provider
```bash
./stt_assemblyai_speaker_mapper.py --llm-detect anthropic/claude-3-5-sonnet audio.json
```

### Local Ollama (Free, Offline)
```bash
./stt_assemblyai_speaker_mapper.py --llm-detect ollama/llama3.2 audio.json
```

### Remote Ollama Server
```bash
./stt_assemblyai_speaker_mapper.py \
  --llm-detect ollama/llama3.2 \
  --llm-endpoint http://gpu-server:11434 \
  audio.json
```

### Ultra-Fast (Groq)
```bash
./stt_assemblyai_speaker_mapper.py --llm-detect groq/llama-3.1-70b audio.json
```

### With Fallback
```bash
./stt_assemblyai_speaker_mapper.py \
  --llm-detect groq/llama-3.1-70b \
  --llm-fallback openai/gpt-4o-mini \
  audio.json
```

### Interactive with AI Suggestions
```bash
./stt_assemblyai_speaker_mapper.py \
  --llm-interactive anthropic/claude-3-5-sonnet \
  audio.json

# Output:
# Speaker A [Alice Anderson]: _    ← Press Enter or type override
# Speaker B [Bob Smith]: Robert    ← User typed "Robert"
```

---

## Provider Comparison for Speaker Detection

| Provider | Speed | Cost/call | Quality | Offline | Recommendation |
|----------|-------|-----------|---------|---------|----------------|
| **groq/llama-3.1-70b** | ⚡⚡⚡ | $0.001 | ⭐⭐⭐⭐ | ❌ | **Best default** (fast + cheap) |
| **openai/gpt-4o-mini** | ⚡⚡ | $0.005 | ⭐⭐⭐⭐⭐ | ❌ | Reliable fallback |
| **anthropic/claude-3-5-sonnet** | ⚡⚡ | $0.020 | ⭐⭐⭐⭐⭐ | ❌ | Best reasoning |
| **anthropic/claude-3-5-haiku** | ⚡⚡⚡ | $0.002 | ⭐⭐⭐⭐ | ❌ | Fast + quality |
| **google/gemini-2.0-flash** | ⚡⚡⚡ | $0.001 | ⭐⭐⭐⭐ | ❌ | Very fast |
| **ollama/llama3.2** | ⚡ | Free | ⭐⭐⭐ | ✅ | **Offline development** |
| **ollama/qwen2.5** | ⚡ | Free | ⭐⭐⭐⭐ | ✅ | Better local quality |

**Recommended default chain:**
1. Groq (fast + cheap first attempt)
2. Claude Haiku (if Groq fails)
3. Local Ollama (if no API access)

---

## Environment Variables

```bash
# API Keys (optional - can use config file)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="..."

# Default provider (optional)
export STT_LLM_PROVIDER="groq/llama-3.1-70b-versatile"

# Ollama endpoint (if not localhost:11434)
export OLLAMA_BASE_URL="http://gpu-server:11434"
```

---

## Bonus: Utilize Claude Subscription

### Problem
User has Claude Pro/Max subscription ($20-200/mo) but `claude` CLI may be restricted.

### Solution 1: Use Anthropic API Key
Even with subscription, get separate API key for pay-per-use:

```bash
# Get API key from console.anthropic.com
export ANTHROPIC_API_KEY="sk-ant-..."

./stt_assemblyai_speaker_mapper.py --llm-detect anthropic/claude-3-5-sonnet audio.json
```

**Note:** This uses API credits (pay-per-token), not subscription quota.

### Solution 2: LiteLLM Proxy (Advanced)
Set up LiteLLM proxy to route through `claude` CLI:

```python
# litellm_config.yaml
model_list:
  - model_name: claude-subscription
    litellm_params:
      model: claude-3-5-sonnet-20241022
      # Custom routing through CLI
```

**Recommendation:** Just use Anthropic API key - simpler and more reliable.

---

## Testing Strategy

### Unit Tests (No API Calls)
```python
@pytest.fixture
def mock_instructor():
    """Mock Instructor client for testing"""
    with patch('instructor.from_provider') as mock:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SpeakerDetection(
            speakers={"A": "Alice", "B": "Bob"},
            confidence="high",
            reasoning="Names mentioned in conversation"
        )
        mock.return_value = mock_client
        yield mock

def test_detect_speakers(mock_instructor):
    result = detect_speakers_llm(
        "openai/gpt-4o-mini",
        "transcript",
        ["A", "B"]
    )
    assert result.speakers == {"A": "Alice", "B": "Bob"}
```

### Integration Tests (Optional, Real APIs)
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_openai_integration():
    result = detect_speakers_llm(
        "openai/gpt-4o-mini",
        "Hi, I'm Alice. Hello Alice, I'm Bob.",
        ["A", "B"]
    )
    assert "Alice" in result.speakers.get("A", "").lower()
    assert "Bob" in result.speakers.get("B", "").lower()

@pytest.mark.integration
def test_ollama_integration():
    """Test with local Ollama (no API key needed)"""
    try:
        result = detect_speakers_llm(
            "ollama/llama3.2",
            "Hi, I'm Alice",
            ["A"]
        )
        assert result.speakers.get("A") is not None
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")
```

---

## Dependencies

```bash
# Core (required)
pip install instructor pydantic

# Provider-specific (install as needed)
pip install openai          # For OpenAI, Azure OpenAI
pip install anthropic       # For Anthropic/Claude
pip install google-generativeai  # For Google Gemini
pip install groq            # For Groq
pip install litellm         # For LiteLLM (100+ providers)
pip install cohere          # For Cohere
pip install mistralai       # For Mistral

# Local models (optional)
pip install llama-cpp-python  # For direct GGUF support
```

**Minimal installation** (just OpenAI + Anthropic + Ollama):
```bash
pip install instructor openai anthropic
```

---

## Implementation Checklist

### Phase 1: Core Instructor Integration
- [ ] Install `instructor` + `pydantic`
- [ ] Create `SpeakerDetection` Pydantic model
- [ ] Implement `detect_speakers_llm()` function
- [ ] Add `--llm-detect PROVIDER/MODEL` CLI argument
- [ ] Test with OpenAI (gpt-4o-mini)

### Phase 2: Multi-Provider Support
- [ ] Add Anthropic support (`anthropic/claude-3-5-haiku`)
- [ ] Add Ollama support (`ollama/llama3.2`)
- [ ] Add Groq support (`groq/llama-3.1-70b-versatile`)
- [ ] Add `--llm-endpoint` for custom URLs

### Phase 3: Fallback Chain
- [ ] Implement provider fallback logic
- [ ] Add `--llm-fallback` argument
- [ ] Configuration file support (`llm_config.yml`)

### Phase 4: Interactive Mode
- [ ] `--llm-interactive` with AI suggestions
- [ ] Format prompts: `Speaker A [Alice]: _`
- [ ] Handle user overrides

### Phase 5: Testing & Documentation
- [ ] Unit tests with mocked Instructor
- [ ] Integration tests (optional, real APIs)
- [ ] Update README with provider examples
- [ ] Cost estimation guide

---

## Cost Optimization

### Strategy 1: Groq First, Others as Fallback
```python
PROVIDER_CHAIN = [
    "groq/llama-3.1-70b-versatile",  # $0.001/call, ultra-fast
    "anthropic/claude-3-5-haiku",     # $0.002/call if Groq fails
    "ollama/llama3.2"                  # Free local fallback
]
```

### Strategy 2: Ollama for Development, Cloud for Production
```python
if os.getenv("ENVIRONMENT") == "production":
    provider = "groq/llama-3.1-70b-versatile"
else:
    provider = "ollama/llama3.2"  # Free local dev
```

### Strategy 3: LiteLLM Router (Advanced)
Automatic load balancing across providers based on cost/availability.

---

## Summary

**Before:** Complex multi-backend architecture with subprocess calls

**After:** Single unified Instructor interface, user chooses provider

**Benefits:**
* ✅ Simpler code (1 backend instead of 3)
* ✅ Better error handling (Pydantic validation)
* ✅ Type safety everywhere
* ✅ Automatic retries
* ✅ 100+ providers accessible
* ✅ No subprocess complexity
* ✅ Better testing (mock Instructor)

**Recommendation:** Use Instructor exclusively, drop mods/sgpt subprocess wrappers.

---

**Status:** Ready for implementation
**Next Step:** Update implementation design document
