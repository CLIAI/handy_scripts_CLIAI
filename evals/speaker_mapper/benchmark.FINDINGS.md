# Speaker Mapper Benchmark Findings

This document captures actual benchmark results and insights from testing various LLM models on the speaker detection task.

## Task Requirements

The speaker mapper uses **structured Pydantic outputs** with the Instructor library. Models must:

* Generate valid JSON matching the `SpeakerDetection` Pydantic schema
* Follow complex nested structures with `List[SpeakerMapping]`
* Handle retry logic with validation errors
* Extract speaker names from conversation context

## Test Results by Model Size

### Cloud API Models (Baseline - 100% Success)

| Model | Pass Rate | Avg Time | Notes |
|-------|-----------|----------|-------|
| **GPT-4o-mini** | 100% (3/3) | 7.8s | Fast, accurate baseline |

### Small Local Models (< 3GB) - All Failed

| Model | Size | Pass Rate | Avg Time | Issues |
|-------|------|-----------|----------|--------|
| **smollm2:360m** | 725MB | 0% (0/7) | 40s | Cannot generate valid Pydantic schema, timeouts |
| **granite3-moe** | 821MB | 0% (0/3) | 21s | Validation errors, wrong JSON format |
| **llama3.2:1b** | 1.3GB | 0% (0/3) | 50s | Returns "Unknown" or validation errors |
| **smollm2:latest (1.7B)** | 1.8GB | Timeout | >180s | Too slow, likely similar issues to 360m |
| **llama3.2:3b** | 2.0GB | Timeout | >180s | Too slow for practical use |
| **nemotron-mini** | 2.7GB | 0-33% | 45s | Wrong JSON format, hallucinates "Alice Anderson" & "Bob Martinez" |

### Key Findings for Small Models

**Structured Output Challenge**: Models under 3GB parameters struggle with:

* Generating valid nested Pydantic schemas
* Following `{"speakers": [...]}` structure vs simple `{"A": "Name"}`
* Handling validation retry logic
* Maintaining schema compliance across multiple attempts

**Common Failure Patterns**:

1. **Schema Mismatch**: Models output `{"A": "Name", "B": "Name"}` instead of required `{"speakers": [{"label": "A", "name": "Name"}, ...]}`
2. **Hallucination**: nemotron-mini fixated on "Alice Anderson" and "Bob Martinez" regardless of actual conversation content
3. **Timeouts**: Models >1B but <4B often too slow (>60s per test)
4. **Validation Loops**: Small models fail validation, retry, fail again (3x attempts → error)

**Speed vs Accuracy**:

* granite3-moe: Fastest failure (21s avg) but 0% accuracy
* smollm2:360m: Medium speed (40s avg) but 0% accuracy
* nemotron-mini: Slow (45s avg) with occasional partial success but unreliable
* llama3.2 family: Too slow (>60s) with poor accuracy

## Recommendations

### For Production Use

**Cloud APIs** (recommended for reliability):

* **GPT-4o-mini**: Best balance of speed (8s) and accuracy (100%)
* **Claude Sonnet**: Expected high accuracy based on model capabilities
* **Gemini**: Worth testing for Google Vertex AI users

### For Local Deployment

**Minimum Requirements**: 7B+ parameter models needed for structured outputs

**Recommended local models to test** (not yet benchmarked):

* **qwen2.5-coder:7b** (4.7GB) - Coding-focused, should excel at structured JSON
* **mistral:7b-instruct** (4.1GB) - Good instruction following
* **llama3.1:8b** (4.7GB) - Latest Meta model with strong capabilities
* **phi4** (9.1GB) - Microsoft's capable model

### Hardware Considerations

**Small models (<3GB) are NOT viable** for this task, even with:

* 16-24GB RAM systems
* CPU-only inference
* Quantized models

The task requires **logical reasoning + structured output** capabilities that only emerge at 7B+ scale.

## Insights for Pydantic/Instructor Users

If you're using **Instructor library for structured outputs**:

* Budget for 7B+ models minimum
* Smaller models will fail validation loops
* Consider schema complexity in model selection
* Cloud APIs more reliable than small local models
* Test with simple 2-speaker conversations first

## Test Cases Used

Results above based on:

* **001-clear-introductions**: Simple 2-speaker with full names (easiest)
* **002-informal-names**: Name variations (Mike/Michael)
* **003-three-speakers**: 3-way conversation
* Full 7-test suite includes edge cases (no names, 4 speakers, ambiguous)

## Future Testing

Models to benchmark:

* [ ] qwen2.5-coder:7b
* [ ] qwen2.5-coder:14b
* [ ] mistral:7b-instruct
* [ ] llama3.1:8b
* [ ] phi4
* [ ] granite3-dense:latest (1.6GB) - might perform better than MoE
* [ ] hermes3:8b
* [ ] llama3-groq-tool-use:8b - optimized for structured outputs

## Reproducing Results

```bash
# Test any model
cd evals/speaker_mapper

# Cloud API (fast, accurate)
./benchmark.py --llm-detect 4o-mini --output ascii

# Local model (7B+ recommended)
./benchmark.py --llm-detect qwen2.5-coder:7b \
  --llm-endpoint http://localhost:11434/v1 \
  --output ascii

# Save results to files
./benchmark.py --llm-detect {model} \
  --save-jsonl results/{model}.jsonl \
  --save-ascii results/{model}.txt
```

## Date

Initial findings: 2025-10-14

Models tested: GPT-4o-mini, smollm2:360m, smollm2:1.7b, llama3.2:1b, llama3.2:3b, granite3-moe, nemotron-mini
