# Speaker Mapper Benchmark Framework

Comprehensive evaluation framework for testing LLM-based speaker name detection across different models and providers.

## Overview

This benchmark framework evaluates the `stt_assemblyai_speaker_mapper.py` tool's ability to correctly identify and map speaker names from conversation transcripts using various LLM models. It provides:

* **Diverse test cases** covering common scenarios (clear introductions, informal names, multiple speakers, ambiguous contexts)
* **Flexible reference answers** that accept multiple valid variations (e.g., "Mike" vs "Michael")
* **Automated scoring** with partial credit for acceptable variants
* **Multiple output formats** (JSONL for analysis, ASCII for human reading)
* **Easy model comparison** across OpenAI, Anthropic, Google, Ollama, and other providers

## Directory Structure

```
evals/speaker_mapper/
├── benchmark.py                    # Main benchmark script
├── benchmark.README.md             # This file
├── tests/                          # Test conversation files
│   ├── 001-clear-introductions.json
│   ├── 002-informal-names.json
│   ├── 003-three-speakers.json
│   ├── 004-no-names-mentioned.json
│   ├── 005-partial-names.json
│   ├── 006-ambiguous-context.json
│   └── 007-four-speakers.json
└── references/                     # Reference answer files
    ├── 001-clear-introductions.ref.json
    ├── 002-informal-names.ref.json
    └── ...
```

## Quick Start

### Basic Usage

```bash
# Run benchmark with OpenAI GPT-4o-mini
cd evals/speaker_mapper
./benchmark.py --llm-detect 4o-mini --output ascii

# Run with local Ollama model
./benchmark.py --llm-detect smollm2:1.7b --llm-endpoint http://localhost:11434/v1 --output ascii

# Run with Anthropic Claude
./benchmark.py --llm-detect sonnet --output ascii
```

### Common Examples

```bash
# Generate JSONL output for analysis (to stdout)
./benchmark.py --llm-detect 4o-mini > results_4o-mini.jsonl

# Run specific tests only
./benchmark.py --llm-detect gemini --tests 001,002,003 --output ascii

# Compare multiple models
./benchmark.py --llm-detect 4o-mini > results_4o-mini.jsonl
./benchmark.py --llm-detect sonnet > results_sonnet.jsonl
./benchmark.py --llm-detect smollm2:1.7b --llm-endpoint http://localhost:11434/v1 > results_smollm2.jsonl
```

### Saving Results to Files

The benchmark script supports saving results directly to files, which automatically generates three files:

* `.jsonl` - Machine-readable JSON Lines format
* `.txt` - Human-readable ASCII table format
* `.sh` - Executable shell script with the exact command used (for reproducibility)

```bash
# Save results to files with both JSONL and ASCII formats
./benchmark.py --llm-detect 4o-mini \
  --save-jsonl results/openai_4o-mini.jsonl \
  --save-ascii results/openai_4o-mini.txt

# This generates three files:
# - results/openai_4o-mini.jsonl (detailed results)
# - results/openai_4o-mini.txt (human-readable report)
# - results/openai_4o-mini.sh (command used to generate results)

# Save only JSONL format
./benchmark.py --llm-detect smollm2:1.7b \
  --llm-endpoint http://localhost:11434/v1 \
  --save-jsonl results/ollama_smollm2_1.7b.jsonl

# Save only ASCII format
./benchmark.py --llm-detect sonnet \
  --save-ascii results/anthropic_sonnet.txt

# Combine with --output flag to also display results on stdout
./benchmark.py --llm-detect 4o-mini \
  --save-jsonl results/openai_4o-mini.jsonl \
  --save-ascii results/openai_4o-mini.txt \
  --output ascii  # Also display ASCII table on screen
```

**Benefits**:

* **Reproducibility**: The `.sh` file captures the exact command with timestamp
* **Organization**: Results organized in `results/` directory with consistent naming
* **Comparison**: Easy to compare multiple model runs side-by-side
* **Archiving**: Keep historical results for tracking model improvements over time

## Test Cases

### 001-clear-introductions.json
**Description**: Clear speaker introductions with full names

**Scenario**: Sarah Johnson (host) interviews David Lee

**Expected behavior**: LLM should easily identify both speakers with full names

**Acceptable answers**:
- A: Sarah Johnson, Sarah, Johnson
- B: David Lee, David, Lee

### 002-informal-names.json
**Description**: Handles informal name variations

**Scenario**: Michael/Mike Thompson talks with Elizabeth/Beth Wilson

**Expected behavior**: LLM should recognize both formal and informal names

**Acceptable answers**:
- A: Michael, Mike, Michael Thompson, Mike Thompson, Thompson
- B: Elizabeth, Beth, Elizabeth Wilson, Beth Wilson, Wilson, Liz

### 003-three-speakers.json
**Description**: Three-way conversation

**Scenario**: Jennifer Martinez (project lead), Robert Chen (backend), Amy Parker (frontend)

**Expected behavior**: LLM should correctly identify all three speakers

**Acceptable answers**:
- A: Jennifer Martinez, Jennifer, Martinez
- B: Robert Chen, Robert, Chen, Bob Chen, Bob
- C: Amy Parker, Amy, Parker

### 004-no-names-mentioned.json
**Description**: No names in transcript

**Scenario**: Generic discussion with no name mentions

**Expected behavior**: LLM should return "Unknown" or generic labels

**Acceptable answers**:
- A: Unknown, Speaker A, Host, Interviewer, Person A
- B: Unknown, Speaker B, Guest, Participant, Person B

### 005-partial-names.json
**Description**: Only first name or last name mentioned

**Scenario**: Jessica (first name only) talks with Thompson (last name only)

**Expected behavior**: LLM should extract available partial names

**Acceptable answers**:
- A: Jessica, Jess
- B: Thompson, Mr. Thompson

### 006-ambiguous-context.json
**Description**: Ambiguous/unclear speaker identity

**Scenario**: Discussing "Chris" but unclear which Chris

**Expected behavior**: LLM should acknowledge ambiguity with "Unknown" or best guess

**Acceptable answers**:
- A/B: Unknown, Speaker A/B, Chris (any), Chris Taylor, Chris Morgan

**Note**: This test is intentionally ambiguous - any reasonable answer accepted

### 007-four-speakers.json
**Description**: Four-speaker panel discussion

**Scenario**: Dr. Patricia Williams (moderator), James Rodriguez (CEO), Maria Santos (CTO), Kevin O'Brien (investor)

**Expected behavior**: LLM should track all four speakers correctly

**Acceptable answers**:
- A: Patricia Williams, Dr. Patricia Williams, Dr. Williams, Patricia, Williams
- B: James Rodriguez, James, Rodriguez
- C: Maria Santos, Maria, Santos
- D: Kevin O'Brien, Kevin, O'Brien, Kevin OBrien

## Reference File Format

Reference files define expected answers with variation support:

```json
{
  "test_id": "002-informal-names",
  "description": "Handles Mike/Michael, Beth/Elizabeth variations",
  "expected_mappings": {
    "A": {
      "acceptable": ["Michael", "Mike", "Michael Thompson", "Mike Thompson", "Thompson"],
      "preferred": "Michael"
    },
    "B": {
      "acceptable": ["Elizabeth", "Beth", "Elizabeth Wilson", "Beth Wilson", "Wilson", "Liz"],
      "preferred": "Elizabeth"
    }
  },
  "scoring": {
    "exact_preferred": 1.0,
    "acceptable_variant": 1.0,
    "partial_match": 0.5,
    "wrong": 0.0
  }
}
```

### Scoring Rules

* **Exact preferred** (1.0): Matches the preferred answer exactly
* **Acceptable variant** (1.0): Matches any acceptable variation
* **Partial match** (0.5): Substring match with acceptable answers
* **Wrong** (0.0): No match with any acceptable answer

**Pass threshold**: 75% accuracy (0.75)

## Output Formats

### JSONL Format (Default)

One JSON object per line - ideal for programmatic analysis:

```jsonl
{"test": "001-clear-introductions", "status": "pass", "accuracy": 1.0, "time_sec": 2.3, ...}
{"test": "002-informal-names", "status": "pass", "accuracy": 1.0, "time_sec": 2.1, ...}
{"summary": {"total": 7, "passed": 6, "failed": 1, "pass_rate": 0.857, ...}}
```

### ASCII Format (Human-Readable)

Formatted table with summary statistics:

```
╔══════════════════════════════════════════════════════════════════════════╗
║           Speaker Mapper Benchmark Results                              ║
║  LLM Args: --llm-detect 4o-mini                                         ║
╚══════════════════════════════════════════════════════════════════════════╝

TEST                         STATUS    ACCURACY  TIME    DETAILS
────────────────────────────────────────────────────────────────────────────
001-clear-introductions      ✓ PASS    100.0%    2.3s    A→Sarah✓, B→David✓
002-informal-names           ✓ PASS    100.0%    2.1s    A→Mike✓, B→Beth✓
003-three-speakers           ✓ PASS    100.0%    3.1s    A→Jennifer✓, B→Robert✓...
004-no-names-mentioned       ✗ FAIL     50.0%    2.0s    A→Unknown✓, B→Person B~
────────────────────────────────────────────────────────────────────────────

SUMMARY
════════════════════════════════════════════════════════════════════════════
  Total Tests:      7
  Passed:           6 (85.7%)
  Failed:           1 (14.3%)
  Avg Accuracy:     93.0%
  Avg Time:         2.5s
  Total Time:       17.5s
════════════════════════════════════════════════════════════════════════════
```

## Creating New Test Cases

### 1. Create Test File

Create a new JSON file in `tests/` directory:

```json
{
  "utterances": [
    {
      "speaker": "A",
      "text": "Hi, I'm Alice Anderson...",
      "start": 0,
      "end": 3500
    },
    {
      "speaker": "B",
      "text": "Hello Alice! I'm Bob...",
      "start": 3600,
      "end": 7200
    }
  ],
  "audio_duration": 7200
}
```

**Naming convention**: `NNN-descriptive-name.json` (e.g., `008-technical-jargon.json`)

### 2. Create Reference File

Create corresponding reference in `references/` directory:

```json
{
  "test_id": "008-technical-jargon",
  "description": "Technical discussion with jargon",
  "expected_mappings": {
    "A": {
      "acceptable": ["Alice Anderson", "Alice", "Anderson"],
      "preferred": "Alice Anderson"
    },
    "B": {
      "acceptable": ["Bob Martinez", "Bob", "Martinez"],
      "preferred": "Bob Martinez"
    }
  },
  "scoring": {
    "exact_preferred": 1.0,
    "acceptable_variant": 1.0,
    "partial_match": 0.5,
    "wrong": 0.0
  }
}
```

**Naming convention**: `NNN-descriptive-name.ref.json` (must match test file name)

### 3. Test Your New Case

```bash
./benchmark.py --llm-detect 4o-mini --tests 008 --output ascii
```

## Comparing Models

### Generate Results for Multiple Models

```bash
# OpenAI models
./benchmark.py --llm-detect 4o-mini \
  --save-jsonl results/openai_4o-mini.jsonl \
  --save-ascii results/openai_4o-mini.txt

./benchmark.py --llm-detect 4o \
  --save-jsonl results/openai_4o.jsonl \
  --save-ascii results/openai_4o.txt

# Anthropic models
./benchmark.py --llm-detect sonnet \
  --save-jsonl results/anthropic_sonnet.jsonl \
  --save-ascii results/anthropic_sonnet.txt

./benchmark.py --llm-detect haiku \
  --save-jsonl results/anthropic_haiku.jsonl \
  --save-ascii results/anthropic_haiku.txt

# Google models
./benchmark.py --llm-detect gemini \
  --save-jsonl results/google_gemini.jsonl \
  --save-ascii results/google_gemini.txt

# Local Ollama models
./benchmark.py --llm-detect smollm2:1.7b --llm-endpoint http://localhost:11434/v1 \
  --save-jsonl results/ollama_smollm2_1.7b.jsonl \
  --save-ascii results/ollama_smollm2_1.7b.txt

./benchmark.py --llm-detect llama3.2:1b --llm-endpoint http://localhost:11434/v1 \
  --save-jsonl results/ollama_llama3.2_1b.jsonl \
  --save-ascii results/ollama_llama3.2_1b.txt
```

### Analyze Results

```bash
# Extract summaries
grep '"summary"' results/*.jsonl

# Compare pass rates
jq -r 'select(.summary) | "\(.summary.passed)/\(.summary.total) (\(.summary.pass_rate*100|floor)%)"' results/*.jsonl

# Compare average accuracy
jq -r 'select(.summary) | .summary.avg_accuracy' results/*.jsonl
```

## Best Practices

### For Test Creation

1. **Clear speaker differentiation**: Use distinct names that are easy to identify
2. **Realistic conversations**: Model real-world scenarios (interviews, meetings, casual chat)
3. **Varied difficulty**: Include both easy and challenging cases
4. **Edge cases**: Test boundary conditions (no names, ambiguous context)
5. **Sufficient context**: Provide enough utterances for LLM to identify patterns

### For Reference Answers

1. **Be generous with acceptable variants**: Include common variations (Mike/Michael, formal/informal)
2. **Consider cultural variations**: Include nicknames, titles (Dr., Mr.), maiden names
3. **Substring matching**: Partial matches should give partial credit
4. **Document ambiguity**: Add notes for intentionally ambiguous tests
5. **Realistic expectations**: Don't expect perfection on impossible cases

### For Benchmarking

1. **Run multiple times**: LLM outputs can vary, average over 3-5 runs for stability
2. **Control for context**: Keep `--llm-sample-size` consistent across runs
3. **Document configuration**: Save exact command used for reproducibility
4. **Track versions**: Note model versions and API changes over time
5. **Fair comparison**: Use same test set and parameters across models

## Troubleshooting

### Benchmark script fails to run

```bash
# Check Python version (requires 3.7+)
python3 --version

# Make script executable
chmod +x benchmark.py

# Check paths
ls tests/*.json
ls references/*.ref.json
```

### Speaker mapper not found

```bash
# Verify path to speaker mapper
ls ../../stt_assemblyai_speaker_mapper.py

# Run from correct directory
cd evals/speaker_mapper
```

### Ollama connection issues

```bash
# Check Ollama is running
ollama list

# Verify endpoint
curl http://localhost:11434/v1/models

# Test model directly
ollama run smollm2:1.7b "Hello"
```

### All tests fail with timeout

```bash
# Increase timeout in benchmark.py (line ~115)
# timeout=60  # Increase this value

# Or use faster model
./benchmark.py --llm-detect llama3.2:1b --llm-endpoint http://localhost:11434/v1
```

## Advanced Usage

### Custom Scoring

Edit reference file to adjust scoring weights:

```json
{
  "scoring": {
    "exact_preferred": 1.0,
    "acceptable_variant": 0.9,    # Reduce if you prefer exact matches
    "partial_match": 0.3,          # Reduce if you want stricter matching
    "wrong": 0.0
  }
}
```

### Filtering Tests

```bash
# Run only clear cases (001-003)
./benchmark.py --llm-detect 4o-mini --tests 001,002,003

# Run only challenging cases (004-006)
./benchmark.py --llm-detect 4o-mini --tests 004,005,006

# Run specific test
./benchmark.py --llm-detect 4o-mini --tests 007
```

### Batch Comparison Script

```bash
#!/bin/bash
# compare_models.sh - Compare multiple models

models=(
    "4o-mini"
    "sonnet"
    "gemini"
    "smollm2:1.7b --llm-endpoint http://localhost:11434/v1"
)

for model in "${models[@]}"; do
    echo "Testing $model..."
    ./benchmark.py --llm-detect $model --output ascii
    echo "---"
done
```

## Future Enhancements

Potential improvements to the benchmark framework:

* **Multi-run aggregation**: Average results over N runs for statistical significance
* **Confidence intervals**: Calculate error margins for accuracy metrics
* **Confusion matrix**: Track specific error patterns (which speakers get confused)
* **Performance profiling**: Detailed breakdown of time spent per component
* **Cost tracking**: Estimate API costs based on token usage
* **Interactive mode**: Test with `--llm-interactive` and human validation
* **Domain-specific tests**: Add tests for medical, legal, technical domains
* **Multilingual tests**: Test speaker detection in non-English languages
* **Audio quality simulation**: Add tests with transcription errors/noise

## Contributing

To contribute new test cases:

1. Create test file in `tests/`
2. Create reference file in `references/`
3. Test with at least 2-3 different models
4. Document any special considerations
5. Submit PR with test description

## License

Same as parent project.

## Contact

For questions or issues with the benchmark framework, see parent project README.
