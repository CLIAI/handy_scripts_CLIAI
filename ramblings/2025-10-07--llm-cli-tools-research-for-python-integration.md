# LLM CLI Tools Research: stdin→stdout Processing for Python Integration

**Research Date:** 2025-10-07
**Purpose:** Document LLM CLI tools for integration into Python script with stdin→stdout processing capabilities
**Status:** Research completed - implementation pending

---

## Executive Summary

This document researches 4 components for LLM integration:

* **3 CLI Tools:** ollama, mods, claude (CLI)
* **1 Python Library:** Instructor

### Quick Recommendation

**For Default Backend:** Use **mods** CLI
* Best balance of features, ease-of-use, and cloud/local flexibility
* Native stdin→stdout piping design
* Works with multiple providers (OpenAI, Anthropic, Groq, LocalAI, etc.)
* Simple model selection and configuration

**For Structured Output:** Use **Instructor** Python library
* Superior validation, automatic retries, type safety
* Supports 15+ providers with unified API
* Built on Pydantic for robust schema enforcement
* Eliminates manual JSON parsing complexity

**For Local/Offline:** Use **ollama**
* Complete local model management
* No API costs, full privacy control
* GPU acceleration support
* Works offline

**For Advanced Workflows:** Consider **claude CLI** with restrictions
* Most powerful reasoning capabilities (Claude Sonnet 4.5)
* Requires careful tool restriction for stdin→stdout mode
* Higher cost, rate limits apply

---

## 1. Ollama - Local LLM Runtime

### Overview

Ollama is a complete local LLM runtime and management system that allows running models like Llama 3, Mistral, CodeLlama locally without cloud dependencies.

### Installation

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
# Download from ollama.com/download/mac
# Drag Ollama.app to Applications folder
```

**Windows:**
```bash
# Download executable from ollama.com/download
# Run installer automatically
```

**System Requirements:**

* CPU: 64-bit processor
* RAM: 8GB minimum (16GB recommended)
  * 7B models: 8GB RAM
  * 13B models: 16GB RAM
  * 70B models: 40GB+ RAM
* Storage: 10GB+ free space (models range 1-80GB each)
* GPU: Optional (NVIDIA Compute 5.0+, AMD ROCm)

### Command-Line Syntax

**Basic Usage:**
```bash
# Interactive mode
ollama run llama3

# With prompt
ollama run llama3 "Explain quantum computing"

# Pipe input
cat README.md | ollama run llama3 "Summarize this document"
```

### Model Selection

**Available Models:**
```bash
# Pull specific models
ollama pull llama3
ollama pull mistral:7b
ollama pull codellama:7b

# List installed models
ollama list

# Remove models
ollama rm llama3:7b
```

**Common Models:**

* `llama3` - Meta's latest model (various sizes)
* `mistral:7b` - Mistral 7B (4.1GB)
* `codellama:7b` - Code-specialized Llama (3.8GB)
* `gemma` - Google's Gemma models
* `phi` - Microsoft's Phi models

### stdin→stdout Usage

**Native ollama:**
```bash
# Pipe file contents
cat myfile.py | ollama run llama3 "Explain this code"

# Chain commands
echo "List 5 programming languages" | ollama run mistral

# Save output
ollama run llama3 "Explain AI" > output.txt
```

**Third-Party ollama-cli (Better for piping):**
```bash
# Install
pip install ollama-cli

# Basic piping
echo "Why is the sky blue?" | ollama-cli

# With options
echo "Write an article" | ollama-cli --max-lines=10 --opts="temperature=0.5"

# Termination control
echo "List animals" | ollama-cli --max-lines=2
```

### Structured Output (JSON)

**API-based (Recommended):**
```bash
# Using curl with API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What color is the sky? Respond using JSON",
  "format": "json",
  "stream": false
}'
```

**Python with Pydantic:**
```python
from ollama import chat
from pydantic import BaseModel

class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]

response = chat(
    messages=[{'role': 'user', 'content': 'Tell me about Canada.'}],
    model='llama3.1',
    format=Country.model_json_schema()
)
```

### Limitations & Gotchas

**1. Streaming Issues with Tool Calling**

* When tools are enabled, streaming may break
* Responses arrive as single block instead of progressive chunks
* Recent updates have improved this

**2. Performance Bottlenecks**

* Memory limitations are the most common issue
* Context window size affects performance
* 32k+ context improves results but increases memory usage

**3. Windows stdin Limitation**

* Piping to stdin fails in git bash on Windows
* Error: "The handle is invalid"
* Use WSL2 or native Windows Terminal

**4. No Native JSON Flag**

* `/set format json` is for interactive mode only
* Must use API endpoint with `format` parameter
* Native CLI doesn't have `--format json` flag

**5. Model Storage**

* Models consume significant disk space
* 70B models require 40GB+ storage
* Must manually manage model deletion

---

## 2. Mods - Multi-Provider AI Pipeline Tool

### Overview

Mods by Charmbracelet is a CLI tool designed for integrating AI into command-line pipelines. It works with OpenAI, Anthropic, Cohere, Groq, Azure OpenAI, Google Gemini, and local models via LocalAI.

### Installation

**macOS/Linux:**
```bash
brew install charmbracelet/tap/mods
```

**Windows:**
```bash
winget install charmbracelet.mods
```

**Arch Linux:**
```bash
yay -S mods
```

**Nix:**
```bash
nix-shell -p mods
```

**From Source:**
```bash
go install github.com/charmbracelet/mods@latest
```

### Configuration

**Initial Setup:**
```bash
# Open settings
mods --settings

# Check configuration location
mods --dirs
# Linux: ~/.config/mods/mods.yml
# macOS: ~/Library/Application Support/mods/mods.yml
```

**API Key Setup:**
```bash
# Environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="..."
export AZURE_OPENAI_KEY="..."
```

**Configuration File (mods.yml):**
```yaml
apis:
  openai:
    api-key: sk-...
    base-url: https://api.openai.com/v1

  anthropic:
    api-key: sk-ant-...

  localai:
    base-url: http://localhost:8080/v1
```

### Command-Line Syntax

**Core Flags:**

* `-m, --model <name>` - Select specific model
* `-M, --ask-model` - Interactive model picker
* `-f, --format <type>` - Request specific output format
* `-r, --raw` - Raw unformatted output
* `-c, --continue` - Continue previous conversation
* `-t, --title <text>` - Set conversation title
* `-l, --list` - List saved conversations
* `--role <name>` - Use custom system prompt role
* `-P, --prompt` - Include prompt in output
* `-p, --prompt-args` - Include arguments in response
* `--word-wrap <n>` - Set output width (default: 80)

### Model Selection

**Examples:**
```bash
# Use specific model
mods -m gpt-4 "Explain this"

# Interactive model selection
mods -M

# Use o1-mini (reasoning model)
mods -m o1-mini "Solve this logic puzzle"

# Claude models
mods -m claude-sonnet-4-5 "Write code"

# Local models
mods -m llama3:local "Process this"
```

**Default Behavior:**

* Uses GPT-4 by default
* Falls back to GPT-3.5-turbo if unavailable
* Models configured in `mods.yml`

### stdin→stdout Usage Examples

**Basic Piping:**
```bash
# Summarize files
ls -la | mods "summarize these files"

# Code explanation
cat script.py | mods "explain this code"

# Refactoring
cat old_code.js | mods "refactor to use modern ES6" > new_code.js

# Git commit messages
git diff | mods "write a commit message for these changes"
```

**Custom Roles:**
```bash
# Shell expert (outputs commands without explanation)
mods --role shell "list files in current directory"

# JSON output
ls | mods -f json "list these as JSON array"
```

**Conversation Management:**
```bash
# Start new titled conversation
mods -t "code-review" "review this code"

# Continue conversation
mods -c

# List conversations
mods -l

# Delete conversation
mods --delete <conversation-id>
```

### Structured Output

**Request Format:**
```bash
# JSON format
cat data.txt | mods -f json "extract names and emails as JSON"

# Markdown format
mods -f markdown "create a table of programming languages"

# Raw output (no formatting)
echo "simple text" | mods -r "process this"
```

### Limitations & Gotchas

**1. API Costs**

* Cloud providers charge per token
* No built-in cost tracking
* GPT-4 significantly more expensive than GPT-3.5
* O1-preview/O1-mini have higher costs

**2. Rate Limits**

* Subject to provider rate limits
* OpenAI: Varies by tier (20 RPM free → 10,000 RPM paid)
* Anthropic: Pro ($20/mo) = 40 prompts/5hrs, Max ($100-200) = more capacity
* No built-in rate limit handling

**3. Environment Variable Prefix**

* `MODS_` prefix overrides config file
* Can cause unexpected behavior
* Example: `MODS_MODEL=gpt-3.5` overrides config

**4. LocalAI Default Port**

* Defaults to port 8080 for LocalAI
* May conflict with other services
* Requires manual configuration change

**5. Provider Cleanup**

* Config includes all providers by default
* Should remove unused providers to avoid confusion
* Edit `~/.config/mods/mods.yml` manually

**6. Conversation Storage**

* Saves all conversations locally by default
* Can consume disk space over time
* Stored in XDG data directory

---

## 3. Claude CLI - Anthropic's Official CLI

### Overview

Claude Code CLI by Anthropic provides access to Claude models (Opus, Sonnet, Haiku) with agentic capabilities. Default mode includes tool use (file operations, bash, web search), but can be restricted to pure stdin→stdout mode.

### Installation

**npm (Global):**
```bash
npm install -g @anthropic-ai/claude-code
# Avoid sudo - causes permission issues
```

**Install Script (Recommended):**
```bash
curl -fsSL https://claude.ai/install.sh | bash

# Specific version
curl -fsSL https://claude.ai/install.sh | bash -s 1.0.58
```

**Verification:**
```bash
claude --version
claude doctor
```

### API Key Configuration

**Method 1: Environment Variable (Recommended)**
```bash
# Export API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Add to shell profile
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

**Method 2: Interactive Config**
```bash
claude config
# Follow prompts to set API key
```

**Authentication Options:**

* **API Key:** From Anthropic Console (pay-per-use)
* **Claude Pro:** $20/mo (10-40 prompts per 5 hours)
* **Claude Max:** $100-200/mo (higher limits)

**Note:** Print mode (`-p`) prefers API keys over OAuth subscriptions, which can cause unexpected billing.

### Command-Line Syntax

**Print Mode Flags:**

* `-p, --print` - Non-interactive print mode (headless)
* `--allowedTools <list>` - Whitelist specific tools
* `--output-format <type>` - Output format: `text`, `json`, `stream-json`
* `--include-partial-messages` - Include streaming partials
* `--max-turns <n>` - Limit agentic iterations
* `--verbose` - Detailed logging
* `--system-prompt <text>` - Custom system prompt

### Restricting to stdin→stdout Mode

**Goal:** Disable all file operations and tool use for pure text processing.

**Approach 1: Empty allowedTools**
```bash
# Minimal tools (none specified = ask for permission)
claude -p "your prompt" --allowedTools ""

# Explicitly empty list
claude -p "process text" --allowedTools
```

**Approach 2: Deny Configuration**
```bash
# Edit ~/.claude/settings.json
{
  "allowedTools": [],
  "permissions": {
    "deny": ["*"]
  }
}
```

**Approach 3: Whitelist Only Read (if needed)**
```bash
# Allow only reading (no writes, no bash)
claude -p "explain file" --allowedTools "Read"

# Allow specific git commands only
claude -p "analyze repo" --allowedTools "Bash(git log:*)" "Bash(git diff:*)"
```

**Known Issues:**

* Non-interactive mode may still request permissions despite configuration
* Bug reported: github.com/anthropics/claude-code/issues/581
* Workaround: Use API directly via Python SDK instead of CLI

### stdin→stdout Usage

**Basic Examples:**
```bash
# Process piped input
cat application.log | claude -p "identify error patterns"

# With explicit prompt
echo "This is test data" | claude -p "summarize"

# JSON output
cat data.json | claude -p "validate this JSON" --output-format json
```

**Streaming JSON:**
```bash
# Stream JSON responses
echo "Long document text..." | claude -p "extract entities" \
  --output-format stream-json \
  --include-partial-messages
```

**Limitations in Print Mode:**

* Must provide input via stdin OR prompt argument (not optional)
* Error if neither provided: "Input must be provided either through stdin or as a prompt argument"
* Requires proper environment variable setup (ANTHROPIC_API_KEY)

### Model Selection

**Claude CLI uses configured model in settings:**
```bash
# Check current model
claude config get model

# Set default model
claude config set model claude-sonnet-4-5

# Available models:
# - claude-opus-4
# - claude-sonnet-4-5 (recommended, most capable)
# - claude-sonnet-3-5
# - claude-haiku-3-5 (fastest, cheapest)
```

**Note:** Model selection is NOT via `-m` flag like other CLIs. Must configure via `claude config`.

### Structured Output

**JSON Output:**
```bash
# Request JSON format
echo "Extract data from: John Doe, age 30" | claude -p \
  "extract as JSON with name and age fields" \
  --output-format json
```

**Best Practice:** Use Instructor library instead for structured outputs (see Section 4).

### Limitations & Gotchas

**1. stdin Raw Mode Errors in CI/CD**

* Error: "Raw mode is not supported on the current process.stdin"
* Occurs in non-interactive environments (CI pipelines, cron jobs)
* Workaround: Use Anthropic Python SDK directly instead of CLI

**2. Authentication Confusion**

* Interactive mode uses OAuth (Pro/Max subscription)
* Print mode uses API key (pay-per-token)
* Can deduct from API balance even if logged into Max subscription
* Check which auth method is active: `claude config get auth`

**3. Permission Prompts Despite Configuration**

* Bug: CLI may still ask for tool permissions even when configured
* Affects non-interactive usage
* Issue: github.com/anthropics/claude-code/issues/581
* Workaround: Use `--allowedTools ""` flag explicitly

**4. High Cost & Rate Limits**

* Claude Sonnet 4.5: Premium pricing
* Pro plan: Only 10-40 prompts per 5 hours
* Max plan: $100-200/mo still has limits
* Affects <5% of users but critical for heavy automation

**5. Verbose stderr Output**

* Prints "no stderr output" messages even on success
* Can clutter scripts
* Redirect stderr: `claude -p "prompt" 2>/dev/null`

**6. Not Ideal for stdin→stdout**

* Designed for agentic workflows with tool use
* Print mode is secondary feature
* Better options exist (mods, direct API, Instructor)

---

## 4. Instructor - Python Library for Structured LLM Outputs

### Overview

Instructor is a Python library that wraps LLM APIs to enforce structured outputs using Pydantic models. It provides automatic validation, retries, type safety, and works with 15+ providers.

### Installation

**Basic Installation:**
```bash
pip install instructor
```

**Python Requirements:**

* Python >=3.9, <4.0
* Pydantic (installed automatically)

**Provider-Specific Installation:**
```bash
# Anthropic (Claude)
pip install "instructor[anthropic]"

# Google Gemini
pip install "instructor[google-generativeai]"

# Multiple providers
pip install "instructor[anthropic,google-genai]"
```

**Available Extras:**

* `anthropic` - Anthropic Claude SDK
* `google-generativeai` - Google Gemini
* `cohere` - Cohere models
* `groq` - Groq inference
* `litellm` - Multi-provider via LiteLLM
* `mistral` - Mistral AI
* `ollama` - Local Ollama models
* `bedrock` - AWS Bedrock
* `vertexai` - Google Vertex AI

### Basic Usage

**OpenAI Example:**
```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

class Person(BaseModel):
    name: str
    age: int
    email: str | None = None

# Patch OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data
person = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Person,
    messages=[{
        "role": "user",
        "content": "Extract: John Doe is 30 years old, email: john@example.com"
    }]
)

print(person.name)  # "John Doe"
print(person.age)   # 30
```

**Anthropic (Claude) Example:**
```python
import instructor
from anthropic import Anthropic

# Patch Anthropic client
client = instructor.from_anthropic(Anthropic())

person = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    response_model=Person,
    messages=[{
        "role": "user",
        "content": "Extract: Jane Smith, age 25"
    }]
)
```

**Multi-Provider Unified API:**
```python
# Unified interface across providers
client = instructor.from_provider("openai/gpt-4o")
client = instructor.from_provider("anthropic/claude-sonnet-4-5")
client = instructor.from_provider("google/gemini-pro")

# API keys via environment or parameters
client = instructor.from_provider(
    "openai/gpt-4o",
    api_key="sk-..."
)
```

### Integration with Different Providers

**Supported Providers (15+):**

* **OpenAI** - GPT-4, GPT-4o, GPT-3.5-turbo, O1, O1-mini
* **Anthropic** - Claude Opus, Sonnet, Haiku (all versions)
* **Google** - Gemini Pro, Gemini Flash, PaLM
* **Mistral** - Mistral Large, Mistral Medium
* **Cohere** - Command, Command-Light
* **Groq** - Fast inference for open models
* **Ollama** - Local open-source models
* **DeepSeek** - DeepSeek models
* **AWS Bedrock** - All Bedrock models
* **Azure OpenAI** - Azure-hosted OpenAI models
* **Vertex AI** - Google Cloud AI models
* **Together AI** - Open model hosting
* **Anyscale** - Ray-based inference
* **Fireworks AI** - Fast open model inference
* **Writer** - Writer LLM

**Ollama Integration:**
```python
import instructor
from openai import OpenAI

# Ollama uses OpenAI-compatible API
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Required but not used
    ),
    mode=instructor.Mode.JSON
)

class Response(BaseModel):
    answer: str
    confidence: float

result = client.chat.completions.create(
    model="llama3",
    response_model=Response,
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

### Code Examples for Structured Parsing

**Simple Extraction:**
```python
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool
    tags: List[str]

text = """
Our premium laptop costs $1299 and is currently in stock.
Tags: electronics, computer, portable
"""

product = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Product,
    messages=[{"role": "user", "content": f"Extract product info: {text}"}]
)
```

**Complex Nested Structures:**
```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    employees: int
    address: Address
    departments: List[str]

# Instructor handles deep nesting automatically
company = client.chat.completions.create(
    model="gpt-4o",
    response_model=Company,
    messages=[{
        "role": "user",
        "content": """
        Extract: Acme Corp has 500 employees located at
        123 Main St, New York, USA.
        Departments: Engineering, Sales, HR
        """
    }]
)
```

**Validation with Constraints:**
```python
from pydantic import Field, field_validator

class UserProfile(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: int = Field(ge=0, le=120)
    email: str

    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()

# Instructor will retry if validation fails
profile = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserProfile,
    messages=[{"role": "user", "content": "Create a user profile for John"}]
)
```

**Lists and Enums:**
```python
from enum import Enum
from typing import List

class Category(str, Enum):
    TECH = "technology"
    HEALTH = "health"
    FINANCE = "finance"

class Article(BaseModel):
    title: str
    category: Category
    keywords: List[str]

articles = client.chat.completions.create(
    model="gpt-4o",
    response_model=List[Article],  # List of structured objects
    messages=[{
        "role": "user",
        "content": "Extract articles from this news feed: ..."
    }]
)
```

### Comparison to Manual JSON Parsing

**Manual JSON Parsing (Traditional Approach):**
```python
import json
from openai import OpenAI

client = OpenAI()

# Request JSON in prompt
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": """
        Extract person info as JSON with fields: name, age, email
        Text: John Doe is 30 years old, email: john@example.com
        """
    }]
)

# Manual parsing
try:
    data = json.loads(response.choices[0].message.content)
    name = data.get('name')  # No type safety
    age = data.get('age')    # Could be string instead of int
    email = data.get('email')  # Might be missing

    # Manual validation
    if not name or not isinstance(age, int):
        raise ValueError("Invalid data")

except json.JSONDecodeError:
    # Handle parsing errors manually
    print("Failed to parse JSON")
except KeyError:
    # Handle missing fields manually
    print("Missing required field")
```

**With Instructor (Modern Approach):**
```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int
    email: str | None = None

# Automatic parsing, validation, retries
person = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Person,
    messages=[{
        "role": "user",
        "content": "Extract: John Doe is 30 years old, email: john@example.com"
    }]
)

# Type-safe access with IDE autocomplete
print(person.name)  # Guaranteed to be string
print(person.age)   # Guaranteed to be int
```

**Key Differences:**

| Aspect | Manual Parsing | Instructor |
|--------|---------------|------------|
| **Schema Definition** | Write JSON schema manually | Use Python classes with type hints |
| **Validation** | Manual checks, try/except blocks | Automatic via Pydantic validators |
| **Type Safety** | No type safety, returns dict | Full type safety with IDE support |
| **Error Handling** | Manual error handling for parsing failures | Automatic retries with error feedback to LLM |
| **Nested Objects** | Complex manual dict navigation | Natural Python object access |
| **Missing Fields** | Manual null checks with `.get()` | Pydantic default values or validation errors |
| **Code Volume** | 20-50+ lines for robust parsing | 5-10 lines with Instructor |
| **Maintenance** | High - changes require updating validation | Low - changes to Pydantic model auto-propagate |
| **Debugging** | Print dict contents, manual inspection | Type errors at IDE time, clear validation messages |

**Benefits of Instructor:**

1. **Automatic Validation & Retries**
   * If LLM returns invalid data, Instructor retries with error details
   * Configurable max retries (default: 3)
   * Validation errors fed back to model for correction

2. **No Manual Schema Writing**
   * JSON schemas auto-generated from Pydantic models
   * Changes to Python class automatically update schema

3. **Built-in Error Handling**
   * Handles parsing errors, missing keys, type mismatches
   * Manual approaches require extensive try/except blocks

4. **Complex Nested Structures**
   * Nested objects, lists, enums work seamlessly
   * Manual parsing becomes exponentially complex with nesting

5. **Time Savings**
   * Manual approach: 2-3 hours implementation + 4-6 hours debugging edge cases
   * Instructor: 15-30 minutes for same functionality

6. **Type Safety**
   * IDE autocomplete for all fields
   * Compile-time type checking with mypy/pyright
   * Refactoring tools work correctly

7. **Validation Types**
   * JSON schema validation (structure)
   * Content validation (custom validators)
   * Range checks, regex patterns, custom logic

---

## Comparison Table

| Feature | Ollama | Mods | Claude CLI | Instructor |
|---------|--------|------|------------|-----------|
| **Type** | Local LLM Runtime | Multi-Provider CLI | Anthropic Official CLI | Python Library |
| **Installation** | curl script / installer | brew / winget | npm / curl script | pip install |
| **API Required** | No (local models) | Yes (cloud) or LocalAI | Yes (Anthropic) | Yes (any provider) |
| **Cost** | Free (local compute) | Varies by provider | $$ (Claude pricing) | Varies by provider |
| **Offline Support** | ✅ Full | ⚠️ With LocalAI only | ❌ Cloud only | ⚠️ With Ollama |
| **stdin→stdout** | ✅ Native | ✅ Native | ⚠️ With restrictions | ✅ Via API |
| **Model Selection** | `ollama run <model>` | `-m <model>` | Via config only | Per-request |
| **JSON Output** | Via API + schema | `-f json` flag | `--output-format json` | ✅ Automatic via Pydantic |
| **Structured Output** | ⚠️ Manual schema | ⚠️ Prompt-based | ⚠️ Prompt-based | ✅✅ Validated Pydantic |
| **Validation** | Manual | Manual | Manual | ✅ Automatic retries |
| **Type Safety** | ❌ None | ❌ None | ❌ None | ✅ Full Pydantic |
| **Providers** | Local models only | OpenAI, Anthropic, Cohere, Groq, LocalAI, Gemini | Anthropic only | 15+ providers |
| **Conversation History** | Manual | ✅ Built-in | ✅ Built-in | Manual |
| **Streaming** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **GPU Acceleration** | ✅ Automatic | N/A | N/A | N/A |
| **RAM Requirements** | 8-40GB depending on model | Minimal | Minimal | Minimal |
| **Best For** | Privacy, offline, local dev | General CLI automation | Advanced reasoning | Structured data extraction |
| **Setup Complexity** | Low-Medium | Low | Low-Medium | Low |
| **Documentation** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Maturity** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ (new) | ⭐⭐⭐⭐⭐ |
| **Community** | Large | Medium | Growing | Very Large |

**Legend:**

* ✅ = Fully supported, works well
* ⚠️ = Partially supported or requires workarounds
* ❌ = Not supported or not recommended

---

## Design Recommendations for Modular Backend Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│            User Python Script                        │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐
│  LLM Backend     │      │  Output Parser       │
│  Interface       │      │  (Instructor)        │
└────────┬─────────┘      └──────────────────────┘
         │
         │ (Strategy Pattern)
         │
    ┌────┴────┬────────┬────────────┐
    ▼         ▼        ▼            ▼
┌─────┐  ┌──────┐  ┌─────────┐  ┌──────────┐
│Mods │  │Ollama│  │Claude   │  │Direct    │
│CLI  │  │CLI   │  │CLI      │  │API       │
└─────┘  └──────┘  └─────────┘  └──────────┘
```

### Recommended Design Patterns

#### 1. Strategy Pattern for Backend Selection

```python
from abc import ABC, abstractmethod
from typing import Protocol

class LLMBackend(Protocol):
    """Protocol for LLM backends supporting stdin→stdout"""

    def process(self, prompt: str, input_text: str, **kwargs) -> str:
        """Process input text with given prompt"""
        ...

    def supports_structured_output(self) -> bool:
        """Check if backend supports native structured output"""
        ...

class ModsBackend:
    """Mods CLI backend - recommended default"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def process(self, prompt: str, input_text: str, **kwargs) -> str:
        import subprocess
        cmd = ["mods", "-m", self.model, prompt]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True
        )
        return result.stdout

    def supports_structured_output(self) -> bool:
        return False  # Prompt-based JSON only

class OllamaBackend:
    """Ollama local backend - for privacy/offline"""

    def __init__(self, model: str = "llama3"):
        self.model = model

    def process(self, prompt: str, input_text: str, **kwargs) -> str:
        import subprocess
        cmd = ["ollama", "run", self.model, prompt]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True
        )
        return result.stdout

    def supports_structured_output(self) -> bool:
        return False

class InstructorBackend:
    """Instructor library backend - for structured output"""

    def __init__(self, provider: str = "openai/gpt-4o-mini"):
        import instructor
        self.client = instructor.from_provider(provider)

    def process(self, prompt: str, input_text: str, response_model=None, **kwargs):
        from pydantic import BaseModel

        if response_model is None:
            # Simple text response
            class TextResponse(BaseModel):
                result: str
            response_model = TextResponse

        result = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4o-mini"),
            response_model=response_model,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nInput:\n{input_text}"}
            ]
        )
        return result

    def supports_structured_output(self) -> bool:
        return True

# Factory pattern for backend selection
def get_backend(backend_type: str = "mods", **config) -> LLMBackend:
    """Factory to create appropriate backend"""
    backends = {
        "mods": ModsBackend,
        "ollama": OllamaBackend,
        "instructor": InstructorBackend,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}")

    return backends[backend_type](**config)
```

#### 2. Configuration Management

```python
# config.yaml
llm:
  default_backend: "mods"

  backends:
    mods:
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"

    ollama:
      model: "llama3"
      base_url: "http://localhost:11434"

    instructor:
      provider: "openai/gpt-4o-mini"
      max_retries: 3

# config.py
import yaml
from pathlib import Path

class LLMConfig:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str):
        with open(path) as f:
            return yaml.safe_load(f)

    def get_backend_config(self, backend_name: str = None):
        backend_name = backend_name or self.config['llm']['default_backend']
        return self.config['llm']['backends'][backend_name]
```

#### 3. Unified Interface for Scripts

```python
from typing import Optional, Type
from pydantic import BaseModel

class LLMProcessor:
    """Unified interface for LLM processing"""

    def __init__(self, backend: str = "mods", **config):
        self.backend = get_backend(backend, **config)

    def process_text(
        self,
        input_text: str,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None
    ):
        """Process text with optional structured output"""

        if response_model and not self.backend.supports_structured_output():
            # Fallback to prompt-based JSON for non-Instructor backends
            json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema: {response_model.model_json_schema()}"
            result = self.backend.process(json_prompt, input_text)
            # Manual parsing
            import json
            return response_model.model_validate(json.loads(result))

        return self.backend.process(prompt, input_text, response_model=response_model)

# Usage
processor = LLMProcessor(backend="mods", model="gpt-4o-mini")

# Simple text processing
result = processor.process_text(
    input_text="Log file contents...",
    prompt="Identify error patterns"
)

# Structured output (auto-switches to Instructor if needed)
class ErrorPattern(BaseModel):
    pattern: str
    severity: str
    count: int

errors = processor.process_text(
    input_text="Log file contents...",
    prompt="Extract error patterns",
    response_model=ErrorPattern
)
```

### Recommended Implementation Strategy

**Phase 1: Start Simple**

* Use **mods** as default backend for prototyping
* Implement basic subprocess calls
* Focus on core functionality

**Phase 2: Add Structure**

* Integrate **Instructor** for structured outputs
* Implement response models with Pydantic
* Add validation and error handling

**Phase 3: Add Flexibility**

* Implement backend abstraction layer
* Add **ollama** backend for local/offline support
* Create configuration system

**Phase 4: Production Hardening**

* Add retry logic with exponential backoff
* Implement rate limiting
* Add cost tracking for API backends
* Create comprehensive error handling

### Feature Priority Matrix

```
High Priority:
├── Mods CLI integration (default backend)
├── Instructor for structured outputs
└── Configuration system

Medium Priority:
├── Ollama backend (offline support)
├── Error handling & retries
└── Logging & debugging

Low Priority:
├── Claude CLI integration
├── Cost tracking
└── Performance optimization
```

### Testing Strategy

```python
# test_backends.py
import pytest
from pydantic import BaseModel

class TestResponse(BaseModel):
    answer: str

def test_mods_backend():
    backend = ModsBackend(model="gpt-4o-mini")
    result = backend.process("Say 'hello'", "")
    assert "hello" in result.lower()

def test_instructor_structured():
    backend = InstructorBackend(provider="openai/gpt-4o-mini")
    result = backend.process(
        "Extract the answer",
        "The answer is 42",
        response_model=TestResponse
    )
    assert isinstance(result, TestResponse)
    assert "42" in result.answer

@pytest.mark.integration
def test_backend_switching():
    # Test seamless backend switching
    for backend_type in ["mods", "ollama", "instructor"]:
        processor = LLMProcessor(backend=backend_type)
        result = processor.process_text("test", "echo back")
        assert result is not None
```

---

## Additional Notes & Resources

### Alternative Tools Not Covered

**LLM by Simon Willison:** (`github.com/simonw/llm`)

* Excellent Unix-style CLI tool
* Supports plugins for various providers
* Built-in conversation management
* May be worth considering as alternative to mods

**Fabric:** (`github.com/danielmiessler/fabric`)

* Framework for augmenting humans with AI
* Breaks problems into pieces
* More opinionated than mods

**SGPT:** (`github.com/TheR1D/shell_gpt`)

* Shell interaction optimized
* OpenAI focused

### Provider Pricing (Approximate)

**OpenAI:**

* GPT-4o: $2.50/M input, $10/M output tokens
* GPT-4o-mini: $0.15/M input, $0.60/M output
* O1-preview: $15/M input, $60/M output
* O1-mini: $3/M input, $12/M output

**Anthropic:**

* Claude Sonnet 4.5: $3/M input, $15/M output
* Claude Haiku: $0.25/M input, $1.25/M output
* Pro subscription: $20/mo (limited prompts)
* Max subscription: $100-200/mo

**Local (Ollama):**

* Free (uses local compute)
* Electricity costs ~$0.10-0.50/hour for GPU

### Rate Limits (Typical)

**OpenAI Free Tier:**

* 20 requests per minute
* 10,000 tokens per minute

**OpenAI Paid Tier 1:**

* 3,500 requests per minute
* 200,000 tokens per minute

**Anthropic Pro:**

* 10-40 prompts per 5 hours (Claude Code)
* API separate limits

**Ollama:**

* No rate limits (local)
* Limited by hardware performance

### Security Considerations

**API Keys:**

* Never commit to git
* Use environment variables
* Consider tools like `direnv` for project-specific keys
* Rotate keys periodically

**Local Models (Ollama):**

* Full data privacy
* No telemetry by default
* Suitable for sensitive data

**Cloud Models:**

* Data sent to provider APIs
* Review provider privacy policies
* Consider data residency requirements

### Performance Benchmarks (Approximate)

**Local (Ollama llama3:7b):**

* ~20-40 tokens/second (GPU)
* ~3-5 tokens/second (CPU)
* Latency: <100ms first token

**Cloud (OpenAI GPT-4o-mini):**

* ~50-100 tokens/second
* Latency: 200-500ms first token
* Network dependent

**Cloud (Claude Sonnet 4.5):**

* ~40-80 tokens/second
* Latency: 300-600ms first token
* Higher quality, slower

---

## Conclusion

### For Your Python Integration Script:

**Recommended Stack:**

1. **Primary Backend:** Mods CLI
   * Best balance of features and ease-of-use
   * Multi-provider support (cloud + local)
   * Native stdin→stdout design
   * Simple model selection

2. **Structured Output:** Instructor Library
   * Superior to manual JSON parsing
   * Automatic validation and retries
   * Type safety with Pydantic
   * Works with all major providers

3. **Offline Fallback:** Ollama
   * Privacy-focused workflows
   * Development without API costs
   * Offline capability

**Implementation Approach:**

```python
# Start with this simple pattern
from instructor import from_provider
from pydantic import BaseModel

# For structured outputs
class Output(BaseModel):
    field1: str
    field2: int

client = from_provider("openai/gpt-4o-mini")
result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Output,
    messages=[{"role": "user", "content": "..."}]
)

# For simple text processing, use mods via subprocess
import subprocess
result = subprocess.run(
    ["mods", "-m", "gpt-4o-mini", "your prompt"],
    input="text to process",
    capture_output=True,
    text=True
).stdout
```

**Next Steps:**

1. Set up mods CLI with API keys
2. Install Instructor library
3. Create Pydantic models for your outputs
4. Implement backend abstraction for flexibility
5. Add error handling and retries
6. Consider Ollama for local development

### Key Takeaways

* **Don't use Claude CLI for stdin→stdout** - too many limitations, better options exist
* **Instructor >>> manual JSON parsing** - saves hours of development and debugging
* **Mods is the most versatile CLI tool** - works with everything
* **Ollama for privacy/offline** - but requires significant RAM
* **Start simple, add complexity as needed** - don't over-engineer upfront

---

## References & Documentation Links

* **Ollama:** `ollama.com` | `github.com/ollama/ollama`
* **Mods:** `github.com/charmbracelet/mods`
* **Claude CLI:** `docs.claude.com/en/docs/claude-code/cli-reference`
* **Instructor:** `python.useinstructor.com` | `github.com/567-labs/instructor`
* **OpenAI API:** `platform.openai.com/docs`
* **Anthropic API:** `docs.anthropic.com`

---

**Document Version:** 1.0
**Last Updated:** 2025-10-07
**Author:** AI Research (Claude Sonnet 4.5)
**Status:** Ready for implementation
