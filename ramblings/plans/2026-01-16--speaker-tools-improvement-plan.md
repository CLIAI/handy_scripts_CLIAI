# Speaker Detection Tools - Comprehensive Improvement Plan

**Created**: 2026-01-16
**Status**: Active
**Scope**: speaker_detection, speaker_samples, speaker_detection_backends

This document consolidates findings from 6 parallel analysis agents examining design, documentation, tests, workflows, CLI ergonomics, and future-proofing.

---

## Executive Summary

The speaker detection system has **solid architectural foundations** (abstract backend pattern, JSON/YAML outputs, environment-based configuration) but has critical gaps in:

1. **UNIX Composability** - Tools share hidden state instead of piping
2. **Schema Versioning** - No migration pathway for breaking changes
3. **Test Coverage** - Core workflows (enroll, identify, verify) untested
4. **Documentation** - Missing quick-start, troubleshooting, copy-paste examples
5. **CLI Consistency** - Inconsistent dry-run, no quiet mode, different filters

---

## Table of Contents

1. [UNIX Philosophy Analysis](#1-unix-philosophy-analysis)
2. [User Workflows Coverage](#2-user-workflows-coverage)
3. [Documentation Clarity](#3-documentation-clarity)
4. [CLI Ergonomics](#4-cli-ergonomics)
5. [Future-Proofing & Extensibility](#5-future-proofing--extensibility)
6. [Test Robustness](#6-test-robustness)
7. [Prioritized TODO List](#7-prioritized-todo-list)

---

## 1. UNIX Philosophy Analysis

### Strengths

* **Composability via JSONL**: `speaker_samples segments` outputs JSON-per-line for piping
* **jq integration**: `speaker_detection query` delegates to jq for flexible filtering
* **Environment-based configuration**: `SPEAKERS_EMBEDDINGS_DIR`, `SPEAKER_DETECTION_BACKEND`
* **Text streams**: All commands support `--format json` for downstream processing
* **Clear error handling**: Exit codes 0/1 consistently, stderr vs stdout separation

### Critical Violations

| Issue | Location | Impact |
|-------|----------|--------|
| **Tight coupling via shared DB** | Both tools manipulate `~/.config/speakers_embeddings/` | Cannot use independently |
| **God functions** | `cmd_enroll()` does 10+ things (125 lines) | Hard to compose/test |
| **No stdin support** | `speaker_detection enroll` requires file args | Breaks pipe workflows |
| **Hidden state/side effects** | `save_speaker()` mutates input dict | Unpredictable behavior |
| **Auto schema upgrades** | `speaker_samples review` silently upgrades version | Hidden migrations |

### Proposed Fixes

1. **P1.1** - Extract segments as standalone tool (`speaker_segments`)
2. **P1.2** - Create `SampleProvider` interface to decouple tools
3. **P1.3** - Add `--from-stdin` support to key commands
4. **P1.4** - Make schema versioning explicit (no auto-upgrade)
5. **P1.5** - Decompose `cmd_enroll()` into pipeable components

---

## 2. User Workflows Coverage

### Identified Workflows (12 total)

1. Initial Speaker Enrollment
2. Adding Samples from New Recordings
3. Reviewing and Approving Samples
4. Re-enrollment After Rejection
5. Identifying Speakers in New Audio
6. Exporting for STT Integration
7. Checking Embedding Validity
8. Batch Operations
9. Speaker Profile Management
10. Context-Specific Naming
11. Tag-Based Organization
12. Query and Export

### Test Coverage Matrix

| Workflow | Tested | Missing |
|----------|--------|---------|
| Profile CRUD | YES (test_cli.py) | - |
| Sample extraction | YES | Multi-source scenarios |
| Review workflow | YES | Bulk review |
| Trust levels | YES | Trust degradation flow |
| **Enroll** | NO | Real enrollment with API |
| **Identify** | NO | Speaker identification |
| **Verify** | NO | Speaker verification |
| Batch operations | NO | Multi-speaker extraction |
| Export with context | NO | Name context switching |

### Critical Test Gaps

* **NO enrollment tests** with actual API/audio processing
* **NO identification tests** (core feature!)
* **NO verification tests**
* **NO integration tests** connecting extract → enroll → identify

---

## 3. Documentation Clarity

### Scorecard

| Category | Score | Notes |
|----------|-------|-------|
| Architecture Clarity | 9/10 | Excellent Mermaid diagrams |
| Quick Start Usability | 5/10 | Incomplete examples |
| Error Documentation | 3/10 | No troubleshooting guide |
| Filesystem Clarity | 6/10 | Naming patterns unclear |
| Copy-Paste Examples | 4/10 | Many require context |

### Missing Documentation

1. **QUICKSTART.md** - Complete runnable workflow (audio → profile → identification)
2. **TROUBLESHOOTING.md** - Common errors + fixes
3. **Filesystem reference** - Exact paths, glob patterns, jq examples
4. **Environment variable table** - Single reference (currently scattered)
5. **Schema version changelog** - Migration notes per version

### Documentation Fixes

* **P3.1** - Create QUICKSTART.md with 5-minute copy-paste example
* **P3.2** - Create TROUBLESHOOTING.md for common errors
* **P3.3** - Add filesystem reference with exact naming patterns
* **P3.4** - Consolidate environment variables in one table
* **P3.5** - Document schema versions and migration path

---

## 4. CLI Ergonomics

### Good Patterns

* Subcommand architecture with help
* Multiple output formats (json/table/yaml)
* Destructive operations protected (force flag)
* Short flags for common options (-f, -v, -t, -s, -l)
* Actionable error messages

### Missing Patterns

| Pattern | Status | Impact |
|---------|--------|--------|
| Quiet mode (`-q`) | MISSING | Can't silence for scripting |
| Config file support | MISSING | All config via env/flags |
| Batch operations | MISSING | No bulk add/review |
| Output pagination | MISSING | Large lists dump all |
| Tab completion | MISSING | No shell completions |
| Progress reporting | MISSING | Large operations are silent |

### Inconsistencies Between Tools

| Aspect | speaker_detection | speaker_samples |
|--------|-------------------|-----------------|
| Dry-run short flag | None | `-n` |
| Dry-run commands | enroll only | extract only |
| Format options | json/table/yaml/ids | json/table/yaml |
| Filter semantics | AND (--tags) | exact match (--status) |

### CLI Fixes

* **P4.1** - Add `-n` dry-run to all state-modifying commands
* **P4.2** - Add `-q/--quiet` to both tools
* **P4.3** - Support batch operations (`--from-file`)
* **P4.4** - Add `--limit/--offset` for pagination
* **P4.5** - Generate shell completion scripts

---

## 5. Future-Proofing & Extensibility

### Extensible Components

* **Backend abstraction**: `EmbeddingBackend` ABC is solid
* **Backend registry**: Dynamic loading via `importlib`
* **Transcript formats**: Multiple providers supported
* **Embedding storage**: Flexible schema with provider-specific fields

### Brittle Components

| Issue | Risk | Location |
|-------|------|----------|
| **No schema migrations** | CRITICAL | No version checks in load functions |
| **Hardcoded backend registry** | MEDIUM | `BACKENDS` dict in base.py |
| **Duplicated transcript parsing** | MEDIUM | 3 locations parse transcripts |
| **Speechmatics API coupling** | CRITICAL | Assumes v2 response format |
| **Hardcoded audio params** | MEDIUM | 16kHz mono in ffmpeg calls |

### Missing Abstractions

1. Backend API compatibility layer (handle v2 vs v3)
2. Embedding metadata schema validation
3. Audio format profiles per backend
4. Provider-specific data isolation
5. Configuration externalization (YAML config files)

### Future-Proofing Fixes

* **P5.1** - Implement schema migration framework
* **P5.2** - Add Speechmatics API version compatibility layer
* **P5.3** - Consolidate transcript handling in single module
* **P5.4** - Make backend registry config-driven
* **P5.5** - Add embedding schema validation

---

## 6. Test Robustness

### Scorecard

| Aspect | Grade | Notes |
|--------|-------|-------|
| Test Isolation | A | Fresh temp dirs per test |
| Cleanup | A | Try/finally with ignore_errors |
| Error Paths | C+ | Some covered, missing edge cases |
| JSON Safety | C | No error handling on parsing |
| Environment Safety | D | Direct os.environ mutation (2 tests) |
| Determinism | A | Deterministic audio, fixed test data |

### Critical Issues

1. **Environment pollution** (lines 342, 390 in test_samples_and_trust.py)
   - Direct `os.environ` modification without cleanup

2. **JSON parsing without try/except**
   - Tests crash on malformed output

3. **Hardcoded sample IDs**
   - Tests assume `sample-001` always first

### Missing Test Coverage

* Subprocess failures (missing binaries)
* YAML parsing failures (corrupt metadata)
* Audio file corruption (truncated WAV)
* Concurrent test execution (race conditions)
* Module import failures (missing functions)

### Test Fixes

* **P6.1** - Fix environment pollution in trust level tests
* **P6.2** - Wrap JSON/YAML parsing with error handling
* **P6.3** - Make sample IDs dynamic (don't assume -001)
* **P6.4** - Add integration tests for enroll/identify/verify
* **P6.5** - Add corrupted file handling tests

---

## 7. Prioritized TODO List

### Tier 1: CRITICAL (Blocks further development)

| ID | Task | Files | Effort |
|----|------|-------|--------|
| **T1.1** | Implement schema migration framework | `migrations/__init__.py` (new), `speaker_detection`, `speaker_samples` | M |
| **T1.2** | Add Speechmatics API version compatibility | `speechmatics_backend.py` | M |
| **T1.3** | Create QUICKSTART.md with complete workflow | `QUICKSTART.md` (new) | S |
| **T1.4** | Fix environment pollution in tests | `test_samples_and_trust.py:342,390` | S |
| **T1.5** | Add integration test for enroll workflow | `test_integration.py` (new) | M |

### Tier 2: HIGH (Significant improvement)

| ID | Task | Files | Effort |
|----|------|-------|--------|
| **T2.1** | Consolidate transcript parsing into single module | `transcript.py` (new), `base.py`, `speaker_samples` | M |
| **T2.2** | Add `--from-stdin` support to enroll command | `speaker_detection` | S |
| **T2.3** | Make backend registry config-driven | `base.py`, `backends.yaml` (new) | M |
| **T2.4** | Add `-q/--quiet` and `-n` consistently | `speaker_detection`, `speaker_samples` | S |
| **T2.5** | Create TROUBLESHOOTING.md | `TROUBLESHOOTING.md` (new) | S |
| **T2.6** | Add integration test for identify workflow | `test_integration.py` | M |
| **T2.7** | Add embedding schema validation | `schemas.py` (new) | M |

### Tier 3: MEDIUM (Quality of life)

| ID | Task | Files | Effort |
|----|------|-------|--------|
| **T3.1** | Add batch operations (`--from-file`) | `speaker_detection`, `speaker_samples` | M |
| **T3.2** | Add `--limit/--offset` pagination | `speaker_detection`, `speaker_samples` | S |
| **T3.3** | Extract segments as standalone tool | `speaker_segments` (new) | M |
| **T3.4** | Create SampleProvider interface | `sample_provider.py` (new) | L |
| **T3.5** | Wrap JSON/YAML parsing with error handling | `test_cli.py`, `test_samples_and_trust.py` | S |
| **T3.6** | Add audio format profiles per backend | `audio.py` (new) | M |
| **T3.7** | Document environment variables in single table | `CONTRIBUTING.md` | S |

### Tier 4: NICE-TO-HAVE (Polish)

| ID | Task | Files | Effort |
|----|------|-------|--------|
| **T4.1** | Generate shell completion scripts | `completions/` (new) | M |
| **T4.2** | Add progress reporting for large operations | `speaker_detection`, `speaker_samples` | M |
| **T4.3** | Decompose `cmd_enroll()` into smaller functions | `speaker_detection` | L |
| **T4.4** | Make sample IDs dynamic in tests | `test_samples_and_trust.py` | S |
| **T4.5** | Add event logging/audit trail | `audit.py` (new) | L |

**Legend**: S = Small (< 1 hour), M = Medium (1-4 hours), L = Large (> 4 hours)

---

## How to Use This Plan

### Picking Up a Task

1. Choose a task from the TODO list by ID (e.g., T1.3)
2. Read the relevant analysis section for context
3. Check the "Files" column for affected code
4. Implement following CONTRIBUTING.md guidelines
5. Add/update tests as needed
6. Update this document when task completes

### Task Dependencies

```
T1.1 (migrations) ← T5.x (any schema changes)
T2.1 (transcript module) ← T3.3 (speaker_segments tool)
T2.3 (backend config) ← T5.2 (API version compat)
T2.5 (troubleshooting) ← T1.3 (quickstart, to link from)
```

### Review Checklist

Before marking a task complete:

- [ ] Tests pass locally (`./evals/speaker_detection/test_all.sh`)
- [ ] Tests pass in Docker (`./evals/run_docker_tests.sh`)
- [ ] Documentation updated if behavior changed
- [ ] No new hardcoded values introduced
- [ ] UNIX principles maintained (composability, no hidden state)

---

## Appendix: Analysis Sources

This plan was created by synthesizing outputs from 6 specialized analysis agents:

1. **UNIX Design Principles** - Composability, pipes, single responsibility
2. **User Workflows Coverage** - User stories, test coverage gaps
3. **Documentation Clarity** - Completeness, copy-paste examples
4. **CLI Ergonomics** - Consistency, missing patterns
5. **Future-Proofing** - Schema evolution, API stability
6. **Test Robustness** - Isolation, error handling, flakiness

Each section above corresponds to one agent's findings.
