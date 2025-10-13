#!/usr/bin/env python3
"""
Benchmark script for evaluating speaker mapper performance across different LLM models.

Usage:
    ./benchmark.py --llm-detect 4o-mini
    ./benchmark.py --llm-detect smollm2:360m --llm-endpoint http://localhost:11434/v1
    ./benchmark.py --llm-detect sonnet --output ascii
    ./benchmark.py --llm-detect gemini --tests 001,002,003
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BenchmarkRunner:
    """Runs benchmark tests for speaker mapper."""

    def __init__(self, script_path: str, tests_dir: str, references_dir: str):
        self.script_path = Path(script_path)
        self.tests_dir = Path(tests_dir)
        self.references_dir = Path(references_dir)
        self.results = []

    def discover_tests(self, test_filter: Optional[List[str]] = None) -> List[Path]:
        """Discover all test files in tests directory."""
        all_tests = sorted(self.tests_dir.glob("*.json"))

        if test_filter:
            # Filter by test IDs (e.g., ["001", "002"])
            filtered = []
            for test in all_tests:
                test_id = test.stem
                if any(tid in test_id for tid in test_filter):
                    filtered.append(test)
            return filtered

        return all_tests

    def load_reference(self, test_file: Path) -> Dict:
        """Load reference answer for a test."""
        ref_file = self.references_dir / f"{test_file.stem}.ref.json"
        if not ref_file.exists():
            return None

        with open(ref_file, 'r') as f:
            return json.load(f)

    def run_mapper(self, test_file: Path, llm_args: List[str]) -> Tuple[Dict, float, Optional[str]]:
        """
        Run speaker mapper on test file.

        Returns:
            (result_dict, execution_time, error_message)
        """
        cmd = [
            str(self.script_path),
            "--stdout-only",
            *llm_args,
            str(test_file)
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            elapsed = time.time() - start_time

            if result.returncode != 0:
                return None, elapsed, f"Exit code {result.returncode}: {result.stderr}"

            output = json.loads(result.stdout)
            return output, elapsed, None

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return None, elapsed, "Timeout (>60s)"
        except json.JSONDecodeError as e:
            elapsed = time.time() - start_time
            return None, elapsed, f"JSON decode error: {e}"
        except Exception as e:
            elapsed = time.time() - start_time
            return None, elapsed, f"Error: {e}"

    def score_mapping(self, actual: str, expected_config: Dict) -> Tuple[float, str]:
        """
        Score a single speaker mapping.

        Returns:
            (score, match_type)
        """
        acceptable = expected_config.get("acceptable", [])
        preferred = expected_config.get("preferred", "")

        # Exact match with preferred
        if actual == preferred:
            return 1.0, "exact"

        # Match with acceptable variant
        if actual in acceptable:
            return 1.0, "acceptable"

        # Partial match (substring)
        for variant in acceptable:
            if variant.lower() in actual.lower() or actual.lower() in variant.lower():
                return 0.5, "partial"

        # No match
        return 0.0, "wrong"

    def compare_results(self, actual: Dict, reference: Dict) -> Dict:
        """
        Compare actual results with reference.

        Returns:
            Comparison results with scores
        """
        mappings = actual.get("mappings", {})
        expected_mappings = reference.get("expected_mappings", {})

        speaker_scores = {}
        total_score = 0.0
        max_score = 0.0

        for speaker, expected_config in expected_mappings.items():
            actual_name = mappings.get(speaker, "")
            score, match_type = self.score_mapping(actual_name, expected_config)

            speaker_scores[speaker] = {
                "actual": actual_name,
                "expected": expected_config.get("preferred", ""),
                "score": score,
                "match_type": match_type
            }

            total_score += score
            max_score += 1.0

        accuracy = (total_score / max_score) if max_score > 0 else 0.0
        status = "pass" if accuracy >= 0.75 else "fail"

        return {
            "status": status,
            "accuracy": accuracy,
            "speaker_scores": speaker_scores,
            "total_score": total_score,
            "max_score": max_score
        }

    def run_test(self, test_file: Path, llm_args: List[str]) -> Dict:
        """Run a single test and return results."""
        test_id = test_file.stem
        reference = self.load_reference(test_file)

        if not reference:
            return {
                "test": test_id,
                "status": "error",
                "error": "No reference file found"
            }

        # Run mapper
        result, elapsed, error = self.run_mapper(test_file, llm_args)

        if error:
            return {
                "test": test_id,
                "status": "error",
                "time_sec": elapsed,
                "error": error
            }

        # Compare with reference
        comparison = self.compare_results(result, reference)

        return {
            "test": test_id,
            "description": reference.get("description", ""),
            "status": comparison["status"],
            "accuracy": comparison["accuracy"],
            "time_sec": round(elapsed, 2),
            "speaker_scores": comparison["speaker_scores"],
            "mappings": result.get("mappings", {}),
            "expected": {k: v["preferred"] for k, v in reference["expected_mappings"].items()}
        }

    def run_all_tests(self, llm_args: List[str], test_filter: Optional[List[str]] = None) -> List[Dict]:
        """Run all tests and return results."""
        tests = self.discover_tests(test_filter)
        results = []

        for test_file in tests:
            print(f"Running {test_file.stem}...", file=sys.stderr)
            result = self.run_test(test_file, llm_args)
            results.append(result)

        return results

    def calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from results."""
        total = len(results)
        passed = sum(1 for r in results if r.get("status") == "pass")
        failed = sum(1 for r in results if r.get("status") == "fail")
        errors = sum(1 for r in results if r.get("status") == "error")

        accuracies = [r.get("accuracy", 0) for r in results if "accuracy" in r]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

        times = [r.get("time_sec", 0) for r in results if "time_sec" in r]
        avg_time = sum(times) / len(times) if times else 0
        total_time = sum(times)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_accuracy": avg_accuracy,
            "avg_time_sec": round(avg_time, 2),
            "total_time_sec": round(total_time, 2)
        }


def output_jsonl(results: List[Dict], summary: Dict):
    """Output results in JSONL format."""
    for result in results:
        print(json.dumps(result))
    print(json.dumps({"summary": summary}))


def output_ascii(results: List[Dict], summary: Dict, llm_args: List[str]):
    """Output results in human-readable ASCII table format."""
    # Header
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║           Speaker Mapper Benchmark Results                              ║")
    print(f"║  LLM Args: {' '.join(llm_args):<58}║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    # Results table
    print("TEST                         STATUS    ACCURACY  TIME    DETAILS")
    print("────────────────────────────────────────────────────────────────────────────")

    for result in results:
        test = result.get("test", "")
        status = result.get("status", "error")
        accuracy = result.get("accuracy", 0) * 100
        time_sec = result.get("time_sec", 0)

        # Status symbol
        status_symbol = "✓" if status == "pass" else "✗" if status == "fail" else "⚠"
        status_str = f"{status_symbol} {status.upper()}"

        # Build details
        details = []
        speaker_scores = result.get("speaker_scores", {})
        for speaker, scores in sorted(speaker_scores.items()):
            actual = scores["actual"]
            match_type = scores["match_type"]
            symbol = "✓" if scores["score"] == 1.0 else "~" if scores["score"] > 0 else "✗"
            details.append(f"{speaker}→{actual}{symbol}")

        details_str = ", ".join(details[:3])  # Limit to 3 speakers for display
        if len(speaker_scores) > 3:
            details_str += "..."

        # Handle errors
        if status == "error":
            details_str = result.get("error", "Unknown error")[:40]

        print(f"{test:<28} {status_str:<9} {accuracy:>5.1f}%  {time_sec:>5.1f}s  {details_str}")

    print("────────────────────────────────────────────────────────────────────────────")
    print()

    # Summary
    print("SUMMARY")
    print("════════════════════════════════════════════════════════════════════════════")
    print(f"  Total Tests:      {summary['total']}")
    print(f"  Passed:           {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
    print(f"  Failed:           {summary['failed']} ({summary['failed']/summary['total']*100:.1f}%)")
    if summary['errors'] > 0:
        print(f"  Errors:           {summary['errors']}")
    print(f"  Avg Accuracy:     {summary['avg_accuracy']*100:.1f}%")
    print(f"  Avg Time:         {summary['avg_time_sec']:.2f}s")
    print(f"  Total Time:       {summary['total_time_sec']:.2f}s")
    print("════════════════════════════════════════════════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark speaker mapper performance across different LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with OpenAI model
  ./benchmark.py --llm-detect 4o-mini

  # Run with local Ollama model
  ./benchmark.py --llm-detect smollm2:360m --llm-endpoint http://localhost:11434/v1

  # Run with ASCII output
  ./benchmark.py --llm-detect sonnet --output ascii

  # Run specific tests only
  ./benchmark.py --llm-detect gemini --tests 001,002,003

  # Verbose mode
  ./benchmark.py --llm-detect 4o-mini -v --output ascii
        """
    )

    # Benchmark-specific arguments
    parser.add_argument(
        "--output",
        choices=["jsonl", "ascii"],
        default="jsonl",
        help="Output format (default: jsonl)"
    )
    parser.add_argument(
        "--tests",
        help="Comma-separated test IDs to run (e.g., 001,002,003)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show stderr from mapper)"
    )

    # LLM arguments (pass-through to speaker mapper)
    parser.add_argument(
        "--llm-detect",
        required=True,
        metavar="MODEL",
        help="LLM model to use (e.g., 4o-mini, smollm2:360m, sonnet)"
    )
    parser.add_argument(
        "--llm-endpoint",
        metavar="URL",
        help="Custom LLM endpoint URL"
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        metavar="N",
        help="Number of utterances to send to LLM"
    )

    args = parser.parse_args()

    # Build LLM arguments list
    llm_args = ["--llm-detect", args.llm_detect]
    if args.llm_endpoint:
        llm_args.extend(["--llm-endpoint", args.llm_endpoint])
    if args.llm_sample_size:
        llm_args.extend(["--llm-sample-size", str(args.llm_sample_size)])

    # Parse test filter
    test_filter = None
    if args.tests:
        test_filter = [t.strip() for t in args.tests.split(",")]

    # Setup paths
    script_dir = Path(__file__).parent
    script_path = script_dir / "../../stt_assemblyai_speaker_mapper.py"
    tests_dir = script_dir / "tests"
    references_dir = script_dir / "references"

    # Run benchmark
    runner = BenchmarkRunner(script_path, tests_dir, references_dir)
    results = runner.run_all_tests(llm_args, test_filter)
    summary = runner.calculate_summary(results)

    # Output results
    if args.output == "jsonl":
        output_jsonl(results, summary)
    else:
        output_ascii(results, summary, llm_args)


if __name__ == "__main__":
    main()
