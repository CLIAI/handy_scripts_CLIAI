#!/usr/bin/env python3
import argparse
import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STT_SCRIPT = Path("../../stt_openai.py")

NORMALIZE_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text):
    text = text.lower()
    text = NORMALIZE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_voices():
    voices_path = SCRIPT_DIR / "voices.txt"
    voices = []
    for line in voices_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        voices.append(line)
    return voices


def read_samples():
    samples = []
    for path in sorted(SCRIPT_DIR.glob("samples/*.test.txt")):
        name = path.name
        if not name.endswith(".test.txt"):
            continue
        stem = name[: -len(".test.txt")]
        samples.append((stem, path))
    return samples


def parse_filter(values):
    if not values:
        return None
    allowed = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                allowed.add(part)
    return allowed if allowed else None


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test ../../stt_openai.py against local audio samples."
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.90,
        help="Minimum similarity ratio to consider a test passing (default: 0.90).",
    )
    parser.add_argument(
        "--voice",
        action="append",
        help="Voice name(s) to include (comma-separated or repeatable).",
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="Sample stem(s) to include (comma-separated or repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete cached transcripts before running.",
    )
    parser.add_argument(
        "--keep-json",
        action="store_true",
        help="Keep *.openai.json files next to audio.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print subprocess output on failures.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY is not set.")
        return 2

    voices = read_voices()
    samples = read_samples()

    allowed_voices = parse_filter(args.voice)
    allowed_samples = parse_filter(args.sample)

    if allowed_voices is not None:
        voices = [voice for voice in voices if voice in allowed_voices]
    if allowed_samples is not None:
        samples = [item for item in samples if item[0] in allowed_samples]

    if not voices:
        print("ERROR: No voices selected. Check voices.txt or --voice.")
        return 1
    if not samples:
        print("ERROR: No samples selected. Check samples/*.test.txt or --sample.")
        return 1

    cases = []
    for stem, sample_path in samples:
        for voice in voices:
            audio_path = SCRIPT_DIR / "audio" / f"{stem}__{voice}.ogg"
            cases.append((stem, sample_path, voice, audio_path))

    missing_audio = [case for case in cases if not case[3].exists()]
    if missing_audio:
        print("ERROR: Missing audio files. Run `make` in evals/stt.")
        for stem, _, voice, audio_path in missing_audio:
            print(f"  - {stem} ({voice}): {audio_path}")
        return 1

    results_dir = SCRIPT_DIR / "results" / "openai"
    results_dir.mkdir(parents=True, exist_ok=True)
    uv_cache_dir = results_dir / "uv-cache"
    uv_cache_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0
    failed = 0

    for stem, sample_path, voice, audio_path in cases:
        total += 1
        expected_text = sample_path.read_text()
        expected_norm = normalize_text(expected_text)

        output_path = results_dir / f"{stem}__{voice}.txt"
        json_path = audio_path.with_name(audio_path.name + ".openai.json")

        if args.force:
            if output_path.exists():
                output_path.unlink()
            if json_path.exists() and not args.keep_json:
                json_path.unlink()

        cmd = [
            str(STT_SCRIPT),
            str(audio_path),
            "--no-meta-message",
            "-q",
            "-o",
            str(output_path),
            "-l",
            "en",  # Specify English for test samples
        ]

        env = os.environ.copy()
        env.setdefault("UV_CACHE_DIR", str(uv_cache_dir))

        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.returncode != 0:
            failed += 1
            print(f"FAIL: {stem} ({voice}) - stt_openai.py returned {proc.returncode}")
            if args.verbose:
                if proc.stdout:
                    print(proc.stdout.strip())
                if proc.stderr:
                    print(proc.stderr.strip())
            continue

        if not output_path.exists():
            failed += 1
            print(f"FAIL: {stem} ({voice}) - missing transcript output")
            continue

        actual_text = output_path.read_text()
        actual_norm = normalize_text(actual_text)

        ratio = difflib.SequenceMatcher(None, expected_norm, actual_norm).ratio()
        if ratio >= args.min_ratio:
            passed += 1
            print(f"PASS: {stem} ({voice}) ratio={ratio:.2f}")
        else:
            failed += 1
            print(f"FAIL: {stem} ({voice}) ratio={ratio:.2f}")
            print(f"  expected: {expected_norm}")
            print(f"  actual:   {actual_norm}")

        if json_path.exists() and not args.keep_json:
            json_path.unlink()

    print(
        "Summary: "
        f"total={total} passed={passed} failed={failed} min_ratio={args.min_ratio:.2f}"
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
