#!/usr/bin/env python3
"""
Unified transcript parsing for speaker detection tools.

Supports multiple transcript formats:
- Speechmatics (results array with start_time/end_time in seconds)
- AssemblyAI (utterances array with start/end in milliseconds)

Usage:
    from speaker_detection_backends.transcript import (
        detect_transcript_format,
        get_available_speakers,
        extract_segments_from_transcript,
    )

    fmt = detect_transcript_format(data)
    speakers = get_available_speakers(data)
    segments = extract_segments_from_transcript(data, "S1")
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple


def detect_transcript_format(data: Dict[str, Any]) -> str:
    """
    Detect transcript format from JSON structure.

    Args:
        data: Parsed transcript JSON

    Returns:
        'assemblyai', 'speechmatics', or 'unknown'
    """
    # AssemblyAI has "utterances" array
    if "utterances" in data:
        return "assemblyai"

    # Speechmatics has "results" array
    if "results" in data and isinstance(data.get("results"), list):
        results = data["results"]
        if results:
            first = results[0]
            # Check for Speechmatics-style fields
            if "alternatives" in first:
                return "speechmatics"
            if "start_time" in first:
                return "speechmatics"
            # Also check type field
            if first.get("type") in ("word", "punctuation"):
                return "speechmatics"

    return "unknown"


def get_available_speakers(data: Dict[str, Any]) -> List[str]:
    """
    Get list of unique speaker labels in transcript.

    Args:
        data: Parsed transcript JSON

    Returns:
        Sorted list of speaker labels found in transcript
    """
    fmt = detect_transcript_format(data)
    speakers = set()

    if fmt == "assemblyai":
        for utt in data.get("utterances", []):
            if "speaker" in utt:
                speakers.add(utt["speaker"])

    elif fmt == "speechmatics":
        for item in data.get("results", []):
            if item.get("type") != "word":
                continue

            # Check top level speaker field
            if "speaker" in item:
                speakers.add(item["speaker"])

            # Check alternatives (Speechmatics with speaker identification)
            for alt in item.get("alternatives", []):
                if "speaker" in alt:
                    speakers.add(alt["speaker"])

    return sorted(speakers)


def extract_segments_from_transcript(
    data: Dict[str, Any],
    speaker_label: str,
    min_duration: float = 0.5,
    max_gap: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Extract time segments for a speaker from transcript.

    Merges adjacent segments within max_gap and filters by min_duration.

    Args:
        data: Transcript JSON data
        speaker_label: Speaker label to extract (e.g., 'S1', 'Alice')
        min_duration: Minimum segment duration in seconds
        max_gap: Maximum gap between segments to merge

    Returns:
        List of segment dicts with keys: start, end, text
    """
    fmt = detect_transcript_format(data)
    raw_segments = []

    if fmt == "assemblyai":
        raw_segments = _extract_assemblyai(data, speaker_label)
    elif fmt == "speechmatics":
        raw_segments = _extract_speechmatics(data, speaker_label)

    # Merge close segments and filter by duration
    return _merge_and_filter_segments(raw_segments, min_duration, max_gap)


def extract_segments_as_tuples(
    data: Dict[str, Any],
    speaker_label: str,
) -> List[Tuple[float, float]]:
    """
    Extract time segments as (start, end) tuples.

    This is a simpler version without merging, for backend use.

    Args:
        data: Transcript JSON data
        speaker_label: Speaker label to extract

    Returns:
        List of (start_sec, end_sec) tuples
    """
    fmt = detect_transcript_format(data)
    segments = []

    if fmt == "assemblyai":
        for utt in data.get("utterances", []):
            if utt.get("speaker") == speaker_label:
                start = utt.get("start", 0) / 1000.0  # ms to sec
                end = utt.get("end", 0) / 1000.0
                segments.append((start, end))

    elif fmt == "speechmatics":
        current_start = None
        current_end = None
        current_speaker = None

        for item in data.get("results", []):
            if item.get("type") != "word":
                continue

            # Get speaker from item or alternatives
            speaker = item.get("speaker")
            if not speaker:
                alts = item.get("alternatives", [])
                if alts:
                    speaker = alts[0].get("speaker")
            speaker = speaker or "UU"

            start = item.get("start_time", 0)
            end = item.get("end_time", 0)

            if speaker == speaker_label:
                if current_speaker != speaker_label:
                    # Save previous segment if any
                    if current_start is not None and current_speaker == speaker_label:
                        segments.append((current_start, current_end))
                    current_start = start
                current_end = end
                current_speaker = speaker_label
            else:
                # Speaker changed
                if current_speaker == speaker_label and current_start is not None:
                    segments.append((current_start, current_end))
                    current_start = None
                current_speaker = speaker

        # Don't forget last segment
        if current_speaker == speaker_label and current_start is not None:
            segments.append((current_start, current_end))

    return segments


def _extract_assemblyai(data: Dict[str, Any], speaker_label: str) -> List[Dict[str, Any]]:
    """Extract segments from AssemblyAI format transcript."""
    segments = []
    for utt in data.get("utterances", []):
        if utt.get("speaker") == speaker_label:
            start = utt.get("start", 0) / 1000.0  # ms to sec
            end = utt.get("end", 0) / 1000.0
            text = utt.get("text", "")
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _extract_speechmatics(data: Dict[str, Any], speaker_label: str) -> List[Dict[str, Any]]:
    """Extract segments from Speechmatics format transcript."""
    segments = []
    current_start = None
    current_end = None
    current_text = []
    current_speaker = None

    for item in data.get("results", []):
        if item.get("type") != "word":
            continue

        # Get speaker - check alternatives first (speaker identification mode)
        speaker = item.get("speaker")
        content = ""
        alternatives = item.get("alternatives", [])
        if alternatives:
            if not speaker:
                speaker = alternatives[0].get("speaker")
            content = alternatives[0].get("content", "")

        speaker = speaker or "UU"
        start = item.get("start_time", 0)
        end = item.get("end_time", 0)

        if speaker == speaker_label:
            if current_speaker != speaker_label:
                # New segment starts - save previous if any
                if current_start is not None:
                    segments.append({
                        "start": current_start,
                        "end": current_end,
                        "text": " ".join(current_text),
                    })
                current_start = start
                current_text = []
            current_end = end
            current_speaker = speaker_label
            if content:
                current_text.append(content)
        else:
            # Speaker changed
            if current_speaker == speaker_label and current_start is not None:
                segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_text),
                })
                current_start = None
                current_text = []
            current_speaker = speaker

    # Don't forget last segment
    if current_speaker == speaker_label and current_start is not None:
        segments.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_text),
        })

    return segments


def _merge_and_filter_segments(
    segments: List[Dict[str, Any]],
    min_duration: float,
    max_gap: float,
) -> List[Dict[str, Any]]:
    """Merge close segments and filter by minimum duration."""
    merged = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration < min_duration:
            continue

        if merged and (seg["start"] - merged[-1]["end"]) <= max_gap:
            # Merge with previous
            merged[-1]["end"] = seg["end"]
            if seg["text"]:
                merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
        else:
            merged.append(seg)

    return merged


def load_transcript(path: Path) -> Dict[str, Any]:
    """
    Load and parse a transcript JSON file.

    Args:
        path: Path to transcript JSON file

    Returns:
        Parsed transcript data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    import json
    with open(path, "r") as f:
        return json.load(f)
