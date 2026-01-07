"""
Abstract base class for speaker embedding backends.

Each backend implements speaker enrollment and identification using
provider-specific APIs or local models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import importlib


class EmbeddingBackend(ABC):
    """Abstract base class for speaker embedding backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'speechmatics', 'pyannote')."""
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this backend requires an API key."""
        pass

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimensionality of embeddings (None for API-based backends)."""
        return None

    @abstractmethod
    def enroll_speaker(
        self,
        audio_path: Path,
        segments: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Enroll a speaker from audio file.

        Args:
            audio_path: Path to audio file
            segments: Optional list of (start_sec, end_sec) tuples to extract

        Returns:
            Embedding metadata dict containing:
            - external_id: Provider-specific ID (for API backends)
            - file: Path to .npy file (for local backends)
            - model_version: Model/API version used
            - source_audio: Original audio path
            - source_segments: Segments used
        """
        pass

    @abstractmethod
    def identify_speaker(
        self,
        audio_path: Path,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.354,
    ) -> List[Dict[str, Any]]:
        """
        Identify speaker in audio from candidate embeddings.

        Args:
            audio_path: Path to audio file
            candidates: List of speaker profiles with embeddings
            threshold: Similarity threshold (0-1)

        Returns:
            List of matches with:
            - speaker_id: Matched speaker ID
            - similarity: Confidence score
            - segment: (start, end) tuple if applicable
        """
        pass

    def verify_speaker(
        self,
        audio_path: Path,
        speaker_profile: Dict[str, Any],
        threshold: float = 0.354,
    ) -> Dict[str, Any]:
        """
        Verify if audio matches a specific speaker.

        Args:
            audio_path: Path to audio file
            speaker_profile: Speaker profile with embeddings
            threshold: Similarity threshold

        Returns:
            Dict with:
            - match: bool
            - similarity: float
            - embedding_id: Which embedding matched (if any)
        """
        results = self.identify_speaker(audio_path, [speaker_profile], threshold)
        if results:
            return {
                "match": True,
                "similarity": results[0]["similarity"],
                "embedding_id": results[0].get("embedding_id"),
            }
        return {"match": False, "similarity": 0.0, "embedding_id": None}

    def extract_segments_from_transcript(
        self,
        transcript_path: Path,
        speaker_label: str,
    ) -> List[Tuple[float, float]]:
        """
        Extract time segments for a speaker from a transcript JSON.

        Supports AssemblyAI and Speechmatics transcript formats.

        Args:
            transcript_path: Path to transcript JSON file
            speaker_label: Speaker label to extract (e.g., 'A', 'S1')

        Returns:
            List of (start_sec, end_sec) tuples
        """
        import json

        with open(transcript_path) as f:
            data = json.load(f)

        segments = []

        # Try AssemblyAI format (utterances array)
        if "utterances" in data:
            for utt in data["utterances"]:
                if utt.get("speaker") == speaker_label:
                    start = utt.get("start", 0) / 1000.0  # ms to sec
                    end = utt.get("end", 0) / 1000.0
                    segments.append((start, end))

        # Try Speechmatics format (results array with speaker field)
        elif "results" in data:
            current_start = None
            current_end = None
            current_speaker = None

            for item in data["results"]:
                if item.get("type") != "word":
                    continue

                speaker = item.get("speaker", "UU")
                start = item.get("start_time", 0)
                end = item.get("end_time", 0)

                if speaker == speaker_label:
                    if current_speaker != speaker_label:
                        # New segment starts
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


# Registry of available backends
BACKENDS = {
    "speechmatics": "speaker_detection_backends.speechmatics_backend",
    # Future backends:
    # "pyannote": "speaker_detection_backends.pyannote_backend",
    # "speechbrain": "speaker_detection_backends.speechbrain_backend",
}


def get_backend(name: str) -> EmbeddingBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend identifier

    Returns:
        Backend instance

    Raises:
        ValueError: If backend not found
    """
    if name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    module_path = BACKENDS[name]
    module = importlib.import_module(module_path)
    return module.Backend()


def list_backends() -> List[str]:
    """List available backend names."""
    return list(BACKENDS.keys())
