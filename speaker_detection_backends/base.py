"""
Abstract base class for speaker embedding backends.

Each backend implements speaker enrollment and identification using
provider-specific APIs or local models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import importlib
import os

# Import transcript parsing functions
from speaker_detection_backends.transcript import (
    extract_segments_as_tuples,
    load_transcript,
)


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

    @property
    def model_version(self) -> str:
        """Model version string stored in embeddings."""
        return f"{self.name}-unknown"

    def check_embedding_compatibility(
        self,
        embedding: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check if an embedding is compatible with current API version.

        Default implementation checks model_version prefix matches backend name.

        Args:
            embedding: Embedding dict with model_version field

        Returns:
            Dict with:
            - compatible: bool
            - version: embedding's model_version
            - current: current model_version
            - warning: warning message if incompatible
        """
        emb_version = embedding.get("model_version", "unknown")
        compatible = emb_version.startswith(f"{self.name}-")
        result = {
            "compatible": compatible,
            "version": emb_version,
            "current": self.model_version,
            "warning": None,
        }
        if not compatible:
            result["warning"] = (
                f"Embedding created with {emb_version} may not work with "
                f"backend {self.name}. Consider re-enrolling."
            )
        return result

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
        data = load_transcript(transcript_path)
        return extract_segments_as_tuples(data, speaker_label)


# Default backend registry (used if config file not found)
_DEFAULT_BACKENDS = {
    "speechmatics": "speaker_detection_backends.speechmatics_backend",
}

# Cached loaded backends
_LOADED_BACKENDS: Optional[Dict[str, str]] = None


def _load_backends_config() -> Dict[str, str]:
    """
    Load backend registry from config file or use defaults.

    Config file locations (in order of precedence):
    1. $SPEAKER_BACKENDS_CONFIG (if set)
    2. speaker_detection_backends/backends.yaml (relative to this file)

    Returns:
        Dict mapping backend names to module paths
    """
    global _LOADED_BACKENDS

    if _LOADED_BACKENDS is not None:
        return _LOADED_BACKENDS

    config_path = None

    # Check environment variable
    env_path = os.environ.get("SPEAKER_BACKENDS_CONFIG")
    if env_path:
        config_path = Path(env_path)
        if not config_path.exists():
            import sys
            print(f"Warning: SPEAKER_BACKENDS_CONFIG not found: {config_path}", file=sys.stderr)
            config_path = None

    # Check default location
    if config_path is None:
        default_path = Path(__file__).parent / "backends.yaml"
        if default_path.exists():
            config_path = default_path

    # Load config or use defaults
    if config_path:
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)

            backends = {}
            for name, info in data.get("backends", {}).items():
                if isinstance(info, dict):
                    backends[name] = info.get("module", "")
                elif isinstance(info, str):
                    backends[name] = info

            _LOADED_BACKENDS = backends
            return backends
        except ImportError:
            # PyYAML not available, use defaults
            pass
        except Exception as e:
            import sys
            print(f"Warning: Failed to load backends config: {e}", file=sys.stderr)

    _LOADED_BACKENDS = _DEFAULT_BACKENDS.copy()
    return _LOADED_BACKENDS


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
    backends = _load_backends_config()

    if name not in backends:
        available = ", ".join(backends.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    module_path = backends[name]
    module = importlib.import_module(module_path)
    return module.Backend()


def list_backends() -> List[str]:
    """List available backend names."""
    return list(_load_backends_config().keys())


def reload_backends_config() -> None:
    """Force reload of backends config (useful for testing)."""
    global _LOADED_BACKENDS
    _LOADED_BACKENDS = None
