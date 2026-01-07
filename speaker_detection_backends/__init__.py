"""
Speaker detection backends for different embedding providers.

Available backends:
- speechmatics: API-based speaker identification
- pyannote: Local PyAnnote 3.1 embeddings (requires torch)
- speechbrain: Local SpeechBrain ECAPA-TDNN (requires torch)
"""

from .base import EmbeddingBackend, get_backend

__all__ = ["EmbeddingBackend", "get_backend"]
