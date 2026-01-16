"""
Audio format profiles for speaker detection backends.

Different backends may require different audio formats for optimal performance.
This module provides a profiles system for specifying audio format requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class AudioProfile:
    """
    Audio format profile specifying requirements for a backend.

    Attributes:
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        format: Audio format/container (default: wav)
        bit_depth: Bit depth for audio samples (default: 16)
        max_duration_sec: Maximum duration in seconds (None = unlimited)
    """

    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    bit_depth: int = 16
    max_duration_sec: Optional[float] = None


# Default profiles for known backends
PROFILES: Dict[str, AudioProfile] = {
    "speechmatics": AudioProfile(
        sample_rate=16000,
        channels=1,
        format="wav",
        bit_depth=16,
    ),
    "pyannote": AudioProfile(
        sample_rate=16000,
        channels=1,
        format="wav",
        bit_depth=16,
    ),
    "default": AudioProfile(),
}


def get_profile(backend_name: str) -> AudioProfile:
    """
    Get audio profile for a backend.

    Args:
        backend_name: Name of the backend (e.g., 'speechmatics', 'pyannote')

    Returns:
        AudioProfile for the backend, or default profile if not found
    """
    return PROFILES.get(backend_name, PROFILES["default"])


def format_ffmpeg_args(profile: AudioProfile) -> List[str]:
    """
    Generate ffmpeg arguments for converting audio to profile specifications.

    Args:
        profile: AudioProfile specifying target format

    Returns:
        List of ffmpeg arguments (without input/output paths)

    Example:
        >>> profile = AudioProfile(sample_rate=16000, channels=1, format="wav")
        >>> format_ffmpeg_args(profile)
        ['-ar', '16000', '-ac', '1', '-f', 'wav']
    """
    args = []

    # Sample rate
    args.extend(["-ar", str(profile.sample_rate)])

    # Channels
    args.extend(["-ac", str(profile.channels)])

    # Format
    args.extend(["-f", profile.format])

    # Bit depth - map to audio codec for wav format
    if profile.format == "wav":
        if profile.bit_depth == 16:
            args.extend(["-acodec", "pcm_s16le"])
        elif profile.bit_depth == 24:
            args.extend(["-acodec", "pcm_s24le"])
        elif profile.bit_depth == 32:
            args.extend(["-acodec", "pcm_s32le"])
        elif profile.bit_depth == 8:
            args.extend(["-acodec", "pcm_u8"])

    return args


def register_profile(name: str, profile: AudioProfile) -> None:
    """
    Register a custom audio profile.

    Args:
        name: Profile name (backend name)
        profile: AudioProfile instance
    """
    PROFILES[name] = profile
