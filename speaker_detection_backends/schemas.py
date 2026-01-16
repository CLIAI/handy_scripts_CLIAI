#!/usr/bin/env python3
"""
Schema validation for speaker detection data structures.

Provides validation functions for:
- Speaker profiles (db/*.json)
- Embedding records
- Sample metadata

Usage:
    from speaker_detection_backends.schemas import (
        validate_profile,
        validate_embedding,
        validate_sample_metadata,
        ValidationError,
    )

    try:
        validate_profile(profile_dict)
    except ValidationError as e:
        print(f"Invalid profile: {e}")
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


# Required fields for each schema
PROFILE_REQUIRED_FIELDS = {"id", "names"}
EMBEDDING_REQUIRED_FIELDS = {"id", "external_id", "created_at"}
SAMPLE_METADATA_REQUIRED_FIELDS = {"sample_id", "source", "segment"}

# Valid trust levels
VALID_TRUST_LEVELS = {"high", "medium", "low", "invalidated"}

# Valid review statuses
VALID_REVIEW_STATUSES = {"pending", "reviewed", "rejected"}


def validate_profile(profile: Dict[str, Any], strict: bool = False) -> List[str]:
    """
    Validate a speaker profile structure.

    Args:
        profile: Profile dict to validate
        strict: If True, raise ValidationError on issues; otherwise return warnings

    Returns:
        List of warning messages (empty if valid)

    Raises:
        ValidationError: If strict=True and validation fails
    """
    warnings = []

    # Check type
    if not isinstance(profile, dict):
        msg = f"Profile must be a dict, got {type(profile).__name__}"
        if strict:
            raise ValidationError(msg)
        return [msg]

    # Check required fields
    missing = PROFILE_REQUIRED_FIELDS - set(profile.keys())
    if missing:
        msg = f"Missing required fields: {', '.join(sorted(missing))}"
        if strict:
            raise ValidationError(msg)
        warnings.append(msg)

    # Validate id
    if "id" in profile:
        if not isinstance(profile["id"], str) or not profile["id"]:
            msg = "Profile 'id' must be a non-empty string"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate names
    if "names" in profile:
        names = profile["names"]
        if not isinstance(names, dict):
            msg = f"Profile 'names' must be a dict, got {type(names).__name__}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)
        elif "default" not in names:
            msg = "Profile 'names' should have a 'default' entry"
            warnings.append(msg)

    # Validate tags
    if "tags" in profile:
        tags = profile["tags"]
        if not isinstance(tags, list):
            msg = f"Profile 'tags' must be a list, got {type(tags).__name__}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)
        elif not all(isinstance(t, str) for t in tags):
            msg = "All tags must be strings"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate embeddings
    if "embeddings" in profile:
        embs = profile["embeddings"]
        if not isinstance(embs, dict):
            msg = f"Profile 'embeddings' must be a dict, got {type(embs).__name__}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)
        else:
            for backend, emb_list in embs.items():
                if not isinstance(emb_list, list):
                    msg = f"Embeddings for '{backend}' must be a list"
                    if strict:
                        raise ValidationError(msg)
                    warnings.append(msg)
                else:
                    for i, emb in enumerate(emb_list):
                        emb_warnings = validate_embedding(emb, strict=False)
                        for w in emb_warnings:
                            warnings.append(f"embeddings.{backend}[{i}]: {w}")

    # Validate version
    if "version" in profile:
        if not isinstance(profile["version"], int):
            msg = f"Profile 'version' must be an int, got {type(profile['version']).__name__}"
            warnings.append(msg)

    return warnings


def validate_embedding(embedding: Dict[str, Any], strict: bool = False) -> List[str]:
    """
    Validate an embedding record structure.

    Args:
        embedding: Embedding dict to validate
        strict: If True, raise ValidationError on issues

    Returns:
        List of warning messages (empty if valid)

    Raises:
        ValidationError: If strict=True and validation fails
    """
    warnings = []

    # Check type
    if not isinstance(embedding, dict):
        msg = f"Embedding must be a dict, got {type(embedding).__name__}"
        if strict:
            raise ValidationError(msg)
        return [msg]

    # Check required fields
    missing = EMBEDDING_REQUIRED_FIELDS - set(embedding.keys())
    if missing:
        msg = f"Missing required fields: {', '.join(sorted(missing))}"
        if strict:
            raise ValidationError(msg)
        warnings.append(msg)

    # Validate id
    if "id" in embedding:
        if not isinstance(embedding["id"], str) or not embedding["id"]:
            msg = "Embedding 'id' must be a non-empty string"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate external_id
    if "external_id" in embedding:
        ext_id = embedding["external_id"]
        if ext_id is not None and not isinstance(ext_id, str):
            msg = f"Embedding 'external_id' must be a string or null, got {type(ext_id).__name__}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate model_version
    if "model_version" in embedding:
        mv = embedding["model_version"]
        if not isinstance(mv, str):
            msg = f"Embedding 'model_version' must be a string, got {type(mv).__name__}"
            warnings.append(msg)
        elif mv == "unknown":
            warnings.append("Embedding has unknown model_version")

    # Validate trust_level
    if "trust_level" in embedding:
        tl = embedding["trust_level"]
        if tl not in VALID_TRUST_LEVELS:
            msg = f"Invalid trust_level '{tl}', expected one of: {', '.join(sorted(VALID_TRUST_LEVELS))}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate created_at (ISO format)
    if "created_at" in embedding:
        ca = embedding["created_at"]
        if isinstance(ca, str):
            try:
                datetime.fromisoformat(ca.replace("Z", "+00:00"))
            except ValueError:
                msg = f"Embedding 'created_at' is not valid ISO format: {ca}"
                warnings.append(msg)
        else:
            msg = f"Embedding 'created_at' must be a string, got {type(ca).__name__}"
            warnings.append(msg)

    # Validate samples structure
    if "samples" in embedding:
        samples = embedding["samples"]
        if isinstance(samples, dict):
            expected_keys = {"reviewed", "unreviewed", "rejected"}
            for key in expected_keys:
                if key in samples:
                    if not isinstance(samples[key], list):
                        msg = f"samples.{key} must be a list"
                        warnings.append(msg)
                    elif not all(isinstance(s, str) for s in samples[key]):
                        msg = f"samples.{key} must contain only strings (b3sum hashes)"
                        warnings.append(msg)
        elif samples is not None:
            msg = f"Embedding 'samples' must be a dict or null, got {type(samples).__name__}"
            warnings.append(msg)

    # Validate source_segments
    if "source_segments" in embedding:
        segs = embedding["source_segments"]
        if segs is not None and not isinstance(segs, list):
            msg = f"Embedding 'source_segments' must be a list or null"
            warnings.append(msg)
        elif isinstance(segs, list):
            for i, seg in enumerate(segs):
                if not isinstance(seg, dict):
                    msg = f"source_segments[{i}] must be a dict"
                    warnings.append(msg)
                elif "start" not in seg or "end" not in seg:
                    msg = f"source_segments[{i}] must have 'start' and 'end' keys"
                    warnings.append(msg)

    return warnings


def validate_sample_metadata(metadata: Dict[str, Any], strict: bool = False) -> List[str]:
    """
    Validate sample metadata structure.

    Args:
        metadata: Metadata dict to validate
        strict: If True, raise ValidationError on issues

    Returns:
        List of warning messages (empty if valid)

    Raises:
        ValidationError: If strict=True and validation fails
    """
    warnings = []

    # Check type
    if not isinstance(metadata, dict):
        msg = f"Metadata must be a dict, got {type(metadata).__name__}"
        if strict:
            raise ValidationError(msg)
        return [msg]

    # Check required fields
    missing = SAMPLE_METADATA_REQUIRED_FIELDS - set(metadata.keys())
    if missing:
        msg = f"Missing required fields: {', '.join(sorted(missing))}"
        if strict:
            raise ValidationError(msg)
        warnings.append(msg)

    # Validate sample_id
    if "sample_id" in metadata:
        if not isinstance(metadata["sample_id"], str):
            msg = "Metadata 'sample_id' must be a string"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate source
    if "source" in metadata:
        source = metadata["source"]
        if not isinstance(source, dict):
            msg = f"Metadata 'source' must be a dict, got {type(source).__name__}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)

    # Validate segment
    if "segment" in metadata:
        segment = metadata["segment"]
        if not isinstance(segment, dict):
            msg = f"Metadata 'segment' must be a dict, got {type(segment).__name__}"
            if strict:
                raise ValidationError(msg)
            warnings.append(msg)
        else:
            if "start_sec" not in segment or "end_sec" not in segment:
                msg = "Metadata 'segment' should have 'start_sec' and 'end_sec'"
                warnings.append(msg)

    # Validate review
    if "review" in metadata:
        review = metadata["review"]
        if not isinstance(review, dict):
            msg = f"Metadata 'review' must be a dict, got {type(review).__name__}"
            warnings.append(msg)
        elif "status" in review:
            status = review["status"]
            if status not in VALID_REVIEW_STATUSES:
                msg = f"Invalid review status '{status}', expected: {', '.join(sorted(VALID_REVIEW_STATUSES))}"
                if strict:
                    raise ValidationError(msg)
                warnings.append(msg)

    # Validate version
    if "version" in metadata:
        if not isinstance(metadata["version"], int):
            msg = f"Metadata 'version' must be an int"
            warnings.append(msg)

    # Validate b3sum
    if "b3sum" in metadata:
        b3sum = metadata["b3sum"]
        if b3sum is not None and not isinstance(b3sum, str):
            msg = "Metadata 'b3sum' must be a string or null"
            warnings.append(msg)
        elif isinstance(b3sum, str) and len(b3sum) < 16:
            msg = f"Metadata 'b3sum' seems too short ({len(b3sum)} chars)"
            warnings.append(msg)

    return warnings


def validate_all(
    profiles: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[Dict[str, Any]]] = None,
    sample_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """
    Validate multiple items and return all warnings.

    Args:
        profiles: List of profiles to validate
        embeddings: List of embeddings to validate
        sample_metadata: List of sample metadata to validate

    Returns:
        Dict mapping item identifiers to their warnings
    """
    all_warnings = {}

    if profiles:
        for i, profile in enumerate(profiles):
            pid = profile.get("id", f"profile[{i}]")
            warnings = validate_profile(profile)
            if warnings:
                all_warnings[f"profile:{pid}"] = warnings

    if embeddings:
        for i, emb in enumerate(embeddings):
            eid = emb.get("id", f"embedding[{i}]")
            warnings = validate_embedding(emb)
            if warnings:
                all_warnings[f"embedding:{eid}"] = warnings

    if sample_metadata:
        for i, meta in enumerate(sample_metadata):
            sid = meta.get("sample_id", f"sample[{i}]")
            warnings = validate_sample_metadata(meta)
            if warnings:
                all_warnings[f"sample:{sid}"] = warnings

    return all_warnings
