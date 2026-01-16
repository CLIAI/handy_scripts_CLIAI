#!/usr/bin/env python3
"""
Schema migration framework for speaker detection tools.

Handles version upgrades for:
- Speaker profiles (db/*.json)
- Sample metadata (samples/*/*.meta.yaml)

Usage:
    from speaker_detection_backends.migrations import (
        migrate_profile,
        migrate_sample_metadata,
        PROFILE_SCHEMA_VERSION,
        SAMPLE_METADATA_VERSION,
    )

    # Migrate profile on load
    profile = migrate_profile(loaded_profile)

    # Migrate sample metadata on load
    metadata = migrate_sample_metadata(loaded_metadata)
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys

# Current schema versions
PROFILE_SCHEMA_VERSION = 1
SAMPLE_METADATA_VERSION = 2

# Type aliases
ProfileType = Dict[str, Any]
MetadataType = Dict[str, Any]
MigrationFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


# =============================================================================
# Profile Migrations (db/*.json)
# =============================================================================

def _migrate_profile_v0_to_v1(profile: ProfileType) -> ProfileType:
    """
    Migration: v0 (no version) -> v1

    Changes:
    - Add version field
    - Ensure required fields exist
    """
    profile = profile.copy()
    profile["version"] = 1

    # Ensure required fields with defaults
    if "tags" not in profile:
        profile["tags"] = []
    if "embeddings" not in profile:
        profile["embeddings"] = {}
    if "metadata" not in profile:
        profile["metadata"] = {}
    if "name_contexts" not in profile:
        profile["name_contexts"] = {}

    return profile


# Registry of profile migrations: (from_version, to_version) -> migration_func
PROFILE_MIGRATIONS: Dict[Tuple[int, int], MigrationFunc] = {
    (0, 1): _migrate_profile_v0_to_v1,
    # Future migrations:
    # (1, 2): _migrate_profile_v1_to_v2,
}


def migrate_profile(
    profile: ProfileType,
    target_version: Optional[int] = None,
    auto_save: bool = False,
) -> ProfileType:
    """
    Migrate a speaker profile to the target schema version.

    Args:
        profile: The loaded profile dict
        target_version: Target version (default: PROFILE_SCHEMA_VERSION)
        auto_save: If True, the profile was modified (caller should save)

    Returns:
        Migrated profile dict (copy if modified, original if no changes)
    """
    if target_version is None:
        target_version = PROFILE_SCHEMA_VERSION

    current_version = profile.get("version", 0)

    # Already at or above target version
    if current_version >= target_version:
        return profile

    # Apply migrations sequentially
    migrated = profile
    while current_version < target_version:
        next_version = current_version + 1
        migration_key = (current_version, next_version)

        if migration_key not in PROFILE_MIGRATIONS:
            # No migration path available
            print(
                f"Warning: No migration from profile v{current_version} to v{next_version}",
                file=sys.stderr,
            )
            break

        migration_func = PROFILE_MIGRATIONS[migration_key]
        migrated = migration_func(migrated)
        migrated["version"] = next_version
        current_version = next_version

        print(
            f"Migrated profile '{migrated.get('id', '?')}' to v{next_version}",
            file=sys.stderr,
        )

    return migrated


# =============================================================================
# Sample Metadata Migrations (samples/*/*.meta.yaml)
# =============================================================================

def _migrate_metadata_v1_to_v2(meta: MetadataType) -> MetadataType:
    """
    Migration: v1 -> v2

    Changes:
    - Add review section with pending status
    - Add b3sum field (set to empty, requires recomputation)
    - Add source.audio_b3sum field (set to empty, requires recomputation)
    """
    meta = meta.copy()
    meta["version"] = 2

    # Add review section if missing
    if "review" not in meta:
        meta["review"] = {
            "status": "pending",
            "reviewed_at": None,
            "notes": None,
        }

    # Note: b3sum fields may need recomputation by caller
    # We just ensure the structure exists
    if "b3sum" not in meta:
        meta["b3sum"] = None  # Requires recomputation

    if "source" in meta and "audio_b3sum" not in meta["source"]:
        meta["source"]["audio_b3sum"] = None  # Requires recomputation

    return meta


def _migrate_metadata_v0_to_v1(meta: MetadataType) -> MetadataType:
    """
    Migration: v0 (no version) -> v1

    Changes:
    - Add version field
    - Ensure basic structure exists
    """
    meta = meta.copy()
    meta["version"] = 1

    # Ensure basic structure
    if "sample_id" not in meta:
        meta["sample_id"] = "unknown"
    if "source" not in meta:
        meta["source"] = {}
    if "segment" not in meta:
        meta["segment"] = {}
    if "extraction" not in meta:
        meta["extraction"] = {}

    return meta


# Registry of sample metadata migrations
METADATA_MIGRATIONS: Dict[Tuple[int, int], MigrationFunc] = {
    (0, 1): _migrate_metadata_v0_to_v1,
    (1, 2): _migrate_metadata_v1_to_v2,
    # Future migrations:
    # (2, 3): _migrate_metadata_v2_to_v3,
}


def migrate_sample_metadata(
    meta: MetadataType,
    target_version: Optional[int] = None,
) -> MetadataType:
    """
    Migrate sample metadata to the target schema version.

    Args:
        meta: The loaded metadata dict
        target_version: Target version (default: SAMPLE_METADATA_VERSION)

    Returns:
        Migrated metadata dict (copy if modified, original if no changes)
    """
    if target_version is None:
        target_version = SAMPLE_METADATA_VERSION

    current_version = meta.get("version", 0)

    # Already at or above target version
    if current_version >= target_version:
        return meta

    # Apply migrations sequentially
    migrated = meta
    while current_version < target_version:
        next_version = current_version + 1
        migration_key = (current_version, next_version)

        if migration_key not in METADATA_MIGRATIONS:
            print(
                f"Warning: No migration from metadata v{current_version} to v{next_version}",
                file=sys.stderr,
            )
            break

        migration_func = METADATA_MIGRATIONS[migration_key]
        migrated = migration_func(migrated)
        migrated["version"] = next_version
        current_version = next_version

    return migrated


def needs_migration(data: Dict[str, Any], target_version: int) -> bool:
    """Check if data needs migration to reach target version."""
    return data.get("version", 0) < target_version


# =============================================================================
# Batch Migration Utilities
# =============================================================================

def get_migration_plan(
    current_version: int,
    target_version: int,
    migrations: Dict[Tuple[int, int], MigrationFunc],
) -> List[Tuple[int, int]]:
    """
    Get the list of migrations needed to reach target version.

    Returns:
        List of (from, to) version tuples representing the migration path
    """
    plan = []
    version = current_version

    while version < target_version:
        next_version = version + 1
        key = (version, next_version)
        if key in migrations:
            plan.append(key)
            version = next_version
        else:
            break

    return plan


def describe_migrations() -> str:
    """Return a human-readable description of available migrations."""
    lines = [
        "Available Schema Migrations",
        "=" * 40,
        "",
        "Profile Migrations:",
    ]

    for (from_v, to_v), func in sorted(PROFILE_MIGRATIONS.items()):
        doc = func.__doc__ or "No description"
        first_line = doc.strip().split("\n")[0]
        lines.append(f"  v{from_v} -> v{to_v}: {first_line}")

    lines.append("")
    lines.append("Sample Metadata Migrations:")

    for (from_v, to_v), func in sorted(METADATA_MIGRATIONS.items()):
        doc = func.__doc__ or "No description"
        first_line = doc.strip().split("\n")[0]
        lines.append(f"  v{from_v} -> v{to_v}: {first_line}")

    lines.append("")
    lines.append(f"Current Profile Schema Version: {PROFILE_SCHEMA_VERSION}")
    lines.append(f"Current Metadata Schema Version: {SAMPLE_METADATA_VERSION}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_migrations())
