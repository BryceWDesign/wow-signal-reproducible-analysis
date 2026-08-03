"""Provenance models and integrity checks for repository evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast
from urllib.parse import urlparse

_SHA256_HEX_LENGTH: Final = 64
_READ_CHUNK_BYTES: Final = 1024 * 1024


class ProvenanceError(ValueError):
    """Raised when provenance metadata is missing, malformed, or inconsistent."""


class ArtifactStatus(StrEnum):
    """Outcome of checking one local artifact against its recorded digest."""

    VERIFIED = "verified"
    MISSING = "missing"
    HASH_MISMATCH = "hash-mismatch"


@dataclass(frozen=True, slots=True)
class SourceReference:
    """One external source supporting specific fields or interpretations."""

    source_id: str
    creator: str
    title: str
    publisher: str
    url: str
    publication_date: date | None
    accessed_date: date
    role: str
    supports: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_id", "creator", "title", "publisher", "role"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceError(f"{field_name} must be non-empty")

        parsed_url = urlparse(self.url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ProvenanceError("source URL must be an absolute HTTPS URL")
        if not self.supports or any(not item.strip() for item in self.supports):
            raise ProvenanceError("supports must contain non-empty entries")
        if len(set(self.supports)) != len(self.supports):
            raise ProvenanceError("supports entries must be unique")
        if self.publication_date is not None and self.publication_date > self.accessed_date:
            raise ProvenanceError("publication_date cannot be after accessed_date")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SourceReference:
        """Create a validated source reference from decoded JSON."""

        publication_text = _optional_text(value, "publication_date")
        publication_date = _parse_date(publication_text, "publication_date")
        return cls(
            source_id=_required_text(value, "source_id"),
            creator=_required_text(value, "creator"),
            title=_required_text(value, "title"),
            publisher=_required_text(value, "publisher"),
            url=_required_text(value, "url"),
            publication_date=publication_date,
            accessed_date=_required_date(value, "accessed_date"),
            role=_required_text(value, "role"),
            supports=_required_text_tuple(value, "supports"),
        )


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """One repository-local dataset file bound to an expected SHA-256 digest."""

    path: str
    sha256: str
    media_type: str
    description: str
    record_count: int

    def __post_init__(self) -> None:
        artifact_path = PurePosixPath(self.path)
        if artifact_path.is_absolute() or not artifact_path.parts or ".." in artifact_path.parts:
            raise ProvenanceError("artifact path must be a safe repository-relative POSIX path")
        if len(self.sha256) != _SHA256_HEX_LENGTH:
            raise ProvenanceError("artifact sha256 must contain 64 hexadecimal characters")
        if self.sha256.lower() != self.sha256:
            raise ProvenanceError("artifact sha256 must use lowercase hexadecimal")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ProvenanceError("artifact sha256 must be hexadecimal") from error
        if not self.media_type.strip() or not self.description.strip():
            raise ProvenanceError("artifact media_type and description must be non-empty")
        if self.record_count <= 0:
            raise ProvenanceError("artifact record_count must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DatasetArtifact:
        """Create a validated artifact record from decoded JSON."""

        return cls(
            path=_required_text(value, "path"),
            sha256=_required_text(value, "sha256"),
            media_type=_required_text(value, "media_type"),
            description=_required_text(value, "description"),
            record_count=_required_int(value, "record_count"),
        )


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Complete provenance record for one normalized analysis dataset."""

    schema_version: int
    dataset_id: str
    title: str
    description: str
    rights_statement: str
    artifacts: tuple[DatasetArtifact, ...]
    sources: tuple[SourceReference, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ProvenanceError("unsupported source manifest schema_version")
        for field_name in ("dataset_id", "title", "description", "rights_statement"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ProvenanceError(f"{field_name} must be non-empty")
        if not self.artifacts:
            raise ProvenanceError("manifest must contain at least one artifact")
        if not self.sources:
            raise ProvenanceError("manifest must contain at least one source")

        artifact_paths = tuple(artifact.path for artifact in self.artifacts)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ProvenanceError("artifact paths must be unique")
        source_ids = tuple(source.source_id for source in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise ProvenanceError("source IDs must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SourceManifest:
        """Create a validated source manifest from decoded JSON."""

        artifact_values = _required_list(value, "artifacts")
        source_values = _required_list(value, "sources")
        return cls(
            schema_version=_required_int(value, "schema_version"),
            dataset_id=_required_text(value, "dataset_id"),
            title=_required_text(value, "title"),
            description=_required_text(value, "description"),
            rights_statement=_required_text(value, "rights_statement"),
            artifacts=tuple(
                DatasetArtifact.from_mapping(_require_mapping(item, "artifact"))
                for item in artifact_values
            ),
            sources=tuple(
                SourceReference.from_mapping(_require_mapping(item, "source"))
                for item in source_values
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    """Integrity-check result for one manifest-bound artifact."""

    path: str
    expected_sha256: str
    actual_sha256: str | None
    status: ArtifactStatus

    @property
    def is_verified(self) -> bool:
        """Return whether the artifact exists and matches its expected digest."""

        return self.status is ArtifactStatus.VERIFIED


def load_source_manifest(path: Path) -> SourceManifest:
    """Load and validate a JSON source manifest."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProvenanceError(f"unable to read source manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise ProvenanceError(f"invalid JSON source manifest: {path}") from error
    return SourceManifest.from_mapping(_require_mapping(payload, "manifest"))


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_artifacts(
    manifest: SourceManifest,
    repository_root: Path,
) -> tuple[ArtifactVerification, ...]:
    """Check every manifest artifact without mutating repository content."""

    root = repository_root.resolve()
    results: list[ArtifactVerification] = []
    for artifact in manifest.artifacts:
        artifact_path = (root / artifact.path).resolve()
        if artifact_path != root and root not in artifact_path.parents:
            raise ProvenanceError(f"artifact escapes repository root: {artifact.path}")
        if not artifact_path.is_file():
            results.append(
                ArtifactVerification(
                    path=artifact.path,
                    expected_sha256=artifact.sha256,
                    actual_sha256=None,
                    status=ArtifactStatus.MISSING,
                )
            )
            continue

        actual_sha256 = sha256_file(artifact_path)
        status = (
            ArtifactStatus.VERIFIED
            if actual_sha256 == artifact.sha256
            else ArtifactStatus.HASH_MISMATCH
        )
        results.append(
            ArtifactVerification(
                path=artifact.path,
                expected_sha256=artifact.sha256,
                actual_sha256=actual_sha256,
                status=status,
            )
        )
    return tuple(results)


def require_verified_artifacts(
    manifest: SourceManifest,
    repository_root: Path,
) -> tuple[str, ...]:
    """Return verified paths or raise when any artifact is absent or altered."""

    results = verify_manifest_artifacts(manifest, repository_root)
    failures = tuple(result for result in results if not result.is_verified)
    if failures:
        details = ", ".join(f"{result.path}={result.status}" for result in failures)
        raise ProvenanceError(f"artifact verification failed: {details}")
    return tuple(result.path for result in results)


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ProvenanceError(f"{field_name} keys must be strings")
    return cast(dict[str, object], value)


def _required_list(value: Mapping[str, object], field_name: str) -> list[object]:
    item = value.get(field_name)
    if not isinstance(item, list):
        raise ProvenanceError(f"{field_name} must be a JSON array")
    return cast(list[object], item)


def _required_text(value: Mapping[str, object], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise ProvenanceError(f"{field_name} must be a non-empty string")
    return item


def _optional_text(value: Mapping[str, object], field_name: str) -> str | None:
    item = value.get(field_name)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        raise ProvenanceError(f"{field_name} must be null or a non-empty string")
    return item


def _required_int(value: Mapping[str, object], field_name: str) -> int:
    item = value.get(field_name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ProvenanceError(f"{field_name} must be an integer")
    return item


def _required_text_tuple(value: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    items = _required_list(value, field_name)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ProvenanceError(f"{field_name} must contain only non-empty strings")
    return tuple(cast(str, item) for item in items)


def _parse_date(value: str | None, field_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ProvenanceError(f"{field_name} must use ISO-8601 YYYY-MM-DD format") from error


def _required_date(value: Mapping[str, object], field_name: str) -> date:
    parsed = _parse_date(_required_text(value, field_name), field_name)
    if parsed is None:
        raise ProvenanceError(f"{field_name} is required")
    return parsed
