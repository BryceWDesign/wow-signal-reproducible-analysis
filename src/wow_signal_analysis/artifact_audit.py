"""Independent audit of generated artifacts from their content manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Final, cast

from wow_signal_analysis.analysis_snapshot import ANALYSIS_SNAPSHOT_ID
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
    ANALYSIS_ARTIFACT_DIRECTORY,
    ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
    ANALYSIS_BEAM_FIT_FIGURE_PATH,
    ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    ANALYSIS_BUNDLE_MANIFEST_PATH,
    ANALYSIS_BUNDLE_MANIFEST_SCHEMA_VERSION,
    ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
    ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
    ANALYSIS_REPORT_ARTIFACT_PATH,
    ANALYSIS_REPORT_CHECKSUM_PATH,
    ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
    ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
    AnalysisBundleManifest,
    ArtifactGenerationError,
    BundleManifestEntry,
)

ARTIFACT_AUDIT_ID: Final = "wow-signal-generated-artifact-audit-v1"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_PAYLOAD_SPEC: Final = (
    (ANALYSIS_SNAPSHOT_ARTIFACT_PATH, "application/json"),
    (ANALYSIS_SNAPSHOT_CHECKSUM_PATH, "text/plain"),
    (ANALYSIS_REPORT_ARTIFACT_PATH, "text/markdown"),
    (ANALYSIS_REPORT_CHECKSUM_PATH, "text/plain"),
    (ANALYSIS_BEAM_FIT_FIGURE_PATH, "image/svg+xml"),
    (ANALYSIS_BEAM_FIT_CHECKSUM_PATH, "text/plain"),
    (ANALYSIS_MODEL_COMPARISON_FIGURE_PATH, "image/svg+xml"),
    (ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH, "text/plain"),
)
_CHECKSUM_PAIRS: Final = (
    (ANALYSIS_SNAPSHOT_ARTIFACT_PATH, ANALYSIS_SNAPSHOT_CHECKSUM_PATH),
    (ANALYSIS_REPORT_ARTIFACT_PATH, ANALYSIS_REPORT_CHECKSUM_PATH),
    (ANALYSIS_BEAM_FIT_FIGURE_PATH, ANALYSIS_BEAM_FIT_CHECKSUM_PATH),
    (
        ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
        ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
    ),
)
_CANONICAL_DIRECTORY_PATHS: Final = frozenset(
    (
        *(path for path, _ in _CANONICAL_PAYLOAD_SPEC),
        ANALYSIS_BUNDLE_MANIFEST_PATH,
        ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    )
)


class ArtifactAuditError(ValueError):
    """Raised when committed generated artifacts fail independent verification."""


@dataclass(frozen=True, slots=True)
class AuditedArtifact:
    """Verified identity for one manifest-inventoried artifact."""

    relative_path: PurePosixPath
    media_type: str
    byte_count: int
    sha256_hex: str

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.relative_path)
        object.__setattr__(self, "relative_path", normalized)

        if (
            normalized.is_absolute()
            or not normalized.parts
            or ".." in normalized.parts
        ):
            raise ArtifactAuditError(
                "audited artifact path must be repository-relative"
            )
        if not self.media_type.strip():
            raise ArtifactAuditError(
                "audited artifact media_type must be non-empty"
            )
        if self.byte_count <= 0:
            raise ArtifactAuditError(
                "audited artifact byte_count must be positive"
            )
        if not _SHA256_PATTERN.fullmatch(self.sha256_hex):
            raise ArtifactAuditError(
                "audited artifact sha256_hex must be a lowercase SHA-256 digest"
            )


@dataclass(frozen=True, slots=True)
class ArtifactAuditReport:
    """Independent verification result for one generated artifact directory."""

    audit_id: str
    repository_root: Path
    bundle_id: str
    analysis_id: str
    manifest_byte_count: int
    manifest_sha256_hex: str
    artifacts: tuple[AuditedArtifact, ...]
    strict_directory: bool

    def __post_init__(self) -> None:
        if self.audit_id != ARTIFACT_AUDIT_ID:
            raise ArtifactAuditError(
                f"audit_id must be {ARTIFACT_AUDIT_ID!r}"
            )
        if not self.repository_root.is_absolute():
            raise ArtifactAuditError(
                "repository_root must be absolute"
            )
        if self.bundle_id != ANALYSIS_ARTIFACT_BUNDLE_ID:
            raise ArtifactAuditError(
                f"bundle_id must be {ANALYSIS_ARTIFACT_BUNDLE_ID!r}"
            )
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise ArtifactAuditError(
                f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}"
            )
        if self.manifest_byte_count <= 0:
            raise ArtifactAuditError(
                "manifest_byte_count must be positive"
            )
        if not _SHA256_PATTERN.fullmatch(self.manifest_sha256_hex):
            raise ArtifactAuditError(
                "manifest_sha256_hex must be a lowercase SHA-256 digest"
            )
        if not isinstance(self.strict_directory, bool):
            raise ArtifactAuditError(
                "strict_directory must be a boolean"
            )

        actual_spec = tuple(
            (artifact.relative_path, artifact.media_type)
            for artifact in self.artifacts
        )
        if actual_spec != _CANONICAL_PAYLOAD_SPEC:
            raise ArtifactAuditError(
                "audited artifacts must preserve the canonical payload contract"
            )

    @property
    def artifact_count(self) -> int:
        """Return the number of independently verified payload artifacts."""

        return len(self.artifacts)

    @property
    def total_byte_count(self) -> int:
        """Return the combined byte count of verified payload artifacts."""

        return sum(artifact.byte_count for artifact in self.artifacts)

    def artifact_by_path(
        self,
        relative_path: PurePosixPath,
    ) -> AuditedArtifact:
        """Return one unique audited artifact by repository-relative path."""

        normalized = PurePosixPath(relative_path)
        matches = tuple(
            artifact
            for artifact in self.artifacts
            if artifact.relative_path == normalized
        )
        if len(matches) != 1:
            raise ArtifactAuditError(
                f"expected one audited artifact for {normalized}, "
                f"found {len(matches)}"
            )
        return matches[0]


def load_analysis_bundle_manifest(
    manifest_path: Path,
) -> AnalysisBundleManifest:
    """Load a generated artifact manifest without trusting package generation state."""

    content = _read_regular_file(
        manifest_path,
        "bundle manifest",
    )
    return _parse_manifest(content)


def audit_generated_artifacts(
    repository_root: Path,
    *,
    strict_directory: bool = True,
) -> ArtifactAuditReport:
    """Verify the manifest, every payload hash, and all detached checksums."""

    if not isinstance(strict_directory, bool):
        raise ArtifactAuditError(
            "strict_directory must be a boolean"
        )

    root = repository_root.resolve()
    if not root.is_dir():
        raise ArtifactAuditError(
            f"repository_root must be an existing directory: {root}"
        )

    generated_directory = _generated_directory(root)
    manifest_path = _absolute_artifact_path(
        root,
        ANALYSIS_BUNDLE_MANIFEST_PATH,
    )
    manifest_checksum_path = _absolute_artifact_path(
        root,
        ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    )
    manifest_content = _read_regular_file(
        manifest_path,
        "bundle manifest",
    )
    manifest_checksum_content = _read_regular_file(
        manifest_checksum_path,
        "bundle manifest checksum",
    )
    manifest_digest = sha256(manifest_content).hexdigest()

    _require_checksum_content(
        manifest_checksum_content,
        digest=manifest_digest,
        source_name=ANALYSIS_BUNDLE_MANIFEST_PATH.name,
        label="bundle manifest checksum",
    )

    manifest = _parse_manifest(manifest_content)
    _require_canonical_manifest(manifest)

    content_by_path: dict[PurePosixPath, bytes] = {}
    audited: list[AuditedArtifact] = []

    for entry in manifest.artifacts:
        absolute_path = _absolute_artifact_path(
            root,
            entry.relative_path,
        )
        content = _read_regular_file(
            absolute_path,
            str(entry.relative_path),
        )
        digest = sha256(content).hexdigest()

        if len(content) != entry.byte_count:
            raise ArtifactAuditError(
                f"byte-count mismatch for {entry.relative_path}: "
                f"expected {entry.byte_count}, received {len(content)}"
            )
        if digest != entry.sha256_hex:
            raise ArtifactAuditError(
                f"digest mismatch for {entry.relative_path}: "
                f"expected {entry.sha256_hex}, received {digest}"
            )

        content_by_path[entry.relative_path] = content
        audited.append(
            AuditedArtifact(
                relative_path=entry.relative_path,
                media_type=entry.media_type,
                byte_count=len(content),
                sha256_hex=digest,
            )
        )

    _verify_detached_checksums(content_by_path)

    if strict_directory:
        _require_exact_directory_inventory(
            root,
            generated_directory,
        )

    return ArtifactAuditReport(
        audit_id=ARTIFACT_AUDIT_ID,
        repository_root=root,
        bundle_id=manifest.bundle_id,
        analysis_id=manifest.analysis_id,
        manifest_byte_count=len(manifest_content),
        manifest_sha256_hex=manifest_digest,
        artifacts=tuple(audited),
        strict_directory=strict_directory,
    )


def _parse_manifest(content: bytes) -> AnalysisBundleManifest:
    try:
        payload: object = json.loads(content.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ArtifactAuditError(
            "bundle manifest must contain valid UTF-8"
        ) from error
    except json.JSONDecodeError as error:
        raise ArtifactAuditError(
            "bundle manifest must contain valid JSON"
        ) from error

    root = _require_mapping(payload, "bundle manifest")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "bundle_id",
            "analysis_id",
            "artifact_count",
            "total_byte_count",
            "artifacts",
        },
        "bundle manifest",
    )

    raw_artifacts = root["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ArtifactAuditError(
            "bundle manifest artifacts must be a JSON array"
        )

    entries = tuple(
        _parse_manifest_entry(
            _require_mapping(item, "bundle manifest entry")
        )
        for item in raw_artifacts
    )

    try:
        manifest = AnalysisBundleManifest(
            schema_version=_required_int(root, "schema_version"),
            bundle_id=_required_text(root, "bundle_id"),
            analysis_id=_required_text(root, "analysis_id"),
            artifacts=entries,
        )
    except ArtifactGenerationError as error:
        raise ArtifactAuditError(str(error)) from error

    declared_count = _required_int(root, "artifact_count")
    declared_bytes = _required_int(root, "total_byte_count")
    if declared_count != manifest.artifact_count:
        raise ArtifactAuditError(
            f"bundle manifest artifact_count is {declared_count}, "
            f"but {manifest.artifact_count} entries are present"
        )
    if declared_bytes != manifest.total_byte_count:
        raise ArtifactAuditError(
            f"bundle manifest total_byte_count is {declared_bytes}, "
            f"but entries declare {manifest.total_byte_count} bytes"
        )

    return manifest


def _parse_manifest_entry(
    value: Mapping[str, object],
) -> BundleManifestEntry:
    _require_exact_keys(
        value,
        {
            "relative_path",
            "media_type",
            "byte_count",
            "sha256",
        },
        "bundle manifest entry",
    )

    try:
        return BundleManifestEntry(
            relative_path=PurePosixPath(
                _required_text(value, "relative_path")
            ),
            media_type=_required_text(value, "media_type"),
            byte_count=_required_int(value, "byte_count"),
            sha256_hex=_required_text(value, "sha256"),
        )
    except ArtifactGenerationError as error:
        raise ArtifactAuditError(str(error)) from error


def _require_canonical_manifest(
    manifest: AnalysisBundleManifest,
) -> None:
    if manifest.schema_version != ANALYSIS_BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ArtifactAuditError(
            "bundle manifest schema does not match the canonical schema"
        )

    actual_spec = tuple(
        (entry.relative_path, entry.media_type)
        for entry in manifest.artifacts
    )
    if actual_spec != _CANONICAL_PAYLOAD_SPEC:
        raise ArtifactAuditError(
            "bundle manifest does not preserve the canonical payload paths, "
            "media types, and order"
        )


def _verify_detached_checksums(
    content_by_path: Mapping[PurePosixPath, bytes],
) -> None:
    for source_path, checksum_path in _CHECKSUM_PAIRS:
        source_content = content_by_path[source_path]
        checksum_content = content_by_path[checksum_path]
        _require_checksum_content(
            checksum_content,
            digest=sha256(source_content).hexdigest(),
            source_name=source_path.name,
            label=str(checksum_path),
        )


def _require_checksum_content(
    content: bytes,
    *,
    digest: str,
    source_name: str,
    label: str,
) -> None:
    expected = f"{digest}  {source_name}\n".encode("ascii")
    if content != expected:
        raise ArtifactAuditError(
            f"{label} does not match the source digest and basename"
        )


def _generated_directory(root: Path) -> Path:
    directory = root.joinpath(*ANALYSIS_ARTIFACT_DIRECTORY.parts)
    if directory.is_symlink():
        raise ArtifactAuditError(
            "generated artifact directory must not be a symbolic link"
        )
    if not directory.is_dir():
        raise ArtifactAuditError(
            f"generated artifact directory is missing: "
            f"{ANALYSIS_ARTIFACT_DIRECTORY}"
        )

    try:
        resolved = directory.resolve(strict=True)
    except OSError as error:
        raise ArtifactAuditError(
            "unable to resolve generated artifact directory"
        ) from error

    if not resolved.is_relative_to(root):
        raise ArtifactAuditError(
            "generated artifact directory escapes repository_root"
        )
    return resolved


def _absolute_artifact_path(
    root: Path,
    relative_path: PurePosixPath,
) -> Path:
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise ArtifactAuditError(
            f"unsafe repository-relative artifact path: {relative_path}"
        )

    candidate = root.joinpath(*relative_path.parts)
    parent = candidate.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ArtifactAuditError(
            f"artifact parent directory is missing: {relative_path.parent}"
        ) from error

    if not resolved_parent.is_relative_to(root):
        raise ArtifactAuditError(
            f"artifact path escapes repository_root: {relative_path}"
        )
    return resolved_parent / candidate.name


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink():
        raise ArtifactAuditError(
            f"{label} must not be a symbolic link"
        )
    if not path.is_file():
        raise ArtifactAuditError(
            f"{label} is missing or is not a regular file"
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise ArtifactAuditError(
            f"unable to read {label}"
        ) from error


def _require_exact_directory_inventory(
    root: Path,
    generated_directory: Path,
) -> None:
    actual_paths: set[PurePosixPath] = set()

    for path in generated_directory.rglob("*"):
        relative_path = PurePosixPath(
            *path.relative_to(root).parts
        )
        if path.is_symlink():
            raise ArtifactAuditError(
                f"generated directory contains a symbolic link: {relative_path}"
            )
        if path.is_dir():
            raise ArtifactAuditError(
                f"generated directory contains an unexpected directory: "
                f"{relative_path}"
            )
        if not path.is_file():
            raise ArtifactAuditError(
                f"generated directory contains a non-regular entry: "
                f"{relative_path}"
            )
        actual_paths.add(relative_path)

    unexpected = sorted(
        actual_paths - _CANONICAL_DIRECTORY_PATHS,
        key=str,
    )
    missing = sorted(
        _CANONICAL_DIRECTORY_PATHS - actual_paths,
        key=str,
    )
    if unexpected or missing:
        details: list[str] = []
        if unexpected:
            details.append(
                "unexpected=" + ",".join(map(str, unexpected))
            )
        if missing:
            details.append(
                "missing=" + ",".join(map(str, missing))
            )
        raise ArtifactAuditError(
            "generated artifact directory inventory mismatch: "
            + "; ".join(details)
        )


def _require_mapping(
    value: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactAuditError(
            f"{label} must be a JSON object"
        )
    if any(not isinstance(key, str) for key in value):
        raise ArtifactAuditError(
            f"{label} keys must be strings"
        )
    return cast(dict[str, object], value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ArtifactAuditError(
            f"{label} fields differ from the schema: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _required_text(
    value: Mapping[str, object],
    field_name: str,
) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise ArtifactAuditError(
            f"{field_name} must be a non-empty string"
        )
    return item


def _required_int(
    value: Mapping[str, object],
    field_name: str,
) -> int:
    item = value.get(field_name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ArtifactAuditError(
            f"{field_name} must be an integer"
        )
    return item
