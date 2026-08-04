"""Isolated end-to-end verification of the reproducible release workflow."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Final, TextIO

from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.artifact_audit import (
    ARTIFACT_AUDIT_ID,
    audit_generated_artifacts,
)
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
    AnalysisArtifactBundle,
    build_analysis_artifact_bundle,
    verify_written_analysis_artifacts,
    write_analysis_artifact_bundle,
)
from wow_signal_analysis.repository_contract import (
    verify_repository_contract,
)

RELEASE_VERIFICATION_ID: Final = "wow-signal-release-verification-v1"
RELEASE_REBUILD_COUNT: Final = 2

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class ReleaseVerificationError(ValueError):
    """Raised when the isolated release workflow is not reproducible."""


@dataclass(frozen=True, slots=True)
class ReleaseVerificationReport:
    """Verified summary of one isolated end-to-end release reproduction."""

    verification_id: str
    repository_root: Path
    bundle_id: str
    analysis_id: str
    audit_id: str
    deterministic_rebuild_count: int
    contract_component_count: int
    contract_record_count: int
    artifact_count: int
    payload_artifact_count: int
    total_bundle_byte_count: int
    payload_byte_count: int
    manifest_sha256_hex: str

    def __post_init__(self) -> None:
        if self.verification_id != RELEASE_VERIFICATION_ID:
            raise ReleaseVerificationError(f"verification_id must be {RELEASE_VERIFICATION_ID!r}")
        if not self.repository_root.is_absolute():
            raise ReleaseVerificationError("repository_root must be absolute")
        if self.bundle_id != ANALYSIS_ARTIFACT_BUNDLE_ID:
            raise ReleaseVerificationError(f"bundle_id must be {ANALYSIS_ARTIFACT_BUNDLE_ID!r}")
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise ReleaseVerificationError(f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}")
        if self.audit_id != ARTIFACT_AUDIT_ID:
            raise ReleaseVerificationError(f"audit_id must be {ARTIFACT_AUDIT_ID!r}")
        if self.deterministic_rebuild_count != RELEASE_REBUILD_COUNT:
            raise ReleaseVerificationError(
                "deterministic_rebuild_count does not match the release contract"
            )
        if self.contract_component_count <= 0:
            raise ReleaseVerificationError("contract_component_count must be positive")
        if self.contract_record_count <= 0:
            raise ReleaseVerificationError("contract_record_count must be positive")
        if self.artifact_count <= 0:
            raise ReleaseVerificationError("artifact_count must be positive")
        if self.payload_artifact_count <= 0:
            raise ReleaseVerificationError("payload_artifact_count must be positive")
        if self.payload_artifact_count >= self.artifact_count:
            raise ReleaseVerificationError(
                "artifact_count must include manifest artifacts beyond the payload"
            )
        if self.total_bundle_byte_count <= 0:
            raise ReleaseVerificationError("total_bundle_byte_count must be positive")
        if self.payload_byte_count <= 0:
            raise ReleaseVerificationError("payload_byte_count must be positive")
        if self.payload_byte_count >= self.total_bundle_byte_count:
            raise ReleaseVerificationError("total bundle bytes must exceed payload bytes")
        if not _SHA256_PATTERN.fullmatch(self.manifest_sha256_hex):
            raise ReleaseVerificationError("manifest_sha256_hex must be a lowercase SHA-256 digest")

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible verification result."""

        return {
            "verification_id": self.verification_id,
            "status": "ok",
            "repository_contract": {
                "component_count": self.contract_component_count,
                "record_count": self.contract_record_count,
            },
            "reproduction": {
                "deterministic_rebuild_count": (self.deterministic_rebuild_count),
                "bundle_id": self.bundle_id,
                "analysis_id": self.analysis_id,
                "artifact_count": self.artifact_count,
                "payload_artifact_count": self.payload_artifact_count,
                "total_bundle_byte_count": self.total_bundle_byte_count,
                "payload_byte_count": self.payload_byte_count,
                "manifest_sha256": self.manifest_sha256_hex,
            },
            "independent_audit": {
                "audit_id": self.audit_id,
                "status": "verified",
            },
        }

    def to_json(self) -> str:
        """Serialize the verification result deterministically."""

        return (
            json.dumps(
                self.to_mapping(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def to_text(self) -> str:
        """Return a stable human-readable verification summary."""

        return "\n".join(
            (
                "Release reproduction: verified",
                f"Verification: {self.verification_id}",
                (
                    "Repository contract: "
                    f"{self.contract_component_count} components, "
                    f"{self.contract_record_count} records"
                ),
                (f"Deterministic rebuilds: {self.deterministic_rebuild_count}"),
                f"Bundle: {self.bundle_id}",
                f"Analysis: {self.analysis_id}",
                (f"Artifacts: {self.artifact_count} total, {self.payload_artifact_count} payload"),
                (f"Bytes: {self.total_bundle_byte_count} total, {self.payload_byte_count} payload"),
                (f"Manifest: sha256:{self.manifest_sha256_hex}"),
                f"Independent audit: {self.audit_id}",
                "",
            )
        )


def verify_release_reproduction(
    repository_root: Path,
    output_root: Path,
    *,
    config: SnapshotConfig | None = None,
) -> ReleaseVerificationReport:
    """Rebuild, compare, write, verify, and independently audit a release."""

    source_root = _require_directory(
        repository_root,
        label="repository_root",
    )
    isolated_root = _require_directory(
        output_root,
        label="output_root",
    )
    _require_empty_directory(isolated_root)

    selected_config = config or SnapshotConfig()
    contract = verify_repository_contract(source_root)

    first_snapshot = build_analysis_snapshot(
        source_root,
        config=selected_config,
    )
    first_bundle = build_analysis_artifact_bundle(first_snapshot)

    second_snapshot = build_analysis_snapshot(
        source_root,
        config=selected_config,
    )
    second_bundle = build_analysis_artifact_bundle(second_snapshot)

    _require_identical_bundles(
        first_bundle,
        second_bundle,
    )

    written = write_analysis_artifact_bundle(
        first_bundle,
        isolated_root,
        overwrite=False,
    )
    verified = verify_written_analysis_artifacts(
        first_bundle,
        isolated_root,
    )

    if written != verified:
        raise ReleaseVerificationError("written and independently re-read artifact results differ")

    if len(written) != len(first_bundle.artifacts):
        raise ReleaseVerificationError("written artifact count does not match the bundle")

    audit = audit_generated_artifacts(
        isolated_root,
        strict_directory=True,
    )

    if audit.artifact_count != len(first_bundle.payload_artifacts):
        raise ReleaseVerificationError("independent audit payload count does not match the bundle")
    if audit.total_byte_count != sum(
        artifact.byte_count for artifact in first_bundle.payload_artifacts
    ):
        raise ReleaseVerificationError("independent audit payload bytes do not match the bundle")
    if audit.manifest_sha256_hex != first_bundle.manifest.sha256_hex:
        raise ReleaseVerificationError(
            "independent audit manifest digest does not match the bundle"
        )
    if audit.bundle_id != first_bundle.bundle_id:
        raise ReleaseVerificationError("independent audit bundle identity does not match")
    if audit.analysis_id != first_bundle.analysis_id:
        raise ReleaseVerificationError("independent audit analysis identity does not match")

    return ReleaseVerificationReport(
        verification_id=RELEASE_VERIFICATION_ID,
        repository_root=source_root,
        bundle_id=first_bundle.bundle_id,
        analysis_id=first_bundle.analysis_id,
        audit_id=audit.audit_id,
        deterministic_rebuild_count=RELEASE_REBUILD_COUNT,
        contract_component_count=contract.verified_component_count,
        contract_record_count=contract.total_record_count,
        artifact_count=len(first_bundle.artifacts),
        payload_artifact_count=len(first_bundle.payload_artifacts),
        total_bundle_byte_count=first_bundle.total_byte_count,
        payload_byte_count=audit.total_byte_count,
        manifest_sha256_hex=audit.manifest_sha256_hex,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run isolated release verification and return a process status."""

    parser = argparse.ArgumentParser(
        prog="wow-signal-release-verification",
        description=(
            "Rebuild the complete analysis twice, compare every generated "
            "byte, write into an isolated directory, and independently audit "
            "the resulting artifact manifest."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic JSON instead of human-readable text.",
    )
    arguments = parser.parse_args(None if argv is None else list(argv))

    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr

    try:
        with TemporaryDirectory(prefix="wow-signal-release-verification-") as temporary_directory:
            report = verify_release_reproduction(
                arguments.root,
                Path(temporary_directory),
            )
    except (OSError, ValueError) as error:
        print(
            f"release verification error: {error}",
            file=error_stream,
        )
        return 1

    if arguments.json:
        output_stream.write(report.to_json())
    else:
        output_stream.write(report.to_text())

    return 0


def _require_directory(
    path: Path,
    *,
    label: str,
) -> Path:
    resolved = path.resolve()

    if not resolved.is_dir():
        raise ReleaseVerificationError(f"{label} must be an existing directory: {resolved}")

    return resolved


def _require_empty_directory(path: Path) -> None:
    try:
        has_entries = next(path.iterdir(), None) is not None
    except OSError as error:
        raise ReleaseVerificationError(f"unable to inspect output_root: {path}") from error

    if has_entries:
        raise ReleaseVerificationError("output_root must be empty before release verification")


def _require_identical_bundles(
    first: AnalysisArtifactBundle,
    second: AnalysisArtifactBundle,
) -> None:
    if first.bundle_id != second.bundle_id:
        raise ReleaseVerificationError("deterministic rebuild changed the bundle identity")
    if first.analysis_id != second.analysis_id:
        raise ReleaseVerificationError("deterministic rebuild changed the analysis identity")

    first_signature = _bundle_signature(first)
    second_signature = _bundle_signature(second)

    if first_signature != second_signature:
        raise ReleaseVerificationError("deterministic rebuild produced different artifact bytes")


def _bundle_signature(
    bundle: AnalysisArtifactBundle,
) -> tuple[tuple[PurePosixPath, str, bytes], ...]:
    return tuple(
        (
            artifact.relative_path,
            artifact.media_type,
            artifact.content,
        )
        for artifact in bundle.artifacts
    )


if __name__ == "__main__":
    raise SystemExit(main())
