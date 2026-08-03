"""Deterministic writing and verification of generated analysis artifacts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Final

from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    AnalysisSnapshot,
)
from wow_signal_analysis.report import build_analysis_report
from wow_signal_analysis.visualization import (
    BEAM_FIT_FIGURE_ID,
    MODEL_COMPARISON_FIGURE_ID,
    build_analysis_figures,
)

ANALYSIS_ARTIFACT_BUNDLE_ID: Final = "wow-signal-analysis-artifacts-v1"
ANALYSIS_ARTIFACT_DIRECTORY: Final = PurePosixPath("artifacts/generated")
ANALYSIS_SNAPSHOT_ARTIFACT_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "analysis_snapshot.json"
)
ANALYSIS_SNAPSHOT_CHECKSUM_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "analysis_snapshot.sha256"
)
ANALYSIS_REPORT_ARTIFACT_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "analysis_report.md"
)
ANALYSIS_REPORT_CHECKSUM_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "analysis_report.sha256"
)
ANALYSIS_BEAM_FIT_FIGURE_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "beam_fit.svg"
)
ANALYSIS_BEAM_FIT_CHECKSUM_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "beam_fit.sha256"
)
ANALYSIS_MODEL_COMPARISON_FIGURE_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "model_comparison.svg"
)
ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH: Final = (
    ANALYSIS_ARTIFACT_DIRECTORY / "model_comparison.sha256"
)

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class ArtifactGenerationError(ValueError):
    """Raised when generated analysis artifacts are unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    """One immutable repository-relative generated file."""

    relative_path: PurePosixPath
    media_type: str
    content: bytes

    def __post_init__(self) -> None:
        normalized_path = PurePosixPath(self.relative_path)
        object.__setattr__(self, "relative_path", normalized_path)

        if (
            normalized_path.is_absolute()
            or not normalized_path.parts
            or ".." in normalized_path.parts
            or normalized_path.name in {"", ".", ".."}
        ):
            raise ArtifactGenerationError(
                "relative_path must be a safe repository-relative POSIX path"
            )

        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ArtifactGenerationError("media_type must be non-empty")

        if not isinstance(self.content, bytes) or not self.content:
            raise ArtifactGenerationError(
                "generated artifact content must be non-empty bytes"
            )

    @property
    def byte_count(self) -> int:
        """Return the exact serialized artifact size."""

        return len(self.content)

    @property
    def sha256_hex(self) -> str:
        """Return the lowercase SHA-256 digest of the serialized content."""

        return sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisArtifactBundle:
    """Canonical snapshot, report, figures, and detached checksum files."""

    bundle_id: str
    analysis_id: str
    snapshot: GeneratedArtifact
    checksum: GeneratedArtifact
    report: GeneratedArtifact
    report_checksum: GeneratedArtifact
    beam_fit_figure: GeneratedArtifact
    beam_fit_checksum: GeneratedArtifact
    model_comparison_figure: GeneratedArtifact
    model_comparison_checksum: GeneratedArtifact

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.bundle_id):
            raise ArtifactGenerationError(
                "bundle_id must be a lowercase hyphen-delimited identifier"
            )
        if self.bundle_id != ANALYSIS_ARTIFACT_BUNDLE_ID:
            raise ArtifactGenerationError(
                f"bundle_id must be {ANALYSIS_ARTIFACT_BUNDLE_ID!r}"
            )
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise ArtifactGenerationError(
                f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}"
            )

        self._validate_snapshot_artifacts()
        self._validate_report_artifacts()
        self._validate_figure_artifacts()

    @property
    def snapshot_checksum(self) -> GeneratedArtifact:
        """Return the snapshot checksum under an explicit property name."""

        return self.checksum

    @property
    def artifacts(self) -> tuple[GeneratedArtifact, ...]:
        """Return bundle artifacts in deterministic write order."""

        return (
            self.snapshot,
            self.checksum,
            self.report,
            self.report_checksum,
            self.beam_fit_figure,
            self.beam_fit_checksum,
            self.model_comparison_figure,
            self.model_comparison_checksum,
        )

    @property
    def total_byte_count(self) -> int:
        """Return the combined serialized size of the bundle."""

        return sum(artifact.byte_count for artifact in self.artifacts)

    def artifact_by_path(
        self,
        relative_path: PurePosixPath,
    ) -> GeneratedArtifact:
        """Return the unique artifact stored at one canonical relative path."""

        normalized = PurePosixPath(relative_path)
        matches = tuple(
            artifact
            for artifact in self.artifacts
            if artifact.relative_path == normalized
        )
        if len(matches) != 1:
            raise ArtifactGenerationError(
                f"expected one artifact for {normalized}, found {len(matches)}"
            )
        return matches[0]

    def _validate_snapshot_artifacts(self) -> None:
        _validate_primary_artifact(
            self.snapshot,
            expected_path=ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
            expected_media_type="application/json",
            label="snapshot artifact",
        )
        _validate_checksum_artifact(
            self.checksum,
            source=self.snapshot,
            expected_path=ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
            label="snapshot checksum",
        )

    def _validate_report_artifacts(self) -> None:
        _validate_primary_artifact(
            self.report,
            expected_path=ANALYSIS_REPORT_ARTIFACT_PATH,
            expected_media_type="text/markdown",
            label="report artifact",
        )

        try:
            report_text = self.report.content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactGenerationError(
                "report artifact must contain valid UTF-8"
            ) from error

        if not report_text.startswith("# "):
            raise ArtifactGenerationError(
                "report artifact must begin with a Markdown level-one heading"
            )
        if not report_text.endswith("\n") or report_text.endswith("\n\n"):
            raise ArtifactGenerationError(
                "report artifact must end with exactly one line terminator"
            )
        if "\r" in report_text:
            raise ArtifactGenerationError(
                "report artifact must use LF line endings"
            )

        _validate_checksum_artifact(
            self.report_checksum,
            source=self.report,
            expected_path=ANALYSIS_REPORT_CHECKSUM_PATH,
            label="report checksum",
        )

    def _validate_figure_artifacts(self) -> None:
        _validate_svg_artifact(
            self.beam_fit_figure,
            expected_path=ANALYSIS_BEAM_FIT_FIGURE_PATH,
            expected_figure_id=BEAM_FIT_FIGURE_ID,
            label="beam-fit figure",
        )
        _validate_checksum_artifact(
            self.beam_fit_checksum,
            source=self.beam_fit_figure,
            expected_path=ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
            label="beam-fit checksum",
        )
        _validate_svg_artifact(
            self.model_comparison_figure,
            expected_path=ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
            expected_figure_id=MODEL_COMPARISON_FIGURE_ID,
            label="model-comparison figure",
        )
        _validate_checksum_artifact(
            self.model_comparison_checksum,
            source=self.model_comparison_figure,
            expected_path=ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
            label="model-comparison checksum",
        )


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Verified result of writing or reading one generated artifact."""

    relative_path: PurePosixPath
    absolute_path: Path
    media_type: str
    byte_count: int
    sha256_hex: str

    def __post_init__(self) -> None:
        normalized_path = PurePosixPath(self.relative_path)
        object.__setattr__(self, "relative_path", normalized_path)

        if (
            normalized_path.is_absolute()
            or not normalized_path.parts
            or ".." in normalized_path.parts
        ):
            raise ArtifactGenerationError(
                "result relative_path must be repository-relative"
            )
        if not self.absolute_path.is_absolute():
            raise ArtifactGenerationError(
                "result absolute_path must be absolute"
            )
        if not self.media_type.strip():
            raise ArtifactGenerationError(
                "result media_type must be non-empty"
            )
        if self.byte_count <= 0:
            raise ArtifactGenerationError(
                "result byte_count must be positive"
            )
        if not _SHA256_PATTERN.fullmatch(self.sha256_hex):
            raise ArtifactGenerationError(
                "result sha256_hex must be a lowercase SHA-256 digest"
            )


def build_analysis_artifact_bundle(
    snapshot: AnalysisSnapshot,
) -> AnalysisArtifactBundle:
    """Serialize the snapshot, report, figures, and detached checksums."""

    if not isinstance(snapshot, AnalysisSnapshot):
        raise ArtifactGenerationError(
            "snapshot must be an AnalysisSnapshot"
        )

    rendered_report = build_analysis_report(snapshot)
    figures = build_analysis_figures(snapshot)

    snapshot_artifact = GeneratedArtifact(
        relative_path=ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
        media_type="application/json",
        content=snapshot.to_json().encode("utf-8"),
    )
    report_artifact = GeneratedArtifact(
        relative_path=ANALYSIS_REPORT_ARTIFACT_PATH,
        media_type="text/markdown",
        content=rendered_report.content,
    )
    beam_fit_figure = GeneratedArtifact(
        relative_path=ANALYSIS_BEAM_FIT_FIGURE_PATH,
        media_type="image/svg+xml",
        content=figures.beam_fit.content,
    )
    model_comparison_figure = GeneratedArtifact(
        relative_path=ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
        media_type="image/svg+xml",
        content=figures.model_comparison.content,
    )

    return AnalysisArtifactBundle(
        bundle_id=ANALYSIS_ARTIFACT_BUNDLE_ID,
        analysis_id=snapshot.analysis_id,
        snapshot=snapshot_artifact,
        checksum=_checksum_artifact(
            snapshot_artifact,
            ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
        ),
        report=report_artifact,
        report_checksum=_checksum_artifact(
            report_artifact,
            ANALYSIS_REPORT_CHECKSUM_PATH,
        ),
        beam_fit_figure=beam_fit_figure,
        beam_fit_checksum=_checksum_artifact(
            beam_fit_figure,
            ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
        ),
        model_comparison_figure=model_comparison_figure,
        model_comparison_checksum=_checksum_artifact(
            model_comparison_figure,
            ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
        ),
    )


def write_analysis_artifact_bundle(
    bundle: AnalysisArtifactBundle,
    repository_root: Path,
    *,
    overwrite: bool = True,
) -> tuple[ArtifactWriteResult, ...]:
    """Atomically replace each generated artifact and verify its final bytes."""

    if not isinstance(bundle, AnalysisArtifactBundle):
        raise ArtifactGenerationError(
            "bundle must be an AnalysisArtifactBundle"
        )
    if not isinstance(overwrite, bool):
        raise ArtifactGenerationError("overwrite must be a boolean")

    root = _require_repository_root(repository_root)
    results: list[ArtifactWriteResult] = []

    for artifact in bundle.artifacts:
        destination = _destination_for(
            root,
            artifact.relative_path,
            create_parent=True,
        )
        destination_exists = destination.exists() or destination.is_symlink()

        if destination_exists and not overwrite:
            raise ArtifactGenerationError(
                f"generated artifact already exists: {artifact.relative_path}"
            )

        _write_artifact_atomically(artifact, destination)
        results.append(_result_for_written_artifact(artifact, destination))

    return tuple(results)


def verify_written_analysis_artifacts(
    bundle: AnalysisArtifactBundle,
    repository_root: Path,
) -> tuple[ArtifactWriteResult, ...]:
    """Verify that written artifacts exactly match their in-memory bundle."""

    if not isinstance(bundle, AnalysisArtifactBundle):
        raise ArtifactGenerationError(
            "bundle must be an AnalysisArtifactBundle"
        )

    root = _require_repository_root(repository_root)
    results: list[ArtifactWriteResult] = []

    for artifact in bundle.artifacts:
        destination = _destination_for(
            root,
            artifact.relative_path,
            create_parent=False,
        )

        if destination.is_symlink():
            raise ArtifactGenerationError(
                f"generated artifact must not be a symbolic link: "
                f"{artifact.relative_path}"
            )
        if not destination.is_file():
            raise ArtifactGenerationError(
                f"generated artifact is missing: {artifact.relative_path}"
            )

        try:
            actual_content = destination.read_bytes()
        except OSError as error:
            raise ArtifactGenerationError(
                f"unable to read generated artifact: {artifact.relative_path}"
            ) from error

        if actual_content != artifact.content:
            actual_digest = sha256(actual_content).hexdigest()
            raise ArtifactGenerationError(
                f"generated artifact content mismatch for "
                f"{artifact.relative_path}: expected {artifact.sha256_hex}, "
                f"received {actual_digest}"
            )

        results.append(_result_for_written_artifact(artifact, destination))

    return tuple(results)


def _validate_primary_artifact(
    artifact: GeneratedArtifact,
    *,
    expected_path: PurePosixPath,
    expected_media_type: str,
    label: str,
) -> None:
    if artifact.relative_path != expected_path:
        raise ArtifactGenerationError(
            f"{label} path does not match the canonical path"
        )
    if artifact.media_type != expected_media_type:
        raise ArtifactGenerationError(
            f"{label} media_type must be {expected_media_type}"
        )


def _validate_checksum_artifact(
    checksum: GeneratedArtifact,
    *,
    source: GeneratedArtifact,
    expected_path: PurePosixPath,
    label: str,
) -> None:
    _validate_primary_artifact(
        checksum,
        expected_path=expected_path,
        expected_media_type="text/plain",
        label=label,
    )
    if checksum.content != _checksum_content(source):
        raise ArtifactGenerationError(
            f"{label} does not match the source digest"
        )


def _validate_svg_artifact(
    artifact: GeneratedArtifact,
    *,
    expected_path: PurePosixPath,
    expected_figure_id: str,
    label: str,
) -> None:
    _validate_primary_artifact(
        artifact,
        expected_path=expected_path,
        expected_media_type="image/svg+xml",
        label=label,
    )

    try:
        svg = artifact.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactGenerationError(
            f"{label} must contain valid UTF-8"
        ) from error

    if not svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"'):
        raise ArtifactGenerationError(
            f"{label} must begin with the canonical SVG root"
        )
    if not svg.endswith("</svg>\n") or svg.endswith("</svg>\n\n"):
        raise ArtifactGenerationError(
            f"{label} must end with exactly one line terminator"
        )
    if "\r" in svg:
        raise ArtifactGenerationError(
            f"{label} must use LF line endings"
        )
    if "<script" in svg.lower():
        raise ArtifactGenerationError(
            f"{label} must not contain executable scripts"
        )
    if " href=" in svg.lower() or "xlink:href" in svg.lower():
        raise ArtifactGenerationError(
            f"{label} must not reference external resources"
        )
    if f'id="{expected_figure_id}-title"' not in svg:
        raise ArtifactGenerationError(
            f"{label} does not contain its canonical accessible title ID"
        )
    if f'id="{expected_figure_id}-description"' not in svg:
        raise ArtifactGenerationError(
            f"{label} does not contain its canonical accessible description ID"
        )


def _checksum_artifact(
    source: GeneratedArtifact,
    checksum_path: PurePosixPath,
) -> GeneratedArtifact:
    return GeneratedArtifact(
        relative_path=checksum_path,
        media_type="text/plain",
        content=_checksum_content(source),
    )


def _checksum_content(artifact: GeneratedArtifact) -> bytes:
    line = f"{artifact.sha256_hex}  {artifact.relative_path.name}\n"
    return line.encode("ascii")


def _require_repository_root(repository_root: Path) -> Path:
    root = repository_root.resolve()

    if not root.is_dir():
        raise ArtifactGenerationError(
            f"repository_root must be an existing directory: {root}"
        )

    return root


def _destination_for(
    root: Path,
    relative_path: PurePosixPath,
    *,
    create_parent: bool,
) -> Path:
    parent = root.joinpath(*relative_path.parts[:-1])

    if create_parent:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArtifactGenerationError(
                f"unable to create artifact directory: {relative_path.parent}"
            ) from error
    elif not parent.is_dir():
        raise ArtifactGenerationError(
            f"generated artifact directory is missing: {relative_path.parent}"
        )

    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ArtifactGenerationError(
            f"unable to resolve artifact directory: {relative_path.parent}"
        ) from error

    if not resolved_parent.is_relative_to(root):
        raise ArtifactGenerationError(
            f"artifact path escapes repository_root: {relative_path}"
        )

    return resolved_parent / relative_path.name


def _write_artifact_atomically(
    artifact: GeneratedArtifact,
    destination: Path,
) -> None:
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(artifact.content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)

        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as error:
        raise ArtifactGenerationError(
            f"unable to write generated artifact: {artifact.relative_path}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _result_for_written_artifact(
    artifact: GeneratedArtifact,
    destination: Path,
) -> ArtifactWriteResult:
    try:
        actual_content = destination.read_bytes()
    except OSError as error:
        raise ArtifactGenerationError(
            f"unable to verify generated artifact: {artifact.relative_path}"
        ) from error

    actual_digest = sha256(actual_content).hexdigest()
    if actual_content != artifact.content:
        raise ArtifactGenerationError(
            f"written artifact differs from expected content: "
            f"{artifact.relative_path}"
        )
    if actual_digest != artifact.sha256_hex:
        raise ArtifactGenerationError(
            f"written artifact digest differs from expected digest: "
            f"{artifact.relative_path}"
        )

    return ArtifactWriteResult(
        relative_path=artifact.relative_path,
        absolute_path=destination.resolve(),
        media_type=artifact.media_type,
        byte_count=len(actual_content),
        sha256_hex=actual_digest,
    )
