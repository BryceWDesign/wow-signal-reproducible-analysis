from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from wow_signal_analysis.analysis_snapshot import (
    AnalysisSnapshot,
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
    ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
    ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
    AnalysisArtifactBundle,
    ArtifactGenerationError,
    GeneratedArtifact,
    build_analysis_artifact_bundle,
    verify_written_analysis_artifacts,
    write_analysis_artifact_bundle,
)
from wow_signal_analysis.beam_model import GaussianSearchConfig

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FAST_CONFIG = SnapshotConfig(
    gaussian_search=GaussianSearchConfig(
        grid_points=21,
        refinement_rounds=3,
    )
)


@pytest.fixture(scope="module")
def snapshot() -> AnalysisSnapshot:
    return build_analysis_snapshot(
        _REPOSITORY_ROOT,
        config=_FAST_CONFIG,
    )


@pytest.fixture(scope="module")
def bundle(snapshot: AnalysisSnapshot) -> AnalysisArtifactBundle:
    return build_analysis_artifact_bundle(snapshot)


def test_bundle_serialization_is_byte_for_byte_deterministic(
    snapshot: AnalysisSnapshot,
    bundle: AnalysisArtifactBundle,
) -> None:
    second = build_analysis_artifact_bundle(snapshot)

    assert bundle == second
    assert bundle.bundle_id == ANALYSIS_ARTIFACT_BUNDLE_ID
    assert bundle.analysis_id == snapshot.analysis_id
    assert bundle.snapshot.content == snapshot.to_json().encode("utf-8")
    assert bundle.snapshot.relative_path == ANALYSIS_SNAPSHOT_ARTIFACT_PATH
    assert bundle.checksum.relative_path == ANALYSIS_SNAPSHOT_CHECKSUM_PATH


def test_checksum_file_uses_the_snapshot_digest_and_basename(
    bundle: AnalysisArtifactBundle,
) -> None:
    expected = (
        f"{bundle.snapshot.sha256_hex}  analysis_snapshot.json\n"
    ).encode("ascii")

    assert bundle.checksum.content == expected
    assert bundle.snapshot.sha256_hex in bundle.checksum.content.decode("ascii")
    assert bundle.total_byte_count == sum(
        artifact.byte_count for artifact in bundle.artifacts
    )


def test_bundle_supports_strict_artifact_lookup(
    bundle: AnalysisArtifactBundle,
) -> None:
    assert (
        bundle.artifact_by_path(ANALYSIS_SNAPSHOT_ARTIFACT_PATH)
        is bundle.snapshot
    )
    assert (
        bundle.artifact_by_path(ANALYSIS_SNAPSHOT_CHECKSUM_PATH)
        is bundle.checksum
    )

    with pytest.raises(ArtifactGenerationError, match="found 0"):
        bundle.artifact_by_path(
            PurePosixPath("artifacts/generated/missing.json")
        )


def test_writer_creates_and_verifies_both_artifacts(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    written = write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )
    verified = verify_written_analysis_artifacts(
        bundle,
        repository_root,
    )

    assert written == verified
    assert tuple(result.relative_path for result in written) == (
        ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
        ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
    )
    assert tuple(result.sha256_hex for result in written) == (
        bundle.snapshot.sha256_hex,
        bundle.checksum.sha256_hex,
    )
    assert all(result.absolute_path.is_file() for result in written)


def test_writer_replaces_existing_artifacts_deterministically(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    first = write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )
    second = write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )

    assert first == second
    assert verify_written_analysis_artifacts(
        bundle,
        repository_root,
    ) == second


def test_writer_can_refuse_overwrite(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="already exists",
    ):
        write_analysis_artifact_bundle(
            bundle,
            repository_root,
            overwrite=False,
        )


def test_verifier_detects_snapshot_tampering(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )

    snapshot_path = repository_root.joinpath(
        *ANALYSIS_SNAPSHOT_ARTIFACT_PATH.parts
    )
    snapshot_path.write_text(
        '{"tampered":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="content mismatch",
    ):
        verify_written_analysis_artifacts(
            bundle,
            repository_root,
        )


def test_verifier_detects_missing_artifact(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )

    checksum_path = repository_root.joinpath(
        *ANALYSIS_SNAPSHOT_CHECKSUM_PATH.parts
    )
    checksum_path.unlink()

    with pytest.raises(
        ArtifactGenerationError,
        match="is missing",
    ):
        verify_written_analysis_artifacts(
            bundle,
            repository_root,
        )


def test_generated_artifact_rejects_unsafe_or_empty_content() -> None:
    with pytest.raises(
        ArtifactGenerationError,
        match="safe repository-relative",
    ):
        GeneratedArtifact(
            relative_path=PurePosixPath("../outside.json"),
            media_type="application/json",
            content=b"{}\n",
        )

    with pytest.raises(
        ArtifactGenerationError,
        match="non-empty bytes",
    ):
        GeneratedArtifact(
            relative_path=PurePosixPath("artifacts/generated/empty.json"),
            media_type="application/json",
            content=b"",
        )


def test_bundle_rejects_checksum_drift(
    bundle: AnalysisArtifactBundle,
) -> None:
    invalid_checksum = GeneratedArtifact(
        relative_path=ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
        media_type="text/plain",
        content=(
            b"0000000000000000000000000000000000000000000000000000000000000000"
            b"  analysis_snapshot.json\n"
        ),
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="does not match",
    ):
        AnalysisArtifactBundle(
            bundle_id=bundle.bundle_id,
            analysis_id=bundle.analysis_id,
            snapshot=bundle.snapshot,
            checksum=invalid_checksum,
        )


def test_writer_requires_an_existing_repository_root(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(
        ArtifactGenerationError,
        match="existing directory",
    ):
        write_analysis_artifact_bundle(
            bundle,
            missing_root,
        )
