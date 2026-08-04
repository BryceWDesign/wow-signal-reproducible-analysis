from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from wow_signal_analysis.analysis_snapshot import (
    AnalysisSnapshot,
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
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
    AnalysisArtifactBundle,
    AnalysisBundleManifest,
    ArtifactGenerationError,
    BundleManifestEntry,
    GeneratedArtifact,
    build_analysis_artifact_bundle,
    verify_written_analysis_artifacts,
    write_analysis_artifact_bundle,
)
from wow_signal_analysis.beam_model import GaussianSearchConfig
from wow_signal_analysis.report import build_analysis_report
from wow_signal_analysis.visualization import build_analysis_figures

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


def _checksum_artifact(
    source: GeneratedArtifact,
    checksum_path: PurePosixPath,
) -> GeneratedArtifact:
    return GeneratedArtifact(
        relative_path=checksum_path,
        media_type="text/plain",
        content=(f"{source.sha256_hex}  {source.relative_path.name}\n").encode("ascii"),
    )


def test_bundle_serialization_is_byte_for_byte_deterministic(
    snapshot: AnalysisSnapshot,
    bundle: AnalysisArtifactBundle,
) -> None:
    second = build_analysis_artifact_bundle(snapshot)
    rendered_report = build_analysis_report(snapshot)
    figures = build_analysis_figures(snapshot)

    assert bundle == second
    assert bundle.bundle_id == ANALYSIS_ARTIFACT_BUNDLE_ID
    assert bundle.analysis_id == snapshot.analysis_id
    assert bundle.snapshot.content == snapshot.to_json().encode("utf-8")
    assert bundle.report.content == rendered_report.content
    assert bundle.beam_fit_figure.content == figures.beam_fit.content
    assert bundle.model_comparison_figure.content == figures.model_comparison.content


def test_bundle_separates_payload_from_manifest_artifacts(
    bundle: AnalysisArtifactBundle,
) -> None:
    assert tuple(artifact.relative_path for artifact in bundle.payload_artifacts) == (
        ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
        ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
        ANALYSIS_REPORT_ARTIFACT_PATH,
        ANALYSIS_REPORT_CHECKSUM_PATH,
        ANALYSIS_BEAM_FIT_FIGURE_PATH,
        ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
        ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
        ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
    )
    assert tuple(artifact.relative_path for artifact in bundle.artifacts) == (
        *(artifact.relative_path for artifact in bundle.payload_artifacts),
        ANALYSIS_BUNDLE_MANIFEST_PATH,
        ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    )
    assert len(bundle.payload_artifacts) == 8
    assert len(bundle.artifacts) == 10


def test_manifest_exactly_inventories_payload_artifacts(
    bundle: AnalysisArtifactBundle,
) -> None:
    manifest = bundle.manifest_model
    decoded = json.loads(bundle.manifest.content)

    assert manifest.schema_version == ANALYSIS_BUNDLE_MANIFEST_SCHEMA_VERSION
    assert manifest.bundle_id == bundle.bundle_id
    assert manifest.analysis_id == bundle.analysis_id
    assert manifest.artifact_count == 8
    assert manifest.total_byte_count == sum(
        artifact.byte_count for artifact in bundle.payload_artifacts
    )
    assert decoded == manifest.to_mapping()
    assert bundle.manifest.content == manifest.to_json().encode("utf-8")
    assert tuple(entry.relative_path for entry in manifest.artifacts) == tuple(
        artifact.relative_path for artifact in bundle.payload_artifacts
    )
    assert tuple(entry.sha256_hex for entry in manifest.artifacts) == tuple(
        artifact.sha256_hex for artifact in bundle.payload_artifacts
    )


def test_checksum_files_use_matching_digests_and_basenames(
    bundle: AnalysisArtifactBundle,
) -> None:
    primary_and_checksum = (
        (bundle.snapshot, bundle.checksum),
        (bundle.report, bundle.report_checksum),
        (bundle.beam_fit_figure, bundle.beam_fit_checksum),
        (
            bundle.model_comparison_figure,
            bundle.model_comparison_checksum,
        ),
        (bundle.manifest, bundle.manifest_checksum),
    )

    for primary, checksum in primary_and_checksum:
        expected = (f"{primary.sha256_hex}  {primary.relative_path.name}\n").encode("ascii")
        assert checksum.content == expected

    assert bundle.total_byte_count == sum(artifact.byte_count for artifact in bundle.artifacts)


def test_bundle_supports_strict_artifact_lookup(
    bundle: AnalysisArtifactBundle,
) -> None:
    for artifact in bundle.artifacts:
        assert bundle.artifact_by_path(artifact.relative_path) is artifact

    with pytest.raises(ArtifactGenerationError, match="found 0"):
        bundle.artifact_by_path(PurePosixPath("artifacts/generated/missing.json"))


def test_svg_artifacts_are_accessible_and_self_contained(
    bundle: AnalysisArtifactBundle,
) -> None:
    expected_identifiers = (
        (
            bundle.beam_fit_figure,
            "wow-signal-beam-fit-v1",
        ),
        (
            bundle.model_comparison_figure,
            "wow-signal-model-comparison-v1",
        ),
    )

    for artifact, figure_id in expected_identifiers:
        svg = artifact.content.decode("utf-8")
        assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        assert svg.endswith("</svg>\n")
        assert f'id="{figure_id}-title"' in svg
        assert f'id="{figure_id}-description"' in svg
        assert "<script" not in svg.lower()
        assert " href=" not in svg.lower()
        assert "xlink:href" not in svg.lower()


def test_writer_creates_and_verifies_all_artifacts(
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
    assert len(written) == 10
    assert tuple(result.relative_path for result in written) == tuple(
        artifact.relative_path for artifact in bundle.artifacts
    )
    assert tuple(result.sha256_hex for result in written) == tuple(
        artifact.sha256_hex for artifact in bundle.artifacts
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
    assert (
        verify_written_analysis_artifacts(
            bundle,
            repository_root,
        )
        == second
    )


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


@pytest.mark.parametrize(
    ("relative_path", "replacement"),
    [
        (
            ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
            b'{"tampered":true}\n',
        ),
        (
            ANALYSIS_REPORT_ARTIFACT_PATH,
            b"# Tampered report\n",
        ),
        (
            ANALYSIS_BEAM_FIT_FIGURE_PATH,
            b'<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
        ),
        (
            ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
            b'<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
        ),
        (
            ANALYSIS_BUNDLE_MANIFEST_PATH,
            b'{"tampered":true}\n',
        ),
    ],
)
def test_verifier_detects_primary_artifact_tampering(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
    relative_path: PurePosixPath,
    replacement: bytes,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )

    target = repository_root.joinpath(*relative_path.parts)
    target.write_bytes(replacement)

    with pytest.raises(
        ArtifactGenerationError,
        match="content mismatch",
    ):
        verify_written_analysis_artifacts(
            bundle,
            repository_root,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
        ANALYSIS_REPORT_CHECKSUM_PATH,
        ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
        ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
        ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    ],
)
def test_verifier_detects_missing_checksum_artifacts(
    tmp_path: Path,
    bundle: AnalysisArtifactBundle,
    relative_path: PurePosixPath,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    write_analysis_artifact_bundle(
        bundle,
        repository_root,
    )

    target = repository_root.joinpath(*relative_path.parts)
    target.unlink()

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


def test_manifest_entries_reject_self_reference_and_duplicate_paths(
    bundle: AnalysisArtifactBundle,
) -> None:
    with pytest.raises(
        ArtifactGenerationError,
        match="must not include the manifest itself",
    ):
        BundleManifestEntry.from_artifact(bundle.manifest)

    entry = BundleManifestEntry.from_artifact(bundle.snapshot)
    with pytest.raises(
        ArtifactGenerationError,
        match="paths must be unique",
    ):
        AnalysisBundleManifest(
            schema_version=ANALYSIS_BUNDLE_MANIFEST_SCHEMA_VERSION,
            bundle_id=bundle.bundle_id,
            analysis_id=bundle.analysis_id,
            artifacts=(entry, entry),
        )


def test_bundle_rejects_snapshot_checksum_drift(
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
        match="snapshot checksum does not match",
    ):
        replace(bundle, checksum=invalid_checksum)


def test_bundle_rejects_figure_checksum_drift(
    bundle: AnalysisArtifactBundle,
) -> None:
    invalid_checksum = GeneratedArtifact(
        relative_path=ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
        media_type="text/plain",
        content=(
            b"0000000000000000000000000000000000000000000000000000000000000000  beam_fit.svg\n"
        ),
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="beam-fit checksum does not match",
    ):
        replace(bundle, beam_fit_checksum=invalid_checksum)


def test_bundle_rejects_manifest_content_or_checksum_drift(
    bundle: AnalysisArtifactBundle,
) -> None:
    invalid_manifest = GeneratedArtifact(
        relative_path=ANALYSIS_BUNDLE_MANIFEST_PATH,
        media_type="application/json",
        content=b"{}\n",
    )
    with pytest.raises(
        ArtifactGenerationError,
        match="does not match the payload artifacts",
    ):
        replace(bundle, manifest=invalid_manifest)

    invalid_checksum = GeneratedArtifact(
        relative_path=ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
        media_type="text/plain",
        content=(
            b"0000000000000000000000000000000000000000000000000000000000000000"
            b"  artifact_manifest.json\n"
        ),
    )
    with pytest.raises(
        ArtifactGenerationError,
        match="bundle manifest checksum does not match",
    ):
        replace(bundle, manifest_checksum=invalid_checksum)


def test_bundle_rejects_invalid_figure_encoding_or_format(
    bundle: AnalysisArtifactBundle,
) -> None:
    invalid_encoding = GeneratedArtifact(
        relative_path=ANALYSIS_BEAM_FIT_FIGURE_PATH,
        media_type="image/svg+xml",
        content=b"\xff",
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="valid UTF-8",
    ):
        replace(bundle, beam_fit_figure=invalid_encoding)

    invalid_format = GeneratedArtifact(
        relative_path=ANALYSIS_BEAM_FIT_FIGURE_PATH,
        media_type="image/svg+xml",
        content=b"<svg></svg>\n",
    )
    invalid_format_checksum = _checksum_artifact(
        invalid_format,
        ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="canonical SVG root",
    ):
        replace(
            bundle,
            beam_fit_figure=invalid_format,
            beam_fit_checksum=invalid_format_checksum,
        )


def test_bundle_rejects_unsafe_figure_content(
    bundle: AnalysisArtifactBundle,
) -> None:
    unsafe_content = bundle.beam_fit_figure.content.replace(
        b"</svg>",
        b"<script>unsafe()</script></svg>",
    )
    unsafe_figure = GeneratedArtifact(
        relative_path=ANALYSIS_BEAM_FIT_FIGURE_PATH,
        media_type="image/svg+xml",
        content=unsafe_content,
    )
    unsafe_checksum = _checksum_artifact(
        unsafe_figure,
        ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
    )

    with pytest.raises(
        ArtifactGenerationError,
        match="executable scripts",
    ):
        replace(
            bundle,
            beam_fit_figure=unsafe_figure,
            beam_fit_checksum=unsafe_checksum,
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
