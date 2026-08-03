from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from wow_signal_analysis.artifact_audit import (
    ARTIFACT_AUDIT_ID,
    ArtifactAuditError,
    audit_generated_artifacts,
    load_analysis_bundle_manifest,
)
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
    ANALYSIS_ARTIFACT_DIRECTORY,
    ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
    ANALYSIS_BEAM_FIT_FIGURE_PATH,
    ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    ANALYSIS_BUNDLE_MANIFEST_PATH,
    ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
    ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
    ANALYSIS_REPORT_ARTIFACT_PATH,
    ANALYSIS_REPORT_CHECKSUM_PATH,
    ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
    ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
    AnalysisBundleManifest,
    GeneratedArtifact,
)


def _checksum_artifact(
    source: GeneratedArtifact,
    checksum_path: PurePosixPath,
) -> GeneratedArtifact:
    return GeneratedArtifact(
        relative_path=checksum_path,
        media_type="text/plain",
        content=(
            f"{source.sha256_hex}  {source.relative_path.name}\n"
        ).encode("ascii"),
    )


def _payload_artifacts() -> tuple[GeneratedArtifact, ...]:
    snapshot = GeneratedArtifact(
        relative_path=ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
        media_type="application/json",
        content=b"{}\n",
    )
    report = GeneratedArtifact(
        relative_path=ANALYSIS_REPORT_ARTIFACT_PATH,
        media_type="text/markdown",
        content=b"# Test report\n",
    )
    beam = GeneratedArtifact(
        relative_path=ANALYSIS_BEAM_FIT_FIGURE_PATH,
        media_type="image/svg+xml",
        content=b'<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
    )
    comparison = GeneratedArtifact(
        relative_path=ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
        media_type="image/svg+xml",
        content=b'<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
    )

    return (
        snapshot,
        _checksum_artifact(
            snapshot,
            ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
        ),
        report,
        _checksum_artifact(
            report,
            ANALYSIS_REPORT_CHECKSUM_PATH,
        ),
        beam,
        _checksum_artifact(
            beam,
            ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
        ),
        comparison,
        _checksum_artifact(
            comparison,
            ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
        ),
    )


def _write_artifacts(
    root: Path,
    payload: tuple[GeneratedArtifact, ...] | None = None,
) -> tuple[GeneratedArtifact, ...]:
    selected_payload = payload or _payload_artifacts()
    generated = root.joinpath(*ANALYSIS_ARTIFACT_DIRECTORY.parts)
    generated.mkdir(parents=True)

    for artifact in selected_payload:
        root.joinpath(*artifact.relative_path.parts).write_bytes(
            artifact.content
        )

    _rewrite_manifest(root, selected_payload)
    return selected_payload


def _rewrite_manifest(
    root: Path,
    payload: tuple[GeneratedArtifact, ...],
) -> None:
    manifest_model = AnalysisBundleManifest.from_artifacts(payload)
    manifest = GeneratedArtifact(
        relative_path=ANALYSIS_BUNDLE_MANIFEST_PATH,
        media_type="application/json",
        content=manifest_model.to_json().encode("utf-8"),
    )
    checksum = _checksum_artifact(
        manifest,
        ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    )
    root.joinpath(*manifest.relative_path.parts).write_bytes(
        manifest.content
    )
    root.joinpath(*checksum.relative_path.parts).write_bytes(
        checksum.content
    )


def test_audit_verifies_manifest_payloads_and_checksums(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)

    report = audit_generated_artifacts(tmp_path)

    assert report.audit_id == ARTIFACT_AUDIT_ID
    assert report.bundle_id == ANALYSIS_ARTIFACT_BUNDLE_ID
    assert report.analysis_id == "wow-signal-canonical-analysis-v1"
    assert report.artifact_count == 8
    assert report.strict_directory
    assert report.total_byte_count == sum(
        artifact.byte_count for artifact in report.artifacts
    )
    assert report.artifact_by_path(
        ANALYSIS_REPORT_ARTIFACT_PATH
    ).media_type == "text/markdown"


def test_manifest_loader_preserves_declared_content_identity(
    tmp_path: Path,
) -> None:
    payload = _write_artifacts(tmp_path)
    manifest_path = tmp_path.joinpath(
        *ANALYSIS_BUNDLE_MANIFEST_PATH.parts
    )

    manifest = load_analysis_bundle_manifest(manifest_path)

    assert manifest.artifact_count == 8
    assert manifest.total_byte_count == sum(
        artifact.byte_count for artifact in payload
    )
    assert tuple(
        entry.relative_path for entry in manifest.artifacts
    ) == tuple(
        artifact.relative_path for artifact in payload
    )


def test_audit_detects_payload_tampering(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    report_path = tmp_path.joinpath(
        *ANALYSIS_REPORT_ARTIFACT_PATH.parts
    )
    report_path.write_bytes(b"# Tampered report\n")

    with pytest.raises(
        ArtifactAuditError,
        match="mismatch for artifacts/generated/analysis_report.md",
    ):
        audit_generated_artifacts(tmp_path)


def test_audit_detects_manifest_checksum_tampering(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    checksum_path = tmp_path.joinpath(
        *ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH.parts
    )
    checksum_path.write_text(
        "0" * 64 + "  artifact_manifest.json\n",
        encoding="ascii",
    )

    with pytest.raises(
        ArtifactAuditError,
        match="bundle manifest checksum does not match",
    ):
        audit_generated_artifacts(tmp_path)


def test_audit_checks_detached_checksum_semantics(
    tmp_path: Path,
) -> None:
    payload = list(_write_artifacts(tmp_path))
    invalid_checksum = GeneratedArtifact(
        relative_path=ANALYSIS_SNAPSHOT_CHECKSUM_PATH,
        media_type="text/plain",
        content=(
            "0" * 64 + "  analysis_snapshot.json\n"
        ).encode("ascii"),
    )
    payload[1] = invalid_checksum
    tmp_path.joinpath(
        *invalid_checksum.relative_path.parts
    ).write_bytes(invalid_checksum.content)
    _rewrite_manifest(tmp_path, tuple(payload))

    with pytest.raises(
        ArtifactAuditError,
        match="analysis_snapshot.sha256 does not match",
    ):
        audit_generated_artifacts(tmp_path)


def test_strict_inventory_rejects_extra_files_but_relaxed_mode_allows_them(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    extra = tmp_path.joinpath(
        *ANALYSIS_ARTIFACT_DIRECTORY.parts,
        "notes.txt",
    )
    extra.write_text("not canonical\n", encoding="utf-8")

    with pytest.raises(
        ArtifactAuditError,
        match="inventory mismatch",
    ):
        audit_generated_artifacts(tmp_path)

    report = audit_generated_artifacts(
        tmp_path,
        strict_directory=False,
    )
    assert not report.strict_directory
    assert report.artifact_count == 8


def test_manifest_rejects_path_traversal_even_with_valid_checksum(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    manifest_path = tmp_path.joinpath(
        *ANALYSIS_BUNDLE_MANIFEST_PATH.parts
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["relative_path"] = "../escape.json"
    manifest_content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_content)
    manifest_artifact = GeneratedArtifact(
        relative_path=ANALYSIS_BUNDLE_MANIFEST_PATH,
        media_type="application/json",
        content=manifest_content,
    )
    checksum = _checksum_artifact(
        manifest_artifact,
        ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
    )
    tmp_path.joinpath(*checksum.relative_path.parts).write_bytes(
        checksum.content
    )

    with pytest.raises(
        ArtifactAuditError,
        match="repository-relative",
    ):
        audit_generated_artifacts(tmp_path)


def test_manifest_rejects_unknown_fields_and_declared_count_drift(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    manifest_path = tmp_path.joinpath(
        *ANALYSIS_BUNDLE_MANIFEST_PATH.parts
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactAuditError,
        match="fields differ from the schema",
    ):
        load_analysis_bundle_manifest(manifest_path)

    payload.pop("unexpected")
    payload["artifact_count"] = 7
    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactAuditError,
        match="artifact_count is 7",
    ):
        load_analysis_bundle_manifest(manifest_path)


def test_audit_rejects_missing_payload_artifact(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path)
    missing_path = tmp_path.joinpath(
        *ANALYSIS_MODEL_COMPARISON_FIGURE_PATH.parts
    )
    missing_path.unlink()

    with pytest.raises(
        ArtifactAuditError,
        match="model_comparison.svg is missing",
    ):
        audit_generated_artifacts(tmp_path)
