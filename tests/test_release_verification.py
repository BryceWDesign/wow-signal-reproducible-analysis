from __future__ import annotations

import json
from io import StringIO
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast

import pytest

import wow_signal_analysis.release_verification as release
from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    AnalysisSnapshot,
)
from wow_signal_analysis.artifact_audit import ARTIFACT_AUDIT_ID
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
    AnalysisArtifactBundle,
    GeneratedArtifact,
)


def _artifact(
    name: str,
    content: bytes,
) -> GeneratedArtifact:
    return GeneratedArtifact(
        relative_path=PurePosixPath(
            "artifacts/generated"
        )
        / name,
        media_type="application/octet-stream",
        content=content,
    )


def _fake_bundle(
    *,
    changed_content: bool = False,
) -> SimpleNamespace:
    payload = tuple(
        _artifact(
            f"payload-{index}.bin",
            (
                f"payload-{index}-changed\n"
                if changed_content and index == 0
                else f"payload-{index}\n"
            ).encode("utf-8"),
        )
        for index in range(8)
    )
    manifest = _artifact(
        "artifact_manifest.json",
        b'{"manifest":true}\n',
    )
    manifest_checksum = _artifact(
        "artifact_manifest.sha256",
        b"0" * 64 + b"  artifact_manifest.json\n",
    )

    return SimpleNamespace(
        bundle_id=ANALYSIS_ARTIFACT_BUNDLE_ID,
        analysis_id=ANALYSIS_SNAPSHOT_ID,
        payload_artifacts=payload,
        manifest=manifest,
        artifacts=(
            *payload,
            manifest,
            manifest_checksum,
        ),
        total_byte_count=sum(
            artifact.byte_count
            for artifact in (
                *payload,
                manifest,
                manifest_checksum,
            )
        ),
    )


def _fake_audit(
    bundle: SimpleNamespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        audit_id=ARTIFACT_AUDIT_ID,
        bundle_id=bundle.bundle_id,
        analysis_id=bundle.analysis_id,
        artifact_count=len(bundle.payload_artifacts),
        total_byte_count=sum(
            artifact.byte_count
            for artifact in bundle.payload_artifacts
        ),
        manifest_sha256_hex=bundle.manifest.sha256_hex,
    )


def _install_successful_workflow(
    monkeypatch: pytest.MonkeyPatch,
    bundle: SimpleNamespace,
) -> None:
    snapshots = iter((object(), object()))
    bundles = iter((bundle, bundle))
    results = tuple(
        SimpleNamespace(
            relative_path=artifact.relative_path,
            byte_count=artifact.byte_count,
            sha256_hex=artifact.sha256_hex,
        )
        for artifact in bundle.artifacts
    )

    monkeypatch.setattr(
        release,
        "verify_repository_contract",
        lambda root: SimpleNamespace(
            verified_component_count=5,
            total_record_count=77,
        ),
    )
    monkeypatch.setattr(
        release,
        "build_analysis_snapshot",
        lambda root, config: cast(
            AnalysisSnapshot,
            next(snapshots),
        ),
    )
    monkeypatch.setattr(
        release,
        "build_analysis_artifact_bundle",
        lambda snapshot: cast(
            AnalysisArtifactBundle,
            next(bundles),
        ),
    )
    monkeypatch.setattr(
        release,
        "write_analysis_artifact_bundle",
        lambda received, root, overwrite: results,
    )
    monkeypatch.setattr(
        release,
        "verify_written_analysis_artifacts",
        lambda received, root: results,
    )
    monkeypatch.setattr(
        release,
        "audit_generated_artifacts",
        lambda root, strict_directory: _fake_audit(bundle),
    )


def test_release_reproduction_verifies_complete_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    bundle = _fake_bundle()

    _install_successful_workflow(
        monkeypatch,
        bundle,
    )

    report = release.verify_release_reproduction(
        source_root,
        output_root,
    )

    assert report.verification_id == release.RELEASE_VERIFICATION_ID
    assert report.repository_root == source_root.resolve()
    assert report.bundle_id == ANALYSIS_ARTIFACT_BUNDLE_ID
    assert report.analysis_id == ANALYSIS_SNAPSHOT_ID
    assert report.audit_id == ARTIFACT_AUDIT_ID
    assert report.deterministic_rebuild_count == 2
    assert report.contract_component_count == 5
    assert report.contract_record_count == 77
    assert report.artifact_count == 10
    assert report.payload_artifact_count == 8
    assert report.manifest_sha256_hex == bundle.manifest.sha256_hex


def test_release_report_serialization_is_deterministic(
    tmp_path: Path,
) -> None:
    report = release.ReleaseVerificationReport(
        verification_id=release.RELEASE_VERIFICATION_ID,
        repository_root=tmp_path.resolve(),
        bundle_id=ANALYSIS_ARTIFACT_BUNDLE_ID,
        analysis_id=ANALYSIS_SNAPSHOT_ID,
        audit_id=ARTIFACT_AUDIT_ID,
        deterministic_rebuild_count=2,
        contract_component_count=5,
        contract_record_count=77,
        artifact_count=10,
        payload_artifact_count=8,
        total_bundle_byte_count=1_000,
        payload_byte_count=800,
        manifest_sha256_hex="a" * 64,
    )

    payload = json.loads(report.to_json())

    assert payload == report.to_mapping()
    assert payload["status"] == "ok"
    assert payload["repository_contract"]["record_count"] == 77
    assert payload["reproduction"]["artifact_count"] == 10
    assert payload["independent_audit"]["status"] == "verified"
    assert report.to_json().endswith("\n")
    assert report.to_text().startswith(
        "Release reproduction: verified\n"
    )


def test_release_reproduction_requires_empty_output_directory(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    (output_root / "stale.txt").write_text(
        "stale\n",
        encoding="utf-8",
    )

    with pytest.raises(
        release.ReleaseVerificationError,
        match="must be empty",
    ):
        release.verify_release_reproduction(
            source_root,
            output_root,
        )


def test_release_reproduction_detects_nondeterministic_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()

    snapshots = iter((object(), object()))
    bundles = iter(
        (
            _fake_bundle(),
            _fake_bundle(changed_content=True),
        )
    )

    monkeypatch.setattr(
        release,
        "verify_repository_contract",
        lambda root: SimpleNamespace(
            verified_component_count=5,
            total_record_count=77,
        ),
    )
    monkeypatch.setattr(
        release,
        "build_analysis_snapshot",
        lambda root, config: cast(
            AnalysisSnapshot,
            next(snapshots),
        ),
    )
    monkeypatch.setattr(
        release,
        "build_analysis_artifact_bundle",
        lambda snapshot: cast(
            AnalysisArtifactBundle,
            next(bundles),
        ),
    )

    with pytest.raises(
        release.ReleaseVerificationError,
        match="different artifact bytes",
    ):
        release.verify_release_reproduction(
            source_root,
            output_root,
        )


def test_release_reproduction_detects_write_verification_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    bundle = _fake_bundle()

    _install_successful_workflow(
        monkeypatch,
        bundle,
    )
    monkeypatch.setattr(
        release,
        "verify_written_analysis_artifacts",
        lambda received, root: (),
    )

    with pytest.raises(
        release.ReleaseVerificationError,
        match="written and independently re-read",
    ):
        release.verify_release_reproduction(
            source_root,
            output_root,
        )


def test_release_reproduction_detects_audit_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    output_root.mkdir()
    bundle = _fake_bundle()

    _install_successful_workflow(
        monkeypatch,
        bundle,
    )
    audit = _fake_audit(bundle)
    audit.manifest_sha256_hex = "f" * 64
    monkeypatch.setattr(
        release,
        "audit_generated_artifacts",
        lambda root, strict_directory: audit,
    )

    with pytest.raises(
        release.ReleaseVerificationError,
        match="manifest digest",
    ):
        release.verify_release_reproduction(
            source_root,
            output_root,
        )


def test_main_emits_json_and_success_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = release.ReleaseVerificationReport(
        verification_id=release.RELEASE_VERIFICATION_ID,
        repository_root=tmp_path.resolve(),
        bundle_id=ANALYSIS_ARTIFACT_BUNDLE_ID,
        analysis_id=ANALYSIS_SNAPSHOT_ID,
        audit_id=ARTIFACT_AUDIT_ID,
        deterministic_rebuild_count=2,
        contract_component_count=5,
        contract_record_count=77,
        artifact_count=10,
        payload_artifact_count=8,
        total_bundle_byte_count=1_000,
        payload_byte_count=800,
        manifest_sha256_hex="b" * 64,
    )
    monkeypatch.setattr(
        release,
        "verify_release_reproduction",
        lambda repository_root, output_root: report,
    )
    stdout = StringIO()
    stderr = StringIO()

    status = release.main(
        (
            "--root",
            str(tmp_path),
            "--json",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue())["status"] == "ok"


def test_main_reports_controlled_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        repository_root: Path,
        output_root: Path,
    ) -> release.ReleaseVerificationReport:
        raise release.ReleaseVerificationError(
            "reproduction failed"
        )

    monkeypatch.setattr(
        release,
        "verify_release_reproduction",
        fail,
    )
    stdout = StringIO()
    stderr = StringIO()

    status = release.main(
        ("--root", str(tmp_path)),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "release verification error: reproduction failed\n"
    )
