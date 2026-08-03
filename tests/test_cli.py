from __future__ import annotations

import json
from hashlib import sha256
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

import wow_signal_analysis.cli as cli
from wow_signal_analysis.analysis_snapshot import AnalysisSnapshot
from wow_signal_analysis.artifact_audit import (
    ARTIFACT_AUDIT_ID,
    ArtifactAuditReport,
    AuditedArtifact,
)
from wow_signal_analysis.artifacts import (
    ANALYSIS_ARTIFACT_BUNDLE_ID,
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
    AnalysisArtifactBundle,
    AnalysisBundleManifest,
    ArtifactWriteResult,
    GeneratedArtifact,
)
from wow_signal_analysis.repository_contract import (
    RepositoryContractError,
    RepositoryContractReport,
    VerifiedComponent,
)


def _repository_report(root: Path) -> RepositoryContractReport:
    return RepositoryContractReport(
        repository_root=root.resolve(),
        printer_sequence="6EQUJ5",
        morse_standard_id="ITU-R M.1677-1",
        claim_ledger_id="wow-signal-evidence-claims-v1",
        hypothesis_matrix_id="wow-signal-hypothesis-matrix-v1",
        components=(
            VerifiedComponent(
                component_id="canonical-observation-dataset",
                artifact_path=PurePosixPath(
                    "data/raw/wow_6equj5.csv"
                ),
                record_count=6,
            ),
            VerifiedComponent(
                component_id="international-morse-registry",
                artifact_path=PurePosixPath(
                    "data/reference/itu_m1677_1_morse_symbols.json"
                ),
                record_count=51,
            ),
            VerifiedComponent(
                component_id="frequency-context",
                artifact_path=PurePosixPath(
                    "data/reference/frequency_context.json"
                ),
                record_count=3,
            ),
            VerifiedComponent(
                component_id="claim-ledger",
                artifact_path=PurePosixPath(
                    "data/reference/claim_ledger.json"
                ),
                record_count=12,
            ),
            VerifiedComponent(
                component_id="hypothesis-matrix",
                artifact_path=PurePosixPath(
                    "data/reference/hypothesis_matrix.json"
                ),
                record_count=5,
            ),
        ),
    )


def _checksum_artifact(
    source: GeneratedArtifact,
    checksum_path: PurePosixPath,
) -> GeneratedArtifact:
    content = (
        f"{source.sha256_hex}  {source.relative_path.name}\n"
    ).encode("ascii")

    return GeneratedArtifact(
        relative_path=checksum_path,
        media_type="text/plain",
        content=content,
    )


def _svg_artifact(
    relative_path: PurePosixPath,
    figure_id: str,
    title: str,
) -> GeneratedArtifact:
    svg = "\n".join(
        (
            (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'role="img" '
                f'aria-labelledby="{figure_id}-title '
                f'{figure_id}-description">'
            ),
            f'<title id="{figure_id}-title">{title}</title>',
            (
                f'<desc id="{figure_id}-description">'
                f"Test figure.</desc>"
            ),
            "</svg>",
            "",
        )
    ).encode("utf-8")

    return GeneratedArtifact(
        relative_path=relative_path,
        media_type="image/svg+xml",
        content=svg,
    )


def _artifact_bundle() -> AnalysisArtifactBundle:
    snapshot = GeneratedArtifact(
        relative_path=ANALYSIS_SNAPSHOT_ARTIFACT_PATH,
        media_type="application/json",
        content=b"{}\n",
    )
    report = GeneratedArtifact(
        relative_path=ANALYSIS_REPORT_ARTIFACT_PATH,
        media_type="text/markdown",
        content=b"# Test analysis report\n",
    )
    beam_fit_figure = _svg_artifact(
        ANALYSIS_BEAM_FIT_FIGURE_PATH,
        "wow-signal-beam-fit-v1",
        "Test beam fit",
    )
    model_comparison_figure = _svg_artifact(
        ANALYSIS_MODEL_COMPARISON_FIGURE_PATH,
        "wow-signal-model-comparison-v1",
        "Test model comparison",
    )
    payload = (
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
        beam_fit_figure,
        _checksum_artifact(
            beam_fit_figure,
            ANALYSIS_BEAM_FIT_CHECKSUM_PATH,
        ),
        model_comparison_figure,
        _checksum_artifact(
            model_comparison_figure,
            ANALYSIS_MODEL_COMPARISON_CHECKSUM_PATH,
        ),
    )
    manifest_model = AnalysisBundleManifest.from_artifacts(payload)
    manifest = GeneratedArtifact(
        relative_path=ANALYSIS_BUNDLE_MANIFEST_PATH,
        media_type="application/json",
        content=manifest_model.to_json().encode("utf-8"),
    )

    return AnalysisArtifactBundle(
        bundle_id=ANALYSIS_ARTIFACT_BUNDLE_ID,
        analysis_id="wow-signal-canonical-analysis-v1",
        snapshot=payload[0],
        checksum=payload[1],
        report=payload[2],
        report_checksum=payload[3],
        beam_fit_figure=payload[4],
        beam_fit_checksum=payload[5],
        model_comparison_figure=payload[6],
        model_comparison_checksum=payload[7],
        manifest=manifest,
        manifest_checksum=_checksum_artifact(
            manifest,
            ANALYSIS_BUNDLE_MANIFEST_CHECKSUM_PATH,
        ),
    )


def _write_results(
    root: Path,
    bundle: AnalysisArtifactBundle,
) -> tuple[ArtifactWriteResult, ...]:
    return tuple(
        ArtifactWriteResult(
            relative_path=artifact.relative_path,
            absolute_path=(
                root.joinpath(*artifact.relative_path.parts).resolve()
            ),
            media_type=artifact.media_type,
            byte_count=artifact.byte_count,
            sha256_hex=sha256(artifact.content).hexdigest(),
        )
        for artifact in bundle.artifacts
    )


def _artifact_audit_report(
    root: Path,
    *,
    strict_directory: bool,
) -> ArtifactAuditReport:
    bundle = _artifact_bundle()
    audited = tuple(
        AuditedArtifact(
            relative_path=artifact.relative_path,
            media_type=artifact.media_type,
            byte_count=artifact.byte_count,
            sha256_hex=artifact.sha256_hex,
        )
        for artifact in bundle.payload_artifacts
    )

    return ArtifactAuditReport(
        audit_id=ARTIFACT_AUDIT_ID,
        repository_root=root.resolve(),
        bundle_id=bundle.bundle_id,
        analysis_id=bundle.analysis_id,
        manifest_byte_count=bundle.manifest.byte_count,
        manifest_sha256_hex=bundle.manifest.sha256_hex,
        artifacts=audited,
        strict_directory=strict_directory,
    )


def test_verify_command_emits_portable_deterministic_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _repository_report(tmp_path)
    monkeypatch.setattr(
        cli,
        "verify_repository_contract",
        lambda repository_root: report,
    )
    stdout = StringIO()
    stderr = StringIO()

    status = cli.main(
        (
            "verify",
            "--root",
            str(tmp_path),
            "--json",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert stderr.getvalue() == ""
    assert payload["command"] == "verify"
    assert payload["status"] == "ok"
    assert payload["printer_sequence"] == "6EQUJ5"
    assert payload["verified_component_count"] == 5
    assert payload["total_record_count"] == 77
    assert len(payload["components"]) == 5


def test_verify_command_emits_human_readable_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _repository_report(tmp_path)
    monkeypatch.setattr(
        cli,
        "verify_repository_contract",
        lambda repository_root: report,
    )
    stdout = StringIO()

    status = cli.main(
        ("verify", "--root", str(tmp_path)),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert status == 0
    assert stdout.getvalue().startswith(
        "Repository contract: verified\n"
    )
    assert "Printer sequence: 6EQUJ5\n" in stdout.getvalue()
    assert "Canonical records: 77\n" in stdout.getvalue()


def test_audit_command_emits_portable_deterministic_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _artifact_audit_report(
        tmp_path,
        strict_directory=True,
    )
    captured: dict[str, object] = {}

    def fake_audit(
        repository_root: Path,
        *,
        strict_directory: bool,
    ) -> ArtifactAuditReport:
        captured["repository_root"] = repository_root
        captured["strict_directory"] = strict_directory
        return report

    monkeypatch.setattr(
        cli,
        "audit_generated_artifacts",
        fake_audit,
    )
    stdout = StringIO()
    stderr = StringIO()

    status = cli.main(
        (
            "audit",
            "--root",
            str(tmp_path),
            "--json",
        ),
        stdout=stdout,
        stderr=stderr,
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert stderr.getvalue() == ""
    assert captured["repository_root"] == tmp_path
    assert captured["strict_directory"] is True
    assert payload["command"] == "audit"
    assert payload["status"] == "ok"
    assert payload["audit_id"] == ARTIFACT_AUDIT_ID
    assert payload["strict_directory"] is True
    assert payload["artifact_count"] == 8
    assert len(payload["artifacts"]) == 8
    assert payload["manifest"]["sha256"] == report.manifest_sha256_hex


def test_audit_command_can_allow_extra_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _artifact_audit_report(
        tmp_path,
        strict_directory=False,
    )
    captured: dict[str, object] = {}

    def fake_audit(
        repository_root: Path,
        *,
        strict_directory: bool,
    ) -> ArtifactAuditReport:
        captured["strict_directory"] = strict_directory
        return report

    monkeypatch.setattr(
        cli,
        "audit_generated_artifacts",
        fake_audit,
    )
    stdout = StringIO()

    status = cli.main(
        (
            "audit",
            "--root",
            str(tmp_path),
            "--allow-extra-files",
        ),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert status == 0
    assert captured["strict_directory"] is False
    assert stdout.getvalue().startswith(
        "Generated artifact audit: verified\n"
    )
    assert "Directory inventory: allow-extra-files\n" in stdout.getvalue()
    assert "Payload artifacts: 8\n" in stdout.getvalue()


def test_generate_command_passes_explicit_reproducibility_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _artifact_bundle()
    results = _write_results(tmp_path, bundle)
    captured: dict[str, object] = {}

    def fake_build_snapshot(
        repository_root: Path,
        *,
        config: object,
    ) -> AnalysisSnapshot:
        captured["repository_root"] = repository_root
        captured["config"] = config
        return cast(AnalysisSnapshot, object())

    monkeypatch.setattr(
        cli,
        "build_analysis_snapshot",
        fake_build_snapshot,
    )
    monkeypatch.setattr(
        cli,
        "build_analysis_artifact_bundle",
        lambda snapshot: bundle,
    )

    def fake_write(
        received_bundle: AnalysisArtifactBundle,
        repository_root: Path,
        *,
        overwrite: bool,
    ) -> tuple[ArtifactWriteResult, ...]:
        captured["bundle"] = received_bundle
        captured["write_root"] = repository_root
        captured["overwrite"] = overwrite
        return results

    monkeypatch.setattr(
        cli,
        "write_analysis_artifact_bundle",
        fake_write,
    )
    stdout = StringIO()

    status = cli.main(
        (
            "generate",
            "--root",
            str(tmp_path),
            "--grid-points",
            "21",
            "--refinement-rounds",
            "3",
            "--glyph",
            "?",
            "--glyph",
            ",",
            "--max-unique-sequences",
            "720",
            "--no-overwrite",
            "--json",
        ),
        stdout=stdout,
        stderr=StringIO(),
    )
    payload = json.loads(stdout.getvalue())
    config = captured["config"]

    assert status == 0
    assert payload["command"] == "generate"
    assert payload["action"] == "write"
    assert payload["result"] == "written"
    assert payload["artifact_count"] == 10
    assert captured["overwrite"] is False
    assert config.gaussian_search.grid_points == 21
    assert config.gaussian_search.refinement_rounds == 3
    assert config.selected_morse_glyphs == ("?", ",")
    assert config.max_unique_sequences == 720


def test_generate_check_verifies_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _artifact_bundle()
    results = _write_results(tmp_path, bundle)

    monkeypatch.setattr(
        cli,
        "build_analysis_snapshot",
        lambda repository_root, config: cast(
            AnalysisSnapshot,
            object(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_analysis_artifact_bundle",
        lambda snapshot: bundle,
    )

    def fail_write(
        received_bundle: AnalysisArtifactBundle,
        repository_root: Path,
        *,
        overwrite: bool,
    ) -> tuple[ArtifactWriteResult, ...]:
        raise AssertionError("write path must not run during --check")

    monkeypatch.setattr(
        cli,
        "write_analysis_artifact_bundle",
        fail_write,
    )
    monkeypatch.setattr(
        cli,
        "verify_written_analysis_artifacts",
        lambda received_bundle, repository_root: results,
    )
    stdout = StringIO()

    status = cli.main(
        (
            "generate",
            "--root",
            str(tmp_path),
            "--check",
            "--json",
        ),
        stdout=stdout,
        stderr=StringIO(),
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert payload["action"] == "check"
    assert payload["result"] == "verified"
    assert payload["artifact_count"] == 10
    assert tuple(
        artifact["relative_path"] for artifact in payload["artifacts"]
    ) == (
        "artifacts/generated/analysis_snapshot.json",
        "artifacts/generated/analysis_snapshot.sha256",
        "artifacts/generated/analysis_report.md",
        "artifacts/generated/analysis_report.sha256",
        "artifacts/generated/beam_fit.svg",
        "artifacts/generated/beam_fit.sha256",
        "artifacts/generated/model_comparison.svg",
        "artifacts/generated/model_comparison.sha256",
        "artifacts/generated/artifact_manifest.json",
        "artifacts/generated/artifact_manifest.sha256",
    )


def test_execution_error_returns_nonzero_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_verification(
        repository_root: Path,
    ) -> RepositoryContractReport:
        raise RepositoryContractError("canonical contract failed")

    monkeypatch.setattr(
        cli,
        "verify_repository_contract",
        fail_verification,
    )
    stdout = StringIO()
    stderr = StringIO()

    status = cli.main(
        ("verify", "--root", str(tmp_path)),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "error: canonical contract failed\n"
    )


def test_invalid_gaussian_search_configuration_fails_before_execution(
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    stderr = StringIO()

    status = cli.main(
        (
            "generate",
            "--root",
            str(tmp_path),
            "--grid-points",
            "4",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 1
    assert stdout.getvalue() == ""
    assert "grid_points must be an odd integer" in stderr.getvalue()
