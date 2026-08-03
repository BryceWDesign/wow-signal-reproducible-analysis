"""Command-line verification, generation, and artifact-audit workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO, cast

from wow_signal_analysis.analysis_snapshot import (
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.artifact_audit import (
    ArtifactAuditReport,
    audit_generated_artifacts,
)
from wow_signal_analysis.artifacts import (
    ArtifactWriteResult,
    build_analysis_artifact_bundle,
    verify_written_analysis_artifacts,
    write_analysis_artifact_bundle,
)
from wow_signal_analysis.beam_model import GaussianSearchConfig
from wow_signal_analysis.repository_contract import (
    RepositoryContractReport,
    verify_repository_contract,
)


class CommandOutput(StrEnum):
    """Supported command-line rendering formats."""

    TEXT = "text"
    JSON = "json"


class ArtifactAction(StrEnum):
    """Requested action for deterministic generated artifacts."""

    WRITE = "write"
    CHECK = "check"


@dataclass(frozen=True, slots=True)
class VerifyOptions:
    """Parsed options for the repository verification command."""

    repository_root: Path
    output: CommandOutput


@dataclass(frozen=True, slots=True)
class AuditOptions:
    """Parsed options for independent generated-artifact auditing."""

    repository_root: Path
    output: CommandOutput
    strict_directory: bool


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    """Parsed options for deterministic snapshot artifact handling."""

    repository_root: Path
    output: CommandOutput
    action: ArtifactAction
    overwrite: bool
    gaussian_search: GaussianSearchConfig
    selected_morse_glyphs: tuple[str, ...]
    max_unique_sequences: int


CommandOptions = VerifyOptions | AuditOptions | GenerateOptions


@dataclass(frozen=True, slots=True)
class VerificationCommandResult:
    """Portable rendering of a successful repository-contract verification."""

    report: RepositoryContractReport

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible command result."""

        return {
            "command": "verify",
            "status": "ok",
            "printer_sequence": self.report.printer_sequence,
            "morse_standard_id": self.report.morse_standard_id,
            "claim_ledger_id": self.report.claim_ledger_id,
            "hypothesis_matrix_id": self.report.hypothesis_matrix_id,
            "verified_component_count": self.report.verified_component_count,
            "total_record_count": self.report.total_record_count,
            "components": [
                {
                    "component_id": component.component_id,
                    "artifact_path": str(component.artifact_path),
                    "record_count": component.record_count,
                }
                for component in self.report.components
            ],
        }

    def to_text(self) -> str:
        """Return a stable human-readable verification summary."""

        lines = [
            "Repository contract: verified",
            f"Printer sequence: {self.report.printer_sequence}",
            f"Morse standard: {self.report.morse_standard_id}",
            f"Claim ledger: {self.report.claim_ledger_id}",
            f"Hypothesis matrix: {self.report.hypothesis_matrix_id}",
            f"Verified components: {self.report.verified_component_count}",
            f"Canonical records: {self.report.total_record_count}",
            "Components:",
        ]

        lines.extend(
            (
                f"  - {component.component_id}: "
                f"{component.record_count} records "
                f"[{component.artifact_path}]"
            )
            for component in self.report.components
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ArtifactAuditCommandResult:
    """Portable rendering of an independent generated-artifact audit."""

    report: ArtifactAuditReport

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible audit result."""

        return {
            "command": "audit",
            "status": "ok",
            "audit_id": self.report.audit_id,
            "bundle_id": self.report.bundle_id,
            "analysis_id": self.report.analysis_id,
            "strict_directory": self.report.strict_directory,
            "artifact_count": self.report.artifact_count,
            "total_byte_count": self.report.total_byte_count,
            "manifest": {
                "byte_count": self.report.manifest_byte_count,
                "sha256": self.report.manifest_sha256_hex,
            },
            "artifacts": [
                {
                    "relative_path": str(artifact.relative_path),
                    "media_type": artifact.media_type,
                    "byte_count": artifact.byte_count,
                    "sha256": artifact.sha256_hex,
                }
                for artifact in self.report.artifacts
            ],
        }

    def to_text(self) -> str:
        """Return a stable human-readable audit summary."""

        inventory_mode = (
            "strict"
            if self.report.strict_directory
            else "allow-extra-files"
        )
        lines = [
            "Generated artifact audit: verified",
            f"Audit: {self.report.audit_id}",
            f"Bundle: {self.report.bundle_id}",
            f"Analysis: {self.report.analysis_id}",
            f"Directory inventory: {inventory_mode}",
            f"Payload artifacts: {self.report.artifact_count}",
            f"Payload bytes: {self.report.total_byte_count}",
            (
                "Manifest: "
                f"{self.report.manifest_byte_count} bytes "
                f"[sha256:{self.report.manifest_sha256_hex}]"
            ),
            "Artifacts:",
        ]
        lines.extend(
            (
                f"  - {artifact.relative_path}: "
                f"{artifact.byte_count} bytes "
                f"[sha256:{artifact.sha256_hex}]"
            )
            for artifact in self.report.artifacts
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ArtifactCommandResult:
    """Portable rendering of written or verified generated artifacts."""

    action: ArtifactAction
    bundle_id: str
    analysis_id: str
    artifacts: tuple[ArtifactWriteResult, ...]

    def __post_init__(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("bundle_id must be non-empty")
        if not self.analysis_id.strip():
            raise ValueError("analysis_id must be non-empty")
        if not self.artifacts:
            raise ValueError("artifact command result must contain artifacts")

    @property
    def action_status(self) -> str:
        """Return the completed action in past-tense reporting form."""

        if self.action is ArtifactAction.CHECK:
            return "verified"
        return "written"

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible command result."""

        return {
            "command": "generate",
            "action": self.action.value,
            "status": "ok",
            "result": self.action_status,
            "bundle_id": self.bundle_id,
            "analysis_id": self.analysis_id,
            "artifact_count": len(self.artifacts),
            "artifacts": [
                {
                    "relative_path": str(artifact.relative_path),
                    "media_type": artifact.media_type,
                    "byte_count": artifact.byte_count,
                    "sha256": artifact.sha256_hex,
                }
                for artifact in self.artifacts
            ],
        }

    def to_text(self) -> str:
        """Return a stable human-readable artifact summary."""

        lines = [
            f"Analysis artifacts: {self.action_status}",
            f"Bundle: {self.bundle_id}",
            f"Analysis: {self.analysis_id}",
            f"Artifact count: {len(self.artifacts)}",
            "Artifacts:",
        ]

        lines.extend(
            (
                f"  - {artifact.relative_path}: "
                f"{artifact.byte_count} bytes "
                f"[sha256:{artifact.sha256_hex}]"
            )
            for artifact in self.artifacts
        )
        return "\n".join(lines) + "\n"


CommandResult = (
    VerificationCommandResult
    | ArtifactAuditCommandResult
    | ArtifactCommandResult
)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the command-line interface and return a process-compatible status."""

    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr

    try:
        options = _parse_arguments(argv)

        if isinstance(options, VerifyOptions):
            result: CommandResult = _run_verify(options)
        elif isinstance(options, AuditOptions):
            result = _run_audit(options)
        else:
            result = _run_generate(options)

        _write_result(result, options.output, output_stream)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=error_stream)
        return 1

    return 0


def _run_verify(options: VerifyOptions) -> VerificationCommandResult:
    report = verify_repository_contract(options.repository_root)
    return VerificationCommandResult(report=report)


def _run_audit(options: AuditOptions) -> ArtifactAuditCommandResult:
    report = audit_generated_artifacts(
        options.repository_root,
        strict_directory=options.strict_directory,
    )
    return ArtifactAuditCommandResult(report=report)


def _run_generate(options: GenerateOptions) -> ArtifactCommandResult:
    snapshot = build_analysis_snapshot(
        options.repository_root,
        config=SnapshotConfig(
            gaussian_search=options.gaussian_search,
            selected_morse_glyphs=options.selected_morse_glyphs,
            max_unique_sequences=options.max_unique_sequences,
        ),
    )
    bundle = build_analysis_artifact_bundle(snapshot)

    if options.action is ArtifactAction.CHECK:
        artifacts = verify_written_analysis_artifacts(
            bundle,
            options.repository_root,
        )
    else:
        artifacts = write_analysis_artifact_bundle(
            bundle,
            options.repository_root,
            overwrite=options.overwrite,
        )

    return ArtifactCommandResult(
        action=options.action,
        bundle_id=bundle.bundle_id,
        analysis_id=bundle.analysis_id,
        artifacts=artifacts,
    )


def _write_result(
    result: CommandResult,
    output: CommandOutput,
    stream: TextIO,
) -> None:
    if output is CommandOutput.JSON:
        stream.write(
            json.dumps(
                result.to_mapping(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        stream.write("\n")
        return

    stream.write(result.to_text())


def _parse_arguments(
    argv: Sequence[str] | None,
) -> CommandOptions:
    parser = _build_parser()
    namespace = parser.parse_args(
        None if argv is None else list(argv)
    )

    command = cast(str, namespace.command)
    repository_root = cast(Path, namespace.repository_root)
    json_output = cast(bool, namespace.json_output)
    output = CommandOutput.JSON if json_output else CommandOutput.TEXT

    if command == "verify":
        return VerifyOptions(
            repository_root=repository_root,
            output=output,
        )

    if command == "audit":
        allow_extra_files = cast(bool, namespace.allow_extra_files)
        return AuditOptions(
            repository_root=repository_root,
            output=output,
            strict_directory=not allow_extra_files,
        )

    glyph_values = cast(list[str] | None, namespace.glyphs)
    selected_glyphs = (
        tuple(glyph_values)
        if glyph_values is not None
        else ("?", ",")
    )

    check = cast(bool, namespace.check)
    no_overwrite = cast(bool, namespace.no_overwrite)

    return GenerateOptions(
        repository_root=repository_root,
        output=output,
        action=ArtifactAction.CHECK if check else ArtifactAction.WRITE,
        overwrite=not no_overwrite,
        gaussian_search=GaussianSearchConfig(
            grid_points=cast(int, namespace.grid_points),
            refinement_rounds=cast(
                int,
                namespace.refinement_rounds,
            ),
            center_padding_cadences=cast(
                float,
                namespace.center_padding_cadences,
            ),
            minimum_sigma_cadences=cast(
                float,
                namespace.minimum_sigma_cadences,
            ),
            maximum_sigma_spans=cast(
                float,
                namespace.maximum_sigma_spans,
            ),
        ),
        selected_morse_glyphs=selected_glyphs,
        max_unique_sequences=cast(
            int,
            namespace.max_unique_sequences,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    defaults = GaussianSearchConfig()
    parser = argparse.ArgumentParser(
        prog="wow-signal-analysis",
        description=(
            "Verify canonical evidence, reproduce deterministic artifacts, "
            "or audit generated files independently."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify canonical artifacts, manifests, and cross-bindings.",
    )
    _add_common_options(verify_parser)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit generated artifacts from their content manifest.",
    )
    _add_common_options(audit_parser)
    audit_parser.add_argument(
        "--allow-extra-files",
        action="store_true",
        help=(
            "Permit unmanifested regular files while still verifying every "
            "canonical artifact, digest, byte count, and checksum."
        ),
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Build and write or verify deterministic analysis artifacts.",
    )
    _add_common_options(generate_parser)

    action_group = generate_parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify existing generated artifacts instead of writing them."
        ),
    )
    action_group.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail when a generated artifact already exists.",
    )

    generate_parser.add_argument(
        "--grid-points",
        type=int,
        default=defaults.grid_points,
        help="Odd number of points per Gaussian search axis.",
    )
    generate_parser.add_argument(
        "--refinement-rounds",
        type=int,
        default=defaults.refinement_rounds,
        help="Number of deterministic Gaussian grid refinements.",
    )
    generate_parser.add_argument(
        "--center-padding-cadences",
        type=float,
        default=defaults.center_padding_cadences,
        help="Gaussian center-search padding measured in sample cadences.",
    )
    generate_parser.add_argument(
        "--minimum-sigma-cadences",
        type=float,
        default=defaults.minimum_sigma_cadences,
        help="Minimum Gaussian sigma measured in sample cadences.",
    )
    generate_parser.add_argument(
        "--maximum-sigma-spans",
        type=float,
        default=defaults.maximum_sigma_spans,
        help="Maximum Gaussian sigma measured in observation spans.",
    )
    generate_parser.add_argument(
        "--glyph",
        action="append",
        dest="glyphs",
        metavar="CHARACTER",
        help=(
            "Printable Morse glyph included in the exact permutation control. "
            "May be supplied more than once. Defaults to '?' and ','."
        ),
    )
    generate_parser.add_argument(
        "--max-unique-sequences",
        type=int,
        default=100_000,
        help="Maximum unique permutations allowed by the exact null model.",
    )

    return parser


def _add_common_options(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--root",
        dest="repository_root",
        type=Path,
        default=Path("."),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit deterministic JSON instead of human-readable text.",
    )
