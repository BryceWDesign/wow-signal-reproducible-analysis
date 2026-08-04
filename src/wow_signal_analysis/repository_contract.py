"""Cumulative verification contract for repository-controlled evidence artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from wow_signal_analysis.claim_ledger import (
    CLAIM_LEDGER_REFERENCE_PATH,
    load_verified_claim_ledger,
)
from wow_signal_analysis.dataset import (
    CANONICAL_DATASET_PATH,
    load_verified_wow_dataset,
)
from wow_signal_analysis.frequency_context import (
    FREQUENCY_REFERENCE_PATH,
    load_verified_frequency_context,
)
from wow_signal_analysis.hypothesis_matrix import (
    HYPOTHESIS_MATRIX_REFERENCE_PATH,
    load_verified_hypothesis_matrix,
)
from wow_signal_analysis.measurements import WOW_PRINTER_SEQUENCE
from wow_signal_analysis.morse import (
    MORSE_REFERENCE_PATH,
    MORSE_STANDARD_ID,
    load_verified_morse_registry,
)

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_EXPECTED_COMPONENT_IDS: Final = (
    "canonical-observation-dataset",
    "international-morse-registry",
    "frequency-context",
    "claim-ledger",
    "hypothesis-matrix",
)


class RepositoryContractError(ValueError):
    """Raised when cumulative repository evidence fails its declared contract."""


@dataclass(frozen=True, slots=True)
class VerifiedComponent:
    """One provenance-verified repository component and its record count."""

    component_id: str
    artifact_path: PurePosixPath
    record_count: int

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.component_id):
            raise RepositoryContractError(
                "component_id must be a lowercase hyphen-delimited identifier"
            )

        if (
            self.artifact_path.is_absolute()
            or not self.artifact_path.parts
            or ".." in self.artifact_path.parts
        ):
            raise RepositoryContractError(
                "artifact_path must be a safe repository-relative POSIX path"
            )

        if self.record_count <= 0:
            raise RepositoryContractError("record_count must be positive")


@dataclass(frozen=True, slots=True)
class RepositoryContractReport:
    """Validated cumulative identity and record counts for canonical evidence."""

    repository_root: Path
    printer_sequence: str
    morse_standard_id: str
    claim_ledger_id: str
    hypothesis_matrix_id: str
    components: tuple[VerifiedComponent, ...]

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute():
            raise RepositoryContractError("repository_root must be an absolute path")

        if self.printer_sequence != WOW_PRINTER_SEQUENCE:
            raise RepositoryContractError(f"printer_sequence must be {WOW_PRINTER_SEQUENCE!r}")

        if self.morse_standard_id != MORSE_STANDARD_ID:
            raise RepositoryContractError(f"morse_standard_id must be {MORSE_STANDARD_ID!r}")

        for field_name in ("claim_ledger_id", "hypothesis_matrix_id"):
            value = getattr(self, field_name)
            if not _IDENTIFIER_PATTERN.fullmatch(value):
                raise RepositoryContractError(
                    f"{field_name} must be a lowercase hyphen-delimited identifier"
                )

        component_ids = tuple(component.component_id for component in self.components)
        if component_ids != _EXPECTED_COMPONENT_IDS:
            raise RepositoryContractError(
                "components must contain the canonical repository contract in its declared order"
            )

        artifact_paths = tuple(component.artifact_path for component in self.components)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise RepositoryContractError("component artifact paths must be unique")

    @property
    def verified_component_count(self) -> int:
        """Return the number of provenance-verified canonical components."""

        return len(self.components)

    @property
    def total_record_count(self) -> int:
        """Return the combined records declared by all canonical components."""

        return sum(component.record_count for component in self.components)

    def component_by_id(self, component_id: str) -> VerifiedComponent:
        """Return one unique verified component."""

        matches = tuple(
            component for component in self.components if component.component_id == component_id
        )
        if len(matches) != 1:
            raise RepositoryContractError(
                f"expected one component for {component_id!r}, found {len(matches)}"
            )
        return matches[0]


def verify_repository_contract(
    repository_root: Path,
) -> RepositoryContractReport:
    """Verify all canonical evidence and cross-component bindings.

    This function verifies hashes and schema contracts through the individual
    verified loaders. Successful return means the committed canonical artifacts
    agree with their manifests and with one another. It does not validate every
    generated report or execute the repository test suite.
    """

    root = repository_root.resolve()
    if not root.is_dir():
        raise RepositoryContractError(f"repository_root must be an existing directory: {root}")

    dataset = load_verified_wow_dataset(root)
    morse_registry = load_verified_morse_registry(root)
    frequency_context = load_verified_frequency_context(root)
    claim_ledger = load_verified_claim_ledger(root)
    hypothesis_matrix = load_verified_hypothesis_matrix(
        root,
        claim_ledger=claim_ledger,
    )

    if hypothesis_matrix.ledger_id != claim_ledger.ledger_id:
        raise RepositoryContractError("hypothesis matrix is not bound to the verified claim ledger")

    return RepositoryContractReport(
        repository_root=root,
        printer_sequence=dataset.printer_sequence,
        morse_standard_id=morse_registry.standard_id,
        claim_ledger_id=claim_ledger.ledger_id,
        hypothesis_matrix_id=hypothesis_matrix.matrix_id,
        components=(
            VerifiedComponent(
                component_id="canonical-observation-dataset",
                artifact_path=CANONICAL_DATASET_PATH,
                record_count=len(dataset.samples),
            ),
            VerifiedComponent(
                component_id="international-morse-registry",
                artifact_path=MORSE_REFERENCE_PATH,
                record_count=len(morse_registry.symbols),
            ),
            VerifiedComponent(
                component_id="frequency-context",
                artifact_path=FREQUENCY_REFERENCE_PATH,
                record_count=frequency_context.record_count,
            ),
            VerifiedComponent(
                component_id="claim-ledger",
                artifact_path=CLAIM_LEDGER_REFERENCE_PATH,
                record_count=len(claim_ledger.claims),
            ),
            VerifiedComponent(
                component_id="hypothesis-matrix",
                artifact_path=HYPOTHESIS_MATRIX_REFERENCE_PATH,
                record_count=len(hypothesis_matrix.hypotheses),
            ),
        ),
    )
