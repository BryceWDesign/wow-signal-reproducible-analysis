"""Evidence-bound claim registry for reproducible Wow! signal reporting."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast
from urllib.parse import urlparse

from wow_signal_analysis.provenance import (
    ProvenanceError,
    load_source_manifest,
    require_verified_artifacts,
)

CLAIM_LEDGER_REFERENCE_PATH: Final = PurePosixPath("data/reference/claim_ledger.json")
CLAIM_LEDGER_MANIFEST_PATH: Final = PurePosixPath("data/provenance/claim_source_manifest.json")

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ClaimLedgerError(ValueError):
    """Raised when claim metadata are malformed or exceed their evidence class."""


class EvidenceKind(StrEnum):
    """Kind of evidence cited by a claim."""

    DATASET = "dataset"
    ANALYSIS = "analysis"
    STANDARD = "standard"
    SOURCE = "source"


class ClaimClassification(StrEnum):
    """Evidentiary class assigned before reports are generated."""

    OBSERVED = "observed"
    DERIVED = "derived"
    COMPATIBILITY = "compatibility"
    INTERPRETIVE = "interpretive"
    SPECULATIVE = "speculative"


class ClaimVerdict(StrEnum):
    """Permitted reporting verdict for a claim classification."""

    SUPPORTED = "supported"
    REPRODUCIBLE = "reproducible"
    COMPATIBLE_NOT_PROVEN = "compatible-not-proven"
    SUMMARY_ONLY = "summary-only"
    NOT_ESTABLISHED = "not-established"


_ALLOWED_VERDICTS: Final = {
    ClaimClassification.OBSERVED: ClaimVerdict.SUPPORTED,
    ClaimClassification.DERIVED: ClaimVerdict.REPRODUCIBLE,
    ClaimClassification.COMPATIBILITY: ClaimVerdict.COMPATIBLE_NOT_PROVEN,
    ClaimClassification.INTERPRETIVE: ClaimVerdict.SUMMARY_ONLY,
    ClaimClassification.SPECULATIVE: ClaimVerdict.NOT_ESTABLISHED,
}

_ALLOWED_DEPENDENCY_CLASSES: Final = {
    ClaimClassification.OBSERVED: frozenset({ClaimClassification.OBSERVED}),
    ClaimClassification.DERIVED: frozenset(
        {
            ClaimClassification.OBSERVED,
            ClaimClassification.DERIVED,
        }
    ),
    ClaimClassification.COMPATIBILITY: frozenset(
        {
            ClaimClassification.OBSERVED,
            ClaimClassification.DERIVED,
            ClaimClassification.COMPATIBILITY,
        }
    ),
    ClaimClassification.INTERPRETIVE: frozenset(
        {
            ClaimClassification.OBSERVED,
            ClaimClassification.DERIVED,
            ClaimClassification.COMPATIBILITY,
            ClaimClassification.INTERPRETIVE,
        }
    ),
    ClaimClassification.SPECULATIVE: frozenset(ClaimClassification),
}


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One immutable dataset, analysis, standard, or external source reference."""

    evidence_id: str
    kind: EvidenceKind
    locator: str
    description: str

    def __post_init__(self) -> None:
        _require_identifier(self.evidence_id, "evidence_id")
        if not self.locator.strip():
            raise ClaimLedgerError("evidence locator must be non-empty")
        if not self.description.strip():
            raise ClaimLedgerError("evidence description must be non-empty")

        if self.kind is EvidenceKind.SOURCE:
            parsed = urlparse(self.locator)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ClaimLedgerError("source evidence locator must be an absolute HTTPS URL")
        elif self.kind is EvidenceKind.ANALYSIS:
            if ":" not in self.locator:
                raise ClaimLedgerError(
                    "analysis evidence locator must identify module and callable"
                )
        else:
            path = PurePosixPath(self.locator)
            if path.is_absolute() or ".." in path.parts:
                raise ClaimLedgerError("dataset and standard locators must be safe relative paths")


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """One statement with an enforced evidence class and reporting verdict."""

    claim_id: str
    statement: str
    classification: ClaimClassification
    verdict: ClaimVerdict
    evidence_ids: tuple[str, ...]
    depends_on: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.claim_id, "claim_id")
        if not self.statement.strip():
            raise ClaimLedgerError("claim statement must be non-empty")

        expected_verdict = _ALLOWED_VERDICTS[self.classification]
        if self.verdict is not expected_verdict:
            raise ClaimLedgerError(
                f"classification {self.classification.value!r} requires "
                f"verdict {expected_verdict.value!r}"
            )

        _require_unique_identifiers(
            self.evidence_ids,
            "evidence_ids",
            allow_empty=False,
        )
        _require_unique_identifiers(
            self.depends_on,
            "depends_on",
            allow_empty=True,
        )
        if self.claim_id in self.depends_on:
            raise ClaimLedgerError("claim cannot depend on itself")
        if not self.limitations:
            raise ClaimLedgerError("claim limitations must not be empty")
        if any(not limitation.strip() for limitation in self.limitations):
            raise ClaimLedgerError("claim limitations must contain only non-empty strings")
        if len(set(self.limitations)) != len(self.limitations):
            raise ClaimLedgerError("claim limitations must be unique")


@dataclass(frozen=True, slots=True)
class ClaimLedger:
    """Validated claim graph separating findings from interpretations."""

    schema_version: int
    ledger_id: str
    title: str
    scope_note: str
    evidence: tuple[EvidenceRecord, ...]
    claims: tuple[ClaimRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ClaimLedgerError("unsupported claim ledger schema_version")
        _require_identifier(self.ledger_id, "ledger_id")
        if not self.title.strip() or not self.scope_note.strip():
            raise ClaimLedgerError("title and scope_note must be non-empty")
        if not self.evidence:
            raise ClaimLedgerError("claim ledger must contain evidence")
        if not self.claims:
            raise ClaimLedgerError("claim ledger must contain claims")

        evidence_ids = tuple(record.evidence_id for record in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ClaimLedgerError("evidence IDs must be unique")

        claim_ids = tuple(claim.claim_id for claim in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ClaimLedgerError("claim IDs must be unique")

        evidence_id_set = set(evidence_ids)
        claim_id_set = set(claim_ids)
        claims_by_id = {claim.claim_id: claim for claim in self.claims}

        for claim in self.claims:
            unknown_evidence = tuple(
                evidence_id
                for evidence_id in claim.evidence_ids
                if evidence_id not in evidence_id_set
            )
            if unknown_evidence:
                raise ClaimLedgerError(
                    f"claim {claim.claim_id!r} references unknown evidence: {unknown_evidence}"
                )

            unknown_dependencies = tuple(
                dependency for dependency in claim.depends_on if dependency not in claim_id_set
            )
            if unknown_dependencies:
                raise ClaimLedgerError(
                    f"claim {claim.claim_id!r} references unknown claims: {unknown_dependencies}"
                )

            allowed_classes = _ALLOWED_DEPENDENCY_CLASSES[claim.classification]
            disallowed = tuple(
                dependency
                for dependency in claim.depends_on
                if claims_by_id[dependency].classification not in allowed_classes
            )
            if disallowed:
                raise ClaimLedgerError(
                    f"claim {claim.claim_id!r} depends on claims outside its "
                    f"permitted evidence classes: {disallowed}"
                )

        _require_acyclic_claim_graph(self.claims)

    @property
    def topological_claims(self) -> tuple[ClaimRecord, ...]:
        """Return dependencies before dependents in deterministic ledger order."""

        by_id = {claim.claim_id: claim for claim in self.claims}
        ordered: list[ClaimRecord] = []
        visited: set[str] = set()

        def visit(claim: ClaimRecord) -> None:
            if claim.claim_id in visited:
                return
            for dependency_id in claim.depends_on:
                visit(by_id[dependency_id])
            visited.add(claim.claim_id)
            ordered.append(claim)

        for claim in self.claims:
            visit(claim)

        return tuple(ordered)

    def evidence_by_id(self, evidence_id: str) -> EvidenceRecord:
        """Return one unique evidence record."""

        matches = tuple(record for record in self.evidence if record.evidence_id == evidence_id)
        if len(matches) != 1:
            raise ClaimLedgerError(
                f"expected one evidence record for {evidence_id!r}, found {len(matches)}"
            )
        return matches[0]

    def claim_by_id(self, claim_id: str) -> ClaimRecord:
        """Return one unique claim record."""

        matches = tuple(claim for claim in self.claims if claim.claim_id == claim_id)
        if len(matches) != 1:
            raise ClaimLedgerError(
                f"expected one claim record for {claim_id!r}, found {len(matches)}"
            )
        return matches[0]

    def claims_by_classification(
        self,
        classification: ClaimClassification,
    ) -> tuple[ClaimRecord, ...]:
        """Return claims in ledger order for one evidence class."""

        return tuple(claim for claim in self.claims if claim.classification is classification)

    def claims_by_verdict(
        self,
        verdict: ClaimVerdict,
    ) -> tuple[ClaimRecord, ...]:
        """Return claims in ledger order for one reporting verdict."""

        return tuple(claim for claim in self.claims if claim.verdict is verdict)


def load_claim_ledger(path: Path) -> ClaimLedger:
    """Load and validate a claim-ledger JSON document."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ClaimLedgerError(f"unable to read claim ledger: {path}") from error
    except json.JSONDecodeError as error:
        raise ClaimLedgerError(f"invalid claim ledger JSON: {path}") from error

    root = _require_mapping(payload, "claim ledger")
    evidence = tuple(
        _evidence_from_mapping(_require_mapping(item, "evidence record"))
        for item in _required_list(root, "evidence")
    )
    claims = tuple(
        _claim_from_mapping(_require_mapping(item, "claim record"))
        for item in _required_list(root, "claims")
    )

    return ClaimLedger(
        schema_version=_required_int(root, "schema_version"),
        ledger_id=_required_text(root, "ledger_id"),
        title=_required_text(root, "title"),
        scope_note=_required_text(root, "scope_note"),
        evidence=evidence,
        claims=claims,
    )


def load_verified_claim_ledger(
    repository_root: Path,
    *,
    manifest_path: PurePosixPath = CLAIM_LEDGER_MANIFEST_PATH,
    ledger_path: PurePosixPath = CLAIM_LEDGER_REFERENCE_PATH,
) -> ClaimLedger:
    """Verify provenance before loading the canonical claim ledger."""

    root = repository_root.resolve()
    manifest = load_source_manifest(root / manifest_path)
    require_verified_artifacts(manifest, root)

    matches = tuple(
        artifact for artifact in manifest.artifacts if artifact.path == str(ledger_path)
    )
    if len(matches) != 1:
        raise ProvenanceError(
            f"expected one manifest artifact for {ledger_path}, found {len(matches)}"
        )

    ledger = load_claim_ledger(root / ledger_path)
    if len(ledger.claims) != matches[0].record_count:
        raise ClaimLedgerError(
            f"manifest record_count is {matches[0].record_count}, "
            f"but the claim ledger contains {len(ledger.claims)} claims"
        )

    manifest_source_urls = {source.url for source in manifest.sources}
    undeclared_urls = tuple(
        sorted(
            record.locator
            for record in ledger.evidence
            if record.kind is EvidenceKind.SOURCE and record.locator not in manifest_source_urls
        )
    )
    if undeclared_urls:
        raise ProvenanceError(
            f"claim ledger references source URLs absent from the manifest: {undeclared_urls}"
        )

    return ledger


def _evidence_from_mapping(value: Mapping[str, object]) -> EvidenceRecord:
    kind_text = _required_text(value, "kind")
    try:
        kind = EvidenceKind(kind_text)
    except ValueError as error:
        raise ClaimLedgerError(f"unsupported evidence kind: {kind_text!r}") from error

    return EvidenceRecord(
        evidence_id=_required_text(value, "evidence_id"),
        kind=kind,
        locator=_required_text(value, "locator"),
        description=_required_text(value, "description"),
    )


def _claim_from_mapping(value: Mapping[str, object]) -> ClaimRecord:
    classification_text = _required_text(value, "classification")
    verdict_text = _required_text(value, "verdict")

    try:
        classification = ClaimClassification(classification_text)
    except ValueError as error:
        raise ClaimLedgerError(
            f"unsupported claim classification: {classification_text!r}"
        ) from error

    try:
        verdict = ClaimVerdict(verdict_text)
    except ValueError as error:
        raise ClaimLedgerError(f"unsupported claim verdict: {verdict_text!r}") from error

    return ClaimRecord(
        claim_id=_required_text(value, "claim_id"),
        statement=_required_text(value, "statement"),
        classification=classification,
        verdict=verdict,
        evidence_ids=_required_text_tuple(value, "evidence_ids"),
        depends_on=_required_text_tuple(value, "depends_on"),
        limitations=_required_text_tuple(value, "limitations"),
    )


def _require_acyclic_claim_graph(claims: tuple[ClaimRecord, ...]) -> None:
    by_id = {claim.claim_id: claim for claim in claims}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(claim_id: str) -> None:
        if claim_id in visited:
            return
        if claim_id in visiting:
            raise ClaimLedgerError(f"claim dependency graph contains a cycle at {claim_id!r}")

        visiting.add(claim_id)
        for dependency_id in by_id[claim_id].depends_on:
            visit(dependency_id)
        visiting.remove(claim_id)
        visited.add(claim_id)

    for claim in claims:
        visit(claim.claim_id)


def _require_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ClaimLedgerError(f"{field_name} must be a lowercase hyphen-delimited identifier")


def _require_unique_identifiers(
    values: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool,
) -> None:
    if not values and not allow_empty:
        raise ClaimLedgerError(f"{field_name} must not be empty")
    for value in values:
        _require_identifier(value, field_name)
    if len(set(values)) != len(values):
        raise ClaimLedgerError(f"{field_name} must contain unique values")


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ClaimLedgerError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ClaimLedgerError(f"{field_name} keys must be strings")
    return cast(dict[str, object], value)


def _required_list(
    value: Mapping[str, object],
    field_name: str,
) -> list[object]:
    item = value.get(field_name)
    if not isinstance(item, list):
        raise ClaimLedgerError(f"{field_name} must be a JSON array")
    return cast(list[object], item)


def _required_text(
    value: Mapping[str, object],
    field_name: str,
) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise ClaimLedgerError(f"{field_name} must be a non-empty string")
    return item


def _required_int(
    value: Mapping[str, object],
    field_name: str,
) -> int:
    item = value.get(field_name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ClaimLedgerError(f"{field_name} must be an integer")
    return item


def _required_text_tuple(
    value: Mapping[str, object],
    field_name: str,
) -> tuple[str, ...]:
    items = _required_list(value, field_name)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ClaimLedgerError(f"{field_name} must contain only non-empty strings")
    return tuple(cast(str, item) for item in items)
