"""Evidence-bound hypothesis matrix for Wow! signal reporting."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast

from wow_signal_analysis.claim_ledger import (
    ClaimClassification,
    ClaimLedger,
    ClaimRecord,
    ClaimVerdict,
    load_verified_claim_ledger,
)
from wow_signal_analysis.provenance import (
    ProvenanceError,
    load_source_manifest,
    require_verified_artifacts,
)

HYPOTHESIS_MATRIX_REFERENCE_PATH: Final = PurePosixPath("data/reference/hypothesis_matrix.json")
HYPOTHESIS_MATRIX_MANIFEST_PATH: Final = PurePosixPath(
    "data/provenance/hypothesis_source_manifest.json"
)

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class HypothesisMatrixError(ValueError):
    """Raised when hypothesis metadata are malformed or overstate the evidence."""


class HypothesisScope(StrEnum):
    """Question layer addressed by one hypothesis record."""

    SHAPE_MODEL = "shape-model"
    SOURCE_FUNCTION = "source-function"
    SOURCE_ORIGIN = "source-origin"
    SYMBOLIC_INTENT = "symbolic-intent"


class HypothesisStatus(StrEnum):
    """Evidence-bound status permitted for a hypothesis."""

    SUPPORTED_AS_MODEL = "supported-as-model"
    COMPATIBLE_NOT_PROVEN = "compatible-not-proven"
    NOT_DISCRIMINATED = "not-discriminated"
    NOT_ESTABLISHED = "not-established"


_ALLOWED_STATUS_BY_SCOPE: Final = {
    HypothesisScope.SHAPE_MODEL: frozenset({HypothesisStatus.SUPPORTED_AS_MODEL}),
    HypothesisScope.SOURCE_FUNCTION: frozenset(
        {
            HypothesisStatus.COMPATIBLE_NOT_PROVEN,
            HypothesisStatus.NOT_ESTABLISHED,
        }
    ),
    HypothesisScope.SOURCE_ORIGIN: frozenset(
        {
            HypothesisStatus.NOT_DISCRIMINATED,
            HypothesisStatus.NOT_ESTABLISHED,
        }
    ),
    HypothesisScope.SYMBOLIC_INTENT: frozenset({HypothesisStatus.NOT_ESTABLISHED}),
}


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """One proposition with an explicit scope, status, and test conditions."""

    hypothesis_id: str
    label: str
    scope: HypothesisScope
    proposition: str
    status: HypothesisStatus
    claim_ids: tuple[str, ...]
    rationale: str
    would_strengthen: tuple[str, ...]
    would_weaken: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.hypothesis_id, "hypothesis_id")
        for field_name in ("label", "proposition", "rationale"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HypothesisMatrixError(f"{field_name} must be non-empty")

        if self.status not in _ALLOWED_STATUS_BY_SCOPE[self.scope]:
            raise HypothesisMatrixError(
                f"scope {self.scope.value!r} does not permit status {self.status.value!r}"
            )

        _require_unique_identifiers(self.claim_ids, "claim_ids")
        _require_nonempty_unique_text(
            self.would_strengthen,
            "would_strengthen",
        )
        _require_nonempty_unique_text(
            self.would_weaken,
            "would_weaken",
        )
        _require_nonempty_unique_text(self.limitations, "limitations")


@dataclass(frozen=True, slots=True)
class HypothesisMatrix:
    """Unbound hypothesis records loaded from the reference artifact."""

    schema_version: int
    matrix_id: str
    title: str
    scope_note: str
    hypotheses: tuple[HypothesisRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise HypothesisMatrixError("unsupported hypothesis matrix schema_version")
        _require_identifier(self.matrix_id, "matrix_id")
        if not self.title.strip() or not self.scope_note.strip():
            raise HypothesisMatrixError("title and scope_note must be non-empty")
        if not self.hypotheses:
            raise HypothesisMatrixError("hypothesis matrix must contain at least one hypothesis")

        hypothesis_ids = tuple(hypothesis.hypothesis_id for hypothesis in self.hypotheses)
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise HypothesisMatrixError("hypothesis IDs must be unique")

    def hypothesis_by_id(self, hypothesis_id: str) -> HypothesisRecord:
        """Return one unique unbound hypothesis record."""

        matches = tuple(
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.hypothesis_id == hypothesis_id
        )
        if len(matches) != 1:
            raise HypothesisMatrixError(
                f"expected one hypothesis for {hypothesis_id!r}, found {len(matches)}"
            )
        return matches[0]

    def hypotheses_by_scope(
        self,
        scope: HypothesisScope,
    ) -> tuple[HypothesisRecord, ...]:
        """Return hypotheses in source order for one question layer."""

        return tuple(hypothesis for hypothesis in self.hypotheses if hypothesis.scope is scope)

    def hypotheses_by_status(
        self,
        status: HypothesisStatus,
    ) -> tuple[HypothesisRecord, ...]:
        """Return hypotheses in source order for one evidence status."""

        return tuple(hypothesis for hypothesis in self.hypotheses if hypothesis.status is status)


@dataclass(frozen=True, slots=True)
class BoundHypothesis:
    """One hypothesis with all cited claim-ledger records resolved."""

    record: HypothesisRecord
    claims: tuple[ClaimRecord, ...]

    def __post_init__(self) -> None:
        if tuple(claim.claim_id for claim in self.claims) != self.record.claim_ids:
            raise HypothesisMatrixError("bound claims must preserve the hypothesis claim_ids order")
        _validate_status_basis(self.record.status, self.claims)

    @property
    def classifications(self) -> tuple[ClaimClassification, ...]:
        """Return cited claim classifications in declared order."""

        return tuple(claim.classification for claim in self.claims)

    @property
    def verdicts(self) -> tuple[ClaimVerdict, ...]:
        """Return cited claim verdicts in declared order."""

        return tuple(claim.verdict for claim in self.claims)


@dataclass(frozen=True, slots=True)
class BoundHypothesisMatrix:
    """Hypothesis matrix validated against one concrete claim ledger."""

    matrix_id: str
    ledger_id: str
    hypotheses: tuple[BoundHypothesis, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.matrix_id, "matrix_id")
        _require_identifier(self.ledger_id, "ledger_id")
        if not self.hypotheses:
            raise HypothesisMatrixError("bound hypothesis matrix must contain hypotheses")

        hypothesis_ids = tuple(hypothesis.record.hypothesis_id for hypothesis in self.hypotheses)
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise HypothesisMatrixError("bound hypothesis IDs must be unique")

    def hypothesis_by_id(self, hypothesis_id: str) -> BoundHypothesis:
        """Return one unique evidence-bound hypothesis."""

        matches = tuple(
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.record.hypothesis_id == hypothesis_id
        )
        if len(matches) != 1:
            raise HypothesisMatrixError(
                f"expected one bound hypothesis for {hypothesis_id!r}, found {len(matches)}"
            )
        return matches[0]

    def hypotheses_by_status(
        self,
        status: HypothesisStatus,
    ) -> tuple[BoundHypothesis, ...]:
        """Return bound hypotheses carrying one evidence status."""

        return tuple(
            hypothesis for hypothesis in self.hypotheses if hypothesis.record.status is status
        )


def load_hypothesis_matrix(path: Path) -> HypothesisMatrix:
    """Load and validate a hypothesis-matrix JSON document."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HypothesisMatrixError(f"unable to read hypothesis matrix: {path}") from error
    except json.JSONDecodeError as error:
        raise HypothesisMatrixError(f"invalid hypothesis matrix JSON: {path}") from error

    root = _require_mapping(payload, "hypothesis matrix")
    hypotheses = tuple(
        _hypothesis_from_mapping(_require_mapping(item, "hypothesis record"))
        for item in _required_list(root, "hypotheses")
    )

    return HypothesisMatrix(
        schema_version=_required_int(root, "schema_version"),
        matrix_id=_required_text(root, "matrix_id"),
        title=_required_text(root, "title"),
        scope_note=_required_text(root, "scope_note"),
        hypotheses=hypotheses,
    )


def bind_hypothesis_matrix(
    matrix: HypothesisMatrix,
    claim_ledger: ClaimLedger,
) -> BoundHypothesisMatrix:
    """Resolve every hypothesis against the claim ledger and enforce status basis."""

    bound_hypotheses: list[BoundHypothesis] = []
    for hypothesis in matrix.hypotheses:
        claims = tuple(claim_ledger.claim_by_id(claim_id) for claim_id in hypothesis.claim_ids)
        bound_hypotheses.append(BoundHypothesis(record=hypothesis, claims=claims))

    return BoundHypothesisMatrix(
        matrix_id=matrix.matrix_id,
        ledger_id=claim_ledger.ledger_id,
        hypotheses=tuple(bound_hypotheses),
    )


def load_verified_hypothesis_matrix(
    repository_root: Path,
    *,
    manifest_path: PurePosixPath = HYPOTHESIS_MATRIX_MANIFEST_PATH,
    matrix_path: PurePosixPath = HYPOTHESIS_MATRIX_REFERENCE_PATH,
    claim_ledger: ClaimLedger | None = None,
) -> BoundHypothesisMatrix:
    """Verify provenance, load the matrix, and bind it to the claim ledger."""

    root = repository_root.resolve()
    manifest = load_source_manifest(root / manifest_path)
    require_verified_artifacts(manifest, root)

    matches = tuple(
        artifact for artifact in manifest.artifacts if artifact.path == str(matrix_path)
    )
    if len(matches) != 1:
        raise ProvenanceError(
            f"expected one manifest artifact for {matrix_path}, found {len(matches)}"
        )

    matrix = load_hypothesis_matrix(root / matrix_path)
    if len(matrix.hypotheses) != matches[0].record_count:
        raise HypothesisMatrixError(
            f"manifest record_count is {matches[0].record_count}, "
            f"but the hypothesis matrix contains "
            f"{len(matrix.hypotheses)} hypotheses"
        )

    ledger = claim_ledger or load_verified_claim_ledger(root)
    return bind_hypothesis_matrix(matrix, ledger)


def _validate_status_basis(
    status: HypothesisStatus,
    claims: tuple[ClaimRecord, ...],
) -> None:
    if not claims:
        raise HypothesisMatrixError("hypothesis status basis must contain at least one claim")

    basis = {(claim.classification, claim.verdict) for claim in claims}
    derived_reproducible = (
        ClaimClassification.DERIVED,
        ClaimVerdict.REPRODUCIBLE,
    )
    compatibility_basis = (
        ClaimClassification.COMPATIBILITY,
        ClaimVerdict.COMPATIBLE_NOT_PROVEN,
    )
    speculative_basis = (
        ClaimClassification.SPECULATIVE,
        ClaimVerdict.NOT_ESTABLISHED,
    )

    if status is HypothesisStatus.SUPPORTED_AS_MODEL:
        if derived_reproducible not in basis:
            raise HypothesisMatrixError("supported-as-model requires a reproducible derived claim")
        if any(
            claim.classification
            not in {
                ClaimClassification.OBSERVED,
                ClaimClassification.DERIVED,
            }
            for claim in claims
        ):
            raise HypothesisMatrixError(
                "supported-as-model cannot rely on compatibility, "
                "interpretive, or speculative claims"
            )
        return

    if status is HypothesisStatus.COMPATIBLE_NOT_PROVEN:
        if compatibility_basis not in basis or speculative_basis not in basis:
            raise HypothesisMatrixError(
                "compatible-not-proven requires compatibility and "
                "not-established speculative claims"
            )
        return

    if status is HypothesisStatus.NOT_DISCRIMINATED:
        required = {
            derived_reproducible,
            compatibility_basis,
            speculative_basis,
        }
        if not required.issubset(basis):
            raise HypothesisMatrixError(
                "not-discriminated requires derived, compatibility, and speculative claim bases"
            )
        return

    if speculative_basis not in basis:
        raise HypothesisMatrixError("not-established requires a not-established speculative claim")


def _hypothesis_from_mapping(
    value: Mapping[str, object],
) -> HypothesisRecord:
    scope_text = _required_text(value, "scope")
    status_text = _required_text(value, "status")

    try:
        scope = HypothesisScope(scope_text)
    except ValueError as error:
        raise HypothesisMatrixError(f"unsupported hypothesis scope: {scope_text!r}") from error

    try:
        status = HypothesisStatus(status_text)
    except ValueError as error:
        raise HypothesisMatrixError(f"unsupported hypothesis status: {status_text!r}") from error

    return HypothesisRecord(
        hypothesis_id=_required_text(value, "hypothesis_id"),
        label=_required_text(value, "label"),
        scope=scope,
        proposition=_required_text(value, "proposition"),
        status=status,
        claim_ids=_required_text_tuple(value, "claim_ids"),
        rationale=_required_text(value, "rationale"),
        would_strengthen=_required_text_tuple(value, "would_strengthen"),
        would_weaken=_required_text_tuple(value, "would_weaken"),
        limitations=_required_text_tuple(value, "limitations"),
    )


def _require_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise HypothesisMatrixError(f"{field_name} must be a lowercase hyphen-delimited identifier")


def _require_unique_identifiers(
    values: tuple[str, ...],
    field_name: str,
) -> None:
    if not values:
        raise HypothesisMatrixError(f"{field_name} must not be empty")
    for value in values:
        _require_identifier(value, field_name)
    if len(set(values)) != len(values):
        raise HypothesisMatrixError(f"{field_name} must contain unique values")


def _require_nonempty_unique_text(
    values: tuple[str, ...],
    field_name: str,
) -> None:
    if not values:
        raise HypothesisMatrixError(f"{field_name} must not be empty")
    if any(not value.strip() for value in values):
        raise HypothesisMatrixError(f"{field_name} must contain only non-empty strings")
    if len(set(values)) != len(values):
        raise HypothesisMatrixError(f"{field_name} must contain unique values")


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise HypothesisMatrixError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise HypothesisMatrixError(f"{field_name} keys must be strings")
    return cast(dict[str, object], value)


def _required_list(
    value: Mapping[str, object],
    field_name: str,
) -> list[object]:
    item = value.get(field_name)
    if not isinstance(item, list):
        raise HypothesisMatrixError(f"{field_name} must be a JSON array")
    return cast(list[object], item)


def _required_text(
    value: Mapping[str, object],
    field_name: str,
) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise HypothesisMatrixError(f"{field_name} must be a non-empty string")
    return item


def _required_int(
    value: Mapping[str, object],
    field_name: str,
) -> int:
    item = value.get(field_name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise HypothesisMatrixError(f"{field_name} must be an integer")
    return item


def _required_text_tuple(
    value: Mapping[str, object],
    field_name: str,
) -> tuple[str, ...]:
    items = _required_list(value, field_name)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise HypothesisMatrixError(f"{field_name} must contain only non-empty strings")
    return tuple(cast(str, item) for item in items)
