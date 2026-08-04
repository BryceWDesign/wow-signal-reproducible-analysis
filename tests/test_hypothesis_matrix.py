from __future__ import annotations

import json
from pathlib import Path

import pytest

from wow_signal_analysis.claim_ledger import (
    ClaimClassification,
    ClaimLedger,
    ClaimRecord,
    ClaimVerdict,
    EvidenceKind,
    EvidenceRecord,
)
from wow_signal_analysis.hypothesis_matrix import (
    HYPOTHESIS_MATRIX_MANIFEST_PATH,
    HYPOTHESIS_MATRIX_REFERENCE_PATH,
    BoundHypothesis,
    HypothesisMatrixError,
    HypothesisRecord,
    HypothesisScope,
    HypothesisStatus,
    bind_hypothesis_matrix,
    load_hypothesis_matrix,
    load_verified_hypothesis_matrix,
)
from wow_signal_analysis.provenance import ProvenanceError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MATRIX_PATH = _REPOSITORY_ROOT / HYPOTHESIS_MATRIX_REFERENCE_PATH


def _claim(
    claim_id: str,
    classification: ClaimClassification,
    verdict: ClaimVerdict,
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        statement=f"Claim record for {claim_id}.",
        classification=classification,
        verdict=verdict,
        evidence_ids=("analysis-test",),
        depends_on=(),
        limitations=("Test claim used to validate hypothesis binding.",),
    )


def _claim_ledger() -> ClaimLedger:
    evidence = EvidenceRecord(
        evidence_id="analysis-test",
        kind=EvidenceKind.ANALYSIS,
        locator="example.module:run",
        description="Test analysis used by hypothesis binding fixtures.",
    )

    return ClaimLedger(
        schema_version=1,
        ledger_id="wow-signal-evidence-claims-v1",
        title="Test claim ledger",
        scope_note="Claims required by the canonical hypothesis matrix.",
        evidence=(evidence,),
        claims=(
            _claim(
                "derived-gaussian-compatibility",
                ClaimClassification.DERIVED,
                ClaimVerdict.REPRODUCIBLE,
            ),
            _claim(
                "derived-gaussian-loocv-ranking",
                ClaimClassification.DERIVED,
                ClaimVerdict.REPRODUCIBLE,
            ),
            _claim(
                "compatible-beacon-hypothesis",
                ClaimClassification.COMPATIBILITY,
                ClaimVerdict.COMPATIBLE_NOT_PROVEN,
            ),
            _claim(
                "speculative-extraterrestrial-technology",
                ClaimClassification.SPECULATIVE,
                ClaimVerdict.NOT_ESTABLISHED,
            ),
            _claim(
                "derived-question-mark-correspondence",
                ClaimClassification.DERIVED,
                ClaimVerdict.REPRODUCIBLE,
            ),
            _claim(
                "derived-comma-correspondence",
                ClaimClassification.DERIVED,
                ClaimVerdict.REPRODUCIBLE,
            ),
            _claim(
                "derived-question-null-frequency",
                ClaimClassification.DERIVED,
                ClaimVerdict.REPRODUCIBLE,
            ),
            _claim(
                "speculative-intentional-message",
                ClaimClassification.SPECULATIVE,
                ClaimVerdict.NOT_ESTABLISHED,
            ),
            _claim(
                "derived-frequency-context",
                ClaimClassification.DERIVED,
                ClaimVerdict.REPRODUCIBLE,
            ),
        ),
    )


def test_verified_matrix_binds_all_hypotheses_to_the_claim_ledger() -> None:
    bound = load_verified_hypothesis_matrix(
        _REPOSITORY_ROOT,
        claim_ledger=_claim_ledger(),
    )

    assert bound.matrix_id == "wow-signal-hypothesis-matrix-v1"
    assert bound.ledger_id == "wow-signal-evidence-claims-v1"
    assert len(bound.hypotheses) == 5
    assert tuple(hypothesis.record.hypothesis_id for hypothesis in bound.hypotheses) == (
        "gaussian-transit-shape",
        "stable-artificial-carrier-or-beacon",
        "natural-versus-artificial-origin",
        "intentional-morse-question",
        "extraterrestrial-technology-origin",
    )


def test_matrix_preserves_scope_and_status_counts() -> None:
    matrix = load_hypothesis_matrix(_MATRIX_PATH)

    assert {scope: len(matrix.hypotheses_by_scope(scope)) for scope in HypothesisScope} == {
        HypothesisScope.SHAPE_MODEL: 1,
        HypothesisScope.SOURCE_FUNCTION: 1,
        HypothesisScope.SOURCE_ORIGIN: 2,
        HypothesisScope.SYMBOLIC_INTENT: 1,
    }
    assert {status: len(matrix.hypotheses_by_status(status)) for status in HypothesisStatus} == {
        HypothesisStatus.SUPPORTED_AS_MODEL: 1,
        HypothesisStatus.COMPATIBLE_NOT_PROVEN: 1,
        HypothesisStatus.NOT_DISCRIMINATED: 1,
        HypothesisStatus.NOT_ESTABLISHED: 2,
    }


def test_shape_model_status_uses_only_reproducible_derived_claims() -> None:
    bound = load_verified_hypothesis_matrix(
        _REPOSITORY_ROOT,
        claim_ledger=_claim_ledger(),
    )
    hypothesis = bound.hypothesis_by_id("gaussian-transit-shape")

    assert hypothesis.record.status is HypothesisStatus.SUPPORTED_AS_MODEL
    assert set(hypothesis.classifications) == {ClaimClassification.DERIVED}
    assert set(hypothesis.verdicts) == {ClaimVerdict.REPRODUCIBLE}
    assert "does not identify" in hypothesis.record.limitations[0]


def test_carrier_hypothesis_is_compatible_but_not_promoted_to_fact() -> None:
    bound = load_verified_hypothesis_matrix(
        _REPOSITORY_ROOT,
        claim_ledger=_claim_ledger(),
    )
    hypothesis = bound.hypothesis_by_id("stable-artificial-carrier-or-beacon")

    assert hypothesis.record.status is HypothesisStatus.COMPATIBLE_NOT_PROVEN
    assert hypothesis.classifications == (
        ClaimClassification.COMPATIBILITY,
        ClaimClassification.SPECULATIVE,
    )
    assert hypothesis.verdicts == (
        ClaimVerdict.COMPATIBLE_NOT_PROVEN,
        ClaimVerdict.NOT_ESTABLISHED,
    )


def test_origin_discrimination_requires_all_evidence_classes() -> None:
    bound = load_verified_hypothesis_matrix(
        _REPOSITORY_ROOT,
        claim_ledger=_claim_ledger(),
    )
    hypothesis = bound.hypothesis_by_id("natural-versus-artificial-origin")

    assert hypothesis.record.status is HypothesisStatus.NOT_DISCRIMINATED
    assert set(hypothesis.classifications) == {
        ClaimClassification.DERIVED,
        ClaimClassification.COMPATIBILITY,
        ClaimClassification.SPECULATIVE,
    }
    assert "information-theoretic" in hypothesis.record.would_weaken[0]


def test_morse_intent_and_extraterrestrial_origin_remain_unestablished() -> None:
    bound = load_verified_hypothesis_matrix(
        _REPOSITORY_ROOT,
        claim_ledger=_claim_ledger(),
    )
    morse = bound.hypothesis_by_id("intentional-morse-question")
    extraterrestrial = bound.hypothesis_by_id("extraterrestrial-technology-origin")

    assert morse.record.status is HypothesisStatus.NOT_ESTABLISHED
    assert extraterrestrial.record.status is HypothesisStatus.NOT_ESTABLISHED
    assert ClaimVerdict.NOT_ESTABLISHED in morse.verdicts
    assert ClaimVerdict.NOT_ESTABLISHED in extraterrestrial.verdicts
    assert "receiver-strength codes" in morse.record.limitations[0]
    assert "does not assign" in extraterrestrial.record.limitations[0]


def test_binding_fails_when_a_claim_id_is_absent() -> None:
    matrix = load_hypothesis_matrix(_MATRIX_PATH)
    complete_ledger = _claim_ledger()
    incomplete_ledger = ClaimLedger(
        schema_version=1,
        ledger_id="incomplete-ledger",
        title="Incomplete ledger",
        scope_note="Validation test.",
        evidence=complete_ledger.evidence,
        claims=complete_ledger.claims[:-1],
    )

    with pytest.raises(ValueError, match="found 0"):
        bind_hypothesis_matrix(matrix, incomplete_ledger)


def test_supported_model_cannot_depend_on_a_speculative_claim() -> None:
    record = HypothesisRecord(
        hypothesis_id="invalid-supported-model",
        label="Invalid supported model",
        scope=HypothesisScope.SHAPE_MODEL,
        proposition="A model is treated as supported by speculation.",
        status=HypothesisStatus.SUPPORTED_AS_MODEL,
        claim_ids=(
            "derived-gaussian-compatibility",
            "speculative-extraterrestrial-technology",
        ),
        rationale="This record is intentionally invalid.",
        would_strengthen=("No valid strengthening condition.",),
        would_weaken=("No valid weakening condition.",),
        limitations=("This record is intentionally invalid.",),
    )
    ledger = _claim_ledger()
    claims = tuple(ledger.claim_by_id(claim_id) for claim_id in record.claim_ids)

    with pytest.raises(
        HypothesisMatrixError,
        match="cannot rely on compatibility",
    ):
        BoundHypothesis(record=record, claims=claims)


def test_scope_rejects_an_incompatible_status() -> None:
    with pytest.raises(HypothesisMatrixError, match="does not permit status"):
        HypothesisRecord(
            hypothesis_id="invalid-symbolic-status",
            label="Invalid symbolic status",
            scope=HypothesisScope.SYMBOLIC_INTENT,
            proposition="An unsupported symbolic claim is promoted.",
            status=HypothesisStatus.SUPPORTED_AS_MODEL,
            claim_ids=("derived-question-mark-correspondence",),
            rationale="This record is intentionally invalid.",
            would_strengthen=("No valid strengthening condition.",),
            would_weaken=("No valid weakening condition.",),
            limitations=("This record is intentionally invalid.",),
        )


def test_loader_rejects_invalid_json_and_unknown_status(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")

    with pytest.raises(
        HypothesisMatrixError,
        match="invalid hypothesis matrix JSON",
    ):
        load_hypothesis_matrix(invalid_json)

    payload = json.loads(_MATRIX_PATH.read_text(encoding="utf-8"))
    payload["hypotheses"][0]["status"] = "proven"
    invalid_status = tmp_path / "invalid-status.json"
    invalid_status.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        HypothesisMatrixError,
        match="unsupported hypothesis status",
    ):
        load_hypothesis_matrix(invalid_status)


def test_verified_loader_detects_matrix_tampering(tmp_path: Path) -> None:
    manifest_target = tmp_path / HYPOTHESIS_MATRIX_MANIFEST_PATH
    matrix_target = tmp_path / HYPOTHESIS_MATRIX_REFERENCE_PATH
    manifest_target.parent.mkdir(parents=True)
    matrix_target.parent.mkdir(parents=True)

    manifest_target.write_bytes((_REPOSITORY_ROOT / HYPOTHESIS_MATRIX_MANIFEST_PATH).read_bytes())
    matrix_target.write_text(
        _MATRIX_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProvenanceError, match="hash-mismatch"):
        load_verified_hypothesis_matrix(
            tmp_path,
            claim_ledger=_claim_ledger(),
        )


def test_verified_loader_rejects_manifest_record_count_drift(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / HYPOTHESIS_MATRIX_MANIFEST_PATH
    matrix_target = tmp_path / HYPOTHESIS_MATRIX_REFERENCE_PATH
    manifest_target.parent.mkdir(parents=True)
    matrix_target.parent.mkdir(parents=True)

    manifest_payload = json.loads(
        (_REPOSITORY_ROOT / HYPOTHESIS_MATRIX_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest_payload["artifacts"][0]["record_count"] = 4
    manifest_target.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    matrix_target.write_bytes(_MATRIX_PATH.read_bytes())

    with pytest.raises(HypothesisMatrixError, match="record_count is 4"):
        load_verified_hypothesis_matrix(
            tmp_path,
            claim_ledger=_claim_ledger(),
        )
