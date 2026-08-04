from __future__ import annotations

import json
from pathlib import Path

import pytest

from wow_signal_analysis.claim_ledger import (
    CLAIM_LEDGER_MANIFEST_PATH,
    CLAIM_LEDGER_REFERENCE_PATH,
    ClaimClassification,
    ClaimLedger,
    ClaimLedgerError,
    ClaimRecord,
    ClaimVerdict,
    EvidenceKind,
    EvidenceRecord,
    load_claim_ledger,
    load_verified_claim_ledger,
)
from wow_signal_analysis.provenance import ProvenanceError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LEDGER_PATH = _REPOSITORY_ROOT / CLAIM_LEDGER_REFERENCE_PATH


def test_verified_ledger_preserves_all_claim_classes() -> None:
    ledger = load_verified_claim_ledger(_REPOSITORY_ROOT)

    assert ledger.schema_version == 1
    assert ledger.ledger_id == "wow-signal-evidence-claims-v1"
    assert len(ledger.evidence) == 9
    assert len(ledger.claims) == 12

    assert {
        classification: len(ledger.claims_by_classification(classification))
        for classification in ClaimClassification
    } == {
        ClaimClassification.OBSERVED: 1,
        ClaimClassification.DERIVED: 7,
        ClaimClassification.COMPATIBILITY: 1,
        ClaimClassification.INTERPRETIVE: 1,
        ClaimClassification.SPECULATIVE: 2,
    }


def test_classifications_are_bound_to_non_overclaiming_verdicts() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)

    assert {claim.classification: claim.verdict for claim in ledger.claims} == {
        ClaimClassification.OBSERVED: ClaimVerdict.SUPPORTED,
        ClaimClassification.DERIVED: ClaimVerdict.REPRODUCIBLE,
        ClaimClassification.COMPATIBILITY: (ClaimVerdict.COMPATIBLE_NOT_PROVEN),
        ClaimClassification.INTERPRETIVE: ClaimVerdict.SUMMARY_ONLY,
        ClaimClassification.SPECULATIVE: ClaimVerdict.NOT_ESTABLISHED,
    }


def test_topological_order_places_dependencies_before_dependents() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)
    ordered = ledger.topological_claims
    positions = {claim.claim_id: index for index, claim in enumerate(ordered)}

    assert len(ordered) == len(ledger.claims)
    assert len({claim.claim_id for claim in ordered}) == len(ordered)

    for claim in ordered:
        assert all(
            positions[dependency] < positions[claim.claim_id] for dependency in claim.depends_on
        )


def test_symbolic_correspondence_is_recorded_as_non_unique() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)
    question = ledger.claim_by_id("derived-question-mark-correspondence")
    comma = ledger.claim_by_id("derived-comma-correspondence")
    null_frequency = ledger.claim_by_id("derived-question-null-frequency")

    assert question.verdict is ClaimVerdict.REPRODUCIBLE
    assert "analyst-declared" in question.limitations[0]
    assert comma.depends_on == ("derived-question-mark-correspondence",)
    assert "non-unique" in comma.limitations[0]
    assert "96 of 720" in null_frequency.statement
    assert "not a probability" in null_frequency.limitations[0]


def test_five_layer_phrase_is_summary_only_not_recovered_plaintext() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)
    summary = ledger.claim_by_id("interpretive-five-layer-summary")

    assert summary.classification is ClaimClassification.INTERPRETIVE
    assert summary.verdict is ClaimVerdict.SUMMARY_ONLY
    assert "do not constitute recovered plaintext" in summary.limitations[0]
    assert "compatible-beacon-hypothesis" in summary.depends_on


def test_extraterrestrial_origin_and_message_intent_remain_unestablished() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)

    intentional_message = ledger.claim_by_id("speculative-intentional-message")
    extraterrestrial = ledger.claim_by_id("speculative-extraterrestrial-technology")

    assert intentional_message.verdict is ClaimVerdict.NOT_ESTABLISHED
    assert extraterrestrial.verdict is ClaimVerdict.NOT_ESTABLISHED
    assert "payload" in intentional_message.limitations[0]
    assert "not evidence sufficient" in extraterrestrial.limitations[0]


def test_evidence_lookups_preserve_local_and_external_provenance() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)

    dataset = ledger.evidence_by_id("dataset-wow-6equj5")
    source = ledger.evidence_by_id("source-big-ear-report")

    assert dataset.kind is EvidenceKind.DATASET
    assert dataset.locator == "data/raw/wow_6equj5.csv"
    assert source.kind is EvidenceKind.SOURCE
    assert source.locator.startswith("https://www.bigear.org/")


def test_unknown_lookups_fail_closed() -> None:
    ledger = load_claim_ledger(_LEDGER_PATH)

    with pytest.raises(ClaimLedgerError, match="found 0"):
        ledger.claim_by_id("missing-claim")

    with pytest.raises(ClaimLedgerError, match="found 0"):
        ledger.evidence_by_id("missing-evidence")


def test_claim_record_rejects_a_verdict_outside_its_classification() -> None:
    with pytest.raises(ClaimLedgerError, match="requires verdict"):
        ClaimRecord(
            claim_id="invalid-derived-claim",
            statement="A derived claim cannot claim direct support.",
            classification=ClaimClassification.DERIVED,
            verdict=ClaimVerdict.SUPPORTED,
            evidence_ids=("analysis-test",),
            depends_on=(),
            limitations=("This record is intentionally invalid.",),
        )


def test_ledger_rejects_unknown_evidence_and_dependency_cycles() -> None:
    evidence = EvidenceRecord(
        evidence_id="analysis-test",
        kind=EvidenceKind.ANALYSIS,
        locator="example.module:run",
        description="Test analysis.",
    )
    unknown_evidence_claim = ClaimRecord(
        claim_id="unknown-evidence-claim",
        statement="This claim cites evidence absent from its ledger.",
        classification=ClaimClassification.DERIVED,
        verdict=ClaimVerdict.REPRODUCIBLE,
        evidence_ids=("missing-evidence",),
        depends_on=(),
        limitations=("This record is intentionally invalid.",),
    )

    with pytest.raises(ClaimLedgerError, match="unknown evidence"):
        ClaimLedger(
            schema_version=1,
            ledger_id="unknown-evidence-ledger",
            title="Unknown evidence test",
            scope_note="Validation test.",
            evidence=(evidence,),
            claims=(unknown_evidence_claim,),
        )

    first = ClaimRecord(
        claim_id="first-cycle-claim",
        statement="First cycle claim.",
        classification=ClaimClassification.DERIVED,
        verdict=ClaimVerdict.REPRODUCIBLE,
        evidence_ids=("analysis-test",),
        depends_on=("second-cycle-claim",),
        limitations=("This record is intentionally invalid.",),
    )
    second = ClaimRecord(
        claim_id="second-cycle-claim",
        statement="Second cycle claim.",
        classification=ClaimClassification.DERIVED,
        verdict=ClaimVerdict.REPRODUCIBLE,
        evidence_ids=("analysis-test",),
        depends_on=("first-cycle-claim",),
        limitations=("This record is intentionally invalid.",),
    )

    with pytest.raises(ClaimLedgerError, match="contains a cycle"):
        ClaimLedger(
            schema_version=1,
            ledger_id="cycle-ledger",
            title="Cycle test",
            scope_note="Validation test.",
            evidence=(evidence,),
            claims=(first, second),
        )


def test_derived_claim_cannot_depend_on_a_speculative_claim() -> None:
    evidence = EvidenceRecord(
        evidence_id="analysis-test",
        kind=EvidenceKind.ANALYSIS,
        locator="example.module:run",
        description="Test analysis.",
    )
    speculative = ClaimRecord(
        claim_id="speculative-parent",
        statement="Speculative parent.",
        classification=ClaimClassification.SPECULATIVE,
        verdict=ClaimVerdict.NOT_ESTABLISHED,
        evidence_ids=("analysis-test",),
        depends_on=(),
        limitations=("This record is intentionally unsupported.",),
    )
    derived = ClaimRecord(
        claim_id="derived-child",
        statement="Derived child.",
        classification=ClaimClassification.DERIVED,
        verdict=ClaimVerdict.REPRODUCIBLE,
        evidence_ids=("analysis-test",),
        depends_on=("speculative-parent",),
        limitations=("This record is intentionally invalid.",),
    )

    with pytest.raises(
        ClaimLedgerError,
        match="outside its permitted evidence classes",
    ):
        ClaimLedger(
            schema_version=1,
            ledger_id="invalid-dependency-ledger",
            title="Dependency strength test",
            scope_note="Validation test.",
            evidence=(evidence,),
            claims=(speculative, derived),
        )


def test_loader_rejects_invalid_json_and_unknown_classification(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")

    with pytest.raises(
        ClaimLedgerError,
        match="invalid claim ledger JSON",
    ):
        load_claim_ledger(invalid_json)

    payload = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    payload["claims"][0]["classification"] = "proven"

    invalid_classification = tmp_path / "invalid-classification.json"
    invalid_classification.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ClaimLedgerError,
        match="unsupported claim classification",
    ):
        load_claim_ledger(invalid_classification)


def test_verified_loader_detects_ledger_tampering(tmp_path: Path) -> None:
    manifest_target = tmp_path / CLAIM_LEDGER_MANIFEST_PATH
    ledger_target = tmp_path / CLAIM_LEDGER_REFERENCE_PATH
    manifest_target.parent.mkdir(parents=True)
    ledger_target.parent.mkdir(parents=True)

    manifest_target.write_bytes((_REPOSITORY_ROOT / CLAIM_LEDGER_MANIFEST_PATH).read_bytes())
    ledger_target.write_text(
        _LEDGER_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProvenanceError, match="hash-mismatch"):
        load_verified_claim_ledger(tmp_path)


def test_verified_loader_rejects_manifest_record_count_drift(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / CLAIM_LEDGER_MANIFEST_PATH
    ledger_target = tmp_path / CLAIM_LEDGER_REFERENCE_PATH
    manifest_target.parent.mkdir(parents=True)
    ledger_target.parent.mkdir(parents=True)

    manifest_payload = json.loads(
        (_REPOSITORY_ROOT / CLAIM_LEDGER_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest_payload["artifacts"][0]["record_count"] = 11
    manifest_target.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    ledger_target.write_bytes(_LEDGER_PATH.read_bytes())

    with pytest.raises(
        ClaimLedgerError,
        match="record_count is 11",
    ):
        load_verified_claim_ledger(tmp_path)


def test_verified_loader_rejects_undeclared_external_source(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / CLAIM_LEDGER_MANIFEST_PATH
    ledger_target = tmp_path / CLAIM_LEDGER_REFERENCE_PATH
    manifest_target.parent.mkdir(parents=True)
    ledger_target.parent.mkdir(parents=True)

    manifest_payload = json.loads(
        (_REPOSITORY_ROOT / CLAIM_LEDGER_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest_payload["sources"] = tuple(
        source
        for source in manifest_payload["sources"]
        if source["source_id"] != "ehman-wow-30th-report"
    )
    manifest_target.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    ledger_target.write_bytes(_LEDGER_PATH.read_bytes())

    with pytest.raises(
        ProvenanceError,
        match="source URLs absent from the manifest",
    ):
        load_verified_claim_ledger(tmp_path)
