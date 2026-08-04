from __future__ import annotations

from pathlib import Path

import pytest

from wow_signal_analysis.repository_contract import (
    RepositoryContractError,
    VerifiedComponent,
    verify_repository_contract,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_repository_contract_verifies_all_canonical_components() -> None:
    report = verify_repository_contract(_REPOSITORY_ROOT)

    assert report.repository_root == _REPOSITORY_ROOT.resolve()
    assert report.printer_sequence == "6EQUJ5"
    assert report.morse_standard_id == "ITU-R M.1677-1"
    assert report.claim_ledger_id == "wow-signal-evidence-claims-v1"
    assert report.hypothesis_matrix_id == "wow-signal-hypothesis-matrix-v1"
    assert report.verified_component_count == 5


def test_repository_contract_preserves_component_order_and_counts() -> None:
    report = verify_repository_contract(_REPOSITORY_ROOT)

    assert tuple(component.component_id for component in report.components) == (
        "canonical-observation-dataset",
        "international-morse-registry",
        "frequency-context",
        "claim-ledger",
        "hypothesis-matrix",
    )

    assert tuple(component.record_count for component in report.components) == (
        6,
        51,
        3,
        12,
        5,
    )

    assert report.total_record_count == 77


def test_repository_contract_preserves_canonical_artifact_paths() -> None:
    report = verify_repository_contract(_REPOSITORY_ROOT)

    assert (
        str(report.component_by_id("canonical-observation-dataset").artifact_path)
        == "data/raw/wow_6equj5.csv"
    )

    assert (
        str(report.component_by_id("international-morse-registry").artifact_path)
        == "data/reference/itu_m1677_1_morse_symbols.json"
    )

    assert (
        str(report.component_by_id("frequency-context").artifact_path)
        == "data/reference/frequency_context.json"
    )

    assert (
        str(report.component_by_id("claim-ledger").artifact_path)
        == "data/reference/claim_ledger.json"
    )

    assert (
        str(report.component_by_id("hypothesis-matrix").artifact_path)
        == "data/reference/hypothesis_matrix.json"
    )


def test_unknown_component_lookup_fails_closed() -> None:
    report = verify_repository_contract(_REPOSITORY_ROOT)

    with pytest.raises(RepositoryContractError, match="found 0"):
        report.component_by_id("missing-component")


def test_contract_rejects_a_nonexistent_repository_root(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(
        RepositoryContractError,
        match="existing directory",
    ):
        verify_repository_contract(missing_root)


def test_verified_component_rejects_unsafe_paths_and_invalid_counts() -> None:
    with pytest.raises(
        RepositoryContractError,
        match="safe repository-relative",
    ):
        VerifiedComponent(
            component_id="unsafe-component",
            artifact_path=Path("../outside.json"),
            record_count=1,
        )

    with pytest.raises(
        RepositoryContractError,
        match="record_count must be positive",
    ):
        VerifiedComponent(
            component_id="empty-component",
            artifact_path=Path("data/reference/empty.json"),
            record_count=0,
        )
