from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from wow_signal_analysis.analysis_snapshot import (
    ANALYSIS_SNAPSHOT_ID,
    ANALYSIS_SNAPSHOT_SCHEMA_VERSION,
    AnalysisSnapshot,
    AnalysisSnapshotError,
    SnapshotConfig,
    build_analysis_snapshot,
)
from wow_signal_analysis.beam_model import GaussianSearchConfig
from wow_signal_analysis.claim_ledger import ClaimVerdict
from wow_signal_analysis.hypothesis_matrix import HypothesisStatus
from wow_signal_analysis.model_comparison import CandidateModel
from wow_signal_analysis.quantization import FitMetric

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FAST_CONFIG = SnapshotConfig(
    gaussian_search=GaussianSearchConfig(
        grid_points=21,
        refinement_rounds=3,
    )
)


@pytest.fixture(scope="module")
def snapshot() -> AnalysisSnapshot:
    return build_analysis_snapshot(
        _REPOSITORY_ROOT,
        config=_FAST_CONFIG,
    )


def test_snapshot_binds_every_canonical_analysis_component(
    snapshot: AnalysisSnapshot,
) -> None:
    assert snapshot.schema_version == ANALYSIS_SNAPSHOT_SCHEMA_VERSION
    assert snapshot.analysis_id == ANALYSIS_SNAPSHOT_ID
    assert snapshot.repository.verified_component_count == 5
    assert snapshot.dataset.printer_sequence == "6EQUJ5"
    assert snapshot.profile.trend_signature == "+++--"
    assert snapshot.profile.is_strict_single_peak
    assert snapshot.quantization.evaluated_corner_count == 64
    assert snapshot.claim_ledger.ledger_id == snapshot.repository.claim_ledger_id
    assert (
        snapshot.hypothesis_matrix.matrix_id
        == snapshot.repository.hypothesis_matrix_id
    )


def test_snapshot_preserves_model_ranking_and_quantization_ranges(
    snapshot: AnalysisSnapshot,
) -> None:
    assert (
        snapshot.model_comparison.best_model.model
        is CandidateModel.GAUSSIAN_TRANSIT
    )
    assert tuple(
        result.model
        for result in snapshot.model_comparison.ranked_by_prediction_error
    ) == (
        CandidateModel.GAUSSIAN_TRANSIT,
        CandidateModel.QUADRATIC,
        CandidateModel.CONSTANT,
        CandidateModel.AFFINE,
    )

    fwhm = snapshot.quantization.envelope(FitMetric.FWHM_SECONDS)
    assert fwhm.minimum < snapshot.gaussian_fit.fwhm_seconds < fwhm.maximum


def test_snapshot_preserves_symbolic_matches_and_exact_null_fractions(
    snapshot: AnalysisSnapshot,
) -> None:
    question_matches = snapshot.morse_correspondence.comparisons_for_glyph("?")
    comma_matches = snapshot.morse_correspondence.comparisons_for_glyph(",")
    question_null = snapshot.permutation_null.summary_for_glyph("?")
    comma_null = snapshot.permutation_null.summary_for_glyph(",")

    assert len(question_matches) == 2
    assert len(comma_matches) == 2
    assert {match.morse_pattern for match in question_matches} == {"..--.."}
    assert {match.morse_pattern for match in comma_matches} == {"--..--"}
    assert question_null.sequence_fraction == Fraction(2, 15)
    assert comma_null.sequence_fraction == Fraction(2, 15)
    assert question_null.comparison_fraction == Fraction(1, 105)
    assert comma_null.comparison_fraction == Fraction(1, 105)


def test_snapshot_retains_non_establishment_verdicts(
    snapshot: AnalysisSnapshot,
) -> None:
    assert tuple(
        claim.claim_id
        for claim in snapshot.claim_ledger.claims_by_verdict(
            ClaimVerdict.NOT_ESTABLISHED
        )
    ) == (
        "speculative-intentional-message",
        "speculative-extraterrestrial-technology",
    )
    assert len(
        snapshot.hypothesis_matrix.hypotheses_by_status(
            HypothesisStatus.NOT_ESTABLISHED
        )
    ) == 2


def test_snapshot_mapping_is_portable_and_json_serialization_is_stable(
    snapshot: AnalysisSnapshot,
) -> None:
    mapping = snapshot.to_mapping()
    encoded = snapshot.to_json()
    decoded = json.loads(encoded)

    assert encoded == snapshot.to_json()
    assert encoded.endswith("\n")
    assert decoded == mapping
    assert mapping["repository"]["repository_root"] == "."
    assert mapping["observation"]["printer_sequence"] == "6EQUJ5"
    assert (
        mapping["symbolic_correspondence"]["standard_id"]
        == "ITU-R M.1677-1"
    )
    assert mapping["claim_ledger"]["claim_count"] == 12
    assert mapping["hypothesis_matrix"]["hypothesis_count"] == 5


def test_snapshot_configuration_fails_closed() -> None:
    with pytest.raises(AnalysisSnapshotError, match="must not be empty"):
        SnapshotConfig(selected_morse_glyphs=())

    with pytest.raises(
        AnalysisSnapshotError,
        match="must not contain duplicates",
    ):
        SnapshotConfig(selected_morse_glyphs=("?", "?"))

    with pytest.raises(
        AnalysisSnapshotError,
        match="single printable glyphs",
    ):
        SnapshotConfig(selected_morse_glyphs=("AB",))

    with pytest.raises(AnalysisSnapshotError, match="must be an integer"):
        SnapshotConfig(max_unique_sequences=True)

    with pytest.raises(AnalysisSnapshotError, match="must be positive"):
        SnapshotConfig(max_unique_sequences=0)


def test_snapshot_rejects_identity_or_component_drift(
    snapshot: AnalysisSnapshot,
) -> None:
    with pytest.raises(AnalysisSnapshotError, match="analysis_id must be"):
        replace(snapshot, analysis_id="different-analysis")

    with pytest.raises(AnalysisSnapshotError, match="schema_version"):
        replace(snapshot, schema_version=2)

    with pytest.raises(AnalysisSnapshotError, match="glyphs do not match"):
        replace(
            snapshot,
            config=SnapshotConfig(
                gaussian_search=_FAST_CONFIG.gaussian_search,
                selected_morse_glyphs=("?",),
            ),
        )
