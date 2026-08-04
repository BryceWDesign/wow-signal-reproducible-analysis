from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from wow_signal_analysis.frequency_context import (
    FREQUENCY_MANIFEST_PATH,
    FREQUENCY_REFERENCE_PATH,
    FrequencyContext,
    FrequencyContextError,
    FrequencyEstimate,
    FrequencyEstimateStatus,
    FrequencyOffset,
    SpectralLineReference,
    load_frequency_context,
    load_verified_frequency_context,
)
from wow_signal_analysis.provenance import ProvenanceError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_PATH = _REPOSITORY_ROOT / FREQUENCY_REFERENCE_PATH


def test_verified_context_preserves_sourced_frequency_records() -> None:
    context = load_verified_frequency_context(_REPOSITORY_ROOT)

    assert context.schema_version == 1
    assert context.record_count == 3
    assert context.rest_line.line_id == "neutral-hydrogen-21cm"
    assert context.rest_line.species == "neutral atomic hydrogen (H I)"
    assert context.rest_line.rest_frequency_mhz == Decimal("1420.405752")
    assert tuple(estimate.estimate_id for estimate in context.estimates) == (
        "ehman-historical-2008",
        "mendez-et-al-2025-v1",
    )


def test_historical_frequency_offset_is_computed_exactly() -> None:
    context = load_frequency_context(_REFERENCE_PATH)
    offset = context.offset_for("ehman-historical-2008")

    assert offset.delta_mhz == Decimal("0.049848")
    assert offset.absolute_offset_khz == Decimal("49.848000")
    assert offset.uncertainty_khz == Decimal("5.000")
    assert offset.relative_offset_ppm == (
        Decimal("0.049848") / Decimal("1420.405752") * Decimal("1000000")
    )
    assert not offset.uncertainty_interval_contains_rest


def test_recalibrated_frequency_offset_is_computed_exactly() -> None:
    context = load_frequency_context(_REFERENCE_PATH)
    offset = context.offset_for("mendez-et-al-2025-v1")

    assert offset.delta_mhz == Decimal("0.320248")
    assert offset.absolute_offset_khz == Decimal("320.248000")
    assert offset.uncertainty_khz == Decimal("5.000")
    assert not offset.uncertainty_interval_contains_rest


def test_context_keeps_historical_analysis_and_preprint_status_distinct() -> None:
    context = load_frequency_context(_REFERENCE_PATH)

    assert context.estimates[0].status is FrequencyEstimateStatus.HISTORICAL_ANALYSIS
    assert context.estimates[1].status is FrequencyEstimateStatus.RESEARCH_PREPRINT


def test_offset_window_is_explicit_and_not_embedded_as_a_near_line_claim() -> None:
    context = load_frequency_context(_REFERENCE_PATH)

    assert context.estimates_within_offset(Decimal("49.847")) == ()
    assert tuple(
        estimate.estimate_id for estimate in context.estimates_within_offset(Decimal("49.848"))
    ) == ("ehman-historical-2008",)
    assert tuple(
        estimate.estimate_id for estimate in context.estimates_within_offset(Decimal("320.248"))
    ) == (
        "ehman-historical-2008",
        "mendez-et-al-2025-v1",
    )
    assert context.maximum_absolute_offset_khz == Decimal("320.248000")


def test_unknown_estimate_and_invalid_offset_window_fail_closed() -> None:
    context = load_frequency_context(_REFERENCE_PATH)

    with pytest.raises(FrequencyContextError, match="found 0"):
        context.estimate_by_id("not-declared")

    with pytest.raises(
        FrequencyContextError,
        match="non-negative and finite",
    ):
        context.estimates_within_offset(Decimal("-1"))

    with pytest.raises(
        FrequencyContextError,
        match="non-negative and finite",
    ):
        context.estimates_within_offset(Decimal("NaN"))


def test_loader_rejects_invalid_json_and_unknown_status(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")

    with pytest.raises(
        FrequencyContextError,
        match="invalid frequency context JSON",
    ):
        load_frequency_context(invalid_json)

    payload = json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))
    payload["wow_frequency_estimates"][0]["status"] = "peer-reviewed"

    unknown_status = tmp_path / "unknown-status.json"
    unknown_status.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        FrequencyContextError,
        match="unsupported frequency estimate status",
    ):
        load_frequency_context(unknown_status)


def test_context_rejects_duplicate_estimate_ids() -> None:
    rest_line = SpectralLineReference(
        line_id="neutral-hydrogen-21cm",
        species="neutral atomic hydrogen (H I)",
        transition="ground-state hyperfine transition",
        rest_frequency_mhz=Decimal("1420.405752"),
        source_id="source",
    )
    estimate = FrequencyEstimate(
        estimate_id="duplicate",
        frequency_mhz=Decimal("1420.4556"),
        uncertainty_mhz=Decimal("0.005"),
        status=FrequencyEstimateStatus.HISTORICAL_ANALYSIS,
        source_id="source",
        notes="test record",
    )

    with pytest.raises(
        FrequencyContextError,
        match="IDs must be unique",
    ):
        FrequencyContext(
            schema_version=1,
            rest_line=rest_line,
            estimates=(estimate, estimate),
        )


def test_frequency_offset_rejects_inconsistent_derived_values() -> None:
    with pytest.raises(
        FrequencyContextError,
        match="does not match",
    ):
        FrequencyOffset(
            estimate_id="bad",
            delta_mhz=Decimal("0.1"),
            absolute_offset_khz=Decimal("99"),
            relative_offset_ppm=Decimal("1"),
            uncertainty_khz=Decimal("5"),
            uncertainty_interval_contains_rest=False,
        )


def test_verified_loader_detects_reference_tampering(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / FREQUENCY_MANIFEST_PATH
    reference_target = tmp_path / FREQUENCY_REFERENCE_PATH

    manifest_target.parent.mkdir(parents=True)
    reference_target.parent.mkdir(parents=True)

    manifest_target.write_bytes((_REPOSITORY_ROOT / FREQUENCY_MANIFEST_PATH).read_bytes())
    reference_target.write_text(
        _REFERENCE_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ProvenanceError,
        match="hash-mismatch",
    ):
        load_verified_frequency_context(tmp_path)


def test_verified_loader_rejects_source_ids_absent_from_manifest(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / FREQUENCY_MANIFEST_PATH
    reference_target = tmp_path / FREQUENCY_REFERENCE_PATH

    manifest_target.parent.mkdir(parents=True)
    reference_target.parent.mkdir(parents=True)

    manifest_payload = json.loads(
        (_REPOSITORY_ROOT / FREQUENCY_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest_payload["sources"] = manifest_payload["sources"][:-1]

    manifest_target.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    reference_target.write_bytes(_REFERENCE_PATH.read_bytes())

    with pytest.raises(
        ProvenanceError,
        match="absent from the manifest",
    ):
        load_verified_frequency_context(tmp_path)


def test_verified_loader_rejects_manifest_record_count_drift(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / FREQUENCY_MANIFEST_PATH
    reference_target = tmp_path / FREQUENCY_REFERENCE_PATH

    manifest_target.parent.mkdir(parents=True)
    reference_target.parent.mkdir(parents=True)

    manifest_payload = json.loads(
        (_REPOSITORY_ROOT / FREQUENCY_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    manifest_payload["artifacts"][0]["record_count"] = 2

    manifest_target.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    reference_target.write_bytes(_REFERENCE_PATH.read_bytes())

    with pytest.raises(
        FrequencyContextError,
        match="record_count is 2",
    ):
        load_verified_frequency_context(tmp_path)
