"""Verified frequency context for the Wow! signal and the H I 21 cm line."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast

from wow_signal_analysis.provenance import (
    ProvenanceError,
    load_source_manifest,
    require_verified_artifacts,
)

FREQUENCY_REFERENCE_PATH: Final = PurePosixPath("data/reference/frequency_context.json")
FREQUENCY_MANIFEST_PATH: Final = PurePosixPath("data/provenance/frequency_source_manifest.json")

_KILOHERTZ_PER_MEGAHERTZ: Final = Decimal("1000")
_PARTS_PER_MILLION: Final = Decimal("1000000")
_ZERO: Final = Decimal("0")


class FrequencyContextError(ValueError):
    """Raised when frequency reference data are malformed or inconsistent."""


class FrequencyEstimateStatus(StrEnum):
    """Declared publication status of one Wow! signal frequency estimate."""

    HISTORICAL_ANALYSIS = "historical-analysis"
    RESEARCH_PREPRINT = "research-preprint"


@dataclass(frozen=True, slots=True)
class SpectralLineReference:
    """One sourced rest-frame spectral-line frequency."""

    line_id: str
    species: str
    transition: str
    rest_frequency_mhz: Decimal
    source_id: str

    def __post_init__(self) -> None:
        for field_name in ("line_id", "species", "transition", "source_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise FrequencyContextError(f"{field_name} must be non-empty")
        _require_positive_finite(
            self.rest_frequency_mhz,
            "rest_frequency_mhz",
        )


@dataclass(frozen=True, slots=True)
class FrequencyEstimate:
    """One explicitly sourced estimate of the observed Wow! signal frequency."""

    estimate_id: str
    frequency_mhz: Decimal
    uncertainty_mhz: Decimal
    status: FrequencyEstimateStatus
    source_id: str
    notes: str

    def __post_init__(self) -> None:
        for field_name in ("estimate_id", "source_id", "notes"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise FrequencyContextError(f"{field_name} must be non-empty")
        _require_positive_finite(self.frequency_mhz, "frequency_mhz")
        _require_nonnegative_finite(self.uncertainty_mhz, "uncertainty_mhz")


@dataclass(frozen=True, slots=True)
class FrequencyOffset:
    """Exact decimal offset between an estimate and the H I rest frequency."""

    estimate_id: str
    delta_mhz: Decimal
    absolute_offset_khz: Decimal
    relative_offset_ppm: Decimal
    uncertainty_khz: Decimal
    uncertainty_interval_contains_rest: bool

    def __post_init__(self) -> None:
        if not self.estimate_id.strip():
            raise FrequencyContextError("estimate_id must be non-empty")
        if not self.delta_mhz.is_finite():
            raise FrequencyContextError("delta_mhz must be finite")
        _require_nonnegative_finite(
            self.absolute_offset_khz,
            "absolute_offset_khz",
        )
        if self.absolute_offset_khz != abs(self.delta_mhz) * _KILOHERTZ_PER_MEGAHERTZ:
            raise FrequencyContextError("absolute_offset_khz does not match delta_mhz")
        if not self.relative_offset_ppm.is_finite():
            raise FrequencyContextError("relative_offset_ppm must be finite")
        _require_nonnegative_finite(self.uncertainty_khz, "uncertainty_khz")


@dataclass(frozen=True, slots=True)
class FrequencyContext:
    """Immutable H I rest line and competing Wow! signal frequency estimates."""

    schema_version: int
    rest_line: SpectralLineReference
    estimates: tuple[FrequencyEstimate, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise FrequencyContextError("unsupported frequency context schema_version")
        if not self.estimates:
            raise FrequencyContextError("frequency context must contain at least one estimate")

        estimate_ids = tuple(estimate.estimate_id for estimate in self.estimates)
        if len(set(estimate_ids)) != len(estimate_ids):
            raise FrequencyContextError("frequency estimate IDs must be unique")

    @property
    def record_count(self) -> int:
        """Return one rest-line record plus all Wow! frequency estimates."""

        return 1 + len(self.estimates)

    @property
    def offsets(self) -> tuple[FrequencyOffset, ...]:
        """Return exact offsets for every estimate in source order."""

        return tuple(self.offset_for(estimate.estimate_id) for estimate in self.estimates)

    @property
    def maximum_absolute_offset_khz(self) -> Decimal:
        """Return the largest absolute H I offset among declared estimates."""

        return max(offset.absolute_offset_khz for offset in self.offsets)

    def estimate_by_id(self, estimate_id: str) -> FrequencyEstimate:
        """Return the unique estimate matching an explicit identifier."""

        matches = tuple(
            estimate for estimate in self.estimates if estimate.estimate_id == estimate_id
        )
        if len(matches) != 1:
            raise FrequencyContextError(
                f"expected one frequency estimate for {estimate_id!r}, found {len(matches)}"
            )
        return matches[0]

    def offset_for(self, estimate_id: str) -> FrequencyOffset:
        """Calculate a sourced estimate's offset from the H I rest frequency."""

        estimate = self.estimate_by_id(estimate_id)
        rest_frequency = self.rest_line.rest_frequency_mhz
        delta_mhz = estimate.frequency_mhz - rest_frequency
        lower = estimate.frequency_mhz - estimate.uncertainty_mhz
        upper = estimate.frequency_mhz + estimate.uncertainty_mhz

        return FrequencyOffset(
            estimate_id=estimate.estimate_id,
            delta_mhz=delta_mhz,
            absolute_offset_khz=(abs(delta_mhz) * _KILOHERTZ_PER_MEGAHERTZ),
            relative_offset_ppm=((delta_mhz / rest_frequency) * _PARTS_PER_MILLION),
            uncertainty_khz=(estimate.uncertainty_mhz * _KILOHERTZ_PER_MEGAHERTZ),
            uncertainty_interval_contains_rest=(lower <= rest_frequency <= upper),
        )

    def estimates_within_offset(
        self,
        maximum_absolute_offset_khz: Decimal,
    ) -> tuple[FrequencyEstimate, ...]:
        """Return estimates inside an analyst-declared absolute H I window."""

        _require_nonnegative_finite(
            maximum_absolute_offset_khz,
            "maximum_absolute_offset_khz",
        )
        accepted_ids = {
            offset.estimate_id
            for offset in self.offsets
            if offset.absolute_offset_khz <= maximum_absolute_offset_khz
        }
        return tuple(
            estimate for estimate in self.estimates if estimate.estimate_id in accepted_ids
        )


def load_frequency_context(path: Path) -> FrequencyContext:
    """Load and validate a normalized frequency-context JSON document."""

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FrequencyContextError(f"unable to read frequency context: {path}") from error
    except json.JSONDecodeError as error:
        raise FrequencyContextError(f"invalid frequency context JSON: {path}") from error

    root = _require_mapping(payload, "frequency context")
    rest_line = _rest_line_from_mapping(_require_mapping(root.get("rest_line"), "rest_line"))
    estimates = tuple(
        _estimate_from_mapping(_require_mapping(item, "frequency estimate"))
        for item in _required_list(root, "wow_frequency_estimates")
    )

    return FrequencyContext(
        schema_version=_required_int(root, "schema_version"),
        rest_line=rest_line,
        estimates=estimates,
    )


def load_verified_frequency_context(
    repository_root: Path,
    *,
    manifest_path: PurePosixPath = FREQUENCY_MANIFEST_PATH,
    reference_path: PurePosixPath = FREQUENCY_REFERENCE_PATH,
) -> FrequencyContext:
    """Verify provenance before loading the canonical frequency reference."""

    root = repository_root.resolve()
    manifest = load_source_manifest(root / manifest_path)
    require_verified_artifacts(manifest, root)

    matches = tuple(
        artifact for artifact in manifest.artifacts if artifact.path == str(reference_path)
    )
    if len(matches) != 1:
        raise ProvenanceError(
            f"expected one manifest artifact for {reference_path}, found {len(matches)}"
        )

    context = load_frequency_context(root / reference_path)
    if context.record_count != matches[0].record_count:
        raise FrequencyContextError(
            f"manifest record_count is {matches[0].record_count}, "
            f"but the frequency context contains {context.record_count} records"
        )

    declared_source_ids = {source.source_id for source in manifest.sources}
    referenced_source_ids = {
        context.rest_line.source_id,
        *(estimate.source_id for estimate in context.estimates),
    }
    undeclared_source_ids = tuple(sorted(referenced_source_ids - declared_source_ids))
    if undeclared_source_ids:
        raise ProvenanceError(
            "frequency context references source IDs absent from the manifest: "
            f"{undeclared_source_ids}"
        )
    return context


def _rest_line_from_mapping(
    value: Mapping[str, object],
) -> SpectralLineReference:
    return SpectralLineReference(
        line_id=_required_text(value, "line_id"),
        species=_required_text(value, "species"),
        transition=_required_text(value, "transition"),
        rest_frequency_mhz=_required_decimal(
            value,
            "rest_frequency_mhz",
        ),
        source_id=_required_text(value, "source_id"),
    )


def _estimate_from_mapping(
    value: Mapping[str, object],
) -> FrequencyEstimate:
    status_text = _required_text(value, "status")
    try:
        status = FrequencyEstimateStatus(status_text)
    except ValueError as error:
        raise FrequencyContextError(
            f"unsupported frequency estimate status: {status_text!r}"
        ) from error

    return FrequencyEstimate(
        estimate_id=_required_text(value, "estimate_id"),
        frequency_mhz=_required_decimal(value, "frequency_mhz"),
        uncertainty_mhz=_required_decimal(value, "uncertainty_mhz"),
        status=status,
        source_id=_required_text(value, "source_id"),
        notes=_required_text(value, "notes"),
    )


def _require_mapping(
    value: object,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FrequencyContextError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise FrequencyContextError(f"{field_name} keys must be strings")
    return cast(dict[str, object], value)


def _required_list(
    value: Mapping[str, object],
    field_name: str,
) -> list[object]:
    item = value.get(field_name)
    if not isinstance(item, list):
        raise FrequencyContextError(f"{field_name} must be a JSON array")
    return cast(list[object], item)


def _required_text(
    value: Mapping[str, object],
    field_name: str,
) -> str:
    item = value.get(field_name)
    if not isinstance(item, str) or not item.strip():
        raise FrequencyContextError(f"{field_name} must be a non-empty string")
    return item


def _required_int(
    value: Mapping[str, object],
    field_name: str,
) -> int:
    item = value.get(field_name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise FrequencyContextError(f"{field_name} must be an integer")
    return item


def _required_decimal(
    value: Mapping[str, object],
    field_name: str,
) -> Decimal:
    text = _required_text(value, field_name)
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise FrequencyContextError(f"{field_name} must contain a decimal string") from error
    if not parsed.is_finite():
        raise FrequencyContextError(f"{field_name} must be finite")
    return parsed


def _require_positive_finite(
    value: Decimal,
    field_name: str,
) -> None:
    if not value.is_finite() or value <= _ZERO:
        raise FrequencyContextError(f"{field_name} must be positive and finite")


def _require_nonnegative_finite(
    value: Decimal,
    field_name: str,
) -> None:
    if not value.is_finite() or value < _ZERO:
        raise FrequencyContextError(f"{field_name} must be non-negative and finite")
