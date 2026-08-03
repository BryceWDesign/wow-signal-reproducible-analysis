"""Strict loading and verification of normalized Wow! observation data."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Final

from wow_signal_analysis.measurements import (
    SignalSample,
    canonical_wow_samples,
    decode_printer_symbol,
)
from wow_signal_analysis.provenance import (
    DatasetArtifact,
    ProvenanceError,
    SourceManifest,
    load_source_manifest,
    require_verified_artifacts,
)

CANONICAL_DATASET_PATH: Final = PurePosixPath("data/raw/wow_6equj5.csv")
CANONICAL_MANIFEST_PATH: Final = PurePosixPath("data/provenance/source_manifest.json")

EXPECTED_COLUMNS: Final = (
    "sample_index",
    "elapsed_seconds",
    "printer_symbol",
    "snr_lower_inclusive",
    "snr_upper_exclusive",
    "snr_midpoint",
)

_ZERO: Final = Decimal("0")


class DatasetError(ValueError):
    """Raised when normalized observation data are malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ObservationDataset:
    """An immutable, schema-validated sequence of telescope samples."""

    source_path: Path
    samples: tuple[SignalSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise DatasetError("observation dataset must contain at least one sample")

        expected_indices = tuple(range(len(self.samples)))
        actual_indices = tuple(sample.sample_index for sample in self.samples)
        if actual_indices != expected_indices:
            raise DatasetError(
                "sample_index values must be contiguous and zero-based: "
                f"expected {expected_indices}, received {actual_indices}"
            )

        elapsed_values = tuple(sample.elapsed_seconds for sample in self.samples)
        if elapsed_values[0] != _ZERO:
            raise DatasetError("the first elapsed_seconds value must be zero")
        if any(current <= previous for previous, current in pairwise(elapsed_values)):
            raise DatasetError("elapsed_seconds values must be strictly increasing")

    @property
    def printer_sequence(self) -> str:
        """Return the ordered printer symbols represented by the dataset."""

        return "".join(sample.intensity.symbol for sample in self.samples)

    @property
    def midpoint_snr(self) -> tuple[Decimal, ...]:
        """Return immutable midpoint signal-to-noise estimates."""

        return tuple(sample.intensity.midpoint_snr for sample in self.samples)


def load_observation_csv(path: Path) -> ObservationDataset:
    """Load one normalized observation CSV with strict schema and value checks."""

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_header(reader.fieldnames)
            samples = tuple(
                _parse_row(row, line_number=line_number)
                for line_number, row in enumerate(reader, start=2)
            )
    except OSError as error:
        raise DatasetError(f"unable to read observation dataset: {path}") from error

    return ObservationDataset(source_path=path, samples=samples)


def require_canonical_wow_dataset(dataset: ObservationDataset) -> None:
    """Require exact agreement with the canonical typed 6EQUJ5 measurements."""

    expected = canonical_wow_samples()
    if dataset.samples != expected:
        raise DatasetError(
            "observation dataset does not exactly match the canonical 6EQUJ5 measurements"
        )


def load_verified_wow_dataset(
    repository_root: Path,
    *,
    manifest_path: PurePosixPath = CANONICAL_MANIFEST_PATH,
    dataset_path: PurePosixPath = CANONICAL_DATASET_PATH,
) -> ObservationDataset:
    """Verify provenance, load the manifest-bound CSV, and require canonical values."""

    root = repository_root.resolve()
    manifest = load_source_manifest(root / manifest_path)
    require_verified_artifacts(manifest, root)
    artifact = _find_artifact(manifest, dataset_path)

    dataset = load_observation_csv(root / dataset_path)
    if len(dataset.samples) != artifact.record_count:
        raise DatasetError(
            f"manifest record_count is {artifact.record_count}, "
            f"but the dataset contains {len(dataset.samples)} records"
        )
    require_canonical_wow_dataset(dataset)
    return dataset


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise DatasetError("observation dataset is missing a CSV header")
    actual = tuple(fieldnames)
    if actual != EXPECTED_COLUMNS:
        raise DatasetError(
            f"unexpected CSV columns: expected {EXPECTED_COLUMNS}, received {actual}"
        )


def _parse_row(
    row: Mapping[str | None, str | list[str] | None],
    *,
    line_number: int,
) -> SignalSample:
    if None in row:
        raise DatasetError(f"line {line_number}: row contains more fields than the header")

    sample_index = _parse_int(
        _required_cell(row, "sample_index", line_number),
        "sample_index",
        line_number,
    )
    elapsed_seconds = _parse_decimal(
        _required_cell(row, "elapsed_seconds", line_number),
        "elapsed_seconds",
        line_number,
    )
    symbol = _required_cell(row, "printer_symbol", line_number)

    try:
        intensity = decode_printer_symbol(symbol)
    except ValueError as error:
        raise DatasetError(f"line {line_number}: invalid printer_symbol {symbol!r}") from error

    documented_values = {
        "snr_lower_inclusive": intensity.lower_snr,
        "snr_upper_exclusive": intensity.upper_snr,
        "snr_midpoint": intensity.midpoint_snr,
    }
    for field_name, expected in documented_values.items():
        actual = _parse_decimal(
            _required_cell(row, field_name, line_number),
            field_name,
            line_number,
        )
        if actual != expected:
            raise DatasetError(
                f"line {line_number}: {field_name}={actual} conflicts with "
                f"printer_symbol {symbol!r}, which requires {expected}"
            )

    try:
        return SignalSample(
            sample_index=sample_index,
            elapsed_seconds=elapsed_seconds,
            intensity=intensity,
        )
    except ValueError as error:
        raise DatasetError(f"line {line_number}: {error}") from error


def _required_cell(
    row: Mapping[str | None, str | list[str] | None],
    field_name: str,
    line_number: int,
) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"line {line_number}: {field_name} must be non-empty")
    return value


def _parse_int(value: str, field_name: str, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise DatasetError(f"line {line_number}: {field_name} must be an integer") from error


def _parse_decimal(value: str, field_name: str, line_number: int) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise DatasetError(f"line {line_number}: {field_name} must be a decimal number") from error
    if not parsed.is_finite():
        raise DatasetError(f"line {line_number}: {field_name} must be finite")
    return parsed


def _find_artifact(
    manifest: SourceManifest,
    dataset_path: PurePosixPath,
) -> DatasetArtifact:
    matches = tuple(
        artifact for artifact in manifest.artifacts if artifact.path == str(dataset_path)
    )
    if not matches:
        raise ProvenanceError(f"manifest does not declare dataset artifact: {dataset_path}")
    if len(matches) != 1:
        raise ProvenanceError(f"manifest declares dataset artifact more than once: {dataset_path}")
    return matches[0]
