from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from wow_signal_analysis.dataset import (
    DatasetError,
    ObservationDataset,
    load_observation_csv,
    load_verified_wow_dataset,
    require_canonical_wow_dataset,
)
from wow_signal_analysis.measurements import canonical_wow_samples, decode_printer_sequence
from wow_signal_analysis.provenance import ProvenanceError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPOSITORY_ROOT / "data/raw/wow_6equj5.csv"
_HEADER = (
    "sample_index,elapsed_seconds,printer_symbol,snr_lower_inclusive,"
    "snr_upper_exclusive,snr_midpoint\n"
)


def _write_csv(tmp_path: Path, body: str, *, header: str = _HEADER) -> Path:
    path = tmp_path / "observations.csv"
    path.write_text(header + body, encoding="utf-8", newline="")
    return path


def test_committed_dataset_loads_as_the_canonical_sequence() -> None:
    dataset = load_observation_csv(_DATASET_PATH)

    assert dataset.source_path == _DATASET_PATH
    assert dataset.samples == canonical_wow_samples()
    assert dataset.printer_sequence == "6EQUJ5"
    assert tuple(str(value) for value in dataset.midpoint_snr) == (
        "6.5",
        "14.5",
        "26.5",
        "30.5",
        "19.5",
        "5.5",
    )
    require_canonical_wow_dataset(dataset)


def test_manifest_bound_loader_verifies_and_loads_the_repository_dataset() -> None:
    dataset = load_verified_wow_dataset(_REPOSITORY_ROOT)

    assert dataset.samples == canonical_wow_samples()


def test_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="unable to read"):
        load_observation_csv(tmp_path / "missing.csv")


def test_loader_rejects_a_missing_or_unexpected_header(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing a CSV header"):
        load_observation_csv(empty_path)

    wrong_header_path = _write_csv(
        tmp_path,
        "0,0,6,6,7\n",
        header=(
            "sample_index,elapsed_seconds,printer_symbol,"
            "snr_lower_inclusive,snr_upper_exclusive\n"
        ),
    )
    with pytest.raises(DatasetError, match="unexpected CSV columns"):
        load_observation_csv(wrong_header_path)


def test_loader_rejects_empty_data_and_malformed_rows(tmp_path: Path) -> None:
    header_only = _write_csv(tmp_path, "")
    with pytest.raises(DatasetError, match="at least one sample"):
        load_observation_csv(header_only)

    extra_field = _write_csv(tmp_path, "0,0,6,6,7,6.5,unexpected\n")
    with pytest.raises(DatasetError, match="more fields"):
        load_observation_csv(extra_field)

    missing_cell = _write_csv(tmp_path, "0,0,6,6,7,\n")
    with pytest.raises(DatasetError, match="snr_midpoint must be non-empty"):
        load_observation_csv(missing_cell)


def test_loader_rejects_invalid_numeric_and_symbol_values(tmp_path: Path) -> None:
    invalid_index = _write_csv(tmp_path, "x,0,6,6,7,6.5\n")
    with pytest.raises(DatasetError, match="sample_index must be an integer"):
        load_observation_csv(invalid_index)

    invalid_elapsed = _write_csv(tmp_path, "0,not-a-number,6,6,7,6.5\n")
    with pytest.raises(DatasetError, match="elapsed_seconds must be a decimal"):
        load_observation_csv(invalid_elapsed)

    infinite_elapsed = _write_csv(tmp_path, "0,Infinity,6,6,7,6.5\n")
    with pytest.raises(DatasetError, match="elapsed_seconds must be finite"):
        load_observation_csv(infinite_elapsed)

    invalid_symbol = _write_csv(tmp_path, "0,0,!,6,7,6.5\n")
    with pytest.raises(DatasetError, match="invalid printer_symbol"):
        load_observation_csv(invalid_symbol)


def test_loader_rejects_values_that_conflict_with_the_printer_symbol(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "0,0,E,15,16,15.5\n")

    with pytest.raises(DatasetError, match="conflicts with printer_symbol"):
        load_observation_csv(path)


def test_dataset_requires_contiguous_indices_and_increasing_time(tmp_path: Path) -> None:
    noncontiguous = _write_csv(tmp_path, "0,0,6,6,7,6.5\n2,12,E,14,15,14.5\n")
    with pytest.raises(DatasetError, match="contiguous and zero-based"):
        load_observation_csv(noncontiguous)

    nonzero_start = _write_csv(tmp_path, "0,12,6,6,7,6.5\n")
    with pytest.raises(DatasetError, match="first elapsed_seconds"):
        load_observation_csv(nonzero_start)

    repeated_time = _write_csv(tmp_path, "0,0,6,6,7,6.5\n1,0,E,14,15,14.5\n")
    with pytest.raises(DatasetError, match="strictly increasing"):
        load_observation_csv(repeated_time)


def test_dataset_wraps_signal_sample_validation_errors(tmp_path: Path) -> None:
    negative_index = _write_csv(tmp_path, "-1,0,6,6,7,6.5\n")

    with pytest.raises(DatasetError, match="sample_index must be non-negative"):
        load_observation_csv(negative_index)


def test_canonical_requirement_rejects_a_different_valid_sequence(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "0,0,6,6,7,6.5\n1,12,E,14,15,14.5\n")
    dataset = load_observation_csv(path)

    with pytest.raises(DatasetError, match="does not exactly match"):
        require_canonical_wow_dataset(dataset)


def test_observation_dataset_rejects_invalid_direct_construction(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="at least one sample"):
        ObservationDataset(source_path=tmp_path / "empty.csv", samples=())

    samples = decode_printer_sequence("6E")
    invalid_order = (samples[1], samples[0])
    with pytest.raises(DatasetError, match="contiguous and zero-based"):
        ObservationDataset(source_path=tmp_path / "wrong.csv", samples=invalid_order)


def test_verified_loader_rejects_an_unlisted_dataset_path() -> None:
    with pytest.raises(ProvenanceError, match="does not declare dataset artifact"):
        load_verified_wow_dataset(
            _REPOSITORY_ROOT,
            dataset_path=PurePosixPath("data/raw/not-declared.csv"),
        )
