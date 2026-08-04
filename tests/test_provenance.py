from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from wow_signal_analysis.measurements import canonical_wow_samples
from wow_signal_analysis.provenance import (
    ArtifactStatus,
    ProvenanceError,
    load_source_manifest,
    require_verified_artifacts,
    sha256_file,
    verify_manifest_artifacts,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _REPOSITORY_ROOT / "data/provenance/source_manifest.json"
_DATASET_PATH = _REPOSITORY_ROOT / "data/raw/wow_6equj5.csv"


def test_source_manifest_identifies_real_sources_and_local_artifact() -> None:
    manifest = load_source_manifest(_MANIFEST_PATH)

    assert manifest.schema_version == 1
    assert manifest.dataset_id == "wow-6equj5-normalized-v1"
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].path == "data/raw/wow_6equj5.csv"
    assert manifest.artifacts[0].record_count == 6
    assert [source.source_id for source in manifest.sources] == [
        "ehman-6equj5-2008",
        "ehman-wow-30th-report",
    ]
    assert all(source.url.startswith("https://www.bigear.org/") for source in manifest.sources)


def test_manifest_digest_matches_the_committed_dataset() -> None:
    manifest = load_source_manifest(_MANIFEST_PATH)
    artifact = manifest.artifacts[0]

    assert sha256_file(_DATASET_PATH) == artifact.sha256
    results = verify_manifest_artifacts(manifest, _REPOSITORY_ROOT)
    assert len(results) == 1
    assert results[0].status is ArtifactStatus.VERIFIED
    assert results[0].is_verified
    assert require_verified_artifacts(manifest, _REPOSITORY_ROOT) == ("data/raw/wow_6equj5.csv",)


def test_normalized_csv_exactly_matches_the_typed_canonical_samples() -> None:
    with _DATASET_PATH.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    samples = canonical_wow_samples()
    assert len(rows) == len(samples)
    for row, sample in zip(rows, samples, strict=True):
        assert int(row["sample_index"]) == sample.sample_index
        assert Decimal(row["elapsed_seconds"]) == sample.elapsed_seconds
        assert row["printer_symbol"] == sample.intensity.symbol
        assert Decimal(row["snr_lower_inclusive"]) == sample.intensity.lower_snr
        assert Decimal(row["snr_upper_exclusive"]) == sample.intensity.upper_snr
        assert Decimal(row["snr_midpoint"]) == sample.intensity.midpoint_snr


def test_verification_reports_a_hash_mismatch_without_hiding_it(tmp_path: Path) -> None:
    manifest = load_source_manifest(_MANIFEST_PATH)
    altered_path = tmp_path / "data/raw/wow_6equj5.csv"
    altered_path.parent.mkdir(parents=True)
    original = _DATASET_PATH.read_text(encoding="utf-8")
    altered_path.write_text(original + "altered\n", encoding="utf-8")

    results = verify_manifest_artifacts(manifest, tmp_path)

    assert results[0].status is ArtifactStatus.HASH_MISMATCH
    assert results[0].actual_sha256 is not None
    assert not results[0].is_verified
    with pytest.raises(ProvenanceError, match="hash-mismatch"):
        require_verified_artifacts(manifest, tmp_path)


def test_verification_reports_a_missing_artifact(tmp_path: Path) -> None:
    manifest = load_source_manifest(_MANIFEST_PATH)

    results = verify_manifest_artifacts(manifest, tmp_path)

    assert results[0].status is ArtifactStatus.MISSING
    assert results[0].actual_sha256 is None
