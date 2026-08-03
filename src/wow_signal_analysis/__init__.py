"""Public package identity, analysis, dataset, measurements, and provenance API."""

from typing import Final

from wow_signal_analysis.dataset import (
    CANONICAL_DATASET_PATH,
    CANONICAL_MANIFEST_PATH,
    EXPECTED_COLUMNS,
    DatasetError,
    ObservationDataset,
    load_observation_csv,
    load_verified_wow_dataset,
    require_canonical_wow_dataset,
)
from wow_signal_analysis.measurements import (
    WOW_INTEGRATION_SECONDS,
    WOW_PRINTER_SEQUENCE,
    WOW_SAMPLE_CADENCE_SECONDS,
    IntensityCode,
    SignalSample,
    canonical_wow_samples,
    decode_printer_sequence,
    decode_printer_symbol,
)
from wow_signal_analysis.profile import (
    AdjacentChange,
    MirrorComparison,
    ProfileError,
    SequenceProfile,
    Trend,
    analyze_midpoint_values,
    analyze_samples,
)
from wow_signal_analysis.provenance import (
    ArtifactStatus,
    ArtifactVerification,
    DatasetArtifact,
    ProvenanceError,
    SourceManifest,
    SourceReference,
    load_source_manifest,
    require_verified_artifacts,
    sha256_file,
    verify_manifest_artifacts,
)
from wow_signal_analysis.thresholds import (
    BitPolarity,
    ThresholdCase,
    ThresholdError,
    apply_threshold,
    enumerate_threshold_cases,
    find_threshold_case,
)

DISPLAY_NAME: Final = "Reproducible Analysis of the Wow! Signal"
PROJECT_SLUG: Final = "wow-signal-reproducible-analysis"
__version__: Final = "0.1.0"

__all__ = [
    "CANONICAL_DATASET_PATH",
    "CANONICAL_MANIFEST_PATH",
    "DISPLAY_NAME",
    "EXPECTED_COLUMNS",
    "PROJECT_SLUG",
    "WOW_INTEGRATION_SECONDS",
    "WOW_PRINTER_SEQUENCE",
    "WOW_SAMPLE_CADENCE_SECONDS",
    "AdjacentChange",
    "ArtifactStatus",
    "ArtifactVerification",
    "BitPolarity",
    "DatasetArtifact",
    "DatasetError",
    "IntensityCode",
    "MirrorComparison",
    "ObservationDataset",
    "ProfileError",
    "ProvenanceError",
    "SequenceProfile",
    "SignalSample",
    "SourceManifest",
    "SourceReference",
    "ThresholdCase",
    "ThresholdError",
    "Trend",
    "__version__",
    "analyze_midpoint_values",
    "analyze_samples",
    "apply_threshold",
    "canonical_wow_samples",
    "decode_printer_sequence",
    "decode_printer_symbol",
    "enumerate_threshold_cases",
    "find_threshold_case",
    "load_observation_csv",
    "load_source_manifest",
    "load_verified_wow_dataset",
    "require_canonical_wow_dataset",
    "require_verified_artifacts",
    "sha256_file",
    "verify_manifest_artifacts",
]
