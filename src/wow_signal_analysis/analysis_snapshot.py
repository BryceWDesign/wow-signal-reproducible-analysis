"""Deterministic cumulative snapshot of the repository's canonical analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Final

from wow_signal_analysis.beam_model import (
    GaussianSearchConfig,
    GaussianTransitFit,
    fit_gaussian_transit,
)
from wow_signal_analysis.claim_ledger import (
    ClaimClassification,
    ClaimLedger,
    ClaimVerdict,
    load_verified_claim_ledger,
)
from wow_signal_analysis.dataset import ObservationDataset, load_verified_wow_dataset
from wow_signal_analysis.frequency_context import (
    FrequencyContext,
    load_verified_frequency_context,
)
from wow_signal_analysis.hypothesis_matrix import (
    BoundHypothesisMatrix,
    HypothesisStatus,
    load_verified_hypothesis_matrix,
)
from wow_signal_analysis.model_comparison import (
    CandidateModel,
    ModelComparisonReport,
    compare_shape_models,
)
from wow_signal_analysis.morse import MorseRegistry, load_verified_morse_registry
from wow_signal_analysis.morse_correspondence import (
    MorseCorrespondenceReport,
    analyze_threshold_morse,
)
from wow_signal_analysis.null_model import (
    PermutationNullReport,
    analyze_permutation_null,
)
from wow_signal_analysis.profile import SequenceProfile, analyze_samples
from wow_signal_analysis.quantization import (
    FitMetric,
    QuantizationSensitivityReport,
    analyze_quantization_corners,
)
from wow_signal_analysis.repository_contract import (
    RepositoryContractReport,
    verify_repository_contract,
)

ANALYSIS_SNAPSHOT_SCHEMA_VERSION: Final = 1
ANALYSIS_SNAPSHOT_ID: Final = "wow-signal-canonical-analysis-v1"


class AnalysisSnapshotError(ValueError):
    """Raised when cumulative analysis components are inconsistent."""


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    """Deterministic controls included in every generated analysis snapshot."""

    gaussian_search: GaussianSearchConfig = field(default_factory=GaussianSearchConfig)
    selected_morse_glyphs: tuple[str, ...] = ("?", ",")
    max_unique_sequences: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.gaussian_search, GaussianSearchConfig):
            raise AnalysisSnapshotError(
                "gaussian_search must be a GaussianSearchConfig"
            )
        if not self.selected_morse_glyphs:
            raise AnalysisSnapshotError("selected_morse_glyphs must not be empty")
        if len(set(self.selected_morse_glyphs)) != len(self.selected_morse_glyphs):
            raise AnalysisSnapshotError(
                "selected_morse_glyphs must not contain duplicates"
            )
        if any(
            len(glyph) != 1 or glyph.isspace()
            for glyph in self.selected_morse_glyphs
        ):
            raise AnalysisSnapshotError(
                "selected_morse_glyphs must contain single printable glyphs"
            )
        if not isinstance(self.max_unique_sequences, int) or isinstance(
            self.max_unique_sequences,
            bool,
        ):
            raise AnalysisSnapshotError("max_unique_sequences must be an integer")
        if self.max_unique_sequences <= 0:
            raise AnalysisSnapshotError("max_unique_sequences must be positive")


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Verified inputs and deterministic analyses for the canonical event."""

    schema_version: int
    analysis_id: str
    config: SnapshotConfig
    repository: RepositoryContractReport
    dataset: ObservationDataset
    profile: SequenceProfile
    gaussian_fit: GaussianTransitFit
    quantization: QuantizationSensitivityReport
    model_comparison: ModelComparisonReport
    morse_registry: MorseRegistry
    morse_correspondence: MorseCorrespondenceReport
    permutation_null: PermutationNullReport
    frequency_context: FrequencyContext
    claim_ledger: ClaimLedger
    hypothesis_matrix: BoundHypothesisMatrix

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_SNAPSHOT_SCHEMA_VERSION:
            raise AnalysisSnapshotError(
                "unsupported analysis snapshot schema_version"
            )
        if self.analysis_id != ANALYSIS_SNAPSHOT_ID:
            raise AnalysisSnapshotError(
                f"analysis_id must be {ANALYSIS_SNAPSHOT_ID!r}"
            )

        midpoint_snr = self.dataset.midpoint_snr
        elapsed_seconds = tuple(
            sample.elapsed_seconds for sample in self.dataset.samples
        )
        gaussian_observed = tuple(
            sample.observed_snr for sample in self.gaussian_fit.samples
        )

        if self.repository.printer_sequence != self.dataset.printer_sequence:
            raise AnalysisSnapshotError(
                "repository contract and dataset printer sequences differ"
            )
        if self.profile.values != midpoint_snr:
            raise AnalysisSnapshotError(
                "sequence profile does not match dataset midpoint values"
            )
        if gaussian_observed != midpoint_snr:
            raise AnalysisSnapshotError(
                "Gaussian fit observations do not match dataset midpoint values"
            )
        if self.quantization.midpoint_fit != self.gaussian_fit:
            raise AnalysisSnapshotError(
                "quantization midpoint fit does not match the canonical Gaussian fit"
            )
        if self.model_comparison.elapsed_seconds != elapsed_seconds:
            raise AnalysisSnapshotError(
                "model comparison times do not match the dataset"
            )
        if self.model_comparison.observed_snr != midpoint_snr:
            raise AnalysisSnapshotError(
                "model comparison values do not match the dataset"
            )
        if self.morse_correspondence.values != midpoint_snr:
            raise AnalysisSnapshotError(
                "Morse correspondence values do not match the dataset"
            )
        if self.permutation_null.original_values != midpoint_snr:
            raise AnalysisSnapshotError(
                "permutation null values do not match the dataset"
            )

        null_glyphs = tuple(
            summary.glyph for summary in self.permutation_null.glyph_summaries
        )
        if null_glyphs != self.config.selected_morse_glyphs:
            raise AnalysisSnapshotError(
                "permutation null glyphs do not match snapshot configuration"
            )
        for glyph in self.config.selected_morse_glyphs:
            self.morse_registry.symbol_for_glyph(glyph)

        if self.claim_ledger.ledger_id != self.repository.claim_ledger_id:
            raise AnalysisSnapshotError(
                "claim ledger does not match the repository contract"
            )
        if self.hypothesis_matrix.ledger_id != self.claim_ledger.ledger_id:
            raise AnalysisSnapshotError(
                "hypothesis matrix is not bound to the snapshot claim ledger"
            )
        if self.hypothesis_matrix.matrix_id != self.repository.hypothesis_matrix_id:
            raise AnalysisSnapshotError(
                "hypothesis matrix does not match the repository contract"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible snapshot representation."""

        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "configuration": self._configuration_mapping(),
            "repository": self._repository_mapping(),
            "observation": self._observation_mapping(),
            "beam_transit": self._beam_mapping(),
            "quantization_sensitivity": self._quantization_mapping(),
            "shape_model_comparison": self._model_comparison_mapping(),
            "symbolic_correspondence": self._symbolic_mapping(),
            "frequency_context": self._frequency_mapping(),
            "claim_ledger": self._claim_mapping(),
            "hypothesis_matrix": self._hypothesis_mapping(),
            "interpretive_limits": [
                "A good beam-shape fit does not identify the emitter or its origin.",
                (
                    "Threshold and Morse correspondences are reproducible "
                    "transformations, not recovered transmitter metadata."
                ),
                (
                    "Hydrogen-line proximity does not establish artificial "
                    "or extraterrestrial origin."
                ),
                (
                    "The snapshot does not establish an intentional message "
                    "or extraterrestrial technology."
                ),
            ],
        }

    def to_json(self) -> str:
        """Serialize the snapshot with stable keys, indentation, and a final newline."""

        return (
            json.dumps(
                self.to_mapping(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _configuration_mapping(self) -> dict[str, object]:
        search = self.config.gaussian_search
        return {
            "gaussian_search": {
                "grid_points": search.grid_points,
                "refinement_rounds": search.refinement_rounds,
                "center_padding_cadences": _float_text(
                    search.center_padding_cadences
                ),
                "minimum_sigma_cadences": _float_text(
                    search.minimum_sigma_cadences
                ),
                "maximum_sigma_spans": _float_text(search.maximum_sigma_spans),
            },
            "selected_morse_glyphs": list(self.config.selected_morse_glyphs),
            "max_unique_sequences": self.config.max_unique_sequences,
        }

    def _repository_mapping(self) -> dict[str, object]:
        return {
            "repository_root": ".",
            "verified_component_count": self.repository.verified_component_count,
            "total_record_count": self.repository.total_record_count,
            "components": [
                {
                    "component_id": component.component_id,
                    "artifact_path": str(component.artifact_path),
                    "record_count": component.record_count,
                }
                for component in self.repository.components
            ],
        }

    def _observation_mapping(self) -> dict[str, object]:
        return {
            "printer_sequence": self.dataset.printer_sequence,
            "sample_count": len(self.dataset.samples),
            "elapsed_seconds": [
                _decimal_text(sample.elapsed_seconds)
                for sample in self.dataset.samples
            ],
            "midpoint_snr": [
                _decimal_text(value) for value in self.dataset.midpoint_snr
            ],
            "trend_signature": self.profile.trend_signature,
            "peak_index": self.profile.peak_index,
            "peak_value": _decimal_text(self.profile.peak_value),
            "strict_single_peak": self.profile.is_strict_single_peak,
            "exact_palindrome": self.profile.is_exact_palindrome,
            "mirror_signed_differences": [
                _decimal_text(comparison.signed_difference)
                for comparison in self.profile.mirror_comparisons
            ],
        }

    def _beam_mapping(self) -> dict[str, object]:
        fit = self.gaussian_fit
        return {
            "model": "zero-baseline-gaussian",
            "amplitude_snr": _float_text(fit.amplitude_snr),
            "center_seconds": _float_text(fit.center_seconds),
            "sigma_seconds": _float_text(fit.sigma_seconds),
            "fwhm_seconds": _float_text(fit.fwhm_seconds),
            "sum_squared_error": _float_text(fit.sum_squared_error),
            "root_mean_squared_error": _float_text(
                fit.root_mean_squared_error
            ),
            "coefficient_of_determination": _float_text(
                fit.coefficient_of_determination
            ),
            "samples": [
                {
                    "sample_index": sample.sample_index,
                    "observed_snr": _decimal_text(sample.observed_snr),
                    "predicted_snr": _float_text(sample.predicted_snr),
                    "residual_snr": _float_text(sample.residual_snr),
                }
                for sample in fit.samples
            ],
        }

    def _quantization_mapping(self) -> dict[str, object]:
        return {
            "interpretation": (
                "Exhaustive lower-bound/upper-supremum corner ranges; "
                "not confidence intervals."
            ),
            "evaluated_corner_count": self.quantization.evaluated_corner_count,
            "metric_envelopes": [
                {
                    "metric": metric.value,
                    "minimum": _float_text(
                        self.quantization.envelope(metric).minimum
                    ),
                    "maximum": _float_text(
                        self.quantization.envelope(metric).maximum
                    ),
                    "span": _float_text(
                        self.quantization.envelope(metric).span
                    ),
                    "minimum_corner_pattern": self.quantization.envelope(
                        metric
                    ).minimum_corner_pattern,
                    "maximum_corner_pattern": self.quantization.envelope(
                        metric
                    ).maximum_corner_pattern,
                }
                for metric in FitMetric
            ],
        }

    def _model_comparison_mapping(self) -> dict[str, object]:
        return {
            "method": "leave-one-out-cross-validation",
            "best_model": self.model_comparison.best_model.model.value,
            "ranking": [
                result.model.value
                for result in self.model_comparison.ranked_by_prediction_error
            ],
            "models": [
                {
                    "model": result.model.value,
                    "parameter_count": result.parameter_count,
                    "prediction_sum_squares": _float_text(
                        result.prediction_sum_squares
                    ),
                    "root_mean_squared_prediction_error": _float_text(
                        result.root_mean_squared_prediction_error
                    ),
                    "mean_absolute_prediction_error": _float_text(
                        result.mean_absolute_prediction_error
                    ),
                }
                for result in (
                    self.model_comparison.result_for(model)
                    for model in CandidateModel
                )
            ],
        }

    def _symbolic_mapping(self) -> dict[str, object]:
        glyphs: list[dict[str, object]] = []

        for glyph in self.config.selected_morse_glyphs:
            observed = self.morse_correspondence.comparisons_for_glyph(glyph)
            null_summary = self.permutation_null.summary_for_glyph(glyph)
            sequence_fraction = null_summary.sequence_fraction
            comparison_fraction = null_summary.comparison_fraction

            glyphs.append(
                {
                    "glyph": glyph,
                    "observed_comparison_count": len(observed),
                    "observed_cases": [
                        {
                            "cut_index": comparison.cut_index,
                            "direction": comparison.direction.value,
                            "polarity": comparison.polarity.value,
                            "lower_bound_inclusive": _optional_decimal_text(
                                comparison.lower_bound_inclusive
                            ),
                            "upper_bound_exclusive": _optional_decimal_text(
                                comparison.upper_bound_exclusive
                            ),
                            "binary_pattern": comparison.binary_pattern,
                            "morse_pattern": comparison.morse_pattern,
                        }
                        for comparison in observed
                    ],
                    "null_model": {
                        "matched_sequence_count": (
                            null_summary.matched_sequence_count
                        ),
                        "total_sequence_count": null_summary.total_sequence_count,
                        "sequence_fraction": {
                            "numerator": sequence_fraction.numerator,
                            "denominator": sequence_fraction.denominator,
                        },
                        "matched_comparison_count": (
                            null_summary.matched_comparison_count
                        ),
                        "total_comparison_count": (
                            null_summary.total_comparison_count
                        ),
                        "comparison_fraction": {
                            "numerator": comparison_fraction.numerator,
                            "denominator": comparison_fraction.denominator,
                        },
                    },
                }
            )

        return {
            "standard_id": self.morse_registry.standard_id,
            "threshold_case_count": len(
                self.morse_correspondence.threshold_cases
            ),
            "comparison_count_per_observed_sequence": len(
                self.morse_correspondence.comparisons
            ),
            "glyphs": glyphs,
            "conclusion": (
                "The correspondences are reproducible but non-unique and do not "
                "establish intentional Morse transmission."
            ),
        }

    def _frequency_mapping(self) -> dict[str, object]:
        return {
            "rest_line": {
                "line_id": self.frequency_context.rest_line.line_id,
                "species": self.frequency_context.rest_line.species,
                "rest_frequency_mhz": _decimal_text(
                    self.frequency_context.rest_line.rest_frequency_mhz
                ),
            },
            "estimates": [
                {
                    "estimate_id": estimate.estimate_id,
                    "status": estimate.status.value,
                    "frequency_mhz": _decimal_text(estimate.frequency_mhz),
                    "uncertainty_mhz": _decimal_text(
                        estimate.uncertainty_mhz
                    ),
                    "delta_mhz": _decimal_text(
                        self.frequency_context.offset_for(
                            estimate.estimate_id
                        ).delta_mhz
                    ),
                    "absolute_offset_khz": _decimal_text(
                        self.frequency_context.offset_for(
                            estimate.estimate_id
                        ).absolute_offset_khz
                    ),
                    "uncertainty_interval_contains_rest": (
                        self.frequency_context.offset_for(
                            estimate.estimate_id
                        ).uncertainty_interval_contains_rest
                    ),
                }
                for estimate in self.frequency_context.estimates
            ],
        }

    def _claim_mapping(self) -> dict[str, object]:
        return {
            "ledger_id": self.claim_ledger.ledger_id,
            "claim_count": len(self.claim_ledger.claims),
            "classification_counts": [
                {
                    "classification": classification.value,
                    "count": len(
                        self.claim_ledger.claims_by_classification(
                            classification
                        )
                    ),
                }
                for classification in ClaimClassification
            ],
            "not_established_claim_ids": [
                claim.claim_id
                for claim in self.claim_ledger.claims_by_verdict(
                    ClaimVerdict.NOT_ESTABLISHED
                )
            ],
        }

    def _hypothesis_mapping(self) -> dict[str, object]:
        return {
            "matrix_id": self.hypothesis_matrix.matrix_id,
            "hypothesis_count": len(self.hypothesis_matrix.hypotheses),
            "status_counts": [
                {
                    "status": status.value,
                    "count": len(
                        self.hypothesis_matrix.hypotheses_by_status(status)
                    ),
                }
                for status in HypothesisStatus
            ],
            "hypotheses": [
                {
                    "hypothesis_id": hypothesis.record.hypothesis_id,
                    "scope": hypothesis.record.scope.value,
                    "status": hypothesis.record.status.value,
                }
                for hypothesis in self.hypothesis_matrix.hypotheses
            ],
        }


def build_analysis_snapshot(
    repository_root: Path,
    *,
    config: SnapshotConfig | None = None,
) -> AnalysisSnapshot:
    """Verify canonical evidence and execute the complete deterministic analysis."""

    snapshot_config = config or SnapshotConfig()
    root = repository_root.resolve()

    repository = verify_repository_contract(root)
    dataset = load_verified_wow_dataset(root)
    morse_registry = load_verified_morse_registry(root)
    frequency_context = load_verified_frequency_context(root)
    claim_ledger = load_verified_claim_ledger(root)
    hypothesis_matrix = load_verified_hypothesis_matrix(
        root,
        claim_ledger=claim_ledger,
    )

    profile = analyze_samples(dataset.samples)
    gaussian_fit = fit_gaussian_transit(
        dataset.samples,
        config=snapshot_config.gaussian_search,
    )
    quantization = analyze_quantization_corners(
        dataset.samples,
        config=snapshot_config.gaussian_search,
    )
    model_comparison = compare_shape_models(
        tuple(sample.elapsed_seconds for sample in dataset.samples),
        dataset.midpoint_snr,
        gaussian_config=snapshot_config.gaussian_search,
    )
    morse_correspondence = analyze_threshold_morse(
        dataset.midpoint_snr,
        morse_registry,
    )
    permutation_null = analyze_permutation_null(
        dataset.midpoint_snr,
        morse_registry,
        glyphs=snapshot_config.selected_morse_glyphs,
        max_unique_sequences=snapshot_config.max_unique_sequences,
    )

    return AnalysisSnapshot(
        schema_version=ANALYSIS_SNAPSHOT_SCHEMA_VERSION,
        analysis_id=ANALYSIS_SNAPSHOT_ID,
        config=snapshot_config,
        repository=repository,
        dataset=dataset,
        profile=profile,
        gaussian_fit=gaussian_fit,
        quantization=quantization,
        model_comparison=model_comparison,
        morse_registry=morse_registry,
        morse_correspondence=morse_correspondence,
        permutation_null=permutation_null,
        frequency_context=frequency_context,
        claim_ledger=claim_ledger,
        hypothesis_matrix=hypothesis_matrix,
    )


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise AnalysisSnapshotError("snapshot decimal values must be finite")
    return str(value)


def _optional_decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return _decimal_text(value)


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise AnalysisSnapshotError("snapshot float values must be finite")
    normalized = 0.0 if abs(value) < 0.5e-12 else value
    return f"{normalized:.12f}"
