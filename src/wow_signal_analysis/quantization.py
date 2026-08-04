"""Corner sensitivity analysis for quantized Big Ear printer intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import product

from wow_signal_analysis.beam_model import (
    GaussianSearchConfig,
    GaussianTransitFit,
    fit_gaussian_series,
    fit_gaussian_transit,
)
from wow_signal_analysis.measurements import SignalSample


class QuantizationError(ValueError):
    """Raised when quantization sensitivity analysis is inconsistent."""


class FitMetric(StrEnum):
    """Gaussian fit metric eligible for a corner sensitivity envelope."""

    AMPLITUDE_SNR = "amplitude_snr"
    CENTER_SECONDS = "center_seconds"
    SIGMA_SECONDS = "sigma_seconds"
    FWHM_SECONDS = "fwhm_seconds"
    SUM_SQUARED_ERROR = "sum_squared_error"
    ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    COEFFICIENT_OF_DETERMINATION = "coefficient_of_determination"


@dataclass(frozen=True, slots=True)
class QuantizationCornerFit:
    """One lower-bound/upper-supremum corner and its Gaussian fit."""

    corner_index: int
    upper_supremum_mask: tuple[bool, ...]
    observed_snr: tuple[Decimal, ...]
    fit: GaussianTransitFit

    def __post_init__(self) -> None:
        if self.corner_index < 0:
            raise QuantizationError("corner_index must be non-negative")
        if not self.upper_supremum_mask:
            raise QuantizationError("upper_supremum_mask must not be empty")
        if len(self.upper_supremum_mask) != len(self.observed_snr):
            raise QuantizationError("upper_supremum_mask and observed_snr must have equal length")
        if len(self.observed_snr) != self.fit.sample_count:
            raise QuantizationError("observed_snr length must match the Gaussian fit sample count")
        if any(not value.is_finite() for value in self.observed_snr):
            raise QuantizationError("observed_snr values must be finite")
        if tuple(sample.observed_snr for sample in self.fit.samples) != self.observed_snr:
            raise QuantizationError("fit observations must match observed_snr")

    @property
    def mask_pattern(self) -> str:
        """Render lower bounds as 0 and upper suprema as 1."""

        return "".join("1" if selected else "0" for selected in self.upper_supremum_mask)

    def metric_value(self, metric: FitMetric) -> float:
        """Return one declared fit metric without arbitrary dynamic access."""

        return float(getattr(self.fit, metric.value))


@dataclass(frozen=True, slots=True)
class MetricEnvelope:
    """Minimum and maximum observed across all evaluated interval corners."""

    metric: FitMetric
    minimum: float
    maximum: float
    minimum_corner_pattern: str
    maximum_corner_pattern: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise QuantizationError("metric envelope values must be finite")
        if self.minimum > self.maximum:
            raise QuantizationError("metric envelope minimum cannot exceed maximum")

        for field_name in (
            "minimum_corner_pattern",
            "maximum_corner_pattern",
        ):
            pattern = getattr(self, field_name)
            if not pattern or any(character not in {"0", "1"} for character in pattern):
                raise QuantizationError(f"{field_name} must be a binary pattern")

    @property
    def span(self) -> float:
        """Return the full observed range across evaluated corners."""

        return self.maximum - self.minimum


@dataclass(frozen=True, slots=True)
class QuantizationSensitivityReport:
    """Midpoint fit and exhaustive lower/upper-supremum corner fits."""

    sample_count: int
    midpoint_fit: GaussianTransitFit
    corners: tuple[QuantizationCornerFit, ...]

    def __post_init__(self) -> None:
        if self.sample_count < 3:
            raise QuantizationError("sample_count must be at least three")
        if self.midpoint_fit.sample_count != self.sample_count:
            raise QuantizationError("midpoint fit sample count does not match report")

        expected_corner_count = 2**self.sample_count
        if len(self.corners) != expected_corner_count:
            raise QuantizationError(
                f"expected {expected_corner_count} corners, found {len(self.corners)}"
            )

        expected_indices = tuple(range(expected_corner_count))
        actual_indices = tuple(corner.corner_index for corner in self.corners)
        if actual_indices != expected_indices:
            raise QuantizationError("corner indices must be contiguous and zero-based")

        patterns = tuple(corner.mask_pattern for corner in self.corners)
        if len(set(patterns)) != expected_corner_count:
            raise QuantizationError("corner mask patterns must be unique")
        if any(len(pattern) != self.sample_count for pattern in patterns):
            raise QuantizationError("corner mask lengths must match sample_count")

    @property
    def evaluated_corner_count(self) -> int:
        """Return the number of interval-corner combinations evaluated."""

        return len(self.corners)

    def corner_for_pattern(
        self,
        pattern: str,
    ) -> QuantizationCornerFit:
        """Return one exact corner by its lower/upper selection pattern."""

        if len(pattern) != self.sample_count or any(
            character not in {"0", "1"} for character in pattern
        ):
            raise QuantizationError(
                f"corner pattern must contain exactly {self.sample_count} binary digits"
            )

        matches = tuple(corner for corner in self.corners if corner.mask_pattern == pattern)
        if len(matches) != 1:
            raise QuantizationError(
                f"expected one corner for pattern {pattern!r}, found {len(matches)}"
            )
        return matches[0]

    def envelope(self, metric: FitMetric) -> MetricEnvelope:
        """Return the observed corner range for one fit metric."""

        minimum_corner = min(
            self.corners,
            key=lambda corner: (
                corner.metric_value(metric),
                corner.mask_pattern,
            ),
        )
        maximum_corner = max(
            self.corners,
            key=lambda corner: (
                corner.metric_value(metric),
                corner.mask_pattern,
            ),
        )

        return MetricEnvelope(
            metric=metric,
            minimum=minimum_corner.metric_value(metric),
            maximum=maximum_corner.metric_value(metric),
            minimum_corner_pattern=minimum_corner.mask_pattern,
            maximum_corner_pattern=maximum_corner.mask_pattern,
        )


def analyze_quantization_corners(
    samples: Sequence[SignalSample],
    *,
    config: GaussianSearchConfig | None = None,
) -> QuantizationSensitivityReport:
    """Fit every lower-bound/upper-supremum printer interval corner.

    Upper printer bounds are exclusive. They are evaluated as mathematical
    suprema for sensitivity analysis, not asserted as observed signal values.
    The resulting envelopes are corner ranges, not confidence intervals.
    """

    normalized = tuple(samples)
    midpoint_fit = fit_gaussian_transit(
        normalized,
        config=config,
    )
    elapsed_seconds = tuple(sample.elapsed_seconds for sample in normalized)
    corners: list[QuantizationCornerFit] = []

    for corner_index, mask in enumerate(product((False, True), repeat=len(normalized))):
        observed_snr = tuple(
            (sample.intensity.upper_snr if use_upper else sample.intensity.lower_snr)
            for sample, use_upper in zip(
                normalized,
                mask,
                strict=True,
            )
        )
        fit = fit_gaussian_series(
            elapsed_seconds,
            observed_snr,
            config=config,
        )
        corners.append(
            QuantizationCornerFit(
                corner_index=corner_index,
                upper_supremum_mask=mask,
                observed_snr=observed_snr,
                fit=fit,
            )
        )

    return QuantizationSensitivityReport(
        sample_count=len(normalized),
        midpoint_fit=midpoint_fit,
        corners=tuple(corners),
    )
