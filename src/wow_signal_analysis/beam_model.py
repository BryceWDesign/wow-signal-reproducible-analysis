"""Deterministic Gaussian beam-transit fitting for ordered signal samples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from typing import Final

from wow_signal_analysis.measurements import SignalSample

_MINIMUM_SAMPLES: Final = 3
_DEFAULT_GRID_POINTS: Final = 161
_DEFAULT_REFINEMENT_ROUNDS: Final = 5
_TWO: Final = 2.0


class BeamModelError(ValueError):
    """Raised when a beam-transit fit cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class GaussianSearchConfig:
    """Deterministic bounded-search controls for the Gaussian transit model."""

    grid_points: int = _DEFAULT_GRID_POINTS
    refinement_rounds: int = _DEFAULT_REFINEMENT_ROUNDS
    center_padding_cadences: float = 1.0
    minimum_sigma_cadences: float = 0.1
    maximum_sigma_spans: float = 2.0

    def __post_init__(self) -> None:
        if self.grid_points < 3 or self.grid_points % 2 == 0:
            raise BeamModelError("grid_points must be an odd integer of at least 3")
        if self.refinement_rounds <= 0:
            raise BeamModelError("refinement_rounds must be positive")
        for field_name in (
            "center_padding_cadences",
            "minimum_sigma_cadences",
            "maximum_sigma_spans",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise BeamModelError(f"{field_name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class BeamSampleFit:
    """Observed, predicted, and residual values for one telescope sample."""

    sample_index: int
    elapsed_seconds: Decimal
    observed_snr: Decimal
    predicted_snr: float
    residual_snr: float

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise BeamModelError("sample_index must be non-negative")
        if not self.elapsed_seconds.is_finite() or self.elapsed_seconds < 0:
            raise BeamModelError("elapsed_seconds must be non-negative and finite")
        if not self.observed_snr.is_finite() or self.observed_snr < 0:
            raise BeamModelError("observed_snr must be non-negative and finite")
        if not math.isfinite(self.predicted_snr) or self.predicted_snr < 0.0:
            raise BeamModelError("predicted_snr must be non-negative and finite")
        if not math.isfinite(self.residual_snr):
            raise BeamModelError("residual_snr must be finite")

        expected_residual = float(self.observed_snr) - self.predicted_snr
        if not math.isclose(
            self.residual_snr,
            expected_residual,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise BeamModelError(
                "residual_snr must equal observed_snr minus predicted_snr"
            )


@dataclass(frozen=True, slots=True)
class GaussianTransitFit:
    """Least-squares fit of a zero-baseline Gaussian beam-transit model."""

    amplitude_snr: float
    center_seconds: float
    sigma_seconds: float
    fwhm_seconds: float
    sum_squared_error: float
    root_mean_squared_error: float
    coefficient_of_determination: float
    samples: tuple[BeamSampleFit, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "amplitude_snr",
            "center_seconds",
            "sigma_seconds",
            "fwhm_seconds",
            "sum_squared_error",
            "root_mean_squared_error",
            "coefficient_of_determination",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise BeamModelError(f"{field_name} must be finite")
        if self.amplitude_snr <= 0.0:
            raise BeamModelError("amplitude_snr must be positive")
        if self.sigma_seconds <= 0.0 or self.fwhm_seconds <= 0.0:
            raise BeamModelError("sigma_seconds and fwhm_seconds must be positive")
        if self.sum_squared_error < 0.0 or self.root_mean_squared_error < 0.0:
            raise BeamModelError("fit errors must be non-negative")
        if len(self.samples) < _MINIMUM_SAMPLES:
            raise BeamModelError("fit must contain at least three samples")
        if tuple(sample.sample_index for sample in self.samples) != tuple(
            range(len(self.samples))
        ):
            raise BeamModelError("fit sample indices must be contiguous and zero-based")

        expected_fwhm = gaussian_fwhm(self.sigma_seconds)
        if not math.isclose(
            self.fwhm_seconds,
            expected_fwhm,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise BeamModelError("fwhm_seconds does not match sigma_seconds")

        expected_sse = sum(sample.residual_snr**2 for sample in self.samples)
        if not math.isclose(
            self.sum_squared_error,
            expected_sse,
            rel_tol=1e-11,
            abs_tol=1e-11,
        ):
            raise BeamModelError("sum_squared_error does not match sample residuals")

        expected_rmse = math.sqrt(self.sum_squared_error / len(self.samples))
        if not math.isclose(
            self.root_mean_squared_error,
            expected_rmse,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise BeamModelError(
                "root_mean_squared_error does not match sum_squared_error"
            )

    @property
    def sample_count(self) -> int:
        """Return the number of observations included in the fit."""

        return len(self.samples)

    def predict(self, elapsed_seconds: float) -> float:
        """Predict signal-to-noise at one elapsed time under the fitted model."""

        if not math.isfinite(elapsed_seconds):
            raise BeamModelError("elapsed_seconds must be finite")
        return gaussian_response(
            elapsed_seconds,
            amplitude_snr=self.amplitude_snr,
            center_seconds=self.center_seconds,
            sigma_seconds=self.sigma_seconds,
        )


@dataclass(frozen=True, slots=True)
class _CandidateFit:
    center_seconds: float
    sigma_seconds: float
    amplitude_snr: float
    sum_squared_error: float


def gaussian_response(
    elapsed_seconds: float,
    *,
    amplitude_snr: float,
    center_seconds: float,
    sigma_seconds: float,
) -> float:
    """Evaluate a zero-baseline Gaussian beam response."""

    for field_name, value in (
        ("elapsed_seconds", elapsed_seconds),
        ("amplitude_snr", amplitude_snr),
        ("center_seconds", center_seconds),
        ("sigma_seconds", sigma_seconds),
    ):
        if not math.isfinite(value):
            raise BeamModelError(f"{field_name} must be finite")
    if amplitude_snr <= 0.0:
        raise BeamModelError("amplitude_snr must be positive")
    if sigma_seconds <= 0.0:
        raise BeamModelError("sigma_seconds must be positive")

    standardized = (elapsed_seconds - center_seconds) / sigma_seconds
    return amplitude_snr * math.exp(-0.5 * standardized**2)


def gaussian_fwhm(sigma_seconds: float) -> float:
    """Convert Gaussian sigma to full width at half maximum."""

    if not math.isfinite(sigma_seconds) or sigma_seconds <= 0.0:
        raise BeamModelError("sigma_seconds must be positive and finite")
    return _TWO * math.sqrt(_TWO * math.log(_TWO)) * sigma_seconds


def fit_gaussian_transit(
    samples: Sequence[SignalSample],
    *,
    config: GaussianSearchConfig | None = None,
) -> GaussianTransitFit:
    """Fit midpoint estimates from ordered signal samples."""

    normalized = tuple(samples)
    if len(normalized) < _MINIMUM_SAMPLES:
        raise BeamModelError("at least three signal samples are required")
    if tuple(sample.sample_index for sample in normalized) != tuple(
        range(len(normalized))
    ):
        raise BeamModelError("sample indices must be contiguous and zero-based")

    return fit_gaussian_series(
        tuple(sample.elapsed_seconds for sample in normalized),
        tuple(sample.intensity.midpoint_snr for sample in normalized),
        config=config,
    )


def fit_gaussian_series(
    elapsed_seconds: Sequence[Decimal],
    observed_snr: Sequence[Decimal],
    *,
    config: GaussianSearchConfig | None = None,
) -> GaussianTransitFit:
    """Fit arbitrary decimal observations using the shared Gaussian search."""

    times_decimal = tuple(elapsed_seconds)
    observed_decimal = tuple(observed_snr)
    _validate_series(times_decimal, observed_decimal)
    search_config = config or GaussianSearchConfig()

    times = tuple(float(value) for value in times_decimal)
    observed = tuple(float(value) for value in observed_decimal)
    average_cadence = (times[-1] - times[0]) / (len(times) - 1)
    span = times[-1] - times[0]

    original_center_lower = (
        times[0] - search_config.center_padding_cadences * average_cadence
    )
    original_center_upper = (
        times[-1] + search_config.center_padding_cadences * average_cadence
    )
    original_sigma_lower = search_config.minimum_sigma_cadences * average_cadence
    original_sigma_upper = search_config.maximum_sigma_spans * span

    center_lower = original_center_lower
    center_upper = original_center_upper
    sigma_lower = original_sigma_lower
    sigma_upper = original_sigma_upper
    best: _CandidateFit | None = None

    for _ in range(search_config.refinement_rounds):
        centers = _linspace(center_lower, center_upper, search_config.grid_points)
        sigmas = _linspace(sigma_lower, sigma_upper, search_config.grid_points)
        round_best: _CandidateFit | None = None

        for center in centers:
            for sigma in sigmas:
                candidate = _fit_candidate(
                    times,
                    observed,
                    center,
                    sigma,
                )
                if candidate is None:
                    continue
                if round_best is None or _candidate_key(candidate) < _candidate_key(
                    round_best
                ):
                    round_best = candidate

        if round_best is None:
            raise BeamModelError("Gaussian search produced no finite candidate fit")
        best = round_best

        center_step = (center_upper - center_lower) / (
            search_config.grid_points - 1
        )
        sigma_step = (sigma_upper - sigma_lower) / (
            search_config.grid_points - 1
        )

        center_lower = max(
            original_center_lower,
            best.center_seconds - center_step,
        )
        center_upper = min(
            original_center_upper,
            best.center_seconds + center_step,
        )
        sigma_lower = max(
            original_sigma_lower,
            best.sigma_seconds - sigma_step,
        )
        sigma_upper = min(
            original_sigma_upper,
            best.sigma_seconds + sigma_step,
        )

    if best is None:
        raise BeamModelError("Gaussian search produced no candidate fit")

    sample_fits = tuple(
        _series_sample_fit(index, elapsed, observed_value, best)
        for index, (elapsed, observed_value) in enumerate(
            zip(times_decimal, observed_decimal, strict=True)
        )
    )
    sse = sum(sample.residual_snr**2 for sample in sample_fits)
    rmse = math.sqrt(sse / len(sample_fits))
    mean_observed = sum(observed) / len(observed)
    total_sum_squares = sum((value - mean_observed) ** 2 for value in observed)

    if total_sum_squares <= 0.0:
        raise BeamModelError("observed midpoint values must not all be equal")

    r_squared = 1.0 - (sse / total_sum_squares)

    return GaussianTransitFit(
        amplitude_snr=best.amplitude_snr,
        center_seconds=best.center_seconds,
        sigma_seconds=best.sigma_seconds,
        fwhm_seconds=gaussian_fwhm(best.sigma_seconds),
        sum_squared_error=sse,
        root_mean_squared_error=rmse,
        coefficient_of_determination=r_squared,
        samples=sample_fits,
    )


def _validate_series(
    elapsed_seconds: tuple[Decimal, ...],
    observed_snr: tuple[Decimal, ...],
) -> None:
    if len(elapsed_seconds) != len(observed_snr):
        raise BeamModelError("elapsed_seconds and observed_snr must have equal length")
    if len(elapsed_seconds) < _MINIMUM_SAMPLES:
        raise BeamModelError("at least three observations are required")
    if any(not value.is_finite() or value < 0 for value in elapsed_seconds):
        raise BeamModelError("elapsed_seconds must be non-negative and finite")
    if any(current <= previous for previous, current in pairwise(elapsed_seconds)):
        raise BeamModelError("elapsed_seconds must be strictly increasing")
    if any(not value.is_finite() or value < 0 for value in observed_snr):
        raise BeamModelError("observed_snr must be non-negative and finite")
    if len(set(observed_snr)) == 1:
        raise BeamModelError("observed_snr values must not all be equal")


def _candidate_key(candidate: _CandidateFit) -> tuple[float, float, float]:
    return (
        candidate.sum_squared_error,
        candidate.sigma_seconds,
        candidate.center_seconds,
    )


def _fit_candidate(
    times: tuple[float, ...],
    observed: tuple[float, ...],
    center_seconds: float,
    sigma_seconds: float,
) -> _CandidateFit | None:
    basis = tuple(
        math.exp(-0.5 * ((time - center_seconds) / sigma_seconds) ** 2)
        for time in times
    )
    denominator = sum(value**2 for value in basis)
    if denominator <= 0.0:
        return None

    amplitude = sum(
        observed_value * basis_value
        for observed_value, basis_value in zip(observed, basis, strict=True)
    ) / denominator

    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise BeamModelError(
            "Gaussian candidate amplitude must be positive and finite"
        )

    sse = sum(
        (observed_value - amplitude * basis_value) ** 2
        for observed_value, basis_value in zip(observed, basis, strict=True)
    )

    return _CandidateFit(
        center_seconds=center_seconds,
        sigma_seconds=sigma_seconds,
        amplitude_snr=amplitude,
        sum_squared_error=sse,
    )


def _series_sample_fit(
    sample_index: int,
    elapsed_seconds: Decimal,
    observed_snr: Decimal,
    candidate: _CandidateFit,
) -> BeamSampleFit:
    predicted = gaussian_response(
        float(elapsed_seconds),
        amplitude_snr=candidate.amplitude_snr,
        center_seconds=candidate.center_seconds,
        sigma_seconds=candidate.sigma_seconds,
    )
    return BeamSampleFit(
        sample_index=sample_index,
        elapsed_seconds=elapsed_seconds,
        observed_snr=observed_snr,
        predicted_snr=predicted,
        residual_snr=float(observed_snr) - predicted,
    )


def _linspace(
    lower: float,
    upper: float,
    count: int,
) -> tuple[float, ...]:
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise BeamModelError("search bounds must be finite")
    if lower >= upper:
        raise BeamModelError("search lower bound must be below upper bound")
    if count < 2:
        raise BeamModelError("linspace count must be at least 2")

    step = (upper - lower) / (count - 1)
    return tuple(lower + index * step for index in range(count))
