from decimal import Decimal

import pytest

from wow_signal_analysis.beam_model import (
    GaussianSearchConfig,
    fit_gaussian_transit,
)
from wow_signal_analysis.measurements import canonical_wow_samples
from wow_signal_analysis.quantization import (
    FitMetric,
    MetricEnvelope,
    QuantizationError,
    QuantizationSensitivityReport,
    analyze_quantization_corners,
)

_FAST_CONFIG = GaussianSearchConfig(
    grid_points=21,
    refinement_rounds=3,
)


@pytest.fixture(scope="module")
def report() -> QuantizationSensitivityReport:
    return analyze_quantization_corners(
        canonical_wow_samples(),
        config=_FAST_CONFIG,
    )


def test_analysis_evaluates_every_quantization_corner(
    report: QuantizationSensitivityReport,
) -> None:
    assert report.sample_count == 6
    assert report.evaluated_corner_count == 64
    assert report.corners[0].mask_pattern == "000000"
    assert report.corners[-1].mask_pattern == "111111"
    assert len(
        {
            corner.mask_pattern
            for corner in report.corners
        }
    ) == 64


def test_midpoint_fit_remains_the_canonical_beam_fit(
    report: QuantizationSensitivityReport,
) -> None:
    expected = fit_gaussian_transit(
        canonical_wow_samples(),
        config=_FAST_CONFIG,
    )

    assert report.midpoint_fit == expected


def test_corner_patterns_select_lower_bounds_or_upper_suprema_exactly(
    report: QuantizationSensitivityReport,
) -> None:
    selected = report.corner_for_pattern("001100")

    assert selected.observed_snr == (
        Decimal("6"),
        Decimal("14"),
        Decimal("27"),
        Decimal("31"),
        Decimal("19"),
        Decimal("5"),
    )
    assert tuple(
        sample.observed_snr
        for sample in selected.fit.samples
    ) == selected.observed_snr


def test_corner_envelopes_are_stable(
    report: QuantizationSensitivityReport,
) -> None:
    amplitude = report.envelope(FitMetric.AMPLITUDE_SNR)
    center = report.envelope(FitMetric.CENTER_SECONDS)
    sigma = report.envelope(FitMetric.SIGMA_SECONDS)
    fwhm = report.envelope(FitMetric.FWHM_SECONDS)
    r_squared = report.envelope(
        FitMetric.COEFFICIENT_OF_DETERMINATION
    )

    assert amplitude.minimum == pytest.approx(
        29.9980,
        abs=0.0002,
    )
    assert amplitude.maximum == pytest.approx(
        31.5430,
        abs=0.0002,
    )
    assert amplitude.minimum_corner_pattern == "110001"
    assert amplitude.maximum_corner_pattern == "001110"

    assert center.minimum == pytest.approx(
        31.218,
        abs=0.001,
    )
    assert center.maximum == pytest.approx(
        32.394,
        abs=0.001,
    )
    assert center.minimum_corner_pattern == "111000"
    assert center.maximum_corner_pattern == "000111"

    assert sigma.minimum == pytest.approx(
        15.8124,
        abs=0.0002,
    )
    assert sigma.maximum == pytest.approx(
        17.0004,
        abs=0.0002,
    )
    assert fwhm.minimum == pytest.approx(
        37.2354,
        abs=0.0002,
    )
    assert fwhm.maximum == pytest.approx(
        40.0329,
        abs=0.0002,
    )

    assert r_squared.minimum == pytest.approx(
        0.972883,
        abs=0.000001,
    )
    assert r_squared.maximum == pytest.approx(
        0.994708,
        abs=0.000001,
    )


def test_error_envelopes_identify_their_extreme_corners(
    report: QuantizationSensitivityReport,
) -> None:
    sse = report.envelope(FitMetric.SUM_SQUARED_ERROR)
    rmse = report.envelope(
        FitMetric.ROOT_MEAN_SQUARED_ERROR
    )

    assert sse.minimum_corner_pattern == "011001"
    assert sse.maximum_corner_pattern == "100110"
    assert rmse.minimum_corner_pattern == "011001"
    assert rmse.maximum_corner_pattern == "100110"
    assert sse.span > 0.0
    assert rmse.span > 0.0


def test_corner_lookup_and_envelope_validation_fail_closed(
    report: QuantizationSensitivityReport,
) -> None:
    with pytest.raises(
        QuantizationError,
        match="exactly 6 binary digits",
    ):
        report.corner_for_pattern("00110")

    with pytest.raises(
        QuantizationError,
        match="binary pattern",
    ):
        MetricEnvelope(
            metric=FitMetric.AMPLITUDE_SNR,
            minimum=1.0,
            maximum=2.0,
            minimum_corner_pattern="lower",
            maximum_corner_pattern="111111",
        )
