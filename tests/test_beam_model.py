from __future__ import annotations

import math
from decimal import Decimal

import pytest

from wow_signal_analysis.beam_model import (
    BeamModelError,
    BeamSampleFit,
    GaussianSearchConfig,
    GaussianTransitFit,
    fit_gaussian_transit,
    gaussian_fwhm,
    gaussian_response,
)
from wow_signal_analysis.measurements import (
    SignalSample,
    canonical_wow_samples,
    decode_printer_sequence,
    decode_printer_symbol,
)


def test_canonical_gaussian_fit_reproduces_the_documented_beam_scale() -> None:
    fit = fit_gaussian_transit(canonical_wow_samples())

    assert fit.amplitude_snr == pytest.approx(30.754, abs=0.002)
    assert fit.center_seconds == pytest.approx(31.79, abs=0.03)
    assert fit.sigma_seconds == pytest.approx(16.40, abs=0.03)
    assert fit.fwhm_seconds == pytest.approx(38.62, abs=0.08)


def test_canonical_fit_exposes_predictions_residuals_and_goodness_of_fit() -> None:
    fit = fit_gaussian_transit(canonical_wow_samples())

    assert fit.sum_squared_error == pytest.approx(7.525, abs=0.01)
    assert fit.root_mean_squared_error == pytest.approx(1.120, abs=0.002)
    assert fit.coefficient_of_determination == pytest.approx(
        0.9857,
        abs=0.0002,
    )
    assert tuple(
        sample.predicted_snr for sample in fit.samples
    ) == pytest.approx(
        (4.69, 14.84, 27.47, 29.76, 18.88, 7.01),
        abs=0.03,
    )
    assert tuple(
        sample.residual_snr for sample in fit.samples
    ) == pytest.approx(
        (1.81, -0.34, -0.97, 0.74, 0.62, -1.51),
        abs=0.03,
    )


def test_fit_predictions_equal_the_public_gaussian_response() -> None:
    fit = fit_gaussian_transit(canonical_wow_samples())

    for sample in fit.samples:
        expected = gaussian_response(
            float(sample.elapsed_seconds),
            amplitude_snr=fit.amplitude_snr,
            center_seconds=fit.center_seconds,
            sigma_seconds=fit.sigma_seconds,
        )
        assert sample.predicted_snr == pytest.approx(expected, abs=1e-12)
        assert fit.predict(float(sample.elapsed_seconds)) == pytest.approx(
            expected,
            abs=1e-12,
        )


def test_fwhm_conversion_is_exactly_bound_to_sigma() -> None:
    sigma = 16.4

    assert gaussian_fwhm(sigma) == pytest.approx(
        2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma,
        abs=1e-12,
    )


def test_search_is_deterministic_under_an_explicit_configuration() -> None:
    config = GaussianSearchConfig(
        grid_points=81,
        refinement_rounds=4,
        center_padding_cadences=1.5,
        minimum_sigma_cadences=0.05,
        maximum_sigma_spans=3.0,
    )

    first = fit_gaussian_transit(
        canonical_wow_samples(),
        config=config,
    )
    second = fit_gaussian_transit(
        canonical_wow_samples(),
        config=config,
    )

    assert first == second


def test_fit_rejects_too_few_noncontiguous_or_nonincreasing_samples() -> None:
    with pytest.raises(BeamModelError, match="at least three"):
        fit_gaussian_transit(decode_printer_sequence("6E"))

    samples = canonical_wow_samples()

    with pytest.raises(
        BeamModelError,
        match="contiguous and zero-based",
    ):
        fit_gaussian_transit(
            (
                samples[1],
                samples[0],
                *samples[2:],
            )
        )

    repeated_time = (
        samples[0],
        SignalSample(
            sample_index=1,
            elapsed_seconds=Decimal("0"),
            intensity=decode_printer_symbol("E"),
        ),
        *samples[2:],
    )

    with pytest.raises(BeamModelError, match="strictly increasing"):
        fit_gaussian_transit(repeated_time)


def test_fit_rejects_a_constant_midpoint_sequence() -> None:
    intensity = decode_printer_symbol("6")
    samples = tuple(
        SignalSample(
            sample_index=index,
            elapsed_seconds=Decimal(index * 12),
            intensity=intensity,
        )
        for index in range(3)
    )

    with pytest.raises(BeamModelError, match="must not all be equal"):
        fit_gaussian_transit(samples)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid_points", 4),
        ("refinement_rounds", 0),
        ("center_padding_cadences", 0.0),
        ("minimum_sigma_cadences", float("nan")),
        ("maximum_sigma_spans", -1.0),
    ],
)
def test_search_configuration_fails_closed(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {field: value}

    with pytest.raises(BeamModelError):
        GaussianSearchConfig(**arguments)  # type: ignore[arg-type]


def test_gaussian_response_rejects_invalid_parameters() -> None:
    with pytest.raises(
        BeamModelError,
        match="amplitude_snr must be positive",
    ):
        gaussian_response(
            0.0,
            amplitude_snr=0.0,
            center_seconds=0.0,
            sigma_seconds=1.0,
        )

    with pytest.raises(
        BeamModelError,
        match="sigma_seconds must be positive",
    ):
        gaussian_response(
            0.0,
            amplitude_snr=1.0,
            center_seconds=0.0,
            sigma_seconds=0.0,
        )

    with pytest.raises(
        BeamModelError,
        match="elapsed_seconds must be finite",
    ):
        gaussian_response(
            float("nan"),
            amplitude_snr=1.0,
            center_seconds=0.0,
            sigma_seconds=1.0,
        )


def test_beam_sample_fit_rejects_an_inconsistent_residual() -> None:
    with pytest.raises(BeamModelError, match="must equal"):
        BeamSampleFit(
            sample_index=0,
            elapsed_seconds=Decimal("0"),
            observed_snr=Decimal("6.5"),
            predicted_snr=5.0,
            residual_snr=2.0,
        )


def test_gaussian_transit_fit_rejects_an_inconsistent_fwhm() -> None:
    samples = tuple(
        BeamSampleFit(
            sample_index=index,
            elapsed_seconds=Decimal(index),
            observed_snr=Decimal("1"),
            predicted_snr=1.0,
            residual_snr=0.0,
        )
        for index in range(3)
    )

    with pytest.raises(
        BeamModelError,
        match="does not match sigma_seconds",
    ):
        GaussianTransitFit(
            amplitude_snr=1.0,
            center_seconds=1.0,
            sigma_seconds=1.0,
            fwhm_seconds=1.0,
            sum_squared_error=0.0,
            root_mean_squared_error=0.0,
            coefficient_of_determination=1.0,
            samples=samples,
        )
