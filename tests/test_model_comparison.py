from decimal import Decimal

import pytest

from wow_signal_analysis.beam_model import GaussianSearchConfig
from wow_signal_analysis.model_comparison import (
    CandidateModel,
    CrossValidationFold,
    ModelComparisonError,
    ModelComparisonReport,
    compare_shape_models,
)

_FAST_CONFIG = GaussianSearchConfig(
    grid_points=21,
    refinement_rounds=3,
)

_TIMES = tuple(
    Decimal(index * 12)
    for index in range(6)
)

_VALUES = (
    Decimal("6.5"),
    Decimal("14.5"),
    Decimal("26.5"),
    Decimal("30.5"),
    Decimal("19.5"),
    Decimal("5.5"),
)


@pytest.fixture(scope="module")
def report() -> ModelComparisonReport:
    return compare_shape_models(
        _TIMES,
        _VALUES,
        gaussian_config=_FAST_CONFIG,
    )


def test_report_evaluates_every_candidate_with_one_fold_per_sample(
    report: ModelComparisonReport,
) -> None:
    assert {
        result.model for result in report.results
    } == set(CandidateModel)

    assert all(
        len(result.folds) == 6
        for result in report.results
    )

    assert all(
        tuple(
            fold.held_out_index
            for fold in result.folds
        ) == tuple(range(6))
        for result in report.results
    )


def test_gaussian_transit_has_the_lowest_held_out_error(
    report: ModelComparisonReport,
) -> None:
    assert tuple(
        result.model
        for result in report.ranked_by_prediction_error
    ) == (
        CandidateModel.GAUSSIAN_TRANSIT,
        CandidateModel.QUADRATIC,
        CandidateModel.CONSTANT,
        CandidateModel.AFFINE,
    )

    assert (
        report.best_model.model
        is CandidateModel.GAUSSIAN_TRANSIT
    )


def test_canonical_cross_validation_errors_are_reproducible(
    report: ModelComparisonReport,
) -> None:
    constant = report.result_for(
        CandidateModel.CONSTANT
    )
    affine = report.result_for(
        CandidateModel.AFFINE
    )
    quadratic = report.result_for(
        CandidateModel.QUADRATIC
    )
    gaussian = report.result_for(
        CandidateModel.GAUSSIAN_TRANSIT
    )

    assert (
        constant.root_mean_squared_prediction_error
        == pytest.approx(
            11.249889,
            abs=0.000001,
        )
    )
    assert (
        affine.root_mean_squared_prediction_error
        == pytest.approx(
            15.952504,
            abs=0.000001,
        )
    )
    assert (
        quadratic.root_mean_squared_prediction_error
        == pytest.approx(
            6.596274,
            abs=0.000001,
        )
    )
    assert (
        gaussian.root_mean_squared_prediction_error
        == pytest.approx(
            2.013396,
            abs=0.000001,
        )
    )


def test_gaussian_held_out_predictions_are_reproducible(
    report: ModelComparisonReport,
) -> None:
    gaussian = report.result_for(
        CandidateModel.GAUSSIAN_TRANSIT
    )

    assert gaussian.predicted_snr == pytest.approx(
        (
            3.933194,
            15.344816,
            28.528249,
            28.671551,
            18.245169,
            8.326436,
        ),
        abs=0.000002,
    )


def test_quadratic_endpoint_extrapolation_is_retained_not_hidden(
    report: ModelComparisonReport,
) -> None:
    quadratic = report.result_for(
        CandidateModel.QUADRATIC
    )

    assert quadratic.predicted_snr[0] == pytest.approx(
        -6.7,
        abs=1e-12,
    )
    assert quadratic.predicted_snr[-1] == pytest.approx(
        9.1,
        abs=1e-12,
    )


def test_rmse_ratio_quantifies_separation_from_the_best_model(
    report: ModelComparisonReport,
) -> None:
    assert report.rmse_ratio_to_best(
        CandidateModel.QUADRATIC
    ) == pytest.approx(
        3.276194,
        abs=0.000001,
    )

    assert report.rmse_ratio_to_best(
        CandidateModel.AFFINE
    ) == pytest.approx(
        7.923184,
        abs=0.000001,
    )


def test_comparison_is_deterministic() -> None:
    first = compare_shape_models(
        _TIMES,
        _VALUES,
        gaussian_config=_FAST_CONFIG,
    )
    second = compare_shape_models(
        _TIMES,
        _VALUES,
        gaussian_config=_FAST_CONFIG,
    )

    assert first == second


@pytest.mark.parametrize(
    ("times", "values", "message"),
    [
        (
            _TIMES[:-1],
            _VALUES,
            "equal length",
        ),
        (
            _TIMES[:3],
            _VALUES[:3],
            "at least four",
        ),
        (
            (
                Decimal("0"),
                Decimal("12"),
                Decimal("12"),
                Decimal("36"),
            ),
            _VALUES[:4],
            "strictly increasing",
        ),
        (
            _TIMES,
            (Decimal("1"),) * 6,
            "must not all be equal",
        ),
    ],
)
def test_invalid_series_fail_closed(
    times: tuple[Decimal, ...],
    values: tuple[Decimal, ...],
    message: str,
) -> None:
    with pytest.raises(
        ModelComparisonError,
        match=message,
    ):
        compare_shape_models(
            times,
            values,
            gaussian_config=_FAST_CONFIG,
        )


def test_fold_validation_rejects_inconsistent_derived_values() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="must equal",
    ):
        CrossValidationFold(
            held_out_index=0,
            elapsed_seconds=Decimal("0"),
            observed_snr=Decimal("6.5"),
            predicted_snr=5.0,
            residual_snr=2.0,
            squared_error=4.0,
        )
