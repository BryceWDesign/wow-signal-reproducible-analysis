"""Leave-one-out comparison of simple shape models for the Wow! sequence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from typing import Final

from wow_signal_analysis.beam_model import (
    GaussianSearchConfig,
    fit_gaussian_series,
)

_MINIMUM_OBSERVATIONS: Final = 4
_SINGULAR_TOLERANCE: Final = 1e-12


class ModelComparisonError(ValueError):
    """Raised when a shape-model comparison cannot be evaluated safely."""


class CandidateModel(StrEnum):
    """Predeclared low-complexity models used in the comparison."""

    CONSTANT = "constant"
    AFFINE = "affine"
    QUADRATIC = "quadratic"
    GAUSSIAN_TRANSIT = "gaussian-transit"

    @property
    def parameter_count(self) -> int:
        """Return the number of fitted shape parameters."""

        if self is CandidateModel.CONSTANT:
            return 1
        if self is CandidateModel.AFFINE:
            return 2
        return 3


@dataclass(frozen=True, slots=True)
class CrossValidationFold:
    """Prediction for one observation omitted from model fitting."""

    held_out_index: int
    elapsed_seconds: Decimal
    observed_snr: Decimal
    predicted_snr: float
    residual_snr: float
    squared_error: float

    def __post_init__(self) -> None:
        if self.held_out_index < 0:
            raise ModelComparisonError("held_out_index must be non-negative")
        if not self.elapsed_seconds.is_finite() or self.elapsed_seconds < 0:
            raise ModelComparisonError(
                "elapsed_seconds must be non-negative and finite"
            )
        if not self.observed_snr.is_finite() or self.observed_snr < 0:
            raise ModelComparisonError(
                "observed_snr must be non-negative and finite"
            )
        if not math.isfinite(self.predicted_snr):
            raise ModelComparisonError("predicted_snr must be finite")
        if not math.isfinite(self.residual_snr):
            raise ModelComparisonError("residual_snr must be finite")
        if not math.isfinite(self.squared_error) or self.squared_error < 0.0:
            raise ModelComparisonError(
                "squared_error must be non-negative and finite"
            )

        expected_residual = float(self.observed_snr) - self.predicted_snr
        if not math.isclose(
            self.residual_snr,
            expected_residual,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ModelComparisonError(
                "residual_snr must equal observed_snr minus predicted_snr"
            )
        if not math.isclose(
            self.squared_error,
            self.residual_snr**2,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ModelComparisonError(
                "squared_error must equal residual_snr squared"
            )


@dataclass(frozen=True, slots=True)
class ModelCrossValidation:
    """Leave-one-out prediction errors for one candidate model."""

    model: CandidateModel
    parameter_count: int
    folds: tuple[CrossValidationFold, ...]
    prediction_sum_squares: float
    root_mean_squared_prediction_error: float
    mean_absolute_prediction_error: float

    def __post_init__(self) -> None:
        if self.parameter_count != self.model.parameter_count:
            raise ModelComparisonError(
                "parameter_count does not match the candidate model"
            )
        if not self.folds:
            raise ModelComparisonError("folds must not be empty")
        if tuple(fold.held_out_index for fold in self.folds) != tuple(
            range(len(self.folds))
        ):
            raise ModelComparisonError(
                "fold indices must be contiguous and zero-based"
            )

        for field_name in (
            "prediction_sum_squares",
            "root_mean_squared_prediction_error",
            "mean_absolute_prediction_error",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ModelComparisonError(
                    f"{field_name} must be non-negative and finite"
                )

        expected_press = sum(fold.squared_error for fold in self.folds)
        if not math.isclose(
            self.prediction_sum_squares,
            expected_press,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ModelComparisonError(
                "prediction_sum_squares does not match fold errors"
            )

        expected_rmse = math.sqrt(expected_press / len(self.folds))
        if not math.isclose(
            self.root_mean_squared_prediction_error,
            expected_rmse,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ModelComparisonError(
                "root_mean_squared_prediction_error does not match fold errors"
            )

        expected_mae = sum(
            abs(fold.residual_snr) for fold in self.folds
        ) / len(self.folds)
        if not math.isclose(
            self.mean_absolute_prediction_error,
            expected_mae,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ModelComparisonError(
                "mean_absolute_prediction_error does not match fold errors"
            )

    @property
    def predicted_snr(self) -> tuple[float, ...]:
        """Return held-out predictions in original observation order."""

        return tuple(fold.predicted_snr for fold in self.folds)


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    """Complete leave-one-out comparison over all predeclared models."""

    elapsed_seconds: tuple[Decimal, ...]
    observed_snr: tuple[Decimal, ...]
    results: tuple[ModelCrossValidation, ...]

    def __post_init__(self) -> None:
        _validate_series(self.elapsed_seconds, self.observed_snr)

        expected_models = set(CandidateModel)
        actual_models = {result.model for result in self.results}
        if (
            actual_models != expected_models
            or len(self.results) != len(expected_models)
        ):
            raise ModelComparisonError(
                "results must contain each candidate model exactly once"
            )

        for result in self.results:
            if len(result.folds) != len(self.observed_snr):
                raise ModelComparisonError(
                    "every model must contain one fold per observation"
                )

            for index, fold in enumerate(result.folds):
                if fold.elapsed_seconds != self.elapsed_seconds[index]:
                    raise ModelComparisonError(
                        "fold elapsed_seconds do not match report input"
                    )
                if fold.observed_snr != self.observed_snr[index]:
                    raise ModelComparisonError(
                        "fold observed_snr values do not match report input"
                    )

    @property
    def ranked_by_prediction_error(
        self,
    ) -> tuple[ModelCrossValidation, ...]:
        """Return models from lowest to highest held-out RMSE."""

        return tuple(
            sorted(
                self.results,
                key=lambda result: (
                    result.root_mean_squared_prediction_error,
                    result.model.value,
                ),
            )
        )

    @property
    def best_model(self) -> ModelCrossValidation:
        """Return the model with the lowest held-out RMSE."""

        return self.ranked_by_prediction_error[0]

    def result_for(
        self,
        model: CandidateModel,
    ) -> ModelCrossValidation:
        """Return the unique result for one predeclared model."""

        matches = tuple(
            result for result in self.results if result.model is model
        )
        if len(matches) != 1:
            raise ModelComparisonError(
                f"expected one result for model {model.value!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def rmse_ratio_to_best(
        self,
        model: CandidateModel,
    ) -> float:
        """Return a model's held-out RMSE divided by the best held-out RMSE."""

        best_rmse = self.best_model.root_mean_squared_prediction_error
        if best_rmse <= 0.0:
            raise ModelComparisonError(
                "best held-out RMSE must be positive"
            )

        result = self.result_for(model)
        return result.root_mean_squared_prediction_error / best_rmse


def compare_shape_models(
    elapsed_seconds: Sequence[Decimal],
    observed_snr: Sequence[Decimal],
    *,
    gaussian_config: GaussianSearchConfig | None = None,
) -> ModelComparisonReport:
    """Compare shape models using deterministic leave-one-out prediction.

    Lower held-out error supports shape compatibility only. It does not identify
    a physical emitter, establish artificial origin, or prove transmission intent.
    """

    times = tuple(elapsed_seconds)
    observed = tuple(observed_snr)
    _validate_series(times, observed)

    results = tuple(
        _cross_validate_model(
            model,
            times,
            observed,
            gaussian_config=gaussian_config,
        )
        for model in CandidateModel
    )

    return ModelComparisonReport(
        elapsed_seconds=times,
        observed_snr=observed,
        results=results,
    )
  def _cross_validate_model(
    model: CandidateModel,
    elapsed_seconds: tuple[Decimal, ...],
    observed_snr: tuple[Decimal, ...],
    *,
    gaussian_config: GaussianSearchConfig | None,
) -> ModelCrossValidation:
    folds: list[CrossValidationFold] = []

    for held_out_index in range(len(observed_snr)):
        training_times = tuple(
            value
            for index, value in enumerate(elapsed_seconds)
            if index != held_out_index
        )
        training_observed = tuple(
            value
            for index, value in enumerate(observed_snr)
            if index != held_out_index
        )

        predicted = _predict_held_out(
            model,
            training_times,
            training_observed,
            elapsed_seconds[held_out_index],
            gaussian_config=gaussian_config,
        )

        residual = float(observed_snr[held_out_index]) - predicted
        folds.append(
            CrossValidationFold(
                held_out_index=held_out_index,
                elapsed_seconds=elapsed_seconds[held_out_index],
                observed_snr=observed_snr[held_out_index],
                predicted_snr=predicted,
                residual_snr=residual,
                squared_error=residual**2,
            )
        )

    prediction_sum_squares = sum(
        fold.squared_error for fold in folds
    )
    root_mean_squared_prediction_error = math.sqrt(
        prediction_sum_squares / len(folds)
    )
    mean_absolute_prediction_error = sum(
        abs(fold.residual_snr) for fold in folds
    ) / len(folds)

    return ModelCrossValidation(
        model=model,
        parameter_count=model.parameter_count,
        folds=tuple(folds),
        prediction_sum_squares=prediction_sum_squares,
        root_mean_squared_prediction_error=(
            root_mean_squared_prediction_error
        ),
        mean_absolute_prediction_error=(
            mean_absolute_prediction_error
        ),
    )


def _predict_held_out(
    model: CandidateModel,
    training_times: tuple[Decimal, ...],
    training_observed: tuple[Decimal, ...],
    target_time: Decimal,
    *,
    gaussian_config: GaussianSearchConfig | None,
) -> float:
    if model is CandidateModel.CONSTANT:
        return sum(
            float(value) for value in training_observed
        ) / len(training_observed)

    if model is CandidateModel.AFFINE:
        return _polynomial_prediction(
            training_times,
            training_observed,
            target_time,
            degree=1,
        )

    if model is CandidateModel.QUADRATIC:
        return _polynomial_prediction(
            training_times,
            training_observed,
            target_time,
            degree=2,
        )

    fit = fit_gaussian_series(
        training_times,
        training_observed,
        config=gaussian_config,
    )
    return fit.predict(float(target_time))


def _polynomial_prediction(
    training_times: tuple[Decimal, ...],
    training_observed: tuple[Decimal, ...],
    target_time: Decimal,
    *,
    degree: int,
) -> float:
    if degree not in {1, 2}:
        raise ModelComparisonError(
            "polynomial degree must be 1 or 2"
        )
    if len(training_times) < degree + 1:
        raise ModelComparisonError(
            "insufficient training observations for polynomial degree"
        )

    times = tuple(float(value) for value in training_times)
    observed = tuple(float(value) for value in training_observed)

    center = sum(times) / len(times)
    scale = max(abs(value - center) for value in times)
    if scale <= 0.0 or not math.isfinite(scale):
        raise ModelComparisonError(
            "training times do not span a finite interval"
        )

    normalized_times = tuple(
        (value - center) / scale for value in times
    )
    target = (float(target_time) - center) / scale
    size = degree + 1

    matrix = [
        [
            sum(
                value ** (row + column)
                for value in normalized_times
            )
            for column in range(size)
        ]
        for row in range(size)
    ]

    right_hand_side = [
        sum(
            observed_value * normalized_time**row
            for observed_value, normalized_time in zip(
                observed,
                normalized_times,
                strict=True,
            )
        )
        for row in range(size)
    ]

    coefficients = _solve_linear_system(
        matrix,
        right_hand_side,
    )

    prediction = sum(
        coefficient * target**power
        for power, coefficient in enumerate(coefficients)
    )
    if not math.isfinite(prediction):
        raise ModelComparisonError(
            "polynomial prediction must be finite"
        )

    return prediction


def _solve_linear_system(
    matrix: list[list[float]],
    right_hand_side: list[float],
) -> tuple[float, ...]:
    size = len(matrix)

    if size == 0 or len(right_hand_side) != size:
        raise ModelComparisonError(
            "linear system dimensions are invalid"
        )
    if any(len(row) != size for row in matrix):
        raise ModelComparisonError(
            "linear system matrix must be square"
        )

    augmented = [
        [*row, right_hand_side[index]]
        for index, row in enumerate(matrix)
    ]

    for column in range(size):
        pivot_row = max(
            range(column, size),
            key=lambda row_index: abs(
                augmented[row_index][column]
            ),
        )
        pivot_value = augmented[pivot_row][column]

        if abs(pivot_value) <= _SINGULAR_TOLERANCE:
            raise ModelComparisonError(
                "linear system is singular"
            )

        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor

        for row_index in range(size):
            if row_index == column:
                continue

            factor = augmented[row_index][column]
            for index in range(column, size + 1):
                augmented[row_index][index] -= (
                    factor * augmented[column][index]
                )

    solution = tuple(
        augmented[index][-1] for index in range(size)
    )
    if any(not math.isfinite(value) for value in solution):
        raise ModelComparisonError(
            "linear system solution must be finite"
        )

    return solution


def _validate_series(
    elapsed_seconds: tuple[Decimal, ...],
    observed_snr: tuple[Decimal, ...],
) -> None:
    if len(elapsed_seconds) != len(observed_snr):
        raise ModelComparisonError(
            "elapsed_seconds and observed_snr must have equal length"
        )
    if len(elapsed_seconds) < _MINIMUM_OBSERVATIONS:
        raise ModelComparisonError(
            "at least four observations are required"
        )
    if any(
        not value.is_finite() or value < 0
        for value in elapsed_seconds
    ):
        raise ModelComparisonError(
            "elapsed_seconds must be non-negative and finite"
        )
    if any(
        current <= previous
        for previous, current in pairwise(elapsed_seconds)
    ):
        raise ModelComparisonError(
            "elapsed_seconds must be strictly increasing"
        )
    if any(
        not value.is_finite() or value < 0
        for value in observed_snr
    ):
        raise ModelComparisonError(
            "observed_snr must be non-negative and finite"
        )
    if len(set(observed_snr)) == 1:
        raise ModelComparisonError(
            "observed_snr values must not all be equal"
        )
