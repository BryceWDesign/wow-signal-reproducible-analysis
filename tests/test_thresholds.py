from decimal import Decimal

import pytest

from wow_signal_analysis.measurements import canonical_wow_samples
from wow_signal_analysis.thresholds import (
    BitPolarity,
    ThresholdCase,
    ThresholdError,
    apply_threshold,
    enumerate_threshold_cases,
    find_threshold_case,
)


def _canonical_values() -> tuple[Decimal, ...]:
    return tuple(sample.intensity.midpoint_snr for sample in canonical_wow_samples())


def test_canonical_threshold_enumeration_is_exhaustive_and_ordered() -> None:
    cases = enumerate_threshold_cases(_canonical_values())

    assert len(cases) == 7
    assert tuple(case.cut_index for case in cases) == tuple(range(7))
    assert tuple(case.bit_pattern() for case in cases) == (
        "111111",
        "111110",
        "011110",
        "001110",
        "001100",
        "000100",
        "000000",
    )


def test_canonical_threshold_intervals_are_explicit() -> None:
    cases = enumerate_threshold_cases(_canonical_values())

    assert tuple((case.lower_bound_inclusive, case.upper_bound_exclusive) for case in cases) == (
        (None, Decimal("5.5")),
        (Decimal("5.5"), Decimal("6.5")),
        (Decimal("6.5"), Decimal("14.5")),
        (Decimal("14.5"), Decimal("19.5")),
        (Decimal("19.5"), Decimal("26.5")),
        (Decimal("26.5"), Decimal("30.5")),
        (Decimal("30.5"), None),
    )
    assert tuple(case.representative_threshold for case in cases) == (
        Decimal("4.5"),
        Decimal("6.0"),
        Decimal("10.5"),
        Decimal("17.0"),
        Decimal("23.0"),
        Decimal("28.5"),
        Decimal("30.5"),
    )


def test_central_core_partition_is_palindromic_under_both_polarities() -> None:
    cases = enumerate_threshold_cases(_canonical_values())
    central_core = cases[4]

    assert central_core.high_mask == (False, False, True, True, False, False)
    assert central_core.high_count == 2
    assert central_core.low_count == 4
    assert central_core.bit_pattern(BitPolarity.HIGH_IS_ONE) == "001100"
    assert central_core.reverse_bit_pattern(BitPolarity.HIGH_IS_ONE) == "001100"
    assert central_core.bit_pattern(BitPolarity.HIGH_IS_ZERO) == "110011"
    assert central_core.reverse_bit_pattern(BitPolarity.HIGH_IS_ZERO) == "110011"


def test_reverse_patterns_are_derived_without_reordering_threshold_levels() -> None:
    cases = enumerate_threshold_cases(_canonical_values())

    assert tuple(case.reverse_bit_pattern() for case in cases) == (
        "111111",
        "011111",
        "011110",
        "011100",
        "001100",
        "001000",
        "000000",
    )


def test_find_threshold_case_obeys_half_open_boundary_semantics() -> None:
    cases = enumerate_threshold_cases(_canonical_values())

    assert find_threshold_case(cases, Decimal("19.499")).cut_index == 3
    assert find_threshold_case(cases, Decimal("19.5")).cut_index == 4
    assert find_threshold_case(cases, Decimal("26.499")).cut_index == 4
    assert find_threshold_case(cases, Decimal("26.5")).cut_index == 5


def test_representative_threshold_reproduces_each_enumerated_mask() -> None:
    values = _canonical_values()

    for case in enumerate_threshold_cases(values):
        assert apply_threshold(values, case.representative_threshold) == case.high_mask


def test_duplicate_measurement_levels_do_not_create_duplicate_partitions() -> None:
    values = (
        Decimal("1"),
        Decimal("2"),
        Decimal("2"),
        Decimal("1"),
    )
    cases = enumerate_threshold_cases(values)

    assert tuple(case.bit_pattern() for case in cases) == (
        "1111",
        "0110",
        "0000",
    )
    assert tuple(case.high_count for case in cases) == (4, 2, 0)


@pytest.mark.parametrize(
    "values",
    [
        (),
        (Decimal("NaN"),),
        (Decimal("Infinity"),),
    ],
)
def test_threshold_analysis_rejects_empty_or_nonfinite_values(
    values: tuple[Decimal, ...],
) -> None:
    with pytest.raises(ThresholdError):
        enumerate_threshold_cases(values)


@pytest.mark.parametrize("threshold", [Decimal("NaN"), Decimal("Infinity")])
def test_threshold_application_rejects_nonfinite_thresholds(
    threshold: Decimal,
) -> None:
    with pytest.raises(ThresholdError, match="threshold must be finite"):
        apply_threshold(_canonical_values(), threshold)


def test_find_threshold_case_requires_one_complete_nonoverlapping_partition() -> None:
    cases = enumerate_threshold_cases(_canonical_values())

    with pytest.raises(ThresholdError, match="threshold must be finite"):
        find_threshold_case(cases, Decimal("NaN"))

    with pytest.raises(ThresholdError, match="found 0"):
        find_threshold_case(cases[:-1], Decimal("100"))


def test_threshold_case_rejects_invalid_direct_construction() -> None:
    with pytest.raises(ThresholdError, match="high_mask"):
        ThresholdCase(
            cut_index=0,
            lower_bound_inclusive=None,
            upper_bound_exclusive=Decimal("1"),
            representative_threshold=Decimal("0"),
            high_mask=(),
        )

    with pytest.raises(ThresholdError, match="strictly ordered"):
        ThresholdCase(
            cut_index=1,
            lower_bound_inclusive=Decimal("2"),
            upper_bound_exclusive=Decimal("1"),
            representative_threshold=Decimal("1.5"),
            high_mask=(True,),
        )

    with pytest.raises(ThresholdError, match="outside the interval"):
        ThresholdCase(
            cut_index=1,
            lower_bound_inclusive=Decimal("1"),
            upper_bound_exclusive=Decimal("2"),
            representative_threshold=Decimal("2"),
            high_mask=(True,),
        )
