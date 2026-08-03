from decimal import Decimal

import pytest

from wow_signal_analysis.measurements import canonical_wow_samples
from wow_signal_analysis.profile import (
    AdjacentChange,
    MirrorComparison,
    ProfileError,
    Trend,
    analyze_midpoint_values,
    analyze_samples,
)


def test_canonical_profile_preserves_forward_and_reverse_values() -> None:
    profile = analyze_samples(canonical_wow_samples())

    assert profile.values == (
        Decimal("6.5"),
        Decimal("14.5"),
        Decimal("26.5"),
        Decimal("30.5"),
        Decimal("19.5"),
        Decimal("5.5"),
    )
    assert profile.reverse_values == tuple(reversed(profile.values))


def test_canonical_profile_has_one_strict_interior_peak() -> None:
    profile = analyze_samples(canonical_wow_samples())

    assert profile.peak_index == 3
    assert profile.peak_value == Decimal("30.5")
    assert profile.peak_count == 1
    assert profile.has_unique_peak
    assert profile.is_strict_single_peak


def test_canonical_profile_reports_all_adjacent_changes() -> None:
    profile = analyze_samples(canonical_wow_samples())

    assert tuple(change.delta_snr for change in profile.changes) == (
        Decimal("8.0"),
        Decimal("12.0"),
        Decimal("4.0"),
        Decimal("-11.0"),
        Decimal("-14.0"),
    )
    assert profile.trend_pattern == (
        Trend.RISING,
        Trend.RISING,
        Trend.RISING,
        Trend.FALLING,
        Trend.FALLING,
    )
    assert profile.trend_signature == "+++--"


def test_canonical_profile_exposes_mirror_residuals_without_claiming_symmetry() -> None:
    profile = analyze_samples(canonical_wow_samples())

    assert tuple(pair.signed_difference for pair in profile.mirror_comparisons) == (
        Decimal("1.0"),
        Decimal("-5.0"),
        Decimal("-4.0"),
    )
    assert tuple(pair.absolute_difference for pair in profile.mirror_comparisons) == (
        Decimal("1.0"),
        Decimal("5.0"),
        Decimal("4.0"),
    )
    assert not profile.is_exact_palindrome


def test_reverse_profile_is_analyzed_independently() -> None:
    forward = analyze_samples(canonical_wow_samples())
    reverse = analyze_midpoint_values(forward.reverse_values)

    assert reverse.trend_signature == "++---"
    assert reverse.peak_index == 2
    assert reverse.is_strict_single_peak
    assert tuple(pair.signed_difference for pair in reverse.mirror_comparisons) == (
        Decimal("-1.0"),
        Decimal("5.0"),
        Decimal("4.0"),
    )


def test_flat_or_repeated_peak_sequences_are_not_strict_single_peaks() -> None:
    flat = analyze_midpoint_values(
        (
            Decimal("1"),
            Decimal("1"),
            Decimal("2"),
        )
    )
    repeated_peak = analyze_midpoint_values(
        (
            Decimal("1"),
            Decimal("2"),
            Decimal("2"),
            Decimal("1"),
        )
    )

    assert flat.trend_signature == "0+"
    assert not flat.is_strict_single_peak

    assert repeated_peak.peak_count == 2
    assert not repeated_peak.has_unique_peak
    assert not repeated_peak.is_strict_single_peak
    assert repeated_peak.is_exact_palindrome


def test_single_measurement_profile_is_valid_but_not_a_strict_peak() -> None:
    profile = analyze_midpoint_values((Decimal("7.5"),))

    assert profile.changes == ()
    assert profile.mirror_comparisons == ()
    assert profile.peak_index == 0
    assert profile.has_unique_peak
    assert not profile.is_strict_single_peak
    assert profile.is_exact_palindrome


@pytest.mark.parametrize(
    "values",
    [
        (),
        (Decimal("NaN"),),
        (Decimal("Infinity"),),
    ],
)
def test_analysis_rejects_empty_or_nonfinite_sequences(
    values: tuple[Decimal, ...],
) -> None:
    with pytest.raises(ProfileError):
        analyze_midpoint_values(values)


def test_adjacent_change_rejects_inconsistent_structure() -> None:
    with pytest.raises(ProfileError, match="immediately follow"):
        AdjacentChange(
            from_index=0,
            to_index=2,
            delta_snr=Decimal("1"),
            trend=Trend.RISING,
        )

    with pytest.raises(ProfileError, match="does not match"):
        AdjacentChange(
            from_index=0,
            to_index=1,
            delta_snr=Decimal("-1"),
            trend=Trend.RISING,
        )


def test_mirror_comparison_rejects_inconsistent_difference() -> None:
    with pytest.raises(ProfileError, match="left_snr minus right_snr"):
        MirrorComparison(
            left_index=0,
            right_index=1,
            left_snr=Decimal("3"),
            right_snr=Decimal("1"),
            signed_difference=Decimal("1"),
        )
