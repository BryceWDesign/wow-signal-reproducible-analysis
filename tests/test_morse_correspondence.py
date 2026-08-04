from decimal import Decimal

import pytest

from wow_signal_analysis.morse import (
    MorseCategory,
    MorseRegistry,
    MorseSymbol,
)
from wow_signal_analysis.morse_correspondence import (
    MorseCorrespondenceError,
    MorseCorrespondenceReport,
    MorsePolarity,
    SequenceDirection,
    ThresholdMorseComparison,
    analyze_threshold_morse,
    morse_pattern_from_mask,
)
from wow_signal_analysis.thresholds import enumerate_threshold_cases


def _canonical_values() -> tuple[Decimal, ...]:
    return (
        Decimal("6.5"),
        Decimal("14.5"),
        Decimal("26.5"),
        Decimal("30.5"),
        Decimal("19.5"),
        Decimal("5.5"),
    )


def _symbol(
    glyph: str,
    pattern: str,
    label: str,
) -> MorseSymbol:
    return MorseSymbol(
        glyph=glyph,
        category=MorseCategory.PUNCTUATION,
        pattern=pattern,
        section="1.1.3",
        label=label,
    )


def _registry() -> MorseRegistry:
    return MorseRegistry(
        schema_version=1,
        standard_id="ITU-R M.1677-1",
        title="Test registry",
        source_url="https://example.test/morse",
        scope_note="Symbols needed to verify exhaustive correspondence behavior.",
        symbols=(
            _symbol("'", ".----.", "apostrophe"),
            _symbol("-", "-....-", "hyphen"),
            _symbol("?", "..--..", "question mark"),
            _symbol(",", "--..--", "comma"),
        ),
    )


def test_analysis_covers_every_threshold_direction_and_polarity() -> None:
    report = analyze_threshold_morse(_canonical_values(), _registry())

    assert len(report.threshold_cases) == 7
    assert len(report.comparisons) == 28
    assert {
        (comparison.cut_index, comparison.direction, comparison.polarity)
        for comparison in report.comparisons
    } == {
        (case.cut_index, direction, polarity)
        for case in report.threshold_cases
        for direction in SequenceDirection
        for polarity in MorsePolarity
    }


def test_question_mark_requires_the_declared_central_threshold_and_polarity() -> None:
    report = analyze_threshold_morse(_canonical_values(), _registry())
    matches = report.comparisons_for_glyph("?")

    assert len(matches) == 2
    assert {match.cut_index for match in matches} == {4}
    assert {match.direction for match in matches} == set(SequenceDirection)
    assert {match.polarity for match in matches} == {MorsePolarity.HIGH_IS_DASH}
    assert {match.lower_bound_inclusive for match in matches} == {Decimal("19.5")}
    assert {match.upper_bound_exclusive for match in matches} == {Decimal("26.5")}
    assert {match.binary_pattern for match in matches} == {"001100"}
    assert {match.morse_pattern for match in matches} == {"..--.."}


def test_opposite_polarity_maps_the_same_partition_to_comma() -> None:
    report = analyze_threshold_morse(_canonical_values(), _registry())
    matches = report.comparisons_for_glyph(",")

    assert len(matches) == 2
    assert {match.cut_index for match in matches} == {4}
    assert {match.polarity for match in matches} == {MorsePolarity.HIGH_IS_DOT}
    assert {match.binary_pattern for match in matches} == {"001100"}
    assert {match.morse_pattern for match in matches} == {"--..--"}


def test_another_threshold_also_matches_official_punctuation() -> None:
    report = analyze_threshold_morse(_canonical_values(), _registry())

    apostrophes = report.comparisons_for_glyph("'")
    hyphens = report.comparisons_for_glyph("-")

    assert len(apostrophes) == 2
    assert len(hyphens) == 2
    assert {match.cut_index for match in apostrophes} == {2}
    assert {match.cut_index for match in hyphens} == {2}
    assert {match.binary_pattern for match in apostrophes} == {"011110"}
    assert {match.binary_pattern for match in hyphens} == {"011110"}


def test_report_exposes_all_matches_instead_of_selecting_question_only() -> None:
    report = analyze_threshold_morse(_canonical_values(), _registry())

    assert len(report.assigned_comparisons) == 8
    assert tuple(symbol.glyph for symbol in report.unique_assigned_symbols) == ("'", "-", "?", ",")


def test_nonpalindromic_partition_changes_when_read_in_reverse() -> None:
    report = analyze_threshold_morse(_canonical_values(), _registry())
    forward = next(
        comparison
        for comparison in report.comparisons
        if comparison.cut_index == 3
        and comparison.direction is SequenceDirection.FORWARD
        and comparison.polarity is MorsePolarity.HIGH_IS_DASH
    )
    reverse = next(
        comparison
        for comparison in report.comparisons
        if comparison.cut_index == 3
        and comparison.direction is SequenceDirection.REVERSE
        and comparison.polarity is MorsePolarity.HIGH_IS_DASH
    )

    assert forward.binary_pattern == "001110"
    assert reverse.binary_pattern == "011100"
    assert forward.morse_pattern == "..---."
    assert reverse.morse_pattern == ".---.."
    assert not forward.is_assigned
    assert not reverse.is_assigned


def test_mask_rendering_requires_explicit_polarity() -> None:
    mask = (False, False, True, True, False, False)

    assert morse_pattern_from_mask(mask, MorsePolarity.HIGH_IS_DASH) == "..--.."
    assert morse_pattern_from_mask(mask, MorsePolarity.HIGH_IS_DOT) == "--..--"

    with pytest.raises(MorseCorrespondenceError, match="must not be empty"):
        morse_pattern_from_mask((), MorsePolarity.HIGH_IS_DASH)


def test_comparison_rejects_a_pattern_inconsistent_with_its_polarity() -> None:
    with pytest.raises(MorseCorrespondenceError, match="does not match"):
        ThresholdMorseComparison(
            cut_index=4,
            direction=SequenceDirection.FORWARD,
            polarity=MorsePolarity.HIGH_IS_DASH,
            lower_bound_inclusive=Decimal("19.5"),
            upper_bound_exclusive=Decimal("26.5"),
            binary_pattern="001100",
            morse_pattern="--..--",
            matched_symbols=(),
        )


def test_report_rejects_an_incomplete_comparison_space() -> None:
    values = _canonical_values()
    cases = enumerate_threshold_cases(values)

    with pytest.raises(MorseCorrespondenceError, match="forward and reverse"):
        MorseCorrespondenceReport(
            values=values,
            threshold_cases=cases,
            comparisons=(),
        )
