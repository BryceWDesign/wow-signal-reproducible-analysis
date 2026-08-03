from decimal import Decimal
from fractions import Fraction

import pytest

from wow_signal_analysis.morse import MorseCategory, MorseRegistry, MorseSymbol
from wow_signal_analysis.morse_correspondence import MorsePolarity
from wow_signal_analysis.null_model import (
    GlyphComparisonCount,
    NullModelError,
    PermutationOutcome,
    analyze_permutation_null,
    iter_unique_permutations,
    unique_permutation_count,
)


def _canonical_values() -> tuple[Decimal, ...]:
    return (
        Decimal("6.5"),
        Decimal("14.5"),
        Decimal("26.5"),
        Decimal("30.5"),
        Decimal("19.5"),
        Decimal("5.5"),
    )


def _symbol(glyph: str, pattern: str, label: str) -> MorseSymbol:
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
        title="Null-model test registry",
        source_url="https://example.test/morse",
        scope_note="Symbols needed to verify exact permutation frequencies.",
        symbols=(
            _symbol("'", ".----.", "apostrophe"),
            _symbol("-", "-....-", "hyphen"),
            _symbol("?", "..--..", "question mark"),
            _symbol(",", "--..--", "comma"),
        ),
    )


def test_canonical_null_enumerates_all_six_factorial_orderings() -> None:
    report = analyze_permutation_null(
        _canonical_values(),
        _registry(),
        glyphs=("?", ","),
    )

    assert report.total_unique_sequences == 720
    assert report.comparisons_per_sequence == 28
    assert report.total_comparison_count == 20_160
    assert tuple(outcome.permutation_index for outcome in report.outcomes) == tuple(
        range(720)
    )
    assert len({outcome.values for outcome in report.outcomes}) == 720


def test_question_and_comma_have_exact_exhaustive_frequencies() -> None:
    report = analyze_permutation_null(
        _canonical_values(),
        _registry(),
        glyphs=("?", ","),
    )
    question = report.summary_for_glyph("?")
    comma = report.summary_for_glyph(",")

    assert question.matched_sequence_count == 96
    assert question.sequence_fraction == Fraction(2, 15)
    assert question.matched_comparison_count == 192
    assert question.comparison_fraction == Fraction(1, 105)

    assert comma.matched_sequence_count == 96
    assert comma.sequence_fraction == Fraction(2, 15)
    assert comma.matched_comparison_count == 192
    assert comma.comparison_fraction == Fraction(1, 105)


def test_question_or_comma_family_matches_the_same_permutations() -> None:
    report = analyze_permutation_null(
        _canonical_values(),
        _registry(),
        glyphs=("?", ","),
    )

    assert report.sequence_count_matching_any(("?", ",")) == 96
    assert report.sequence_fraction_matching_any(("?", ",")) == Fraction(2, 15)


def test_predeclared_high_is_dash_polarity_halves_the_sequence_frequency() -> None:
    report = analyze_permutation_null(
        _canonical_values(),
        _registry(),
        glyphs=("?",),
        polarities=(MorsePolarity.HIGH_IS_DASH,),
    )
    question = report.summary_for_glyph("?")

    assert report.comparisons_per_sequence == 14
    assert report.total_comparison_count == 10_080
    assert question.matched_sequence_count == 48
    assert question.sequence_fraction == Fraction(1, 15)
    assert question.matched_comparison_count == 96
    assert question.comparison_fraction == Fraction(1, 105)


def test_observed_ordering_has_two_directional_matches_per_polarity() -> None:
    values = _canonical_values()
    report = analyze_permutation_null(values, _registry(), glyphs=("?", ","))
    observed = report.outcome_for_values(values)

    assert observed.matched_glyphs == ("?", ",")
    assert observed.comparison_count_for_glyph("?") == 2
    assert observed.comparison_count_for_glyph(",") == 2
    assert observed.matches_glyph("?")
    assert observed.matches_glyph(",")


def test_unique_permutation_enumerator_handles_duplicate_values_exactly() -> None:
    values = (Decimal("1"), Decimal("1"), Decimal("2"))

    assert unique_permutation_count(values) == 3
    assert tuple(iter_unique_permutations(values)) == (
        (Decimal("1"), Decimal("1"), Decimal("2")),
        (Decimal("1"), Decimal("2"), Decimal("1")),
        (Decimal("2"), Decimal("1"), Decimal("1")),
    )


def test_exact_enumeration_limit_fails_before_large_search() -> None:
    with pytest.raises(NullModelError, match="exceeding the configured limit"):
        analyze_permutation_null(
            _canonical_values(),
            _registry(),
            glyphs=("?",),
            max_unique_sequences=719,
        )


@pytest.mark.parametrize(
    "glyphs",
    [
        (),
        ("?", "?"),
        ("$",),
    ],
)
def test_glyph_selection_fails_closed(glyphs: tuple[str, ...]) -> None:
    with pytest.raises(NullModelError):
        analyze_permutation_null(
            _canonical_values(),
            _registry(),
            glyphs=glyphs,
        )


def test_family_queries_require_symbols_in_the_completed_report() -> None:
    report = analyze_permutation_null(
        _canonical_values(),
        _registry(),
        glyphs=("?",),
    )

    with pytest.raises(NullModelError, match="must not be empty"):
        report.sequence_count_matching_any(())

    with pytest.raises(NullModelError, match="absent from this report"):
        report.sequence_fraction_matching_any((",",))


def test_outcome_validation_rejects_impossible_counts() -> None:
    with pytest.raises(NullModelError, match="cannot exceed"):
        PermutationOutcome(
            permutation_index=0,
            values=(Decimal("1"),),
            total_comparisons=1,
            glyph_counts=(
                GlyphComparisonCount(glyph="?", comparison_count=2),
            ),
        )


def test_input_validation_rejects_nonfinite_values_and_bad_limits() -> None:
    with pytest.raises(NullModelError, match="must be finite"):
        analyze_permutation_null(
            (Decimal("NaN"),),
            _registry(),
            glyphs=("?",),
        )

    with pytest.raises(NullModelError, match="must be positive"):
        analyze_permutation_null(
            _canonical_values(),
            _registry(),
            glyphs=("?",),
            max_unique_sequences=0,
        )

    with pytest.raises(NullModelError, match="must be an integer"):
        analyze_permutation_null(
            _canonical_values(),
            _registry(),
            glyphs=("?",),
            max_unique_sequences=True,
        )

    with pytest.raises(NullModelError, match="sequence direction"):
        analyze_permutation_null(
            _canonical_values(),
            _registry(),
            glyphs=("?",),
            directions=(),
        )

    with pytest.raises(NullModelError, match="Morse polarity"):
        analyze_permutation_null(
            _canonical_values(),
            _registry(),
            glyphs=("?",),
            polarities=(),
        )
