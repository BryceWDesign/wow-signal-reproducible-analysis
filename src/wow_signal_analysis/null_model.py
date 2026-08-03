"""Exact permutation controls for threshold-to-Morse correspondences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import factorial

from wow_signal_analysis.morse import MorseError, MorseRegistry
from wow_signal_analysis.morse_correspondence import (
    MorsePolarity,
    SequenceDirection,
    analyze_threshold_morse,
)


class NullModelError(ValueError):
    """Raised when an exact permutation control cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class GlyphComparisonCount:
    """Number of declared comparisons matching one glyph in one permutation."""

    glyph: str
    comparison_count: int

    def __post_init__(self) -> None:
        if len(self.glyph) != 1 or self.glyph.isspace():
            raise NullModelError(
                "glyph must contain exactly one non-whitespace character"
            )
        if self.comparison_count <= 0:
            raise NullModelError("comparison_count must be positive")


@dataclass(frozen=True, slots=True)
class PermutationOutcome:
    """Morse matches produced by one unique temporal ordering of the values."""

    permutation_index: int
    values: tuple[Decimal, ...]
    total_comparisons: int
    glyph_counts: tuple[GlyphComparisonCount, ...]

    def __post_init__(self) -> None:
        if self.permutation_index < 0:
            raise NullModelError("permutation_index must be non-negative")
        if not self.values:
            raise NullModelError("permutation values must not be empty")
        if any(not value.is_finite() for value in self.values):
            raise NullModelError("permutation values must be finite")
        if self.total_comparisons <= 0:
            raise NullModelError("total_comparisons must be positive")

        glyphs = tuple(item.glyph for item in self.glyph_counts)
        if len(set(glyphs)) != len(glyphs):
            raise NullModelError("glyph_counts must not contain duplicate glyphs")
        if any(
            item.comparison_count > self.total_comparisons
            for item in self.glyph_counts
        ):
            raise NullModelError(
                "glyph comparison_count cannot exceed total_comparisons"
            )

    @property
    def matched_glyphs(self) -> tuple[str, ...]:
        """Return every selected glyph matched by this permutation."""

        return tuple(item.glyph for item in self.glyph_counts)

    def comparison_count_for_glyph(self, glyph: str) -> int:
        """Return the comparison count for a glyph, or zero when unmatched."""

        return sum(
            item.comparison_count
            for item in self.glyph_counts
            if item.glyph == glyph
        )

    def matches_glyph(self, glyph: str) -> bool:
        """Return whether at least one declared comparison matches the glyph."""

        return self.comparison_count_for_glyph(glyph) > 0


@dataclass(frozen=True, slots=True)
class GlyphNullSummary:
    """Exact occurrence frequency for one glyph under temporal permutations."""

    glyph: str
    matched_sequence_count: int
    total_sequence_count: int
    matched_comparison_count: int
    total_comparison_count: int

    def __post_init__(self) -> None:
        if len(self.glyph) != 1 or self.glyph.isspace():
            raise NullModelError(
                "glyph must contain exactly one non-whitespace character"
            )
        if self.total_sequence_count <= 0:
            raise NullModelError("total_sequence_count must be positive")
        if not 0 <= self.matched_sequence_count <= self.total_sequence_count:
            raise NullModelError(
                "matched_sequence_count must be within the sequence total"
            )
        if self.total_comparison_count <= 0:
            raise NullModelError("total_comparison_count must be positive")
        if not 0 <= self.matched_comparison_count <= self.total_comparison_count:
            raise NullModelError(
                "matched_comparison_count must be within the comparison total"
            )

    @property
    def sequence_fraction(self) -> Fraction:
        """Return the exact fraction of permutations matching this glyph."""

        return Fraction(self.matched_sequence_count, self.total_sequence_count)

    @property
    def comparison_fraction(self) -> Fraction:
        """Return the exact fraction of declared comparisons matching this glyph."""

        return Fraction(
            self.matched_comparison_count,
            self.total_comparison_count,
        )


@dataclass(frozen=True, slots=True)
class PermutationNullReport:
    """Exact exchangeability control over all unique temporal permutations."""

    original_values: tuple[Decimal, ...]
    search_directions: tuple[SequenceDirection, ...]
    search_polarities: tuple[MorsePolarity, ...]
    comparisons_per_sequence: int
    outcomes: tuple[PermutationOutcome, ...]
    glyph_summaries: tuple[GlyphNullSummary, ...]

    def __post_init__(self) -> None:
        if not self.original_values:
            raise NullModelError("original_values must not be empty")
        if any(not value.is_finite() for value in self.original_values):
            raise NullModelError("original_values must be finite")
        if not self.search_directions:
            raise NullModelError("search_directions must not be empty")
        if len(set(self.search_directions)) != len(self.search_directions):
            raise NullModelError("search_directions must not contain duplicates")
        if not self.search_polarities:
            raise NullModelError("search_polarities must not be empty")
        if len(set(self.search_polarities)) != len(self.search_polarities):
            raise NullModelError("search_polarities must not contain duplicates")
        if self.comparisons_per_sequence <= 0:
            raise NullModelError("comparisons_per_sequence must be positive")
        if not self.outcomes:
            raise NullModelError("outcomes must not be empty")

        expected_indices = tuple(range(len(self.outcomes)))
        actual_indices = tuple(
            outcome.permutation_index for outcome in self.outcomes
        )
        if actual_indices != expected_indices:
            raise NullModelError(
                "outcome indices must be contiguous and zero-based"
            )

        expected_multiset = Counter(self.original_values)
        for outcome in self.outcomes:
            if Counter(outcome.values) != expected_multiset:
                raise NullModelError(
                    "every outcome must preserve the original value multiset"
                )
            if outcome.total_comparisons != self.comparisons_per_sequence:
                raise NullModelError(
                    "every outcome must use comparisons_per_sequence"
                )

        if len({outcome.values for outcome in self.outcomes}) != len(self.outcomes):
            raise NullModelError("outcomes must contain unique permutations")

        summary_glyphs = tuple(summary.glyph for summary in self.glyph_summaries)
        if len(set(summary_glyphs)) != len(summary_glyphs):
            raise NullModelError("glyph_summaries must not contain duplicate glyphs")

        summary_glyph_set = set(summary_glyphs)
        if any(
            glyph not in summary_glyph_set
            for outcome in self.outcomes
            for glyph in outcome.matched_glyphs
        ):
            raise NullModelError(
                "outcome glyphs must be declared by glyph_summaries"
            )

        for summary in self.glyph_summaries:
            if summary.total_sequence_count != self.total_unique_sequences:
                raise NullModelError(
                    "glyph summary sequence totals must match the report"
                )
            if summary.total_comparison_count != self.total_comparison_count:
                raise NullModelError(
                    "glyph summary comparison totals must match the report"
                )

            expected_sequence_count = sum(
                outcome.matches_glyph(summary.glyph)
                for outcome in self.outcomes
            )
            expected_comparison_count = sum(
                outcome.comparison_count_for_glyph(summary.glyph)
                for outcome in self.outcomes
            )

            if summary.matched_sequence_count != expected_sequence_count:
                raise NullModelError(
                    "glyph summary matched_sequence_count does not match outcomes"
                )
            if summary.matched_comparison_count != expected_comparison_count:
                raise NullModelError(
                    "glyph summary matched_comparison_count does not match outcomes"
                )

    @property
    def total_unique_sequences(self) -> int:
        """Return the number of unique temporal permutations evaluated."""

        return len(self.outcomes)

    @property
    def total_comparison_count(self) -> int:
        """Return the total declared threshold/direction/polarity comparisons."""

        return self.total_unique_sequences * self.comparisons_per_sequence

    def summary_for_glyph(self, glyph: str) -> GlyphNullSummary:
        """Return the unique null summary for one selected glyph."""

        matches = tuple(
            summary for summary in self.glyph_summaries if summary.glyph == glyph
        )
        if len(matches) != 1:
            raise NullModelError(
                f"expected one null summary for glyph {glyph!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def outcome_for_values(
        self,
        values: Sequence[Decimal],
    ) -> PermutationOutcome:
        """Return the unique evaluated outcome for an exact temporal ordering."""

        normalized = tuple(values)
        matches = tuple(
            outcome for outcome in self.outcomes if outcome.values == normalized
        )
        if len(matches) != 1:
            raise NullModelError(
                "expected one outcome for the requested value ordering, "
                f"found {len(matches)}"
            )
        return matches[0]

    def sequence_count_matching_any(self, glyphs: Sequence[str]) -> int:
        """Count permutations matching at least one glyph in a declared family."""

        selected = self._require_selected_glyphs(glyphs)
        return sum(
            any(outcome.matches_glyph(glyph) for glyph in selected)
            for outcome in self.outcomes
        )

    def sequence_fraction_matching_any(self, glyphs: Sequence[str]) -> Fraction:
        """Return the exact familywise fraction matching at least one glyph."""

        return Fraction(
            self.sequence_count_matching_any(glyphs),
            self.total_unique_sequences,
        )

    def _require_selected_glyphs(self, glyphs: Sequence[str]) -> tuple[str, ...]:
        selected = tuple(glyphs)
        if not selected:
            raise NullModelError("glyph family must not be empty")
        if len(set(selected)) != len(selected):
            raise NullModelError("glyph family must not contain duplicates")

        available = {summary.glyph for summary in self.glyph_summaries}
        unknown = tuple(glyph for glyph in selected if glyph not in available)
        if unknown:
            raise NullModelError(
                f"glyph family contains symbols absent from this report: {unknown}"
            )
        return selected
      def analyze_permutation_null(
    values: Sequence[Decimal],
    registry: MorseRegistry,
    *,
    glyphs: Sequence[str] | None = None,
    directions: Sequence[SequenceDirection] = tuple(SequenceDirection),
    polarities: Sequence[MorsePolarity] = tuple(MorsePolarity),
    max_unique_sequences: int = 100_000,
) -> PermutationNullReport:
    """Evaluate Morse matches under every unique temporal permutation.

    The control preserves the observed amplitudes and assumes all unique temporal
    orderings are exchangeable. It reports occurrence frequencies; it does not
    infer transmission intent or assign a linguistic meaning to the measurements.
    """

    normalized = _normalize_values(values)
    selected_glyphs = _normalize_glyphs(registry, glyphs)
    selected_directions = _normalize_directions(directions)
    selected_polarities = _normalize_polarities(polarities)
    unique_count = unique_permutation_count(normalized)

    if not isinstance(max_unique_sequences, int) or isinstance(
        max_unique_sequences,
        bool,
    ):
        raise NullModelError("max_unique_sequences must be an integer")
    if max_unique_sequences <= 0:
        raise NullModelError("max_unique_sequences must be positive")
    if unique_count > max_unique_sequences:
        raise NullModelError(
            f"exact null requires {unique_count} unique permutations, "
            f"exceeding the configured limit of {max_unique_sequences}"
        )

    sequence_counts = {glyph: 0 for glyph in selected_glyphs}
    comparison_counts = {glyph: 0 for glyph in selected_glyphs}
    outcomes: list[PermutationOutcome] = []
    comparisons_per_sequence: int | None = None

    for permutation_index, permutation in enumerate(
        iter_unique_permutations(normalized)
    ):
        correspondence = analyze_threshold_morse(permutation, registry)
        searched_comparisons = tuple(
            comparison
            for comparison in correspondence.comparisons
            if comparison.direction in selected_directions
            and comparison.polarity in selected_polarities
        )
        current_comparisons = len(searched_comparisons)

        if comparisons_per_sequence is None:
            comparisons_per_sequence = current_comparisons
        elif current_comparisons != comparisons_per_sequence:
            raise NullModelError(
                "comparison count changed across permutations of one value multiset"
            )

        glyph_counts: list[GlyphComparisonCount] = []
        for glyph in selected_glyphs:
            count = sum(
                glyph in comparison.matched_glyphs
                for comparison in searched_comparisons
            )
            if count > 0:
                glyph_counts.append(
                    GlyphComparisonCount(
                        glyph=glyph,
                        comparison_count=count,
                    )
                )
                sequence_counts[glyph] += 1
                comparison_counts[glyph] += count

        outcomes.append(
            PermutationOutcome(
                permutation_index=permutation_index,
                values=permutation,
                total_comparisons=current_comparisons,
                glyph_counts=tuple(glyph_counts),
            )
        )

    if comparisons_per_sequence is None:
        raise NullModelError("exact permutation enumeration produced no outcomes")
    if len(outcomes) != unique_count:
        raise NullModelError(
            f"expected {unique_count} unique permutations, produced {len(outcomes)}"
        )

    total_comparisons = unique_count * comparisons_per_sequence
    summaries = tuple(
        GlyphNullSummary(
            glyph=glyph,
            matched_sequence_count=sequence_counts[glyph],
            total_sequence_count=unique_count,
            matched_comparison_count=comparison_counts[glyph],
            total_comparison_count=total_comparisons,
        )
        for glyph in selected_glyphs
    )

    return PermutationNullReport(
        original_values=normalized,
        search_directions=selected_directions,
        search_polarities=selected_polarities,
        comparisons_per_sequence=comparisons_per_sequence,
        outcomes=tuple(outcomes),
        glyph_summaries=summaries,
    )


def unique_permutation_count(values: Sequence[Decimal]) -> int:
    """Return the exact number of unique permutations of a finite multiset."""

    normalized = _normalize_values(values)
    denominator = 1

    for count in Counter(normalized).values():
        denominator *= factorial(count)

    return factorial(len(normalized)) // denominator


def iter_unique_permutations(
    values: Sequence[Decimal],
) -> Iterator[tuple[Decimal, ...]]:
    """Yield unique permutations in deterministic ascending-value order."""

    normalized = _normalize_values(values)
    counts = Counter(normalized)
    ordered_values = tuple(sorted(counts))
    permutation: list[Decimal] = []

    def visit() -> Iterator[tuple[Decimal, ...]]:
        if len(permutation) == len(normalized):
            yield tuple(permutation)
            return

        for value in ordered_values:
            if counts[value] == 0:
                continue

            counts[value] -= 1
            permutation.append(value)
            yield from visit()
            permutation.pop()
            counts[value] += 1

    yield from visit()


def _normalize_values(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    normalized = tuple(values)
    if not normalized:
        raise NullModelError("measurement sequence must not be empty")
    if any(not value.is_finite() for value in normalized):
        raise NullModelError("measurement values must be finite")
    return normalized


def _normalize_glyphs(
    registry: MorseRegistry,
    glyphs: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = (
        tuple(symbol.glyph for symbol in registry.symbols)
        if glyphs is None
        else tuple(glyphs)
    )
    if not selected:
        raise NullModelError("at least one Morse glyph must be selected")
    if len(set(selected)) != len(selected):
        raise NullModelError("selected Morse glyphs must be unique")

    for glyph in selected:
        try:
            registry.symbol_for_glyph(glyph)
        except MorseError as error:
            raise NullModelError(
                f"selected glyph is absent from the Morse registry: {glyph!r}"
            ) from error

    return selected


def _normalize_directions(
    directions: Sequence[SequenceDirection],
) -> tuple[SequenceDirection, ...]:
    selected = tuple(directions)
    if not selected:
        raise NullModelError("at least one sequence direction must be selected")
    if len(set(selected)) != len(selected):
        raise NullModelError("selected sequence directions must be unique")
    if any(not isinstance(direction, SequenceDirection) for direction in selected):
        raise NullModelError(
            "selected sequence directions must be SequenceDirection values"
        )
    return selected


def _normalize_polarities(
    polarities: Sequence[MorsePolarity],
) -> tuple[MorsePolarity, ...]:
    selected = tuple(polarities)
    if not selected:
        raise NullModelError("at least one Morse polarity must be selected")
    if len(set(selected)) != len(selected):
        raise NullModelError("selected Morse polarities must be unique")
    if any(not isinstance(polarity, MorsePolarity) for polarity in selected):
        raise NullModelError(
            "selected Morse polarities must be MorsePolarity values"
        )
    return selected
