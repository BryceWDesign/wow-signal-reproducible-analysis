"""Exhaustive, non-semantic Morse comparisons for threshold partitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from wow_signal_analysis.morse import MorseRegistry, MorseSymbol
from wow_signal_analysis.thresholds import ThresholdCase, enumerate_threshold_cases


class MorseCorrespondenceError(ValueError):
    """Raised when a threshold-to-Morse comparison is internally inconsistent."""


class SequenceDirection(StrEnum):
    """Temporal order used when rendering a threshold partition."""

    FORWARD = "forward"
    REVERSE = "reverse"


class MorsePolarity(StrEnum):
    """Explicit assignment of high and low states to Morse elements."""

    HIGH_IS_DASH = "high-is-dash"
    HIGH_IS_DOT = "high-is-dot"

    def render(self, is_high: bool) -> str:
        """Render one threshold state as a Morse dot or dash."""

        if self is MorsePolarity.HIGH_IS_DASH:
            return "-" if is_high else "."
        return "." if is_high else "-"


@dataclass(frozen=True, slots=True)
class ThresholdMorseComparison:
    """One fully declared threshold, direction, and polarity comparison."""

    cut_index: int
    direction: SequenceDirection
    polarity: MorsePolarity
    lower_bound_inclusive: Decimal | None
    upper_bound_exclusive: Decimal | None
    binary_pattern: str
    morse_pattern: str
    matched_symbols: tuple[MorseSymbol, ...]

    def __post_init__(self) -> None:
        if self.cut_index < 0:
            raise MorseCorrespondenceError("cut_index must be non-negative")
        if (
            self.lower_bound_inclusive is not None
            and self.upper_bound_exclusive is not None
            and self.lower_bound_inclusive >= self.upper_bound_exclusive
        ):
            raise MorseCorrespondenceError(
                "threshold interval bounds must be strictly ordered"
            )
        if not self.binary_pattern or any(
            character not in {"0", "1"} for character in self.binary_pattern
        ):
            raise MorseCorrespondenceError(
                "binary_pattern must contain only '0' and '1'"
            )
        if not self.morse_pattern or any(
            character not in {".", "-"} for character in self.morse_pattern
        ):
            raise MorseCorrespondenceError(
                "morse_pattern must contain only '.' and '-'"
            )
        if len(self.binary_pattern) != len(self.morse_pattern):
            raise MorseCorrespondenceError(
                "binary_pattern and morse_pattern must have equal length"
            )

        expected_pattern = "".join(
            self.polarity.render(character == "1")
            for character in self.binary_pattern
        )
        if self.morse_pattern != expected_pattern:
            raise MorseCorrespondenceError(
                "morse_pattern does not match binary_pattern and polarity"
            )
        if any(
            symbol.pattern != self.morse_pattern for symbol in self.matched_symbols
        ):
            raise MorseCorrespondenceError(
                "matched_symbols must use the comparison's Morse pattern"
            )

    @property
    def is_assigned(self) -> bool:
        """Return whether the official registry assigns this pattern."""

        return bool(self.matched_symbols)

    @property
    def matched_glyphs(self) -> tuple[str, ...]:
        """Return assigned printable glyphs in registry order."""

        return tuple(symbol.glyph for symbol in self.matched_symbols)


@dataclass(frozen=True, slots=True)
class MorseCorrespondenceReport:
    """Complete threshold-to-Morse comparison space for one numeric sequence."""

    values: tuple[Decimal, ...]
    threshold_cases: tuple[ThresholdCase, ...]
    comparisons: tuple[ThresholdMorseComparison, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise MorseCorrespondenceError("report values must not be empty")
        if self.threshold_cases != enumerate_threshold_cases(self.values):
            raise MorseCorrespondenceError(
                "report threshold_cases do not match report values"
            )

        expected_comparison_count = len(self.threshold_cases) * 4
        if len(self.comparisons) != expected_comparison_count:
            raise MorseCorrespondenceError(
                "report must include forward and reverse comparisons "
                "under both polarities"
            )

        expected_keys = {
            (case.cut_index, direction, polarity)
            for case in self.threshold_cases
            for direction in SequenceDirection
            for polarity in MorsePolarity
        }
        actual_keys = {
            (comparison.cut_index, comparison.direction, comparison.polarity)
            for comparison in self.comparisons
        }
        if actual_keys != expected_keys or len(actual_keys) != len(self.comparisons):
            raise MorseCorrespondenceError(
                "report comparisons must cover each parameter combination exactly once"
            )

    @property
    def assigned_comparisons(self) -> tuple[ThresholdMorseComparison, ...]:
        """Return only comparisons matching official printable symbols."""

        return tuple(
            comparison for comparison in self.comparisons if comparison.is_assigned
        )

    @property
    def unique_assigned_symbols(self) -> tuple[MorseSymbol, ...]:
        """Return unique matched symbols in first-observed comparison order."""

        symbols: list[MorseSymbol] = []
        seen_glyphs: set[str] = set()
        for comparison in self.assigned_comparisons:
            for symbol in comparison.matched_symbols:
                if symbol.glyph not in seen_glyphs:
                    symbols.append(symbol)
                    seen_glyphs.add(symbol.glyph)
        return tuple(symbols)

    def comparisons_for_glyph(
        self,
        glyph: str,
    ) -> tuple[ThresholdMorseComparison, ...]:
        """Return every declared comparison that matches one printable glyph."""

        return tuple(
            comparison
            for comparison in self.assigned_comparisons
            if glyph in comparison.matched_glyphs
        )


def analyze_threshold_morse(
    values: Sequence[Decimal],
    registry: MorseRegistry,
) -> MorseCorrespondenceReport:
    """Enumerate every threshold, direction, and polarity before matching Morse."""

    normalized = tuple(values)
    threshold_cases = enumerate_threshold_cases(normalized)
    comparisons: list[ThresholdMorseComparison] = []

    for case in threshold_cases:
        for direction in SequenceDirection:
            directed_mask = _directed_mask(case.high_mask, direction)
            binary_pattern = _binary_pattern(directed_mask)
            for polarity in MorsePolarity:
                morse_pattern = morse_pattern_from_mask(directed_mask, polarity)
                comparisons.append(
                    ThresholdMorseComparison(
                        cut_index=case.cut_index,
                        direction=direction,
                        polarity=polarity,
                        lower_bound_inclusive=case.lower_bound_inclusive,
                        upper_bound_exclusive=case.upper_bound_exclusive,
                        binary_pattern=binary_pattern,
                        morse_pattern=morse_pattern,
                        matched_symbols=registry.symbols_for_pattern(morse_pattern),
                    )
                )

    return MorseCorrespondenceReport(
        values=normalized,
        threshold_cases=threshold_cases,
        comparisons=tuple(comparisons),
    )


def morse_pattern_from_mask(
    high_mask: Sequence[bool],
    polarity: MorsePolarity,
) -> str:
    """Render a non-empty threshold mask under an explicit Morse polarity."""

    normalized = tuple(high_mask)
    if not normalized:
        raise MorseCorrespondenceError("high_mask must not be empty")
    return "".join(polarity.render(is_high) for is_high in normalized)


def _directed_mask(
    high_mask: tuple[bool, ...],
    direction: SequenceDirection,
) -> tuple[bool, ...]:
    if direction is SequenceDirection.FORWARD:
        return high_mask
    return tuple(reversed(high_mask))


def _binary_pattern(high_mask: tuple[bool, ...]) -> str:
    return "".join("1" if is_high else "0" for is_high in high_mask)
