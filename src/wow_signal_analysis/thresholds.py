"""Exhaustive threshold partitions for short signal-strength sequences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class ThresholdError(ValueError):
    """Raised when threshold analysis cannot be performed safely."""


class BitPolarity(StrEnum):
    """Mapping from the above-threshold state to a rendered binary digit."""

    HIGH_IS_ONE = "high-is-one"
    HIGH_IS_ZERO = "high-is-zero"

    def render(self, is_high: bool) -> str:
        """Render one threshold state as a binary digit."""

        high_digit = "1" if self is BitPolarity.HIGH_IS_ONE else "0"
        low_digit = "0" if self is BitPolarity.HIGH_IS_ONE else "1"
        return high_digit if is_high else low_digit


@dataclass(frozen=True, slots=True)
class ThresholdCase:
    """One unique threshold partition of an ordered measurement sequence."""

    cut_index: int
    lower_bound_inclusive: Decimal | None
    upper_bound_exclusive: Decimal | None
    representative_threshold: Decimal
    high_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.cut_index < 0:
            raise ThresholdError("cut_index must be non-negative")
        if not self.high_mask:
            raise ThresholdError("high_mask must not be empty")
        if not self.representative_threshold.is_finite():
            raise ThresholdError("representative_threshold must be finite")
        if self.lower_bound_inclusive is not None and not self.lower_bound_inclusive.is_finite():
            raise ThresholdError("lower_bound_inclusive must be finite when present")
        if self.upper_bound_exclusive is not None and not self.upper_bound_exclusive.is_finite():
            raise ThresholdError("upper_bound_exclusive must be finite when present")
        if (
            self.lower_bound_inclusive is not None
            and self.upper_bound_exclusive is not None
            and self.lower_bound_inclusive >= self.upper_bound_exclusive
        ):
            raise ThresholdError("threshold interval bounds must be strictly ordered")
        if (
            self.lower_bound_inclusive is not None
            and self.representative_threshold < self.lower_bound_inclusive
        ):
            raise ThresholdError("representative_threshold is below the interval")
        if (
            self.upper_bound_exclusive is not None
            and self.representative_threshold >= self.upper_bound_exclusive
        ):
            raise ThresholdError("representative_threshold is outside the interval")

    @property
    def high_count(self) -> int:
        """Return the number of measurements classified above threshold."""

        return sum(self.high_mask)

    @property
    def low_count(self) -> int:
        """Return the number of measurements classified at or below threshold."""

        return len(self.high_mask) - self.high_count

    def bit_pattern(self, polarity: BitPolarity = BitPolarity.HIGH_IS_ONE) -> str:
        """Render the ordered threshold partition under an explicit polarity."""

        return "".join(polarity.render(is_high) for is_high in self.high_mask)

    def reverse_bit_pattern(
        self,
        polarity: BitPolarity = BitPolarity.HIGH_IS_ONE,
    ) -> str:
        """Render the threshold partition in reverse temporal order."""

        return "".join(polarity.render(is_high) for is_high in reversed(self.high_mask))

    def contains_threshold(self, threshold: Decimal) -> bool:
        """Return whether a threshold belongs to this equivalence interval."""

        if not threshold.is_finite():
            return False
        if self.lower_bound_inclusive is not None and threshold < self.lower_bound_inclusive:
            return False
        return not (
            self.upper_bound_exclusive is not None and threshold >= self.upper_bound_exclusive
        )


def apply_threshold(
    values: Sequence[Decimal],
    threshold: Decimal,
) -> tuple[bool, ...]:
    """Classify values using the declared rule: high means strictly above threshold."""

    normalized = _normalize_values(values)
    if not threshold.is_finite():
        raise ThresholdError("threshold must be finite")
    return tuple(value > threshold for value in normalized)


def enumerate_threshold_cases(values: Sequence[Decimal]) -> tuple[ThresholdCase, ...]:
    """Enumerate every distinct binary partition induced by scalar thresholds."""

    normalized = _normalize_values(values)
    levels = tuple(sorted(set(normalized)))
    cases: list[ThresholdCase] = []

    for cut_index in range(len(levels) + 1):
        lower_bound = levels[cut_index - 1] if cut_index > 0 else None
        upper_bound = levels[cut_index] if cut_index < len(levels) else None
        threshold = _representative_threshold(lower_bound, upper_bound)
        mask = tuple(value > threshold for value in normalized)

        cases.append(
            ThresholdCase(
                cut_index=cut_index,
                lower_bound_inclusive=lower_bound,
                upper_bound_exclusive=upper_bound,
                representative_threshold=threshold,
                high_mask=mask,
            )
        )

    patterns = tuple(case.high_mask for case in cases)
    if len(set(patterns)) != len(patterns):
        raise ThresholdError("threshold enumeration produced duplicate partitions")

    return tuple(cases)


def find_threshold_case(
    cases: Sequence[ThresholdCase],
    threshold: Decimal,
) -> ThresholdCase:
    """Return the unique enumerated case containing a finite threshold."""

    if not threshold.is_finite():
        raise ThresholdError("threshold must be finite")
    matches = tuple(case for case in cases if case.contains_threshold(threshold))
    if len(matches) != 1:
        raise ThresholdError(
            f"expected exactly one threshold case for {threshold}, found {len(matches)}"
        )
    return matches[0]


def _normalize_values(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ThresholdError("measurement sequence must not be empty")
    if any(not value.is_finite() for value in normalized):
        raise ThresholdError("measurement values must be finite")
    return normalized


def _representative_threshold(
    lower_bound: Decimal | None,
    upper_bound: Decimal | None,
) -> Decimal:
    if lower_bound is None and upper_bound is None:
        raise ThresholdError("threshold interval must have at least one finite bound")
    if lower_bound is None:
        assert upper_bound is not None
        return upper_bound - Decimal(1)
    if upper_bound is None:
        return lower_bound
    return (lower_bound + upper_bound) / Decimal(2)
