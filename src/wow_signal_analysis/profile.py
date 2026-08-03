"""Deterministic shape analysis for ordered signal-to-noise measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from wow_signal_analysis.measurements import SignalSample


class ProfileError(ValueError):
    """Raised when a sequence cannot support a valid deterministic profile."""


class Trend(StrEnum):
    """Direction of one adjacent measurement change."""

    RISING = "rising"
    FLAT = "flat"
    FALLING = "falling"

    @property
    def symbol(self) -> str:
        """Return a compact, language-neutral direction symbol."""

        if self is Trend.RISING:
            return "+"
        if self is Trend.FALLING:
            return "-"
        return "0"


@dataclass(frozen=True, slots=True)
class AdjacentChange:
    """Difference between two neighboring measurements."""

    from_index: int
    to_index: int
    delta_snr: Decimal
    trend: Trend

    def __post_init__(self) -> None:
        if self.from_index < 0:
            raise ProfileError("from_index must be non-negative")
        if self.to_index != self.from_index + 1:
            raise ProfileError("to_index must immediately follow from_index")

        expected_trend = _classify_delta(self.delta_snr)
        if self.trend is not expected_trend:
            raise ProfileError("trend does not match delta_snr")


@dataclass(frozen=True, slots=True)
class MirrorComparison:
    """Comparison of positions equally distant from opposite sequence ends."""

    left_index: int
    right_index: int
    left_snr: Decimal
    right_snr: Decimal
    signed_difference: Decimal

    def __post_init__(self) -> None:
        if self.left_index < 0 or self.right_index <= self.left_index:
            raise ProfileError("mirror indices must be ordered and non-negative")
        if self.signed_difference != self.left_snr - self.right_snr:
            raise ProfileError("signed_difference must equal left_snr minus right_snr")

    @property
    def absolute_difference(self) -> Decimal:
        """Return the unsigned difference between mirrored measurements."""

        return abs(self.signed_difference)


@dataclass(frozen=True, slots=True)
class SequenceProfile:
    """Auditable, non-semantic description of one ordered numeric sequence."""

    values: tuple[Decimal, ...]
    changes: tuple[AdjacentChange, ...]
    mirror_comparisons: tuple[MirrorComparison, ...]
    peak_index: int
    peak_value: Decimal
    peak_count: int

    def __post_init__(self) -> None:
        if not self.values:
            raise ProfileError("profile values must not be empty")
        if any(not value.is_finite() for value in self.values):
            raise ProfileError("profile values must be finite")
        if self.changes != _build_changes(self.values):
            raise ProfileError("changes do not match the profile values")
        if self.mirror_comparisons != _build_mirror_comparisons(self.values):
            raise ProfileError("mirror comparisons do not match the profile values")
        if not 0 <= self.peak_index < len(self.values):
            raise ProfileError("peak_index is outside the sequence")
        if self.values[self.peak_index] != self.peak_value:
            raise ProfileError("peak_index does not identify peak_value")
        if self.peak_value != max(self.values):
            raise ProfileError("peak_value is not the sequence maximum")
        if self.peak_count != self.values.count(self.peak_value):
            raise ProfileError("peak_count does not match the sequence")

    @property
    def reverse_values(self) -> tuple[Decimal, ...]:
        """Return the values in reverse temporal order."""

        return tuple(reversed(self.values))

    @property
    def trend_pattern(self) -> tuple[Trend, ...]:
        """Return the adjacent trend classification in sequence order."""

        return tuple(change.trend for change in self.changes)

    @property
    def trend_signature(self) -> str:
        """Return rising, flat, and falling changes as +, 0, and -."""

        return "".join(change.trend.symbol for change in self.changes)

    @property
    def has_unique_peak(self) -> bool:
        """Return whether exactly one measurement attains the maximum."""

        return self.peak_count == 1

    @property
    def is_strict_single_peak(self) -> bool:
        """Return whether values strictly rise to one interior peak and then fall."""

        if not self.has_unique_peak or self.peak_index in {0, len(self.values) - 1}:
            return False

        rising_side = self.changes[: self.peak_index]
        falling_side = self.changes[self.peak_index :]

        return all(change.trend is Trend.RISING for change in rising_side) and all(
            change.trend is Trend.FALLING for change in falling_side
        )

    @property
    def is_exact_palindrome(self) -> bool:
        """Return whether forward and reverse measurement orders are identical."""

        return self.values == self.reverse_values


def analyze_midpoint_values(values: Sequence[Decimal]) -> SequenceProfile:
    """Build a deterministic profile without assigning linguistic meaning."""

    normalized = tuple(values)
    if not normalized:
        raise ProfileError("measurement sequence must not be empty")
    if any(not value.is_finite() for value in normalized):
        raise ProfileError("measurement values must be finite")

    peak_value = max(normalized)
    peak_index = normalized.index(peak_value)

    return SequenceProfile(
        values=normalized,
        changes=_build_changes(normalized),
        mirror_comparisons=_build_mirror_comparisons(normalized),
        peak_index=peak_index,
        peak_value=peak_value,
        peak_count=normalized.count(peak_value),
    )


def analyze_samples(samples: Sequence[SignalSample]) -> SequenceProfile:
    """Profile the midpoint estimates represented by ordered signal samples."""

    return analyze_midpoint_values(
        tuple(sample.intensity.midpoint_snr for sample in samples)
    )


def _build_changes(values: tuple[Decimal, ...]) -> tuple[AdjacentChange, ...]:
    return tuple(
        AdjacentChange(
            from_index=index,
            to_index=index + 1,
            delta_snr=values[index + 1] - values[index],
            trend=_classify_delta(values[index + 1] - values[index]),
        )
        for index in range(len(values) - 1)
    )


def _build_mirror_comparisons(
    values: tuple[Decimal, ...],
) -> tuple[MirrorComparison, ...]:
    comparisons: list[MirrorComparison] = []

    for left_index in range(len(values) // 2):
        right_index = len(values) - 1 - left_index
        left_value = values[left_index]
        right_value = values[right_index]

        comparisons.append(
            MirrorComparison(
                left_index=left_index,
                right_index=right_index,
                left_snr=left_value,
                right_snr=right_value,
                signed_difference=left_value - right_value,
            )
        )

    return tuple(comparisons)


def _classify_delta(delta: Decimal) -> Trend:
    if delta > 0:
        return Trend.RISING
    if delta < 0:
        return Trend.FALLING
    return Trend.FLAT
