"""Typed representation of the surviving 6EQUJ5 intensity sequence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

WOW_PRINTER_SEQUENCE: Final = "6EQUJ5"
WOW_INTEGRATION_SECONDS: Final = Decimal("10")
WOW_SAMPLE_CADENCE_SECONDS: Final = Decimal("12")

_ZERO: Final = Decimal("0")
_HALF: Final = Decimal("0.5")
_ONE: Final = Decimal("1")
_VALID_DIGITS: Final = "123456789"
_VALID_LETTERS: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True, slots=True)
class IntensityCode:
    """One Big Ear printer symbol and its signal-to-noise interval."""

    symbol: str
    truncated_snr: int

    def __post_init__(self) -> None:
        expected = _decode_truncated_snr(self.symbol)
        if self.truncated_snr != expected:
            message = (
                f"truncated_snr {self.truncated_snr} does not match "
                f"printer symbol {self.symbol!r} ({expected})"
            )
            raise ValueError(message)

    @property
    def lower_snr(self) -> Decimal:
        """Return the inclusive lower bound represented by the symbol."""

        return Decimal(self.truncated_snr)

    @property
    def upper_snr(self) -> Decimal:
        """Return the exclusive upper bound represented by the symbol."""

        return self.lower_snr + _ONE

    @property
    def midpoint_snr(self) -> Decimal:
        """Return the interval midpoint used as the best point estimate."""

        return self.lower_snr + _HALF


@dataclass(frozen=True, slots=True)
class SignalSample:
    """One decoded sample in an ordered printer sequence."""

    sample_index: int
    elapsed_seconds: Decimal
    intensity: IntensityCode

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.elapsed_seconds < _ZERO:
            raise ValueError("elapsed_seconds must be non-negative")


def _decode_truncated_snr(symbol: str) -> int:
    if len(symbol) != 1:
        raise ValueError("printer symbol must contain exactly one character")
    if symbol == " ":
        return 0
    if symbol in _VALID_DIGITS:
        return int(symbol)
    if symbol in _VALID_LETTERS:
        return ord(symbol) - ord("A") + 10
    raise ValueError(f"unsupported Big Ear printer symbol: {symbol!r}")


def decode_printer_symbol(symbol: str) -> IntensityCode:
    """Decode one printer symbol without inferring unavailable precision."""

    return IntensityCode(symbol=symbol, truncated_snr=_decode_truncated_snr(symbol))


def decode_printer_sequence(
    sequence: str,
    *,
    sample_cadence_seconds: Decimal = WOW_SAMPLE_CADENCE_SECONDS,
) -> tuple[SignalSample, ...]:
    """Decode an ordered sequence into immutable, time-indexed samples."""

    if not sequence:
        raise ValueError("printer sequence must not be empty")
    if sample_cadence_seconds <= _ZERO:
        raise ValueError("sample_cadence_seconds must be positive")

    return tuple(
        SignalSample(
            sample_index=index,
            elapsed_seconds=sample_cadence_seconds * index,
            intensity=decode_printer_symbol(symbol),
        )
        for index, symbol in enumerate(sequence)
    )


def canonical_wow_samples() -> tuple[SignalSample, ...]:
    """Return the canonical six decoded Wow! signal samples."""

    return decode_printer_sequence(WOW_PRINTER_SEQUENCE)
